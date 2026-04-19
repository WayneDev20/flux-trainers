"""
RunPod Serverless handler for FLUX LoRA training.

Slim orchestrator. All real work lives in the underscore modules so it's
trivially unit-testable (see TESTING.md).

Flow (DESIGN.md §2):
    [1] parse request              -> exit 4 on bad schema/R2 key
    [2] idempotency probe          -> exit 0 if LoRA already at output prefix
    [3] SHA-verify FLUX weights    -> exit 2 on mismatch
    [4] download dataset zip       -> exit 5 on R2 read fail
    [5] write caption .txt files   -> exit 4 on missing, exit 3 on partial
    [6] subprocess train.py        -> exit 6 on non-zero
    [7] upload artifacts           -> exit 7 on R2 write fail
    [8] POST HMAC-signed webhook   -> exit 0 (webhook fail is non-fatal, state in R2)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import runpod

# sentry_sdk is optional — present in prod, absent in tests/CI
try:
    import sentry_sdk  # type: ignore
    _SENTRY = True
except ImportError:
    _SENTRY = False

from . import _storage, _training, _webhook, _weights
from ._validation import (
    EXIT_BAD_REQUEST,
    EXIT_OK,
    ValidationError,
    check_captions_complete,
    list_images_in_zip,
    parse_request,
    validate_lora_prefix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging — structured JSON lines to stdout (RunPod captures stdout)
# ─────────────────────────────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k in ("job_id", "user_id", "stage", "duration_ms"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, separators=(",", ":"))


_root = logging.getLogger()
if not _root.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_JsonFormatter())
    _root.addHandler(h)
_root.setLevel(logging.INFO)
log = logging.getLogger("handler")


# ─────────────────────────────────────────────────────────────────────────────
# Paths / env
# ─────────────────────────────────────────────────────────────────────────────
_HERE              = Path(__file__).parent
SCHEMA_PATH        = _HERE / "schemas" / "request.v1.json"
MANIFEST_PATH      = _HERE / "flux_weights_manifest.json"
FLUX_WEIGHTS_DIR   = Path(os.environ.get("FLUX_WEIGHTS_DIR", "/workspace/FLUX.1-dev"))
FLUX_VERSION       = os.environ.get("FLUX_WEIGHTS_VERSION", "v1")
GIT_SHA            = os.environ.get("GIT_SHA", "unknown")


def _sentry_init() -> None:
    dsn = os.environ.get("SENTRY_DSN")
    if _SENTRY and dsn:
        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.05, release=GIT_SHA)


def _stage(job_id: str, stage: str, **kw: Any) -> None:
    """Emit a structured stage marker for observability timelines."""
    log.info(stage, extra={"job_id": job_id, "stage": stage, **kw})


# ─────────────────────────────────────────────────────────────────────────────
# Main handler
# ─────────────────────────────────────────────────────────────────────────────
def handler(event: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    raw = event.get("input") or {}
    job_id = str(raw.get("job_id", "unknown"))
    webhook_url = raw.get("webhook_url") or event.get("webhook")
    secret = os.environ.get("WEBHOOK_HMAC_SECRET", "").encode()
    bucket_loras    = os.environ["R2_BUCKET_LORAS"]
    bucket_datasets = os.environ["R2_BUCKET_DATASETS"]
    bucket_weights  = os.environ["R2_BUCKET_WEIGHTS_MIRROR"]

    try:
        # ── [1] parse + schema validate ──────────────────────────────────────
        req = parse_request(raw, SCHEMA_PATH)
        job_id = req["job_id"]
        user_id = req["user_id"]
        output_prefix = validate_lora_prefix(user_id, job_id)
        _stage(job_id, "request.parsed", user_id=user_id)

        s3 = _storage.make_client()

        # ── [2] idempotency probe ────────────────────────────────────────────
        lora_key = output_prefix + "lora.safetensors"
        head = _storage.head_exists(s3, bucket_loras, lora_key)
        if head is not None:
            _stage(job_id, "idempotent.hit", bytes=head.get("ContentLength"))
            # Re-emit the original manifest as the webhook body for bit-identical replay
            manifest_bytes = _storage.get_bytes(
                s3, bucket_loras, output_prefix + "manifest.json",
            )
            payload = json.loads(manifest_bytes)
            payload["status"] = "succeeded"
            payload["idempotent_replay"] = True
            _webhook.send(webhook_url, payload, secret)
            return payload
        _stage(job_id, "idempotent.miss")

        # ── [3] SHA-verify FLUX weights ──────────────────────────────────────
        weights_manifest = _weights.load_manifest(MANIFEST_PATH, FLUX_VERSION)

        def _fetch_weight(filename: str, local: Path) -> None:
            _storage.download(
                s3, bucket_weights,
                f"{weights_manifest['mirror_prefix']}{filename}",
                local,
            )

        _weights.verify_weights(
            weights_manifest, FLUX_WEIGHTS_DIR, fetch_missing=_fetch_weight,
        )
        _stage(job_id, "sha.verified", version=FLUX_VERSION)

        # ── [4] download dataset ─────────────────────────────────────────────
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"lora_{job_id}_") as tmp:
            workdir = Path(tmp)
            dataset_zip = workdir / "dataset.zip"
            _storage.download(s3, bucket_datasets, req["dataset_r2_key"], dataset_zip)
            _stage(job_id, "dataset.downloaded",
                   bytes=dataset_zip.stat().st_size)

            # ── [5] captions ────────────────────────────────────────────────
            image_names = list_images_in_zip(dataset_zip)
            if not image_names:
                raise ValidationError("dataset zip contains no recognized images")
            captions = req["captions"]  # required (ADR-0004 revised)
            check_captions_complete(image_names, captions)

            # train.py extracts the zip into INPUT_DIR on launch. We can't
            # pre-seed .txt files there yet. Instead we inject captions by
            # rebuilding the zip with matching .txt entries inside it — then
            # train.py's extract_zip puts them alongside the images.
            dataset_zip = _inject_caption_txt(dataset_zip, captions, workdir)
            _stage(job_id, "captions.written", count=len(captions))

            # ── [6] train ────────────────────────────────────────────────────
            # Multi-GPU: when RunPod allocates >1 GPU to this worker, wrap
            # with `accelerate launch` so ai-toolkit's FSDP v2 path triggers.
            # Single-GPU behavior is unchanged.
            try:
                import torch  # imported lazily; handler boot shouldn't need CUDA
                num_gpus = torch.cuda.device_count()
            except Exception:
                num_gpus = 1
            _stage(job_id, "train.gpu_count", num_gpus=num_gpus)

            argv = _training.build_argv(
                req["config"],
                input_zip=dataset_zip,
                trigger_word=req["trigger_word"],
                captions_provided=True,
                num_gpus=num_gpus,
            )

            def _progress(tail: str) -> None:
                try:
                    runpod.serverless.progress_update(event, {"log_tail": tail})
                except Exception:
                    pass

            _stage(job_id, "train.started")
            lora_path = _training.run_training(
                argv, log_path=workdir / "train.log", progress_cb=_progress,
            )
            _stage(job_id, "train.finished")

            # ── [7] upload artifacts ────────────────────────────────────────
            duration_s = int(time.time() - t0)
            training_manifest = _build_manifest(
                req=req, weights_manifest=weights_manifest,
                duration_s=duration_s, image_count=len(image_names),
                lora_path=lora_path,
            )
            artifacts = _upload_artifacts(
                s3, bucket_loras, output_prefix, lora_path, workdir,
                training_manifest,
            )
            _stage(job_id, "artifacts.uploaded", **artifacts)

        # ── Cleanup for the next warm invocation ─────────────────────────────
        if _training.TRAIN_OUTPUT_DIR.exists():
            shutil.rmtree(_training.TRAIN_OUTPUT_DIR, ignore_errors=True)

        # ── [8] webhook ──────────────────────────────────────────────────────
        payload = {**training_manifest, "status": "succeeded"}
        _webhook.send(webhook_url, payload, secret)
        _stage(job_id, "webhook.sent", duration_ms=int((time.time() - t0) * 1000))
        return payload

    except ValidationError as e:
        return _fail(job_id, webhook_url, secret, e, getattr(e, "exit_code", EXIT_BAD_REQUEST))
    except _weights.WeightParityError as e:
        return _fail(job_id, webhook_url, secret, e, e.exit_code)
    except _storage.StorageReadError as e:
        return _fail(job_id, webhook_url, secret, e, e.exit_code)
    except _storage.StorageWriteError as e:
        return _fail(job_id, webhook_url, secret, e, e.exit_code)
    except _training.TrainingError as e:
        return _fail(job_id, webhook_url, secret, e, e.exit_code,
                     extra={"oom": e.oom, "stderr_tail": e.stderr_tail})
    except Exception as e:  # noqa: BLE001 — catch-all for unexpected, still reports
        if _SENTRY:
            sentry_sdk.capture_exception(e)
        return _fail(job_id, webhook_url, secret, e, 1,
                     extra={"traceback": traceback.format_exc()[-4000:]})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _inject_caption_txt(
    src_zip: Path, captions: dict[str, str], workdir: Path,
) -> Path:
    """
    Rebuild the dataset zip with `<image_stem>.txt` entries alongside each
    image. Case-insensitive filename match. train.py's extract_zip preserves
    the structure so the .txt files end up in INPUT_DIR next to their images.
    """
    out = workdir / "dataset_with_captions.zip"
    cap_by_lower = {k.lower(): v for k, v in captions.items()}
    with zipfile.ZipFile(src_zip, "r") as zin, \
         zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("__MACOSX/") or "/._" in name or name.startswith("._"):
                continue
            base = Path(name).name
            if base.startswith("."):
                continue
            zout.writestr(base, zin.read(info))
            caption = cap_by_lower.get(base.lower())
            if caption is not None:
                zout.writestr(Path(base).stem + ".txt", caption)
    return out


def _build_manifest(
    *, req: dict[str, Any], weights_manifest: dict[str, Any],
    duration_s: int, image_count: int, lora_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "training_manifest.v1",
        "job_id":          req["job_id"],
        "user_id":         req["user_id"],
        "trigger_word":    req["trigger_word"],
        "config":          req["config"],
        "weights_version": weights_manifest["version"],
        "weights_source":  weights_manifest.get("source"),
        "image_count":     image_count,
        "duration_s":      duration_s,
        "platform":        "runpod",
        "gpu_name":        os.environ.get("RUNPOD_GPU_NAME", "unknown"),
        "git_sha":         GIT_SHA,
        "lora_sha256":     _weights.sha256_file(lora_path),
        "ts":              int(time.time()),
    }


def _upload_artifacts(
    s3: Any, bucket: str, prefix: str,
    lora_path: Path, workdir: Path, manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Ordering matters: samples → manifest → lora.safetensors. The LoRA blob is
    the commit marker; a partial run can never masquerade as complete because
    the idempotency probe keys on `lora.safetensors` existence.
    """
    uploaded_samples: list[str] = []

    samples_dir = _training.TRAIN_OUTPUT_DIR / "samples"
    if samples_dir.exists():
        for s in sorted(samples_dir.glob("*.jpg")):
            key = f"{prefix}samples/{s.name}"
            _storage.upload(s3, bucket, key, s, content_type="image/jpeg")
            uploaded_samples.append(key)

    log_local = workdir / "train.log"
    logs_key: str | None = None
    if log_local.exists():
        logs_key = prefix + "logs.txt"
        _storage.upload(s3, bucket, logs_key, log_local, content_type="text/plain")

    manifest["samples_r2_prefix"] = f"{prefix}samples/"
    manifest["logs_r2_key"] = logs_key
    manifest["lora_r2_key"] = prefix + "lora.safetensors"
    manifest["manifest_r2_key"] = prefix + "manifest.json"

    # Manifest before the LoRA so a retry after LoRA upload but before
    # manifest upload is impossible (manifest upload is cheap and comes first).
    _storage.put_bytes(
        s3, bucket, manifest["manifest_r2_key"],
        json.dumps(manifest, indent=2).encode(),
        "application/json",
    )
    _storage.upload(s3, bucket, manifest["lora_r2_key"], lora_path)

    return {
        "sample_count": len(uploaded_samples),
        "lora_sha256":  manifest["lora_sha256"],
    }


def _fail(
    job_id: str, webhook_url: str | None, secret: bytes,
    exc: Exception, exit_code: int,
    *, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status":   "failed",
        "job_id":   job_id,
        "error":    str(exc),
        "exit_code": exit_code,
        "ts":       int(time.time()),
    }
    if extra:
        payload.update(extra)
    log.error("job.failed", extra={"job_id": job_id, "exit_code": exit_code},
              exc_info=exc)
    if webhook_url:
        _webhook.send(webhook_url, payload, secret)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# RunPod entrypoint
# ─────────────────────────────────────────────────────────────────────────────
_sentry_init()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
