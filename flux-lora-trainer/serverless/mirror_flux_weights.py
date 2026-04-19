"""
One-time tool: mirror FLUX.1-dev base weights from Replicate's CDN to our R2
bucket and emit a committed SHA-256 manifest (ADR-0003).

The Replicate CDN tar contains the HuggingFace Diffusers directory structure
rooted at `FLUX.1-dev/`:

    FLUX.1-dev/
        ae.safetensors              (optional top-level BFL-style single-file)
        model_index.json
        scheduler/scheduler_config.json
        text_encoder/config.json
        text_encoder/model.safetensors                      # CLIP-L
        text_encoder_2/config.json
        text_encoder_2/model.safetensors.index.json
        text_encoder_2/model-00001-of-00002.safetensors     # T5-xxl shard 1
        text_encoder_2/model-00002-of-00002.safetensors     # T5-xxl shard 2
        tokenizer/...
        tokenizer_2/...
        transformer/config.json
        transformer/diffusion_pytorch_model.safetensors.index.json
        transformer/diffusion_pytorch_model-00001-of-0000N.safetensors
        ...
        vae/config.json
        vae/diffusion_pytorch_model.safetensors

We mirror everything under `FLUX.1-dev/` (stripping that prefix) except preview
images, record SHA + size for every file, and write a single committed manifest.

Run once at bootstrap, and again for every new FLUX revision (v2/, v3/, …).
Never overwrites an existing version — each version is an immutable snapshot
so in-flight LoRAs keep parity with the weights they were trained against.

Required env:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    R2_ENDPOINT_URL (optional; derives from account id if absent)

Usage:
    python -m serverless.mirror_flux_weights \
        --version v1 \
        --bucket flux-weights-mirror \
        [--tar-path /root/files.tar]          # reuse download across retries
        [--extract-dir /root/extracted]       # reuse extraction across retries
        [--also-write-local /workspace/FLUX.1-dev] \
        [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

from . import _storage
from ._weights import sha256_file

log = logging.getLogger("mirror")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


REPLICATE_CDN_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)

# Top-level dir inside the tar; stripped when flattening into out_dir.
TAR_ROOT = "FLUX.1-dev"

# Files we skip entirely — not needed for training or inference parity.
SKIP_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif")

# macOS / git artifacts that may sneak in
SKIP_PATH_PARTS = {"__MACOSX", ".DS_Store", ".git", ".gitattributes"}

MANIFEST_PATH = Path(__file__).parent / "flux_weights_manifest.json"


def _download_tar(dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        log.info("tar already present at %s (size=%.1f GB), skipping download",
                 dest, dest.stat().st_size / 1e9)
        return
    log.info("downloading %s -> %s", REPLICATE_CDN_URL, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("pget"):
        subprocess.check_call(["pget", "-f", REPLICATE_CDN_URL, str(dest)])
    else:
        subprocess.check_call(
            ["curl", "-L", "--retry", "3", "--fail", "-o", str(dest), REPLICATE_CDN_URL]
        )


def _should_skip(relpath: str) -> bool:
    if any(part in SKIP_PATH_PARTS for part in Path(relpath).parts):
        return True
    if relpath.lower().endswith(SKIP_SUFFIXES):
        return True
    return False


def _extract_all(tar_path: Path, out_dir: Path) -> list[str]:
    """
    Extract every regular file under TAR_ROOT/ from the tar into out_dir,
    stripping the TAR_ROOT/ prefix. Skip non-essential files (images, MACOSX).

    Idempotent: if out_dir already contains the extracted tree (heuristic:
    has at least 5 files and total size > 1 GB), skip extraction.

    Returns sorted list of relative paths that were extracted (or preexisting).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cheap heuristic for "already extracted"
    existing = [p for p in out_dir.rglob("*") if p.is_file()]
    existing_bytes = sum(p.stat().st_size for p in existing)
    if len(existing) >= 5 and existing_bytes > 1_000_000_000:
        log.info("extract dir %s already populated (%d files, %.1f GB) — skipping extract",
                 out_dir, len(existing), existing_bytes / 1e9)
    else:
        log.info("extracting %s -> %s", tar_path, out_dir)
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                # Only pull files under TAR_ROOT/
                if not (name == TAR_ROOT or name.startswith(TAR_ROOT + "/")):
                    log.debug("skip outside-root: %s", name)
                    continue
                rel = name[len(TAR_ROOT) + 1:]  # strip "FLUX.1-dev/"
                if not rel:
                    continue
                if _should_skip(rel):
                    log.info("skip non-weight: %s", rel)
                    continue
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Use a fileobj-based extract so we can control the dest name
                # regardless of member.name mutations.
                src = tf.extractfile(member)
                if src is None:
                    continue
                with dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                log.info("extracted %s (%.1f MB)", rel, dest.stat().st_size / 1e6)

    # Final listing
    rels = sorted(
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*")
        if p.is_file() and not _should_skip(str(p.relative_to(out_dir)))
    )
    if not rels:
        raise RuntimeError(f"no files extracted under {out_dir}")
    return rels


def _load_existing_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError:
        return None


def _remote_sha(s3, bucket: str, key: str) -> str | None:
    """Return the SHA recorded in object metadata, or None if object absent."""
    head = _storage.head_exists(s3, bucket, key)
    if head is None:
        return None
    return head.get("Metadata", {}).get("sha256")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. 'v1'")
    ap.add_argument("--bucket", required=True, help="R2 bucket for weight mirror")
    ap.add_argument("--tar-path", type=Path, default=None,
                    help="Persistent tar cache path (reused across retries)")
    ap.add_argument("--extract-dir", type=Path, default=None,
                    help="Persistent extract dir (reused across retries)")
    ap.add_argument("--also-write-local", type=Path, default=None,
                    help="Also copy weights to this dir (e.g. a Network Volume mount)")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting the manifest file for this version")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download + hash, but do not upload or write manifest")
    args = ap.parse_args()

    existing = _load_existing_manifest()
    if existing and existing.get("version") == args.version and not args.force:
        log.error("manifest for version %s already exists at %s. "
                  "Use --force to reroll, or bump to a new version.",
                  args.version, MANIFEST_PATH)
        return 2

    s3 = _storage.make_client()

    # Decide on working paths: persistent if provided, temp otherwise.
    cleanup_tmp: tempfile.TemporaryDirectory | None = None
    if args.tar_path and args.extract_dir:
        tar_path = args.tar_path
        extract_dir = args.extract_dir
    else:
        cleanup_tmp = tempfile.TemporaryDirectory(prefix="flux_mirror_")
        tmp_path = Path(cleanup_tmp.name)
        tar_path = args.tar_path or (tmp_path / "files.tar")
        extract_dir = args.extract_dir or (tmp_path / "extracted")

    try:
        _download_tar(tar_path)
        rels = _extract_all(tar_path, extract_dir)
        log.info("extracted %d files", len(rels))

        files: dict[str, dict] = {}
        for rel in rels:
            local = extract_dir / rel
            sha = sha256_file(local)
            size = local.stat().st_size
            key = f"{args.version}/{rel}"
            log.info("local file=%s sha=%s bytes=%s", rel, sha[:16], size)
            files[rel] = {"sha256": sha, "bytes": size}

            if args.dry_run:
                continue

            # Idempotency: if the R2 object already exists with matching SHA, skip.
            remote = _remote_sha(s3, args.bucket, key)
            if remote == sha:
                log.info("r2 already has %s with matching sha — skipping upload", key)
            elif remote is not None:
                log.error(
                    "r2 has %s with DIFFERENT sha (remote=%s local=%s). "
                    "This would silently break parity. Pick a new --version.",
                    key, remote[:16], sha[:16],
                )
                return 3
            else:
                log.info("uploading r2://%s/%s", args.bucket, key)
                _storage.upload(s3, args.bucket, key, local)
                # Stamp metadata in a separate copy-in-place.
                s3.copy_object(
                    Bucket=args.bucket,
                    Key=key,
                    CopySource={"Bucket": args.bucket, "Key": key},
                    Metadata={"sha256": sha},
                    MetadataDirective="REPLACE",
                )

            if args.also_write_local is not None:
                dest = Path(args.also_write_local) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, dest)
                copied_sha = sha256_file(dest)
                if copied_sha != sha:
                    log.error("local copy at %s has wrong sha — aborting", dest)
                    return 4
                log.info("also wrote + verified %s", dest)

        manifest = {
            "version":       args.version,
            "source":        REPLICATE_CDN_URL,
            "mirror_bucket": args.bucket,
            "mirror_prefix": f"{args.version}/",
            "snapshot_date": date.today().isoformat(),
            "files":         files,
        }

        if args.dry_run:
            log.info("dry-run manifest:\n%s", json.dumps(manifest, indent=2))
            return 0

        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        log.info("wrote %s — commit this file to git", MANIFEST_PATH)
        return 0
    finally:
        if cleanup_tmp is not None:
            cleanup_tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
