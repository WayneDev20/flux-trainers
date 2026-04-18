"""
FLUX base-weight parity verification (ADR-0003).

On boot: every file in the pinned manifest must exist locally and its SHA-256
must match. On mismatch, abort with EXIT_SHA_MISMATCH — LoRAs trained against
drifted weights are not interchangeable with inference-time weights.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from ._validation import EXIT_SHA_MISMATCH, ValidationError

log = logging.getLogger("handler._weights")

_CHUNK = 1 << 20  # 1 MiB


class WeightParityError(Exception):
    """Raised for any base-weight integrity issue. Maps to exit code 2."""
    def __init__(self, msg: str):
        super().__init__(msg)
        self.exit_code = EXIT_SHA_MISMATCH


def sha256_file(path: Path) -> str:
    """Stream-hash a file. Does NOT load into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(manifest_path: Path, expected_version: str) -> dict[str, Any]:
    """Load + sanity-check the committed manifest."""
    if not manifest_path.exists():
        raise WeightParityError(
            f"manifest missing at {manifest_path} — run mirror_flux_weights.py"
        )
    m = json.loads(manifest_path.read_text())
    if m.get("version") != expected_version:
        raise WeightParityError(
            f"manifest version {m.get('version')!r} != env {expected_version!r}"
        )
    if not m.get("files"):
        raise WeightParityError("manifest has no files entry")
    return m


_MARKER_NAME = ".verified.json"


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Stable fingerprint over the files map — independent of key order."""
    canonical = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_weights(
    manifest: dict[str, Any],
    weights_dir: Path,
    *,
    fetch_missing: Callable[[str, Path], None] | None = None,
) -> None:
    """
    Verify each file in `manifest["files"]` exists under weights_dir and its
    SHA matches. If `fetch_missing(filename, local_path)` is provided, it's
    called to populate any missing file before hashing — used for cold-cold
    starts where the Network Volume hasn't been warmed.

    Fast-path: if `.verified.json` exists in `weights_dir` with the same
    manifest fingerprint, we trust the previous full SHA sweep and only
    stat-check that each file still exists with the expected size. This
    is the difference between a 5s cold-start (fast path) and a ~4 min
    cold-start (re-hashing 34 GB from a network volume every boot).

    Raises WeightParityError on any mismatch or missing file that can't be
    fetched.
    """
    fingerprint = _manifest_fingerprint(manifest)
    marker = weights_dir / _MARKER_NAME

    # ── Fast path: marker present + fingerprint match + all files present ──
    if marker.exists():
        try:
            cached = json.loads(marker.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get("fingerprint") == fingerprint:
            all_ok = True
            for filename, meta in manifest["files"].items():
                local = weights_dir / filename
                if not local.exists():
                    log.info("weights.fastpath.miss file=%s reason=absent", filename)
                    all_ok = False
                    break
                if "bytes" in meta and local.stat().st_size != meta["bytes"]:
                    log.info("weights.fastpath.miss file=%s reason=size", filename)
                    all_ok = False
                    break
            if all_ok:
                log.info("weights.fastpath.hit fingerprint=%s", fingerprint[:16])
                return

    # ── Slow path: fetch any missing + SHA-verify all ──
    log.info("weights.fullcheck.start fingerprint=%s", fingerprint[:16])
    for filename, meta in manifest["files"].items():
        local = weights_dir / filename
        if not local.exists():
            if fetch_missing is None:
                raise WeightParityError(f"{filename} missing and no fetcher supplied")
            log.info("weights.fetching file=%s", filename)
            local.parent.mkdir(parents=True, exist_ok=True)
            fetch_missing(filename, local)

        actual = sha256_file(local)
        expected = meta["sha256"]
        if actual != expected:
            raise WeightParityError(
                f"sha mismatch file={filename} expected={expected[:16]}... "
                f"actual={actual[:16]}..."
            )

        size = local.stat().st_size
        if "bytes" in meta and size != meta["bytes"]:
            raise WeightParityError(
                f"size mismatch file={filename} expected={meta['bytes']} "
                f"actual={size}"
            )
        log.info("weights.verified file=%s sha=%s...", filename, expected[:16])

    # Write the marker so the next boot can take the fast path.
    try:
        marker.write_text(json.dumps({
            "fingerprint": fingerprint,
            "version":     manifest.get("version"),
            "file_count":  len(manifest["files"]),
        }))
        log.info("weights.marker.written fingerprint=%s", fingerprint[:16])
    except OSError as e:
        # Writing the marker is an optimization — don't fail the job over it.
        log.warning("weights.marker.write_failed err=%s", e)
