"""
Request + R2 key validators.

Pure functions. No I/O, no GPU, no boto3 — safe to unit-test.
See DESIGN.md §3 for exit code contract.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import jsonschema

# ── Exit codes (DESIGN.md §3) ────────────────────────────────────────────────
EXIT_OK            = 0
EXIT_SHA_MISMATCH  = 2
EXIT_PARTIAL_CAPS  = 3
EXIT_BAD_REQUEST   = 4
EXIT_R2_READ_FAIL  = 5
EXIT_TRAIN_FAIL    = 6
EXIT_R2_WRITE_FAIL = 7


class ValidationError(Exception):
    """Raised for any client-side contract violation. Maps to exit code 4."""
    def __init__(self, msg: str, exit_code: int = EXIT_BAD_REQUEST):
        super().__init__(msg)
        self.exit_code = exit_code


# ── R2 key validator (ADR-0005) ──────────────────────────────────────────────
_ALLOWED_DATASET_PREFIXES = ("datasets/",)
_ALLOWED_LORA_PREFIXES    = ("loras/",)


def validate_r2_key(key: str, allowed_prefixes: tuple[str, ...]) -> str:
    """
    Reject anything that could cause SSRF or escape the allowed prefix.

    Rules:
      - must be a non-empty str
      - no URL scheme (://)
      - no parent dir refs (..)
      - no absolute path (leading /)
      - no null bytes / control chars
      - must start with one of allowed_prefixes
      - length <= 512
    """
    if not isinstance(key, str) or not key:
        raise ValidationError(f"r2 key must be non-empty string, got {type(key).__name__}")
    if len(key) > 512:
        raise ValidationError("r2 key too long (>512)")
    if "://" in key:
        raise ValidationError(f"r2 key contains URL scheme: {key!r}")
    if ".." in key:
        raise ValidationError(f"r2 key contains parent ref: {key!r}")
    if key.startswith("/"):
        raise ValidationError(f"r2 key is absolute path: {key!r}")
    if any(ord(c) < 0x20 for c in key):
        raise ValidationError("r2 key contains control chars")
    if not key.startswith(allowed_prefixes):
        raise ValidationError(
            f"r2 key {key!r} not in allowed prefixes {allowed_prefixes}"
        )
    return key


def validate_dataset_key(key: str) -> str:
    return validate_r2_key(key, _ALLOWED_DATASET_PREFIXES)


def validate_lora_prefix(user_id: str, job_id: str) -> str:
    """Return the canonical R2 prefix for a job's artifacts. Never taken from input."""
    # Both already validated as UUIDs by schema; safe to interpolate.
    return f"loras/{user_id}/{job_id}/"


# ── Request schema ───────────────────────────────────────────────────────────
_SCHEMA_CACHE: dict[str, Any] = {}


def _load_schema(schema_path: Path) -> dict[str, Any]:
    key = str(schema_path)
    if key not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[key] = json.loads(schema_path.read_text())
    return _SCHEMA_CACHE[key]


def parse_request(raw: dict[str, Any], schema_path: Path) -> dict[str, Any]:
    """
    Validate the RunPod event's `input` against request.v1.json.
    Also runs the R2-key validator (schema alone can't express the prefix rule).
    Returns the parsed dict (same object — no mutation).
    """
    schema = _load_schema(schema_path)
    try:
        jsonschema.validate(raw, schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"schema: {e.message} at {list(e.absolute_path)}") from e

    validate_dataset_key(raw["dataset_r2_key"])
    return raw


# ── Caption completeness (ADR-0004) ──────────────────────────────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images_in_zip(zip_path: Path) -> list[str]:
    """Return basenames of images inside the zip, skipping macOS artifacts."""
    names = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir():
                continue
            if name.startswith("__MACOSX/") or "/._" in name or name.startswith("._"):
                continue
            base = Path(name).name
            if base.startswith(".") or Path(base).suffix.lower() not in _IMG_EXTS:
                continue
            names.append(base)
    return names


def check_captions_complete(
    image_names: list[str],
    captions: dict[str, str] | None,
) -> None:
    """
    ADR-0004:
      captions absent     -> handler relies on --autocaption (caller's choice)
      captions present    -> must cover every image, case-insensitive match
      partial captions    -> abort with EXIT_PARTIAL_CAPS (no silent autocaption)
    """
    if captions is None:
        return  # autocaption path — handler signals --autocaption later

    cap_keys = {k.lower() for k in captions.keys()}
    missing = [img for img in image_names if img.lower() not in cap_keys]
    if missing:
        raise ValidationError(
            f"{len(missing)} image(s) missing captions: {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
            exit_code=EXIT_PARTIAL_CAPS,
        )
