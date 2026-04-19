"""
R2 (S3-compatible) client wrapper with explicit retry semantics for reads + writes.

Boto3 has its own retry config, but we layer on a small extra retry for the
specific operations we care about (head/get/put) so DESIGN.md's exit codes 5
and 7 are emitted deterministically instead of bubbling raw ClientError.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError, EndpointConnectionError

from ._validation import EXIT_R2_READ_FAIL, EXIT_R2_WRITE_FAIL

log = logging.getLogger("handler._storage")


class StorageReadError(Exception):
    exit_code = EXIT_R2_READ_FAIL


class StorageWriteError(Exception):
    exit_code = EXIT_R2_WRITE_FAIL


def make_client(env: dict[str, str] | None = None) -> Any:
    """
    Build a boto3 S3 client pointed at R2. `env` is injectable for tests
    (pass moto creds + endpoint there).
    """
    e = env if env is not None else os.environ
    account_id = e["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=e.get(
            "R2_ENDPOINT_URL",
            f"https://{account_id}.r2.cloudflarestorage.com",
        ),
        aws_access_key_id=e["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=e["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            # Training runs > 1 h; don't let boto time out mid-upload.
            read_timeout=300,
            connect_timeout=30,
        ),
        region_name="auto",
    )


# ── Retry helper ─────────────────────────────────────────────────────────────
_RETRYABLE = (EndpointConnectionError,)
_RETRYABLE_CODES = {"RequestTimeout", "SlowDown", "ServiceUnavailable",
                    "InternalError", "503", "500"}


def _with_retry(op: Callable[[], Any], *, what: str, max_attempts: int = 3,
                _sleep=time.sleep) -> Any:
    backoff = 1.0
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return op()
        except _RETRYABLE as e:
            last = e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in _RETRYABLE_CODES:
                raise
            last = e
        log.warning("storage.retry op=%s attempt=%d err=%s", what, attempt, last)
        if attempt < max_attempts:
            _sleep(backoff)
            backoff *= 2
    assert last is not None
    raise last


# ── Typed operations with mapped exceptions ──────────────────────────────────
def head_exists(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    """Return the head response dict if the object exists, None on 404. Raises StorageReadError on anything else."""
    try:
        return _with_retry(
            lambda: client.head_object(Bucket=bucket, Key=key),
            what=f"head {bucket}/{key}",
        )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise StorageReadError(f"head {bucket}/{key}: {e}") from e


def download(client: Any, bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _with_retry(
            lambda: client.download_file(bucket, key, str(dest)),
            what=f"get {bucket}/{key}",
        )
    except Exception as e:
        raise StorageReadError(f"download {bucket}/{key} -> {dest}: {e}") from e


def upload(client: Any, bucket: str, key: str, src: Path,
           content_type: str = "application/octet-stream") -> None:
    try:
        _with_retry(
            lambda: client.upload_file(
                str(src), bucket, key,
                ExtraArgs={"ContentType": content_type},
            ),
            what=f"put {bucket}/{key}",
        )
    except Exception as e:
        raise StorageWriteError(f"upload {src} -> {bucket}/{key}: {e}") from e


def put_bytes(client: Any, bucket: str, key: str, body: bytes,
              content_type: str) -> None:
    try:
        _with_retry(
            lambda: client.put_object(
                Bucket=bucket, Key=key, Body=body, ContentType=content_type,
            ),
            what=f"put {bucket}/{key}",
        )
    except Exception as e:
        raise StorageWriteError(f"put {bucket}/{key}: {e}") from e


def get_bytes(client: Any, bucket: str, key: str) -> bytes:
    try:
        resp = _with_retry(
            lambda: client.get_object(Bucket=bucket, Key=key),
            what=f"get {bucket}/{key}",
        )
        return resp["Body"].read()
    except Exception as e:
        raise StorageReadError(f"get {bucket}/{key}: {e}") from e
