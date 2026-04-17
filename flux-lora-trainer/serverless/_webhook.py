"""
HMAC-SHA256 webhook signing + send with retry.
Pure enough: the HTTP POST is isolated in one function for mocking.
See ADR-0005.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import requests

log = logging.getLogger("handler._webhook")


def sign(body: bytes, secret: bytes) -> str:
    """Return hex-encoded HMAC-SHA256 signature."""
    if not secret:
        raise ValueError("webhook secret is empty; refusing to sign")
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def verify(body: bytes, secret: bytes, signature: str) -> bool:
    """Constant-time signature check. Use this in avatar-backend; included here for symmetry + tests."""
    if not secret or not signature:
        return False
    expected = sign(body, secret)
    return hmac.compare_digest(expected, signature)


def canonical_body(payload: dict[str, Any]) -> bytes:
    """Canonical JSON — stable field order, no whitespace — so sender and receiver compute the same bytes."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def send(
    url: str | None,
    payload: dict[str, Any],
    secret: bytes,
    *,
    max_retries: int = 3,
    timeout_s: float = 30.0,
    _post=requests.post,  # injectable for tests
    _sleep=time.sleep,
) -> bool:
    """
    POST the payload with X-Signature header. Retry on 5xx / network errors
    with exponential backoff (1s, 2s, 4s). Returns True on 2xx, False otherwise.
    Non-fatal: the completion record is in R2; backend reconciler covers gaps.
    """
    if not url:
        log.warning("webhook.skipped: no url configured")
        return False

    body = canonical_body(payload)
    headers = {
        "Content-Type": "application/json",
        "X-Signature": f"sha256={sign(body, secret)}",
        "X-Webhook-Version": "v1",
    }

    backoff = 1.0
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = _post(url, data=body, headers=headers, timeout=timeout_s)
            if 200 <= r.status_code < 300:
                log.info("webhook.sent status=%d attempt=%d", r.status_code, attempt)
                return True
            last_err = f"HTTP {r.status_code}"
            if r.status_code < 500:
                # 4xx = client bug, no point retrying
                log.error("webhook.4xx status=%d body=%s", r.status_code, r.text[:200])
                return False
        except requests.RequestException as e:
            last_err = type(e).__name__

        if attempt < max_retries:
            _sleep(backoff)
            backoff *= 2

    log.error("webhook.failed after %d attempts: %s", max_retries, last_err)
    return False
