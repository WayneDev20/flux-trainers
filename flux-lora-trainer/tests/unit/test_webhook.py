"""Unit tests for serverless._webhook."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from serverless._webhook import canonical_body, send, sign, verify


class TestSignVerify:
    def test_sign_deterministic(self, webhook_secret):
        b = b'{"a":1}'
        assert sign(b, webhook_secret) == sign(b, webhook_secret)

    def test_sign_different_for_different_bodies(self, webhook_secret):
        assert sign(b"x", webhook_secret) != sign(b"y", webhook_secret)

    def test_sign_different_for_different_secrets(self):
        assert sign(b"x", b"s1") != sign(b"x", b"s2")

    def test_sign_refuses_empty_secret(self):
        with pytest.raises(ValueError):
            sign(b"x", b"")

    def test_verify_roundtrip(self, webhook_secret):
        body = b"hello"
        s = sign(body, webhook_secret)
        assert verify(body, webhook_secret, s) is True

    def test_verify_detects_bitflip(self, webhook_secret):
        body = b"hello"
        s = sign(body, webhook_secret)
        tampered = s[:-1] + ("0" if s[-1] != "0" else "1")
        assert verify(body, webhook_secret, tampered) is False

    def test_verify_rejects_wrong_secret(self, webhook_secret):
        body = b"hello"
        s = sign(body, webhook_secret)
        assert verify(body, b"other-secret", s) is False


class TestCanonicalBody:
    def test_sorted_keys_and_no_whitespace(self):
        b = canonical_body({"b": 1, "a": 2})
        assert b == b'{"a":2,"b":1}'


class TestSend:
    def test_skips_without_url(self, webhook_secret):
        assert send(None, {"ok": 1}, webhook_secret) is False

    def test_happy_path(self, webhook_secret):
        resp = MagicMock(status_code=200)
        post = MagicMock(return_value=resp)
        ok = send("https://x", {"ok": 1}, webhook_secret, _post=post)
        assert ok is True
        post.assert_called_once()
        _, kw = post.call_args
        assert "X-Signature" in kw["headers"]
        assert kw["headers"]["X-Signature"].startswith("sha256=")

    def test_4xx_no_retry(self, webhook_secret):
        resp = MagicMock(status_code=400, text="bad")
        post = MagicMock(return_value=resp)
        ok = send("https://x", {}, webhook_secret, _post=post, _sleep=lambda _: None)
        assert ok is False
        assert post.call_count == 1  # no retry on 4xx

    def test_5xx_retries_then_fails(self, webhook_secret):
        resp = MagicMock(status_code=503, text="")
        post = MagicMock(return_value=resp)
        ok = send("https://x", {}, webhook_secret,
                  max_retries=3, _post=post, _sleep=lambda _: None)
        assert ok is False
        assert post.call_count == 3

    def test_5xx_recovers_on_retry(self, webhook_secret):
        bad = MagicMock(status_code=503, text="")
        good = MagicMock(status_code=200)
        post = MagicMock(side_effect=[bad, good])
        ok = send("https://x", {}, webhook_secret,
                  max_retries=3, _post=post, _sleep=lambda _: None)
        assert ok is True
        assert post.call_count == 2
