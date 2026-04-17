"""Unit tests for serverless._weights."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from serverless._weights import (
    WeightParityError,
    load_manifest,
    sha256_file,
    verify_weights,
)


class TestSha256File:
    def test_matches_hashlib(self, tmp_path):
        import hashlib
        p = tmp_path / "x.bin"
        body = b"hello world" * 1000
        p.write_bytes(body)
        assert sha256_file(p) == hashlib.sha256(body).hexdigest()

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty"
        p.write_bytes(b"")
        assert sha256_file(p) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class TestLoadManifest:
    def test_happy_path(self, tmp_path):
        m = {"version": "v1", "files": {"f": {"sha256": "a", "bytes": 1}}}
        p = tmp_path / "mf.json"
        p.write_text(json.dumps(m))
        assert load_manifest(p, "v1")["version"] == "v1"

    def test_missing_file(self, tmp_path):
        with pytest.raises(WeightParityError):
            load_manifest(tmp_path / "nope.json", "v1")

    def test_wrong_version(self, tmp_path):
        p = tmp_path / "mf.json"
        p.write_text(json.dumps({"version": "v1", "files": {"f": {}}}))
        with pytest.raises(WeightParityError):
            load_manifest(p, "v2")

    def test_empty_files(self, tmp_path):
        p = tmp_path / "mf.json"
        p.write_text(json.dumps({"version": "v1", "files": {}}))
        with pytest.raises(WeightParityError):
            load_manifest(p, "v1")


class TestVerifyWeights:
    def test_happy_path(self, fake_weights):
        wdir, manifest = fake_weights
        verify_weights(manifest, wdir)  # no raise

    def test_detects_sha_mismatch(self, fake_weights):
        wdir, manifest = fake_weights
        # Tamper one file
        f = next(iter(manifest["files"]))
        (wdir / f).write_bytes(b"different content")
        with pytest.raises(WeightParityError) as ei:
            verify_weights(manifest, wdir)
        assert "sha mismatch" in str(ei.value)

    def test_missing_file_no_fetcher(self, fake_weights):
        wdir, manifest = fake_weights
        f = next(iter(manifest["files"]))
        (wdir / f).unlink()
        with pytest.raises(WeightParityError):
            verify_weights(manifest, wdir)

    def test_missing_file_with_fetcher(self, fake_weights, tmp_path):
        wdir, manifest = fake_weights
        f = next(iter(manifest["files"]))
        original = (wdir / f).read_bytes()
        (wdir / f).unlink()

        def fetcher(filename, local):
            local.write_bytes(original)

        verify_weights(manifest, wdir, fetch_missing=fetcher)  # no raise

    def test_size_mismatch(self, fake_weights):
        wdir, manifest = fake_weights
        # Keep SHA field correct by recomputing, but poison byte count
        f = next(iter(manifest["files"]))
        manifest["files"][f]["bytes"] = 999_999
        with pytest.raises(WeightParityError) as ei:
            verify_weights(manifest, wdir)
        assert "size mismatch" in str(ei.value)
