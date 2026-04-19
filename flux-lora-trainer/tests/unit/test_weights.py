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


class TestFastPath:
    def test_writes_marker_after_full_verify(self, fake_weights):
        wdir, manifest = fake_weights
        verify_weights(manifest, wdir)
        marker = wdir / ".verified.json"
        assert marker.exists()
        body = json.loads(marker.read_text())
        assert "fingerprint" in body
        assert body["version"] == manifest["version"]
        assert body["file_count"] == len(manifest["files"])

    def test_second_call_takes_fast_path_without_hashing(self, fake_weights):
        """If the fast path is taken, tampering a file's contents (but keeping
        size) won't be detected — that's the whole point of the optimization."""
        wdir, manifest = fake_weights
        verify_weights(manifest, wdir)
        f = next(iter(manifest["files"]))
        body = (wdir / f).read_bytes()
        # Overwrite with different content but same length → slow path would
        # catch this via SHA; fast path (size+presence only) must not.
        (wdir / f).write_bytes(b"X" * len(body))
        verify_weights(manifest, wdir)  # no raise — fast path

    def test_size_change_invalidates_fast_path(self, fake_weights):
        wdir, manifest = fake_weights
        verify_weights(manifest, wdir)
        # Corrupt size only → fast path falls through to slow path → raises
        f = next(iter(manifest["files"]))
        (wdir / f).write_bytes(b"short")
        with pytest.raises(WeightParityError):
            verify_weights(manifest, wdir)

    def test_missing_file_invalidates_fast_path(self, fake_weights):
        wdir, manifest = fake_weights
        verify_weights(manifest, wdir)
        f = next(iter(manifest["files"]))
        (wdir / f).unlink()
        with pytest.raises(WeightParityError):
            verify_weights(manifest, wdir)  # no fetcher → slow path raises

    def test_manifest_fingerprint_change_forces_slow_path(self, fake_weights):
        wdir, manifest = fake_weights
        verify_weights(manifest, wdir)
        # New manifest (different fingerprint) but same files on disk →
        # slow path runs; since the new manifest's SHAs don't match the
        # files on disk, it raises.
        bumped = {**manifest, "files": dict(manifest["files"])}
        f = next(iter(bumped["files"]))
        bumped["files"][f] = {**bumped["files"][f], "sha256": "deadbeef" * 8}
        with pytest.raises(WeightParityError):
            verify_weights(bumped, wdir)
