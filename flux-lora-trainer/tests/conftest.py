"""Shared pytest fixtures."""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def schema_path() -> Path:
    return Path(__file__).parents[1] / "serverless" / "schemas" / "request.v1.json"


@pytest.fixture
def webhook_secret() -> bytes:
    return b"test-secret-do-not-use-in-prod"


@pytest.fixture
def valid_request() -> dict:
    return {
        "job_id":  "11111111-2222-4333-8444-555555555555",
        "user_id": "66666666-7777-4888-8999-aaaaaaaaaaaa",
        "dataset_r2_key": "datasets/11111111-2222-4333-8444-555555555555.zip",
        "trigger_word": "TOK",
        "captions": {
            "a.jpg": "TOK, person, front, studio",
            "b.jpg": "TOK, person, 3/4, studio",
            "c.jpg": "TOK, person, profile, studio",
        },
        "webhook_url": "https://backend.example.com/hooks/training",
        "config": {
            "steps": 1000,
            "lora_rank": 32,
            "learning_rate": 0.001,
            "batch_size": 1,
            "resolution": "512,768,1024",
            "optimizer": "adamw8bit",
            "caption_dropout_rate": 0.05,
            "gpu_tier": "a100_80gb",
        },
    }


@pytest.fixture
def sample_zip(tmp_path: Path) -> Path:
    """Build a zip with three tiny JPEGs whose basenames match `valid_request`."""
    zp = tmp_path / "sample.zip"
    # 3-byte JPEG magic is enough to make the file "look like" a jpeg —
    # we only test list_images_in_zip, which keys on extension + name.
    magic = b"\xff\xd8\xff" + b"\x00" * 60
    with zipfile.ZipFile(zp, "w") as z:
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            z.writestr(name, magic)
        # macOS artifact must be ignored
        z.writestr("__MACOSX/a.jpg", b"")
        z.writestr("._b.jpg", b"")
    return zp


@pytest.fixture
def fake_weights_dir(tmp_path: Path) -> Path:
    """
    Build a fake FLUX weights dir with 4 small files and return (dir, manifest dict).
    Sizes/contents differ per-file so SHAs are deterministic and distinct.
    """
    import hashlib

    wdir = tmp_path / "FLUX.1-dev"
    wdir.mkdir()
    return wdir


@pytest.fixture
def fake_weights(fake_weights_dir: Path, tmp_path: Path):
    """Creates 4 fake weight files + a matching manifest dict."""
    import hashlib

    filenames = ["flux1-dev.safetensors", "ae.safetensors",
                 "t5xxl_fp16.safetensors", "clip_l.safetensors"]
    files: dict = {}
    for i, fn in enumerate(filenames):
        p = fake_weights_dir / fn
        body = (f"fake-{fn}-".encode() + bytes([i]) * 1024)
        p.write_bytes(body)
        files[fn] = {
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes":  len(body),
        }
    manifest = {
        "version": "v1",
        "source": "https://example/files.tar",
        "mirror_bucket": "flux-weights-mirror",
        "mirror_prefix": "v1/",
        "snapshot_date": "2026-04-17",
        "files": files,
    }
    return fake_weights_dir, manifest
