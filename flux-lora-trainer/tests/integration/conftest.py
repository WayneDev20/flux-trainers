"""Integration-test fixtures: moto-backed S3, env vars, fake train.py subprocess."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws


BUCKET_DATASETS = "avatar-datasets"
BUCKET_LORAS    = "avatar-loras"
BUCKET_WEIGHTS  = "flux-weights-mirror"


@pytest.fixture
def moto_env(monkeypatch):
    """Set env vars so _storage.make_client talks to moto, not real R2."""
    monkeypatch.setenv("R2_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://s3.amazonaws.com")  # moto intercepts
    monkeypatch.setenv("R2_BUCKET_DATASETS", BUCKET_DATASETS)
    monkeypatch.setenv("R2_BUCKET_LORAS", BUCKET_LORAS)
    monkeypatch.setenv("R2_BUCKET_WEIGHTS_MIRROR", BUCKET_WEIGHTS)
    monkeypatch.setenv("WEBHOOK_HMAC_SECRET", "test-secret")
    monkeypatch.setenv("FLUX_WEIGHTS_VERSION", "v1")


@pytest.fixture
def s3(moto_env):
    with mock_aws():
        client = boto3.client(
            "s3",
            endpoint_url="https://s3.amazonaws.com",
            aws_access_key_id="test-access",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
        )
        for b in (BUCKET_DATASETS, BUCKET_LORAS, BUCKET_WEIGHTS):
            client.create_bucket(Bucket=b)
        yield client


@pytest.fixture
def populated_weights_mirror(s3, fake_weights):
    """Upload fake FLUX weights to moto-backed weights-mirror bucket."""
    wdir, manifest = fake_weights
    for fn in manifest["files"]:
        key = manifest["mirror_prefix"] + fn
        s3.put_object(Bucket=BUCKET_WEIGHTS, Key=key, Body=(wdir / fn).read_bytes())
    return manifest


@pytest.fixture
def installed_manifest(tmp_path, monkeypatch, fake_weights):
    """
    Put the manifest file where handler.py expects it + point
    FLUX_WEIGHTS_DIR at the tmp weights dir so the fetch path isn't exercised.
    """
    wdir, manifest = fake_weights
    # Manifest in place
    mf_path = tmp_path / "flux_weights_manifest.json"
    mf_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("serverless.handler.MANIFEST_PATH", mf_path)
    monkeypatch.setattr("serverless.handler.FLUX_WEIGHTS_DIR", wdir)
    return manifest


@pytest.fixture
def uploaded_dataset(s3, sample_zip, valid_request):
    """Upload the test dataset to moto at the location valid_request expects."""
    s3.put_object(
        Bucket=BUCKET_DATASETS,
        Key=valid_request["dataset_r2_key"],
        Body=sample_zip.read_bytes(),
    )


@pytest.fixture
def fake_train(monkeypatch, tmp_path):
    """
    Replace subprocess.Popen inside _training with a fake that "produces" a
    lora.safetensors in TRAIN_OUTPUT_DIR, then exits 0.

    Returns a MagicMock so tests can tweak returncode / stdout per-case.
    """
    train_out = tmp_path / "flux_train_replicate"
    train_out.mkdir()
    monkeypatch.setattr("serverless._training.TRAIN_OUTPUT_DIR", train_out)

    proc = MagicMock()
    proc.stdout = iter(["loaded\n", "step 1/100\n", "step 100/100\n", "done\n"])
    proc.wait = MagicMock()
    proc.returncode = 0

    def popen(argv, **kw):
        # Side effect: write a fake LoRA as if train.py succeeded
        (train_out / "lora.safetensors").write_bytes(b"fake-lora-bytes")
        (train_out / "samples").mkdir(exist_ok=True)
        (train_out / "samples" / "0001.jpg").write_bytes(b"\xff\xd8\xff")
        return proc

    # run_training binds `_popen=subprocess.Popen` as a default at import time,
    # so rebinding subprocess.Popen via monkeypatch does nothing. Replace the
    # function's captured default directly (kw-only → __kwdefaults__).
    import serverless._training as _training_mod
    rt = _training_mod.run_training
    if rt.__kwdefaults__ and "_popen" in rt.__kwdefaults__:
        patched = dict(rt.__kwdefaults__)
        patched["_popen"] = popen
        monkeypatch.setattr(rt, "__kwdefaults__", patched)
    return proc


@pytest.fixture
def no_progress_updates(monkeypatch):
    monkeypatch.setattr("runpod.serverless.progress_update", lambda *a, **k: None,
                        raising=False)
