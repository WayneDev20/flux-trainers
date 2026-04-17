"""Unit tests for the pure helpers in handler.py."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


class TestInjectCaptionTxt:
    def test_adds_txt_next_to_each_image(self, sample_zip, tmp_path):
        from serverless.handler import _inject_caption_txt
        caps = {"a.jpg": "cap-a", "b.jpg": "cap-b", "c.jpg": "cap-c"}
        out = _inject_caption_txt(sample_zip, caps, tmp_path)

        with zipfile.ZipFile(out, "r") as z:
            names = set(z.namelist())
        assert {"a.jpg", "b.jpg", "c.jpg",
                "a.txt", "b.txt", "c.txt"}.issubset(names)
        assert "__MACOSX/a.jpg" not in names  # stripped
        assert "._b.jpg" not in names
        with zipfile.ZipFile(out, "r") as z:
            assert z.read("a.txt").decode() == "cap-a"

    def test_case_insensitive_match(self, tmp_path):
        from serverless.handler import _inject_caption_txt
        zp = tmp_path / "src.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("MyPic.JPG", b"\xff\xd8\xff")
        out = _inject_caption_txt(zp, {"mypic.jpg": "lowercase-cap"}, tmp_path)
        with zipfile.ZipFile(out, "r") as z:
            assert z.read("MyPic.txt").decode() == "lowercase-cap"


class TestBuildManifest:
    def test_shape(self, valid_request, tmp_path):
        from serverless.handler import _build_manifest
        lora = tmp_path / "lora.safetensors"
        lora.write_bytes(b"deadbeef")

        weights_m = {"version": "v1", "source": "https://example"}
        m = _build_manifest(
            req=valid_request, weights_manifest=weights_m,
            duration_s=123, image_count=15, lora_path=lora,
        )
        assert m["schema_version"] == "training_manifest.v1"
        assert m["weights_version"] == "v1"
        assert m["duration_s"] == 123
        assert m["image_count"] == 15
        assert len(m["lora_sha256"]) == 64
        assert m["job_id"] == valid_request["job_id"]
        # json-serialisable
        json.dumps(m)
