"""Unit tests for serverless._validation — per TESTING.md §2."""
from __future__ import annotations

import copy
import zipfile
from pathlib import Path

import pytest

from serverless._validation import (
    EXIT_BAD_REQUEST,
    EXIT_PARTIAL_CAPS,
    ValidationError,
    check_captions_complete,
    list_images_in_zip,
    parse_request,
    validate_dataset_key,
    validate_lora_prefix,
    validate_r2_key,
)


# ── validate_r2_key ─────────────────────────────────────────────────────────
class TestValidateR2Key:
    def test_accepts_normal_key(self):
        assert validate_r2_key("datasets/x.zip", ("datasets/",)) == "datasets/x.zip"

    @pytest.mark.parametrize("bad", [
        "",                             # empty
        "../etc/passwd",                # parent ref
        "s3://other-bucket/x",          # URL scheme
        "http://evil.com/x",            # URL scheme
        "/absolute/path",               # absolute
        "loras/x",                      # wrong prefix
        "datasets/x\x00null",           # control char
        "a" * 513,                      # too long
    ])
    def test_rejects(self, bad):
        with pytest.raises(ValidationError) as ei:
            validate_r2_key(bad, ("datasets/",))
        assert ei.value.exit_code == EXIT_BAD_REQUEST

    def test_rejects_non_string(self):
        with pytest.raises(ValidationError):
            validate_r2_key(42, ("datasets/",))  # type: ignore[arg-type]

    def test_validate_dataset_key_uses_datasets_prefix(self):
        validate_dataset_key("datasets/ok.zip")
        with pytest.raises(ValidationError):
            validate_dataset_key("loras/nope.zip")

    def test_validate_lora_prefix_format(self):
        p = validate_lora_prefix("user-a", "job-b")
        assert p == "loras/user-a/job-b/"


# ── parse_request ───────────────────────────────────────────────────────────
class TestParseRequest:
    def test_valid(self, valid_request, schema_path):
        got = parse_request(valid_request, schema_path)
        assert got["job_id"] == valid_request["job_id"]

    def test_missing_required(self, valid_request, schema_path):
        r = copy.deepcopy(valid_request)
        del r["trigger_word"]
        with pytest.raises(ValidationError):
            parse_request(r, schema_path)

    def test_bad_dataset_key(self, valid_request, schema_path):
        r = copy.deepcopy(valid_request)
        r["dataset_r2_key"] = "../../etc/passwd"
        with pytest.raises(ValidationError):
            parse_request(r, schema_path)

    @pytest.mark.parametrize("field,value", [
        ("steps", 50),           # below min
        ("steps", 100000),       # above max
        ("lora_rank", 0),
        ("lora_rank", 200),
        ("learning_rate", 1.0),  # way too high
        ("optimizer", "sgd"),    # not in enum
        ("gpu_tier", "v100"),    # not in enum
        ("resolution", "foo"),
    ])
    def test_config_bounds(self, valid_request, schema_path, field, value):
        r = copy.deepcopy(valid_request)
        r["config"][field] = value
        with pytest.raises(ValidationError):
            parse_request(r, schema_path)

    def test_extra_fields_rejected(self, valid_request, schema_path):
        r = copy.deepcopy(valid_request)
        r["surprise"] = True
        with pytest.raises(ValidationError):
            parse_request(r, schema_path)

    def test_captions_required(self, valid_request, schema_path):
        # ADR-0004 revised: captions is required at schema level.
        r = copy.deepcopy(valid_request)
        del r["captions"]
        with pytest.raises(ValidationError):
            parse_request(r, schema_path)


# ── list_images_in_zip + check_captions_complete ────────────────────────────
class TestCaptions:
    def test_list_images_skips_hidden_and_macos(self, sample_zip):
        names = list_images_in_zip(sample_zip)
        assert sorted(names) == ["a.jpg", "b.jpg", "c.jpg"]

    def test_all_present_ok(self):
        check_captions_complete(["a.jpg", "b.jpg"], {"a.jpg": "x", "b.jpg": "y"})

    def test_case_insensitive(self):
        check_captions_complete(["A.JPG"], {"a.jpg": "x"})

    def test_missing_raises_partial_caps(self):
        with pytest.raises(ValidationError) as ei:
            check_captions_complete(["a.jpg", "b.jpg"], {"a.jpg": "x"})
        assert ei.value.exit_code == EXIT_PARTIAL_CAPS

    def test_captions_none_raises_bad_request(self):
        # ADR-0004 revised: LLaVA removed, captions are required.
        with pytest.raises(ValidationError) as ei:
            check_captions_complete(["a.jpg"], None)
        assert ei.value.exit_code == EXIT_BAD_REQUEST

    def test_extra_captions_ignored(self):
        check_captions_complete(["a.jpg"], {"a.jpg": "x", "b.jpg": "y"})
