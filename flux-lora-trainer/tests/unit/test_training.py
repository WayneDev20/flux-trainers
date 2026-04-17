"""Unit tests for serverless._training."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from serverless._training import (
    TrainingError,
    _classify_error,
    build_argv,
    run_training,
)


class TestBuildArgv:
    def test_all_flags_present(self, valid_request):
        argv = build_argv(
            valid_request["config"],
            input_zip=Path("/tmp/x.zip"),
            trigger_word="TOK",
            captions_provided=True,
        )
        # Required flags that must appear
        for f in ("--input_images", "--trigger_word", "--steps", "--lora_rank",
                  "--learning_rate", "--batch_size", "--resolution",
                  "--optimizer", "--caption_dropout_rate"):
            assert f in argv

    def test_captions_provided_disables_autocaption(self, valid_request):
        argv = build_argv(
            valid_request["config"],
            input_zip=Path("/tmp/x.zip"),
            trigger_word="TOK",
            captions_provided=True,
        )
        assert "--no-autocaption" in argv
        assert "--autocaption" not in argv

    def test_captions_absent_enables_autocaption(self, valid_request):
        argv = build_argv(
            valid_request["config"],
            input_zip=Path("/tmp/x.zip"),
            trigger_word="TOK",
            captions_provided=False,
        )
        assert "--autocaption" in argv
        assert "--no-autocaption" not in argv

    def test_values_pass_through(self, valid_request):
        argv = build_argv(
            valid_request["config"],
            input_zip=Path("/z.zip"),
            trigger_word="MYTRIG",
            captions_provided=True,
        )
        # argv is flat list of strings; find index of each flag
        assert argv[argv.index("--trigger_word") + 1] == "MYTRIG"
        assert argv[argv.index("--lora_rank") + 1] == "32"
        assert argv[argv.index("--optimizer") + 1] == "adamw8bit"


class TestClassifyError:
    def test_oom(self):
        oom, reason = _classify_error("foo\nCUDA out of memory. Tried to allocate")
        assert oom is True
        assert reason == "cuda_oom"

    def test_disk_full(self):
        oom, reason = _classify_error("No space left on device")
        assert oom is False
        assert reason == "disk_full"

    def test_import_error(self):
        oom, reason = _classify_error("ModuleNotFoundError: no module named 'foo'")
        assert oom is False
        assert reason == "import_error"

    def test_unknown(self):
        oom, reason = _classify_error("weird error")
        assert oom is False
        assert reason == "nonzero_exit"


class TestRunTraining:
    def test_happy_path(self, tmp_path, monkeypatch):
        # Point the output dir at tmp_path so lora.safetensors resolution works
        monkeypatch.setattr("serverless._training.TRAIN_OUTPUT_DIR", tmp_path)
        lora = tmp_path / "lora.safetensors"
        lora.write_bytes(b"fake-lora")

        proc = MagicMock()
        proc.stdout = iter(["step 1/100\n", "step 2/100\n"])
        proc.wait = MagicMock()
        proc.returncode = 0
        popen = MagicMock(return_value=proc)

        out = run_training(
            ["python", "train.py"],
            log_path=tmp_path / "log.txt",
            _popen=popen,
        )
        assert out == lora

    def test_nonzero_exit_raises_training_error(self, tmp_path):
        proc = MagicMock()
        proc.stdout = iter(["boom\n", "CUDA out of memory\n"])
        proc.wait = MagicMock()
        proc.returncode = 1
        popen = MagicMock(return_value=proc)

        with pytest.raises(TrainingError) as ei:
            run_training(
                ["python", "train.py"],
                log_path=tmp_path / "log.txt",
                _popen=popen,
            )
        assert ei.value.oom is True
        assert "cuda_oom" in str(ei.value)

    def test_missing_lora_raises(self, tmp_path, monkeypatch):
        # Empty output dir — no .safetensors
        monkeypatch.setattr("serverless._training.TRAIN_OUTPUT_DIR", tmp_path)

        proc = MagicMock()
        proc.stdout = iter([])
        proc.wait = MagicMock()
        proc.returncode = 0
        popen = MagicMock(return_value=proc)

        with pytest.raises(TrainingError):
            run_training(
                ["python", "train.py"],
                log_path=tmp_path / "log.txt",
                _popen=popen,
            )
