"""
Subprocess wrapper around /workspace/train.py.

Pure-ish: `build_argv` has no I/O and is fully unit-testable. `run_training`
does I/O (subprocess) but takes a `_popen` injection point for tests.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ._validation import EXIT_TRAIN_FAIL

log = logging.getLogger("handler._training")


class TrainingError(Exception):
    """train.py exited non-zero or output was malformed. Maps to exit code 6."""
    def __init__(self, msg: str, oom: bool = False, stderr_tail: str = ""):
        super().__init__(msg)
        self.exit_code = EXIT_TRAIN_FAIL
        self.oom = oom
        self.stderr_tail = stderr_tail


# train.py constants (mirror of its hard-coded paths)
TRAIN_PY = Path("/workspace/train.py")
TRAIN_OUTPUT_DIR = Path("/workspace/output/flux_train_replicate")
TRAIN_LORA_NAME = "lora.safetensors"  # train.py renames its output to this name


def build_argv(
    config: dict[str, Any],
    *,
    input_zip: Path,
    trigger_word: str,
    captions_provided: bool,  # retained for call-site compatibility; no-op
    num_gpus: int = 1,
    train_py: Path = TRAIN_PY,
    python: str = sys.executable,
) -> list[str]:
    """
    Construct argv for train.py from the validated 8-knob config{}.
    Schema guarantees all 8 keys exist — no `.get(..., default)` needed.

    Captions are expected to be present as <stem>.txt alongside each image
    in the dataset zip. The handler injects them from the request's
    `captions` map before calling train.py. LLaVA's in-process autocaption
    fallback was removed — see ADR-0004 revision log and the base
    Dockerfile header.

    When num_gpus > 1 we wrap with `accelerate launch` so ai-toolkit's
    FSDP v2 path (luke9705 PR #774) triggers: it keys off
    `Accelerator.num_processes > 1` and shards the frozen FLUX transformer
    across GPUs while LoRA params stay replicated with synced grads.
    """
    del captions_provided  # signal that we intentionally ignore this
    train_args = [
        "--input_images",  str(input_zip),
        "--trigger_word",  trigger_word,
        "--steps",                 str(config["steps"]),
        "--lora_rank",             str(config["lora_rank"]),
        "--learning_rate",         str(config["learning_rate"]),
        "--batch_size",            str(config["batch_size"]),
        "--resolution",            config["resolution"],
        "--optimizer",             config["optimizer"],
        "--caption_dropout_rate",  str(config["caption_dropout_rate"]),
    ]
    if num_gpus > 1:
        return [
            "accelerate", "launch",
            f"--num_processes={num_gpus}",
            "--multi_gpu",
            "--mixed_precision=bf16",
            str(train_py),
            *train_args,
        ]
    return [python, str(train_py), *train_args]


def _classify_error(stderr_tail: str) -> tuple[bool, str]:
    """
    Inspect the tail of train.py's output and classify.
    Returns (is_oom, short_reason).
    """
    low = stderr_tail.lower()
    if "cuda out of memory" in low or "cublas_status_alloc_failed" in low:
        return True, "cuda_oom"
    if "no space left on device" in low:
        return False, "disk_full"
    if "modulenotfounderror" in low:
        return False, "import_error"
    return False, "nonzero_exit"


def run_training(
    argv: list[str],
    *,
    log_path: Path,
    progress_cb: Callable[[str], None] | None = None,
    _popen=subprocess.Popen,
) -> Path:
    """
    Launch train.py. Stream combined stdout/stderr to log_path AND to our
    stdout (for RunPod capture). Rate-limit progress callbacks to once per
    10 s to avoid blowing up RunPod's log volume.

    Returns the path to the produced LoRA.
    Raises TrainingError on non-zero exit.
    """
    log.info("train.launch argv=%s", " ".join(argv))
    last_progress = 0.0
    tail: list[str] = []  # last ~200 lines, for error classification
    TAIL_MAX = 200

    with log_path.open("w") as logf:
        proc = _popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            sys.stdout.write(line)
            tail.append(line)
            if len(tail) > TAIL_MAX:
                tail.pop(0)
            if progress_cb and (time.time() - last_progress > 10) and "step" in line.lower():
                last_progress = time.time()
                try:
                    progress_cb(line.strip()[-200:])
                except Exception:
                    pass  # progress reporting is never fatal
        proc.wait()

    if proc.returncode != 0:
        stderr_tail = "".join(tail)
        oom, reason = _classify_error(stderr_tail)
        raise TrainingError(
            f"train.py exit={proc.returncode} reason={reason}",
            oom=oom,
            stderr_tail=stderr_tail[-4000:],
        )

    lora = TRAIN_OUTPUT_DIR / TRAIN_LORA_NAME
    if not lora.exists():
        # train.py renames to lora.safetensors at the end; if that step failed
        # partway through, fall back to any .safetensors in the job dir.
        candidates = sorted(TRAIN_OUTPUT_DIR.glob("*.safetensors"))
        if not candidates:
            raise TrainingError(
                f"train.py exited 0 but no .safetensors in {TRAIN_OUTPUT_DIR}"
            )
        lora = candidates[-1]
    return lora
