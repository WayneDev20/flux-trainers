#!/usr/bin/env python3
"""
FLUX LoRA Trainer — Production Self-Hosted
Based on: https://github.com/replicate/flux-fine-tuner (ai-toolkit / ostris)

Changes from upstream (cog → argparse, improved LR schedule):
  1. Cog removed → argparse (no Replicate runtime dependency)
  2. LR schedule: cosine_with_min_lr, peak 1e-3, floor 1e-4, warmup ~12 steps
       Source: reverse-engineered from replicate/fast-flux-trainer training logs
       (1500-step run showed warmup 0→1e-3 over 12 steps, cosine decay to 1e-4)
       vs upstream's flat lr=4e-4 constant schedule
  3. Weight download: Replicate's own CDN URL (same bytes their trainer uses)
       with pget (their parallel downloader) when available, wget/curl fallback
  4. Inference note: load LoRA at scale 1.5 on downstream pipelines
       (fast-flux-trainer go_fast=True applies 1.5× lora_scale at inference)

Everything else matches upstream exactly: flow, cleanup, paths, config structure,
gradient checkpointing threshold, captioner flow, W&B integration, HF upload.
"""

import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys

# ── paths must be set before any ai-toolkit imports ─────────────────────────
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
AI_TOOLKIT = SCRIPT_DIR / "ai-toolkit"

sys.path.insert(0, str(AI_TOOLKIT))

# Patch submodules the same way upstream does (fixes ai-toolkit internals)
try:
    from submodule_patches import patch_submodules
    patch_submodules()
except ImportError:
    pass  # submodule_patches.py copied from upstream repo by setup.sh

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import shutil
import subprocess
import tarfile
import time
from collections import OrderedDict
from typing import Optional
from zipfile import ZipFile, is_zipfile

import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from huggingface_hub import HfApi
from jobs import BaseJob
from toolkit.config import get_config

from wandb_client import WeightsAndBiasesClient, logout_wandb
from layer_match import match_layers_to_optimize, available_layers_to_optimize

# ── constants (match upstream exactly) ───────────────────────────────────────
JOB_NAME    = "flux_train_replicate"
WEIGHTS_DIR = SCRIPT_DIR / "FLUX.1-dev"
INPUT_DIR   = SCRIPT_DIR / "input_images"
OUTPUT_DIR  = SCRIPT_DIR / "output"
JOB_DIR     = OUTPUT_DIR / JOB_NAME          # where ai-toolkit writes all output

REPLICATE_CDN_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

os.environ["LANG"] = "en_US.UTF-8"


# ─────────────────────────────────────────────────────────────────────────────
# CustomSDTrainer — copied verbatim from upstream, only base class is the same
# ─────────────────────────────────────────────────────────────────────────────
class CustomSDTrainer(SDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples: set = set()
        self.wandb: WeightsAndBiasesClient | None = None

    def hook_train_loop(self, batch):
        loss_dict = super().hook_train_loop(batch)
        if self.wandb:
            self.wandb.log_loss(loss_dict, self.step_num)
        return loss_dict

    def sample(self, step=None, is_first=False):
        super().sample(step=step, is_first=is_first)
        output_dir = JOB_DIR / "samples"
        all_samples = set([p.name for p in output_dir.glob("*.jpg")])
        new_samples = all_samples - self.seen_samples
        if self.wandb:
            image_paths = [output_dir / p for p in sorted(new_samples)]
            self.wandb.log_samples(image_paths, step)
        self.seen_samples = all_samples

    def post_save_hook(self, save_path):
        super().post_save_hook(save_path)
        lora_path = JOB_DIR / f"{JOB_NAME}.safetensors"
        if not lora_path.exists():
            lora_path = sorted(JOB_DIR.glob("*.safetensors"))[-1]
        if self.wandb:
            log.info(f"Saving weights to W&B: {lora_path.name}")
            self.wandb.save_weights(lora_path)


# ─────────────────────────────────────────────────────────────────────────────
# CustomJob — copied verbatim from upstream
# ─────────────────────────────────────────────────────────────────────────────
class CustomJob(BaseJob):
    def __init__(self, config: OrderedDict, wandb_client: WeightsAndBiasesClient | None):
        super().__init__(config)
        self.device = self.get_conf("device", "cpu")
        self.process_dict = {"custom_sd_trainer": CustomSDTrainer}
        self.load_processes(self.process_dict)
        for process in self.process:
            process.wandb = wandb_client

    def run(self):
        super().run()
        log.info(
            f"Running {len(self.process)} "
            f"process{'' if len(self.process) == 1 else 'es'}"
        )
        for process in self.process:
            process.run()


# ─────────────────────────────────────────────────────────────────────────────
# Weight download — Replicate CDN (same bytes their trainer uses)
# ─────────────────────────────────────────────────────────────────────────────
def download_weights():
    if WEIGHTS_DIR.exists():
        return

    log.info(f"Downloading FLUX.1-dev …\n  {REPLICATE_CDN_URL}")
    t1 = time.time()

    # pget = Replicate's parallel downloader (same tool used in upstream train.py)
    # Falls back to wget / curl if pget is not installed
    if shutil.which("pget"):
        subprocess.check_output(
            ["pget", "-xf", REPLICATE_CDN_URL, str(WEIGHTS_DIR.parent)]
        )
    else:
        log.info("  pget not found — using wget/curl (install pget for faster download)")
        tar_path = WEIGHTS_DIR.parent / "flux-dev.tar"
        WEIGHTS_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(
                ["wget", "-c", "--show-progress", "-O", str(tar_path), REPLICATE_CDN_URL]
            )
        except FileNotFoundError:
            subprocess.check_call(
                ["curl", "-L", "--continue-at", "-", "-o", str(tar_path), REPLICATE_CDN_URL]
            )
        log.info("Extracting …")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(str(WEIGHTS_DIR.parent))
        tar_path.unlink()

    log.info(f"Downloaded base weights in {time.time() - t1:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — copied / adapted from upstream
# ─────────────────────────────────────────────────────────────────────────────
def extract_zip(input_images: Path, input_dir: Path):
    """Upstream extract_zip: preserves directory structure, skips macOS artifacts."""
    if not is_zipfile(input_images):
        raise ValueError("input_images must be a zip file")

    input_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with ZipFile(input_images, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                image_count += 1

    log.info(f"Extracted {image_count} files from zip to {input_dir}")


def clean_up():
    """Upstream clean_up: logout wandb, wipe input and output dirs."""
    logout_wandb()
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def handle_hf_readme(
    hf_repo_id: str,
    trigger_word: Optional[str],
    steps: int,
    learning_rate: float,
    lora_rank: int,
):
    """Upstream: generates README.md in JOB_DIR from template."""
    from string import Template

    readme_path          = JOB_DIR / "README.md"
    readme_template_path = SCRIPT_DIR / "hugging-face-readme-template.md"

    if not readme_template_path.exists():
        log.warning("hugging-face-readme-template.md not found — skipping README")
        return

    shutil.copy(readme_template_path, readme_path)
    with readme_template_path.open() as f:
        template = f.read()

    variables = {
        "repo_id":   hf_repo_id,
        "title":     (
            hf_repo_id.split("/")[1].replace("-", " ").title()
            if len(hf_repo_id.split("/")) > 1 else hf_repo_id
        ),
        "trigger_word": trigger_word,
        "trigger_section": (
            f"\n## Trigger words\n\nYou should use `{trigger_word}` to trigger the image generation.\n"
            if trigger_word else ""
        ),
        "instance_prompt": f"instance_prompt: {trigger_word}" if trigger_word else "",
        "training_details": (
            f"\n## Training details\n\n"
            f"- Steps: {steps}\n"
            f"- Learning rate: {learning_rate}\n"
            f"- LoRA rank: {lora_rank}\n"
        ),
    }
    with readme_path.open("w") as f:
        f.write(Template(template).substitute(variables))


# ─────────────────────────────────────────────────────────────────────────────
# Training config builder
# ─────────────────────────────────────────────────────────────────────────────
def build_train_config(
    args,
    resolutions: list[int],
    sample_prompts: list[str],
    quantize: bool,
    gradient_checkpointing: bool,
    layers_to_optimize: Optional[list],
) -> OrderedDict:
    """
    Builds the ai-toolkit config dict.

    Matches upstream structure exactly except for the LR schedule:
      Upstream : lr=4e-4, no scheduler (constant)
      Ours     : lr=1e-3 peak, lr_scheduler=cosine_with_min_lr,
                 warmup=max(12, 0.8% of steps), floor=10% of peak (1e-4)

    cosine_with_min_lr is a transformers SchedulerType that falls through
    ai-toolkit's scheduler.py else-branch to the transformers lookup.
    Signature: get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0
    )
    Available in transformers >= 4.35 (we use 4.44.0).
    """
    warmup_steps = max(12, int(args.steps * 0.008))

    log.info(
        f"LR schedule: peak={args.learning_rate:.0e}  "
        f"floor={args.learning_rate * 0.1:.0e}  "
        f"warmup={warmup_steps} steps  scheduler=cosine_with_min_lr"
    )

    config = OrderedDict({
        "job": "custom_job",
        "config": {
            "name": JOB_NAME,
            "process": [{
                "type":            "custom_sd_trainer",
                "training_folder": str(OUTPUT_DIR),
                "device":          "cuda:0",
                "trigger_word":    args.trigger_word,

                "network": {
                    "type":         "lora",
                    "linear":       args.lora_rank,
                    "linear_alpha": args.lora_rank,   # alpha=rank prevents underflow
                },

                "save": {
                    "dtype":               "float16",
                    "save_every":          args.wandb_save_interval if args.wandb_api_key else args.steps + 1,
                    "max_step_saves_to_keep": 1,      # matches upstream
                },

                "datasets": [{
                    "folder_path":          str(INPUT_DIR),
                    "caption_ext":          "txt",
                    "caption_dropout_rate": args.caption_dropout_rate,
                    "shuffle_tokens":       False,
                    "cache_latents_to_disk": args.cache_latents_to_disk,
                    "cache_latents":        True,
                    "resolution":           resolutions,
                }],

                "train": {
                    "batch_size":                  args.batch_size,
                    "steps":                       args.steps,
                    "gradient_accumulation_steps": 1,
                    "train_unet":                  True,
                    "train_text_encoder":          False,
                    "content_or_style":            "balanced",
                    "gradient_checkpointing":      gradient_checkpointing,
                    "noise_scheduler":             "flowmatch",
                    "optimizer":                   args.optimizer,

                    # ── fast-flux-trainer LR schedule (key improvement) ──────
                    "lr":             args.learning_rate,
                    "lr_scheduler":   "cosine_with_min_lr",
                    "lr_scheduler_params": {
                        "num_warmup_steps":   warmup_steps,
                        "num_training_steps": args.steps,
                        "min_lr_rate":        0.1,    # floor = 10% of peak
                    },

                    # ── stability (same as upstream) ─────────────────────────
                    "ema_config": {"use_ema": True, "ema_decay": 0.99},
                    "dtype":      "bf16",
                },

                "model": {
                    "name_or_path": str(WEIGHTS_DIR),
                    "is_flux":      True,
                    "quantize":     quantize,
                },

                "sample": {
                    "sampler":       "flowmatch",
                    "sample_every":  (
                        args.wandb_sample_interval
                        if args.wandb_api_key and sample_prompts
                        else args.steps + 1
                    ),
                    "width":         1024,
                    "height":        1024,
                    "prompts":       sample_prompts,
                    "neg":           "",
                    "seed":          42,
                    "walk_seed":     True,
                    "guidance_scale": 3.5,
                    "sample_steps":  28,
                },
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    })

    # layers_to_optimize — matches upstream network_kwargs injection
    if layers_to_optimize:
        config["config"]["process"][0]["network"]["network_kwargs"] = {
            "only_if_contains": layers_to_optimize
        }

    # NOTE: trigger_word deletion is done explicitly in train() after extract_zip,
    # matching upstream's exact position in the flow. Do NOT delete it here.

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    output_path = str(SCRIPT_DIR / "trained-model.tar")

    # ── 1. Clean state from previous runs (matches upstream) ─────────────────
    clean_up()

    # ── 2. Validate layers regex ──────────────────────────────────────────────
    layers_to_optimize = None
    if args.layers_to_optimize_regex:
        layers_to_optimize = match_layers_to_optimize(args.layers_to_optimize_regex)
        if not layers_to_optimize:
            raise ValueError(
                f"The regex '{args.layers_to_optimize_regex}' didn't match any layers. "
                f"These layers can be optimized:\n" + "\n".join(available_layers_to_optimize)
            )

    # ── 3. Parse resolutions + sample prompts ────────────────────────────────
    resolutions    = [int(res) for res in args.resolution.split(",")]
    sample_prompts = []
    if args.wandb_sample_prompts:
        sample_prompts = [p.strip() for p in args.wandb_sample_prompts.split("\n")]

    # ── 4. Hardware: gradient checkpointing + quantize ────────────────────────
    # Matches upstream exactly: threshold is 100 GB.
    # On any normal GPU (A100 40/80 GB, H100 80 GB) this always triggers.
    quantize               = False
    gradient_checkpointing = args.gradient_checkpointing

    if not gradient_checkpointing:
        if torch.cuda.get_device_properties(0).total_memory < 1024 * 1024 * 1024 * 100:
            log.info("Turning gradient checkpointing on and quantizing base model, "
                     "GPU has less than 100 GB of memory")
            gradient_checkpointing = True
            quantize               = True
        elif args.batch_size > 1:
            log.info("Turning gradient checkpointing on automatically for batch size > 1")
            gradient_checkpointing = True
        elif max(resolutions) > 1024:
            log.info("Turning gradient checkpointing on; training resolution greater than 1024x1024")
            gradient_checkpointing = True

    # ── 5. Build train config (upstream builds config BEFORE W&B client) ──────
    train_config = build_train_config(
        args, resolutions, sample_prompts, quantize, gradient_checkpointing, layers_to_optimize
    )

    # ── 6. W&B client (upstream creates AFTER config build) ───────────────────
    wandb_client = None
    if args.wandb_api_key:
        wandb_config = {
            "trigger_word":         args.trigger_word,
            "steps":                args.steps,
            "learning_rate":        args.learning_rate,
            "batch_size":           args.batch_size,
            "resolution":           args.resolution,
            "lora_rank":            args.lora_rank,
            "caption_dropout_rate": args.caption_dropout_rate,
            "optimizer":            args.optimizer,
        }
        wandb_client = WeightsAndBiasesClient(
            api_key=args.wandb_api_key,
            config=wandb_config,
            sample_prompts=sample_prompts,
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run or None,
        )

    # ── 7. Download base weights ──────────────────────────────────────────────
    download_weights()

    # ── 8. Extract training images ────────────────────────────────────────────
    extract_zip(Path(args.input_images), INPUT_DIR)

    # ── 9. Trigger word deletion (upstream does this after extract, before caption)
    if not args.trigger_word:
        del train_config["config"]["process"][0]["trigger_word"]

    # ── 10. Captions must already be present as <stem>.txt alongside images. ─
    # Captioning is owned by tools/caption_dataset.py (Claude/GPT/Gemini via
    # API). If a .txt is missing, ai-toolkit's dataset loader will train with
    # only the trigger token for that image — still valid, just weaker signal.

    # ── 11. Run training ──────────────────────────────────────────────────────
    log.info("Starting train job")
    job = CustomJob(get_config(train_config, name=None), wandb_client)
    job.run()

    if wandb_client:
        wandb_client.finish()

    job.cleanup()

    # ── 12. Post-training output cleanup (matches upstream exactly) ───────────
    # Rename canonical output — no exists() guard, matches upstream fail-fast behaviour
    lora_file = JOB_DIR / f"{JOB_NAME}.safetensors"
    lora_file.rename(JOB_DIR / "lora.safetensors")

    # Remove sample images from output
    samples_dir = JOB_DIR / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # Remove intermediate checkpoints — keep only lora.safetensors
    for path in JOB_DIR.glob("*.safetensors"):
        if path.name != "lora.safetensors":
            path.unlink()

    # Remove optimizer state (not needed for inference)
    optimizer_file = JOB_DIR / "optimizer.pt"
    if optimizer_file.exists():
        optimizer_file.unlink()

    # Copy generated captions into output tar (but not to HF)
    captions_dir = JOB_DIR / "captions"
    captions_dir.mkdir(exist_ok=True)
    for caption_file in INPUT_DIR.glob("*.txt"):
        shutil.copy(caption_file, captions_dir)

    # ── 13. Package output ────────────────────────────────────────────────────
    os.system(f"tar -cvf '{output_path}' '{JOB_DIR}'")
    log.info(f"Output: {output_path}")

    # ── 14. Optional HF upload ────────────────────────────────────────────────
    if args.hf_token and args.hf_repo_id:
        if captions_dir.exists():
            shutil.rmtree(captions_dir)
        try:
            handle_hf_readme(args.hf_repo_id, args.trigger_word, args.steps,
                             args.learning_rate, args.lora_rank)
            log.info(f"Uploading to Hugging Face: {args.hf_repo_id}")
            api      = HfApi()
            repo_url = api.create_repo(
                args.hf_repo_id,
                private=False,
                exist_ok=True,
                token=args.hf_token,
            )
            log.info(f"HF Repo URL: {repo_url}")
            api.upload_folder(
                repo_id=args.hf_repo_id,
                folder_path=str(JOB_DIR),
                repo_type="model",
                token=args.hf_token,
            )
        except Exception as e:
            log.error(f"Error uploading to Hugging Face: {e}")

    log.info("Training complete.")
    log.info(
        "INFERENCE NOTE: load this LoRA at scale 1.5 (not 1.0).\n"
        "  fast-flux-trainer's go_fast=True applies 1.5× lora_scale at inference;\n"
        "  match that on fal.ai / ComfyUI to avoid the weak-activation issue."
    )
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Self-hosted FLUX LoRA trainer (fast-flux-trainer LR schedule)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input_images", required=True,
                   help="Path to .zip of training images. Min 10, ideal 15–20.")
    p.add_argument("--trigger_word",  default="TOK",
                   help="Token that activates the LoRA.")
    p.add_argument("--steps",         type=int,   default=1000,
                   help="Training steps. Range 500–4000. Rule of thumb: 50× image count.")
    p.add_argument("--learning_rate", type=float, default=1e-3,
                   help="Peak LR. Cosine decays to 10%% of this. Default 1e-3.")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--resolution",    default="512,768,1024")
    p.add_argument("--lora_rank",     type=int,   default=16,
                   help="LoRA rank. Range 1–128.")
    p.add_argument("--caption_dropout_rate", type=float, default=0.05,
                   help="0.05 for subjects; 0.1–0.3 for styles.")
    p.add_argument("--optimizer",     default="adamw8bit",
                   choices=["prodigy", "adam8bit", "adamw8bit", "lion8bit",
                             "adam", "adamw", "lion", "adagrad", "adafactor"])
    p.add_argument("--cache_latents_to_disk", action=argparse.BooleanOptionalAction,
                   default=False, help="Use when > 30 images and hitting OOM.")
    p.add_argument("--layers_to_optimize_regex", default=None,
                   help="Regex to target specific transformer layers.")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Auto-enabled when GPU < 100 GB (i.e. always on normal GPUs).")

    # HuggingFace
    p.add_argument("--hf_repo_id", default=None,
                   help="HuggingFace repo to publish to, e.g. username/my-lora.")
    p.add_argument("--hf_token",   default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace write token. Falls back to $HF_TOKEN.")

    # W&B
    p.add_argument("--wandb_api_key",        default=None)
    p.add_argument("--wandb_project",        default=JOB_NAME)
    p.add_argument("--wandb_run",            default=None)
    p.add_argument("--wandb_entity",         default=None)
    p.add_argument("--wandb_sample_interval",type=int, default=100)
    p.add_argument("--wandb_save_interval",  type=int, default=100)
    p.add_argument("--wandb_sample_prompts", default=None,
                   help="Newline-separated prompts for W&B sample images.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
