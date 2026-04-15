#!/usr/bin/env python3
"""
FLUX DoRA Trainer — Production Self-Hosted
Based on: https://github.com/replicate/flux-fine-tuner (ai-toolkit / ostris)

DoRA (Weight-Decomposed Low-Rank Adaptation) vs plain LoRA:
  - Decomposes weight updates into magnitude + direction components separately
  - Magnitude: how much to change each weight (new in DoRA)
  - Direction: which way to change it (what standard LoRA captures)
  - Result: ~15-20% better face/identity resemblance at same rank
  - Same output file size as LoRA (~50-200 MB)
  - ~5-10% slower training

Defaults tuned for face/identity capture:
  - Rank 64  (vs LoRA default 16 — more capacity for identity details)
  - LR 1e-4  (vs LoRA default 1e-3 — higher rank needs lower LR)
  - Same cosine_with_min_lr warmup schedule

DoRA enabled in ai-toolkit via network.type = "dora" (activates DoRAModule)
"""

import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
AI_TOOLKIT = SCRIPT_DIR / "ai-toolkit"
LLAVA_DIR  = SCRIPT_DIR / "LLaVA"

sys.path.insert(0, str(AI_TOOLKIT))
if LLAVA_DIR.exists():
    sys.path.insert(0, str(LLAVA_DIR))

try:
    from submodule_patches import patch_submodules
    patch_submodules()
except ImportError:
    pass

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

from caption import Captioner
from wandb_client import WeightsAndBiasesClient, logout_wandb
from layer_match import match_layers_to_optimize, available_layers_to_optimize

JOB_NAME    = "flux_train_replicate"
WEIGHTS_DIR = SCRIPT_DIR / "FLUX.1-dev"
INPUT_DIR   = SCRIPT_DIR / "input_images"
OUTPUT_DIR  = SCRIPT_DIR / "output"
JOB_DIR     = OUTPUT_DIR / JOB_NAME

REPLICATE_CDN_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
os.environ["LANG"] = "en_US.UTF-8"


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
            self.wandb.save_weights(lora_path)


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
        log.info(f"Running {len(self.process)} process{'es' if len(self.process) != 1 else ''}")
        for process in self.process:
            process.run()


def download_weights():
    if WEIGHTS_DIR.exists():
        return
    log.info(f"Downloading FLUX.1-dev …\n  {REPLICATE_CDN_URL}")
    t1 = time.time()
    if shutil.which("pget"):
        subprocess.check_output(["pget", "-xf", REPLICATE_CDN_URL, str(WEIGHTS_DIR.parent)])
    else:
        tar_path = WEIGHTS_DIR.parent / "flux-dev.tar"
        WEIGHTS_DIR.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(["wget", "-c", "--show-progress", "-O", str(tar_path), REPLICATE_CDN_URL])
        except FileNotFoundError:
            subprocess.check_call(["curl", "-L", "--continue-at", "-", "-o", str(tar_path), REPLICATE_CDN_URL])
        log.info("Extracting …")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(str(WEIGHTS_DIR.parent))
        tar_path.unlink()
    log.info(f"Downloaded in {time.time() - t1:.1f}s")


def extract_zip(input_images: Path, input_dir: Path):
    if not is_zipfile(input_images):
        raise ValueError("input_images must be a zip file")
    input_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(input_images, "r") as zip_ref:
        for fi in zip_ref.infolist():
            if not fi.filename.startswith("__MACOSX/") and not fi.filename.startswith("._"):
                zip_ref.extract(fi, input_dir)
                count += 1
    log.info(f"Extracted {count} files to {input_dir}")


def clean_up():
    logout_wandb()
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def handle_hf_readme(hf_repo_id, trigger_word, steps, learning_rate, lora_rank):
    from string import Template
    readme_path          = JOB_DIR / "README.md"
    readme_template_path = SCRIPT_DIR / "hugging-face-readme-template.md"
    if not readme_template_path.exists():
        return
    with readme_template_path.open() as f:
        template = f.read()
    variables = {
        "repo_id":   hf_repo_id,
        "title": (hf_repo_id.split("/")[1].replace("-", " ").title()
                  if "/" in hf_repo_id else hf_repo_id),
        "trigger_word": trigger_word,
        "trigger_section": (
            f"\n## Trigger words\n\nUse `{trigger_word}` to trigger the image generation.\n"
            if trigger_word else ""
        ),
        "instance_prompt": f"instance_prompt: {trigger_word}" if trigger_word else "",
        "training_details": (
            f"\n## Training details\n\n"
            f"- Method: DoRA (rank={lora_rank})\n"
            f"- Steps: {steps}\n"
            f"- Learning rate: {learning_rate}\n"
        ),
    }
    with readme_path.open("w") as f:
        f.write(Template(template).substitute(variables))


def build_train_config(args, resolutions, sample_prompts, quantize,
                       gradient_checkpointing, layers_to_optimize) -> OrderedDict:
    """
    Builds ai-toolkit config with DoRA enabled.

    Key difference from LoRA config:
      network.type = "dora" swaps in ai-toolkit's DoRAModule (toolkit/models/DoRA.py)
      instead of the standard LoRA module. This is the correct toggle — NOT a kwarg.

    LR is lower (1e-4 vs 1e-3) because rank-64 has more parameters and
    DoRA's magnitude decomposition needs more careful tuning.
    """
    warmup_steps = max(12, int(args.steps * 0.008))

    log.info(
        f"DoRA config: type=dora  rank={args.lora_rank}  "
        f"lr={args.learning_rate:.0e}  warmup={warmup_steps}"
    )

    # Build network config — type "dora" activates DoRAModule in ai-toolkit
    network_config: dict = {
        "type":         "dora",          # activates toolkit/models/DoRA.py::DoRAModule
        "linear":       args.lora_rank,
        "linear_alpha": args.lora_rank,
    }
    if layers_to_optimize:
        network_config["network_kwargs"] = {"only_if_contains": layers_to_optimize}

    config = OrderedDict({
        "job": "custom_job",
        "config": {
            "name": JOB_NAME,
            "process": [{
                "type":            "custom_sd_trainer",
                "training_folder": str(OUTPUT_DIR),
                "device":          "cuda:0",
                "trigger_word":    args.trigger_word,

                "network": network_config,

                "save": {
                    "dtype":                  "float16",
                    "save_every":             args.wandb_save_interval if args.wandb_api_key else args.steps + 1,
                    "max_step_saves_to_keep": 1,
                },

                "datasets": [{
                    "folder_path":           str(INPUT_DIR),
                    "caption_ext":           "txt",
                    "caption_dropout_rate":  args.caption_dropout_rate,
                    "shuffle_tokens":        False,
                    "cache_latents_to_disk": args.cache_latents_to_disk,
                    "cache_latents":         True,
                    "resolution":            resolutions,
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

                    "lr":             args.learning_rate,
                    "lr_scheduler":   "cosine_with_min_lr",
                    "lr_scheduler_params": {
                        "num_warmup_steps":   warmup_steps,
                        "num_training_steps": args.steps,
                        "min_lr_rate":        0.1,
                    },

                    "ema_config": {"use_ema": True, "ema_decay": 0.99},
                    "dtype":      "bf16",
                },

                "model": {
                    "name_or_path": str(WEIGHTS_DIR),
                    "is_flux":      True,
                    "quantize":     quantize,
                },

                "sample": {
                    "sampler":        "flowmatch",
                    "sample_every":   (args.wandb_sample_interval
                                      if args.wandb_api_key and sample_prompts
                                      else args.steps + 1),
                    "width":          1024,
                    "height":         1024,
                    "prompts":        sample_prompts,
                    "neg":            "",
                    "seed":           42,
                    "walk_seed":      True,
                    "guidance_scale": 3.5,
                    "sample_steps":   28,
                },
            }],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    })

    return config


def train(args):
    output_path = str(SCRIPT_DIR / "trained-model.tar")
    clean_up()

    layers_to_optimize = None
    if args.layers_to_optimize_regex:
        layers_to_optimize = match_layers_to_optimize(args.layers_to_optimize_regex)
        if not layers_to_optimize:
            raise ValueError(
                f"Regex '{args.layers_to_optimize_regex}' matched no layers.\n"
                + "\n".join(available_layers_to_optimize)
            )

    resolutions    = [int(r) for r in args.resolution.split(",")]
    sample_prompts = []
    if args.wandb_sample_prompts:
        sample_prompts = [p.strip() for p in args.wandb_sample_prompts.split("\n")]

    quantize               = False
    gradient_checkpointing = args.gradient_checkpointing
    if not gradient_checkpointing:
        if torch.cuda.get_device_properties(0).total_memory < 1024 ** 3 * 100:
            log.info("GPU < 100 GB — enabling gradient checkpointing + quantization")
            gradient_checkpointing = True
            quantize               = True
        elif args.batch_size > 1:
            gradient_checkpointing = True
        elif max(resolutions) > 1024:
            gradient_checkpointing = True

    train_config = build_train_config(
        args, resolutions, sample_prompts, quantize, gradient_checkpointing, layers_to_optimize
    )

    wandb_client = None
    if args.wandb_api_key:
        wandb_client = WeightsAndBiasesClient(
            api_key=args.wandb_api_key,
            config={"method": "DoRA", "rank": args.lora_rank, "steps": args.steps,
                    "lr": args.learning_rate, "trigger_word": args.trigger_word},
            sample_prompts=sample_prompts,
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run or None,
        )

    download_weights()
    extract_zip(Path(args.input_images), INPUT_DIR)

    if not args.trigger_word:
        del train_config["config"]["process"][0]["trigger_word"]

    captioner = Captioner()
    if args.autocaption and not captioner.all_images_are_captioned(INPUT_DIR):
        captioner.load_models()
        captioner.caption_images(INPUT_DIR, args.autocaption_prefix, args.autocaption_suffix)
    del captioner
    torch.cuda.empty_cache()

    log.info("Starting DoRA training…")
    job = CustomJob(get_config(train_config, name=None), wandb_client)
    job.run()

    if wandb_client:
        wandb_client.finish()
    job.cleanup()

    lora_file = JOB_DIR / f"{JOB_NAME}.safetensors"
    lora_file.rename(JOB_DIR / "lora.safetensors")

    samples_dir = JOB_DIR / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    for path in JOB_DIR.glob("*.safetensors"):
        if path.name != "lora.safetensors":
            path.unlink()
    optimizer_file = JOB_DIR / "optimizer.pt"
    if optimizer_file.exists():
        optimizer_file.unlink()

    captions_dir = JOB_DIR / "captions"
    captions_dir.mkdir(exist_ok=True)
    for cf in INPUT_DIR.glob("*.txt"):
        shutil.copy(cf, captions_dir)

    os.system(f"tar -cvf '{output_path}' '{JOB_DIR}'")
    log.info(f"Output: {output_path}")

    if args.hf_token and args.hf_repo_id:
        if captions_dir.exists():
            shutil.rmtree(captions_dir)
        try:
            handle_hf_readme(args.hf_repo_id, args.trigger_word, args.steps,
                             args.learning_rate, args.lora_rank)
            api = HfApi()
            repo_url = api.create_repo(args.hf_repo_id, private=False, exist_ok=True, token=args.hf_token)
            api.upload_folder(repo_id=args.hf_repo_id, folder_path=str(JOB_DIR),
                              repo_type="model", token=args.hf_token)
            log.info(f"Uploaded to HF: {repo_url}")
        except Exception as e:
            log.error(f"HF upload error: {e}")

    log.info("DoRA training complete.")
    log.info("INFERENCE: load this DoRA at scale 1.5 (not 1.0).")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Self-hosted FLUX DoRA trainer — rank-64, best face resemblance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_images", required=True)
    p.add_argument("--trigger_word",  default="TOK")
    p.add_argument("--steps",         type=int,   default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Peak LR. Lower than LoRA (1e-4 vs 1e-3) for high-rank DoRA.")
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--resolution",    default="512,768,1024")
    p.add_argument("--lora_rank",     type=int,   default=64,
                   help="DoRA rank. 64 is the sweet spot for face capture.")
    p.add_argument("--caption_dropout_rate", type=float, default=0.05)
    p.add_argument("--optimizer",     default="adamw8bit",
                   choices=["prodigy", "adam8bit", "adamw8bit", "lion8bit",
                             "adam", "adamw", "lion", "adagrad", "adafactor"])
    p.add_argument("--cache_latents_to_disk", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--layers_to_optimize_regex", default=None)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--autocaption",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autocaption_prefix", default=None)
    p.add_argument("--autocaption_suffix", default=None)
    p.add_argument("--hf_repo_id",    default=None)
    p.add_argument("--hf_token",      default=os.environ.get("HF_TOKEN"))
    p.add_argument("--wandb_api_key",         default=None)
    p.add_argument("--wandb_project",         default=JOB_NAME)
    p.add_argument("--wandb_run",             default=None)
    p.add_argument("--wandb_entity",          default=None)
    p.add_argument("--wandb_sample_interval", type=int, default=100)
    p.add_argument("--wandb_save_interval",   type=int, default=100)
    p.add_argument("--wandb_sample_prompts",  default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
