#!/usr/bin/env python3
"""
FLUX DreamBooth LoRA Trainer — Production Self-Hosted
Uses diffusers train_dreambooth_lora_flux.py with prior preservation loss.

What makes DreamBooth different from plain LoRA / DoRA:
  - Prior preservation loss: generates N class images ("a person") BEFORE training,
    then interleaves them during training to prevent catastrophic forgetting.
    This is the defining feature of DreamBooth.
  - Without it, the model gradually forgets what a generic "person" looks like —
    your trigger word bleeds into everything.
  - With it, identity is sharper AND the model stays general outside the trigger word.

Comparison:
  LoRA     — trains on your images only. Fast, good resemblance.
  DoRA     — same but weight-decomposed. Better resemblance, same speed.
  DreamBooth — trains on your images + generated class images. Best consistency,
               especially when generating in varied styles/poses.

Defaults:
  - lora_rank: 32  (balanced; DreamBooth's prior preservation compensates for lower rank)
  - learning_rate: 1e-4
  - num_class_images: 10  (generated via FLUX before training; ~3 min)
  - steps: 1000
"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile, is_zipfile

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["LANG"] = "en_US.UTF-8"

# LLaVA lives at /workspace/LLaVA (cloned from flux-fine-tuner in Dockerfile)
LLAVA_DIR = Path("/workspace/LLaVA")
if LLAVA_DIR.exists():
    sys.path.insert(0, str(LLAVA_DIR))

import torch

SCRIPT_DIR  = Path(__file__).parent.resolve()
WEIGHTS_DIR = SCRIPT_DIR / "FLUX.1-dev"
INPUT_DIR   = SCRIPT_DIR / "input_images"
CLASS_DIR   = SCRIPT_DIR / "class_images"
OUTPUT_DIR  = SCRIPT_DIR / "output" / "dreambooth"

REPLICATE_CDN_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
DREAMBOOTH_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0"
    "/examples/dreambooth/train_dreambooth_lora_flux.py"
)

# Single-GPU accelerate config (no interaction needed)
ACCELERATE_CONFIG_YAML = """\
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_weights():
    if WEIGHTS_DIR.exists():
        log.info("FLUX.1-dev weights already present.")
        return
    log.info(f"Downloading FLUX.1-dev …\n  {REPLICATE_CDN_URL}")
    t1 = time.time()
    if shutil.which("pget"):
        subprocess.check_call(["pget", "-xf", REPLICATE_CDN_URL, str(WEIGHTS_DIR.parent)])
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
    log.info(f"Downloaded FLUX.1-dev in {time.time() - t1:.1f}s")


def extract_zip(zip_path: Path, out_dir: Path):
    if not is_zipfile(zip_path):
        raise ValueError(f"Not a zip file: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if not info.filename.startswith("__MACOSX/") and not info.filename.startswith("._"):
                z.extract(info, out_dir)
                count += 1
    log.info(f"Extracted {count} files to {out_dir}")


def generate_class_images(class_prompt: str, num_images: int):
    """
    Generate class images using FLUX for prior preservation.
    These teach the model what a generic 'person' looks like so it doesn't
    forget after being trained on your specific subject.
    ~15-20s per image at 512px on RTX 4090.
    """
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(CLASS_DIR.glob("*.jpg")) + list(CLASS_DIR.glob("*.png"))
    if len(existing) >= num_images:
        log.info(f"Class images already exist ({len(existing)}), skipping generation.")
        return

    to_generate = num_images - len(existing)
    log.info(f"Generating {to_generate} class images: '{class_prompt}'")
    log.info("Loading FLUX pipeline (~30s)…")

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        str(WEIGHTS_DIR),
        torch_dtype=torch.bfloat16,
    )
    # CPU offload to stay within 24 GB VRAM while also having training weights loaded
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    start_idx = len(existing)
    for i in range(to_generate):
        img = pipe(
            class_prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            height=512,
            width=512,
            generator=torch.Generator("cpu").manual_seed(i + 1000),
        ).images[0]
        img.save(CLASS_DIR / f"class_{start_idx + i:04d}.jpg")
        log.info(f"  Class image {i + 1}/{to_generate}")

    del pipe
    torch.cuda.empty_cache()
    log.info(f"Class images ready: {CLASS_DIR}")


def download_dreambooth_script() -> Path:
    script_path = SCRIPT_DIR / "train_dreambooth_lora_flux.py"
    if script_path.exists():
        log.info(f"DreamBooth script already present: {script_path.name}")
        return script_path
    log.info(f"Downloading DreamBooth training script…\n  {DREAMBOOTH_SCRIPT_URL}")
    urllib.request.urlretrieve(DREAMBOOTH_SCRIPT_URL, script_path)
    log.info("Downloaded train_dreambooth_lora_flux.py")
    return script_path


def write_accelerate_config() -> Path:
    config_path = SCRIPT_DIR / "accelerate_config.yaml"
    config_path.write_text(ACCELERATE_CONFIG_YAML)
    return config_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    output_path = str(SCRIPT_DIR / "trained-model.tar")

    # 1. Clean previous run
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Download FLUX weights
    download_weights()

    # 3. Extract training images
    extract_zip(Path(args.input_images), INPUT_DIR)

    # 3b. LLaVA auto-captioning (same as LoRA/DoRA)
    # diffusers DreamBooth script reads per-image .txt captions automatically
    # when they exist alongside image files — identical behaviour to ai-toolkit.
    if args.autocaption:
        try:
            from caption import Captioner
            captioner = Captioner()
            if not captioner.all_images_are_captioned(INPUT_DIR):
                log.info("Running LLaVA captioner on instance images…")
                captioner.load_models()
                captioner.caption_images(
                    INPUT_DIR,
                    args.autocaption_prefix,
                    args.autocaption_suffix,
                )
                log.info("LLaVA captioning complete.")
            else:
                log.info("All images already have captions — skipping LLaVA.")
            del captioner
            torch.cuda.empty_cache()
        except ImportError:
            log.warning("caption.py not found — falling back to fixed instance_prompt.")

    # 4. Prior preservation: generate class images
    if args.num_class_images > 0:
        generate_class_images(args.class_prompt, args.num_class_images)
    else:
        log.info("Prior preservation disabled (--num_class_images 0).")

    # 5. Download training script + write accelerate config
    db_script      = download_dreambooth_script()
    accel_config   = write_accelerate_config()

    # 6. Install any missing script deps
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "peft>=0.7.0", "bitsandbytes>=0.43.1", "prodigyopt",
    ])

    # 7. Build training command
    warmup_steps = max(50, int(args.steps * 0.05))
    cmd = [
        "accelerate", "launch",
        "--config_file", str(accel_config),
        str(db_script),
        "--pretrained_model_name_or_path", str(WEIGHTS_DIR),
        "--instance_data_dir",             str(INPUT_DIR),
        "--output_dir",                    str(OUTPUT_DIR),
        "--instance_prompt",               f"a photo of {args.trigger_word} person",
        "--resolution",                    "1024",
        "--train_batch_size",              str(args.batch_size),
        "--gradient_accumulation_steps",   "4",
        "--learning_rate",                 str(args.learning_rate),
        "--lr_scheduler",                  "cosine",
        "--lr_warmup_steps",               str(warmup_steps),
        "--max_train_steps",               str(args.steps),
        "--rank",                          str(args.lora_rank),
        "--seed",                          "42",
        "--mixed_precision",               "bf16",
        "--gradient_checkpointing",
        "--use_8bit_adam",
    ]

    if args.num_class_images > 0:
        cmd += [
            "--with_prior_preservation",
            "--prior_loss_weight",  "1.0",
            "--class_data_dir",     str(CLASS_DIR),
            "--class_prompt",       args.class_prompt,
            "--num_class_images",   str(args.num_class_images),
        ]

    if args.train_text_encoder:
        cmd.append("--train_text_encoder")

    log.info("Starting DreamBooth training…")
    log.info(f"Instance prompt: 'a photo of {args.trigger_word} person'")
    log.info(f"Class prompt:    '{args.class_prompt}'")
    log.info(f"Class images:    {args.num_class_images}")
    log.info(f"Steps:           {args.steps}  |  rank: {args.lora_rank}  |  lr: {args.learning_rate}")

    subprocess.check_call(cmd)

    # 8. Rename output for consistency
    lora_files = sorted(OUTPUT_DIR.glob("**/*.safetensors"))
    if not lora_files:
        # diffusers saves adapter_model.safetensors in a subdirectory
        lora_files = sorted(OUTPUT_DIR.rglob("*.safetensors"))

    if lora_files:
        lora_dst = OUTPUT_DIR / "lora.safetensors"
        if lora_files[-1] != lora_dst:
            shutil.copy(lora_files[-1], lora_dst)
            log.info(f"Renamed output: {lora_files[-1].name} → lora.safetensors")

    # 9. Package
    os.system(f"tar -cvf '{output_path}' '{OUTPUT_DIR}'")
    log.info(f"Output: {output_path}")
    log.info("DreamBooth training complete.")
    log.info("INFERENCE: load LoRA at scale 1.0–1.5. Start with 1.0 for DreamBooth.")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FLUX DreamBooth LoRA trainer with prior preservation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_images",   required=True,
                   help="Path to .zip of training images.")
    p.add_argument("--trigger_word",   default="TOK",
                   help="Token that activates the subject. Used as: 'a photo of TOK person'.")
    p.add_argument("--steps",          type=int,   default=1000)
    p.add_argument("--learning_rate",  type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=1)
    p.add_argument("--lora_rank",      type=int,   default=32,
                   help="LoRA rank. 32 is good for DreamBooth (prior preservation compensates).")
    p.add_argument("--class_prompt",   default="a photo of a person",
                   help="Prompt used to generate prior preservation class images.")
    p.add_argument("--num_class_images", type=int, default=10,
                   help="Class images to generate. 0 = disable prior preservation.")
    p.add_argument("--train_text_encoder", action="store_true", default=False,
                   help="Also fine-tune the text encoder (more VRAM, potentially better).")
    # Captioning (same as LoRA/DoRA — LLaVA per-image captions)
    p.add_argument("--autocaption",        action=argparse.BooleanOptionalAction, default=True,
                   help="Run LLaVA per-image captioning before training.")
    p.add_argument("--autocaption_prefix", default=None,
                   help="Prefix added to every caption, e.g. 'a photo of TOK, '.")
    p.add_argument("--autocaption_suffix", default=None,
                   help="Suffix added to every caption.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
