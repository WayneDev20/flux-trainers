# ============================================================
# FLUX DreamBooth LoRA Trainer — Pre-baked Production Image
#
# Simpler than the LoRA/DoRA image — no ai-toolkit, no LLaVA.
# Uses diffusers train_dreambooth_lora_flux.py directly.
#
# Baked in at build time:
#   • PyTorch 2.4.1 + CUDA 12.4
#   • diffusers, transformers, accelerate, peft, bitsandbytes
#   • train_dreambooth_lora_flux.py from diffusers v0.31.0
#
# Downloaded at runtime (~5 min):
#   • FLUX.1-dev weights via pget → /workspace/FLUX.1-dev/
#
# Build:
#   ./build_push.sh <dockerhub_user>
#
# Train:
#   ./vastai_train.sh --images ./my_images.zip \
#       --docker_image <user>/flux-dreambooth-trainer:latest
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git curl wget unzip \
        libgl1 libglib2.0-0 \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip setuptools wheel

# pget — parallel downloader for FLUX weights
RUN curl -o /usr/local/bin/pget -fsSL \
        "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" \
    && chmod +x /usr/local/bin/pget

WORKDIR /workspace

# ---------------------------------------------------------------------------
# 1. PyTorch (CUDA 12.4)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 2. Core diffusion stack (pinned to match LoRA/DoRA trainers exactly)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    safetensors==0.4.4 \
    diffusers==0.31.0 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    peft==0.13.0 \
    huggingface-hub==0.24.5 \
    hf_transfer==0.1.8 \
    sentencepiece==0.2.0 \
    "tokenizers==0.19.0"

# ---------------------------------------------------------------------------
# 3. Training utilities
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    bitsandbytes==0.43.3 \
    prodigyopt==1.0 \
    lion-pytorch==0.2.2 \
    wandb==0.17.8 \
    pillow==10.4.0 \
    numpy==1.26.0 \
    opencv-python-headless==4.10.0.84 \
    python-dotenv \
    omegaconf==2.3.0 \
    einops==0.8.0

# ---------------------------------------------------------------------------
# 4. LLaVA captioner (same as LoRA/DoRA — for per-image caption generation)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    deepspeed==0.9.5 \
    einops-exts==0.0.4 \
    shortuuid==1.0.11 \
    markdown2==2.4.10

RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
 && pip install --no-cache-dir "git+https://github.com/haotian-liu/LLaVA.git"

# Re-pin pydantic v1 + torch 2.4.1 (LLaVA upgrades both)
RUN pip install --no-cache-dir "pydantic==1.10.17"
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 5. Clone flux-fine-tuner for caption.py + supporting files
#    (we only need caption.py, wandb_client.py, layer_match.py — not ai-toolkit)
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/replicate/flux-fine-tuner /tmp/flux-fine-tuner \
 && for f in caption.py wandb_client.py layer_match.py lora_loading_patch.py; do \
        cp /tmp/flux-fine-tuner/$f /workspace/$f 2>/dev/null || true; \
    done \
 && cp -r /tmp/flux-fine-tuner/LLaVA /workspace/LLaVA 2>/dev/null || true \
 && rm -rf /tmp/flux-fine-tuner

# ---------------------------------------------------------------------------
# 6. DreamBooth training script from diffusers v0.31.0
# ---------------------------------------------------------------------------
RUN curl -fsSL \
    "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/dreambooth/train_dreambooth_lora_flux.py" \
    -o /workspace/train_dreambooth_lora_flux.py

# ---------------------------------------------------------------------------
# 7. Copy our train.py wrapper
# ---------------------------------------------------------------------------
COPY train.py /workspace/

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
ENTRYPOINT ["python", "/workspace/train.py"]
CMD ["--help"]
