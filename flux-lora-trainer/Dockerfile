# ============================================================
# FLUX LoRA Trainer — Pre-baked Production Image
#
# Baked in at build time:
#   • All Python packages (correct install order)
#   • ai-toolkit + LLaVA + all supporting files from flux-fine-tuner
#   • scheduler.py patched for cosine_with_min_lr
#
# Downloaded at runtime (~5 min, done once per instance):
#   • FLUX.1-dev weights via pget → /workspace/FLUX.1-dev/
#
# Uploaded per run by vastai_train.sh:
#   • Training images zip
#
# Build:
#   ./build_push.sh <dockerhub_user>
#
# Train:
#   ./vastai_train.sh --images ./my_images.zip \
#       --docker_image <user>/flux-lora-trainer:latest
# ============================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git git-lfs curl wget unzip tar \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip setuptools wheel

# pget — Replicate's parallel downloader (same version as upstream cog.yaml)
RUN curl -o /usr/local/bin/pget -fsSL \
        "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" \
    && chmod +x /usr/local/bin/pget

WORKDIR /workspace

# ---------------------------------------------------------------------------
# 1. PyTorch (CUDA 12.4) — install first so LLaVA can't downgrade it
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 2. Core diffusion stack (pinned to match Replicate exactly)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    safetensors==0.4.4 \
    diffusers==0.30.0 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    peft==0.13.0 \
    huggingface-hub==0.24.5 \
    hf_transfer==0.1.8 \
    sentencepiece==0.2.0 \
    "tokenizers==0.19.0"

# Diffusers fork with FLUX inpaint support
RUN pip install --no-cache-dir \
    "git+https://github.com/Gothos/diffusers.git@flux-inpaint"

# ---------------------------------------------------------------------------
# 3. LoRA / quantization / optimizers
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    lycoris-lora==1.8.3 \
    bitsandbytes==0.43.3 \
    optimum-quanto==0.2.4 \
    prodigyopt==1.0 \
    lion-pytorch==0.2.2

# ---------------------------------------------------------------------------
# 4. Vision / image stack
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    pillow==10.4.0 \
    numpy==1.26.0 \
    opencv-python-headless==4.10.0.84 \
    scikit-image==0.24.0 \
    scikit-learn>=1.3.2 \
    kornia==0.7.3 \
    albumentations==1.4.3 \
    lpips==0.1.4 \
    open_clip_torch==2.26.1 \
    timm==1.0.8

# ---------------------------------------------------------------------------
# 5. ai-toolkit extras + misc
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    einops==0.8.0 \
    omegaconf==2.3.0 \
    pyyaml==6.0.1 \
    oyaml==1.0 \
    toml==0.10.2 \
    flatten_json==0.1.14 \
    tensorboard==2.17.0 \
    controlnet_aux==0.0.7 \
    k-diffusion \
    invisible-watermark \
    pytorch_fid \
    python-dotenv \
    wandb==0.17.8 \
    Pygments==2.16.1 \
    httpx==0.24.0

# ---------------------------------------------------------------------------
# 6. LLaVA captioner (installs from source — upgrades pydantic + torch)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    deepspeed==0.9.5 \
    einops-exts==0.0.4 \
    shortuuid==1.0.11 \
    markdown2==2.4.10

RUN pip install --no-cache-dir \
    "git+https://github.com/haotian-liu/LLaVA.git"

# ---------------------------------------------------------------------------
# 7. Re-pin pydantic v1 + torch 2.4.1 (LLaVA upgrades both)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir "pydantic==1.10.17"

RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 8. Clone flux-fine-tuner (ai-toolkit + LLaVA + supporting files)
#    Rewrite SSH → HTTPS so git@github.com: submodules work without a key
# ---------------------------------------------------------------------------
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
 && git clone --recurse-submodules \
        https://github.com/replicate/flux-fine-tuner /tmp/flux-fine-tuner \
 && for f in caption.py lora_loading_patch.py wandb_client.py \
             layer_match.py submodule_patches.py \
             hugging-face-readme-template.md; do \
        cp /tmp/flux-fine-tuner/$f /workspace/$f 2>/dev/null || true; \
    done \
 && cp -r /tmp/flux-fine-tuner/ai-toolkit /workspace/ai-toolkit \
 && cp -r /tmp/flux-fine-tuner/LLaVA      /workspace/LLaVA \
 && rm -rf /tmp/flux-fine-tuner

# ---------------------------------------------------------------------------
# 9. Patch ai-toolkit scheduler to support cosine_with_min_lr via LambdaLR
# ---------------------------------------------------------------------------
RUN python3 - << 'PYEOF'
import math

path = '/workspace/ai-toolkit/toolkit/scheduler.py'
with open(path) as f:
    content = f.read()

patch = '''    elif name == 'cosine_with_min_lr':
        import math as _math
        kwargs.pop('total_iters', None)
        warmup = kwargs.get('num_warmup_steps', 0)
        total  = kwargs.get('num_training_steps', 1000)
        floor  = kwargs.get('min_lr_rate', 0.0)
        def _lr_lambda(step, w=warmup, t=total, f=floor):
            if step < w:
                return step / max(1, w)
            progress = (step - w) / max(1, t - w)
            decay = 0.5 * (1.0 + _math.cos(_math.pi * progress))
            return f + (1.0 - f) * decay
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
'''

target = '    else:\n        # try to use a diffusers scheduler'
assert target in content, f'Target not found in {path}'
content = content.replace(target, patch + target)
with open(path, 'w') as f:
    f.write(content)
print('scheduler.py patched')
PYEOF

# ---------------------------------------------------------------------------
# 10. Copy our custom train.py and config files
# ---------------------------------------------------------------------------
COPY train.py requirements.txt /workspace/

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
# Weights are downloaded at container start via pget (~5 min).
# Output is written to /workspace/output/ and downloaded by vastai_train.sh.

ENTRYPOINT ["python", "/workspace/train.py"]
CMD ["--help"]
