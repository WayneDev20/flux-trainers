#!/usr/bin/env bash
# ============================================================
# FLUX LoRA Trainer — Setup Script
# Run once to clone submodules and download model weights.
# Tested on: Ubuntu 22.04 + CUDA 12.4 (Vast.ai / bare metal)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn] ${NC} $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. System dependencies (Linux only — skip on macOS)
# ---------------------------------------------------------------------------
if [[ "$(uname)" == "Linux" ]]; then
    info "Installing system dependencies …"
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git git-lfs curl wget unzip \
        libgl1 libglib2.0-0 \
        build-essential \
    || warn "apt-get install failed — continuing (packages may already be present)"
fi

# ---------------------------------------------------------------------------
# 1. Clone flux-fine-tuner (parent repo) if not already here
# ---------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/caption.py" ]; then
    info "Cloning replicate/flux-fine-tuner …"

    # Rewrite git@github.com: → https://github.com/ so cloning works on
    # machines without a GitHub SSH key (e.g. fresh Vast.ai instances)
    git config --global url."https://github.com/".insteadOf "git@github.com:"

    TMP=$(mktemp -d)
    git clone --recurse-submodules https://github.com/replicate/flux-fine-tuner "$TMP/flux-fine-tuner"

    # Copy all supporting files verbatim from upstream (NOT train.py — we keep ours)
    for f in \
        caption.py \
        lora_loading_patch.py \
        wandb_client.py \
        layer_match.py \
        submodule_patches.py \
        hugging-face-readme-template.md; do
        if [ -f "$TMP/flux-fine-tuner/$f" ]; then
            cp "$TMP/flux-fine-tuner/$f" "$SCRIPT_DIR/$f"
            info "  Copied $f"
        else
            warn "  $f not found in upstream repo — skipping"
        fi
    done

    # Copy ai-toolkit submodule
    if [ -d "$TMP/flux-fine-tuner/ai-toolkit" ]; then
        cp -r "$TMP/flux-fine-tuner/ai-toolkit" "$SCRIPT_DIR/ai-toolkit"
        info "  Copied ai-toolkit submodule"
    fi

    # Copy LLaVA submodule (needed for captioner imports)
    if [ -d "$TMP/flux-fine-tuner/LLaVA" ]; then
        cp -r "$TMP/flux-fine-tuner/LLaVA" "$SCRIPT_DIR/LLaVA"
        info "  Copied LLaVA submodule"
    fi

    rm -rf "$TMP"
    info "Done cloning supporting files."
else
    info "Supporting files already present — skipping clone."
fi

# Ensure ai-toolkit submodule is initialised if we're inside the repo
if [ -f "$SCRIPT_DIR/.gitmodules" ] && [ ! -f "$SCRIPT_DIR/ai-toolkit/README.md" ]; then
    info "Initialising git submodules …"
    git submodule update --init --recursive
fi

# ---------------------------------------------------------------------------
# 2. Python environment check
# ---------------------------------------------------------------------------
PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" &>/dev/null; then
    error "Python not found. Set PYTHON= or install Python 3.10."
fi

# Ensure pip is up to date
$PYTHON -m pip install --quiet --upgrade pip

PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VER" != "3.10" ]]; then
    warn "Detected Python $PY_VER — recommended is 3.10.14. Proceeding anyway."
fi

# ---------------------------------------------------------------------------
# 2b. Install pget (Replicate's parallel downloader — same tool their train.py uses)
# ---------------------------------------------------------------------------
if ! command -v pget &>/dev/null; then
    info "Installing pget (parallel downloader) …"
    PGET_BIN="/usr/local/bin/pget"
    curl -o "$PGET_BIN" -fsSL \
        "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64"
    chmod +x "$PGET_BIN"
    info "pget installed."
else
    info "pget already installed: $(pget --version 2>/dev/null || echo 'ok')"
fi

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------
info "Installing Python dependencies …"

# PyTorch (CUDA 12.4)
$PYTHON -m pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Core requirements
$PYTHON -m pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# Diffusers fork with FLUX inpaint
$PYTHON -m pip install --no-cache-dir \
    "git+https://github.com/Gothos/diffusers.git@flux-inpaint"

# LLaVA (captioner)
$PYTHON -m pip install --no-cache-dir \
    "git+https://github.com/haotian-liu/LLaVA.git"

# ai-toolkit internal deps (not in cog.yaml but required at import time)
$PYTHON -m pip install --no-cache-dir \
    k-diffusion invisible-watermark pytorch_fid python-dotenv

# LLaVA upgrades pydantic to v2 and torch to 2.1.2 — re-pin both.
# pydantic v2 breaks diffusers/SDTrainer; torch must match our CUDA index.
$PYTHON -m pip install --no-cache-dir "pydantic==1.10.17"
$PYTHON -m pip install --no-cache-dir --force-reinstall \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

info "Python dependencies installed."

# ---------------------------------------------------------------------------
# 4. Download FLUX.1-dev weights
# ---------------------------------------------------------------------------
WEIGHTS_DIR="$SCRIPT_DIR/FLUX.1-dev"
SENTINEL="$WEIGHTS_DIR/transformer/config.json"
CDN_URL="https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"

if [ -f "$SENTINEL" ]; then
    info "FLUX.1-dev weights already at $WEIGHTS_DIR — skipping download."
else
    info "Downloading FLUX.1-dev from Replicate CDN …"
    info "  Source: $CDN_URL"
    info "  Size: ~24 GB — no HF token required"

    mkdir -p "$WEIGHTS_DIR"

    # pget is now installed — use it (parallel download, 3-5x faster than wget)
    if command -v pget &>/dev/null; then
        pget -xf "$CDN_URL" "$WEIGHTS_DIR/.."
    else
        TAR_PATH="$SCRIPT_DIR/flux-dev.tar"
        if command -v wget &>/dev/null; then
            wget -c --show-progress -O "$TAR_PATH" "$CDN_URL"
        else
            curl -L --continue-at - -o "$TAR_PATH" "$CDN_URL"
        fi
        info "Extracting archive …"
        tar -xf "$TAR_PATH" -C "$WEIGHTS_DIR/.."
        rm -f "$TAR_PATH"
    fi

    # Hash transformer shards so we can identify the exact version
    info "Transformer weight hashes (paste these to verify / pin exact version):"
    for shard in "$WEIGHTS_DIR"/transformer/*.safetensors; do
        [ -f "$shard" ] && echo "  $(sha256sum "$shard")"
    done

    info "FLUX.1-dev downloaded to $WEIGHTS_DIR"
fi

# ---------------------------------------------------------------------------
# 5. Smoke test
# ---------------------------------------------------------------------------
info "Running smoke test …"
$PYTHON -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/ai-toolkit')
import torch
print(f'  torch     : {torch.__version__}')
print(f'  CUDA      : {torch.version.cuda}')
print(f'  GPU count : {torch.cuda.device_count()}')
try:
    import diffusers
    print(f'  diffusers : {diffusers.__version__}')
except Exception as e:
    print(f'  diffusers : {e}')
try:
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    print('  ai-toolkit: OK')
except ImportError as e:
    print(f'  ai-toolkit: FAILED — {e}')
"

touch "$SCRIPT_DIR/.setup_done"

echo ""
info "Setup complete. Example run:"
echo ""
echo "  python train.py \\"
echo "    --input_images  ./my_images.zip \\"
echo "    --trigger_word  TOK \\"
echo "    --steps         1000 \\"
echo "    --lora_rank     16"
echo ""
info "Output will be written to ./output/ and packaged as ./trained-model.tar"
info "INFERENCE: load the LoRA at scale 1.5 (not 1.0) for best resemblance."
