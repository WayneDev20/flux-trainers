#!/usr/bin/env bash
# ============================================================
# Deploy FLUX LoRA Trainer to a Vast.ai instance
#
# Usage:
#   ./deploy.sh <user@host> [-p <port>] [--images <path.zip>] [--setup] [--train <args>]
#
# Examples:
#   # Upload files only
#   ./deploy.sh root@123.45.67.89 -p 12345
#
#   # Upload + run setup (installs deps + downloads 24 GB weights)
#   ./deploy.sh root@123.45.67.89 -p 12345 --setup
#
#   # Upload + run setup + start training
#   ./deploy.sh root@123.45.67.89 -p 12345 --images ./my_images.zip --setup --train \
#       "--trigger_word TOK --steps 1000 --lora_rank 16"
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]  ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/workspace/flux-lora-trainer"

# ── argument parsing ──────────────────────────────────────────────────────────
HOST=""
PORT="22"
IMAGES=""
RUN_SETUP=false
TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)     PORT="$2";       shift 2 ;;
        --images)      IMAGES="$2";     shift 2 ;;
        --setup)       RUN_SETUP=true;  shift ;;
        --train)       TRAIN_ARGS="$2"; shift 2 ;;
        -*)            error "Unknown flag: $1" ;;
        *)
            if [[ -z "$HOST" ]]; then HOST="$1"; shift
            else error "Unexpected argument: $1"; fi
            ;;
    esac
done

[[ -z "$HOST" ]] && error "Usage: $0 <user@host> [-p port] [--images path.zip] [--setup] [--train \"args\"]"

SSH_OPTS="-p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"
RSYNC_OPTS="-avz --progress -e \"ssh $SSH_OPTS\""

# ── 1. Upload trainer files ───────────────────────────────────────────────────
info "Uploading trainer to ${HOST}:${REMOTE_DIR} …"

# shellcheck disable=SC2086
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    --exclude "FLUX.1-dev/" \
    --exclude "output/" \
    --exclude "input_images/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "ai-toolkit/" \
    --exclude "LLaVA/" \
    "$SCRIPT_DIR/" \
    "${HOST}:${REMOTE_DIR}/"

info "Trainer files uploaded."

# ── 2. Upload training images (optional) ─────────────────────────────────────
if [[ -n "$IMAGES" ]]; then
    if [[ ! -f "$IMAGES" ]]; then
        error "Images file not found: $IMAGES"
    fi
    IMAGES_REMOTE="${REMOTE_DIR}/$(basename "$IMAGES")"
    info "Uploading images: $(basename "$IMAGES") …"
    # shellcheck disable=SC2086
    rsync -avz --progress \
        -e "ssh $SSH_OPTS" \
        "$IMAGES" \
        "${HOST}:${IMAGES_REMOTE}"
    info "Images uploaded to ${IMAGES_REMOTE}"
fi

# ── 3. Run setup (optional) ───────────────────────────────────────────────────
if $RUN_SETUP; then
    info "Running setup on remote (this downloads ~24 GB weights — takes ~5 min on Vast.ai) …"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$HOST" \
        "cd ${REMOTE_DIR} && bash setup.sh"
    info "Setup complete."
fi

# ── 4. Start training (optional) ─────────────────────────────────────────────
if [[ -n "$TRAIN_ARGS" ]]; then
    [[ -z "$IMAGES" ]] && error "--train requires --images to also be specified"
    IMAGES_REMOTE="${REMOTE_DIR}/$(basename "$IMAGES")"
    info "Starting training …"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$HOST" \
        "cd ${REMOTE_DIR} && nohup python train.py --input_images '${IMAGES_REMOTE}' ${TRAIN_ARGS} \
         > train.log 2>&1 &
         echo \"Training started (PID \$!)\"
         echo \"Tail logs:  ssh ${HOST} -p ${PORT} tail -f ${REMOTE_DIR}/train.log\"
         echo \"Get output: scp ${HOST}:${REMOTE_DIR}/trained-model.tar ./ -P ${PORT}\""
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
info "Done. Useful commands:"
echo ""
echo "  # SSH in"
echo "  ssh ${HOST} -p ${PORT}"
echo ""
echo "  # Tail training logs"
echo "  ssh ${HOST} -p ${PORT} 'tail -f ${REMOTE_DIR}/train.log'"
echo ""
echo "  # Download trained LoRA"
echo "  scp -P ${PORT} ${HOST}:${REMOTE_DIR}/trained-model.tar ./"
echo ""
