#!/usr/bin/env bash
# ============================================================
# FLUX DreamBooth Trainer — Vast.ai end-to-end automation
#
# Usage:
#   ./vastai_train.sh --images <path.zip> [options]
#
# Options:
#   --images <path.zip>         Training images zip (required)
#   --trigger_word <word>       DreamBooth subject token (default: TOK)
#   --steps <n>                 Training steps (default: 1000)
#   --lora_rank <n>             LoRA rank (default: 32)
#   --gpu <name>                GPU filter (default: RTX_4090)
#   --min_inet_up <Mbps>        Min upload speed filter (default: 200)
#   --keep                      Don't destroy instance after training
#   --instance_id <id>          Re-use a running instance (skip rent+setup)
#   --skip_setup                Skip setup (weights already downloaded)
#
# Examples:
#   # Full run with prior preservation (recommended)
#   ./vastai_train.sh --images ./my_images.zip --trigger_word TOK --steps 1000 \
#       --docker_image vickwayne/flux-dreambooth-trainer:latest
#
#   # Keep instance for re-use
#   ./vastai_train.sh --images ./my_images.zip --steps 50 --keep \
#       --docker_image vickwayne/flux-dreambooth-trainer:latest
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[vastai]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]  ${NC} $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${CYAN}━━━ $* ${NC}"; }

# ── defaults ──────────────────────────────────────────────────────────────────
IMAGES=""
TRIGGER_WORD="TOK"
STEPS=1000
LORA_RANK=32
GPU="RTX_4090"
MIN_INET_UP=200
DISK_GB=80
KEEP=false
INSTANCE_ID=""
SKIP_SETUP=false
DOCKER_IMAGE="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"   # override with pre-baked image
VOLUME_ID=""                                                  # Vast.ai network volume with FLUX weights

REMOTE_DIR="/workspace/flux-dreambooth-trainer"

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --images)        IMAGES="$2";        shift 2 ;;
        --trigger_word)  TRIGGER_WORD="$2";  shift 2 ;;
        --steps)         STEPS="$2";         shift 2 ;;
        --lora_rank)     LORA_RANK="$2";     shift 2 ;;
        --gpu)           GPU="$2";           shift 2 ;;
        --min_inet_up)   MIN_INET_UP="$2";   shift 2 ;;
        --keep)          KEEP=true;          shift ;;
        --instance_id)   INSTANCE_ID="$2";   shift 2 ;;
        --skip_setup)    SKIP_SETUP=true;    shift ;;
        --docker_image)  DOCKER_IMAGE="$2";  shift 2 ;;
        --volume_id)     VOLUME_ID="$2";     shift 2 ;;
        -h|--help)
            sed -n '3,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

[[ -z "$IMAGES" ]] && error "--images is required. Usage: $0 --images my_images.zip"
[[ ! -f "$IMAGES" ]] && error "Images file not found: $IMAGES"

IMAGES_ABS="$(cd "$(dirname "$IMAGES")" && pwd)/$(basename "$IMAGES")"
IMAGES_REMOTE="${REMOTE_DIR}/$(basename "$IMAGES")"

# ── instance cleanup trap ─────────────────────────────────────────────────────
# Destroys instance on script exit unless --keep was passed
_destroy_on_exit() {
    local exit_code=$?
    if [[ -n "$INSTANCE_ID" ]] && ! $KEEP; then
        echo ""
        warn "Destroying instance $INSTANCE_ID …"
        vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
        rm -f "$SCRIPT_DIR/.vastai_last_instance"
    elif [[ -n "$INSTANCE_ID" ]] && $KEEP; then
        echo ""
        info "Instance kept: $INSTANCE_ID"
        info "  Re-use: ./vastai_train.sh --images <zip> --instance_id $INSTANCE_ID --skip_setup"
        info "  Destroy: vastai destroy instance $INSTANCE_ID"
    fi
    exit $exit_code
}
trap _destroy_on_exit EXIT

# ── step 1: find or reuse instance ───────────────────────────────────────────
if [[ -z "$INSTANCE_ID" ]]; then
    step "1/7  Finding cheapest $GPU on Vast.ai"

    OFFER_JSON=$(vastai search offers \
        "gpu_name=${GPU} num_gpus=1 disk_space>=${DISK_GB} inet_up>=${MIN_INET_UP}" \
        --order 'dph_total' --raw)

    OFFER_COUNT=$(echo "$OFFER_JSON" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
    [[ "$OFFER_COUNT" -eq 0 ]] && error "No $GPU offers found matching filters. Try --min_inet_up 0 or --gpu RTX_3090"

    OFFER_INFO=$(echo "$OFFER_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
o = data[0]
print(o['id'])
print(o['dph_total'])
print(o.get('geolocation', 'unknown'))
print(o.get('inet_up', 0))
")
    OFFER_ID=$(echo "$OFFER_INFO"  | sed -n '1p')
    DPH=$(echo "$OFFER_INFO"       | sed -n '2p')
    GEO=$(echo "$OFFER_INFO"       | sed -n '3p')
    INET=$(echo "$OFFER_INFO"      | sed -n '4p')

    info "Best offer: id=$OFFER_ID  price=\$$DPH/hr  location=$GEO  upload=${INET%.*}Mbps"

    step "2/7  Renting instance"
    # If a pre-baked image is used with a volume, disk only needs to hold
    # the training images + output (~5 GB). Without a volume, needs 60 GB for weights.
    EFFECTIVE_DISK=$DISK_GB
    [[ -n "$VOLUME_ID" ]] && EFFECTIVE_DISK=20

    VOLUME_ARGS=""
    if [[ -n "$VOLUME_ID" ]]; then
        VOLUME_ARGS="--link-volume $VOLUME_ID --mount-path /workspace/FLUX.1-dev"
        info "Attaching weights volume: $VOLUME_ID → /workspace/FLUX.1-dev"
    fi

    # shellcheck disable=SC2086
    CREATE_JSON=$(vastai create instance "$OFFER_ID" \
        --image "$DOCKER_IMAGE" \
        --disk  "$EFFECTIVE_DISK" \
        --ssh --direct \
        $VOLUME_ARGS \
        --raw)

    INSTANCE_ID=$(echo "$CREATE_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if 'new_contract' not in d:
    print('ERROR: ' + str(d), file=sys.stderr)
    sys.exit(1)
print(d['new_contract'])
")
    echo "$INSTANCE_ID" > "$SCRIPT_DIR/.vastai_last_instance"
    info "Instance created: $INSTANCE_ID  (saved to .vastai_last_instance)"

    # ── wait for running ──────────────────────────────────────────────────────
    step "3/7  Waiting for instance to start (pulling $IMAGE …)"
    ELAPSED=0
    while true; do
        STATUS=$(vastai show instance "$INSTANCE_ID" --raw | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('actual_status', d.get('status', 'unknown')))
")
        case "$STATUS" in
            running) info "Instance running (${ELAPSED}s)"; break ;;
            exited|error|delpending)
                error "Instance entered status '$STATUS' — check Vast.ai dashboard" ;;
        esac
        printf "\r  status: %-12s  elapsed: %ds" "$STATUS" "$ELAPSED"
        sleep 10; ELAPSED=$((ELAPSED + 10))
    done

else
    step "1/7  Re-using instance $INSTANCE_ID (skipping rent)"
    echo "$INSTANCE_ID" > "$SCRIPT_DIR/.vastai_last_instance"
fi

# ── get SSH details ───────────────────────────────────────────────────────────
SSH_INFO=$(vastai show instance "$INSTANCE_ID" --raw | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['ssh_host'])
print(d['ssh_port'])
")
SSH_HOST=$(echo "$SSH_INFO" | sed -n '1p')
SSH_PORT=$(echo "$SSH_INFO" | sed -n '2p')

SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

# ── wait for SSH ──────────────────────────────────────────────────────────────
info "Waiting for SSH on ${SSH_HOST}:${SSH_PORT} …"
ELAPSED=0
until ssh $SSH_OPTS "root@$SSH_HOST" 'echo ssh_ready' 2>/dev/null | grep -q ssh_ready; do
    printf "\r  retrying ssh … %ds" "$ELAPSED"
    sleep 5; ELAPSED=$((ELAPSED + 5))
    [[ $ELAPSED -gt 300 ]] && error "SSH not available after 5 min — check instance"
done
info "SSH connected."

# ── step 2: upload trainer files ─────────────────────────────────────────────
step "4/7  Uploading trainer files"
ssh $SSH_OPTS "root@$SSH_HOST" "mkdir -p '${REMOTE_DIR}'"
rsync -az --progress \
    -e "ssh $SSH_OPTS" \
    --exclude "FLUX.1-dev/" \
    --exclude "output/" \
    --exclude "input_images/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "ai-toolkit/" \
    --exclude "LLaVA/" \
    --exclude ".vastai_last_instance" \
    --exclude "trained-model*.tar" \
    "$SCRIPT_DIR/" \
    "root@${SSH_HOST}:${REMOTE_DIR}/"
info "Trainer uploaded."

# ── step 3: upload images ─────────────────────────────────────────────────────
step "5/7  Uploading images: $(basename "$IMAGES_ABS")"
rsync -az --progress \
    -e "ssh $SSH_OPTS" \
    "$IMAGES_ABS" \
    "root@${SSH_HOST}:${IMAGES_REMOTE}"
info "Images uploaded."

# ── step 4: setup ─────────────────────────────────────────────────────────────
if $SKIP_SETUP; then
    step "6/7  Skipping setup (--skip_setup)"
elif [[ "$DOCKER_IMAGE" != "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04" && -n "$VOLUME_ID" ]]; then
    step "6/7  Pre-baked image + volume: skipping setup entirely"
    info "All packages baked into image. Weights on volume. Ready immediately."
elif [[ "$DOCKER_IMAGE" != "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04" ]]; then
    step "6/7  Pre-baked image: only downloading weights (~5 min)"
    ssh $SSH_OPTS "root@$SSH_HOST" "
        mkdir -p /workspace/FLUX.1-dev
        pget -xf 'https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar' /workspace/
    "
    info "Weights downloaded."
else
    step "6/7  Running full setup (cloning ai-toolkit + downloading ~24 GB FLUX weights)"
    warn "This takes ~25-35 min on a fresh instance. Hang tight …"
    ssh $SSH_OPTS "root@$SSH_HOST" \
        "cd '${REMOTE_DIR}' && bash setup.sh"
    info "Setup complete."
fi

# ── step 5: train ─────────────────────────────────────────────────────────────
step "7/7  Training: $STEPS steps | rank=$LORA_RANK | trigger=$TRIGGER_WORD"
info "Live output below. Ctrl-C will cancel training AND destroy the instance."
echo ""

# -t gives a live TTY so training logs stream in real time
ssh -t $SSH_OPTS "root@$SSH_HOST" \
    "cd '${REMOTE_DIR}' && python train.py \
        --input_images  '${IMAGES_REMOTE}' \
        --trigger_word  '${TRIGGER_WORD}' \
        --steps         ${STEPS} \
        --lora_rank     ${LORA_RANK}"

# ── step 6: download output ───────────────────────────────────────────────────
echo ""
OUTPUT_LOCAL="${SCRIPT_DIR}/trained-model-$(date +%Y%m%d-%H%M%S).tar"
info "Downloading trained-model.tar → $(basename "$OUTPUT_LOCAL") …"
scp $SSH_OPTS \
    "root@${SSH_HOST}:${REMOTE_DIR}/trained-model.tar" \
    "$OUTPUT_LOCAL"

info "Saved: $OUTPUT_LOCAL"
info ""
info "INFERENCE: load the DreamBooth LoRA at scale 1.0–1.5. Start with 1.0."
info "  tar -tf '$OUTPUT_LOCAL'   # inspect contents"
info "  tar -xf '$OUTPUT_LOCAL'   # extract lora.safetensors"
echo ""

# trap handles destroy / keep on exit
