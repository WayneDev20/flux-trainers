#!/usr/bin/env bash
# ============================================================
# Build and push the pre-baked FLUX LoRA trainer image
#
# Usage:
#   ./build_push.sh <dockerhub_user> [tag]
#
# Example:
#   ./build_push.sh yourname
#   ./build_push.sh yourname v2
#
# Prerequisites:
#   docker login
#   docker buildx create --use   (first time only)
# ============================================================
set -euo pipefail

USER="${1:?Usage: $0 <dockerhub_user> [tag]}"
TAG="${2:-latest}"
IMAGE="${USER}/flux-lora-trainer:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn] ${NC} $*"; }

info "Building: ${IMAGE}"
info "Platform: linux/amd64"
warn "First build takes ~20–30 min (pip installs + git clones baked in)"
warn "Subsequent builds use Docker layer cache — much faster"

# Ensure buildx builder exists
if ! docker buildx inspect flux-builder &>/dev/null; then
    info "Creating buildx builder …"
    docker buildx create --name flux-builder --use
fi
docker buildx use flux-builder

# Build + push
docker buildx build \
    --platform linux/amd64 \
    --tag "${IMAGE}" \
    --push \
    "${SCRIPT_DIR}"

info "Pushed: ${IMAGE}"
info ""
info "Use in vastai_train.sh:"
info "  ./vastai_train.sh --images ./my_images.zip --docker_image ${IMAGE} [--volume_id <id>]"
