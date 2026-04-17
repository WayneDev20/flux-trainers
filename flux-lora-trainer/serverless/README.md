# Serverless Training Endpoint

Self-hosted FLUX LoRA trainer. Primary lane = RunPod Serverless (A100 80 GB),
fallback = Modal (Phase 3), batch = existing vast.ai script.

**Read first:** `../adr/0001..0005.md` → `DESIGN.md` → `TESTING.md`.

## Layout

| Path | Purpose |
|------|---------|
| `DESIGN.md` | Handler dataflow, failure modes, idempotency, cold-start budget |
| `TESTING.md` | Test pyramid, coverage rules, fixtures |
| `handler.py` | Slim orchestrator. RunPod entry. |
| `_validation.py` | Request schema + R2-key SSRF guard + caption completeness |
| `_webhook.py` | HMAC-SHA256 sign / verify / retry POST |
| `_weights.py` | SHA-256 parity check (ADR-0003) |
| `_storage.py` | boto3 R2 wrapper with typed read/write errors |
| `_training.py` | `train.py` subprocess wrapper |
| `schemas/request.v1.json` | Frozen request contract (ADR-0004) |
| `mirror_flux_weights.py` | One-time: mirror Replicate CDN tar → R2 + manifest |
| `flux_weights_manifest.json` | **Committed.** Source of truth for parity. Created by `mirror_flux_weights.py`. |
| `Dockerfile.runpod` | Thin layer on `flux-lora-trainer:latest` |
| `requirements-serverless.txt` | Pinned Python deps for the serverless layer only |

## Bootstrap checklist

Prereq: accounts in `../adr/0001-gpu-platform.md` and `../adr/0002-object-storage.md` exist.

### 1. R2 buckets + tokens
```bash
wrangler login                                    # OAuth once
wrangler r2 bucket create flux-weights-mirror
wrangler r2 bucket create avatar-datasets
wrangler r2 bucket create avatar-loras
wrangler r2 bucket lifecycle add avatar-datasets \
    --prefix=datasets/ --expire-days=30
# Create two scoped API tokens in the Cloudflare dashboard:
#   mirror-writer: R/W on flux-weights-mirror (laptop only)
#   handler:       R on flux-weights-mirror + avatar-datasets,
#                  R/W on avatar-loras       (RunPod endpoint env only)
```

### 2. Mirror FLUX weights (one-time)
```bash
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...  # mirror-writer
cd flux-lora-trainer
python -m serverless.mirror_flux_weights \
    --version v1 --bucket flux-weights-mirror \
    --also-write-local ~/flux-cache
git add serverless/flux_weights_manifest.json
git commit -m "pin FLUX.1-dev weights v1 (ADR-0003)"
```

### 3. Build + push serverless image
```bash
gh auth status                                                # must show Concept-fashion-ML access
echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin
# (or: gh auth token | docker login ghcr.io -u <username> --password-stdin)

docker build -f serverless/Dockerfile.runpod \
    --build-arg BASE_IMAGE=ghcr.io/concept-fashion-ml/flux-lora-trainer:latest \
    --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
    -t ghcr.io/concept-fashion-ml/flux-lora-trainer-runpod:v1 .
docker push ghcr.io/concept-fashion-ml/flux-lora-trainer-runpod:v1
```

### 4. Populate RunPod Network Volume
Copy `~/flux-cache/*.safetensors` to a fresh 30 GB Network Volume in US-East,
mount point `/workspace/FLUX.1-dev`.

### 5. Create RunPod endpoint
```bash
runpodctl config --apiKey $RUNPOD_API_KEY
runpodctl project create-endpoint \
    --name flux-lora-trainer \
    --image ghcr.io/concept-fashion-ml/flux-lora-trainer-runpod:v1 \
    --gpu-type NVIDIA_A100_80GB_PCIE \
    --network-volume-id <vol-id> --mount-path /workspace/FLUX.1-dev \
    --execution-timeout 3600 --max-retries 2 --min-workers 0 \
    --env R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    --env R2_BUCKET_DATASETS=avatar-datasets \
    --env R2_BUCKET_LORAS=avatar-loras \
    --env R2_BUCKET_WEIGHTS_MIRROR=flux-weights-mirror \
    --env WEBHOOK_HMAC_SECRET=$(openssl rand -hex 32) \
    --env FLUX_WEIGHTS_VERSION=v1 \
    --env SENTRY_DSN=...
```
(Save the WEBHOOK_HMAC_SECRET value — avatar-backend needs the same string.)

### 6. Smoke test
See `../tests/smoke/README.md`.

## Ongoing ops

- **New FLUX revision** → `python -m serverless.mirror_flux_weights --version v2 ...`, commit manifest, rebuild image, deploy. Old LoRAs keep pointing at their training-time version via the manifest stored alongside each LoRA.
- **Rollback** → set `FLUX_WEIGHTS_VERSION=v1` on the endpoint. Cold start picks it up. LoRAs are safe — they were trained against the version recorded in their manifest.
- **Rotate webhook secret** → two-step deploy (add second secret accepted by backend; flip handler to new secret; remove old from backend).
