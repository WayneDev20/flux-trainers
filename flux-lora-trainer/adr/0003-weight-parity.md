# ADR-0003: Base weight parity between training and inference

**Status:** Accepted
**Date:** 2026-04-17
**Deciders:** @vn
**Related:** ADR-0002 (storage)

## Context

Replicate's hosted `ostris/flux-dev-lora-trainer` downloads FLUX.1-dev base
weights from a hardcoded CDN URL:

```
https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar
```

LoRAs produced against one set of base weights will silently underperform when
loaded against a subtly different set (different precision, different VAE
variant, different text-encoder slicing). Replicate can re-upload `files.tar`
at any time without versioning — we'd drift without knowing. User's own words:
*"make sure the models used match both training and inference."*

The same weights must be used at:
- **training time** (this pipeline's RunPod handler)
- **inference time** (future inference service, ADR-xxxx later)

A URL-based reference is not sufficient. We need a content-addressed anchor.

## Decision

**SHA-256-pinned mirror of the upstream tar, versioned by directory prefix in
R2.** The pipeline's bootstrap script (`mirror_flux_weights.py`) does a one-time
download from Replicate's CDN, extracts the four required files, computes and
records SHA-256 for each, uploads to `R2://flux-weights-mirror/v1/`, and emits
a committed-to-git manifest:

```json
{
  "version": "v1",
  "source": "https://weights.replicate.delivery/.../files.tar",
  "mirror_prefix": "v1/",
  "files": {
    "flux1-dev.safetensors":  { "sha256": "…", "bytes": … },
    "ae.safetensors":         { "sha256": "…", "bytes": … },
    "t5xxl_fp16.safetensors": { "sha256": "…", "bytes": … },
    "clip_l.safetensors":     { "sha256": "…", "bytes": … }
  }
}
```

Every training job verifies each file's SHA on container boot. Mismatch = abort
with a loud error. Inference service (later) consumes the same manifest.
Version bumps create a new directory (`v2/`), never overwrite `v1/`; old LoRAs'
manifests point at their originating weight version forever.

## Options Considered

### Option A: SHA-pinned R2 mirror with versioned prefixes — CHOSEN

| Dimension | Assessment |
|-----------|------------|
| Parity guarantee | Cryptographic |
| Drift detection | Automatic at boot |
| Rollback | Swap `FLUX_WEIGHTS_VERSION` env var |

**Pros:** bit-identical guarantee; independent of upstream mutations; cheap to bump versions; makes provenance auditable.
**Cons:** one-time 24 GB download + upload per version; manifest is a new object to keep in git.

### Option B: Pull directly from Replicate CDN at boot (status quo in `train.py`)

**Pros:** zero bootstrap; less of our storage used.
**Cons:** silent drift if Replicate re-uploads; cold-start download each time unless cached; no integrity check.

### Option C: HuggingFace pinned revision SHA (`black-forest-labs/FLUX.1-dev@<sha>`)

| Dimension | Assessment |
|-----------|------------|
| Parity guarantee | HF commit SHA |
| File equivalence with Replicate CDN | **Unverified — precision/format may differ** |

**Pros:** canonical upstream; HF Hub tooling.
**Cons:** files differ subtly between Replicate's packaging and HF's raw files. Using HF breaks parity with Replicate-trained LoRAs (the user's existing LoRAs); ecosystem mismatch.

### Option D: Bake weights into the Docker image

**Pros:** no runtime download.
**Cons:** 24 GB image makes every cold start slow (~3–5 min pull); pushes GHCR rate limits; version rollback = full image rebuild.

## Trade-off Analysis

Option A costs one 24 GB upload per version (< $0.50 on R2) and gives us the
strongest guarantee available short of embedding weights in source control. The
HF path (Option C) is tempting but verifiably produces different safetensors —
any LoRA we've ever trained via Replicate would need to be re-evaluated.

## Consequences

**Easier:**
- Training↔inference parity is a boolean check, not a trust exercise
- Future FLUX revisions slot in as `v2/`, `v3/` without touching existing LoRAs
- Audit: "what weights was user X's LoRA trained against?" → read their manifest

**Harder:**
- Adds a bootstrap step (run `mirror_flux_weights.py` once)
- Requires holding a mirror copy even though Replicate hosts the originals
- Network Volume weights must be re-populated when version bumps

**Revisit when:**
- Replicate deprecates their CDN (switch to HF `@sha` pinning)
- Multiple FLUX variants (Schnell, DoRA-compatible, etc.) are in play simultaneously

## Action Items

1. [ ] Run `mirror_flux_weights.py --version v1 --bucket flux-weights-mirror --also-write-local ~/flux-cache`
2. [ ] Commit `flux_weights_manifest.json` to git (source of truth)
3. [ ] Populate RunPod Network Volume from `~/flux-cache`
4. [ ] Handler implements SHA verification on boot; aborts with exit code 2 on mismatch
5. [ ] Training manifest written alongside each LoRA records the weights version it was trained against
6. [ ] Future inference service MUST consume the same `flux_weights_manifest.json` — enforce via a shared package or git submodule
