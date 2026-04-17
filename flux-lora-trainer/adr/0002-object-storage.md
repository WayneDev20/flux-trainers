# ADR-0002: Object storage for datasets, LoRAs, and base weight mirror

**Status:** Accepted
**Date:** 2026-04-17
**Deciders:** @vn
**Related:** ADR-0001 (GPU platform), ADR-0003 (weight parity)

## Context

Training produces three classes of blobs that need durable storage:

1. **Base FLUX weights** (~24 GB total) — shared, read-heavy, version-pinned (ADR-0003). Every training and inference job reads them.
2. **Per-user datasets** (~2–50 MB zipped) — ingested once, referenced once, retained 30 days for retrain.
3. **Per-user LoRAs + samples + manifest** (~200–400 MB per user) — written once, read many times by the inference service, kept forever.

Egress dominates the cost model: every inference call downloads the LoRA. 10 k
users × 20 inference fetches/month × 250 MB = 50 TB/month egress.

## Decision

**Cloudflare R2** with three buckets, versioned paths, and a matching IAM
pattern:

| Bucket | Content | Lifecycle |
|--------|---------|-----------|
| `flux-weights-mirror` | Base FLUX weights under `v1/`, `v2/`, … prefixes | retain forever |
| `avatar-datasets` | Per-user training zips, keyed by `<job_id>.zip` | 30-day TTL |
| `avatar-loras` | Per-user LoRA + samples + manifest under `<user_id>/<job_id>/` | retain forever; user-deletable |

Two scoped API tokens:
- **mirror-writer**: R/W on `flux-weights-mirror`, scripted from laptop only
- **handler**: R on `flux-weights-mirror` + `avatar-datasets`, R/W on `avatar-loras`

## Options Considered

### Option A: Cloudflare R2 — CHOSEN

| Dimension | Assessment |
|-----------|------------|
| Storage | $0.015/GB/mo |
| Egress | **$0** |
| Request cost | $0.36/M class-A, $4.50/M class-B |
| Latency | Sub-100 ms globally via Workers edge |
| S3 API compat | Full `boto3` / `aws-sdk` compatibility |

**Pros:** free egress is a 10× cost lever at scale; S3-compatible (zero code change if we ever migrate); CF Workers integration if we want signed URL issuance at the edge.
**Cons:** single-region writes (no multi-region replication without extra tooling); smaller ecosystem than S3 for third-party tools.

### Option B: AWS S3

| Dimension | Assessment |
|-----------|------------|
| Storage | $0.023/GB/mo |
| Egress | $0.09/GB after first GB/mo |
| Ecosystem | Best-in-class |

**Pros:** de facto standard; lifecycle + IAM battle-tested.
**Cons:** 50 TB/mo egress = $4,500/mo at 10 k users. Dealbreaker.

### Option C: Backblaze B2

| Dimension | Assessment |
|-----------|------------|
| Storage | $0.006/GB/mo |
| Egress | $0.01/GB after 3× stored |

**Pros:** cheapest raw storage.
**Cons:** egress fee kicks in; smaller global PoP footprint; weaker S3 API compat than R2.

### Option D: Self-host MinIO on a VPS

**Pros:** no per-GB fees.
**Cons:** we run storage now; durability on single-node VPS is awful; operator burden antithetical to "production-grade".

## Trade-off Analysis

Egress economics pick R2 by ~10×. The single-region-writes limitation is
acceptable because:
- Weights are read-mostly and can be fronted by CF Workers cache if latency
  becomes an issue.
- LoRAs are written once; write region doesn't matter much.
- Datasets are write-once, read-once.

S3 would be defensible if we had existing AWS infra or a compliance requirement.
We have neither.

## Consequences

**Easier:**
- Scale to 10 k+ users without egress-cost panic
- Sub-100 ms reads globally once we front with Workers
- Standard `boto3` code — zero R2-specific APIs in our handlers

**Harder:**
- Cloudflare outages affect all three buckets (single-vendor risk)
- No native cross-region replication — if we need multi-region DR, tooling to build
- Fewer third-party monitoring integrations than S3

**Revisit when:**
- We need compliance attestations R2 doesn't carry (HIPAA, FedRAMP — currently only AWS/GCP)
- Multi-region active-active becomes a product requirement

## Action Items

1. [ ] User creates Cloudflare account; completes OAuth for `cloudflare-bindings` MCP
2. [ ] Create three buckets via Wrangler: `wrangler r2 bucket create <name>`
3. [ ] Create two API tokens with the scoped permissions described above, store in `runpodctl` endpoint env and laptop's shell profile respectively
4. [ ] Configure 30-day TTL lifecycle on `avatar-datasets` via `wrangler r2 bucket lifecycle add`
5. [ ] Document bucket layout in `flux-lora-trainer/serverless/README.md`
