# ADR-0001: GPU platform for per-user FLUX LoRA training

**Status:** Accepted
**Date:** 2026-04-17
**Deciders:** @vn
**Supersedes:** none

## Context

We're replacing our use of Replicate's hosted `ostris/flux-dev-lora-trainer` with
a self-hosted equivalent. Each job trains a per-user LoRA in ~25–35 min on
A100-class GPUs. Target volume (launch): 10–100 jobs/day. Parity requirement
with Replicate's FLUX.1-dev weights is absolute (ADR-0003).

Non-functional constraints:
- production-grade, hardened, no silent drift
- cold starts OK (user tolerates a queued job state)
- no warm-worker spend on launch (user directive)
- must tolerate a single-provider outage

## Decision

Three-lane GPU strategy:

- **Primary: RunPod Serverless** (A100 80GB → H100/H200 later, `min_workers=0`)
- **Fallback: Modal** (same Python handler logic, separate SDK, invoked by orchestrator on RunPod 5xx / capacity error)
- **Batch lane: existing vast.ai `vastai_train.sh`** for non-user-facing bulk runs

**Container registry: PENDING** — recommend GHCR under a new GitHub org
(`ghcr.io/<new-org>/flux-lora-trainer`) over Docker Hub: zero new credentials
(reuses `gh` auth), unlimited private repos, higher pull rate limits. Decision
deferred pending user confirmation.

## Options Considered

### Option A: RunPod Serverless (primary) + Modal (fallback) + vast.ai (batch) — CHOSEN

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — two platforms to keep wired |
| Cost | A100 ~$1.08/30-min job; no idle spend |
| Scalability | Auto-scale to endpoint max |
| Team familiarity | vast.ai script exists; RunPod/Modal new |
| Max job duration | 7 days (RunPod) / 24h (Modal) — both fine |
| Cold start | 8–30 s with Network Volume |

**Pros:** production-grade serverless on primary, clean fallback path, keeps existing vast.ai script as escape hatch, H100/H200 upgrade = env var change.
**Cons:** two handlers to maintain (RunPod `handler.py` + `modal_app.py`); two auth systems; two billing dashboards.

### Option B: RunPod Pod pool (persistent)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low — one long-lived pod, SSH train-and-kill |
| Cost | ~$1.60/hr 24/7 = $1,150/mo for one A100 idle |
| Scalability | Manual — one pod, one job at a time |

**Pros:** warmest possible starts; simpler ops.
**Cons:** burns money when idle; doesn't scale horizontally; violates user's "no warm worker" directive.

### Option C: Modal as primary

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low — Python-native, excellent DX |
| Cost | ~$1.60/30-min job (~40% more than RunPod) |
| Scalability | Excellent |

**Pros:** fastest dev velocity; best-in-class cold starts with volumes.
**Cons:** costlier per job; ecosystem lock-in to Modal's runtime abstractions.

### Option D: vast.ai on-demand only (status quo)

**Pros:** cheapest ($0.65–$2/hr A100).
**Cons:** community hosts disappear mid-job; no webhook SLA; not production-grade for per-user jobs.

## Trade-off Analysis

We accept +40% cost over vast.ai and +20% operational surface (two SDKs) to get
production reliability. A single-provider outage is a real risk at our scale
(RunPod has had regional 503s in the past); Modal as warm fallback costs us
almost nothing unless we actually fail over to it. The vast.ai lane stays for
internal experiments where interruption is acceptable.

## Consequences

**Easier:**
- Scales from zero without idle spend
- Single-tenant isolation per job (fresh container each time)
- GPU upgrade (A100 → H100/H200) is an endpoint config change, not a code change

**Harder:**
- Two separate deploys (RunPod image + Modal app) must stay in sync
- Debugging cold-start weirdness across two runtimes
- Cost attribution split across three providers

**Revisit when:**
- Daily volume > 200 jobs — evaluate RunPod Pod pool (becomes cheaper than serverless past ~16 GPU-hr/day)
- RunPod regional outages become frequent (> monthly)

## Action Items

1. [ ] Confirm container registry (new GHCR org vs Docker Hub) — user input pending
2. [ ] Create RunPod account, generate API key, set region to US-East
3. [ ] Create Network Volume (30 GB) in US-East, mount path `/workspace/FLUX.1-dev`
4. [ ] Build serverless image (see ADR-0003 for weight-baking strategy)
5. [ ] Register RunPod endpoint with `executionTimeout: 3600`, `maxRetries: 2`, `min_workers: 0`
6. [ ] Defer Modal fallback to Phase 3 (per user's phased plan); stub `modal_app.py` with a TODO
