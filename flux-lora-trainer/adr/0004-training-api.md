# ADR-0004: Training API config surface and captioning ownership

**Status:** Accepted
**Date:** 2026-04-17
**Deciders:** @vn
**Related:** ADR-0001 (GPU platform), ADR-0005 (invocation security)

## Context

The handler wraps the existing `/workspace/train.py` argparse wrapper. We need
to decide:
- Which training hyperparameters are exposed to the caller vs. hardcoded.
- How captions are supplied, and who owns VLM captioning.

Goal: small, stable request contract that the avatar-backend can call without
knowing ai-toolkit internals, while still letting us bump `lora_rank` /
`steps` / GPU tier without redeploying the backend.

## Decision

**One `config{}` object with exactly 8 knobs** in the RunPod request body:

```
steps, lora_rank, learning_rate, batch_size, resolution,
optimizer, caption_dropout_rate, gpu_tier
```

Everything else (scheduler, EMA, noise schedule, etc.) is locked in the
handler. Defaults match Replicate's `ostris/flux-dev-lora-trainer` so existing
user expectations hold.

**Dataset shape:** `dataset_r2_key` (zip of images) + optional `captions{}`
JSON map from filename → caption.

**Caption rule (no silent autocaption):**
- `captions` field present AND complete → use as-is
- `captions` field present but missing entries → **abort with error**
- `captions` field absent entirely → handler invokes ai-toolkit's built-in
  autocaptioner as last-resort fallback

**Captioning owner = avatar-backend (Phase 2).** Phase 1 ships a manual dev
tool `tools/caption_dataset.py`. VLM = **Claude Sonnet via Anthropic API with
prompt caching** (PENDING USER CONFIRMATION). Prompt instructs the VLM to
describe face, hair, beard, skin, and identity markers exhaustively — we do
not plan to generate with different hair, so caption diversity is not a goal.

## Options Considered

### Option A: 8-knob `config{}` + backend-owned captions — CHOSEN

**Pros:** small surface, versionable, backend can A/B captioning strategies without touching the handler, handler stays a thin training wrapper.
**Cons:** backend must implement VLM captioning before Phase 2; the no-silent-autocaption rule surprises users who expect "just work" behaviour.

### Option B: Expose all ~30 ai-toolkit knobs

**Pros:** maximum flexibility for research.
**Cons:** huge request surface; every backend caller becomes coupled to ai-toolkit internals; breaking changes on upstream bump propagate everywhere.

### Option C: Handler owns VLM captioning

**Pros:** one-stop shop — caller sends images, gets LoRA.
**Cons:** VLM latency lives inside the GPU-billed window (wastes A100 seconds on API calls); harder to swap VLMs; mixes concerns.

### Option D: Always autocaption if `captions` missing any entries

**Pros:** callers never get errors on partial captions.
**Cons:** silent mixing of human + machine captions degrades LoRA quality unpredictably; violates "no silent drift" directive.

## Trade-off Analysis

Option A trades convenience (C's one-shot API) for clean separation: the GPU
handler does GPU work, VLM captioning runs on CPU time in the backend where
retries, caching, and cost tracking are already solved. The strict caption
validation (Option A vs D) is explicitly a reliability choice — loud errors
beat silent quality loss.

## Consequences

**Easier:**
- Backend can iterate on caption prompts without redeploying the GPU image
- Request contract is small enough to version as `v1`, `v2`
- Autocaption fallback still exists as an escape hatch for manual testing

**Harder:**
- Phase 2 blocks on backend VLM integration; Phase 1 users must caption via `tools/caption_dataset.py` manually
- Any new training knob (e.g., DoRA rank) requires a handler bump + backend change
- Claude Sonnet VLM cost is now a line item in the backend's per-job economics

**Revisit when:**
- We want to expose FLUX Schnell / DoRA / multi-resolution (will need more knobs)
- VLM captioning latency becomes the critical path (move to streaming / parallel)

## Action Items

1. [ ] User confirms Claude Sonnet as VLM choice (vs. GPT-4o / Gemini)
2. [ ] Freeze `config{}` JSON schema in `serverless/schemas/request.v1.json`
3. [ ] Implement caption-completeness check in handler, exit code 3 on mismatch
4. [ ] Ship `tools/caption_dataset.py` (Anthropic API, prompt caching, writes `captions.json`)
5. [ ] Document the prompt in `tools/prompts/describe_identity.md` (version-pinned)
6. [ ] Phase 2: avatar-backend integrates the same captioning module as a library

## Revision log

### 2026-04-17 — `captions` is now REQUIRED; LLaVA autocaption fallback removed

The "autocaption fallback when `captions` is absent" path has been deleted.

**Why:** the base image's LLaVA install pulled a fragile dep regime —
pydantic v1 pin (conflicted with fastapi 0.115 in the serverless layer) and
numpy <2 pin (deepspeed's `from numpy import BUFSIZE` breaks on numpy 2.x).
Two full CI rebuilds (+ one GPU-billed training run that OOM'd at
import-time) were burned on resolving those conflicts. The autocaption
fallback was never actually exercised in production — avatar-backend always
pre-captions — so it was paying ongoing complexity tax for zero value.

**What changed:**
- `Dockerfile` no longer installs LLaVA / deepspeed / einops-exts / etc.
- `train.py` no longer imports `Captioner` or accepts `--autocaption*` args.
- `serverless/_validation.py::check_captions_complete` now raises
  `ValidationError(EXIT_BAD_REQUEST)` when `captions is None`.
- `serverless/schemas/request.v1.json` lists `captions` in `required`.
- `serverless/Dockerfile.runpod` has a build-time guard that fails the
  image build if `llava` or `deepspeed` ever reappear in the base.

**Caption prompt** also revised from "describe everything including
identity markers" to "describe only variable-at-inference attributes"
(background, clothing, accessories, pose, framing, lighting). Identity
features (skin, hair, face, build) are intentionally *omitted* from
captions so the LoRA learns to bake them into the trigger token rather
than leaving them dangling as VLM-controllable dials. See
`tools/prompts/describe_identity.md` v2.

**Migration for callers:** none — avatar-backend already pre-captions via
`tools/caption_dataset.py`, so the request payload shape is unchanged for
real traffic. Manual testers who previously submitted without `captions`
now get exit 4 instead of a slow LLaVA fallback.
