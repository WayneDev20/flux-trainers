# ADR-0005: Invocation security and abuse controls

**Status:** Accepted
**Date:** 2026-04-17
**Deciders:** @vn
**Related:** ADR-0001 (GPU platform), ADR-0002 (storage), ADR-0004 (API)

## Context

An unprotected RunPod endpoint is a $1-per-invocation DoS target. The iOS app
must never be able to call RunPod directly — a leaked bundle secret would let
any attacker burn our GPU credit. We also need to be sure the webhook that
reports job completion actually came from our handler and not a forgery that
points `lora_url` at an attacker's R2 key.

Threat model (in scope):
- Stolen iOS API key → unlimited free training
- Leaked RunPod endpoint ID → same
- Forged webhook tricks backend into publishing an attacker's LoRA
- Handler coerced into reading/writing to an attacker-controlled R2 bucket
- One user spamming the training endpoint to grief

Out of scope: RunPod account compromise, R2 token compromise (handled by token
rotation, separate doc).

## Decision

Five controls, applied together:

1. **Backend-only RunPod calls.** iOS talks to avatar-backend; avatar-backend
   holds the RunPod API key and is the *only* caller. The RunPod endpoint ID
   never leaves server-side code.
2. **R2 keys only, never arbitrary URLs.** Handler's request schema accepts
   `dataset_r2_key` (string path within `avatar-datasets`). The handler
   prepends a fixed bucket name from env; any `s3://`, `http://`, or `../`
   in the key is rejected at parse time.
3. **HMAC-SHA256 signed webhooks.** Handler signs the completion payload with
   a shared secret (`WEBHOOK_HMAC_SECRET`, per-environment). Backend verifies
   before trusting `lora_r2_key`. Timestamp in the payload; reject if skew
   > 5 min.
4. **Per-user rate limit: 3 training jobs per rolling 24 h.** Enforced in
   avatar-backend (Postgres `training_jobs` table, count by `user_id` where
   `created_at > now() - '24h'`).
5. **Idempotency on `job_id`.** Handler refuses to re-train a `job_id` that
   already has a LoRA at `avatar-loras/<user_id>/<job_id>/lora.safetensors`
   (exits 0 with "already complete"). Backend retries are free.

## Options Considered

### Option A: Five controls above — CHOSEN

**Pros:** defense in depth; each control covers a different failure mode; HMAC + R2-key-whitelist is cheap to implement.
**Cons:** five things to get right; webhook secret rotation requires handler redeploy (acceptable, rare).

### Option B: mTLS between backend and RunPod

**Pros:** cryptographic transport-level auth.
**Cons:** RunPod Serverless doesn't support client certs on incoming requests; would need a sidecar proxy. Over-engineered for our threat model.

### Option C: Signed presigned R2 URLs instead of R2 keys

**Pros:** handler doesn't need R2 credentials at all.
**Cons:** URLs leak bucket structure; expiry windows are fragile across 30-min training jobs; handler would still need write creds for LoRA upload. Saves nothing net.

### Option D: Allow any URL, trust the caller

**Pros:** simplest.
**Cons:** SSRF risk (handler pulls from attacker-controlled host); exfiltration risk (handler pushes LoRA to attacker bucket). Unacceptable.

## Trade-off Analysis

The R2-keys-only rule (control 2) costs us the ability to test with
one-off URLs — a real annoyance during dev — but closes the SSRF vector
cleanly. HMAC webhooks (control 3) are cheap, standard, and the alternative
(trusting RunPod's source IP) is fragile because serverless worker IPs rotate.
Rate limiting at 3 jobs/24h is a product guess, not a security floor; it's
intentionally tight for launch and will be relaxed once we have abuse telemetry.

## Consequences

**Easier:**
- Every training job has a cryptographically attributable completion record
- Backend is the single chokepoint for billing, auth, and quota — one place to audit
- Dev + prod use the same contract; `WEBHOOK_HMAC_SECRET` is the only thing that differs

**Harder:**
- Dev-time testing with arbitrary image URLs is blocked — must upload to `avatar-datasets` first
- Rate limit (3/24h) will frustrate power users; we need a product-level override path
- HMAC secret rotation is a coordinated deploy (handler + backend at once)

**Revisit when:**
- Abuse patterns surface that the 5 controls don't cover (e.g., credential stuffing → add per-IP limits)
- We add a second backend consumer (e.g., admin retool) — need a second HMAC key
- Serverless platforms start supporting mTLS natively

## Action Items

1. [ ] Generate `WEBHOOK_HMAC_SECRET` (32-byte random), store in RunPod endpoint env + avatar-backend secrets
2. [ ] Implement R2-key validator in handler (`_validate_r2_key()`, rejects anything with `://`, `..`, or outside allowed prefixes); exit code 4 on violation
3. [ ] Handler signs webhook body with `hmac.new(secret, body, 'sha256').hexdigest()` → `X-Signature` header
4. [ ] Backend middleware verifies signature + timestamp skew before routing
5. [ ] Postgres migration: `training_jobs(user_id, created_at, job_id, status)` with index on `(user_id, created_at)` for the rate-limit query
6. [ ] Runbook: "rotating the webhook HMAC secret" (two-step deploy)
