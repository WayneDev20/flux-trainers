# Handler design — RunPod Serverless worker

**Scope:** the RunPod worker only. Not the backend, not the iOS app, not
inference. ADRs 0001–0005 are inputs to this document.

## 1. Request / response contract

**Request** (RunPod `input` field):
```json
{
  "job_id": "uuid-v4",
  "user_id": "uuid-v4",
  "dataset_r2_key": "datasets/<job_id>.zip",
  "captions": { "img01.jpg": "...", "img02.jpg": "..." },
  "trigger_word": "TOK",
  "config": {
    "steps": 1000,
    "lora_rank": 32,
    "learning_rate": 1e-3,
    "batch_size": 1,
    "resolution": "512,768,1024",
    "optimizer": "adamw8bit",
    "caption_dropout_rate": 0.05,
    "gpu_tier": "a100_80gb"
  }
}
```
Rationale for the 8-knob surface → ADR-0004. `captions` is **required**: the
in-process LLaVA autocaption fallback was removed from the base image (it
dragged a fragile dep regime — pydantic v1 + numpy <2 — with it). Captioning
is owned by `tools/caption_dataset.py` in avatar-backend (Claude Sonnet
primary, GPT / Gemini fallback). Missing `captions` → exit 4. Partial
`captions` → exit 3.

**Webhook** (HMAC-SHA256 signed, `X-Signature` header; ADR-0005):
```json
{
  "job_id": "...", "user_id": "...", "status": "succeeded|failed",
  "lora_r2_key": "loras/<user_id>/<job_id>/lora.safetensors",
  "samples_r2_prefix": "loras/<user_id>/<job_id>/samples/",
  "manifest_r2_key": "loras/<user_id>/<job_id>/manifest.json",
  "weights_version": "v1", "duration_s": 1834, "ts": 1713398400
}
```

## 2. Dataflow

```
RunPod input
    │
    ▼
[1] parse + validate request       ── exit 4 on bad R2 key / schema
    │
    ▼
[2] idempotency probe              ── HEAD avatar-loras/<user_id>/<job_id>/lora.safetensors
    │                                 hit → emit "already_complete" webhook, exit 0
    ▼
[3] SHA verify FLUX weights        ── exit 2 on mismatch (ADR-0003)
    │
    ▼
[4] download dataset.zip           ── boto3 get_object → /tmp/<job_id>.zip
    │
    ▼
[5] write caption .txt files       ── captions{} required; rebuild zip with
    │                                 <stem>.txt entries so train.py's
    │                                 extract_zip places them in INPUT_DIR/
    │                                 alongside their images
    ▼
[6] subprocess train.py            ── argparse flags built from config{}
    │                                 stdout/stderr streamed with job_id prefix
    ▼
[7] upload artifacts               ── lora.safetensors, samples/*.jpg, manifest.json
    │                                 → avatar-loras/<user_id>/<job_id>/
    ▼
[8] POST webhook (HMAC signed)     ── retry 3× w/ exp backoff, then log+exit 0
                                      (status already durable in R2)
```

## 3. Failure modes

| Mode | Detect | Exit | Recovery |
|------|--------|------|----------|
| Bad FLUX SHA | SHA256 mismatch on boot | 2 | Abort. Alert — weights corrupted or version drift |
| Schema violation / bad R2 key / missing captions | jsonschema + `_validate_r2_key()` | 4 | Client bug. Webhook `failed`; no retry |
| Partial captions | set diff vs. zip contents | 3 | Client bug. Webhook `failed`; no retry |
| R2 read fail (dataset) | boto3 ClientError | 5 | Retry 3× internally; then `failed` |
| train.py crash | non-zero returncode | 6 | Capture last 200 stderr lines into manifest; `failed`. Backend may retry |
| R2 write fail (artifact) | boto3 ClientError | 7 | Retry 3×; then `failed` (training $ already spent) |
| Webhook POST fail | HTTP != 2xx after retries | 0 | Log only. State is in R2; backend reconciles |
| OOM | CUDA OOM in train.py stderr | 6 | `failed` w/ oom=true hint. Backend re-queues at smaller batch |
| GPU timeout (>3600s) | RunPod `executionTimeout` | — | RunPod kills worker; no webhook. Backend reconciler sweeps |

## 4. Idempotency

`job_id` is the idempotency key. Before any GPU work, step [2] does a boto3
`head_object` on `avatar-loras/<user_id>/<job_id>/lora.safetensors`. Hit →
re-emit the success webhook from the existing `manifest.json` and exit 0. This
makes RunPod's `maxRetries: 2` safe: a retry after a transient network failure
never re-trains a completed job. Artifacts upload in the order
`samples → manifest → lora.safetensors` so the LoRA blob is the commit marker
— a partial run can't masquerade as complete.

## 5. Cold-start budget (target < 30 s to first training step)

Assumes Network Volume warm (FLUX weights + ai-toolkit cached at
`/workspace/FLUX.1-dev/`).

| Stage | Budget |
|-------|--------|
| Container start + Python import | 8 s |
| SHA verify 4 files (mmap, ~24 GB read) | 10 s |
| Dataset download (~30 MB zip) | 2 s |
| Zip extract + write caption .txt | 1 s |
| train.py import + model load | 8 s |
| **Total** | **29 s** |

Cold-cold (no Network Volume): +120 s for FLUX weight pull from R2.

## 6. Observability

- Structured JSON logs to stdout: `{ts, level, job_id, stage, duration_ms, ...}`. RunPod captures stdout into its log UI.
- Stage markers: `request.parsed`, `idempotency.checked`, `sha.verified`, `dataset.downloaded`, `captions.written`, `train.started`, `train.finished`, `artifacts.uploaded`, `webhook.sent`.
- Exceptions → Sentry via `sentry_sdk` (DSN from endpoint env). Tag with `job_id`, `user_id`, `weights_version`.
- No PII in logs. `user_id` is a UUID, captions never logged in full (first 40 chars only).

## 7. Non-goals

- **Queueing / scheduling** — RunPod's queue owns it.
- **Multi-tenant within one worker** — one job per container; no shared state.
- **Inference** — separate service, separate ADR (future).
- **VLM captioning** — handler only consumes `captions{}`; production captioning runs in avatar-backend (ADR-0004).
- **Caching datasets across jobs** — `/tmp` wiped per invocation; dataset re-downloaded each retry.
- **Multi-GPU training** — single-GPU only; `batch_size` knob is per-device.
- **Model output diversity** — samples are debug artifacts, not a product surface.
