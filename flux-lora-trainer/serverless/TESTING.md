# Handler testing strategy

**Scope:** the RunPod handler (`serverless/handler.py`) and its helpers. Does
NOT cover `train.py` or ai-toolkit — those are exercised by the existing
vast.ai batch lane.

## 1. Test pyramid

```
     ┌──────────────┐  ~5%  Smoke (manual)     — 1 test, real A100
     │   Smoke      │
     ├──────────────┤  ~20% Integration         — moto S3, mocked subprocess
     │ Integration  │
     ├──────────────┤  ~75% Unit                — pure funcs, no I/O
     │    Unit      │
     └──────────────┘
```

Unit + integration run in CI on every PR (< 30 s wall clock). Smoke runs
manually before any RunPod endpoint promotion.

## 2. Unit tests (`tests/unit/`)

| Test target | Why |
|-------------|-----|
| `_validate_r2_key()` | Reject `://`, `..`, absolute `/`, empty, wrong prefix. Security-critical (ADR-0005) |
| `_parse_request()` | 8-knob schema; bounds on `steps` (1–10k), `lora_rank` (1–128), `learning_rate` (1e-6 – 1e-2) |
| `_sign_webhook(body, secret)` | Deterministic HMAC-SHA256; stable across reruns; rejects empty secret |
| `_verify_webhook(body, sig, secret)` | Round-trip with `_sign_webhook`; detect bit flips |
| `_verify_sha256(path, expected)` | Match + mismatch; missing file → FileNotFoundError not silently pass |
| `_check_captions_complete(zip_path, captions_dict)` | All-present ok; one-missing abort; `None` abort (captions required); extra-key ok (ignored) |
| `_build_train_argv(config, paths)` | Correct flags from `config{}`; no autocaption flags (LLaVA removed) |
| `_artifact_keys(user_id, job_id)` | Stable R2 path construction; no injection via user_id |
| `_manifest_body(...)` | Includes `weights_version`, `job_id`, `git_sha`, timing; valid JSON |
| `_classify_train_error(stderr)` | Maps "CUDA out of memory" → `oom=true`; unknown → generic |

Target: one happy path + 2–3 edge cases per function.

## 3. Integration tests (`tests/integration/`)

Use `moto` for S3 (R2 API-compatible) and `unittest.mock.patch` on
`subprocess.run` to fake `train.py`. Each test spins up moto fixtures, invokes
`handler(event)` directly.

| Scenario | Asserts |
|----------|---------|
| Happy path | Dataset downloaded, captions written, train.py called with correct argv, lora uploaded, webhook signed+sent |
| Idempotent retry | Pre-populate `lora.safetensors` in moto → handler exits 0, no train.py call, "already_complete" webhook |
| Bad SHA | Tamper fake weight file → exit 2, no dataset download |
| Partial captions | Zip has 15 images, captions has 14 → exit 3, no train.py call |
| Bad R2 key (`../etc/passwd`) | Exit 4, no boto3 call |
| train.py nonzero exit | Mock returncode=1, stderr="CUDA OOM" → exit 6, `oom=true` in webhook |
| R2 write fail | moto raises on put_object → 3 retries → exit 7 |
| Webhook fail | httpretty denies POST → retries → exit 0, logs warning (state in R2) |

## 4. Smoke test (manual, pre-promotion)

Run against the real RunPod endpoint on a fresh deploy. Checklist in
`tests/smoke/README.md`:

- [ ] Upload 15-image fixture zip to `avatar-datasets/smoke-<ts>.zip`
- [ ] `curl -X POST $RUNPOD_URL -d @smoke_request.json -H "auth: ..."`
- [ ] Poll `/status/<runId>` until complete (< 35 min A100)
- [ ] Verify `lora.safetensors` present at `avatar-loras/smoke/<job_id>/`
- [ ] Verify manifest `weights_version == "v1"`
- [ ] Verify webhook received at test endpoint, signature validates
- [ ] Run retry: same `job_id` → exits immediately, no new LoRA
- [ ] Cleanup: delete smoke R2 objects

## 5. CI/CD

| Trigger | What runs |
|---------|-----------|
| PR opened / push to branch | `ruff check`, `mypy --strict`, unit + integration (< 30 s) |
| Merge to `main` | Above + `docker build` (no push) to verify image still composes |
| Manual workflow (`deploy.yml`) | Build + push to GHCR, tag `v<n>`, then prompt for smoke run |
| Nightly | Re-run integration against latest `moto` to catch upstream S3 API drift |

Smoke is never auto-triggered (costs $ per run).

## 6. Fixtures (`tests/fixtures/`)

- `sample_dataset.zip` — 3 tiny (64×64) JPEGs, < 10 KB total
- `sample_captions.json` — matching filenames → short strings
- `fake_flux_weights/` — 4 empty files named as per manifest with pre-computed SHAs recorded in `tests/fixtures/fake_manifest.json`
- `webhook_secret.txt` — fixed string `test-secret-do-not-use-in-prod`
- `runpod_event.json` — canonical request for `handler(event)`

## 7. Coverage

- Target: **90% line coverage** on `serverless/*.py` excluding GPU paths.
- `.coveragerc` excludes: any module imported only when `torch.cuda.is_available()`, the `download_weights()` fallback paths, and `_classify_train_error` branches that require real CUDA stderr samples (covered by smoke).
- Ratcheted — PR that drops coverage below prior `main` fails CI.
