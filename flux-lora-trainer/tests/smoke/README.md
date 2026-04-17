# Smoke test — manual, pre-promotion

Run against the real RunPod endpoint after every push to `main` that builds a
new image tag. Costs ~$1 per run (one full A100 training job). **Never
automate.**

## Prereqs

- [ ] Endpoint deployed with new image: `$RUNPOD_ENDPOINT_ID`
- [ ] `runpodctl` authed (`runpodctl config --apiKey ...`)
- [ ] `awscli` configured for the R2 account (alias `aws --profile r2`)
- [ ] `SMOKE_HMAC_SECRET` matches the endpoint's `WEBHOOK_HMAC_SECRET`
- [ ] Dataset fixture ready: `tests/smoke/fixtures/smoke-15.zip` (15 images, all captioned)
- [ ] Webhook receiver running locally: `python tests/smoke/receiver.py` (prints each POST + signature verdict)

## Procedure

```bash
export JOB_ID=$(uuidgen | tr 'A-Z' 'a-z')
export USER_ID="00000000-0000-4000-8000-000000000001"   # fixed smoke user
export TS=$(date +%s)

# 1. Upload fixture
aws --profile r2 s3 cp tests/smoke/fixtures/smoke-15.zip \
    s3://avatar-datasets/datasets/${JOB_ID}.zip

# 2. Build request
cat > /tmp/smoke_request.json <<EOF
{
  "input": {
    "job_id":  "${JOB_ID}",
    "user_id": "${USER_ID}",
    "dataset_r2_key": "datasets/${JOB_ID}.zip",
    "trigger_word": "SMKTOK",
    "captions": $(cat tests/smoke/fixtures/smoke-captions.json),
    "webhook_url": "https://smoke-receiver.ngrok.io/hook",
    "config": {
      "steps": 500, "lora_rank": 16, "learning_rate": 0.001,
      "batch_size": 1, "resolution": "512,768,1024",
      "optimizer": "adamw8bit", "caption_dropout_rate": 0.05,
      "gpu_tier": "a100_80gb"
    }
  }
}
EOF

# 3. Submit
RUN_ID=$(curl -sS -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @/tmp/smoke_request.json | jq -r '.id')
echo "RUN_ID=$RUN_ID"
```

## Checklist (tick each before declaring the deploy green)

- [ ] **Request accepted** — RUN_ID returned, not an auth error
- [ ] **Poll to COMPLETED** — `curl ... /status/$RUN_ID` reaches status=COMPLETED within 35 minutes
- [ ] **Exit status succeeded** — `.output.status == "succeeded"`
- [ ] **LoRA exists in R2** — `aws s3 ls s3://avatar-loras/${USER_ID}/${JOB_ID}/lora.safetensors` returns a file ≥ 100 MB
- [ ] **Manifest version correct** — `aws s3 cp ... manifest.json - | jq .weights_version` returns `v1`
- [ ] **Sample images uploaded** — `aws s3 ls .../samples/ | wc -l` > 0
- [ ] **Webhook received** — receiver logs show one POST
- [ ] **Webhook signature validates** — receiver reports `signature: OK`
- [ ] **Idempotent retry** — re-submit same request body; handler returns immediately with `idempotent_replay=true`; no new files written (same mtimes)
- [ ] **Bad-SHA guard** — temporarily bump endpoint env `FLUX_WEIGHTS_VERSION=v2` (no such version); next run fails with exit_code=2; restore env
- [ ] **Logs are structured** — `runpodctl logs $RUN_ID | head` shows one-JSON-per-line with `job_id` + `stage` fields

## Cleanup

```bash
aws --profile r2 s3 rm s3://avatar-datasets/datasets/${JOB_ID}.zip
aws --profile r2 s3 rm s3://avatar-loras/${USER_ID}/${JOB_ID}/ --recursive
```

## If anything fails

Do not promote the image. File an issue with:
- `$RUN_ID`
- The full `runpodctl logs $RUN_ID` output
- The webhook receiver log (or "no webhook")
- The manifest.json if it exists
