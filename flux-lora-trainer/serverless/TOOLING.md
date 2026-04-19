# Tooling state — what's installed, what still needs auth

Set up during the "install all" pass. Everything below is on this machine;
no further downloads are required to resume work.

## CLIs installed (all on PATH)

| Tool | Version | Path | Purpose |
|------|---------|------|---------|
| `brew` | 5.1.6 | `/opt/homebrew/bin/brew` | package manager |
| `gh` | 2.86.0 | `/opt/homebrew/bin/gh` | GitHub ops, GHCR push/pull, feeds github MCP token |
| `jq` | 1.7.1 | `/usr/bin/jq` | JSON munging |
| `yq` | 4.53.2 | `/opt/homebrew/bin/yq` | YAML munging |
| `pipx` | 1.11.1 | `/opt/homebrew/bin/pipx` | isolated python tool installs |
| `wrangler` | 4.83.0 | `~/.npm-global/bin/wrangler` | Cloudflare / R2 bucket + secret management |
| `runpodctl` | 2.1.9 | `~/.local/bin/runpodctl` | RunPod endpoint deploy + pod mgmt |
| `modal` | 1.3.3 | `~/Library/Python/3.14/bin/modal` | Modal (fallback platform) |
| `vastai` | 0.5.0 | `~/Library/Python/3.14/bin/vastai` | vast.ai rent/train/destroy (batch lane) |
| `hf` / `huggingface-cli` | latest | `/opt/homebrew/bin/` | FLUX weight downloads |
| `docker` | 29.3.1 | `/usr/local/bin/docker` | container builds. **Daemon not running** — start Docker Desktop when needed |
| `node` / `npm` | 24 / 11 | `/usr/local/bin/` | for wrangler and MCP packages |

`~/.npm-global` is configured as the npm prefix (no-sudo `npm install -g`).
Both `~/.npm-global/bin` and `~/.local/bin` are already on `PATH`.

## MCP servers (user scope — available in every project)

Registered via `claude mcp add -s user`. Listed here in dependency order.

| MCP | Transport | URL / command | Status | Auth needed |
|-----|-----------|--------------|--------|-------------|
| `github` | stdio | `~/.local/bin/github-mcp-wrapper` (wraps the official Go binary; pulls fresh `gh auth token` on each launch) | **✓ Connected** | none (uses existing `gh auth login` as `WayneDev20`) |
| `cloudflare-bindings` | SSE | `https://bindings.mcp.cloudflare.com/sse` | Needs OAuth | trigger any cloudflare-bindings tool → browser OAuth flow |
| `cloudflare-observability` | SSE | `https://observability.mcp.cloudflare.com/sse` | Needs OAuth | same |
| `sentry` | HTTP | `https://mcp.sentry.dev/mcp` | Needs OAuth | trigger any sentry tool → browser OAuth |

**To complete auth** (30 seconds per MCP):
- Next time you use a Cloudflare or Sentry tool, a browser tab opens. Log in
  once. Token is cached — you will not see it again.
- If a tool fails with "unauthenticated", run `claude mcp list` to see status.

**Postgres MCP**: deliberately **not** registered. Register it once avatar-backend
has a database. The command will be:
```bash
claude mcp add -s user -e AVATAR_BACKEND_DATABASE_URL=<url> \
    postgres-avatar -- npx -y @modelcontextprotocol/server-postgres \
    '$AVATAR_BACKEND_DATABASE_URL'
```

## Supporting files

- `~/.local/bin/github-mcp-wrapper` — shell wrapper that refreshes the GitHub
  token on every MCP launch. If `gh auth login` ever changes accounts, the MCP
  automatically picks up the new token — no re-registration needed.
- `~/.local/bin/github-mcp-server` — the Go binary the wrapper launches (v1.0.0).

## Rollback / reinstall

```bash
# Remove every MCP
for n in cloudflare-bindings cloudflare-observability sentry github; do
  claude mcp remove -s user "$n"
done

# Uninstall CLIs
brew uninstall yq pipx
npm uninstall -g wrangler
rm -f ~/.local/bin/runpodctl ~/.local/bin/github-mcp-server \
      ~/.local/bin/github-mcp-wrapper
```

## Known gaps (not blocking)

1. **Xcode Command Line Tools outdated** — prevented `brew install
   runpod/runpodctl/runpodctl` (worked around via direct binary download).
   Update from System Settings → Software Update when convenient; not required
   for this project.
2. **Docker daemon** — image isn't running. Start Docker Desktop before
   `docker build -f serverless/Dockerfile.runpod ...`
3. **RunPod + Modal accounts** — not checked. When ready:
   `runpodctl config --apiKey <key>` and `modal token new`.
4. **Cloudflare account** — when ready: `wrangler login`. R2 buckets get created
   via either `wrangler r2 bucket create` or the `cloudflare-bindings` MCP
   once authed.
