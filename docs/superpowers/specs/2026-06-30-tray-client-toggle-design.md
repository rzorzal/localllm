# Tray toggle — route apps through localllm — design

**Date:** 2026-06-30
**Status:** approved (design)
**Sub-project:** A of 2. The user-facing half of the "auto-route clients through
localllm" feature. Depends on sub-project B (`/v1/responses`, already on this
branch) so the Codex path actually works. Builds on the **same branch**
(`feat/openai-responses-api`).

## Why

A non-technical user should be able to flip one switch in the tray and have
Claude Code and Codex automatically talk to the local server — no environment
variables, no manual config editing. Today that requires hand-editing
`~/.claude/settings.json` and `~/.codex/config.toml`. This sub-project injects
(and cleanly reverts) that config from a checkable tray item.

When **ON**, the clients point at `http://127.0.0.1:31415`; localllm's router
then decides local-vs-cloud per request (cloud reverse-proxies the client's own
key). When **OFF**, the clients talk directly to their provider exactly as
before.

## Scope

### In scope
- One checkable tray item: **"Route apps through localllm"**.
- Pluggable per-client config injection, two clients:
  - **Claude Code** — `~/.claude/settings.json`, JSON.
  - **Codex CLI** — `~/.codex/config.toml`, TOML.
- Surgical, reversible edits: record each key's **prior value** (absent vs
  present-with-value-V) on enable; on disable restore V or delete the key.
- Atomic writes (temp file + rename, mode preserved).
- Persist toggle state + recorded priors in localllm's own `settings.rs`;
  restore the checkbox on launch.
- Skip clients whose config dir/file is absent; report which were wired.

### Out of scope (YAGNI)
- **Claude Desktop** — its GUI has no custom-endpoint support (only transparent
  hosts + TLS interception); cannot be wired by config edit. Excluded.
- Any client beyond Claude Code + Codex (the trait makes adding more trivial
  later; we don't build them now).
- Auto-detecting a non-default port from a *running external* process — we use
  the server's own configured port (the tray and server share one process).
- Live config hot-reload of the *clients* — the user restarts their CLI after
  toggling (standard for these tools); we don't signal them.
- Mutating any key other than the ones listed under "Keys written".

## Keys written

### Claude Code (`~/.claude/settings.json`, JSON object)
Under the top-level `env` object:
- `ANTHROPIC_BASE_URL` = `http://127.0.0.1:<port>`
- `ENABLE_TOOL_SEARCH` = `"1"`

`ENABLE_TOOL_SEARCH` shrinks Claude Code's large tool payload (the ~37k-token
static tool prefix documented in the progress ledger), which is what makes the
local-model path practical on cold turns. Both keys are recorded for surgical
revert. The client's existing auth (e.g. `ANTHROPIC_API_KEY`) is untouched and
flows through to the cloud reverse-proxy.

### Codex (`~/.codex/config.toml`, TOML)
- Top-level `model_provider` = `"localllm"` (prior recorded).
- Add table `[model_providers.localllm]`:
  - `name = "localllm"`
  - `base_url = "http://127.0.0.1:<port>/v1"`
  - `wire_api = "responses"`
  - `env_key = "OPENAI_API_KEY"`
- `model` is **left untouched** — localllm ignores it on the local path and
  forwards it verbatim on the cloud path, so it must remain a real OpenAI model
  name for cloud fallback to work. `env_key` names the env var Codex reads the
  bearer token from; localllm forwards that key to `api.openai.com`.

On disable: restore prior `model_provider` (or delete if it was absent), and
remove the `[model_providers.localllm]` table (or restore its prior contents if
it pre-existed).

## Architecture

New module `src/integrations/`:
- `mod.rs` — the `ClientInjector` trait + orchestration (`enable_all` /
  `disable_all`) + the `Prior` type and atomic-write helper.
- `claude_code.rs` — `ClaudeCode` injector (serde_json).
- `codex.rs` — `Codex` injector (toml_edit, format-preserving).

```rust
trait ClientInjector {
    fn id(&self) -> &'static str;            // "claude-code" | "codex"
    fn display_name(&self) -> &'static str;  // "Claude Code" | "Codex"
    fn detect(&self) -> bool;                // config dir/file present
    fn enable(&self, port: u16) -> Result<Prior, IntegrationError>;
    fn disable(&self, prior: &Prior) -> Result<(), IntegrationError>;
}
```

`Prior` captures, per key the injector touches, whether it was Absent or
`Present(value)`. It is `Serialize`/`Deserialize` so it round-trips through
`settings.rs`. The injector for each client owns the exact shape of its `Prior`
(JSON values for Claude Code, TOML items for Codex), exposed through a small
serializable representation so `settings.rs` stays client-agnostic.

Config paths are **injectable** (constructor takes the base dir, defaulting to
the real home dir) so unit tests run against temp dirs with no real home access.

```
Tray checkbox click ─▶ integrations::enable_all(port) / disable_all()
                          │  for each injector with detect()==true
                          ▼
                   injector.enable(port) ─▶ Prior   (record prior, atomic write)
                          │
                          ▼
              settings.rs persists {enabled, priors per client}
                          │
                          ▼
              tray checkbox reflects settings.enabled (restored on launch)
```

## State & persistence

`settings.rs` gains an `integrations` section:
- `enabled: bool`
- `priors: Map<client_id, Prior>` — the recorded prior values, the **source of
  truth** for revert (not a file backup).

ON: for each detected client, `enable(port)` → store its `Prior` → set
`enabled = true` → persist. OFF: for each client with a recorded `Prior`,
`disable(prior)` → drop its prior → set `enabled = false` → persist.

Idempotence: ON only records a prior when `enabled == false` in our settings, so
a double-ON never captures our own already-written keys as the "prior".

## Tray integration

A single `CheckMenuItem` "Route apps through localllm", checked from
`settings.integrations.enabled` at build time. On click: flip, call
`enable_all`/`disable_all`, persist, re-check the item, and update a
non-clickable sub-line listing wired clients (e.g. `Wired: Claude Code, Codex`)
or `Wired: none (no client configs found)`. Follows the existing Routing-submenu
pattern in `tray.rs` (sync handler in the event loop, no lock across await).

## Error handling

- Each client edit is independent: one failing (malformed JSON/TOML,
  permissions) is logged and surfaced in the sub-line; the others still apply.
- Writes are **atomic**: serialize to a temp file in the same dir, preserve the
  original file mode, then rename over the target. A crash mid-write never
  corrupts the user's config and never leaves a stray temp file on the success
  path.
- If `enable` fails for a client, no prior is recorded for it (nothing to
  revert later).
- Malformed existing config → `Err(IntegrationError)`, never a panic.

## Testing

Pure unit tests per injector against temp dirs (no network, no real home):
- **Claude Code enable** writes exactly `ANTHROPIC_BASE_URL` +
  `ENABLE_TOOL_SEARCH` under `env`; returns `Prior` marking both Absent when the
  file/keys didn't exist.
- **enable with pre-existing value** records `Present(V)`; **disable** restores
  exactly `V`; a key that was Absent is deleted on disable.
- **Unrelated keys preserved** across enable+disable (e.g. a user's own
  `env.FOO` and other top-level settings untouched).
- **Codex enable** adds the `[model_providers.localllm]` table with the four
  fields and sets `model_provider`; leaves `model` and other tables/comments
  intact (toml_edit format-preserving); **disable** removes the table and
  restores prior `model_provider`.
- **Malformed input** (invalid JSON / invalid TOML) → `Err`, not panic.
- **Atomic write** leaves no temp file on success; mode preserved.
- **Missing config** → `detect()==false`, `enable_all` skips it.
- **Orchestration**: mixed detected/undetected clients → only detected ones get
  priors; idempotent double-enable does not overwrite recorded priors.
- **settings round-trip**: `integrations` block with priors serializes and
  reloads equal.

## Files touched
- `src/integrations/mod.rs` — **new** (trait, `Prior`, orchestration, atomic
  write, error type).
- `src/integrations/claude_code.rs` — **new** (JSON injector + tests).
- `src/integrations/codex.rs` — **new** (TOML injector + tests).
- `src/lib.rs` (or `src/main.rs` module list) — register `mod integrations;`.
- `src/settings.rs` — add the `integrations` section (enabled + priors) with
  load/save; round-trip test.
- `src/tray.rs` — the `CheckMenuItem` + click handler + wired-clients sub-line.
- `Cargo.toml` — add `toml_edit` dependency.

## Follow-up
None within this feature — sub-projects A + B together complete the
"auto-route clients through localllm" goal. macOS is the verified target;
Linux/Windows path handling falls out of `dirs`/home-dir resolution and is
compile-best-effort, consistent with the rest of the tray.
