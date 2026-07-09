# Design: Strip launch/routing plumbing, add "Copy env vars" tray item

Date: 2026-07-09
Status: approved (design), pending spec review

## Motivation

The tray's client-launch machinery never earned its complexity: launching
Claude Code / Codex from the tray, the right-click folder picker, the
`localllm claude|codex` CLI wrapper (frequent launch errors), the terminal-app
chooser, and the "route apps through localllm" config injection (the `Wired:`
line). All of it is being removed.

Replacement is one small, reliable primitive: a tray item that copies the
environment variables a user needs to point Claude Code / Codex at the local
server, formatted for the running OS. The user prepends the copied string to
their own `claude` / `codex` invocation. No process launching, no config file
mutation, no CLI symlink.

## Part 1 — Removals

### Files deleted
- `src/launch.rs` — `localllm claude|codex` dispatch, `env_for`, `install_cli`.
- `src/terminal.rs` — terminal-app abstraction (Terminal/iTerm/Warp/Wave).
- `src/integrations/mod.rs`, `src/integrations/claude_code.rs`,
  `src/integrations/codex.rs` — client-config injection ("route apps").

### `src/main.rs`
- Remove the `launch::detect` / `launch::run` dispatch at the top of `main`
  (currently lines 40–43). `localllm claude|codex` no longer exists.
- Remove the `launch::install_cli()` call (currently line 86). No CLI symlink.

### `src/tray.rs`
- Remove menu items `launch_claude`, `launch_codex` and their ids/handlers,
  including the `osascript` folder-picker + `terminal::open` block.
- Remove `wired_label`, the `wired_line` menu item, `wired_handle`/`last_wired`,
  and the per-tick `load_integrations` sync that updates it.
- Quit handler: drop the integration-unwiring block; Quit becomes a plain
  `hard_exit(0)`.
- (Part 2 adds the new copy-env item here.)

### `src/server.rs`
- Remove routes and handlers for `/admin/integrations`
  (`handle_integrations_get` / `handle_integrations_set`, `IntegrationsSetBody`)
  and `/admin/terminal` (`handle_terminal_get` / `handle_terminal_set`).

### `src/settings.rs`
- Remove `IntegrationState`, the `terminal` field, and the functions
  `load_integrations`, `save_integrations`, `load_terminal`, `save_terminal`,
  plus the now-unused `ClientPrior` import and related tests.
- KEEP: routing profile and the cold-prefill gate
  (`load_cold_prefill_gate` / `save_cold_prefill_gate`). These are unrelated
  routing tunables and stay in config.

### `src/manager_ui/` (app.js + index.html)
- Remove the integrations toggle UI and the terminal-chooser UI that call
  `/admin/integrations` and `/admin/terminal`.

### Relocation catch — `atomic_write`
`atomic_write` currently lives in `src/integrations/mod.rs` but is used by
`src/settings.rs` (persisting settings atomically). It MUST survive the deletion
of `integrations/`. Move it to a small surviving module (`src/fsutil.rs`) and
update `settings.rs` to call `crate::fsutil::atomic_write`. Keep its two unit
tests with it. The `uuid` dependency stays (used by `atomic_write`).

## Part 2 — New: Copy env vars

### `src/env_snippet.rs` (new, pure/testable)
- `pub fn build(port: u16) -> String`.
- OS selection is compile-time via `cfg!(target_os = "windows")`. The tray
  binary runs on the user's machine, so the compiled target OS *is* the user's
  OS — no runtime detection needed.
- Variables emitted (all four, one combined blob):
  - `ANTHROPIC_BASE_URL=http://127.0.0.1:{port}`
  - `API_TIMEOUT_MS=1200000`
  - `CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000`
  - `OPENAI_BASE_URL=http://127.0.0.1:{port}/v1`
- `const TIMEOUT_MS: &str = "1200000";` (20 min; covers slow cold prefill)
  moves here from the deleted `launch.rs`. Fixed constant — the cold-prefill
  *gate* in config is a separate routing knob and is not involved.
- Unix/macOS format — inline shell prefix, space-joined, no `export`, no
  trailing command (the user appends `claude` or `codex`):
  ```
  ANTHROPIC_BASE_URL=http://127.0.0.1:31415 API_TIMEOUT_MS=1200000 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000 OPENAI_BASE_URL=http://127.0.0.1:31415/v1
  ```
- Windows format — PowerShell fallback (inline env-prefix is not possible in
  cmd/PowerShell), semicolon-joined `$env:` assignments:
  ```
  $env:ANTHROPIC_BASE_URL="http://127.0.0.1:31415"; $env:API_TIMEOUT_MS="1200000"; $env:CLAUDE_STREAM_IDLE_TIMEOUT_MS="1200000"; $env:OPENAI_BASE_URL="http://127.0.0.1:31415/v1"
  ```

### `src/tray.rs` — new menu item
- One item labelled `📋  Copy env vars`, clickable, placed directly under the
  existing URL-copy line. (Emoji chosen because `⎘` did not render in the macOS
  status menu; `📋` matches the existing emoji style — 🗎 / ⚙ / ↻.)
- Click handler: `platform::copy_to_clipboard(&crate::env_snippet::build(port))`
  and a `tracing::info!` log line. `port` is already in scope in `run_tray`.

### Tests
- `env_snippet::build` unit tests:
  - unix build: contains all four `K=v` pairs, space-joined, no `export`, port
    interpolated into both base URLs.
  - windows build (via a testable inner fn taking an `is_windows: bool`, so both
    branches are testable on any host): `$env:K="v"` form, semicolon-joined.
  - port interpolation reflected in `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`.

## Non-goals / YAGNI
- No process launching, no terminal integration, no CLI symlink, no config-file
  injection into client apps.
- No runtime OS detection (compile-time `cfg!` is exact for a local tray app).
- No configurable client timeout — fixed 20-minute constant.

## Verification
- `cargo build` and `cargo test` green after removals + addition.
- No dangling references to deleted modules (`launch`, `terminal`,
  `integrations`) anywhere in `src/` or `tests/`.
- Rebuild the `.app` bundle (`scripts/build-app.sh`) and confirm the tray shows
  `📋  Copy env vars`, clicking copies the correct inline blob, and the old
  launch/wired items are gone.
