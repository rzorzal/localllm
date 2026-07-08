# Client Launcher (sub-3): `localllm claude`/`codex` + tray + terminal picker

**Date:** 2026-07-08
**Status:** approved (design)
**Parent:** routing/timeout fix. sub-1 (cold-prefill routing) done; sub-2 (never-error fallback + abort) pending; sub-3 = this.

## Problem

To route a client through localllm the user must manually export env vars (README: `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL`) every session, and the default client timeouts abort on a slow cold prefill. There is no in-app way to launch a client already wired to the proxy with raised timeouts. The user wants: a CLI wrapper, a tray item, and a Config choice of terminal app.

## Goals

- One command `localllm claude [args]` / `localllm codex [args]` that runs the client in the current directory, wired to the proxy with raised timeouts — zero manual env.
- A tray item "🚀 Launch Claude Code via LocalLLM" (and Codex) that picks a folder and opens the user's chosen terminal there, running the wrapper.
- A Config page control to pick the terminal app, auto-detecting installed ones and defaulting to an installed choice.

## Non-goals

- KV pre-warming of the picked folder (user confirmed "launch-and-ready" only).
- Windows/Linux terminal launching in v1 (the tray is macOS; the CLI wrapper itself is cross-platform).
- Configurable timeout values in v1 (use a generous default; may expose later).

## Layer 1 — CLI wrapper (foundation)

`src/main.rs` currently does `MainArgs::parse()` (clap, with `Config` flattened). Before that parse, inspect `std::env::args()`: if the first positional arg is `claude` or `codex`, dispatch to a new `launch` module and never enter server/tray mode.

`launch::run(client, passthrough_args)`:
- Resolve the proxy port (default `31415`; honor a `--port`/`LOCALLLM_PORT` if trivially available, else the default constant shared with `Config`).
- Build env for the child:
  - **claude:** `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`, `API_TIMEOUT_MS=<default>`, `CLAUDE_STREAM_IDLE_TIMEOUT_MS=<default>` (the idle timeout is the one that actually fires on a silent cold prefill). Default = `1200000` (20 min) for both.
  - **codex:** `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` only (v1). Codex has no request-timeout env var; its `stream_idle_timeout_ms` lives in `config.toml` under a provider id we don't own here, so v1 does not touch it — claude is the primary target and carries the timeout fix.
- Preserve the caller's existing env (only add/override the keys above) and inherit the current working directory.
- Exec the client binary (`claude` / `codex`) with the passthrough args, replacing the current process (`std::os::unix::process::CommandExt::exec` on unix) so the tty, signals, and exit code pass through cleanly. On non-unix, spawn + wait + propagate status.
- If the client binary isn't on PATH, print a clear error ("claude not found on PATH — install Claude Code") and exit non-zero.

This layer is independently usable and unit-testable (env-map construction is a pure function `launch::env_for(client, port) -> Vec<(String,String)>`).

## Layer 2 — Terminal launcher (tray helper)

`src/tray.rs` (or a small `src/terminal.rs`): `open_terminal(app: TerminalApp, dir: &Path, command: &str)`.
- `TerminalApp` enum: `Terminal` (Apple), `ITerm`, `Warp`, `Wave`. Each knows its bundle id / app name and how to open a new window running a command in `dir`.
- macOS implementation per app:
  - **Apple Terminal / iTerm:** `osascript` telling the app to `do script "cd <dir> && <command>"` in a new window.
  - **Warp / Wave:** these don't expose the same `do script` AppleScript. Fallback: `open -a <App> <dir>` to open the app at the folder, then the user runs the wrapper — OR write the command to a temp shell script and `open -a <App> <script>` if supported. v1: for apps without scriptable `do script`, open the app at `dir` and rely on the wrapper being on PATH (document the limitation).
- `command` is the wrapper invocation, e.g. `localllm claude` (resolved to the running binary's absolute path so it works regardless of PATH).

`installed_terminals() -> Vec<TerminalApp>`: check `/Applications/<App>.app` and `~/Applications/<App>.app`; return those present. Default pick = first installed in preference order (Apple Terminal always exists on macOS, so there is always a default).

## Layer 3 — Config: terminal picker

- Persist `terminal: Option<TerminalApp>` in `settings.rs` (mirror the cold-prefill-gate persistence: `load_terminal()` default = first installed; `save_terminal()`).
- `GET/POST /admin/terminal` → `{ "terminal": "<id>", "available": ["terminal","iterm",...] }` (token-guarded via `check_admin`, mirroring `/admin/cold-prefill-gate`).
- Config page (`app.js`): `renderTerminalConfig(container)` — a segmented/select control listing `available` terminals, current selected, POST on change. Copy: "Terminal usado pelo botão Launch do tray."

## Tray items

Single top-level items (no submenu — consistent with the flattened Config menu):
- `🚀  Launch Claude Code via LocalLLM`
- `🚀  Launch Codex via LocalLLM`

On click: `osascript -e 'choose folder'` → POSIX path (user cancels → no-op). Then `open_terminal(load_terminal(), picked_dir, "<abs-binary> claude")`. Best-effort: a failure logs + optionally a `usage::notify`, never crashes the tray.

## Data flow

tray click → choose-folder (osascript) → open_terminal(chosen app, dir, `localllm claude`) → new terminal window runs the wrapper in `dir` → wrapper sets proxy env + execs `claude` → Claude Code talks to localllm with raised idle/total timeouts.

## Error handling

- Wrapper: missing client binary → clear message + non-zero exit. Bad port → still runs (client will fail to connect, its own error).
- Tray: folder-pick cancelled → no-op. Terminal-open failure → log + best-effort notify; tray stays alive.
- Config: unknown/absent terminal id → fall back to first installed.

## Testing

- `launch::env_for(client, port)` unit tests: claude yields the three keys with the right URL/timeouts; codex yields `OPENAI_BASE_URL`.
- Arg dispatch: a `claude`/`codex` first-positional routes to launch (unit-test the detector function, not the exec).
- `settings` round-trip for `terminal` + default = an installed terminal.
- `installed_terminals` is filesystem-dependent → test the pure preference-ordering/selection given a set of "present" apps (inject the presence check).
- Terminal `open`/osascript are side-effecting → not unit-tested; manual verification.

## Rollout

Backend + embedded frontend; rebuild `.app` bundle (`scripts/build-app.sh --fast`) + restart to test. The `localllm claude` CLI works from the built binary on PATH (or via absolute path).
