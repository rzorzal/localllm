# Stop http tests (and bogus switches) poisoning real settings

**Date:** 2026-07-09
**Status:** approved (design)
**Component:** `src/server.rs` (`handle_admin_switch` validation), `tests/http.rs` (isolation).

## Problem

The user's real settings file
(`~/Library/Application Support/localllm/settings.json`) repeatedly gets
`active_model` corrupted to the test fixture `{"repo":"r2","file":"f2"}`. This
breaks model loading (HF 401 on `r2/f2`) and silently disables per-model
features (e.g. smart-history: `r2/f2` has no `history_turns`, so
`resolve_history_turns` returns `None` and `apply_history_window` skips
selection entirely).

**Root cause (confirmed 2026-07-09):** `tests/http.rs::admin_switch_accepts_with_token`
POSTs `{"repo":"r2","file":"f2"}` to `/admin/model` with a valid admin token.
The handler `handle_admin_switch` calls `crate::settings::save_active_model(...)`
**synchronously** (before returning 202) and the test sets no `LOCALLLM_SETTINGS`
override, so the write lands on the real config. Every `cargo test` run
re-poisons. The isolation helpers `IsolatedSettings` + `ENV_LOCK` already exist
in `tests/http.rs` (used by the tool tests) — this test and other
settings-writing tests simply omit them. A few `POST /admin/model/ctx` tests
(which persist a per-model profile) are additional unguarded writers.

## Goal

`cargo test` must never modify the real settings file, and the switch endpoint
must not persist an obviously-invalid model ref that can only break loading.

## Non-goals

- No lib change to export a test-support helper — the isolation helper already
  lives in `tests/http.rs` and is reachable from the same crate.
- No validation of HF repo existence (cannot check synchronously).
- No change to how `save_active_model` resolves the settings path.

## Two defense layers

### 1. Handler validation (defense in depth)

Extract a pure helper in `src/server.rs`:

```rust
/// A model ref that could plausibly be a real GGUF model on Hugging Face:
/// an `owner/name` repo and a bare `*.gguf` filename. Rejects fixtures like
/// `r2`/`f2` that only break loading if persisted.
fn is_valid_model_ref(repo: &str, file: &str) -> bool {
    !repo.is_empty()
        && repo.contains('/')
        && is_safe_model_file(file)
        && file.to_ascii_lowercase().ends_with(".gguf")
}
```

Replace the inline check in `handle_admin_switch` (currently
`if body.repo.is_empty() || !is_safe_model_file(&body.file)`) with
`if !is_valid_model_ref(&body.repo, &body.file)` → return `400` with a message
naming the requirement (`owner/name` repo + `.gguf` file). This runs before
`start_switch`/`save_active_model`, so a bogus ref never persists.

**Assumption:** every real catalog model has an `owner/name` repo and a `.gguf`
file (true across the catalog). `is_safe_model_file` (existing, non-path
filename) is retained inside the helper.

### 2. Test isolation (root-cause fix)

Wrap every `tests/http.rs` test that hits a settings-writing endpoint without
isolation in the existing guard, at the top of the test:

```rust
let _guard = ENV_LOCK.lock().await;
let _settings = IsolatedSettings::new("<tag>");
```

`IsolatedSettings::new` points `LOCALLLM_SETTINGS` at a throwaway temp file
(removed on drop); `ENV_LOCK` (a `tokio::sync::Mutex`) serializes so the
process-global env var can't race parallel tests. Tests to wrap: the confirmed
`admin_switch_accepts_with_token`, plus each unguarded `POST /admin/model/ctx`
writer. GET-only tests and reject-path tests (`401`/`400`, which return before
any write) do not need the guard.

## Ripple on existing tests

- `admin_switch_accepts_with_token` currently sends `r2/f2` and asserts `202`.
  With layer 1, `r2/f2` now returns `400`. Change its body to a valid-looking
  ref (`owner/name` repo + a `*.gguf` file) AND add the isolation guard; it
  still asserts `202`.
- Add a new test `admin_switch_rejects_invalid_ref`: POST `{"repo":"r2","file":"f2"}`
  with a valid token → asserts `400`. No isolation needed (rejected before any
  write). This documents layer 1 and pins the fixture that used to poison.
- `admin_switch_rejects_bad_token` (401) and `admin_switch_bad_body_is_400`
  (malformed JSON) are unaffected.

## Testing

- Unit tests for `is_valid_model_ref`: `("owner/name","model.gguf") → true`;
  `("r2","f2") → false`; `("owner/name","f2") → false` (no `.gguf`);
  `("r2","model.gguf") → false` (no slash); `("owner/name","../x.gguf") → false`
  (path, via `is_safe_model_file`); `("owner/name","MODEL.GGUF") → true`
  (case-insensitive).
- Acceptance: after the change, run the whole suite and confirm the real
  settings file's `active_model` is unchanged (no `r2/f2`). Practically: the
  isolation guard means the writing tests touch only their temp file.

## Rollout

Backend + test-only. No bundle rebuild strictly required for the fix to hold
(it is test/handler logic), but rebuild for consistency. The user's live config
was already re-repaired 2026-07-09 (backup `settings.json.bak-r2f2-*`).
