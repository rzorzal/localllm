# Settings-Poison Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `cargo test` from corrupting the real settings file with the `r2/f2` fixture, via two layers: a handler guard that rejects invalid model refs, and isolation of the one unguarded settings-writing http test.

**Architecture:** Add a pure `is_valid_model_ref(repo, file)` used by `handle_admin_switch` to reject non-`owner/name` / non-`.gguf` specs with 400 before any `save_active_model`. Fix `tests/http.rs::admin_switch_accepts_with_token` to use a valid spec plus the existing `ENV_LOCK`/`IsolatedSettings` guard, and add a reject-path test.

**Tech Stack:** Rust, axum, `tokio::test` integration tests.

## Global Constraints

- `is_valid_model_ref(repo, file)` = `!repo.is_empty() && repo.contains('/') && is_safe_model_file(file) && file.to_ascii_lowercase().ends_with(".gguf")`. Reuses the existing `is_safe_model_file` (non-path filename).
- Handler rejects an invalid ref with `400` BEFORE `start_switch`/`save_active_model`.
- Test isolation reuses the existing `tests/http.rs` helpers `ENV_LOCK` (`tokio::sync::Mutex`) + `IsolatedSettings::new("<tag>")` — no new helper, no lib change.
- Acceptance: running the suite must not alter the real settings file's `active_model` (no `r2/f2`).
- No change to `save_active_model` or the settings-path resolution.

---

### Task 1: `is_valid_model_ref` helper + handler wiring

**Files:**
- Modify: `src/server.rs` (add pure fn near `is_safe_model_file` ~line 1600; use it in `handle_admin_switch` ~line 1623; add unit tests to the `#[cfg(test)] mod tests` in `src/server.rs`)

**Interfaces:**
- Consumes: existing `fn is_safe_model_file(file: &str) -> bool` (server.rs ~1600).
- Produces: `fn is_valid_model_ref(repo: &str, file: &str) -> bool`.

- [ ] **Step 1: Write the failing unit test**

Add to the `#[cfg(test)] mod tests` block in `src/server.rs`:

```rust
    #[test]
    fn is_valid_model_ref_accepts_real_and_rejects_fixtures() {
        assert!(super::is_valid_model_ref("owner/name", "model.gguf"));
        assert!(super::is_valid_model_ref("owner/name", "MODEL.GGUF")); // case-insensitive ext
        assert!(!super::is_valid_model_ref("r2", "f2")); // the poison fixture
        assert!(!super::is_valid_model_ref("owner/name", "f2")); // no .gguf
        assert!(!super::is_valid_model_ref("r2", "model.gguf")); // no owner/ slash
        assert!(!super::is_valid_model_ref("", "model.gguf")); // empty repo
        assert!(!super::is_valid_model_ref("owner/name", "../x.gguf")); // path (is_safe_model_file)
    }
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cargo test -p localllm is_valid_model_ref`
Expected: FAIL — `cannot find function is_valid_model_ref`.

- [ ] **Step 3: Implement the helper**

Add immediately after `is_safe_model_file` (server.rs ~line 1601) at module scope:

```rust
/// A model ref that could plausibly be a real GGUF model on Hugging Face: an
/// `owner/name` repo and a bare `*.gguf` filename. Rejects fixtures like
/// `r2`/`f2` that only break loading (and disable per-model features) if
/// persisted as the active model.
fn is_valid_model_ref(repo: &str, file: &str) -> bool {
    !repo.is_empty()
        && repo.contains('/')
        && is_safe_model_file(file)
        && file.to_ascii_lowercase().ends_with(".gguf")
}
```

- [ ] **Step 4: Wire it into `handle_admin_switch`**

In `handle_admin_switch` (server.rs ~1623), replace:

```rust
    if body.repo.is_empty() || !is_safe_model_file(&body.file) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo and a valid (non-path) file are required"})),
        )
            .into_response();
    }
```

with:

```rust
    if !is_valid_model_ref(&body.repo, &body.file) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "repo must be owner/name and file must be a .gguf filename"})),
        )
            .into_response();
    }
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cargo test -p localllm is_valid_model_ref`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): validate model ref (owner/name + .gguf) before switch/persist"
```

---

### Task 2: Isolate + fix the switch http tests

**Files:**
- Modify: `tests/http.rs` (`admin_switch_accepts_with_token` ~line 434; add `admin_switch_rejects_invalid_ref`)

**Interfaces:**
- Consumes: `is_valid_model_ref` behavior (Task 1) — invalid refs now 400. Existing `tests/http.rs` helpers `ENV_LOCK`, `IsolatedSettings`.

- [ ] **Step 1: Fix `admin_switch_accepts_with_token` (valid spec + isolation)**

In `tests/http.rs`, replace the whole `admin_switch_accepts_with_token` test with:

```rust
#[tokio::test]
async fn admin_switch_accepts_with_token() {
    let _guard = ENV_LOCK.lock().await;
    let _settings = IsolatedSettings::new("admin-switch-accepts");
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app,
        "/admin/model",
        r#"{"repo":"owner/name","file":"model.gguf"}"#,
        "x-admin-token",
        "test-token",
    )
    .await;
    assert_eq!(status, 202);
}
```

(The valid-format ref passes `is_valid_model_ref`; `start_switch` spawns the load
and the handler returns 202 after persisting to the ISOLATED temp settings — the
real config is untouched. The ref need not be a real model; the load happens
after the 202 and is isolated.)

- [ ] **Step 2: Add the reject-path test**

Immediately after `admin_switch_accepts_with_token`, add:

```rust
#[tokio::test]
async fn admin_switch_rejects_invalid_ref() {
    // The `r2/f2` fixture that used to poison the real settings must now be
    // rejected (no owner/ slash, no .gguf) BEFORE any persist.
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app,
        "/admin/model",
        r#"{"repo":"r2","file":"f2"}"#,
        "x-admin-token",
        "test-token",
    )
    .await;
    assert_eq!(status, 400);
}
```

(No isolation guard needed: the request is rejected before any settings write.)

- [ ] **Step 3: Build + full suite**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean build; all tests pass (the two switch tests, the `is_valid_model_ref` unit test, and everything else).

- [ ] **Step 4: Acceptance — real settings untouched**

Confirm the suite did not write the real config. Run:

```bash
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/Library/Application Support/localllm/settings.json")
am = json.load(open(p)).get("active_model")
print("active_model =", am)
assert am != {"repo": "r2", "file": "f2"}, "REAL SETTINGS POISONED"
print("OK: real settings not poisoned")
PY
```

Expected: prints the user's real Llama `active_model` and `OK: real settings not poisoned`.

- [ ] **Step 5: Commit**

```bash
git add tests/http.rs
git commit -m "test(http): isolate switch-accepts test; add invalid-ref reject test"
```

---

## Self-Review

**Spec coverage:**
- `is_valid_model_ref` pure helper with exact predicate → Task 1 Step 3 + unit test. ✓
- Handler returns 400 for invalid ref before persist → Task 1 Step 4. ✓
- Reuse existing `ENV_LOCK`/`IsolatedSettings`, no lib change → Task 2 Step 1. ✓
- `admin_switch_accepts_with_token` uses valid spec + isolation, still 202 → Task 2 Step 1. ✓
- New `admin_switch_rejects_invalid_ref` → 400 → Task 2 Step 2. ✓
- Acceptance: suite leaves real settings' `active_model` unchanged → Task 2 Step 4. ✓
- The only unguarded writer was `admin_switch_accepts_with_token` (the `/admin/model/ctx` writer already isolates; 401/400 tests return before write) → no other test needs wrapping. ✓

**Placeholder scan:** all code steps carry concrete code; the acceptance is a real check. No TBD/TODO.

**Type consistency:** `is_valid_model_ref(&str, &str) -> bool` defined in Task 1, its behavior consumed by Task 2's tests. `is_safe_model_file(&str) -> bool` is the existing helper. The handler edit matches the existing `handle_admin_switch` body. Test helpers `ENV_LOCK`, `IsolatedSettings`, `axum_test_request_status_with_header` all exist in `tests/http.rs`/lib.
