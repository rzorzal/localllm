# Phase C — Tray Profile Selector + Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pick the routing profile from a menu-bar "Routing" submenu (live, no restart) and persist the choice across launches, with a `--profile` CLI override.

**Architecture:** Add serde + clap derives to `route::Profile` and a small `settings` module that loads/saves the chosen profile to `<config-dir>/localllm/settings.json`. Resolve the startup profile (CLI override → saved setting → default) when building the shared `Arc<RwLock<RoutingPolicy>>`. Create that shared policy in `run_tray` so both the server thread and the menu hold it; the tray's CheckMenuItems write the policy live and persist the choice.

**Tech Stack:** Rust, clap 4, serde/serde_json, dirs; macOS tray via tray-icon 0.24 (muda menu items: `Submenu`, `CheckMenuItem`).

## Global Constraints

- Builds on Phases A+B (branch `feat/model-router`): `route::{Profile, RoutingPolicy}` with `Profile::policy()`; `AppState.policy: Arc<std::sync::RwLock<RoutingPolicy>>`; `server::router(gen, model_id, policy, local_ctx_window)`; `lib::run_server_with_ready(cfg, ready)` builds the policy internally; `tray::macos::run_tray(cfg)`.
- Default profile is **SaveTokens** (`Profile::default()`), used when no CLI flag and no saved setting.
- Profile precedence: **`--profile` CLI flag > saved settings.json > default**. The CLI flag is a per-run override and is NOT persisted; only tray clicks persist.
- Settings file: `<config-dir>/localllm/settings.json` via `dirs::config_dir()`; overridable for tests via the `LOCALLLM_SETTINGS` env var (full file path).
- Live update: a tray click changes routing for the NEXT request without restart (write through the shared `RwLock`).
- The tray module is macOS-only (`#[cfg(target_os = "macos")]`); the event-loop closure runs on the main thread, so `Rc`-based menu items (CheckMenuItem) live inside it (never sent across threads). The shared policy `Arc<RwLock<..>>` is `Send + Sync` and is shared with the server thread.
- TDD for the testable units (settings round-trip, profile resolution, CLI parse, the pure policy-apply helper). The macOS event-loop glue is verified by compile + a manual smoke note.
- Commit per task when green; test output pristine.

---

### Task 1: `Profile` serde/clap derives + `settings` module

**Files:**
- Modify: `src/route/policy.rs` (add derives to `Profile`)
- Create: `src/settings.rs`
- Modify: `src/lib.rs` (add `pub mod settings;`)
- Test: `src/settings.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces:
  - `Profile` additionally derives `serde::Serialize, serde::Deserialize, clap::ValueEnum` (kebab-case names: `save-tokens`, `balanced`, `max-quality`, `local-only`).
  - `settings::settings_path() -> Option<std::path::PathBuf>`
  - `settings::load_profile() -> crate::route::Profile`
  - `settings::save_profile(p: crate::route::Profile) -> anyhow::Result<()>`

- [ ] **Step 1: Add derives to `Profile`**

In `src/route/policy.rs`, change the `Profile` derive/attribute lines from:

```rust
/// User-facing routing choice. Set from the tray in a later phase.
/// Defaults to `SaveTokens` (the token-thrift, local-first profile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
```

to:

```rust
/// User-facing routing choice. Selected from the tray; persisted in settings.
/// Defaults to `SaveTokens` (the token-thrift, local-first profile).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default,
    serde::Serialize, serde::Deserialize, clap::ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
```

(The `#[default]` attribute already on the `SaveTokens` variant stays.)

- [ ] **Step 2: Write the failing settings tests**

Create `src/settings.rs`:

```rust
//! Persisted user settings (currently just the routing profile).
//!
//! Stored as JSON at `<config-dir>/localllm/settings.json` (e.g.
//! `~/Library/Application Support/localllm/settings.json` on macOS). The path
//! is overridable via the `LOCALLLM_SETTINGS` env var (full file path), used by
//! tests and power users.

use std::path::PathBuf;

use crate::route::Profile;

/// On-disk settings shape.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Settings {
    profile: Profile,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_settings<F: FnOnce()>(f: F) {
        let dir = std::env::temp_dir().join(format!("localllm-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::env::set_var("LOCALLLM_SETTINGS", &path);
        f();
        std::env::remove_var("LOCALLLM_SETTINGS");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_then_load_round_trips() {
        with_temp_settings(|| {
            save_profile(Profile::Balanced).unwrap();
            assert_eq!(load_profile(), Profile::Balanced);
        });
    }

    #[test]
    fn load_missing_file_returns_default() {
        with_temp_settings(|| {
            // no save → file does not exist
            assert_eq!(load_profile(), Profile::default());
        });
    }

    #[test]
    fn load_corrupt_file_returns_default() {
        with_temp_settings(|| {
            std::fs::write(settings_path().unwrap(), b"not json {{{").unwrap();
            assert_eq!(load_profile(), Profile::default());
        });
    }

    #[test]
    fn save_overwrites_previous() {
        with_temp_settings(|| {
            save_profile(Profile::MaxQuality).unwrap();
            save_profile(Profile::LocalOnly).unwrap();
            assert_eq!(load_profile(), Profile::LocalOnly);
        });
    }
}
```

In `src/lib.rs`, add after `pub mod route;`:

```rust
pub mod settings;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib settings:: 2>&1 | head -25`
Expected: FAIL — `settings_path`/`load_profile`/`save_profile` not defined.

- [ ] **Step 4: Implement the settings functions**

In `src/settings.rs`, add above the `#[cfg(test)]` block:

```rust
/// Resolve the settings file path. `LOCALLLM_SETTINGS` (full file path) wins;
/// otherwise `<config-dir>/localllm/settings.json`. Returns `None` only if no
/// config directory can be determined and no override is set.
pub fn settings_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LOCALLLM_SETTINGS") {
        return Some(PathBuf::from(p));
    }
    dirs::config_dir().map(|d| d.join("localllm").join("settings.json"))
}

/// Load the saved routing profile, or the default if the file is absent,
/// unreadable, or malformed. Never fails — a bad settings file must not stop
/// the server from starting.
pub fn load_profile() -> Profile {
    let Some(path) = settings_path() else {
        return Profile::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Profile::default();
    };
    match serde_json::from_str::<Settings>(&text) {
        Ok(s) => s.profile,
        Err(_) => Profile::default(),
    }
}

/// Persist the chosen routing profile, creating the parent directory if needed.
pub fn save_profile(p: Profile) -> anyhow::Result<()> {
    let path = settings_path()
        .ok_or_else(|| anyhow::anyhow!("no settings path (no config dir)"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&Settings { profile: p })?;
    std::fs::write(&path, json)?;
    Ok(())
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib settings:: 2>&1 | tail -15`
Expected: PASS (4 tests).

- [ ] **Step 6: Confirm the full suite + clippy are clean**

Run: `cargo test 2>&1 | tail -8 && cargo clippy --lib 2>&1 | grep -E "src/settings.rs|src/route/policy.rs" | grep -- "-->" || echo "no clippy in changed files"`
Expected: full suite green; no clippy warnings in the changed files.

- [ ] **Step 7: Commit**

```bash
git add src/route/policy.rs src/settings.rs src/lib.rs
git commit -m "feat(settings): persist routing profile + serde/clap derives on Profile"
```

---

### Task 2: `--profile` CLI flag + profile resolution into the shared policy

**Files:**
- Modify: `src/config.rs` (add `profile` field + a parse test)
- Modify: `src/settings.rs` (add `resolve_profile`)
- Modify: `src/lib.rs` (`run_server_with_ready` resolves the profile; add `run_server_with_ready_and_policy`)
- Test: `src/config.rs` (`#[cfg(test)]`), `src/settings.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `settings::{load_profile}`, `route::Profile`, `RoutingPolicy`.
- Produces:
  - `Config.profile: Option<crate::route::Profile>` (clap `--profile`)
  - `settings::resolve_profile(cli: Option<crate::route::Profile>) -> crate::route::Profile`
  - `lib::run_server_with_ready_and_policy(cfg, ready, policy: Arc<RwLock<RoutingPolicy>>)` (shared-policy entry point)

- [ ] **Step 1: Add the `--profile` flag with a failing parse test**

In `src/config.rs`, add this field to the `Config` struct (after the existing `no_kv_persist` field, before the closing brace):

```rust
    /// Routing profile (local↔cloud heuristic). Overrides the saved setting for
    /// this run only; tray selections persist, this flag does not. When unset,
    /// the saved setting (or SaveTokens default) is used.
    #[arg(long, value_enum)]
    pub profile: Option<crate::route::Profile>,
```

Add this test to the `#[cfg(test)] mod tests` block in `src/config.rs`:

```rust
    #[test]
    fn profile_flag_parses_and_defaults_none() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.profile, None);
        let c = Config::parse_from(["localllm", "--profile", "balanced"]);
        assert_eq!(c.profile, Some(crate::route::Profile::Balanced));
        let c = Config::parse_from(["localllm", "--profile", "local-only"]);
        assert_eq!(c.profile, Some(crate::route::Profile::LocalOnly));
    }
```

- [ ] **Step 2: Run the parse test to verify it fails**

Run: `cargo test --lib config::tests::profile_flag 2>&1 | head -20`
Expected: FAIL — `Config` has no `profile` field yet (won't compile).

- [ ] **Step 3: (field added in Step 1) Add `resolve_profile` with a failing test**

In `src/settings.rs`, add this test to the `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn resolve_prefers_cli_then_saved_then_default() {
        with_temp_settings(|| {
            // nothing saved, no CLI → default
            assert_eq!(resolve_profile(None), Profile::default());
            // saved setting, no CLI → saved
            save_profile(Profile::MaxQuality).unwrap();
            assert_eq!(resolve_profile(None), Profile::MaxQuality);
            // CLI overrides saved
            assert_eq!(resolve_profile(Some(Profile::LocalOnly)), Profile::LocalOnly);
        });
    }
```

- [ ] **Step 4: Run to verify it fails**

Run: `cargo test --lib settings::tests::resolve 2>&1 | head -15`
Expected: FAIL — `resolve_profile` undefined.

- [ ] **Step 5: Implement `resolve_profile`**

In `src/settings.rs`, add (above the test module, after `save_profile`):

```rust
/// Resolve the effective startup profile: an explicit CLI choice wins; else the
/// saved setting (or the default if none/unreadable).
pub fn resolve_profile(cli: Option<Profile>) -> Profile {
    cli.unwrap_or_else(load_profile)
}
```

- [ ] **Step 6: Build the policy from the resolved profile and add the shared-policy entry point**

In `src/lib.rs`, replace the body of `run_server_with_ready` so it resolves the profile and delegates to a new shared-policy function. Find the current function:

```rust
pub async fn run_server_with_ready(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<()> {
```

REPLACE its entire body (everything between the opening `{` and the matching closing `}` of that function) with:

```rust
    use std::sync::{Arc, RwLock};
    let profile = crate::settings::resolve_profile(cfg.profile);
    tracing::info!("routing profile: {profile:?}");
    let policy = Arc::new(RwLock::new(profile.policy()));
    run_server_with_ready_and_policy(cfg, ready, policy).await
}

/// Like [`run_server_with_ready`] but takes an externally-owned routing policy
/// so a caller (the tray) can share the same `Arc<RwLock<RoutingPolicy>>` and
/// mutate it live while the server reads it per request.
pub async fn run_server_with_ready_and_policy(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    policy: std::sync::Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
) -> anyhow::Result<()> {
```

Then, in what is now `run_server_with_ready_and_policy`, find and DELETE the old local policy construction (the lines that built the default policy in Phase A):

```rust
    let policy = std::sync::Arc::new(std::sync::RwLock::new(
        crate::route::Profile::default().policy(),
    ));
```

(The `router(engine, cfg.model_id.clone(), policy, cfg.ctx_len)` call right after it now uses the passed-in `policy` parameter — leave that call unchanged.)

> Note: the rest of the original body (engine load, bind, serve) stays exactly as-is inside `run_server_with_ready_and_policy`.

- [ ] **Step 7: Run config + settings + full suite**

Run: `cargo test --lib config:: 2>&1 | tail -10 && cargo test --lib settings:: 2>&1 | tail -10`
Expected: PASS — `profile_flag_parses_and_defaults_none` and `resolve_prefers_cli_then_saved_then_default` green.

Run: `cargo test 2>&1 | tail -8`
Expected: full suite green (the headless server path now logs the resolved profile; integration tests use `router_for_test*` and are unaffected).

- [ ] **Step 8: Commit**

```bash
git add src/config.rs src/settings.rs src/lib.rs
git commit -m "feat(config): --profile flag + resolve profile (cli>saved>default) into shared policy"
```

---

### Task 3: Tray "Routing" submenu (live switch + persist)

**Files:**
- Modify: `src/tray.rs` (build the shared policy in `run_tray`, add the Routing submenu + click handling)
- Modify: `src/route/policy.rs` (add the pure `apply_profile` helper + label, with tests)
- Test: `src/route/policy.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `settings::{resolve_profile, save_profile}`, `route::{Profile, RoutingPolicy}`, `lib::run_server_with_ready_and_policy`.
- Produces:
  - `Profile::label(&self) -> &'static str` (display text for menu/status)
  - `Profile::ALL: [Profile; 4]` (iteration order for the menu)
  - `route::policy::apply_profile(lock: &std::sync::RwLock<RoutingPolicy>, p: Profile)` (writes the policy under the lock)

- [ ] **Step 1: Add the pure helpers with failing tests**

In `src/route/policy.rs`, add to the `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn all_lists_every_profile_in_menu_order() {
        assert_eq!(
            Profile::ALL,
            [Profile::SaveTokens, Profile::Balanced, Profile::MaxQuality, Profile::LocalOnly]
        );
    }

    #[test]
    fn labels_are_human_readable() {
        assert_eq!(Profile::SaveTokens.label(), "Save tokens (local-first)");
        assert_eq!(Profile::LocalOnly.label(), "Local only (offline)");
    }

    #[test]
    fn apply_profile_updates_policy_live() {
        let lock = std::sync::RwLock::new(Profile::SaveTokens.policy());
        apply_profile(&lock, Profile::MaxQuality);
        assert_eq!(*lock.read().unwrap(), Profile::MaxQuality.policy());
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib route::policy 2>&1 | head -20`
Expected: FAIL — `Profile::ALL`, `Profile::label`, `apply_profile` undefined.

- [ ] **Step 3: Implement the helpers**

In `src/route/policy.rs`, add to the `impl Profile` block:

```rust
    /// All profiles in the order they appear in the tray menu.
    pub const ALL: [Profile; 4] = [
        Profile::SaveTokens,
        Profile::Balanced,
        Profile::MaxQuality,
        Profile::LocalOnly,
    ];

    /// Human-readable menu/status label.
    pub fn label(&self) -> &'static str {
        match self {
            Profile::SaveTokens => "Save tokens (local-first)",
            Profile::Balanced => "Balanced (smart)",
            Profile::MaxQuality => "Max quality (cloud-first)",
            Profile::LocalOnly => "Local only (offline)",
        }
    }
```

And add this free function at the end of `src/route/policy.rs` (outside any `impl`, above the test module):

```rust
/// Write `p`'s knobs into the shared policy under the lock. Used by the tray to
/// switch routing live without a restart.
pub fn apply_profile(lock: &std::sync::RwLock<RoutingPolicy>, p: Profile) {
    *lock.write().unwrap() = p.policy();
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --lib route::policy 2>&1 | tail -12`
Expected: PASS.

- [ ] **Step 5: Build the shared policy in `run_tray` and pass it to the server thread**

In `src/tray.rs`, inside `run_tray`, the function currently spawns the server with `crate::run_server_with_ready(cfg, Some(ready_for_server))`. Change it to build the shared policy first and pass it through.

Add imports at the top of the `macos` module (near the other `use` lines):

```rust
    use tray_icon::menu::{CheckMenuItem, Submenu};
    use std::sync::{Arc, RwLock};
    use crate::route::{Profile, RoutingPolicy};
```

Just BEFORE the `// ---- Spawn server ... ----` block, add:

```rust
        // Resolve the startup profile (CLI > saved > default) and build the
        // policy the menu and server share. The menu mutates it live.
        let initial_profile = crate::settings::resolve_profile(cfg.profile);
        let policy: Arc<RwLock<RoutingPolicy>> =
            Arc::new(RwLock::new(initial_profile.policy()));
        let policy_for_server = policy.clone();
```

Then REPLACE the server-thread spawn closure call:

```rust
                if let Err(e) = rt.block_on(crate::run_server_with_ready(cfg, Some(ready_for_server))) {
```

with:

```rust
                if let Err(e) = rt.block_on(crate::run_server_with_ready_and_policy(cfg, Some(ready_for_server), policy_for_server)) {
```

- [ ] **Step 6: Add routing-menu state vars and build the submenu in `Init`**

In `src/tray.rs`, alongside the other closure-captured `mut` state (near `let mut status_handle: Option<MenuItem> = None;`), add:

```rust
        // Routing submenu state: (menu id, profile, check item) for each profile,
        // populated in Init and used to dispatch clicks + re-check on change.
        let mut routing_items: Vec<(tray_icon::menu::MenuId, Profile, CheckMenuItem)> = Vec::new();
        let mut routing_status: Option<MenuItem> = None;
```

In the `Event::NewEvents(StartCause::Init)` arm, AFTER the `backend_line` is created and BEFORE the final `separator`/`logs_item`/`quit_item` block, build the submenu:

```rust
                    // Routing profile selector (live; persists on click).
                    let routing_submenu = Submenu::new("Routing", true);
                    for p in Profile::ALL {
                        let item = CheckMenuItem::new(
                            p.label(),
                            true,
                            p == initial_profile,
                            None,
                        );
                        routing_items.push((item.id().clone(), p, item.clone()));
                        routing_submenu.append(&item).expect("append routing item");
                    }
                    let routing_line = MenuItem::new(
                        format!("Routing: {}", initial_profile.label()),
                        false,
                        None,
                    );
                    routing_status = Some(routing_line.clone());
```

Then add these to the menu append sequence (insert AFTER `menu.append(&backend_line)...` and before the existing `separator` append):

```rust
                    menu.append(&PredefinedMenuItem::separator()).expect("sep2");
                    menu.append(&routing_line).expect("append routing line");
                    menu.append(&routing_submenu).expect("append routing submenu");
```

> `initial_profile` is `Copy`, so using it in the Init closure body is fine even though the closure is `move`.

- [ ] **Step 7: Handle routing clicks in the menu-event poll**

In `src/tray.rs`, in the `while let Ok(menu_event) = MenuEvent::receiver().try_recv()` loop, add a branch BEFORE the closing `}` of the while loop (after the existing `url_id` branch):

```rust
                        } else if let Some((_, profile, _)) =
                            routing_items.iter().find(|(id, _, _)| id == &menu_event.id)
                        {
                            let chosen = *profile;
                            crate::route::policy::apply_profile(&policy, chosen);
                            if let Err(e) = crate::settings::save_profile(chosen) {
                                tracing::warn!("failed to persist routing profile: {e}");
                            }
                            // Re-check exactly the chosen item; update the status line.
                            for (_, p, item) in &routing_items {
                                item.set_checked(*p == chosen);
                            }
                            if let Some(s) = &routing_status {
                                s.set_text(format!("Routing: {}", chosen.label()));
                            }
                            tracing::info!("routing profile set via tray: {chosen:?}");
                        }
```

> The closure already owns `policy` (moved into `event_loop.run(move |...|)`); `routing_items` and `routing_status` are closure-captured `mut` state. No additional clones needed.

- [ ] **Step 8: Build the whole binary (macOS) and run the full suite**

Run: `cargo build 2>&1 | tail -20`
Expected: clean compile (tray module included) — no errors, no new warnings in `src/tray.rs`/`src/route/policy.rs`.

Run: `cargo test 2>&1 | tail -8`
Expected: full suite green (route::policy helper tests included).

Run: `cargo clippy --all-targets 2>&1 | grep -E "src/tray.rs|src/route/policy.rs|src/settings.rs" | grep -- "-->" || echo "no clippy in changed files"`
Expected: no clippy warnings in the changed files.

- [ ] **Step 9: Manual smoke (record result in the report)**

This step is a manual check — perform it if a macOS GUI session is available; otherwise note it as "deferred to user".

```bash
# Build the app bundle and launch the tray, then verify by hand:
cargo run -- --tray --port 31999
```
Verify: the menu shows a "Routing: Save tokens (local-first)" line and a "Routing" submenu with four items, the current one checked. Selecting another profile re-checks it, updates the status line, and writes `<config-dir>/localllm/settings.json`. Relaunch → the previously chosen profile is checked. Report what you observed (or that it was deferred).

- [ ] **Step 10: Commit**

```bash
git add src/tray.rs src/route/policy.rs
git commit -m "feat(tray): Routing profile submenu — live switch + persist"
```

---

## Phase C Acceptance

- The tray menu has a "Routing" submenu of four profiles (SaveTokens/Balanced/MaxQuality/LocalOnly) with the active one checked, plus a "Routing: <label>" status line.
- Selecting a profile changes routing for the next request without restart (shared `RwLock` write) and persists to `settings.json`; the checks and status line update.
- On launch, the profile is resolved as CLI `--profile` > saved settings > SaveTokens default; the server and tray share one policy.
- `settings` load/save round-trips and tolerates a missing/corrupt file (returns default); `resolve_profile` precedence is unit-tested; `--profile` parses; `apply_profile`/`label`/`ALL` are unit-tested.
- `cargo test`, `cargo build`, and `cargo clippy --all-targets` are clean (no new warnings in changed files).

## Self-Review

- **Spec coverage (Phase C scope):** tray submenu of 4 profiles + checks (Task 3), live policy write (Task 3 `apply_profile`), persistence (Task 1 `save_profile`/`load_profile`), `--profile` CLI + precedence (Task 2 `resolve_profile`), shared policy for tray+server (Task 2 `run_server_with_ready_and_policy`, Task 3 `run_tray`), status line (Task 3). Usage tracking + degrade alerts remain Phase D.
- **Placeholder scan:** none — every step shows complete code or an exact command.
- **Type consistency:** `Profile` (now serde+clap), `settings::{settings_path, load_profile, save_profile, resolve_profile}`, `Config.profile: Option<Profile>`, `run_server_with_ready_and_policy(cfg, ready, Arc<RwLock<RoutingPolicy>>)`, `route::policy::apply_profile(&RwLock<RoutingPolicy>, Profile)`, `Profile::{ALL, label}` are referenced consistently across tasks. `Profile::policy()` (Phase A) reused unchanged.
- **macOS-only note:** Task 3 edits live entirely in `#[cfg(target_os="macos")] mod macos`; the testable logic (`apply_profile`, `label`, `ALL`, settings, resolution) lives in cross-platform modules and carries the unit tests, since the event-loop glue cannot be driven in a unit test.
