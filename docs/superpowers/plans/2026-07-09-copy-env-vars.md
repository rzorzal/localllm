# Copy Env Vars — Strip Launch/Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the tray's client-launch / CLI / terminal-chooser / route-apps machinery and replace it with one tray item that copies OS-appropriate inline env vars for pointing Claude Code / Codex at the local server.

**Architecture:** Add a small pure `env_snippet` module that formats the four env pairs (base URLs + timeouts) as a POSIX inline prefix or a PowerShell fallback, selected at compile time. Wire it to a new `📋 Copy env vars` tray item. Then delete `launch.rs`, `terminal.rs`, and `integrations/` along with every reference (tray items, server admin endpoints, settings fields, manager-UI panels, CLI dispatch), relocating the shared `atomic_write` helper to a surviving `fsutil` module.

**Tech Stack:** Rust, axum, tao/tray-icon, vanilla JS manager SPA.

## Global Constraints

- Client timeout is a fixed constant `TIMEOUT_MS = "1200000"` (20 min). Not configurable.
- The cold-prefill **gate** (`load_cold_prefill_gate` / `save_cold_prefill_gate`) is a separate routing knob and MUST be kept untouched.
- OS format is selected at compile time via `cfg!(target_os = "windows")` — no runtime detection.
- Combined single snippet contains all four vars; one tray item (not per-client).
- Tray menu emoji must render in the macOS status menu — use `📋` (not `⎘`).
- Each task ends compiling green (`cargo build`) and committed. Deleted modules stay declared until Task 7 so intermediate tasks compile.

---

### Task 1: `env_snippet` module

**Files:**
- Create: `src/env_snippet.rs`
- Modify: `src/lib.rs` (add module declaration near the other `pub mod` lines, ~line 19)
- Test: inline `#[cfg(test)] mod tests` in `src/env_snippet.rs`

**Interfaces:**
- Produces: `pub fn build(port: u16) -> String` — the OS-appropriate snippet.
- Internal (test-visible): `fn format_snippet(port: u16, windows: bool) -> String`.

- [ ] **Step 1: Write the failing tests**

Create `src/env_snippet.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::format_snippet;

    #[test]
    fn unix_is_inline_space_joined_no_export() {
        assert_eq!(
            format_snippet(31415, false),
            "ANTHROPIC_BASE_URL=http://127.0.0.1:31415 \
API_TIMEOUT_MS=1200000 \
CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000 \
OPENAI_BASE_URL=http://127.0.0.1:31415/v1"
        );
    }

    #[test]
    fn windows_is_powershell_env_assignments() {
        assert_eq!(
            format_snippet(31415, true),
            "$env:ANTHROPIC_BASE_URL=\"http://127.0.0.1:31415\"; \
$env:API_TIMEOUT_MS=\"1200000\"; \
$env:CLAUDE_STREAM_IDLE_TIMEOUT_MS=\"1200000\"; \
$env:OPENAI_BASE_URL=\"http://127.0.0.1:31415/v1\""
        );
    }

    #[test]
    fn port_is_interpolated_into_both_base_urls() {
        let out = format_snippet(9000, false);
        assert!(out.contains("ANTHROPIC_BASE_URL=http://127.0.0.1:9000 "));
        assert!(out.contains("OPENAI_BASE_URL=http://127.0.0.1:9000/v1"));
    }
}
```

Add the declaration to `src/lib.rs` (alphabetical-ish, next to `pub mod engine_llama;`):

```rust
pub mod env_snippet;
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib env_snippet`
Expected: FAIL — `cannot find function format_snippet in this scope`.

- [ ] **Step 3: Write minimal implementation**

Prepend to `src/env_snippet.rs` (above the test module):

```rust
//! Builds the inline env-var snippet the tray copies to the clipboard, so a
//! user can prepend it to their own `claude`/`codex` invocation to point the
//! client at the local server. Format matches the OS this binary runs on.

/// Client request timeout (ms). 20 min — covers a slow cold-prefill on the
/// local model so the client does not give up mid-request.
const TIMEOUT_MS: &str = "1200000";

/// The four (key, value) env pairs, port interpolated into the base URLs.
fn pairs(port: u16) -> [(&'static str, String); 4] {
    [
        ("ANTHROPIC_BASE_URL", format!("http://127.0.0.1:{port}")),
        ("API_TIMEOUT_MS", TIMEOUT_MS.to_string()),
        ("CLAUDE_STREAM_IDLE_TIMEOUT_MS", TIMEOUT_MS.to_string()),
        ("OPENAI_BASE_URL", format!("http://127.0.0.1:{port}/v1")),
    ]
}

/// Format the snippet for the given OS. `windows` selects the PowerShell
/// `$env:K="v";` form (space-joined); otherwise the POSIX inline `K=v` prefix
/// (space-joined, no `export`, no trailing command — the user appends it).
fn format_snippet(port: u16, windows: bool) -> String {
    let pairs = pairs(port);
    let parts: Vec<String> = if windows {
        pairs.iter().map(|(k, v)| format!("$env:{k}=\"{v}\";")).collect()
    } else {
        pairs.iter().map(|(k, v)| format!("{k}={v}")).collect()
    };
    parts.join(" ")
}

/// Build the snippet for the OS this binary was compiled for (the tray runs on
/// the user's own machine, so the compile-time target OS is the user's OS).
pub fn build(port: u16) -> String {
    format_snippet(port, cfg!(target_os = "windows"))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib env_snippet`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/env_snippet.rs src/lib.rs
git commit -m "feat(env-snippet): OS-formatted inline env-var snippet for clients"
```

---

### Task 2: Tray — add Copy env vars item, remove launch/wired

**Files:**
- Modify: `src/tray.rs`

**Interfaces:**
- Consumes: `crate::env_snippet::build(port)` (Task 1).

- [ ] **Step 1: Add the menu item in the `Init` block**

In `src/tray.rs`, after the `url_id = Some(url_line.id().clone());` line (~636), add a new item:

```rust
                // Copies the OS-appropriate inline env vars for pointing
                // Claude Code / Codex at this server (user prepends to their cmd).
                let copy_env_line = MenuItem::new("📋  Copy env vars", true, None);
                let copy_env_id_local = copy_env_line.id().clone();
```

Declare the id holder next to the other `let mut *_id` declarations (~line 543-546):

```rust
    let mut copy_env_id: Option<tray_icon::menu::MenuId> = None;
```

Assign it in the `Init` block right after creating the item:

```rust
                copy_env_id = Some(copy_env_id_local);
```

Append it to the menu immediately after `menu.append(&url_line).expect("append url");`:

```rust
                menu.append(&copy_env_line).expect("append copy-env");
```

- [ ] **Step 2: Handle the click**

In the `while let Ok(menu_event) = MenuEvent::receiver().try_recv()` loop, add a branch (e.g. after the `url_id` branch):

```rust
                    } else if copy_env_id.as_ref() == Some(&menu_event.id) {
                        let snippet = crate::env_snippet::build(port);
                        platform::copy_to_clipboard(&snippet);
                        tracing::info!("copied env-var snippet to clipboard");
```

- [ ] **Step 3: Remove the launch items and their handler**

Delete these declarations (~line 570-571 and 627-630):

```rust
    let mut launch_claude_id: Option<tray_icon::menu::MenuId> = None;
    let mut launch_codex_id: Option<tray_icon::menu::MenuId> = None;
```
```rust
                let launch_claude = MenuItem::new("🚀  Launch Claude Code via LocalLLM", true, None);
                let launch_codex = MenuItem::new("🚀  Launch Codex via LocalLLM", true, None);
                launch_claude_id = Some(launch_claude.id().clone());
                launch_codex_id = Some(launch_codex.id().clone());
```

Delete their appends (~line 667-668):

```rust
                menu.append(&launch_claude).expect("append launch claude");
                menu.append(&launch_codex).expect("append launch codex");
```

Delete the entire `else if launch_claude_id... || launch_codex_id...` handler block (the `osascript` folder-picker + `crate::terminal::open` block, ~lines 810-843).

- [ ] **Step 4: Remove the wired line + its live sync**

Delete `wired_handle` / `last_wired` declarations (~line 576-577):

```rust
    let mut wired_handle: Option<MenuItem> = None;
    let mut last_wired: Option<String> = None;
```

Delete the wired item creation + append (~line 643-645, 662):

```rust
                let init_state = crate::settings::load_integrations();
                let wired_line = MenuItem::new(wired_label(&init_state), false, None);
                wired_handle = Some(wired_line.clone());
```
```rust
                menu.append(&wired_line).expect("append wired line");
```

Also delete one of the two adjacent `separator` appends that bracketed the wired line (~line 660-664) — keep a single separator between the retry-cloud block and the logs block so the menu doesn't get a double divider.

Delete the per-tick wired sync block (~line 736-746):

```rust
                // Keep the read-only wired line in sync with the Config-page toggle.
                {
                    let st = crate::settings::load_integrations();
                    let label = wired_label(&st);
                    if last_wired.as_deref() != Some(label.as_str()) {
                        if let Some(line) = &wired_handle {
                            line.set_text(&label);
                        }
                        last_wired = Some(label);
                    }
                }
```

Delete the `wired_label` fn (~lines 402-421).

- [ ] **Step 5: Simplify the Quit handler**

Replace the Quit branch body (~lines 760-773) so it no longer unwires integrations:

```rust
                    if quit_id.as_ref() == Some(&menu_event.id) {
                        tracing::info!("quit requested via tray menu");
                        hard_exit(0);
                    } else if logs_id.as_ref() == Some(&menu_event.id) {
```

- [ ] **Step 6: Build**

Run: `cargo build`
Expected: compiles green (warnings about now-unused `crate::terminal` / `crate::integrations` are fine — removed in Task 7). No errors.

- [ ] **Step 7: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): Copy env vars item; drop launch/wired items"
```

---

### Task 3: `main.rs` — drop CLI dispatch + install_cli

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Remove the client dispatch**

Delete the block at the top of `main` (~lines 38-43):

```rust
    // `localllm claude|codex [args]` → run the client wired to the proxy and
    // exit; never start the server or emit logs.
    let raw: Vec<String> = std::env::args().collect();
    if let Some((client, rest)) = localllm::launch::detect(&raw) {
        localllm::launch::run(client, rest); // never returns
    }
```

- [ ] **Step 2: Remove the CLI symlink install**

Delete the block (~lines 84-86):

```rust
    // Keep the `localllm` CLI symlink fresh with this running binary (after
    // tracing init so its result is logged; after the claude/codex dispatch so
    // the wrapper itself doesn't reinstall).
    localllm::launch::install_cli();
```

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: compiles green (unused-module warnings fine).

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "refactor(cli): remove localllm claude|codex dispatch + symlink install"
```

---

### Task 4: `server.rs` — remove integrations + terminal endpoints

**Files:**
- Modify: `src/server.rs`
- Test: `tests/http.rs`

- [ ] **Step 1: Remove the failing integration tests first**

In `tests/http.rs`, delete the two tests `admin_integrations_get_requires_token` and `admin_integrations_get_reports_state_shape` (~lines 855-870, including their `#[tokio::test]` attributes).

- [ ] **Step 2: Remove the routes**

In `src/server.rs`, delete the two `.route(...)` registrations:

```rust
        .route(
            "/admin/integrations",
            get(handle_integrations_get).post(handle_integrations_set),
        )
```
```rust
        .route(
            "/admin/terminal",
            get(handle_terminal_get).post(handle_terminal_set),
        )
```

- [ ] **Step 3: Remove the handlers**

Delete the handler fns and their bodies: `handle_terminal_get`, `handle_terminal_set` (~lines 1173-1222), and `handle_integrations_get`, `handle_integrations_set`, and the `IntegrationsSetBody` struct (~lines 1312-1370).

- [ ] **Step 4: Build + test**

Run: `cargo build && cargo test --test http`
Expected: compiles green; http tests pass (the two removed tests are gone). Unused-import warnings for `get`/`json` only if they become unused elsewhere — remove any that the compiler flags as unused.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs tests/http.rs
git commit -m "refactor(server): drop /admin/integrations and /admin/terminal"
```

---

### Task 5: Manager UI — remove terminal + integration panels

**Files:**
- Modify: `src/manager_ui/app.js`

- [ ] **Step 1: Remove the mount calls**

In `src/manager_ui/app.js`, delete these two lines (~line 140 and 142):

```javascript
  renderTerminalConfig(shell);
```
```javascript
  renderIntegrationToggle(shell);
```

- [ ] **Step 2: Remove the panel definitions**

Delete the `TERMINAL_LABELS` const + `renderTerminalConfig` function (~lines 220-257) and the `renderIntegrationToggle` function (~lines 325-370).

- [ ] **Step 3: Verify no dangling references**

Run: `grep -nE "renderTerminalConfig|renderIntegrationToggle|admin/terminal|admin/integrations|TERMINAL_LABELS" src/manager_ui/app.js`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add src/manager_ui/app.js
git commit -m "refactor(ui): remove terminal chooser + route-apps toggle panels"
```

---

### Task 6: `settings.rs` — remove integration + terminal state

**Files:**
- Modify: `src/settings.rs`

- [ ] **Step 1: Remove the terminal accessors + tests**

Delete `load_terminal` and `save_terminal` (~lines 200-219) and the `terminal_round_trips_and_defaults_installed` test (~line 522 region, whole `#[test]` fn).

- [ ] **Step 2: Remove the integration accessors**

Delete `load_integrations` and `save_integrations` (~lines 282-291) and any integration-specific tests.

- [ ] **Step 3: Remove the struct + fields**

Delete the `IntegrationState` struct (~lines 14-17 region), the `integrations: IntegrationState` field (~line 54) and the `terminal: Option<String>` field (~lines 85-88) from the persisted-settings struct. Remove the `use crate::integrations::ClientPrior;` import (~line 11).

- [ ] **Step 4: Build + test**

Run: `cargo build && cargo test --lib settings`
Expected: compiles green; remaining settings tests pass. If the compiler flags any leftover reference to `IntegrationState` / `load_integrations` / `load_terminal`, that reference belongs to an earlier task — re-check Tasks 2/4 removed them.

- [ ] **Step 5: Commit**

```bash
git add src/settings.rs
git commit -m "refactor(settings): drop IntegrationState + terminal field/accessors"
```

---

### Task 7: Relocate `atomic_write`, delete dead modules

**Files:**
- Create: `src/fsutil.rs`
- Modify: `src/lib.rs`, `src/settings.rs`, `src/route_log.rs`
- Delete: `src/launch.rs`, `src/terminal.rs`, `src/integrations/mod.rs`, `src/integrations/claude_code.rs`, `src/integrations/codex.rs`

**Interfaces:**
- Produces: `pub fn atomic_write(path: &Path, contents: &[u8]) -> anyhow::Result<()>` in `crate::fsutil`.

- [ ] **Step 1: Create `src/fsutil.rs` with `atomic_write` + its tests**

Move the `atomic_write` fn verbatim from `src/integrations/mod.rs` (~lines 36-64) into a new `src/fsutil.rs`, plus its two unit tests `atomic_write_creates_file_and_leaves_no_temp` and `atomic_write_overwrites_existing` (~lines 147-173):

```rust
//! Small filesystem helpers shared across modules.

use std::path::Path;

/// Write `contents` to `path` atomically: a temp file in the same directory is
/// written, fsync-flushed, given the original file's permissions (when it
/// existed), then renamed over the target. No temp file remains on success.
pub fn atomic_write(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent dir: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".localllm-tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> anyhow::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents)?;
        f.sync_all()?;
        #[cfg(unix)]
        if let Ok(meta) = std::fs::metadata(path) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                &tmp,
                std::fs::Permissions::from_mode(meta.permissions().mode()),
            )?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_creates_file_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!("llm-aw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("config.json");
        atomic_write(&target, b"hello").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name() != "config.json")
            .collect();
        assert!(leftovers.is_empty(), "temp file left: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn atomic_write_overwrites_existing() {
        let dir = std::env::temp_dir().join(format!("llm-aw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("config.json");
        std::fs::write(&target, b"old").unwrap();
        atomic_write(&target, b"new").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        std::fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 2: Repoint callers to `crate::fsutil::atomic_write`**

In `src/settings.rs` change `crate::integrations::atomic_write(&path, json.as_bytes())?;` (~line 140) to `crate::fsutil::atomic_write(&path, json.as_bytes())?;`.

In `src/route_log.rs` change all four `crate::integrations::atomic_write(...)` calls (~lines 204, 217, 236, 247) to `crate::fsutil::atomic_write(...)`.

- [ ] **Step 3: Update `src/lib.rs` module declarations**

Add near the other decls:

```rust
pub mod fsutil;
```

Delete these three lines:

```rust
pub mod integrations;
pub mod launch;
pub mod terminal;
```

- [ ] **Step 4: Delete the dead files**

```bash
git rm src/launch.rs src/terminal.rs src/integrations/mod.rs src/integrations/claude_code.rs src/integrations/codex.rs
```

- [ ] **Step 5: Build + full test + dangling-reference sweep**

Run: `cargo build && cargo test`
Expected: compiles green; all tests pass.

Run: `grep -rnE "crate::(integrations|launch|terminal)\b|localllm::(integrations|launch|terminal)\b" src/ tests/`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: relocate atomic_write to fsutil; delete launch/terminal/integrations"
```

---

### Task 8: Verify in the running app

**Files:** none (manual verification)

- [ ] **Step 1: Rebuild the .app bundle**

Run: `scripts/build-app.sh`
Expected: builds the bundle (running app is the bundle, not `cargo build`).

- [ ] **Step 2: Launch + inspect the tray**

Launch the bundled app. Open the tray menu. Confirm:
- `📋  Copy env vars` appears directly under the URL line and its icon renders.
- No `Launch … via LocalLLM` items, no `Wired:` / `Apps:` line.

- [ ] **Step 3: Verify the clipboard payload**

Click `📋  Copy env vars`, then paste. Expected (macOS, port 31415):

```
ANTHROPIC_BASE_URL=http://127.0.0.1:31415 API_TIMEOUT_MS=1200000 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000 OPENAI_BASE_URL=http://127.0.0.1:31415/v1
```

- [ ] **Step 4: Verify the Config page**

Open Config. Confirm the "Terminal do Launch" and "Rotear apps pelo localllm" panels are gone; routing, threshold, cold-gate, and smart-history panels remain.

---

## Self-Review

**Spec coverage:**
- Remove launch.rs / CLI dispatch / install_cli → Tasks 3, 7. ✓
- Remove terminal.rs + chooser → Tasks 5, 6, 7. ✓
- Remove integrations (route apps) + tray wired + unwire-on-quit → Tasks 2, 4, 5, 6, 7. ✓
- Remove /admin/integrations + /admin/terminal → Task 4. ✓
- Keep profile + cold-prefill gate → untouched (asserted in Global Constraints; Task 6 removes only integration/terminal). ✓
- `atomic_write` relocation (settings + route_log callers) → Task 7. ✓
- New env_snippet (POSIX inline + PowerShell, port interp, fixed 20-min const) → Task 1. ✓
- New `📋 Copy env vars` tray item → Task 2. ✓
- Unit tests unix/windows/port → Task 1. ✓
- Verification (build, dangling sweep, .app rebuild) → Tasks 7, 8. ✓

**Placeholder scan:** none — all steps carry concrete code/commands.

**Type consistency:** `format_snippet(port, windows: bool)` / `build(port)` used consistently (Tasks 1, 2). `crate::fsutil::atomic_write` signature matches original and all repointed callers (Task 7). `copy_env_id: Option<MenuId>` matches the pattern of sibling `*_id` holders (Task 2).
