# Client Launcher Implementation Plan (sub-3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user launch Claude Code / Codex already wired to the localllm proxy with raised timeouts — via a `localllm claude`/`codex` CLI wrapper, a tray "Launch … via LocalLLM" item (folder-pick → chosen terminal), and a Config terminal-app picker.

**Architecture:** A new `launch` module builds the proxy env and execs the client; `main.rs` dispatches to it when the first CLI positional is `claude`/`codex` (before server/tray setup). A `terminal` module maps terminal apps to how they open a new window running a command in a directory, and detects installed ones. Settings persist the chosen terminal, exposed via `/admin/terminal` + a Config control. Tray items pick a folder (osascript) and open the chosen terminal running the wrapper.

**Tech Stack:** Rust (clap, std::process, unix exec, osascript via std::process::Command), vanilla JS/CSS.

## Global Constraints

- CLI wrapper dispatch happens at the TOP of `main()` BEFORE tracing init and `MainArgs::parse()`, so `localllm claude` produces no server logs and never enters server/tray mode.
- claude env: `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`, `API_TIMEOUT_MS=1200000`, `CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000` (the idle timeout is the one that fires on a silent cold prefill). codex env (v1): `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` only.
- Default proxy port `31415` (must match `Config`'s `--port` default).
- Preserve caller env; only add/override the keys above. Inherit cwd. Unix: replace the process via `exec`. Missing client binary → clear error + non-zero exit.
- Tray items are single top-level entries (no submenu — matches the flattened Config menu). Terminal launch + folder pick are best-effort; failure logs (+ optional `usage::notify`) and never crashes the tray.
- New admin endpoint token-guarded via `check_admin`, mirroring `/admin/cold-prefill-gate`.
- macOS is the only supported tray/terminal-launch platform; the CLI wrapper is cross-platform (unix exec, else spawn+wait).

---

### Task 1: `launch` module — CLI wrapper + arg dispatch

**Files:**
- Create: `src/launch.rs`
- Modify: `src/lib.rs` (add `pub mod launch;`), `src/main.rs` (dispatch at top of `main`)
- Test: inline `#[cfg(test)]` in `src/launch.rs`

**Interfaces:**
- Produces: `launch::Client` enum (`Claude`, `Codex`), `launch::detect(args: &[String]) -> Option<(Client, Vec<String>)>`, `launch::env_for(client: Client, port: u16) -> Vec<(String, String)>`, `launch::run(client: Client, passthrough: Vec<String>) -> !` (or `-> anyhow::Result<()>` on non-unix), `launch::DEFAULT_PORT: u16`.

- [ ] **Step 1: Write failing tests for `detect` + `env_for`**

Create `src/launch.rs` with a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_claude_and_passthrough() {
        let args = vec!["localllm".into(), "claude".into(), "--resume".into()];
        let (c, rest) = detect(&args).unwrap();
        assert_eq!(c, Client::Claude);
        assert_eq!(rest, vec!["--resume".to_string()]);
    }

    #[test]
    fn detect_codex() {
        let args = vec!["localllm".into(), "codex".into()];
        assert_eq!(detect(&args).unwrap().0, Client::Codex);
    }

    #[test]
    fn detect_none_for_server_mode() {
        let args = vec!["localllm".into(), "--port".into(), "31415".into()];
        assert!(detect(&args).is_none());
    }

    #[test]
    fn env_for_claude_sets_base_url_and_timeouts() {
        let env = env_for(Client::Claude, 31415);
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:31415"));
        assert_eq!(get("API_TIMEOUT_MS"), Some("1200000"));
        assert_eq!(get("CLAUDE_STREAM_IDLE_TIMEOUT_MS"), Some("1200000"));
    }

    #[test]
    fn env_for_codex_sets_openai_base_url() {
        let env = env_for(Client::Codex, 31415);
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("OPENAI_BASE_URL"), Some("http://127.0.0.1:31415/v1"));
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p localllm launch`
Expected: FAIL — module/functions not defined.

- [ ] **Step 3: Implement `launch.rs`**

```rust
//! `localllm claude|codex`: run the client in the current directory wired to the
//! local proxy with raised timeouts, so no manual env export is needed.

use std::process::Command;

/// Default proxy port. MUST match `Config`'s `--port` default (31415).
pub const DEFAULT_PORT: u16 = 31415;

const TIMEOUT_MS: &str = "1200000"; // 20 min; covers slow cold-prefill on the local model

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Client {
    Claude,
    Codex,
}

impl Client {
    fn bin(self) -> &'static str {
        match self {
            Client::Claude => "claude",
            Client::Codex => "codex",
        }
    }
}

/// If the first positional arg is `claude`/`codex`, return the client and the
/// remaining passthrough args. Otherwise None (normal server/tray mode).
pub fn detect(args: &[String]) -> Option<(Client, Vec<String>)> {
    let first = args.get(1)?;
    let client = match first.as_str() {
        "claude" => Client::Claude,
        "codex" => Client::Codex,
        _ => return None,
    };
    Some((client, args[2..].to_vec()))
}

/// Proxy env to inject for a client. Pure — unit-tested.
pub fn env_for(client: Client, port: u16) -> Vec<(String, String)> {
    match client {
        Client::Claude => vec![
            ("ANTHROPIC_BASE_URL".into(), format!("http://127.0.0.1:{port}")),
            ("API_TIMEOUT_MS".into(), TIMEOUT_MS.into()),
            ("CLAUDE_STREAM_IDLE_TIMEOUT_MS".into(), TIMEOUT_MS.into()),
        ],
        Client::Codex => vec![
            ("OPENAI_BASE_URL".into(), format!("http://127.0.0.1:{port}/v1")),
        ],
    }
}

/// Exec the client with proxy env in the current directory. Replaces this
/// process on unix so tty/signals/exit-code pass through.
pub fn run(client: Client, passthrough: Vec<String>) -> ! {
    let mut cmd = Command::new(client.bin());
    cmd.args(&passthrough);
    for (k, v) in env_for(client, DEFAULT_PORT) {
        cmd.env(k, v);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec(); // only returns on failure
        eprintln!("localllm: failed to launch `{}`: {err}. Is it installed and on PATH?", client.bin());
        std::process::exit(127);
    }
    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(s) => std::process::exit(s.code().unwrap_or(1)),
            Err(e) => {
                eprintln!("localllm: failed to launch `{}`: {e}. Is it installed and on PATH?", client.bin());
                std::process::exit(127);
            }
        }
    }
}
```

- [ ] **Step 4: Register the module**

In `src/lib.rs`, add near the other `pub mod` declarations:

```rust
pub mod launch;
```

- [ ] **Step 5: Dispatch at the top of `main`**

In `src/main.rs`, make the FIRST statements of `fn main()` (before the tracing setup at line ~38):

```rust
fn main() -> anyhow::Result<()> {
    // `localllm claude|codex [args]` → run the client wired to the proxy and
    // exit; never start the server or emit logs.
    let raw: Vec<String> = std::env::args().collect();
    if let Some((client, rest)) = localllm::launch::detect(&raw) {
        localllm::launch::run(client, rest); // never returns
    }

    use tracing_subscriber::fmt::writer::MakeWriterExt;
    // ... existing body unchanged ...
```

- [ ] **Step 6: Run tests + build**

Run: `cargo test -p localllm launch && cargo build -p localllm`
Expected: 5 launch tests pass; clean build.

- [ ] **Step 7: Commit**

```bash
git add src/launch.rs src/lib.rs src/main.rs
git commit -m "feat(launch): localllm claude|codex CLI wrapper (proxy env + exec)"
```

---

### Task 2: `terminal` module — app enum, detection, open

**Files:**
- Create: `src/terminal.rs`
- Modify: `src/lib.rs` (`pub mod terminal;`)
- Test: inline `#[cfg(test)]` in `src/terminal.rs`

**Interfaces:**
- Produces: `terminal::TerminalApp` enum (`Terminal`, `ITerm`, `Warp`, `Wave`) with `id()`/`from_id()`/`app_name()`, `terminal::PREFERENCE: [TerminalApp; 4]`, `terminal::pick_default(present: &[TerminalApp]) -> TerminalApp`, `terminal::installed() -> Vec<TerminalApp>`, `terminal::open(app: TerminalApp, dir: &std::path::Path, command: &str) -> std::io::Result<()>`.

- [ ] **Step 1: Write failing tests for id round-trip + default pick**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_round_trips() {
        for a in PREFERENCE {
            assert_eq!(TerminalApp::from_id(a.id()), Some(a));
        }
        assert_eq!(TerminalApp::from_id("nope"), None);
    }

    #[test]
    fn pick_default_prefers_first_present_in_order() {
        // Warp present but Apple Terminal also present → Terminal wins (earlier in PREFERENCE)
        let present = vec![TerminalApp::Warp, TerminalApp::Terminal];
        assert_eq!(pick_default(&present), TerminalApp::Terminal);
        // Only Warp present → Warp
        assert_eq!(pick_default(&[TerminalApp::Warp]), TerminalApp::Warp);
        // Nothing present → Terminal (always exists on macOS)
        assert_eq!(pick_default(&[]), TerminalApp::Terminal);
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p localllm terminal`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement `terminal.rs`**

```rust
//! Terminal-app abstraction for the tray "Launch … via LocalLLM" items (macOS).

use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalApp {
    Terminal, // Apple Terminal (always present on macOS)
    ITerm,
    Warp,
    Wave,
}

/// Preference order for auto-default: Apple Terminal last-resort but always available.
pub const PREFERENCE: [TerminalApp; 4] = [
    TerminalApp::Terminal,
    TerminalApp::ITerm,
    TerminalApp::Warp,
    TerminalApp::Wave,
];

impl TerminalApp {
    pub fn id(self) -> &'static str {
        match self {
            TerminalApp::Terminal => "terminal",
            TerminalApp::ITerm => "iterm",
            TerminalApp::Warp => "warp",
            TerminalApp::Wave => "wave",
        }
    }
    pub fn from_id(s: &str) -> Option<Self> {
        PREFERENCE.into_iter().find(|a| a.id() == s)
    }
    /// The .app bundle name under /Applications.
    pub fn app_name(self) -> &'static str {
        match self {
            TerminalApp::Terminal => "Terminal",
            TerminalApp::ITerm => "iTerm",
            TerminalApp::Warp => "Warp",
            TerminalApp::Wave => "Wave",
        }
    }
    fn is_present(self) -> bool {
        let name = self.app_name();
        std::path::Path::new(&format!("/Applications/{name}.app")).exists()
            || dirs::home_dir()
                .map(|h| h.join("Applications").join(format!("{name}.app")).exists())
                .unwrap_or(false)
    }
}

/// First present app in PREFERENCE order; Apple Terminal always qualifies.
pub fn pick_default(present: &[TerminalApp]) -> TerminalApp {
    PREFERENCE
        .into_iter()
        .find(|a| present.contains(a))
        .unwrap_or(TerminalApp::Terminal)
}

/// Installed terminals in preference order.
pub fn installed() -> Vec<TerminalApp> {
    PREFERENCE.into_iter().filter(|a| a.is_present()).collect()
}

/// Open a new terminal window in `dir` running `command`. Best-effort.
pub fn open(app: TerminalApp, dir: &Path, command: &str) -> std::io::Result<()> {
    let dir = dir.display();
    match app {
        // Scriptable: run the command directly in a new window.
        TerminalApp::Terminal => {
            let script = format!("tell application \"Terminal\" to do script \"cd {dir} && {command}\"");
            Command::new("osascript").arg("-e").arg(script).status().map(|_| ())
        }
        TerminalApp::ITerm => {
            let script = format!(
                "tell application \"iTerm\"\ncreate window with default profile\ntell current session of current window to write text \"cd {dir} && {command}\"\nend tell"
            );
            Command::new("osascript").arg("-e").arg(script).status().map(|_| ())
        }
        // Warp/Wave: no stable `do script`. Open the app at the folder; the user
        // runs the wrapper (which is on PATH). Documented limitation.
        TerminalApp::Warp | TerminalApp::Wave => {
            Command::new("open").arg("-a").arg(app.app_name()).arg(dir.to_string()).status().map(|_| ())
        }
    }
}
```

- [ ] **Step 4: Register module + run tests**

Add `pub mod terminal;` to `src/lib.rs`. Run: `cargo test -p localllm terminal && cargo build -p localllm`
Expected: id/default tests pass; clean build.

- [ ] **Step 5: Commit**

```bash
git add src/terminal.rs src/lib.rs
git commit -m "feat(terminal): TerminalApp enum, install detection, open helper"
```

---

### Task 3: Settings + `/admin/terminal` + Config picker

**Files:**
- Modify: `src/settings.rs` (persist `terminal`), `src/server.rs` (endpoint), `src/manager_ui/app.js` (picker)
- Test: inline settings round-trip in `src/settings.rs`

**Interfaces:**
- Consumes: `terminal::TerminalApp`, `terminal::installed`, `terminal::pick_default` (Task 2)
- Produces: `settings::load_terminal() -> TerminalApp`, `settings::save_terminal(app: TerminalApp)`, `GET/POST /admin/terminal`

- [ ] **Step 1: Persist terminal in settings (mirror cold-prefill-gate)**

Read how `cold_prefill_gate_secs` persists in `src/settings.rs` (a `Settings` field + `load_/save_` fns). Add a `terminal: Option<String>` field to `Settings` and:

```rust
pub fn load_terminal() -> crate::terminal::TerminalApp {
    match load_settings().terminal.as_deref().and_then(crate::terminal::TerminalApp::from_id) {
        Some(a) => a,
        None => crate::terminal::pick_default(&crate::terminal::installed()),
    }
}

pub fn save_terminal(app: crate::terminal::TerminalApp) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.terminal = Some(app.id().to_string());
    save_settings(&s)
}
```

- [ ] **Step 2: Write + run the settings round-trip test**

```rust
#[test]
fn terminal_round_trips_and_defaults_installed() {
    with_temp_settings(|| {
        // unset → a default that is among installed (or Terminal)
        let d = load_terminal();
        assert!(crate::terminal::installed().contains(&d) || d == crate::terminal::TerminalApp::Terminal);
        save_terminal(crate::terminal::TerminalApp::ITerm).unwrap();
        assert_eq!(load_terminal(), crate::terminal::TerminalApp::ITerm);
    });
}
```

Use the file's real settings-isolation helper (match the name used by the cold-prefill-gate test — e.g. `with_temp_settings`/`LOCALLLM_SETTINGS` override). Run: `cargo test -p localllm terminal_round_trips` → PASS.

- [ ] **Step 3: Add the admin endpoint (mirror `/admin/cold-prefill-gate`)**

In `src/server.rs`, next to the cold-gate handlers:

```rust
async fn handle_terminal_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    let available: Vec<&str> = crate::terminal::installed().iter().map(|a| a.id()).collect();
    Json(json!({ "terminal": crate::settings::load_terminal().id(), "available": available })).into_response()
}
async fn handle_terminal_set(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<serde_json::Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    let Some(app) = body.get("terminal").and_then(|v| v.as_str()).and_then(crate::terminal::TerminalApp::from_id) else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":"unknown terminal"}))).into_response();
    };
    let _ = crate::settings::save_terminal(app);
    Json(json!({ "terminal": app.id() })).into_response()
}
```

Register next to the cold-gate route:

```rust
        .route("/admin/terminal", get(handle_terminal_get).post(handle_terminal_set))
```

- [ ] **Step 4: Config UI picker**

In `src/manager_ui/app.js`, add `renderTerminalConfig(container)` (mirror `renderColdGateConfig`): GET `/admin/terminal`, build a `<select>` from `available` (label each by a friendly name map `{terminal:"Terminal",iterm:"iTerm",warp:"Warp",wave:"Wave"}`), select the current `terminal`, POST `{terminal: value}` on change, toast on success. Panel title "Terminal do Launch", sub-text "Terminal usado pelo botão Launch do tray." Call it from `renderConfig` alongside `renderColdGateConfig(shell)`. Use `textContent` for any status text.

Run: `node --check src/manager_ui/app.js` → exit 0.

- [ ] **Step 5: Build + test + commit**

Run: `cargo build -p localllm && cargo test -p localllm terminal`
Expected: green.

```bash
git add src/settings.rs src/server.rs src/manager_ui/app.js
git commit -m "feat(config): persist + pick the Launch terminal (/admin/terminal + UI)"
```

---

### Task 4: Tray "Launch … via LocalLLM" items

**Files:**
- Modify: `src/tray.rs` (menu items + handlers)

**Interfaces:**
- Consumes: `terminal::open`, `settings::load_terminal` (Tasks 2–3); `launch` semantics (the wrapper subcommand).

- [ ] **Step 1: Add two menu items next to the Config item**

In `src/tray.rs` where `config_item` is built (the flattened menu), add:

```rust
                let launch_claude = MenuItem::new("🚀  Launch Claude Code via LocalLLM", true, None);
                let launch_codex = MenuItem::new("🚀  Launch Codex via LocalLLM", true, None);
                launch_claude_id = Some(launch_claude.id().clone());
                launch_codex_id = Some(launch_codex.id().clone());
```

Declare `let mut launch_claude_id: Option<tray_icon::menu::MenuId> = None;` and `launch_codex_id` next to `config_home_id` (~line 535). Append both to the menu right after `config_item` (before `quit_item`):

```rust
                menu.append(&launch_claude).expect("append launch claude");
                menu.append(&launch_codex).expect("append launch codex");
```

- [ ] **Step 2: Add a launch helper in the menu-event handler**

In the menu-event `while let Ok(menu_event)` block, add branches after the config branch:

```rust
                    } else if launch_claude_id.as_ref() == Some(&menu_event.id)
                        || launch_codex_id.as_ref() == Some(&menu_event.id)
                    {
                        let sub = if launch_claude_id.as_ref() == Some(&menu_event.id) { "claude" } else { "codex" };
                        // Native folder picker.
                        let out = std::process::Command::new("osascript")
                            .arg("-e").arg("POSIX path of (choose folder)")
                            .output();
                        if let Ok(o) = out {
                            let dir = String::from_utf8_lossy(&o.stdout).trim().to_string();
                            if !dir.is_empty() {
                                let exe = std::env::current_exe()
                                    .map(|p| p.display().to_string())
                                    .unwrap_or_else(|_| "localllm".into());
                                let command = format!("{exe:?} {sub}"); // quoted abs path + subcommand
                                let app = crate::settings::load_terminal();
                                if let Err(e) = crate::terminal::open(app, std::path::Path::new(&dir), &command) {
                                    tracing::error!("launch {sub}: terminal open failed: {e}");
                                }
                            }
                        }
                    }
```

- [ ] **Step 3: Build**

Run: `cargo build -p localllm`
Expected: clean (macOS). Confirm no unused-variable warnings for the new ids.

- [ ] **Step 4: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): Launch Claude Code / Codex via LocalLLM items"
```

---

### Task 5: Install/refresh the `localllm` CLI symlink on startup

**Files:**
- Modify: `src/launch.rs` (`install_cli`), `src/main.rs` (call at boot)
- Test: inline `#[cfg(test)]` in `src/launch.rs`

**Interfaces:**
- Produces: `launch::install_cli()` (best-effort, no return), `launch::cli_target_dir(candidates: &[&Path], home_local_bin: &Path, is_writable: impl Fn(&Path)->bool) -> PathBuf` (pure selection, unit-tested)

Context: the app runs from a `.app` bundle, so `localllm` is not on PATH. Re-symlinking to `current_exe()` on every launch keeps `$ localllm claude` matching the running version.

- [ ] **Step 1: Write the failing test for the dir-selection logic**

Pure selection is testable without touching the real filesystem:

```rust
#[test]
fn cli_target_dir_prefers_first_writable_then_home() {
    use std::path::{Path, PathBuf};
    let usr = Path::new("/usr/local/bin");
    let brew = Path::new("/opt/homebrew/bin");
    let home = PathBuf::from("/Users/x/.local/bin");
    // /usr/local/bin writable → chosen
    assert_eq!(cli_target_dir(&[usr, brew], &home, |p| p == usr), usr.to_path_buf());
    // only brew writable → brew
    assert_eq!(cli_target_dir(&[usr, brew], &home, |p| p == brew), brew.to_path_buf());
    // none writable → home fallback
    assert_eq!(cli_target_dir(&[usr, brew], &home, |_| false), home);
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cargo test -p localllm cli_target_dir`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement `install_cli` + `cli_target_dir`**

Add to `src/launch.rs`:

```rust
use std::path::{Path, PathBuf};

/// Pick the CLI install dir: first writable candidate, else the home fallback.
pub fn cli_target_dir(candidates: &[&Path], home_local_bin: &Path, is_writable: impl Fn(&Path) -> bool) -> PathBuf {
    for c in candidates {
        if is_writable(c) {
            return c.to_path_buf();
        }
    }
    home_local_bin.to_path_buf()
}

/// (Re)symlink `localllm` → the running binary into a PATH dir, so `localllm
/// claude` works from the user's terminal and always matches this app version.
/// Best-effort — never blocks boot.
pub fn install_cli() {
    let Ok(exe) = std::env::current_exe() else { return };
    let home_local_bin = dirs::home_dir().map(|h| h.join(".local/bin")).unwrap_or_else(|| PathBuf::from("/usr/local/bin"));
    let candidates = [Path::new("/usr/local/bin"), Path::new("/opt/homebrew/bin")];
    let is_writable = |p: &Path| p.exists() && std::fs::metadata(p).map(|m| !m.permissions().readonly()).unwrap_or(false)
        && { let t = p.join(".localllm-wtest"); let ok = std::fs::write(&t, b"").is_ok(); let _ = std::fs::remove_file(&t); ok };
    let dir = cli_target_dir(&candidates, &home_local_bin, is_writable);
    let _ = std::fs::create_dir_all(&dir);
    let link = dir.join("localllm");
    // Idempotent: skip if already pointing at the current exe.
    if std::fs::read_link(&link).ok().as_deref() == Some(exe.as_path()) {
        return;
    }
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    let res = std::os::unix::fs::symlink(&exe, &link);
    #[cfg(not(unix))]
    let res: std::io::Result<()> = std::fs::copy(&exe, &link).map(|_| ());
    match res {
        Ok(_) => {
            tracing::info!(target: "localllm", "CLI installed: {} -> {}", link.display(), exe.display());
            if !std::env::var("PATH").unwrap_or_default().split(':').any(|p| Path::new(p) == dir) {
                tracing::warn!(target: "localllm", "{} is not on PATH — add it to use `localllm claude`", dir.display());
            }
        }
        Err(e) => tracing::warn!(target: "localllm", "CLI install skipped: {e}"),
    }
}
```

- [ ] **Step 4: Run test, verify pass**

Run: `cargo test -p localllm cli_target_dir`
Expected: PASS.

- [ ] **Step 5: Call at boot (after dispatch, before server/tray)**

In `src/main.rs`, right after the `launch::detect`/`run` dispatch block (Task 1 Step 5), add:

```rust
    // Keep the `localllm` CLI symlink fresh with this running binary.
    localllm::launch::install_cli();
```

(This runs for both tray and headless startup, and only after the `claude`/`codex` wrapper has already exited — the wrapper returns early above.)

- [ ] **Step 6: Build + commit**

Run: `cargo build -p localllm && cargo test -p localllm launch`
Expected: green.

```bash
git add src/launch.rs src/main.rs
git commit -m "feat(launch): (re)install localllm CLI symlink on startup"
```

---

### Task 6: Build + verify

**Files:** none (build + manual verify)

- [ ] **Step 1: Full build + tests**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean, all pass.

- [ ] **Step 2: CLI wrapper smoke test (no client needed)**

Run: `./target/debug/localllm claude --nonexistent-flag-xyz; echo "exit=$?"`
Expected: either Claude Code launches (if installed) OR the "failed to launch `claude` … on PATH?" message with a non-zero exit — NOT the server starting / logs.

- [ ] **Step 3: Rebuild bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS`.

- [ ] **Step 4: Manual checks**

- Config page shows "Terminal do Launch" with the installed terminals; pick one.
- Tray shows "🚀 Launch Claude Code via LocalLLM" / Codex (no submenu). Click → folder picker → chosen terminal opens in that folder and runs `localllm claude`, which launches Claude Code wired to the proxy.

- [ ] **Step 5: Commit any tweaks**

```bash
git add -A && git commit -m "chore: client launcher verification tweaks"
```

---

## Self-Review

**Spec coverage:**
- CLI wrapper `localllm claude`/`codex` (env + exec, cwd, timeouts) → Task 1. ✓
- Arg dispatch before server/tray, no logs → Task 1 Step 5 + Global Constraints. ✓
- Terminal enum + install detection + open (scriptable + Warp/Wave fallback) → Task 2. ✓
- Settings persist terminal + `/admin/terminal` + Config picker → Task 3. ✓
- Tray "Launch … via LocalLLM" items, folder pick, chosen terminal, no submenu, best-effort → Task 4. ✓
- CLAUDE_STREAM_IDLE_TIMEOUT_MS carried → Task 1 `env_for`. ✓
- macOS-only terminal/tray; cross-platform wrapper → Task 1 (`cfg(unix)`) + Global Constraints. ✓
- CLI (re)installed on every startup to match the running version, best-effort, no uninstall on quit → Task 5 (`install_cli`, called at boot). ✓

**Placeholder scan:** all code steps carry concrete code; the settings test names the real isolation helper by matching the cold-prefill-gate test (implementer reads it). No TBD/handle-edge-cases left.

**Type consistency:** `Client`/`env_for`/`detect`/`DEFAULT_PORT` (launch) and `TerminalApp`/`id`/`from_id`/`installed`/`pick_default`/`open` (terminal) used consistently across Tasks 1–4. `load_terminal`/`save_terminal` return/accept `TerminalApp`. Endpoint `/admin/terminal` → `{terminal, available}` consistent between Task 3 handler and UI.
