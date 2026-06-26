# Model Manager Window + Cross-Platform Tray Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tray-launched desktop window (webview) for browsing/switching/deleting models, served by axum and backed by the existing `/admin/*` API, with the tray made cross-platform.

**Architecture:** axum serves an embedded vanilla SPA at `GET /manager`; a `tao::Window` + `wry::WebView` (created on demand from the tray's event loop) opens that URL and injects the admin token in-memory; the SPA drives `/admin/*`. The tray module is de-macOS-gated with a thin `platform` abstraction for OS-specific calls.

**Tech Stack:** Rust, axum, tao 0.35, wry, tray-icon 0.24, vanilla HTML/CSS/JS.

## Global Constraints

- Builds on sub-projects 1+2 (branch `feat/model-manager-window`, stacked on `feat/model-catalog`): `/admin/model` (POST switch), `/admin/model/status` (GET), `/admin/models` (GET catalog) + `DELETE /admin/models`, all guarded by `check_admin` (header `x-admin-token`, constant-time). `ModelManager`, `catalog::FamilyView`, `SwitchStatus`. `resolve_admin_token`/`write_admin_token_file` exist.
- **UI delivery:** SPA embedded (`include_str!`) + served at `GET /manager`(`.js`/`.css`); webview opens `http://127.0.0.1:PORT/manager` (same origin → no CORS); token injected via wry init script `window.__ADMIN_TOKEN__='<hex>'` (never read from the token file).
- **Window:** single on-demand instance held in the tray event loop; re-open shows/focuses; close hides.
- **Cross-platform:** `tao`/`wry`/`tray-icon` cross-platform; OS-specific calls (clipboard/open/activation/bundle-detect) behind a cfg-dispatched `platform` module. **macOS verified here; Linux/Windows compile-best-effort + flagged for on-device verification.**
- **Preserve ALL current tray features** (status 🟡/🟢, URL→copy, Model/Context/KV/Backend, Routing submenu+persistence, Open Logs, Quit) + add "Open Model Manager".
- v1 actions: Switch (with progress) + Delete; no download-only. Token is 32-hex (safe to inject into a JS string literal).
- TDD where testable (routes, `manager_init_script`); the SPA is built with the `frontend-design` skill; the window/tray shell is verified by build + macOS manual smoke. Commit per task; pristine build on macOS.

---

### Task 1: `/manager` route + embedded SPA + `manager_init_script` (cross-platform, headless-testable)

**Files:**
- Create: `src/manager_ui/index.html`, `src/manager_ui/app.js`, `src/manager_ui/style.css`
- Modify: `src/server.rs` (3 static routes + `manager_init_script`)
- Test: `tests/http.rs`; `src/server.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces: routes `GET /manager`, `GET /manager/app.js`, `GET /manager/style.css`; `pub fn manager_init_script(token: &str) -> String`.

- [ ] **Step 1: Write the failing tests**

Add to `src/server.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn manager_init_script_injects_token() {
        let s = super::manager_init_script("abc123");
        assert_eq!(s, "window.__ADMIN_TOKEN__='abc123';");
        assert!(s.contains("abc123"));
    }
```

Append to `tests/http.rs`:

```rust
#[tokio::test]
async fn manager_page_served_no_auth() {
    let app = localllm::router_for_test();
    let (status, ctype, body) = localllm::axum_test_get_full(app, "/manager").await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("text/html"));
    assert!(body.contains("<html") || body.contains("<!doctype") || body.contains("<!DOCTYPE"));
}

#[tokio::test]
async fn manager_assets_served_with_types() {
    let app = localllm::router_for_test();
    let (s1, c1, _b1) = localllm::axum_test_get_full(app.clone(), "/manager/app.js").await;
    assert_eq!(s1, 200);
    assert!(c1.contains("javascript"));
    let (s2, c2, _b2) = localllm::axum_test_get_full(app, "/manager/style.css").await;
    assert_eq!(s2, 200);
    assert!(c2.contains("css"));
}
```

Add a test helper to `src/lib.rs` (next to the other `axum_test_*`):

```rust
/// GET `path` → (status, content-type, body string).
pub async fn axum_test_get_full(app: Router, path: &str) -> (u16, String, String) {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let request = axum::http::Request::builder().method("GET").uri(path).body(Body::empty()).unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let ctype = response.headers().get("content-type")
        .and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, ctype, String::from_utf8_lossy(&bytes).into_owned())
}
```

> `router_for_test` must be `Clone`able for the asset test — axum `Router` is `Clone`, so `app.clone()` works.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib server::tests::manager_init 2>&1 | head; cargo test --test http manager_ 2>&1 | tail -15`
Expected: FAIL — `manager_init_script` undefined; `/manager` routes 404.

- [ ] **Step 3: Create the SPA assets with the frontend-design skill**

Invoke the **frontend-design skill** to author the three files below. They are a small vanilla (no framework, no build) single-page app. Required behavior + data contracts (consume verbatim):

- `GET /admin/models` → `FamilyView[]` where each `FamilyView = { family: string, models: ModelView[] }` and `ModelView = { display_name, params, quant, repo, file, size_mb, est_ram_mb, status: "in_use"|"downloaded"|"needs_download", fit: "fits"|"tight"|"wont_fit", recommended: bool }`.
- `GET /admin/model/status` → `{ state: "ready"|"switching"|"error", current: {repo,file}, target: {repo,file}|null, phase: "idle"|"draining"|"downloading"|"loading", progress: 0..100, error: string|null }`.
- `POST /admin/model` body `{repo,file}` → 202 / 409. `DELETE /admin/models` body `{repo,file}` → `{deleted:bool}` / 409 (in use).
- Every `/admin/*` `fetch` sends header `x-admin-token: window.__ADMIN_TOKEN__`.

UI: **Firestore-style drilldown** — Pane 1 families → Pane 2 model cards (params/quant, est-RAM + fit badge, status badge, ⭐ recommended) → Pane 3 detail with **Switch** (then poll `/admin/model/status` ~1s, show a progress bar by `phase`/`progress`, refresh on `ready`) and **Delete** (disabled when `in_use`; confirm; refresh). Header shows the active model + a "switching…" state (poll while `state=="switching"`; disable actions). Inline toast on non-2xx (401→"restart app", 409→message). Distinctive, polished styling (frontend-design quality) — not generic.

Save the produced files to `src/manager_ui/index.html`, `app.js`, `style.css`. `index.html` must reference `/manager/app.js` and `/manager/style.css` and contain a `<!DOCTYPE html>`.

- [ ] **Step 4: Implement the routes + init script**

In `src/server.rs`, add `manager_init_script` (module level):

```rust
/// The webview initialization script that injects the admin token in-memory so
/// the SPA can authenticate against /admin/* without reading the token file.
/// `token` is a 32-hex string (no quotes/escapes), safe in a JS string literal.
pub fn manager_init_script(token: &str) -> String {
    format!("window.__ADMIN_TOKEN__='{token}';")
}
```

Add three handlers (static, no auth):

```rust
async fn handle_manager_page() -> axum::response::Response {
    use axum::response::IntoResponse;
    ([("content-type", "text/html; charset=utf-8")],
     include_str!("manager_ui/index.html")).into_response()
}
async fn handle_manager_js() -> axum::response::Response {
    use axum::response::IntoResponse;
    ([("content-type", "text/javascript; charset=utf-8")],
     include_str!("manager_ui/app.js")).into_response()
}
async fn handle_manager_css() -> axum::response::Response {
    use axum::response::IntoResponse;
    ([("content-type", "text/css; charset=utf-8")],
     include_str!("manager_ui/style.css")).into_response()
}
```

Register in `router(...)` (before the `DefaultBodyLimit` layer):

```rust
        .route("/manager", get(handle_manager_page))
        .route("/manager/app.js", get(handle_manager_js))
        .route("/manager/style.css", get(handle_manager_css))
```

- [ ] **Step 5: Run to verify they pass**

Run: `cargo test --lib server::tests::manager_init 2>&1 | tail -8 && cargo test --test http manager_ 2>&1 | tail -15`
Expected: PASS — init-script unit test + the two `/manager` integration tests.

- [ ] **Step 6: Full suite + commit**

Run: `cargo test 2>&1 | tail -8`
Expected: full suite green.

```bash
git add src/manager_ui src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): /manager SPA route + admin-token init script"
```

---

### Task 2: Cross-platform tray refactor (preserve all features; no window yet)

**Files:**
- Modify: `src/tray.rs` (de-gate; `platform` submodule), `src/main.rs` (cross-platform dispatch), `src/lib.rs` (thread `admin_token` to the server fn), `Cargo.toml` (move `tao`/`tray-icon`/`png` to `[dependencies]`)
- Test: build + macOS run (no new unit tests; the deliverable is "compiles cross-platform + macOS behaviour unchanged")

**Interfaces:**
- Consumes: existing `run_tray` body, `resolve_admin_token`.
- Produces: cross-platform `tray::run_tray(cfg: Config, admin_token: std::sync::Arc<str>)`; `lib::run_server_with_ready_policy_token(cfg, ready, policy, admin_token)`.

- [ ] **Step 1: Move the GUI deps to all-platform**

In `Cargo.toml`, delete the `[target.'cfg(target_os = "macos")'.dependencies]` section and add to the main `[dependencies]`:

```toml
tao = { version = "0.35", default-features = false }
tray-icon = { version = "0.24", default-features = false }
png = "0.17"
wry = "0.45"
```

> Pin `wry` to a version compatible with `tao = 0.35`; if `cargo build` reports a tao version conflict, adjust `wry` to the matching release (verify the build before Step 6).

- [ ] **Step 2: Thread the admin token to the server entry**

In `src/lib.rs`, make `run_server_with_ready_and_policy` resolve the token then delegate to a token-taking variant (so the tray can pass a shared token). Replace the token-resolution + `router(...)` region with:

```rust
    // (resolve+write token now happens in the _token variant)
    run_server_with_ready_policy_token(
        cfg, ready, policy, std::sync::Arc::from(crate::server::resolve_admin_token(None)),
    ).await
}

/// Like [`run_server_with_ready_and_policy`] but takes an externally-resolved
/// admin token (so the tray shares the same token with its Model Manager window).
pub async fn run_server_with_ready_policy_token(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    policy: std::sync::Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    admin_token: std::sync::Arc<str>,
) -> anyhow::Result<()> {
```

Move the engine-build + manager-build body into `run_server_with_ready_policy_token`; replace its internal `resolve_admin_token`/`write_admin_token_file` with: `crate::server::write_admin_token_file(&admin_token);` and pass `admin_token.clone()` to `router(...)`.

> Net: `run_server_with_ready_and_policy` resolves a token (headless path, CLI override honored: use `cfg.admin_token.clone()` not `None` — pass `crate::server::resolve_admin_token(cfg.admin_token.clone())`) and delegates; the `_token` variant does the real work.

(Correct the delegation to honor the CLI flag:)
```rust
    let admin_token = std::sync::Arc::from(crate::server::resolve_admin_token(cfg.admin_token.clone()));
    run_server_with_ready_policy_token(cfg, ready, policy, admin_token).await
```

- [ ] **Step 3: De-gate the tray module + add `platform`**

In `src/tray.rs`, change the module so it compiles everywhere. Replace the `#[cfg(target_os = "macos")] pub mod macos {` wrapper so the contents become the crate-level `tray` module body (i.e. `tray::run_tray`), and adjust imports:

- Remove `use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};` from the unconditional imports.
- Add an internal `platform` module:

```rust
mod platform {
    /// Copy text to the OS clipboard, best-effort.
    pub fn copy_to_clipboard(text: &str) {
        use std::io::Write;
        use std::process::{Command, Stdio};
        #[cfg(target_os = "macos")]
        let mut cmd = Command::new("pbcopy");
        #[cfg(target_os = "windows")]
        let mut cmd = Command::new("clip");
        #[cfg(all(unix, not(target_os = "macos")))]
        let mut cmd = { // prefer wl-copy, else xclip
            if Command::new("wl-copy").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok() {
                Command::new("wl-copy")
            } else {
                let mut c = Command::new("xclip"); c.args(["-selection","clipboard"]); c
            }
        };
        if let Ok(mut child) = cmd.stdin(Stdio::piped()).spawn() {
            if let Some(si) = child.stdin.as_mut() { let _ = si.write_all(text.as_bytes()); }
            let _ = child.wait();
        }
    }

    /// Open a file/path in the OS default handler, best-effort.
    pub fn open_path(path: &str) {
        use std::process::Command;
        #[cfg(target_os = "macos")]
        { let _ = Command::new("open").arg("-t").arg(path).spawn(); }
        #[cfg(target_os = "windows")]
        { let _ = Command::new("cmd").args(["/C", "start", "", path]).spawn(); }
        #[cfg(all(unix, not(target_os = "macos")))]
        { let _ = Command::new("xdg-open").arg(path).spawn(); }
    }
}
```

Replace the existing `copy_to_clipboard(...)`/`open_log_file(...)` call sites with `platform::copy_to_clipboard(...)` / `platform::open_path(...)`.

Guard the macOS activation policy:

```rust
        let mut event_loop = EventLoopBuilder::<()>::new().build();
        #[cfg(target_os = "macos")]
        {
            use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
            event_loop.set_activation_policy(ActivationPolicy::Accessory);
        }
```

Change `run_tray` to take the token: `pub fn run_tray(cfg: Config, admin_token: std::sync::Arc<str>) -> !` and pass it to the server thread via `run_server_with_ready_policy_token(cfg, Some(ready_for_server), policy_for_server, admin_token.clone())`. Keep `admin_token` in scope for Task 3's window (store it in a closure-captured variable now, unused this task — prefix `_admin_token_for_window` to avoid the unused warning, OR add the window in Task 3 which consumes it).

- [ ] **Step 4: Cross-platform dispatch in `main.rs`**

Replace the `#[cfg(target_os = "macos")]` tray block with a cross-platform one:

```rust
    let want_tray = args.tray || {
        #[cfg(target_os = "macos")]
        { std::env::var_os("__CFBundleIdentifier").is_some() }
        #[cfg(not(target_os = "macos"))]
        { false }
    };
    if want_tray {
        let token = std::sync::Arc::from(localllm::server::resolve_admin_token(args.config.admin_token.clone()));
        localllm::tray::run_tray(args.config, token); // -> !
    }
```

(Update the `localllm::tray::macos::run_tray` reference to `localllm::tray::run_tray`.)

- [ ] **Step 5: Build (macOS) + run the suite**

Run: `cargo build 2>&1 | tail -20`
Expected: clean compile on macOS (no `macos` module path left dangling; `wry` resolves against tao 0.35).

Run: `cargo test 2>&1 | tail -8`
Expected: full suite green (server/lib tests unaffected).

- [ ] **Step 6: macOS manual smoke + commit**

Run (manual, macOS): `cargo run -- --tray --port 31998` → confirm the menu-bar icon appears with ALL existing items (status, URL, Model/Context/KV/Backend, Routing submenu, Open Logs, Quit) and behaves as before (URL copies, Routing switches, Open Logs opens). Record the result in the task report (or "deferred to user" if no GUI session).

```bash
git add Cargo.toml Cargo.lock src/tray.rs src/main.rs src/lib.rs
git commit -m "refactor(tray): cross-platform tray (de-gate + platform abstraction) + share admin token"
```

---

### Task 3: The Model Manager window

**Files:**
- Create: `src/tray/window.rs` (or a `window` submodule inside `tray.rs`)
- Modify: `src/tray.rs` ("Open Model Manager" item + event dispatch + window state)
- Test: build + macOS manual smoke (window logic is GUI; no unit test beyond `manager_init_script` from Task 1)

**Interfaces:**
- Consumes: `manager_init_script` (Task 1), `tao`/`wry`, `admin_token` + `port` from `run_tray`.
- Produces: `ModelManagerWindow { window: tao::window::Window, _webview: wry::WebView }` + an open/show helper.

- [ ] **Step 1: Add the window module**

Create the window helper (as a submodule of `tray`). It builds a window + webview and injects the token:

```rust
//! The Model Manager webview window (tao + wry). Single-instance, opened from
//! the tray. Cross-platform (macOS WKWebView / Linux WebKitGTK / Windows WebView2).
pub(crate) struct ModelManagerWindow {
    pub window: tao::window::Window,
    _webview: wry::WebView,
}

impl ModelManagerWindow {
    /// Build the window + webview pointed at the local /manager SPA, injecting
    /// the admin token in-memory.
    pub fn open<T: 'static>(
        target: &tao::event_loop::EventLoopWindowTarget<T>,
        port: u16,
        admin_token: &str,
    ) -> anyhow::Result<Self> {
        use tao::dpi::LogicalSize;
        use tao::window::WindowBuilder;
        let window = WindowBuilder::new()
            .with_title("localllm — Model Manager")
            .with_inner_size(LogicalSize::new(900.0, 640.0))
            .build(target)?;
        let url = format!("http://127.0.0.1:{port}/manager");
        let webview = wry::WebViewBuilder::new(&window)
            .with_url(&url)
            .with_initialization_script(&crate::server::manager_init_script(admin_token))
            .build()?;
        Ok(Self { window, _webview: webview })
    }
}
```

> If the installed `wry` version's `WebViewBuilder::new` takes the window by a different reference/handle shape, adapt minimally to that version's API and note it; the contract (open URL + init script) is fixed.

- [ ] **Step 2: Add the menu item + state + dispatch in `run_tray`**

In `src/tray.rs`:
- Near the other closure state, add: `let mut manager_window: Option<window::ModelManagerWindow> = None;` and capture `let port = cfg.port;` (already present) and `let admin_token = admin_token;` (moved into the loop).
- In the `Init` arm, create the item and remember its id: `let manager_item = MenuItem::new("Open Model Manager", true, None); let manager_id = manager_item.id().clone();` and `menu.append(&manager_item).expect("append manager item");` (place it near "Open Logs").
- In the `MenuEvent` poll loop, add a branch:

```rust
                        } else if manager_id.as_ref() == Some(&menu_event.id) {
                            // (manager_id stored as Option to match the existing pattern)
                            match &manager_window {
                                Some(w) => { w.window.set_visible(true); w.window.set_focus(); }
                                None => match window::ModelManagerWindow::open(target, port, &admin_token) {
                                    Ok(w) => manager_window = Some(w),
                                    Err(e) => tracing::error!("Model Manager window failed: {e} (needs WebView2/WebKitGTK)"),
                                },
                            }
                        }
```

> The event-loop closure exposes the window target as the second `run` callback arg (currently `_`). Rename it to `target` so `ModelManagerWindow::open(target, …)` can build the window. Store `manager_id: Option<MenuId>` alongside the other ids and set it in `Init`.

- Handle close-to-hide: add a `Event::WindowEvent { event: tao::event::WindowEvent::CloseRequested, window_id }` arm — if `manager_window`'s `window.id() == window_id`, call `set_visible(false)` (keep the instance).

- [ ] **Step 3: Build (macOS) + suite**

Run: `cargo build 2>&1 | tail -20`
Expected: clean compile on macOS (wry + tao window APIs line up).

Run: `cargo test 2>&1 | tail -8`
Expected: full suite green.

- [ ] **Step 4: macOS manual smoke**

Run (manual, macOS): `cargo run -- --tray --port 31997`, click **Open Model Manager** → window opens to the catalog; drilldown family → model; **Switch** a small model and watch the progress bar; **Delete** a cached non-active model; close the window (hides) and re-open (same instance, no duplicate). Record observations (or "deferred to user").

- [ ] **Step 5: Commit**

```bash
git add src/tray.rs Cargo.toml Cargo.lock
git commit -m "feat(tray): Model Manager window (tao+wry) with in-memory token injection"
```

---

## Acceptance

- `GET /manager`(`.js`/`.css`) serve the embedded SPA with correct content-types, no auth; `manager_init_script` injects the token.
- The tray compiles cross-platform (deps un-gated, OS calls behind `platform`), preserves every existing menu item, and adds "Open Model Manager".
- On macOS: the window opens to the catalog, drilldown works, Switch shows progress and activates, Delete frees a cached model, single-instance re-open + close-to-hide work.
- `cargo test`/`build`/`clippy` clean on macOS. Linux/Windows: code present + compile-best-effort, flagged for on-device verification.

## Self-Review

- **Spec coverage:** SPA + /manager route + token injection (T1); cross-platform tray + preserved features + platform abstraction + token threading (T2); the window + Open Model Manager + single-instance + close-to-hide (T3). Switch/Delete/progress are SPA behaviors specified via the frontend-design brief in T1 Step 3 against the exact `/admin/*` contracts.
- **Placeholder scan:** the SPA markup is produced via the frontend-design skill (a justified deviation — it's a design artifact, not transcribable Rust) against precise data contracts + behaviors; all Rust glue has complete code. No TBDs.
- **Type consistency:** `manager_init_script(&str)->String`, the three `/manager` routes, `ModelManagerWindow::open(target, port, &str)`, `run_tray(cfg, Arc<str>)`, `run_server_with_ready_policy_token(cfg, ready, policy, Arc<str>)`, `axum_test_get_full`, and the FamilyView/SwitchStatus JSON contracts are consistent across tasks and match sub-projects 1+2.
- **Verification reality:** macOS build+run verified; Linux/Windows compile-best-effort + flagged (per spec) — not a gap, a documented constraint.
