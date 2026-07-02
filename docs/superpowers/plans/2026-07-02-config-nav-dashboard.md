# Config Nav + Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the tray + Manager SPA around a Config hub, move the route-apps toggle into the SPA, add a Dashboard backed by a persistent routing log, hide windows on blur, and un-wire integrations on Quit.

**Architecture:** Rust axum server + macOS `tao`/`tray-icon`/`wry` tray. One shared webview window points at the `/manager` SPA (vanilla JS) and is deep-linked via hash routes. New per-request routing history persists as JSONL with rolling retention; rollups exposed via a token-guarded admin endpoint.

**Tech Stack:** Rust (axum, serde, tokio), vanilla JS SPA, `tao`+`wry`+`tray-icon`.

## Global Constraints

- New `settings.json` fields MUST be `#[serde(default)]` (older files must still load). Copy the existing pattern in `src/settings.rs`.
- Admin endpoints MUST call `check_admin(&headers, &state)` first and return its response if `Some`.
- Route-log / app-log writes MUST be best-effort: a failure never fails a request and never panics.
- Retention window ≈ 30 days for the routing log; ≈ 7 days for the app log. Prune on boot.
- Token metrics are raw token counts (no dollar cost). `tokens_saved` = Σ(prompt+completion) of **local** requests; `tokens_if_all_cloud` = Σ(prompt+completion) of **all** requests.
- Dashboard **visual layout** is co-designed with the user via the frontend-design + brainstorming visual companion BEFORE building the Dashboard SPA view (Task 15). Data contract is fixed by Task 12.
- Settings tests use the `with_temp_settings` helper + `ENV_LOCK` in `src/settings.rs::tests`.
- HTTP handler tests use the `axum_test_*` helpers (see `tests/http.rs`).
- macOS is the only supported tray runtime; tray tasks are verified by build + manual run.

---

## File Structure

- `src/settings.rs` — add `active_model` field + `save_active_model`/`load_active_model`/`resolve_active_model`.
- `src/route_log.rs` — **new**. JSONL persistence, `RouteEntry`, `record`, `prune`, `rollup`, `Rollups`, `Dashboard`.
- `src/lib.rs` — register `pub mod route_log;`; apply saved model at boot; prune logs at boot.
- `src/server.rs` — write a route-log entry in `route_decision`; add `/admin/dashboard`, `/admin/integrations` (GET/POST); save `active_model` on switch.
- `src/main.rs` — apply saved model to `cfg` before launching.
- `src/manager_ui/app.js` — hash router, Config landing, integrations toggle, Dashboard view.
- `src/manager_ui/style.css` — styles for new views (frontend-design skill).
- `src/tray.rs` — Config submenu, read-only wired line, un-wire on Quit, hide-on-blur, navigate-to-route.
- `tests/http.rs` — tests for `/admin/dashboard`, `/admin/integrations`.

---

## Phase 1 — Model persistence (sub-project B backend)

### Task 1: `active_model` persistence + resolution (pure)

**Files:**
- Modify: `src/settings.rs`
- Test: `src/settings.rs::tests`

**Interfaces:**
- Produces:
  - `struct ActiveModel { repo: String, file: String, quant: Option<String> }` (serde, in `settings.rs`)
  - `fn load_active_model() -> Option<ActiveModel>`
  - `fn save_active_model(repo: &str, file: &str, quant: Option<&str>) -> anyhow::Result<()>`
  - `fn resolve_active_model(cli_model: &str, cli_files: &[String], default_model: &str, default_file: &str, saved: Option<ActiveModel>) -> (String, Vec<String>)`

- [ ] **Step 1: Write the failing test**

Add to `src/settings.rs::tests`:

```rust
#[test]
fn active_model_round_trips() {
    with_temp_settings(|| {
        assert!(load_active_model().is_none());
        save_active_model("Qwen/Qwen2.5-7B-Instruct-GGUF", "q7.gguf", Some("Q4_K_M")).unwrap();
        let a = load_active_model().unwrap();
        assert_eq!(a.repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
        assert_eq!(a.file, "q7.gguf");
        assert_eq!(a.quant.as_deref(), Some("Q4_K_M"));
    });
}

#[test]
fn resolve_active_model_prefers_explicit_cli() {
    // CLI differs from default → CLI wins even if a model is saved.
    let saved = Some(ActiveModel { repo: "saved/repo".into(), file: "s.gguf".into(), quant: None });
    let (m, f) = resolve_active_model(
        "cli/repo", &["c.gguf".to_string()], "default/repo", "d.gguf", saved);
    assert_eq!(m, "cli/repo");
    assert_eq!(f, vec!["c.gguf".to_string()]);
}

#[test]
fn resolve_active_model_uses_saved_when_cli_is_default() {
    let saved = Some(ActiveModel { repo: "saved/repo".into(), file: "s.gguf".into(), quant: None });
    let (m, f) = resolve_active_model(
        "default/repo", &["d.gguf".to_string()], "default/repo", "d.gguf", saved);
    assert_eq!(m, "saved/repo");
    assert_eq!(f, vec!["s.gguf".to_string()]);
}

#[test]
fn resolve_active_model_falls_back_to_default_when_nothing_saved() {
    let (m, f) = resolve_active_model(
        "default/repo", &["d.gguf".to_string()], "default/repo", "d.gguf", None);
    assert_eq!(m, "default/repo");
    assert_eq!(f, vec!["d.gguf".to_string()]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib settings::tests::active_model -- --nocapture` and `cargo test --lib settings::tests::resolve_active_model`
Expected: FAIL to compile (`ActiveModel`, `load_active_model`, `resolve_active_model` undefined).

- [ ] **Step 3: Implement**

In `src/settings.rs`, add the struct near `ExecProfile`:

```rust
/// The last successfully-activated model, restored at boot unless the CLI
/// explicitly overrides `--model-id`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ActiveModel {
    pub repo: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}
```

Add to `struct Settings`:

```rust
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_model: Option<ActiveModel>,
```

Add the accessors (place beside `load_profile`/`save_profile`):

```rust
/// Load the last-activated model, if any was saved.
pub fn load_active_model() -> Option<ActiveModel> {
    load_settings().active_model
}

/// Persist the last-activated model (called on a successful switch).
pub fn save_active_model(repo: &str, file: &str, quant: Option<&str>) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.active_model = Some(ActiveModel {
        repo: repo.to_string(),
        file: file.to_string(),
        quant: quant.map(|q| q.to_string()),
    });
    save_settings(&s)
}

/// Resolve the boot model: an explicit CLI `--model-id` (i.e. one that differs
/// from the compiled default) wins; otherwise the saved model; otherwise the
/// default. Split-model file lists from the saved entry are not restored
/// (single-file only) — a saved model always resolves to `[file]`.
pub fn resolve_active_model(
    cli_model: &str,
    cli_files: &[String],
    default_model: &str,
    default_file: &str,
    saved: Option<ActiveModel>,
) -> (String, Vec<String>) {
    if cli_model != default_model {
        return (cli_model.to_string(), cli_files.to_vec());
    }
    match saved {
        Some(a) => (a.repo, vec![a.file]),
        None => (default_model.to_string(), vec![default_file.to_string()]),
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib settings::`
Expected: PASS (all settings tests green).

- [ ] **Step 5: Commit**

```bash
git add src/settings.rs
git commit -m "feat(settings): persist + resolve last active model"
```

### Task 2: Save on switch + restore at boot

**Files:**
- Modify: `src/server.rs` (`handle_admin_switch`)
- Modify: `src/main.rs` (apply saved model to `cfg`)

**Interfaces:**
- Consumes: `settings::save_active_model`, `settings::resolve_active_model`, `settings::load_active_model` (Task 1).

- [ ] **Step 1: Save on successful switch**

In `src/server.rs::handle_admin_switch`, capture the spec fields before the move and save on `Ok(())`:

```rust
    let spec = crate::model_manager::ModelSpec { repo: body.repo, file: body.file, quant: body.quant };
    let (save_repo, save_file, save_quant) = (spec.repo.clone(), spec.file.clone(), spec.quant.clone());
    match state.manager.start_switch(spec) {
        Ok(()) => {
            if let Err(e) = crate::settings::save_active_model(&save_repo, &save_file, save_quant.as_deref()) {
                tracing::warn!("failed to persist active model: {e}");
            }
            (StatusCode::ACCEPTED, Json(json!({"state": "switching"}))).into_response()
        }
        Err(crate::model_manager::SwitchError::AlreadySwitching) => (
            StatusCode::CONFLICT,
            Json(json!({"error": "switch already in progress"})),
        )
            .into_response(),
    }
```

- [ ] **Step 2: Apply saved model at boot**

In `src/main.rs`, after `let args = MainArgs::parse();` and before the model is loaded/tray launched, mutate `cfg`:

```rust
    let mut cfg = args.config;
    // Restore the last activated model unless --model-id was explicitly passed.
    {
        const DEFAULT_MODEL: &str = "Qwen/Qwen2.5-3B-Instruct-GGUF";
        const DEFAULT_FILE: &str = "qwen2.5-3b-instruct-q4_k_m.gguf";
        let (model, files) = localllm::settings::resolve_active_model(
            &cfg.model_id, &cfg.gguf_files, DEFAULT_MODEL, DEFAULT_FILE,
            localllm::settings::load_active_model(),
        );
        cfg.model_id = model;
        cfg.gguf_files = files;
    }
```

Then use `cfg` (not `args.config`) everywhere below in `main`. Verify every later `args.config`/`cfg` reference is consistent — the file currently reads `args.config`; update those on the launch path to `cfg`.

> NOTE: keep `DEFAULT_MODEL`/`DEFAULT_FILE` in sync with the `#[arg(default_value …)]` in `src/config.rs`. If they drift, explicit-CLI detection breaks. A brief comment on the config defaults pointing here is worthwhile.

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: compiles clean.

- [ ] **Step 4: Manual verification**

Run the app, switch to a second model in the Manager, quit, relaunch. Expected: boots into the second model (tray "Model:" line shows it).

- [ ] **Step 5: Commit**

```bash
git add src/server.rs src/main.rs
git commit -m "feat(model): persist active model on switch, restore at boot"
```

---

## Phase 2 — Integrations API (shared; enables moved toggle)

### Task 3: `GET/POST /admin/integrations`

**Files:**
- Modify: `src/server.rs` (handlers + route table)
- Test: `tests/http.rs`

**Interfaces:**
- Produces:
  - `GET /admin/integrations` → `{ "enabled": bool, "wired": [String] }`
  - `POST /admin/integrations` body `{ "enabled": bool }` → same shape after applying.
- Consumes: `crate::integrations::{injectors_default, enable_all, disable_all}`, `crate::settings::{load_integrations, save_integrations, IntegrationState}`, `state.port`.

> **`AppState` has no `port` field today, and `enable_all(port, …)` needs it.** First add it:
> - Add `pub port: u16,` to `struct AppState`.
> - Add a `port: u16` parameter to `pub fn router(...)` and set `port` in the `AppState { … }` literal.
> - Update the real caller in `src/lib.rs` (~line 241) to pass `cfg.port`.
> - Update `router_for_test_with` (`src/lib.rs` ~line 450) to pass `31415` (any value; tests don't bind).
>
> **Test note:** only GET is tested (401 + response shape). Do **not** write an automated POST test — `POST /admin/integrations {enabled:true}` runs `enable_all`, which rewrites the developer's real agent client configs (Claude Code / Codex) on disk. The POST path is verified manually in Task 10.

- [ ] **Step 1: Add `port` then write the failing GET test**

First make the `AppState`/`router` changes above. Then add to `tests/http.rs` (mirrors the existing `admin_models_*` tests):

```rust
#[tokio::test]
async fn admin_integrations_get_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_get_status(app, "/admin/integrations").await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_integrations_get_reports_state_shape() {
    let app = localllm::router_for_test();
    let body = localllm::axum_test_get_with_header(
        app, "/admin/integrations", "x-admin-token", "test-token").await;
    assert!(body.get("enabled").and_then(|v| v.as_bool()).is_some());
    assert!(body.get("wired").map(|v| v.is_array()).unwrap_or(false));
}
```

`axum_test_get_status(app, path) -> u16` and `axum_test_get_with_header(app, path, hkey, hval) -> serde_json::Value` already exist in `src/lib.rs` — no new helper needed.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test http admin_integrations`
Expected: FAIL (route 404).

- [ ] **Step 3: Implement handlers**

In `src/server.rs`:

```rust
/// GET /admin/integrations — current wiring state (token-guarded).
async fn handle_integrations_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let st = crate::settings::load_integrations();
    let wired: Vec<String> = st.priors.keys().cloned().collect();
    Json(json!({ "enabled": st.enabled, "wired": wired })).into_response()
}

#[derive(serde::Deserialize)]
struct IntegrationsSetBody { enabled: bool }

/// POST /admin/integrations {enabled} — wire or unwire agent clients (token-guarded).
async fn handle_integrations_set(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: IntegrationsSetBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    };
    let injectors = crate::integrations::injectors_default();
    let new_state = if body.enabled {
        let outcome = crate::integrations::enable_all(state.port, &injectors);
        crate::settings::IntegrationState { enabled: !outcome.priors.is_empty(), priors: outcome.priors }
    } else {
        let mut st = crate::settings::load_integrations();
        let summary = crate::integrations::disable_all(&st.priors, &injectors);
        let failed: std::collections::HashSet<&String> = summary.failed.iter().map(|(id, _)| id).collect();
        st.priors.retain(|id, _| failed.contains(id));
        st.enabled = !st.priors.is_empty();
        st
    };
    let _ = crate::settings::save_integrations(&new_state);
    let wired: Vec<String> = new_state.priors.keys().cloned().collect();
    Json(json!({ "enabled": new_state.enabled, "wired": wired })).into_response()
}
```

Register in the route table (next to `/admin/tools`):

```rust
        .route("/admin/integrations", get(handle_integrations_get).post(handle_integrations_set))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test http admin_integrations`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): GET/POST /admin/integrations wiring state + toggle"
```

---

## Phase 3 — Routing log + Dashboard API (sub-project C backend)

### Task 4: `route_log` core — entry + append + prune (pure)

**Files:**
- Create: `src/route_log.rs`
- Modify: `src/lib.rs` (`pub mod route_log;`)
- Test: `src/route_log.rs::tests`

**Interfaces:**
- Produces:
  - `struct RouteEntry { ts: i64, surface: String, dest: String, reason: Option<String>, score: f64, prompt_tok: u64, completion_tok: Option<u64> }` (serde)
  - `fn log_path() -> Option<PathBuf>` (env `LOCALLLM_ROUTE_LOG` overrides; else `<config-dir>/localllm/routing-log.jsonl`)
  - `fn append(entry: &RouteEntry)` — best-effort, one JSON line.
  - `fn read_all() -> Vec<RouteEntry>` — skips malformed lines.
  - `fn prune(entries: &[RouteEntry], now: i64, max_age_secs: i64) -> Vec<RouteEntry>` (pure)
  - `fn prune_file(now: i64, max_age_secs: i64)` — read, prune, atomic rewrite; best-effort.

- [ ] **Step 1: Write the failing test**

Create `src/route_log.rs` with tests at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn entry(ts: i64, dest: &str) -> RouteEntry {
        RouteEntry { ts, surface: "openai".into(), dest: dest.into(),
            reason: None, score: 0.1, prompt_tok: 100, completion_tok: Some(20) }
    }

    #[test]
    fn prune_drops_entries_older_than_max_age() {
        let now = 1_000_000i64;
        let month = 30 * 24 * 3600;
        let kept = prune(&[entry(now - month - 1, "local"), entry(now - 10, "cloud")], now, month);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].dest, "cloud");
    }

    #[test]
    fn prune_keeps_everything_when_all_fresh() {
        let now = 1_000_000i64;
        let kept = prune(&[entry(now - 5, "local"), entry(now - 6, "cloud")], now, 100);
        assert_eq!(kept.len(), 2);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib route_log::tests::prune`
Expected: FAIL to compile (module not declared / symbols missing).

- [ ] **Step 3: Implement**

Top of `src/route_log.rs`:

```rust
//! Persistent per-request routing history (JSONL) with rolling retention.
//!
//! One JSON object per line at `<config-dir>/localllm/routing-log.jsonl`
//! (override with `LOCALLLM_ROUTE_LOG`). Writes are best-effort — a failure
//! never fails a request. Old lines are pruned on boot (see `prune_file`).

use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RouteEntry {
    /// Unix seconds.
    pub ts: i64,
    /// Client surface: "anthropic" | "openai" | "openai-responses".
    pub surface: String,
    /// "local" | "cloud".
    pub dest: String,
    /// RouteReason as a short string when dest == "cloud"; else None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub score: f64,
    pub prompt_tok: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tok: Option<u64>,
}

/// Resolve the log path. `LOCALLLM_ROUTE_LOG` (full file path) wins.
pub fn log_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LOCALLLM_ROUTE_LOG") {
        return Some(PathBuf::from(p));
    }
    dirs::config_dir().map(|d| d.join("localllm").join("routing-log.jsonl"))
}

/// Append one entry as a JSON line. Best-effort; never panics.
pub fn append(entry: &RouteEntry) {
    let Some(path) = log_path() else { return };
    let Ok(line) = serde_json::to_string(entry) else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Read every entry, skipping malformed lines. Empty when the file is absent.
pub fn read_all() -> Vec<RouteEntry> {
    let Some(path) = log_path() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    text.lines()
        .filter_map(|l| serde_json::from_str::<RouteEntry>(l).ok())
        .collect()
}

/// Pure retention filter: keep entries with `ts >= now - max_age_secs`.
pub fn prune(entries: &[RouteEntry], now: i64, max_age_secs: i64) -> Vec<RouteEntry> {
    let cutoff = now - max_age_secs;
    entries.iter().filter(|e| e.ts >= cutoff).cloned().collect()
}

/// Read, prune, and atomically rewrite the log. Best-effort.
pub fn prune_file(now: i64, max_age_secs: i64) {
    let Some(path) = log_path() else { return };
    let kept = prune(&read_all(), now, max_age_secs);
    let mut buf = String::new();
    for e in &kept {
        if let Ok(l) = serde_json::to_string(e) { buf.push_str(&l); buf.push('\n'); }
    }
    let _ = crate::integrations::atomic_write(&path, buf.as_bytes());
}

/// Current unix seconds.
pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
```

Add to `src/lib.rs` module list: `pub mod route_log;`

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib route_log::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/route_log.rs src/lib.rs
git commit -m "feat(route_log): JSONL routing history with retention prune"
```

### Task 5: Rollup + Dashboard aggregation (pure)

**Files:**
- Modify: `src/route_log.rs`
- Test: `src/route_log.rs::tests`

**Interfaces:**
- Produces:
  - `struct Bucket { local_count: u64, cloud_count: u64, tokens_saved: u64, tokens_if_all_cloud: u64 }` (serde)
  - `struct Dashboard { hour: Bucket, day: Bucket, month: Bucket, recent: Vec<RouteEntry> }` (serde)
  - `fn build_dashboard(entries: &[RouteEntry], now: i64, recent_n: usize) -> Dashboard`

Bucket windows are rolling: hour = last 3600s, day = last 86400s, month = last 2_592_000s.

- [ ] **Step 1: Write the failing test**

Add to `route_log::tests`:

```rust
fn e(ts: i64, dest: &str, p: u64, c: u64) -> RouteEntry {
    RouteEntry { ts, surface: "openai".into(), dest: dest.into(),
        reason: None, score: 0.1, prompt_tok: p, completion_tok: Some(c) }
}

#[test]
fn dashboard_buckets_and_token_math() {
    let now = 10_000_000i64;
    let entries = vec![
        e(now - 10, "local", 100, 20),      // in hour/day/month
        e(now - 7200, "cloud", 200, 50),    // in day/month, not hour
        e(now - 200_000, "local", 300, 30), // in month only
    ];
    let d = build_dashboard(&entries, now, 10);
    // hour: only the first local entry
    assert_eq!(d.hour.local_count, 1);
    assert_eq!(d.hour.cloud_count, 0);
    assert_eq!(d.hour.tokens_saved, 120);
    assert_eq!(d.hour.tokens_if_all_cloud, 120);
    // day: local(120) + cloud(250)
    assert_eq!(d.day.local_count, 1);
    assert_eq!(d.day.cloud_count, 1);
    assert_eq!(d.day.tokens_saved, 120);           // only local counts as saved
    assert_eq!(d.day.tokens_if_all_cloud, 370);    // all requests
    // month: all three
    assert_eq!(d.month.tokens_saved, 120 + 330);   // two local
    assert_eq!(d.month.tokens_if_all_cloud, 120 + 250 + 330);
    // recent newest-first
    assert_eq!(d.recent.len(), 3);
    assert_eq!(d.recent[0].ts, now - 10);
}

#[test]
fn dashboard_recent_is_capped() {
    let now = 100i64;
    let entries: Vec<RouteEntry> = (0..20).map(|i| e(now - i, "local", 1, 1)).collect();
    let d = build_dashboard(&entries, now, 5);
    assert_eq!(d.recent.len(), 5);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib route_log::tests::dashboard`
Expected: FAIL (symbols missing).

- [ ] **Step 3: Implement**

Add to `src/route_log.rs`:

```rust
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Bucket {
    pub local_count: u64,
    pub cloud_count: u64,
    /// Σ(prompt+completion) of local requests — tokens NOT sent to the provider.
    pub tokens_saved: u64,
    /// Σ(prompt+completion) of ALL requests — the hypothetical all-cloud cost.
    pub tokens_if_all_cloud: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub hour: Bucket,
    pub day: Bucket,
    pub month: Bucket,
    /// Newest-first, capped at `recent_n`.
    pub recent: Vec<RouteEntry>,
}

const HOUR: i64 = 3600;
const DAY: i64 = 86_400;
const MONTH: i64 = 2_592_000; // 30 days

fn accumulate(bucket: &mut Bucket, e: &RouteEntry) {
    let toks = e.prompt_tok + e.completion_tok.unwrap_or(0);
    bucket.tokens_if_all_cloud += toks;
    if e.dest == "local" {
        bucket.local_count += 1;
        bucket.tokens_saved += toks;
    } else {
        bucket.cloud_count += 1;
    }
}

pub fn build_dashboard(entries: &[RouteEntry], now: i64, recent_n: usize) -> Dashboard {
    let (mut hour, mut day, mut month) = (Bucket::default(), Bucket::default(), Bucket::default());
    for e in entries {
        let age = now - e.ts;
        if age <= MONTH { accumulate(&mut month, e); }
        if age <= DAY { accumulate(&mut day, e); }
        if age <= HOUR { accumulate(&mut hour, e); }
    }
    let mut recent: Vec<RouteEntry> = entries.to_vec();
    recent.sort_by(|a, b| b.ts.cmp(&a.ts));
    recent.truncate(recent_n);
    Dashboard { hour, day, month, recent }
}
```

Add `use serde::{Serialize, Deserialize};` at the top (or fully-qualify as above — keep consistent with the file's existing derive style).

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib route_log::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/route_log.rs
git commit -m "feat(route_log): dashboard rollups (hour/day/month + recent)"
```

### Task 6: Write log entry at the routing decision point

**Files:**
- Modify: `src/server.rs` (`route_decision`)

**Interfaces:**
- Consumes: `route_log::{RouteEntry, append, now_secs}`.
- `route_decision` must learn the surface. Add a `surface: &str` param and pass `"anthropic"`, `"openai"`, `"openai-responses"` from the three call sites.

- [ ] **Step 1: Add surface param + write**

Change the signature:

```rust
fn route_decision(
    state: &AppState,
    internal: &ChatRequest,
    headers: &axum::http::HeaderMap,
    rid: &str,
    surface: &str,
) -> (crate::route::Decision, usize) {
```

At the end of `route_decision`, before the return, map decision → dest/reason and append (completion tokens unknown at decision time → `None`):

```rust
    let (dest, reason) = match decision {
        crate::route::Decision::Cloud(r) => ("cloud", Some(format!("{r:?}"))),
        _ => ("local", None),
    };
    crate::route_log::append(&crate::route_log::RouteEntry {
        ts: crate::route_log::now_secs(),
        surface: surface.to_string(),
        dest: dest.to_string(),
        reason,
        score,
        prompt_tok: prompt_tokens as u64,
        completion_tok: None,
    });

    (decision, prompt_tokens)
```

Update the three call sites (search `route_decision(&state, &internal, &headers, &rid)`):
- anthropic handler → add `, "anthropic"`
- openai chat handler → add `, "openai"`
- openai responses handler → add `, "openai-responses"`

> DESIGN NOTE (known limitation): `dest` reflects the *decision*. When a cloud decision degrades to local (cascade/degrade path), the log still reads "cloud". This matches "where it was routed / why", and keeps a single write site. Completion tokens are not backfilled in v1 (streaming has no single completion count); the dashboard token math is therefore prompt-dominant. Revisit during the Dashboard UI co-design if finer accuracy is wanted.

- [ ] **Step 2: Build + existing tests**

Run: `cargo build && cargo test --test http`
Expected: compiles; existing HTTP tests still pass.

- [ ] **Step 3: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): record routing decision to route_log"
```

### Task 7: `GET /admin/dashboard`

**Files:**
- Modify: `src/server.rs` (handler + route)
- Test: `tests/http.rs`

**Interfaces:**
- Produces: `GET /admin/dashboard` → `Dashboard` JSON (`{hour, day, month, recent}`).

- [ ] **Step 1: Write the failing test**

Add to `tests/http.rs`:

```rust
#[tokio::test]
async fn admin_dashboard_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_get_status(app, "/admin/dashboard").await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_dashboard_returns_shape() {
    let app = localllm::router_for_test();
    let body = localllm::axum_test_get_with_header(
        app, "/admin/dashboard", "x-admin-token", "test-token").await;
    for k in ["hour", "day", "month", "recent"] {
        assert!(body.get(k).is_some(), "missing {k}");
    }
    assert!(body["hour"].get("tokens_saved").is_some());
}
```

> The GET reads the routing log from the real config dir; with no log present `build_dashboard` returns zeroed buckets, so the shape assertions hold regardless. Set `LOCALLLM_ROUTE_LOG` to a temp path if you want a hermetic data assertion.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test http admin_dashboard`
Expected: FAIL (404).

- [ ] **Step 3: Implement**

```rust
/// GET /admin/dashboard — routing rollups + recent decisions (token-guarded).
async fn handle_dashboard(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let entries = crate::route_log::read_all();
    let dash = crate::route_log::build_dashboard(&entries, crate::route_log::now_secs(), 50);
    Json(dash).into_response()
}
```

Register:

```rust
        .route("/admin/dashboard", get(handle_dashboard))
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --test http admin_dashboard`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs tests/http.rs
git commit -m "feat(server): GET /admin/dashboard rollups endpoint"
```

### Task 8: Prune logs at boot

**Files:**
- Modify: `src/lib.rs` (`run_server_with_ready_policy_token`)

**Interfaces:**
- Consumes: `route_log::{prune_file, now_secs}`.

- [ ] **Step 1: Implement**

Near the top of `run_server_with_ready_policy_token` (after the RAM probe, before/after model load — order does not matter, keep it early), add:

```rust
    // Rolling retention: drop routing-log lines older than ~30 days at boot.
    crate::route_log::prune_file(crate::route_log::now_secs(), 30 * 24 * 3600);
    // App-log rotation: keep ~7 days of the plain-text log (see rotate_app_log).
    crate::route_log::rotate_app_log(crate::route_log::now_secs(), 7 * 24 * 3600);
```

- [ ] **Step 2: Implement `rotate_app_log`**

Add to `src/route_log.rs` (the app log is line-oriented `tracing` text; we can only prune by a leading RFC3339/`tracing` timestamp when present — keep it simple and cap by line count as a robust fallback):

```rust
/// Best-effort app-log rotation: cap the plain-text app log at a line budget so
/// it cannot grow unbounded. `LOCALLLM_LOG` (default `/tmp/localllm.log`).
/// `_max_age_secs` is accepted for symmetry but line-count capping is the
/// robust mechanism (the app log has no guaranteed machine-parsable timestamp).
pub fn rotate_app_log(_now: i64, _max_age_secs: i64) {
    const MAX_LINES: usize = 50_000;
    let path = std::env::var("LOCALLLM_LOG").unwrap_or_else(|_| "/tmp/localllm.log".to_string());
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= MAX_LINES { return; }
    let tail = lines[lines.len() - MAX_LINES..].join("\n");
    let _ = crate::integrations::atomic_write(std::path::Path::new(&path), format!("{tail}\n").as_bytes());
}
```

> The spec asked for ~1 week retention on the app log; because the app log lacks a guaranteed parsable timestamp per line, we cap by line count (bounded size), which satisfies the underlying goal (don't keep unbounded history). If precise time-based pruning is later required, switch the app log to JSONL too.

- [ ] **Step 3: Build**

Run: `cargo build && cargo test --lib route_log::`
Expected: compiles; tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/route_log.rs
git commit -m "feat(boot): prune routing log + rotate app log on startup"
```

---

## Phase 4 — SPA: Config hub, integrations toggle, Dashboard (sub-projects A + C UI)

> All Phase 4 UI tasks: **use the frontend-design skill** for markup/CSS quality. There is no JS test runner in this repo; verify by launching the app (tray → Config) and exercising the flows. Keep the existing vanilla-JS, no-build style.

### Task 9: Hash router + Config landing

**Files:**
- Modify: `src/manager_ui/app.js`
- Modify: `src/manager_ui/style.css`

**Interfaces:**
- Produces: `renderConfig()`, `navigate(route)`, a `hashchange` listener mapping `#/config #/models #/tools #/dashboard` to render fns. `#/models` → `renderFamilies`, `#/tools` → `renderTools`, `#/dashboard` → `renderDashboard` (Task 15 stub until then), default/`#/config` → `renderConfig`.

- [ ] **Step 1: Add the router**

In `app.js`, add near the boot section:

```js
function routeFromHash() {
  const h = (location.hash || "#/config").replace(/^#/, "");
  if (h.startsWith("/models")) return renderFamilies();
  if (h.startsWith("/tools")) return renderTools();
  if (h.startsWith("/dashboard")) return renderDashboard();
  return renderConfig();
}
function navigate(route) {
  if (location.hash === "#" + route) routeFromHash();
  else location.hash = route; // triggers hashchange → routeFromHash
}
window.addEventListener("hashchange", () => routeFromHash());
```

- [ ] **Step 2: Add `renderConfig` landing**

```js
// ---- Pane 0: Config landing ----
function renderConfig() {
  currentView = renderConfig;
  setCrumbs([{ label: "Config" }]);
  const grid = el("div", "grid");

  const modelsCard = el("div", "card");
  modelsCard.append(el("h3", null, "Models"));
  modelsCard.append(el("div", "meta", "Escolher, baixar e configurar modelos locais"));
  modelsCard.onclick = () => navigate("/models");
  grid.append(modelsCard);

  const toolsCard = el("div", "card");
  toolsCard.append(el("h3", null, "Tools"));
  toolsCard.append(el("div", "meta", "Filtrar tools por cliente"));
  toolsCard.onclick = () => navigate("/tools");
  grid.append(toolsCard);

  const dashCard = el("div", "card");
  dashCard.append(el("h3", null, "Dashboard"));
  dashCard.append(el("div", "meta", "Roteamento local↔cloud e tokens economizados"));
  dashCard.onclick = () => navigate("/dashboard");
  grid.append(dashCard);

  view.innerHTML = "";
  view.append(grid);
  renderIntegrationToggle(view); // Task 10 (define a no-op stub first if implementing out of order)
}
```

- [ ] **Step 3: Update breadcrumb roots + boot**

- In `renderFamilies`, `renderModels`, `renderDetail`, `renderTools`, change the first breadcrumb from `{ label: "Models", onClick: renderFamilies }` to `{ label: "Config", onClick: renderConfig }` (keep a "Models" crumb where the pane is a model pane, e.g. `renderModels` → `[Config, Models, family]`). Preserve existing deeper crumbs.
- Change `boot()` to call `routeFromHash()` instead of `renderFamilies()`.

- [ ] **Step 4: Manual verification**

Launch app → tray → Config. Expected: landing shows Models/Tools/Dashboard cards; clicking each updates the hash and view; browser-style back via hash works; breadcrumb "Config" returns to landing.

- [ ] **Step 5: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): config landing + hash router"
```

### Task 10: Integrations toggle in Config

**Files:**
- Modify: `src/manager_ui/app.js`
- Modify: `src/manager_ui/style.css`

**Interfaces:**
- Consumes: `GET/POST /admin/integrations` (Task 3).
- Produces: `renderIntegrationToggle(container)`.

- [ ] **Step 1: Implement**

```js
async function renderIntegrationToggle(container) {
  const box = el("div", "ctxbox");
  box.append(el("div", "ctxtitle", "Rotear apps pelo localllm"));
  box.append(el("div", "ctxhint",
    "Liga/desliga o roteamento dos clientes (Claude Code, Codex) por este servidor. Ao sair do localllm, o roteamento é removido automaticamente."));
  const status = el("div", "ctxhint", "carregando…");
  const btn = el("button", "btn primary", "…");
  btn.disabled = true;
  let enabled = false;

  const paint = (st) => {
    enabled = !!st.enabled;
    btn.textContent = enabled ? "Desligar" : "Ligar";
    btn.disabled = false;
    const wired = (st.wired || []).join(", ");
    status.textContent = enabled
      ? `Ligado — wired: ${wired || "nenhum cliente encontrado"}`
      : "Desligado — apps vão direto ao provider";
  };

  try { paint(await api("GET", "/admin/integrations")); }
  catch (e) { status.textContent = e.message; }

  btn.onclick = async () => {
    btn.disabled = true;
    try { paint(await api("POST", "/admin/integrations", { enabled: !enabled })); toast("Integração atualizada"); }
    catch (e) { toast(e.message, true); btn.disabled = false; }
  };

  const actions = el("div", "actions");
  actions.append(btn);
  box.append(status, actions);
  container.append(box);
}
```

- [ ] **Step 2: Manual verification**

Config landing shows the toggle with live state; Ligar/Desligar flips it and updates the wired list; the tray wired line reflects the change on its next poll (Task 13).

- [ ] **Step 3: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): route-apps toggle on config page"
```

---

## Phase 5 — Tray restructure + blur-hide + Quit cleanup (sub-projects B/D)

### Task 11: Config submenu + navigate-to-route

**Files:**
- Modify: `src/tray.rs`

**Interfaces:**
- Produces: a `Config` submenu with `Models`/`Tools`/`Dashboard` items; each opens/show-focuses the shared window and navigates it to `#/models`/`#/tools`/`#/dashboard`.
- The window struct gains `fn navigate(&self, route: &str)` that evals `location.hash`.

- [ ] **Step 1: Add navigate to the window**

In `mod window`, add to `ModelManagerWindow`:

```rust
    /// Navigate the SPA to a hash route (e.g. "#/dashboard") without a reload.
    pub fn navigate(&self, route: &str) {
        let js = format!("location.hash = {:?};", route);
        let _ = self._webview.evaluate_script(&js);
    }
```

`_webview` is currently prefixed `_` (unused). Rename the field to `webview` and update the struct + `open` accordingly so `navigate` can use it.

- [ ] **Step 2: Replace the manager item with a Config submenu**

In the `Init` arm, replace the single `manager_item` with:

```rust
                let config_submenu = Submenu::new("Config", true);
                let cfg_models = MenuItem::new("Models", true, None);
                let cfg_tools = MenuItem::new("Tools", true, None);
                let cfg_dash = MenuItem::new("Dashboard", true, None);
                config_models_id = Some(cfg_models.id().clone());
                config_tools_id = Some(cfg_tools.id().clone());
                config_dash_id = Some(cfg_dash.id().clone());
                config_submenu.append(&cfg_models).expect("append models");
                config_submenu.append(&cfg_tools).expect("append tools");
                config_submenu.append(&cfg_dash).expect("append dashboard");
```

Append `config_submenu` where `manager_item` was appended. Declare the three ids beside the other `Option<MenuId>` state:

```rust
    let mut config_models_id: Option<tray_icon::menu::MenuId> = None;
    let mut config_tools_id: Option<tray_icon::menu::MenuId> = None;
    let mut config_dash_id: Option<tray_icon::menu::MenuId> = None;
```

- [ ] **Step 3: Handle the three items**

Add a helper closure and dispatch. Replace the old `manager_id` click arm with:

```rust
                    } else if config_models_id.as_ref() == Some(&menu_event.id)
                        || config_tools_id.as_ref() == Some(&menu_event.id)
                        || config_dash_id.as_ref() == Some(&menu_event.id)
                    {
                        let route = if config_models_id.as_ref() == Some(&menu_event.id) { "#/models" }
                            else if config_tools_id.as_ref() == Some(&menu_event.id) { "#/tools" }
                            else { "#/dashboard" };
                        match &manager_window {
                            Some(w) => {
                                w.window.set_visible(true);
                                w.window.set_focus();
                                w.navigate(route);
                            }
                            None => match window::ModelManagerWindow::open(target, port, &admin_token) {
                                Ok(w) => { w.navigate(route); manager_window = Some(w); }
                                Err(e) => tracing::error!("Config window failed: {e}"),
                            },
                        }
                    }
```

> Opening fresh then immediately `navigate` may race the initial page load. Acceptable: the SPA reads `location.hash` on boot (`routeFromHash`), and `navigate` sets the hash; if the eval lands before the script, the boot read still routes correctly. If flaky in practice, pass the route via the initial URL (`…/manager#{route}`) in `open`.

Remove `manager_id`/`manager_item` declarations and their old handler arm.

- [ ] **Step 4: Build + manual**

Run: `cargo build`
Launch: tray shows a Config submenu; each item opens the window at the right section.

- [ ] **Step 5: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): config submenu deep-links window routes"
```

### Task 12: Remove toggle from tray; wired line becomes read-only live

**Files:**
- Modify: `src/tray.rs`

- [ ] **Step 1: Remove the CheckMenuItem toggle**

Delete `toggle_item`/`toggle_id`/`toggle_item_handle` creation, the `menu.append(&toggle_item)`, and the `toggle_id` click-handler arm. Keep `wired_line`/`wired_handle` and the `wired_label` helper.

- [ ] **Step 2: Refresh the wired line each poll tick from live state**

In the `MainEventsCleared`/`ResumeTimeReached` arm (where status/model are refreshed), add:

```rust
                // Keep the read-only wired line in sync with the SPA toggle.
                {
                    let st = crate::settings::load_integrations();
                    let label = wired_label(&st);
                    if last_wired.as_deref() != Some(label.as_str()) {
                        if let Some(line) = &wired_handle { line.set_text(&label); }
                        last_wired = Some(label);
                    }
                }
```

Declare `let mut last_wired: Option<String> = None;` beside the other `last_*` state.

- [ ] **Step 3: Build + manual**

Run: `cargo build`. Toggle in the SPA (Task 10) → within ~100ms the tray wired line updates. No toggle item remains in the tray.

- [ ] **Step 4: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): drop toggle; wired line is read-only live state"
```

### Task 13: Un-wire integrations on Quit

**Files:**
- Modify: `src/tray.rs` (Quit handler)

- [ ] **Step 1: Implement**

Replace the Quit arm body:

```rust
                    if quit_id.as_ref() == Some(&menu_event.id) {
                        tracing::info!("quit requested via tray menu — unwiring integrations");
                        let st = crate::settings::load_integrations();
                        if st.enabled {
                            let injectors = crate::integrations::injectors_default();
                            let summary = crate::integrations::disable_all(&st.priors, &injectors);
                            let failed: std::collections::HashSet<&String> =
                                summary.failed.iter().map(|(id, _)| id).collect();
                            let mut new_state = st.clone();
                            new_state.priors.retain(|id, _| failed.contains(id));
                            new_state.enabled = !new_state.priors.is_empty();
                            let _ = crate::settings::save_integrations(&new_state);
                        }
                        std::process::exit(0);
                    } else if ...
```

- [ ] **Step 2: Build + manual**

Wire apps via Config, confirm the client config points at localllm, Quit, then check the client config reverted to provider-direct.

- [ ] **Step 3: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): unwire integrations on quit"
```

### Task 14: Hide window on blur

**Files:**
- Modify: `src/tray.rs` (event loop)

- [ ] **Step 1: Implement**

Extend the existing `Event::WindowEvent` handling. Alongside the `CloseRequested` arm, handle focus loss:

```rust
            Event::WindowEvent {
                event: tao::event::WindowEvent::Focused(false),
                window_id,
                ..
            } => {
                if let Some(w) = &manager_window {
                    if w.window.id() == window_id {
                        w.window.set_visible(false);
                    }
                }
            }
```

(Keep the existing `CloseRequested` arm.)

- [ ] **Step 2: Build + manual**

Open Config, click another app (blur). Expected: window hides. Reopen from tray → reappears at the last/selected route. Confirm the SPA's `visibilitychange` pause still holds (no console errors).

> If transient blurs (e.g. the tray menu itself) hide the window annoyingly, note it for follow-up; behavior is per the approved spec.

- [ ] **Step 3: Commit**

```bash
git add src/tray.rs
git commit -m "feat(tray): hide config window on blur"
```

---

## Phase 6 — Dashboard SPA view (co-designed)

### Task 15: Dashboard view (`#/dashboard`)

**Files:**
- Modify: `src/manager_ui/app.js`
- Modify: `src/manager_ui/style.css`

**Interfaces:**
- Consumes: `GET /admin/dashboard` (Task 7) → `{hour, day, month, recent}`.
- Produces: `renderDashboard()`.

> **CO-DESIGN GATE (per spec):** before building this view, run the frontend-design skill and the brainstorming visual companion WITH THE USER to settle the layout (cards vs table vs charts, what to headline). The data contract is fixed; only the presentation is open. Do not finalize this task without that step.

- [ ] **Step 1: Minimal functional stub (unblocks routing before co-design)**

So the router (Task 9) has a target, add a stub that will be replaced after co-design:

```js
async function renderDashboard() {
  currentView = renderDashboard;
  setCrumbs([{ label: "Config", onClick: renderConfig }, { label: "Dashboard" }]);
  let d;
  try { d = await api("GET", "/admin/dashboard"); }
  catch (e) { view.innerHTML = ""; view.append(el("div", "detail", e.message)); return; }
  const wrap = el("div", "detail");
  wrap.append(el("h2", null, "Dashboard"));
  ["hour", "day", "month"].forEach((k) => {
    const b = d[k];
    const box = el("div", "ctxbox");
    box.append(el("div", "ctxtitle", { hour: "Última hora", day: "Hoje", month: "Este mês" }[k]));
    box.append(el("div", "ctxhint",
      `local ${b.local_count} · cloud ${b.cloud_count} · tokens salvos ${b.tokens_saved} · se tudo cloud ${b.tokens_if_all_cloud}`));
    wrap.append(box);
  });
  const rec = el("div", "ctxbox");
  rec.append(el("div", "ctxtitle", "Recentes"));
  (d.recent || []).forEach((e) => {
    rec.append(el("div", "toolrow",
      `${e.surface} → ${e.dest}${e.reason ? " (" + e.reason + ")" : ""} · score ${e.score.toFixed(2)} · ${e.prompt_tok} tok`));
  });
  wrap.append(rec);
  view.innerHTML = ""; view.append(wrap);
}
```

- [ ] **Step 2: Manual verification of the stub**

Send a few requests through the server (local + cloud), open tray → Config → Dashboard. Expected: buckets populate; recent list shows surface/dest/reason/score/tokens.

- [ ] **Step 3: Co-design the real layout**

Run frontend-design + visual companion with the user. Replace the stub markup/CSS with the agreed design. Keep the same `GET /admin/dashboard` data.

- [ ] **Step 4: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): dashboard view (routing + tokens saved)"
```

---

## Self-Review (completed against the spec)

- **A — Config landing + nav:** Tasks 9, 10 (landing, router, breadcrumbs, moved toggle). ✅
- **B — Tray + model persistence + Quit cleanup:** Tasks 1, 2 (persist/restore model), 11 (Config submenu), 12 (read-only wired line), 13 (unwire on quit). ✅
- **C — Dashboard:** Tasks 4–8 (log core, rollups, write site, endpoint, boot prune + app-log rotation), 15 (view, co-designed). ✅
- **D — Blur hide:** Task 14. ✅
- **Integrations API:** Task 3. ✅
- **App-log rotation:** Task 8 (line-cap fallback; deviation from time-based noted inline, goal met). ✅
- **Placeholders:** none — every code step has concrete code. The Dashboard *visual* is an intentional co-design gate (stub provided so routing isn't blocked), not a plan placeholder.
- **Type consistency:** `ActiveModel`, `RouteEntry`, `Bucket`, `Dashboard`, `resolve_active_model`, `build_dashboard`, `prune`, `append`, `navigate` are used with consistent signatures across tasks.
- **Known limitations documented:** decision-vs-outcome `dest` on degrade; completion-token omission in v1; single-file saved-model restore; line-cap app-log rotation.
