# Model Catalog + Recommendation + Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a curated model catalog annotated with per-model status, estimated RAM, fit verdict, and a single recommendation, plus delete-a-downloaded-model, over token-guarded localhost endpoints.

**Architecture:** A pure `catalog` module holds a compiled-in `CATALOG` and a pure `catalog_view(entries, total_ram_mb, active, is_downloaded)` that annotates each entry — no HTTP, no `sysinfo`, no filesystem (all injected). The server reads total RAM once at startup (via `sysinfo`), and exposes `GET /admin/models` (annotated tree) and `DELETE /admin/models` (free disk; refuses the in-use model), both token-guarded like the existing `/admin/model`.

**Tech Stack:** Rust, axum 0.7, serde, sysinfo; dev tests headless.

## Global Constraints

- Builds on sub-project 1 (branch `feat/model-hotswap`): `ModelManager.status() -> SwitchStatus { current: ModelSpec, … }`; `/admin/*` guarded by `check_admin(&headers, &state) -> Option<Response>` (returns `Some(401)` to reject); `AppState` currently has 7 fields and `router(...)` 7 params; `download::cache_path(repo,file) -> PathBuf` (existence = downloaded).
- Catalog is **embedded/curated/offline** — no network to list models. The default model `Qwen/Qwen2.5-3B-Instruct-GGUF` / `qwen2.5-3b-instruct-q4_k_m.gguf` MUST be a catalog entry so the active model always appears.
- **Recommendation:** largest model whose `est_ram_mb ≤ budget`; budget = `total_ram_mb * 65 / 100`. Exactly one global `recommended` (or none if catalog empty); if nothing fits → smallest `est_ram_mb`.
- **RAM estimate:** `est_ram_mb = round(size_mb * 1.2)`.
- **Fit:** `est ≤ budget → Fits`; `budget < est ≤ total*85/100 → Tight`; else `WontFit`.
- **Delete:** refuse the in-use model with **409**; not-cached → `200 {"deleted":false}`.
- New dep `sysinfo`. RAM read once at startup; tests inject a fixed value (16384).
- `/admin/*` token-guarded; `/v1/*` unchanged. TDD; complete code each step; commit per task; pristine build.

---

### Task 1: `catalog` module — data + pure `catalog_view`

**Files:**
- Create: `src/catalog.rs`
- Modify: `src/lib.rs` (`pub mod catalog;`)
- Test: `src/catalog.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `crate::model_manager::ModelSpec`.
- Produces: `CatalogEntry`, `CATALOG: &[CatalogEntry]`, `ModelStatus`, `FitVerdict`, `ModelView`, `FamilyView`, and `fn catalog_view(entries: &[CatalogEntry], total_ram_mb: u64, active: Option<&ModelSpec>, is_downloaded: impl Fn(&str,&str)->bool) -> Vec<FamilyView>`.

- [ ] **Step 1: Register the module**

In `src/lib.rs`, after `pub mod catalog;`’s neighbours (add near `pub mod model_manager;`):

```rust
pub mod catalog;
```

- [ ] **Step 2: Write the failing tests**

Create `src/catalog.rs`:

```rust
//! Curated, embedded model catalog + pure annotation (status / RAM / fit /
//! recommendation). No HTTP, no sysinfo, no filesystem — the caller injects
//! total RAM, the active model, and a downloaded-probe, so this is fully
//! unit-testable.

use crate::model_manager::ModelSpec;

/// One curated model the user can switch to.
pub struct CatalogEntry {
    pub family: &'static str,
    pub display_name: &'static str,
    pub params: &'static str,
    pub params_b: f32,
    pub quant: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size_mb: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    InUse,
    Downloaded,
    NeedsDownload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    Fits,
    Tight,
    WontFit,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelView {
    pub display_name: String,
    pub params: String,
    pub quant: String,
    pub repo: String,
    pub file: String,
    pub size_mb: u32,
    pub est_ram_mb: u32,
    pub status: ModelStatus,
    pub fit: FitVerdict,
    pub recommended: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FamilyView {
    pub family: String,
    pub models: Vec<ModelView>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(family: &'static str, name: &'static str, pb: f32, repo: &'static str, file: &'static str, size_mb: u32) -> CatalogEntry {
        CatalogEntry { family, display_name: name, params: "x", params_b: pb, quant: "Q4_K_M", repo, file, size_mb }
    }

    fn sample() -> Vec<CatalogEntry> {
        vec![
            entry("Qwen2.5", "Qwen 3B", 3.0, "q/3b", "3b.gguf", 2000),
            entry("Qwen2.5", "Qwen 7B", 7.0, "q/7b", "7b.gguf", 4700),
            entry("Qwen2.5", "Qwen 32B", 32.0, "q/32b", "32b.gguf", 20000),
            entry("Llama", "Llama 8B", 8.0, "l/8b", "8b.gguf", 4900),
        ]
    }

    #[test]
    fn annotates_status_fit_and_single_recommendation() {
        // 16 GB machine → budget 10649 MB; total*0.85 = 13926 MB.
        // est = size*1.2: 3B→2400 fits, 7B→5640 fits, 8B→5880 fits,
        // 32B→24000 wont_fit. Largest-that-fits = 8B (params_b 8 > 7).
        let cat = sample();
        let active = ModelSpec { repo: "q/7b".into(), file: "7b.gguf".into() };
        let downloaded = |r: &str, _f: &str| r == "q/3b"; // 3B cached
        let view = catalog_view(&cat, 16384, Some(&active), downloaded);

        // grouped by family in first-seen order
        assert_eq!(view[0].family, "Qwen2.5");
        assert_eq!(view[1].family, "Llama");

        // statuses
        let qwen = &view[0].models;
        assert_eq!(qwen[0].status, ModelStatus::Downloaded);   // 3B cached
        assert_eq!(qwen[1].status, ModelStatus::InUse);        // 7B active
        assert_eq!(qwen[2].status, ModelStatus::NeedsDownload);// 32B

        // est ram + fit
        assert_eq!(qwen[0].est_ram_mb, 2400);
        assert_eq!(qwen[0].fit, FitVerdict::Fits);
        assert_eq!(qwen[2].fit, FitVerdict::WontFit);          // 32B

        // exactly one recommended, and it's the 8B (largest that fits)
        let recs: Vec<_> = view.iter().flat_map(|f| f.models.iter()).filter(|m| m.recommended).collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].display_name, "Llama 8B");
    }

    #[test]
    fn tiny_ram_recommends_smallest() {
        let cat = sample();
        // 2 GB → budget 1331; nothing fits → smallest est_ram (3B → 2400).
        let view = catalog_view(&cat, 2048, None, |_, _| false);
        let recs: Vec<_> = view.iter().flat_map(|f| f.models.iter()).filter(|m| m.recommended).collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].display_name, "Qwen 3B");
        // and a non-fitting model is marked won't_fit
        assert!(view.iter().flat_map(|f| f.models.iter()).any(|m| m.fit == FitVerdict::WontFit));
    }

    #[test]
    fn tight_band_between_budget_and_85_percent() {
        // total 10000 → budget 6500, 85% = 8500. Pick est in (6500, 8500].
        // size 6000 → est 7200 → Tight.
        let cat = vec![entry("F", "M", 5.0, "r", "f", 6000)];
        let view = catalog_view(&cat, 10000, None, |_, _| false);
        assert_eq!(view[0].models[0].fit, FitVerdict::Tight);
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test --lib catalog:: 2>&1 | head -20`
Expected: FAIL — `catalog_view` (and `CATALOG`) not defined.

- [ ] **Step 4: Implement `catalog_view` + the seed `CATALOG`**

In `src/catalog.rs`, above the test module, add the curated catalog and the pure annotator:

```rust
/// Curated models, smallest→largest within each family. Sizes are approximate
/// download sizes (MB) used only for the RAM estimate; verify against the repo
/// when editing. The default model MUST appear here.
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 3B Instruct", params: "3B", params_b: 3.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-3B-Instruct-GGUF", file: "qwen2.5-3b-instruct-q4_k_m.gguf", size_mb: 2000 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 7B Instruct", params: "7B", params_b: 7.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-7B-Instruct-GGUF", file: "qwen2.5-7b-instruct-q4_k_m.gguf", size_mb: 4700 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 14B Instruct", params: "14B", params_b: 14.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-14B-Instruct-GGUF", file: "qwen2.5-14b-instruct-q4_k_m.gguf", size_mb: 9000 },
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 32B Instruct", params: "32B", params_b: 32.0, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-32B-Instruct-GGUF", file: "qwen2.5-32b-instruct-q4_k_m.gguf", size_mb: 20000 },
    CatalogEntry { family: "Llama 3.1", display_name: "Llama 3.1 8B Instruct", params: "8B", params_b: 8.0, quant: "Q4_K_M",
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", size_mb: 4900 },
    CatalogEntry { family: "Phi 3.5", display_name: "Phi 3.5 Mini Instruct", params: "3.8B", params_b: 3.8, quant: "Q4_K_M",
        repo: "bartowski/Phi-3.5-mini-instruct-GGUF", file: "Phi-3.5-mini-instruct-Q4_K_M.gguf", size_mb: 2400 },
];

/// Annotate the catalog for this machine + the active model. Pure.
pub fn catalog_view(
    entries: &[CatalogEntry],
    total_ram_mb: u64,
    active: Option<&ModelSpec>,
    is_downloaded: impl Fn(&str, &str) -> bool,
) -> Vec<FamilyView> {
    let budget = total_ram_mb * 65 / 100;
    let tight_ceiling = total_ram_mb * 85 / 100;

    // First pass: build ModelView (without recommendation) and track the best
    // recommendation candidate index in the flattened order.
    let mut flat: Vec<ModelView> = Vec::with_capacity(entries.len());
    let mut best_fit: Option<usize> = None; // largest params_b among Fits
    let mut smallest: Option<usize> = None; // fallback: smallest est_ram

    for (i, e) in entries.iter().enumerate() {
        let est = (e.size_mb as f32 * 1.2).round() as u32;
        let fit = if (est as u64) <= budget {
            FitVerdict::Fits
        } else if (est as u64) <= tight_ceiling {
            FitVerdict::Tight
        } else {
            FitVerdict::WontFit
        };
        let status = if active.map(|a| a.repo == e.repo && a.file == e.file).unwrap_or(false) {
            ModelStatus::InUse
        } else if is_downloaded(e.repo, e.file) {
            ModelStatus::Downloaded
        } else {
            ModelStatus::NeedsDownload
        };

        if fit == FitVerdict::Fits {
            let better = match best_fit {
                None => true,
                Some(j) => {
                    e.params_b > entries[j].params_b
                        || (e.params_b == entries[j].params_b && e.size_mb > entries[j].size_mb)
                }
            };
            if better {
                best_fit = Some(i);
            }
        }
        if smallest.map(|j| est < (entries[j].size_mb as f32 * 1.2).round() as u32).unwrap_or(true) {
            smallest = Some(i);
        }

        flat.push(ModelView {
            display_name: e.display_name.to_string(),
            params: e.params.to_string(),
            quant: e.quant.to_string(),
            repo: e.repo.to_string(),
            file: e.file.to_string(),
            size_mb: e.size_mb,
            est_ram_mb: est,
            status,
            fit,
            recommended: false,
        });
    }

    if let Some(idx) = best_fit.or(smallest) {
        flat[idx].recommended = true;
    }

    // Group into families in first-seen order.
    let mut families: Vec<FamilyView> = Vec::new();
    for (e, mv) in entries.iter().zip(flat.into_iter()) {
        match families.iter_mut().find(|f| f.family == e.family) {
            Some(f) => f.models.push(mv),
            None => families.push(FamilyView { family: e.family.to_string(), models: vec![mv] }),
        }
    }
    families
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test --lib catalog:: 2>&1 | tail -15`
Expected: PASS (3 tests).

- [ ] **Step 6: Confirm the default model is in the catalog (guard test)**

Add one more test to `src/catalog.rs` tests and run it:

```rust
    #[test]
    fn default_model_is_in_catalog() {
        assert!(CATALOG.iter().any(|e|
            e.repo == "Qwen/Qwen2.5-3B-Instruct-GGUF"
            && e.file == "qwen2.5-3b-instruct-q4_k_m.gguf"));
    }
```

Run: `cargo test --lib catalog:: 2>&1 | tail -8`
Expected: PASS (4 tests). Build warning-free.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs src/catalog.rs
git commit -m "feat(catalog): curated catalog + pure catalog_view (status/RAM/fit/recommend)"
```

---

### Task 2: `download::delete_cached`

**Files:**
- Modify: `src/download.rs`
- Test: `src/download.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces: `pub fn delete_cached(repo: &str, file: &str) -> std::io::Result<bool>` — removes `cache_path(repo,file)` (+ a stray `.part`); `Ok(true)` if the main file was removed, `Ok(false)` if it wasn't present.

- [ ] **Step 1: Write the failing test**

Add to `src/download.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn delete_cached_removes_file_then_reports_absent() {
        // Create a real cached file at the canonical path for a throwaway repo.
        let repo = format!("test--del-{}", uuid::Uuid::new_v4());
        let file = "m.gguf";
        let path = cache_path(&repo, file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"data").unwrap();

        assert!(delete_cached(&repo, file).unwrap()); // removed
        assert!(!path.exists());
        assert!(!delete_cached(&repo, file).unwrap()); // already gone

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib download::tests::delete_cached 2>&1 | head -15`
Expected: FAIL — `delete_cached` undefined.

- [ ] **Step 3: Implement**

In `src/download.rs`, add (module level):

```rust
/// Delete a cached model file (and any leftover `.part`). Returns whether the
/// main file existed and was removed. Best-effort on the `.part`.
pub fn delete_cached(repo: &str, file: &str) -> std::io::Result<bool> {
    let path = cache_path(repo, file);
    let _ = std::fs::remove_file(path.with_extension("part"));
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib download::tests::delete_cached 2>&1 | tail -8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/download.rs
git commit -m "feat(download): delete_cached (remove a cached model file)"
```

---

### Task 3: Wire endpoints + total-RAM detection

**Files:**
- Modify: `Cargo.toml` (`sysinfo`)
- Modify: `src/server.rs` (`AppState.total_ram_mb`; `router` param; `GET`/`DELETE /admin/models` handlers)
- Modify: `src/lib.rs` (read total RAM once; pass to `router`; update `router_for_test_with`)
- Test: `tests/http.rs`

**Interfaces:**
- Consumes: `catalog::{catalog_view, CATALOG}`, `download::{cache_path, delete_cached}`, `model_manager::ModelSpec`, `check_admin`, `ModelManager::status()`.
- Produces: `AppState { …, total_ram_mb: u64 }`; `router(manager, model_id, policy, local_ctx_window, usage, cloud_token_alert, admin_token, total_ram_mb)`.

- [ ] **Step 1: Add the dependency**

In `Cargo.toml` `[dependencies]`:

```toml
sysinfo = "0.32"
```

- [ ] **Step 2: Add `total_ram_mb` to `AppState` + `router`**

In `src/server.rs`, add to the end of `AppState`:

```rust
    /// Total physical RAM in MB (read once at startup), for catalog fit/recommend.
    pub total_ram_mb: u64,
```

Extend `router(...)` with a trailing `total_ram_mb: u64` param and add it to the `AppState { … }` it builds, then add the two routes (before `.layer(DefaultBodyLimit...)`):

```rust
        .route("/admin/models", get(handle_models_catalog).delete(handle_model_delete))
```

- [ ] **Step 3: Write the failing integration tests**

Append to `tests/http.rs`:

```rust
#[tokio::test]
async fn admin_models_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_get_status(app, "/admin/models").await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_models_returns_catalog_with_recommendation() {
    let app = localllm::router_for_test();
    let resp = localllm::axum_test_get_with_header(
        app, "/admin/models", "x-admin-token", "test-token",
    ).await;
    // grouped families, non-empty, with exactly one recommended across all
    let families = resp.as_array().expect("array of families");
    assert!(!families.is_empty());
    let rec_count: usize = families.iter()
        .flat_map(|f| f["models"].as_array().unwrap())
        .filter(|m| m["recommended"] == true)
        .count();
    assert_eq!(rec_count, 1);
}

#[tokio::test]
async fn admin_delete_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app, "/admin/models", r#"{"repo":"r","file":"f"}"#, "x-admin-token", "wrong",
    ).await;
    // wrong token → 401 (note: DELETE via the POST-status helper won't match the
    // route's method; use the DELETE helper added in Step 5 instead)
    assert_eq!(status, 401);
}
```

> The DELETE method needs a dedicated test helper (Step 5). Replace the third
> test's body once the helper exists; see Step 5.

- [ ] **Step 4: Implement the handlers**

In `src/server.rs`, add:

```rust
/// GET /admin/models — the annotated catalog for this machine (token-guarded).
async fn handle_models_catalog(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let active = state.manager.status().current;
    let view = crate::catalog::catalog_view(
        crate::catalog::CATALOG,
        state.total_ram_mb,
        Some(&active),
        |r, f| crate::download::cache_path(r, f).exists(),
    );
    Json(view).into_response()
}

/// DELETE /admin/models — free disk for a downloaded model (token-guarded).
/// Refuses the in-use model.
async fn handle_model_delete(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: AdminSwitchBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response()
        }
    };
    if body.repo.is_empty() || body.file.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "repo and file are required"})))
            .into_response();
    }
    let active = state.manager.status().current;
    if active.repo == body.repo && active.file == body.file {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "cannot delete the model in use"})),
        )
            .into_response();
    }
    match crate::download::delete_cached(&body.repo, &body.file) {
        Ok(deleted) => Json(json!({"deleted": deleted})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("delete failed: {e}")})),
        )
            .into_response(),
    }
}
```

(`AdminSwitchBody { repo, file }` already exists from sub-project 1 — reuse it.)

- [ ] **Step 5: Read total RAM in `lib.rs`, pass to `router`, update test helper + add a DELETE helper**

In `src/lib.rs` `run_server_with_ready_and_policy`, before the `router(...)` call:

```rust
    let total_ram_mb = {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        sys.total_memory() / (1024 * 1024) // bytes → MB
    };
```

Add `total_ram_mb` as the final argument to the `router(...)` call.

In `router_for_test_with`, add `16384` as the final `router(...)` argument (fixed RAM so catalog tests are deterministic).

Add a DELETE test helper to `src/lib.rs` (next to the other helpers):

```rust
/// DELETE with a JSON body + header → HTTP status code.
pub async fn axum_test_delete_status_with_header(
    app: Router, path: &str, body: &str, hname: &str, hval: &str,
) -> u16 {
    use axum::body::Body;
    use tower::ServiceExt;
    let request = axum::http::Request::builder()
        .method("DELETE")
        .uri(path)
        .header("content-type", "application/json")
        .header(hname, hval)
        .body(Body::from(body.to_owned()))
        .unwrap();
    app.oneshot(request).await.unwrap().status().as_u16()
}
```

Now replace the third integration test (`admin_delete_requires_token`) body and add the in-use 409 test in `tests/http.rs`:

```rust
#[tokio::test]
async fn admin_delete_rejects_bad_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_delete_status_with_header(
        app, "/admin/models", r#"{"repo":"r","file":"f"}"#, "x-admin-token", "wrong",
    ).await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_delete_in_use_model_is_409() {
    // router_for_test's ModelManager current spec is {repo:"test", file:"test"}.
    let app = localllm::router_for_test();
    let status = localllm::axum_test_delete_status_with_header(
        app, "/admin/models", r#"{"repo":"test","file":"test"}"#, "x-admin-token", "test-token",
    ).await;
    assert_eq!(status, 409);
}
```

Remove the earlier placeholder `admin_delete_requires_token` test (replaced by `admin_delete_rejects_bad_token`).

- [ ] **Step 6: Run new tests, full suite, build, clippy**

Run: `cargo test --test http 2>&1 | tail -30`
Expected: PASS — `admin_models_requires_token`, `admin_models_returns_catalog_with_recommendation`, `admin_delete_rejects_bad_token`, `admin_delete_in_use_model_is_409`, plus all pre-existing tests.

Run: `cargo build 2>&1 | tail -8 && cargo test 2>&1 | tail -8`
Expected: clean build; full suite green.

Run: `cargo clippy --all-targets 2>&1 | grep -E "src/catalog.rs|src/server.rs|src/download.rs" | grep -- "-->" || echo "no clippy in changed files (pre-existing SSE to_string excepted)"`
Expected: no NEW clippy in changed files (the two pre-existing SSE `to_string` in server.rs are out of scope).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): /admin/models catalog + delete endpoints + sysinfo RAM"
```

---

## Acceptance

- `GET /admin/models` (valid token) returns families → models annotated with status, est RAM, fit, and exactly one `recommended`; missing/invalid token → 401.
- `DELETE /admin/models` removes a cached model (`{"deleted":true/false}`); the in-use model → 409; bad body → 400; bad token → 401.
- Total RAM is read once via `sysinfo`; the recommendation is the largest model fitting 65% of RAM (smallest if none fit).
- `catalog_view` + `delete_cached` are unit-tested headless; the HTTP surface is integration-tested. `cargo test`/`build`/`clippy` clean.

## Self-Review

- **Spec coverage:** data model + pure annotation (T1), delete primitive (T2), endpoints + RAM detection + delete guard (T3). Recommendation/fit/status formulas, 409-in-use, embedded curated catalog incl. the default model — all covered.
- **Placeholder scan:** none — complete code/commands each step. (Curated `size_mb`/filenames are approximate-but-concrete per the spec; flagged to verify when editing.)
- **Type consistency:** `CatalogEntry`, `CATALOG`, `ModelStatus`, `FitVerdict`, `ModelView`, `FamilyView`, `catalog_view(entries, total_ram_mb: u64, active: Option<&ModelSpec>, is_downloaded)`, `delete_cached(&str,&str)->io::Result<bool>`, `AppState.total_ram_mb`, `router(... total_ram_mb)`, reused `AdminSwitchBody`, and the test helpers (`axum_test_get_status`, `axum_test_get_with_header`, new `axum_test_delete_status_with_header`) are consistent across tasks.
