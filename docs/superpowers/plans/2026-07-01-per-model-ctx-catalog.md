# Per-model Ctx + KV-aware Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a per-model context override, make the catalog's RAM estimate/fit verdict/recommendation account for the KV cache at each model's context, and expose an endpoint to set a per-model ctx (reloading the active model).

**Architecture:** `settings.rs` stores an opt-in `model_ctx` map. `catalog.rs` gains a curated `ctx_train` per entry and a KV-aware `catalog_view` (uses `src/fit.rs` from sub-project 1). `server.rs` adds a token-guarded `POST /admin/model/ctx` and carries the ctx ceiling + KV kind on `AppState`; the load path resolves the requested ctx from the override.

**Tech Stack:** Rust, serde, axum, the pure `src/fit.rs` module (sub-project 1).

## Global Constraints

- Build/test command prefix: `MISTRALRS_METAL_PRECOMPILE=0` on every `cargo` invocation.
- Default stays a **ceiling**: a model with no override requests `cfg.ctx_len` (32768); the small per-model default is only a suggestion until saved as an override.
- Override key format: `"{repo}/{file}"` (via `settings::model_ctx_key`).
- Consumes sub-project 1's `crate::fit`: `KvKind{F16,Q8,Q4}`, `device_budget_mb(total_ram_mb:u64, gpu_present:bool)->u32`, `est_kv_bytes_per_token(params_b:f32, KvKind)->u64`, `ctx_bounds(weights_mb:u32, kv_per_token_bytes:u64, budget_mb:u32, n_ctx_train:u32)->CtxBounds{min,default,max}`, and `pub const COMPUTE_HEADROOM_MB: u32`.
- Catalog verdict thresholds stay `budget = ram*65%`, `tight = ram*85%` (comfort bands); the 78% Metal budget is used ONLY inside `ctx_bounds` for the max-ctx math. Keep these distinct.
- `est_ram_mb = size_mb + kv_mb(ctx_current) + COMPUTE_HEADROOM_MB` where `kv_mb(c) = (kv_per_token * c / (1024*1024)) as u32`. Replaces the old `size_mb * 1.2`.
- `ctx == 0` on the endpoint means "clear the override".
- Curated `ctx_train` per family (verify against model cards; these are the standard native windows): Qwen2.5 = 32768; Qwen3 = 32768; Llama 3.2/3.1 = 131072; Gemma 2 = 8192; Gemma 3 1B = 32768, Gemma 3 4B/12B/27B = 131072; Phi 3.5 Mini = 131072; Phi-4 Mini = 131072; Phi-4 14B = 16384; Mistral 7B v0.3 = 32768.

---

### Task 1: settings — per-model ctx override

**Files:**
- Modify: `src/settings.rs` (add `model_ctx` to `Settings`, key helper, load/save/clear)
- Test: extend the inline `#[cfg(test)]` module

**Interfaces:**
- Consumes: the existing `load_settings()`/`save_settings()` private helpers and `with_temp_settings`/`ENV_LOCK` test scaffolding.
- Produces:
  - `pub fn model_ctx_key(repo: &str, file: &str) -> String` → `"{repo}/{file}"`
  - `pub fn load_model_ctx(key: &str) -> Option<u32>`
  - `pub fn save_model_ctx(key: &str, ctx: u32) -> anyhow::Result<()>`
  - `pub fn clear_model_ctx(key: &str) -> anyhow::Result<()>`

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/settings.rs`:

```rust
    #[test]
    fn model_ctx_round_trip() {
        with_temp_settings(|| {
            let k = model_ctx_key("bartowski/phi-4-GGUF", "phi-4-Q4_K_M.gguf");
            assert_eq!(k, "bartowski/phi-4-GGUF/phi-4-Q4_K_M.gguf");
            assert_eq!(load_model_ctx(&k), None);
            save_model_ctx(&k, 16384).unwrap();
            assert_eq!(load_model_ctx(&k), Some(16384));
        });
    }

    #[test]
    fn saving_model_ctx_preserves_profile_and_integrations() {
        with_temp_settings(|| {
            save_profile(Profile::MaxQuality).unwrap();
            let state = IntegrationState { enabled: true, priors: Default::default() };
            save_integrations(&state).unwrap();
            save_model_ctx("r/f", 8192).unwrap();
            assert_eq!(load_profile(), Profile::MaxQuality);
            assert!(load_integrations().enabled);
            assert_eq!(load_model_ctx("r/f"), Some(8192));
        });
    }

    #[test]
    fn clear_model_ctx_removes_only_that_key() {
        with_temp_settings(|| {
            save_model_ctx("a/1", 4096).unwrap();
            save_model_ctx("b/2", 8192).unwrap();
            clear_model_ctx("a/1").unwrap();
            assert_eq!(load_model_ctx("a/1"), None);
            assert_eq!(load_model_ctx("b/2"), Some(8192));
        });
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib settings::tests::model_ctx_round_trip`
Expected: FAIL to compile (`model_ctx_key`, `load_model_ctx`, `save_model_ctx` not found).

- [ ] **Step 3: Add the field + functions**

In `src/settings.rs`, add `model_ctx` to the `Settings` struct (must be `#[serde(default)]` for backward compat), and add the helpers. The `Settings` struct currently has `profile` and `integrations`:

```rust
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    profile: Profile,
    #[serde(default)]
    integrations: IntegrationState,
    #[serde(default)]
    model_ctx: std::collections::BTreeMap<String, u32>,
}
```

Add these public functions (near `save_integrations`):

```rust
/// Settings key for a model's per-model ctx override: `"{repo}/{file}"`.
pub fn model_ctx_key(repo: &str, file: &str) -> String {
    format!("{repo}/{file}")
}

/// Load a model's persisted ctx override, or `None` if unset.
pub fn load_model_ctx(key: &str) -> Option<u32> {
    load_settings().model_ctx.get(key).copied()
}

/// Persist a model's ctx override, preserving the rest of settings.
pub fn save_model_ctx(key: &str, ctx: u32) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.model_ctx.insert(key.to_string(), ctx);
    save_settings(&s)
}

/// Remove a model's ctx override, preserving the rest of settings.
pub fn clear_model_ctx(key: &str) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.model_ctx.remove(key);
    save_settings(&s)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib settings::`
Expected: PASS (existing settings tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/settings.rs
git commit -m "feat(settings): persist per-model ctx override"
```

---

### Task 2: KV-aware catalog view + AppState ctx/kv fields

**Files:**
- Modify: `src/catalog.rs` (`CatalogEntry.ctx_train`; all CATALOG entries; the test helper; `ModelView` fields; `catalog_view`)
- Modify: `src/server.rs` (`AppState.requested_ctx_ceiling` + `kv_kind`; `router(...)` signature; `handle_models_catalog` passes the new args)
- Modify: `src/lib.rs` (pass the two new values at both `router(...)` call sites — the real one and both `router_for_test*` helpers)
- Test: inline `#[cfg(test)]` in `src/catalog.rs`

**Interfaces:**
- Consumes: `crate::fit` (see Global Constraints); `crate::settings::load_model_ctx`/`model_ctx_key` (Task 1).
- Produces:
  - `CatalogEntry` gains `pub ctx_train: u32`.
  - `ModelView` gains `pub ctx_min: u32, pub ctx_default: u32, pub ctx_max: u32, pub ctx_current: u32`.
  - `pub fn catalog_view(entries, total_ram_mb: u64, requested_ctx_ceiling: u32, kv: crate::fit::KvKind, active: Option<&ModelSpec>, is_downloaded: impl Fn(&str,&str)->bool, ctx_override: impl Fn(&str,&str)->Option<u32>) -> Vec<FamilyView>`
  - `AppState.requested_ctx_ceiling: u32`, `AppState.kv_kind: crate::fit::KvKind`.

- [ ] **Step 1: Add `ctx_train` to `CatalogEntry` and every entry**

In `src/catalog.rs`, add the field to the struct:

```rust
pub struct CatalogEntry {
    pub family: &'static str,
    pub display_name: &'static str,
    pub params: &'static str,
    pub params_b: f32,
    pub quant: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size_mb: u32,
    pub ctx_train: u32,
}
```

Add `ctx_train: <value>,` to each of the 24 CATALOG entries per the Global Constraints values:
- Every `family: "Qwen2.5"` and `family: "Qwen3"` and `family: "Mistral"` → `ctx_train: 32768`.
- Every `family: "Llama"` → `ctx_train: 131072`.
- Every `family: "Gemma 2"` → `ctx_train: 8192`.
- `family: "Gemma 3"`: the `"Gemma 3 1B Instruct"` entry → `ctx_train: 32768`; the 4B/12B/27B entries → `ctx_train: 131072`.
- `family: "Phi 3.5"` → `ctx_train: 131072`.
- `family: "Phi-4"`: `"Phi-4 Mini Instruct"` → `ctx_train: 131072`; `"Phi-4 (14B)"` → `ctx_train: 16384`.

Also update the **test helper constructor** (search for the `CatalogEntry { family, display_name: name, ... size_mb }` line, ~line 245) to include `ctx_train: 32768,` so the catalog test fixtures compile.

- [ ] **Step 2: Write the failing `catalog_view` tests**

Replace/extend the catalog test module. Add these tests (they use the new signature):

```rust
    #[test]
    fn est_ram_includes_kv_and_grows_with_ctx() {
        use crate::fit::KvKind;
        // One 3B model; compare est at a small override vs the 32k ceiling.
        let entries = [CatalogEntry {
            family: "T", display_name: "t3", params: "3B", params_b: 3.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 2000, ctx_train: 32768,
        }];
        let big = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let small = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| Some(4096));
        let big_est = big[0].models[0].est_ram_mb;
        let small_est = small[0].models[0].est_ram_mb;
        // KV at 32768 costs more than at 4096 → bigger est. Both exceed size_mb.
        assert!(big_est > small_est, "big {big_est} small {small_est}");
        assert!(small_est > 2000);
        assert_eq!(small[0].models[0].ctx_current, 4096);
    }

    #[test]
    fn ctx_current_defaults_to_min_ceiling_and_max() {
        use crate::fit::KvKind;
        // Phi-4 14B on 16GB: ctx_train 16384 caps the max below the 32768 ceiling.
        let entries = [CatalogEntry {
            family: "P", display_name: "phi4", params: "14B", params_b: 14.0, quant: "Q4_K_M",
            repo: "bartowski/phi-4-GGUF", file: "phi-4-Q4_K_M.gguf", size_mb: 8634, ctx_train: 16384,
        }];
        let v = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let m = &v[0].models[0];
        assert_eq!(m.ctx_max, 16384);          // trained ctx binds
        assert_eq!(m.ctx_current, 16384);       // min(32768 ceiling, 16384 max)
        assert_eq!(m.ctx_min, crate::fit::MIN_CTX);
        assert_eq!(m.ctx_default, crate::fit::DEFAULT_SMALL_CTX.min(16384));
        // weights 8634 + KV(16384)≈1700 + headroom 1024 ≈ 11.4k < 65% of 16384 (10649)? 
        // 11358 > 10649 → Tight, not Fits. Assert it is at least not WontFit.
        assert_ne!(m.fit, FitVerdict::WontFit);
    }

    #[test]
    fn override_out_of_nothing_uses_ceiling_min_max() {
        use crate::fit::KvKind;
        let entries = [CatalogEntry {
            family: "T", display_name: "t3", params: "3B", params_b: 3.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 2000, ctx_train: 131072,
        }];
        // Big trained ctx, small model → ctx_max is memory- or GLOBAL_MAX-bound.
        let v = catalog_view(&entries, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None);
        let m = &v[0].models[0];
        // ceiling 32768 <= max → ctx_current = 32768
        assert_eq!(m.ctx_current, 32768.min(m.ctx_max));
    }
```

- [ ] **Step 3: Run to verify failure**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib catalog::`
Expected: FAIL to compile (new `catalog_view` arity, new `ModelView` fields).

- [ ] **Step 4: Add the `ModelView` fields and rewrite `catalog_view`**

Add to `ModelView`:

```rust
    pub ctx_min: u32,
    pub ctx_default: u32,
    pub ctx_max: u32,
    pub ctx_current: u32,
```

Rewrite `catalog_view` with the new signature and KV-aware estimate:

```rust
pub fn catalog_view(
    entries: &[CatalogEntry],
    total_ram_mb: u64,
    requested_ctx_ceiling: u32,
    kv: crate::fit::KvKind,
    active: Option<&ModelSpec>,
    is_downloaded: impl Fn(&str, &str) -> bool,
    ctx_override: impl Fn(&str, &str) -> Option<u32>,
) -> Vec<FamilyView> {
    let budget = total_ram_mb * 65 / 100;
    let tight_ceiling = total_ram_mb * 85 / 100;
    let budget_mb = crate::fit::device_budget_mb(total_ram_mb, true);

    let mut flat: Vec<ModelView> = Vec::with_capacity(entries.len());
    let mut best_fit: Option<usize> = None;
    let mut smallest: Option<usize> = None;

    for (i, e) in entries.iter().enumerate() {
        let kv_per_token = crate::fit::est_kv_bytes_per_token(e.params_b, kv);
        let bounds = crate::fit::ctx_bounds(e.size_mb, kv_per_token, budget_mb, e.ctx_train);
        let ctx_current = if bounds.max == 0 {
            0
        } else {
            ctx_override(e.repo, e.file).unwrap_or_else(|| requested_ctx_ceiling.min(bounds.max))
        };
        let kv_mb = (kv_per_token * ctx_current as u64 / (1024 * 1024)) as u32;
        let est = e.size_mb + kv_mb + crate::fit::COMPUTE_HEADROOM_MB;

        let fit = if bounds.max == 0 {
            FitVerdict::WontFit
        } else if (est as u64) <= budget {
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
        if smallest.map(|j| est < flat[j].est_ram_mb).unwrap_or(true) {
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
            ctx_min: bounds.min,
            ctx_default: bounds.default,
            ctx_max: bounds.max,
            ctx_current,
        });
    }

    if let Some(idx) = best_fit.or(smallest) {
        flat[idx].recommended = true;
    }

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

(Note: the `smallest` comparison now reads `flat[j].est_ram_mb` — the already-computed est of the current best-smallest — instead of recomputing `size*1.2`.)

- [ ] **Step 5: Add `AppState` fields + thread them through `router`**

In `src/server.rs`, add to `AppState`:

```rust
    pub requested_ctx_ceiling: u32,
    pub kv_kind: crate::fit::KvKind,
```

Add two params to the `router(...)` function signature (place them after `total_ram_mb`):

```rust
    requested_ctx_ceiling: u32,
    kv_kind: crate::fit::KvKind,
```

…and set them in the `AppState { ... }` it builds. Update `handle_models_catalog` to pass the new args + an override closure:

```rust
    let view = crate::catalog::catalog_view(
        crate::catalog::CATALOG,
        state.total_ram_mb,
        state.requested_ctx_ceiling,
        state.kv_kind,
        Some(&active),
        |r, f| crate::download::cache_path(r, f).exists(),
        |r, f| crate::settings::load_model_ctx(&crate::settings::model_ctx_key(r, f)),
    );
```

- [ ] **Step 6: Update the three `router(...)` call sites in `src/lib.rs`**

At the real call site in `run_server_with_ready_policy_token`, pass the ceiling + kv kind after `total_ram_mb`:

```rust
        total_ram_mb,
        cfg.ctx_len as u32,
        match cfg.kv_type {
            crate::config::KvType::Q8 => crate::fit::KvKind::Q8,
            crate::config::KvType::Q4 => crate::fit::KvKind::Q4,
            crate::config::KvType::F16 => crate::fit::KvKind::F16,
        },
```

In `router_for_test_with` (and thus `router_for_test`), pass a default ceiling `32768` and `crate::fit::KvKind::Q8` in the same positions so the test router compiles.

- [ ] **Step 7: Run catalog tests + full suite + clippy**

```
MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib catalog::
MISTRALRS_METAL_PRECOMPILE=0 cargo test
MISTRALRS_METAL_PRECOMPILE=0 cargo clippy --all-targets
```
Expected: catalog tests pass; full suite passes (existing `/admin/models` integration tests still return a valid view — now with the extra ctx fields); no new clippy warnings in `catalog`/`server`/`lib`.

- [ ] **Step 8: Commit**

```bash
git add src/catalog.rs src/server.rs src/lib.rs
git commit -m "feat(catalog): KV-aware fit + per-model ctx bounds in the catalog view"
```

---

### Task 3: POST /admin/model/ctx + load-path override

**Files:**
- Modify: `src/server.rs` (`AdminCtxBody`, `handle_model_set_ctx`, route registration)
- Modify: `src/lib.rs` (resolve the override for the initial load and the switch builder)
- Test: `tests/http.rs` (endpoint tests)

**Interfaces:**
- Consumes: `crate::settings::{model_ctx_key, save_model_ctx, clear_model_ctx}` (Task 1); `crate::fit::{ctx_bounds, est_kv_bytes_per_token, device_budget_mb}`; `state.{total_ram_mb, kv_kind, manager}`; `crate::catalog::CATALOG`; `is_safe_model_file`; `ModelManager::start_switch`.

- [ ] **Step 1: Write the failing HTTP tests**

Add to `tests/http.rs`:

```rust
#[tokio::test]
async fn set_ctx_unknown_model_returns_400() {
    let resp = localllm::axum_test_request_with_header(
        localllm::router_for_test(),
        "/admin/model/ctx",
        r#"{"repo":"nope","file":"nope.gguf","ctx":8192}"#,
        "x-admin-token",
        "test-token",
    )
    .await;
    assert_eq!(resp["error"].is_string(), true);
}

#[tokio::test]
async fn set_ctx_out_of_range_returns_400() {
    // Qwen2.5-3B is in the catalog; 999999 is above its max.
    let resp = localllm::axum_test_request_with_header(
        localllm::router_for_test(),
        "/admin/model/ctx",
        r#"{"repo":"Qwen/Qwen2.5-3B-Instruct-GGUF","file":"qwen2.5-3b-instruct-q4_k_m.gguf","ctx":999999}"#,
        "x-admin-token",
        "test-token",
    )
    .await;
    assert!(resp["error"].as_str().unwrap().contains("range")
        || resp["error"].as_str().unwrap().contains("between"));
}

#[tokio::test]
async fn set_ctx_valid_non_active_returns_saved() {
    // router_for_test's active model is "test"/"test", so this catalog model is NOT active → 200 saved.
    let _guard = ENV_LOCK.lock().await;
    let dir = std::env::temp_dir().join(format!("llm-ctx-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("LOCALLLM_SETTINGS", dir.join("settings.json"));
    let resp = localllm::axum_test_request_with_header(
        localllm::router_for_test(),
        "/admin/model/ctx",
        r#"{"repo":"Qwen/Qwen2.5-3B-Instruct-GGUF","file":"qwen2.5-3b-instruct-q4_k_m.gguf","ctx":8192}"#,
        "x-admin-token",
        "test-token",
    )
    .await;
    assert_eq!(resp["saved"], true);
    std::env::remove_var("LOCALLLM_SETTINGS");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn set_ctx_without_token_returns_401() {
    let status = localllm::axum_test_request_status(
        localllm::router_for_test(),
        "/admin/model/ctx",
        r#"{"repo":"r","file":"f","ctx":8192}"#,
    )
    .await;
    assert_eq!(status, 401);
}
```

(If `axum_test_request_status` / `axum_test_request_with_header` / `ENV_LOCK` names differ in `tests/http.rs`, match the existing admin tests' helpers — reuse exactly what the `/admin/models` tests use.)

- [ ] **Step 2: Run to verify failure**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --test http set_ctx`
Expected: FAIL — route `/admin/model/ctx` not registered (404/handler missing).

- [ ] **Step 3: Add the body type, handler, and route**

In `src/server.rs`, add the request body near `AdminSwitchBody`:

```rust
#[derive(serde::Deserialize)]
struct AdminCtxBody {
    repo: String,
    file: String,
    ctx: u32,
}
```

Add the handler:

```rust
/// POST /admin/model/ctx — set (or clear, ctx=0) a model's per-model context.
/// Validates against the model's [min,max]; reloads the model if it is active.
async fn handle_model_set_ctx(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let body: AdminCtxBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response()
        }
    };
    if body.repo.is_empty() || !is_safe_model_file(&body.file) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "repo and a valid (non-path) file are required"}))).into_response();
    }
    let Some(entry) = crate::catalog::CATALOG.iter().find(|e| e.repo == body.repo && e.file == body.file) else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "unknown model"}))).into_response();
    };

    let key = crate::settings::model_ctx_key(&body.repo, &body.file);
    if body.ctx == 0 {
        if let Err(e) = crate::settings::clear_model_ctx(&key) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response();
        }
    } else {
        let budget_mb = crate::fit::device_budget_mb(state.total_ram_mb, true);
        let kv_per_token = crate::fit::est_kv_bytes_per_token(entry.params_b, state.kv_kind);
        let bounds = crate::fit::ctx_bounds(entry.size_mb, kv_per_token, budget_mb, entry.ctx_train);
        if bounds.max == 0 || body.ctx < bounds.min || body.ctx > bounds.max {
            return (StatusCode::BAD_REQUEST, Json(json!({
                "error": format!("ctx must be in the range {}..{}", bounds.min, bounds.max)
            }))).into_response();
        }
        if let Err(e) = crate::settings::save_model_ctx(&key, body.ctx) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response();
        }
    }

    // Reload if this is the active model so the new ctx takes effect now.
    let active = state.manager.status().current;
    if active.repo == body.repo && active.file == body.file {
        let spec = crate::model_manager::ModelSpec { repo: body.repo, file: body.file };
        match state.manager.start_switch(spec) {
            Ok(()) => (StatusCode::ACCEPTED, Json(json!({"reloading": true}))).into_response(),
            Err(crate::model_manager::SwitchError::AlreadySwitching) => {
                (StatusCode::CONFLICT, Json(json!({"error": "a switch is already in progress"}))).into_response()
            }
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
        }
    } else {
        (StatusCode::OK, Json(json!({"saved": true}))).into_response()
    }
}
```

Register the route (next to `/admin/model`):

```rust
        .route("/admin/model/ctx", post(handle_model_set_ctx))
```

- [ ] **Step 4: Resolve the override in the load path (`src/lib.rs`)**

Initial load — before the `LlamaEngine::load(...)` in the `Backend::Llama` arm, resolve the requested ctx:

```rust
        let requested_ctx = crate::settings::load_model_ctx(
            &crate::settings::model_ctx_key(&cfg.model_id, &cfg.gguf_files[0]),
        ).unwrap_or(cfg.ctx_len as u32) as usize;
```

…and pass `requested_ctx` instead of `cfg.ctx_len` to `LlamaEngine::load`.

Switch builder closure — replace the `b_ctx_len` argument to the switched `LlamaEngine::load` with a per-spec resolution:

```rust
            let requested_ctx = crate::settings::load_model_ctx(
                &crate::settings::model_ctx_key(&spec.repo, &spec.file),
            ).unwrap_or(b_ctx_len as u32) as usize;
            let engine =
                LlamaEngine::load(&spec.repo, &[spec.file], requested_ctx, b_kv_type, kv_dir, b_total_ram_mb).await?;
```

(`b_ctx_len` is the captured `cfg.ctx_len`. Keep the capture; it is now only the fallback.)

- [ ] **Step 5: Run endpoint tests + full suite + clippy**

```
MISTRALRS_METAL_PRECOMPILE=0 cargo test --test http set_ctx
MISTRALRS_METAL_PRECOMPILE=0 cargo test
MISTRALRS_METAL_PRECOMPILE=0 cargo clippy --all-targets
```
Expected: the 4 endpoint tests pass; full suite passes; no new clippy warnings in `server`/`lib`.

- [ ] **Step 6: Commit**

```bash
git add src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): POST /admin/model/ctx + load-path per-model ctx override"
```

---

## Notes for the executor
- The active-model **reload path** (202/409) is thin: it calls `ModelManager::start_switch`, which is covered by the manager's own unit tests. The HTTP tests here can't easily make a CATALOG model the active one (the test manager's active spec is `"test"/"test"`), so they cover validation + persist + the non-active `200` path; the reload branch is verified by reading. If you can cheaply make a catalog model active in a test router, add a `202` test — but do not restructure the manager to do so.
- Do not change `cfg.ctx_len`'s default (32768). It is the fallback ceiling when a model has no override.
- Curated `ctx_train` values are the standard native windows; if you can quickly confirm one against its HF model card while implementing, do, but do not block on it.
