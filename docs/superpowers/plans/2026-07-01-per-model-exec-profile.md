# Per-model Execution Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each model its own saved execution profile — context length, KV-cache quantization, GPU-layer count, and history window — with a per-model recommended default and a `saved → catalog → CLI` precedence.

**Architecture:** Extend the proven per-model `model_ctx` settings pattern into an `ExecProfile` struct persisted in `settings.json`. A pure resolver combines the saved profile, the catalog's recommended defaults, and the global CLI config into resolved values. Load-time values (ctx, kv_type, gpu_layers) flow through the existing `EngineBuilder` closure into `LlamaEngine::load`; the request-time value (history_turns) truncates the message array in `api/common` before inference. The model manager's hot-swap path is unchanged — the builder closure simply resolves the target model's profile at build time.

**Tech Stack:** Rust, llama-cpp-2, axum, serde, clap. Frontend is vanilla JS (`src/manager_ui`).

## Global Constraints

- New serde fields on persisted structs MUST be `#[serde(default)]` so older `settings.json` files still load. (settings.rs invariant)
- `settings.json` load/save MUST never panic or block startup on a bad file — return defaults. (existing behavior)
- Per-model settings key is `"{repo}/{file}"` via `crate::settings::model_ctx_key`. Reuse it.
- KV-cache figures: F16 = 2.0 B/elem, Q8_0 = 1.0625, Q4_0 = 0.5625 (already in `fit::bytes_per_elem`).
- `gpu_layers` does NOT change the fit verdict in this sub-project — it is an advisory-only manual knob.
- History truncation unit is **conversation turns** (a turn begins at a `Role::User` message). Never split a `tool_use` from its `tool_result`; always keep leading `Role::System` messages.
- Run the full suite with `cargo test` after each task; it must stay green.

---

### Task 1: `ExecProfile` in settings with migration

**Files:**
- Modify: `src/config.rs` (add serde derives to `KvType`)
- Modify: `src/settings.rs` (add `ExecProfile`, `model_profiles`, migration, profile API)
- Test: `src/settings.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `crate::config::KvType`, `crate::settings::model_ctx_key`.
- Produces:
  - `pub struct ExecProfile { pub ctx: Option<u32>, pub kv_type: Option<crate::config::KvType>, pub gpu_layers: Option<u32>, pub history_turns: Option<u32> }` (derives `Debug, Clone, Default, PartialEq, Serialize, Deserialize`)
  - `pub fn load_model_profile(key: &str) -> ExecProfile`
  - `pub fn save_model_profile(key: &str, p: &ExecProfile) -> anyhow::Result<()>`
  - `pub fn clear_model_profile(key: &str) -> anyhow::Result<()>`
  - Existing `load_model_ctx/save_model_ctx/clear_model_ctx` keep working (now backed by `profile.ctx`).

- [ ] **Step 1: Make `KvType` serde-serializable**

In `src/config.rs`, change the `KvType` derive line and add serde rename so it persists as lowercase:

```rust
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvType {
    /// 8-bit quantized KV cache. ~50% the size of F16. Default.
    #[default]
    Q8,
    /// Full 16-bit float KV cache. Maximum quality, most RAM.
    F16,
    /// 4-bit quantized KV cache. ~25% the size of F16. May reduce quality.
    Q4,
}
```

- [ ] **Step 2: Write the failing migration + round-trip tests**

Add to `src/settings.rs` tests module:

```rust
#[test]
fn model_profile_round_trips() {
    with_temp_settings(|| {
        let k = model_ctx_key("r", "f");
        assert_eq!(load_model_profile(&k), ExecProfile::default());
        let p = ExecProfile {
            ctx: Some(8192),
            kv_type: Some(crate::config::KvType::Q4),
            gpu_layers: Some(20),
            history_turns: Some(3),
        };
        save_model_profile(&k, &p).unwrap();
        assert_eq!(load_model_profile(&k), p);
    });
}

#[test]
fn legacy_model_ctx_migrates_into_profile() {
    with_temp_settings(|| {
        // Write an old-shape settings file that only has model_ctx.
        let path = settings_path().unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{"model_ctx":{"r/f":16384}}"#).unwrap();
        // Reading the profile should surface the migrated ctx.
        assert_eq!(load_model_profile("r/f").ctx, Some(16384));
        // And the ctx back-compat wrapper still sees it.
        assert_eq!(load_model_ctx("r/f"), Some(16384));
    });
}

#[test]
fn ctx_wrapper_writes_through_to_profile() {
    with_temp_settings(|| {
        save_model_ctx("r/f", 4096).unwrap();
        assert_eq!(load_model_profile("r/f").ctx, Some(4096));
        clear_model_ctx("r/f").unwrap();
        assert_eq!(load_model_profile("r/f").ctx, None);
    });
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib settings:: 2>&1 | tail -20`
Expected: FAIL — `ExecProfile`, `load_model_profile`, etc. do not exist.

- [ ] **Step 4: Implement `ExecProfile`, migration, and the profile API**

In `src/settings.rs`, add the struct near `IntegrationState`:

```rust
/// Per-model execution profile. Every field optional: `None` means "fall back
/// to the catalog recommendation, then the global CLI default".
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_type: Option<crate::config::KvType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_turns: Option<u32>,
}
```

Extend the on-disk `Settings` struct with the new map and keep the legacy field for reading:

```rust
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    profile: Profile,
    #[serde(default)]
    integrations: IntegrationState,
    /// Legacy per-model ctx map. Read-only for migration; new writes go to
    /// `model_profiles`. Kept so old files still load.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    model_ctx: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    model_profiles: std::collections::BTreeMap<String, ExecProfile>,
}
```

Fold the legacy map into `model_profiles` right after loading:

```rust
fn load_settings() -> Settings {
    let mut s = match settings_path()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .and_then(|t| serde_json::from_str::<Settings>(&t).ok())
    {
        Some(s) => s,
        None => Settings::default(),
    };
    // Migrate any legacy model_ctx entries into model_profiles.ctx.
    for (k, ctx) in std::mem::take(&mut s.model_ctx) {
        s.model_profiles.entry(k).or_default().ctx.get_or_insert(ctx);
    }
    s
}
```

Replace the existing `model_ctx` helpers with profile-backed versions and add the profile API:

```rust
pub fn load_model_profile(key: &str) -> ExecProfile {
    load_settings().model_profiles.get(key).cloned().unwrap_or_default()
}

pub fn save_model_profile(key: &str, p: &ExecProfile) -> anyhow::Result<()> {
    let mut s = load_settings();
    if *p == ExecProfile::default() {
        s.model_profiles.remove(key);
    } else {
        s.model_profiles.insert(key.to_string(), p.clone());
    }
    save_settings(&s)
}

pub fn clear_model_profile(key: &str) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.model_profiles.remove(key);
    save_settings(&s)
}

// --- ctx back-compat wrappers, now backed by the profile's ctx field ---

pub fn load_model_ctx(key: &str) -> Option<u32> {
    load_model_profile(key).ctx
}

pub fn save_model_ctx(key: &str, ctx: u32) -> anyhow::Result<()> {
    let mut p = load_model_profile(key);
    p.ctx = Some(ctx);
    save_model_profile(key, &p)
}

pub fn clear_model_ctx(key: &str) -> anyhow::Result<()> {
    let mut p = load_model_profile(key);
    p.ctx = None;
    save_model_profile(key, &p)
}
```

Delete the old `model_ctx`-map-based bodies of `load_model_ctx/save_model_ctx/clear_model_ctx` (replaced above). Keep `model_ctx_key` unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib settings:: 2>&1 | tail -20`
Expected: PASS (including the pre-existing `model_ctx_round_trip`, `clear_model_ctx_removes_only_that_key`, etc.).

- [ ] **Step 6: Commit**

```bash
git add src/config.rs src/settings.rs
git commit -m "feat(settings): per-model ExecProfile with model_ctx migration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Catalog recommendations + per-entry KV in `catalog_view`

**Files:**
- Modify: `src/catalog.rs` (`CatalogEntry`, `ModelView`, `catalog_view`, `CATALOG` literals)
- Test: `src/catalog.rs` tests

**Interfaces:**
- Consumes: `crate::fit::KvKind`, `crate::config::KvType`, `ExecProfile` fields (via injected closures).
- Produces:
  - `CatalogEntry` gains `pub rec_kv: Option<crate::config::KvType>`, `pub rec_gpu_layers: Option<u32>`, `pub rec_history_turns: Option<u32>`.
  - `ModelView` gains `pub kv_current: String`, `pub kv_default: String`, `pub gpu_layers_current: Option<u32>`, `pub history_turns_current: Option<u32>`, `pub history_turns_default: Option<u32>`.
  - `catalog_view(entries, total_ram_mb, requested_ctx_ceiling, kv, active, is_downloaded, ctx_override, profile_override)` — new final param `profile_override: impl Fn(&str,&str) -> crate::settings::ExecProfile`.

- [ ] **Step 1: Write the failing test for per-entry KV resolution**

Add to `src/catalog.rs` tests:

```rust
#[test]
fn catalog_view_uses_per_model_kv_from_profile() {
    use crate::settings::ExecProfile;
    let cat = vec![entry("Qwen2.5", "Qwen 7B", 7.0, "q/7b", "7b.gguf", 4700)];
    // Global kv = Q8, but this model's saved profile forces F16 → larger KV est.
    let f16_profile = |_r: &str, _f: &str| ExecProfile {
        kv_type: Some(crate::config::KvType::F16), ..Default::default()
    };
    let none = |_r: &str, _f: &str| ExecProfile::default();
    let with_f16 = catalog_view(&cat, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, f16_profile);
    let with_q8  = catalog_view(&cat, 16384, 32768, KvKind::Q8, None, |_, _| false, |_, _| None, none);
    // F16 KV is heavier per token → smaller max ctx for the same budget.
    assert!(with_f16[0].models[0].ctx_max <= with_q8[0].models[0].ctx_max);
    assert_eq!(with_f16[0].models[0].kv_current, "f16");
    assert_eq!(with_q8[0].models[0].kv_current, "q8");
}
```

Update the `entry(...)` test helper to set the new recommendation fields to `None`:

```rust
fn entry(family: &'static str, name: &'static str, pb: f32, repo: &'static str, file: &'static str, size_mb: u32) -> CatalogEntry {
    CatalogEntry { family, display_name: name, params: "x", params_b: pb, quant: "Q4_K_M",
        repo, file, size_mb, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib catalog:: 2>&1 | tail -20`
Expected: FAIL — new fields/params/closure do not exist.

- [ ] **Step 3: Add the struct fields and resolve KV per entry**

In `src/catalog.rs`:

Add to `CatalogEntry` (after `ctx_train`):

```rust
    /// Recommended per-model overrides. `None` = use the global default.
    pub rec_kv: Option<crate::config::KvType>,
    pub rec_gpu_layers: Option<u32>,
    pub rec_history_turns: Option<u32>,
```

Add to `ModelView` (after `ctx_current`):

```rust
    pub kv_current: String,
    pub kv_default: String,
    pub gpu_layers_current: Option<u32>,
    pub history_turns_current: Option<u32>,
    pub history_turns_default: Option<u32>,
```

Add a small mapping helper near the top of `catalog.rs`:

```rust
/// Map a `KvType` to the fit-math `KvKind` and a lowercase tag.
fn kv_to_kind(t: crate::config::KvType) -> (crate::fit::KvKind, &'static str) {
    match t {
        crate::config::KvType::Q8 => (crate::fit::KvKind::Q8, "q8"),
        crate::config::KvType::Q4 => (crate::fit::KvKind::Q4, "q4"),
        crate::config::KvType::F16 => (crate::fit::KvKind::F16, "f16"),
    }
}

/// Lowercase tag for the global fallback `KvKind`.
fn kv_kind_tag(k: crate::fit::KvKind) -> &'static str {
    match k {
        crate::fit::KvKind::Q8 => "q8",
        crate::fit::KvKind::Q4 => "q4",
        crate::fit::KvKind::F16 => "f16",
    }
}
```

Change the `catalog_view` signature to add the trailing closure:

```rust
pub fn catalog_view(
    entries: &[CatalogEntry],
    total_ram_mb: u64,
    requested_ctx_ceiling: u32,
    kv: crate::fit::KvKind,
    active: Option<&ModelSpec>,
    is_downloaded: impl Fn(&str, &str) -> bool,
    ctx_override: impl Fn(&str, &str) -> Option<u32>,
    profile_override: impl Fn(&str, &str) -> crate::settings::ExecProfile,
) -> Vec<FamilyView> {
```

Inside the per-entry loop, before computing `kv_per_token`, resolve the effective KV (saved profile → `rec_kv` → global `kv`):

```rust
        let prof = profile_override(e.repo, e.file);
        let (eff_kv_kind, kv_tag) = match prof.kv_type.or(e.rec_kv) {
            Some(t) => kv_to_kind(t),
            None => (kv, kv_kind_tag(kv)),
        };
        let kv_default_tag = match e.rec_kv {
            Some(t) => kv_to_kind(t).1,
            None => kv_kind_tag(kv),
        };
        let kv_per_token = crate::fit::est_kv_bytes_per_token(e.params_b, eff_kv_kind);
```

(Replace the old `let kv_per_token = crate::fit::est_kv_bytes_per_token(e.params_b, kv);` line.)

When constructing the `ModelView`, add the new fields:

```rust
            kv_current: kv_tag.to_string(),
            kv_default: kv_default_tag.to_string(),
            gpu_layers_current: prof.gpu_layers.or(e.rec_gpu_layers),
            history_turns_current: prof.history_turns.or(e.rec_history_turns),
            history_turns_default: e.rec_history_turns,
```

- [ ] **Step 4: Add the recommendation fields to every `CATALOG` literal**

Each `CatalogEntry { ... ctx_train: N }` literal in the `CATALOG` const must gain the three new fields. Use `None` for all of them (no non-default recommendations ship in sub-1). Apply this to all 24 entries — add `, rec_kv: None, rec_gpu_layers: None, rec_history_turns: None` before the closing `}` of each entry.

Example (Qwen 0.5B), the rest follow identically:

```rust
    CatalogEntry { family: "Qwen2.5", display_name: "Qwen2.5 0.5B Instruct", params: "0.5B", params_b: 0.5, quant: "Q4_K_M",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", file: "qwen2.5-0.5b-instruct-q4_k_m.gguf", size_mb: 469, ctx_train: 32768,
        rec_kv: None, rec_gpu_layers: None, rec_history_turns: None },
```

Also update the two per-model literal `CatalogEntry`s inside the existing tests (`est_ram_includes_kv_and_grows_with_ctx`, `ctx_current_defaults_to_min_ceiling_and_max`, `override_out_of_nothing_uses_ceiling_min_max`) the same way.

- [ ] **Step 5: Update existing `catalog_view` callers in tests**

Every existing `catalog_view(...)` call in the `catalog.rs` tests takes one more argument now. Append `, |_, _| crate::settings::ExecProfile::default()` to each call in the test module.

- [ ] **Step 6: Update the production caller in `server.rs`**

`handle_models_catalog` (~`src/server.rs:514`) calls `catalog_view(...)` and will not compile without the new trailing closure. Add it, loading each entry's saved profile:

```rust
        |r, f| crate::settings::load_model_ctx(&crate::settings::model_ctx_key(r, f)), // existing ctx_override
        |r, f| crate::settings::load_model_profile(&crate::settings::model_ctx_key(r, f)),
```

(The first line already exists; add the second as the new final argument.)

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test --lib catalog:: 2>&1 | tail -20`
Expected: PASS — including the new `catalog_view_uses_per_model_kv_from_profile`.

- [ ] **Step 8: Commit**

```bash
git add src/catalog.rs src/server.rs
git commit -m "feat(catalog): per-model KV in fit view + recommendation fields

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Pure profile resolver

**Files:**
- Create: `src/profile.rs`
- Modify: `src/lib.rs` (add `pub mod profile;`)
- Test: `src/profile.rs` tests

**Interfaces:**
- Consumes: `crate::settings::ExecProfile`, `crate::config::KvType`, `crate::catalog::CatalogEntry`.
- Produces:
  - `pub struct Resolved { pub ctx: u32, pub kv_type: KvType, pub gpu_layers: Option<u32>, pub history_turns: Option<u32> }`
  - `pub fn resolve(saved: &ExecProfile, catalog: Option<&CatalogEntry>, global_ctx: u32, global_kv: KvType) -> Resolved`

- [ ] **Step 1: Write the failing resolver tests**

Create `src/profile.rs`:

```rust
//! Pure resolution of a model's effective execution parameters from three
//! layers of precedence: saved per-model profile → catalog recommendation →
//! global CLI default. No I/O.

use crate::catalog::CatalogEntry;
use crate::config::KvType;
use crate::settings::ExecProfile;

/// Fully resolved execution parameters for a model load.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub ctx: u32,
    pub kv_type: KvType,
    pub gpu_layers: Option<u32>,
    pub history_turns: Option<u32>,
}

/// Resolve each field: saved profile wins, else catalog recommendation, else
/// the global CLI default. `ctx` has no catalog recommendation here (catalog ctx
/// defaults are memory-derived elsewhere), so it is `saved.ctx` or `global_ctx`.
pub fn resolve(
    saved: &ExecProfile,
    catalog: Option<&CatalogEntry>,
    global_ctx: u32,
    global_kv: KvType,
) -> Resolved {
    let rec_kv = catalog.and_then(|c| c.rec_kv);
    let rec_gpu = catalog.and_then(|c| c.rec_gpu_layers);
    let rec_hist = catalog.and_then(|c| c.rec_history_turns);
    Resolved {
        ctx: saved.ctx.unwrap_or(global_ctx),
        kv_type: saved.kv_type.or(rec_kv).unwrap_or(global_kv),
        gpu_layers: saved.gpu_layers.or(rec_gpu),
        history_turns: saved.history_turns.or(rec_hist),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat(rec_kv: Option<KvType>, rec_gpu: Option<u32>, rec_hist: Option<u32>) -> CatalogEntry {
        CatalogEntry {
            family: "F", display_name: "M", params: "7B", params_b: 7.0, quant: "Q4_K_M",
            repo: "r", file: "f", size_mb: 4700, ctx_train: 32768,
            rec_kv, rec_gpu_layers: rec_gpu, rec_history_turns: rec_hist,
        }
    }

    #[test]
    fn saved_wins_over_catalog_and_global() {
        let saved = ExecProfile {
            ctx: Some(8192), kv_type: Some(KvType::Q4),
            gpu_layers: Some(10), history_turns: Some(2),
        };
        let c = cat(Some(KvType::F16), Some(99), Some(9));
        let r = resolve(&saved, Some(&c), 32768, KvType::Q8);
        assert_eq!(r, Resolved { ctx: 8192, kv_type: KvType::Q4, gpu_layers: Some(10), history_turns: Some(2) });
    }

    #[test]
    fn catalog_fills_gaps_when_saved_empty() {
        let saved = ExecProfile::default();
        let c = cat(Some(KvType::F16), Some(20), Some(4));
        let r = resolve(&saved, Some(&c), 32768, KvType::Q8);
        assert_eq!(r, Resolved { ctx: 32768, kv_type: KvType::F16, gpu_layers: Some(20), history_turns: Some(4) });
    }

    #[test]
    fn global_fallback_when_nothing_set() {
        let r = resolve(&ExecProfile::default(), None, 32768, KvType::Q8);
        assert_eq!(r, Resolved { ctx: 32768, kv_type: KvType::Q8, gpu_layers: None, history_turns: None });
    }
}
```

Add to `src/lib.rs` near the other `pub mod` declarations:

```rust
pub mod profile;
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `cargo test --lib profile:: 2>&1 | tail -20`
Expected: PASS (module is written complete; this is a pure unit with its own tests).

- [ ] **Step 3: Commit**

```bash
git add src/profile.rs src/lib.rs
git commit -m "feat(profile): pure saved->catalog->global resolver

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire `gpu_layers` through the llama engine

**Files:**
- Modify: `src/engine_llama.rs` (`LlamaEngine::load`, `worker_thread`, `with_n_gpu_layers` sites)
- Test: `src/engine_llama.rs` tests (pure helper only)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `fn resolve_n_gpu_layers(opt: Option<u32>) -> u32` (pure; `None → u32::MAX`).
  - `LlamaEngine::load(..., kv_cache_dir, gpu_layers: Option<u32>, total_ram_mb)` — new `gpu_layers` param inserted before `total_ram_mb`.

- [ ] **Step 1: Write the failing helper test**

Add to the `engine_llama.rs` test module:

```rust
#[test]
fn resolve_n_gpu_layers_maps_none_to_all() {
    assert_eq!(super::resolve_n_gpu_layers(None), u32::MAX);
    assert_eq!(super::resolve_n_gpu_layers(Some(0)), 0);
    assert_eq!(super::resolve_n_gpu_layers(Some(24)), 24);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib engine_llama::tests::resolve_n_gpu_layers 2>&1 | tail -20`
Expected: FAIL — `resolve_n_gpu_layers` not defined.

- [ ] **Step 3: Add the helper and thread the param**

Add the helper near `make_ctx_params` in `src/engine_llama.rs`:

```rust
/// GPU-layer count for `with_n_gpu_layers`: `None` means "all layers on GPU".
fn resolve_n_gpu_layers(opt: Option<u32>) -> u32 {
    opt.unwrap_or(u32::MAX)
}
```

In `LlamaEngine::load`, add the `gpu_layers` parameter (before `total_ram_mb`):

```rust
    pub async fn load(
        model_id: &str,
        gguf_files: &[String],
        ctx_len: usize,
        kv_cache_type: KvCacheType,
        kv_cache_dir: Option<PathBuf>,
        gpu_layers: Option<u32>,
        total_ram_mb: u64,
    ) -> Result<Self> {
```

Pass it into the worker spawn:

```rust
        std::thread::spawn(move || {
            worker_thread(path, ctx_len_u32, kv_cache_type, gpu_layers, provenance_base, kv_cache_dir, total_ram_mb, rx, load_tx);
        });
```

Add the parameter to `worker_thread` (before `provenance_base`):

```rust
fn worker_thread(
    model_path: PathBuf,
    ctx_len: u32,
    kv_cache_type: KvCacheType,
    gpu_layers: Option<u32>,
    provenance_base: String,
    kv_cache_dir: Option<PathBuf>,
    total_ram_mb: u64,
    rx: std_mpsc::Receiver<Job>,
    load_tx: tokio::sync::oneshot::Sender<Result<u32>>,
) {
```

Replace the hardcoded model-params line in `worker_thread`:

```rust
    let model_params = LlamaModelParams::default().with_n_gpu_layers(resolve_n_gpu_layers(gpu_layers));
```

The other `with_n_gpu_layers(u32::MAX)` site (around line 809) is in a standalone/spike helper — leave it as `u32::MAX` (it is not on the serving path).

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `cargo test --lib engine_llama::tests::resolve_n_gpu_layers 2>&1 | tail -20`
Expected: PASS. (The two `LlamaEngine::load` call sites in `lib.rs` are updated in Task 5; the crate will not fully compile until then — that is expected mid-task and resolved by the next task's commit.)

- [ ] **Step 5: Commit**

```bash
git add src/engine_llama.rs
git commit -m "feat(engine): thread gpu_layers into llama model load

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Resolve profiles at startup and on model switch

**Files:**
- Modify: `src/lib.rs` (startup load ~L89-106, builder closure ~L130-160)

**Interfaces:**
- Consumes: `crate::profile::resolve`, `crate::settings::load_model_profile`, `crate::catalog::CATALOG`, `crate::config::KvType`, updated `LlamaEngine::load`.
- Produces: nothing new (internal wiring; both load paths now pass resolved `kv_type` + `gpu_layers`).

- [ ] **Step 1: Add a resolver helper in `lib.rs`**

Add a small private helper in `src/lib.rs` (top-level fn) that resolves a spec's parameters from settings + catalog + globals:

```rust
/// Resolve a model's effective load parameters (ctx, kv_type, gpu_layers) from
/// its saved profile, the catalog recommendation, and the global CLI defaults.
fn resolve_load_params(
    repo: &str,
    file: &str,
    global_ctx: u32,
    global_kv: crate::config::KvType,
) -> crate::profile::Resolved {
    let key = crate::settings::model_ctx_key(repo, file);
    let saved = crate::settings::load_model_profile(&key);
    let catalog = crate::catalog::CATALOG.iter().find(|e| e.repo == repo && e.file == file);
    crate::profile::resolve(&saved, catalog, global_ctx, global_kv)
}

/// Map `KvType` to the llama-cpp-2 KV cache type.
fn kv_type_to_llama(t: crate::config::KvType) -> llama_cpp_2::context::params::KvCacheType {
    use llama_cpp_2::context::params::KvCacheType;
    match t {
        crate::config::KvType::Q8 => KvCacheType::Q8_0,
        crate::config::KvType::Q4 => KvCacheType::Q4_0,
        crate::config::KvType::F16 => KvCacheType::F16,
    }
}
```

- [ ] **Step 2: Use the resolver in the startup load path**

Replace the startup block (currently resolving only `requested_ctx` and using `cfg.llama_kv_cache_type()`) so it resolves the full profile and passes `gpu_layers`:

```rust
            let kv_cache_dir = cfg.resolved_kv_cache_dir();
            let r = resolve_load_params(
                &cfg.model_id,
                &cfg.gguf_files[0],
                cfg.ctx_len as u32,
                cfg.kv_type.clone(),
            );
            tracing::info!(
                "resolved load params: ctx={} kv={:?} gpu_layers={:?}",
                r.ctx, r.kv_type, r.gpu_layers
            );
            let llama = LlamaEngine::load(
                &cfg.model_id,
                &cfg.gguf_files,
                r.ctx as usize,
                kv_type_to_llama(r.kv_type),
                kv_cache_dir,
                r.gpu_layers,
                total_ram_mb,
            )
```

(Match the existing surrounding `.await?` / assignment exactly.)

- [ ] **Step 3: Use the resolver in the builder closure**

In the `EngineBuilder` closure, replace the `requested_ctx` resolution + `LlamaEngine::load` call. Because the closure currently captures `b_kv_type` (a `KvCacheType`), change it to capture the global `KvType` and ctx instead:

Change the captured-globals block (~L130-133) to:

```rust
    let b_ctx_len = cfg.ctx_len as u32;
    let b_kv_type = cfg.kv_type.clone(); // crate::config::KvType
    let b_kv_dir = cfg.resolved_kv_cache_dir();
    let b_total_ram_mb = total_ram_mb;
```

Change the closure body's load call to:

```rust
            let r = resolve_load_params(&spec.repo, &spec.file, b_ctx_len, b_kv_type.clone());
            let engine =
                LlamaEngine::load(
                    &spec.repo,
                    &[spec.file.clone()],
                    r.ctx as usize,
                    kv_type_to_llama(r.kv_type),
                    kv_dir,
                    r.gpu_layers,
                    b_total_ram_mb,
                ).await?;
```

(Keep the surrounding `kv_dir` clone and `Ok(engine)` return as they are; only the resolution + call change.)

- [ ] **Step 4: Verify the whole crate compiles and the suite is green**

Run: `cargo test 2>&1 | tail -25`
Expected: PASS — the previously-incomplete Task 4 call sites now compile; all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat(startup): resolve per-model profile on load and switch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: History truncation (pure)

**Files:**
- Modify: `src/api/common.rs` (add `truncate_history`)
- Test: `src/api/common.rs` tests

**Interfaces:**
- Consumes: `ChatMessage`, `Role`.
- Produces: `pub fn truncate_history(messages: Vec<ChatMessage>, keep_turns: Option<u32>) -> Vec<ChatMessage>`.

- [ ] **Step 1: Write the failing truncation tests**

Add to `src/api/common.rs` tests module:

```rust
fn m(role: Role, text: &str) -> ChatMessage {
    ChatMessage { role, text: Some(text.into()), tool_calls: vec![], tool_result: None }
}

#[test]
fn truncate_keeps_system_and_last_n_turns() {
    let msgs = vec![
        m(Role::System, "sys"),
        m(Role::User, "u1"), m(Role::Assistant, "a1"),
        m(Role::User, "u2"), m(Role::Assistant, "a2"),
    ];
    let out = truncate_history(msgs, Some(1));
    let texts: Vec<_> = out.iter().map(|x| x.text.clone().unwrap()).collect();
    assert_eq!(texts, vec!["sys", "u2", "a2"]);
}

#[test]
fn truncate_never_splits_a_tool_turn() {
    // Turn 1 has a tool call + result; keeping 1 turn must keep turn 2 whole.
    let msgs = vec![
        m(Role::System, "sys"),
        m(Role::User, "u1"),
        ChatMessage { role: Role::Assistant, text: None,
            tool_calls: vec![ToolCall { id: "c1".into(), name: "t".into(), arguments: "{}".into() }],
            tool_result: None },
        ChatMessage { role: Role::Tool, text: None, tool_calls: vec![],
            tool_result: Some(ToolResult { tool_call_id: "c1".into(), content: "ok".into() }) },
        m(Role::Assistant, "a1"),
        m(Role::User, "u2"), m(Role::Assistant, "a2"),
    ];
    let out = truncate_history(msgs.clone(), Some(1));
    let roles: Vec<_> = out.iter().map(|x| x.role.clone()).collect();
    assert_eq!(roles, vec![Role::System, Role::User, Role::Assistant]);
    // Keeping 2 turns returns everything.
    assert_eq!(truncate_history(msgs.clone(), Some(2)).len(), msgs.len());
}

#[test]
fn truncate_none_is_passthrough() {
    let msgs = vec![m(Role::System, "sys"), m(Role::User, "u1")];
    assert_eq!(truncate_history(msgs.clone(), None), msgs);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib api::common 2>&1 | tail -20`
Expected: FAIL — `truncate_history` not defined.

- [ ] **Step 3: Implement `truncate_history`**

Add to `src/api/common.rs`:

```rust
/// Keep only the last `keep_turns` conversation turns, plus all leading system
/// messages. A turn begins at a `Role::User` message and runs until the next
/// `Role::User`, so cutting on a user boundary never splits a `tool_use` from
/// its `tool_result` (both live inside the same turn). `None` = passthrough.
pub fn truncate_history(messages: Vec<ChatMessage>, keep_turns: Option<u32>) -> Vec<ChatMessage> {
    let Some(n) = keep_turns else { return messages };
    let n = n as usize;

    // Leading system messages are always kept.
    let lead_sys = messages.iter().take_while(|m| m.role == Role::System).count();

    // Indices (in the full vec) where a turn starts.
    let user_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect();

    if user_starts.len() <= n {
        return messages; // nothing to drop
    }

    // Start of the first turn we keep.
    let cut = user_starts[user_starts.len() - n];

    let mut out = Vec::with_capacity(lead_sys + (messages.len() - cut));
    out.extend(messages.iter().take(lead_sys).cloned());
    out.extend(messages.iter().skip(cut).cloned());
    out
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib api::common 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/common.rs
git commit -m "feat(api): turn-based history truncation helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Apply history truncation on the request path

**Files:**
- Modify: `src/server.rs` (resolve active model's `history_turns`, truncate before generate/stream)
- Test: `src/server.rs` or `src/lib.rs` router tests

**Interfaces:**
- Consumes: `crate::api::common::truncate_history`, `crate::settings::load_model_profile`, `crate::catalog::CATALOG`, `AppState.requested_ctx_ceiling`/existing state, `ModelManager::status().current`.
- Produces: internal — requests are truncated per the active model's resolved `history_turns` before inference.

- [ ] **Step 1: Locate the single choke point**

Both `/v1/chat/completions` and `/v1/messages` build a `ChatRequest` and call `state.generator.generate(...)` / `generate_stream(...)`. Find the shared helper that owns the `ChatRequest` just before those calls (search `generate(` and `generate_stream(` in `src/server.rs`). Truncation must be applied to `req.messages` there, once, so both APIs are covered.

- [ ] **Step 2: Write a failing integration test**

Add a router test (in `src/server.rs` tests or the `lib.rs` router tests) that saves a `history_turns = 1` profile for the active model, posts a 3-turn conversation, and asserts the generator received a truncated message list. Use the existing test `Generator` seam. Concretely, add a capturing generator that records the last `ChatRequest.messages.len()` and assert it equals `lead_sys + 2` (system + last turn's user+assistant) rather than the full history.

```rust
#[tokio::test]
async fn active_model_history_turns_truncates_request() {
    // Arrange: temp settings with history_turns=1 for the active model spec,
    // an AppState whose generator records the messages it receives.
    // Act: POST /v1/messages with [system,u1,a1,u2,a2].
    // Assert: recorded messages == [system,u2,a2] (len 3).
    // (Wire using the existing test harness patterns in this module.)
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test --lib active_model_history_turns_truncates 2>&1 | tail -20`
Expected: FAIL — no truncation is applied yet.

- [ ] **Step 4: Resolve `history_turns` for the active model and truncate**

At the shared choke point, before calling the generator, resolve the active model's `history_turns` and apply it:

```rust
    // Trim old history to the active model's configured window (request-time).
    let active = state.manager.status().current;
    let key = crate::settings::model_ctx_key(&active.repo, &active.file);
    let saved = crate::settings::load_model_profile(&key);
    let rec = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == active.repo && e.file == active.file)
        .and_then(|e| e.rec_history_turns);
    let keep_turns = saved.history_turns.or(rec);
    req.messages = crate::api::common::truncate_history(req.messages, keep_turns);
```

Adjust field access (`state.manager` vs however `AppState` exposes the `ModelManager`) to match the actual struct in `server.rs`. If `AppState` does not already hold the manager, use whatever handle exposes `status()` / the active `ModelSpec` (the catalog endpoint already reads the active spec — reuse that path).

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test --lib active_model_history_turns_truncates 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): truncate history per active model's window

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `POST /admin/model/profile` endpoint

**Files:**
- Modify: `src/server.rs` (route, handler, keep `/ctx` wrapper)
- Test: `src/server.rs` / `lib.rs` router tests

**Interfaces:**
- Consumes: `crate::settings::{load_model_profile, save_model_profile}`, `crate::fit::ctx_bounds`, `crate::config::KvType`, `crate::catalog::CATALOG`.
- Produces: `POST /admin/model/profile` accepting `{ repo, file, ctx?, kv_type?, gpu_layers?, history_turns? }` (each optional; sentinel `0` clears `ctx`).

- [ ] **Step 1: Write the failing endpoint test**

Add a router test that POSTs a profile and asserts it persists:

```rust
#[tokio::test]
async fn set_model_profile_persists_kv_and_history() {
    // POST /admin/model/profile {repo,file,kv_type:"q4",history_turns:3}
    // with the correct X-Admin-Token; expect 200.
    // Then load_model_profile(key) == { kv_type: Q4, history_turns: 3, .. }.
}
```

Mirror the auth/token wiring from the existing `handle_model_set_ctx` test if present; otherwise assert via `crate::settings::load_model_profile` after the call.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test --lib set_model_profile_persists 2>&1 | tail -20`
Expected: FAIL — route/handler absent.

- [ ] **Step 3: Add the route and handler**

Register the route next to the existing ctx route in `build_router` (near `src/server.rs:407`):

```rust
        .route("/admin/model/profile", post(handle_model_set_profile))
```

Add the request body + handler:

```rust
#[derive(serde::Deserialize)]
struct SetProfileBody {
    repo: String,
    file: String,
    #[serde(default)]
    ctx: Option<u32>,          // 0 clears
    #[serde(default)]
    kv_type: Option<crate::config::KvType>,
    #[serde(default)]
    gpu_layers: Option<u32>,
    #[serde(default)]
    history_turns: Option<u32>,
}

/// POST /admin/model/profile — set a subset of a model's execution profile.
/// Validates `ctx` against the model's fit bounds; `ctx == 0` clears it.
/// Reloads the model if it is the active one.
async fn handle_model_set_profile(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<SetProfileBody>,
) -> impl IntoResponse {
    if let Err(resp) = require_admin(&state, &headers) {
        return resp;
    }
    let Some(entry) = crate::catalog::CATALOG.iter().find(|e| e.repo == body.repo && e.file == body.file) else {
        return json_error(StatusCode::NOT_FOUND, "unknown model");
    };
    let key = crate::settings::model_ctx_key(&body.repo, &body.file);
    let mut prof = crate::settings::load_model_profile(&key);

    if let Some(ctx) = body.ctx {
        if ctx == 0 {
            prof.ctx = None;
        } else {
            let budget_mb = crate::fit::device_budget_mb(state.total_ram_mb, true);
            let kv_kind = match prof.kv_type.or(entry.rec_kv) {
                Some(crate::config::KvType::Q4) => crate::fit::KvKind::Q4,
                Some(crate::config::KvType::F16) => crate::fit::KvKind::F16,
                _ => crate::fit::KvKind::Q8,
            };
            let kv_per_token = crate::fit::est_kv_bytes_per_token(entry.params_b, kv_kind);
            let bounds = crate::fit::ctx_bounds(entry.size_mb, kv_per_token, budget_mb, entry.ctx_train);
            if bounds.max == 0 || ctx < bounds.min || ctx > bounds.max {
                return json_error(StatusCode::BAD_REQUEST,
                    &format!("ctx must be in the range {}..{}", bounds.min, bounds.max));
            }
            prof.ctx = Some(ctx);
        }
    }
    if let Some(kv) = body.kv_type { prof.kv_type = Some(kv); }
    if let Some(g) = body.gpu_layers { prof.gpu_layers = if g == u32::MAX { None } else { Some(g) }; }
    if let Some(h) = body.history_turns { prof.history_turns = if h == 0 { None } else { Some(h) }; }

    if let Err(e) = crate::settings::save_model_profile(&key, &prof) {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("save failed: {e}"));
    }

    // Reload if this is the active model so the new profile takes effect now.
    reload_if_active(&state, &body.repo, &body.file).await
}
```

Match the actual names in `server.rs` for the admin guard (`require_admin` or the inline token check used by `handle_model_set_ctx`), the JSON error helper (`json_error` or the inline `Json(json!(...))` pattern), the `AppState` field for total RAM (`state.total_ram_mb`), and the active-model reload used by the ctx handler (factor its reload tail into `reload_if_active(&state, repo, file)` and call it from both, or inline the same logic). Keep `handle_model_set_ctx` working by delegating: build a `SetProfileBody` with only `ctx` set and call the shared save+reload path.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test --lib set_model_profile_persists 2>&1 | tail -20`
Expected: PASS. Then run `cargo test 2>&1 | tail -25` — full suite green (the ctx endpoint test must still pass through the shared path).

- [ ] **Step 5: Commit**

```bash
git add src/server.rs
git commit -m "feat(server): POST /admin/model/profile sets exec profile

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Model-picker UI — KV, history, advanced gpu_layers

**Files:**
- Modify: `src/manager_ui/app.js` (drilldown panel + `saveProfile`)
- Modify: `src/manager_ui/style.css` (styling for the new controls)

**Interfaces:**
- Consumes: `ModelView.kv_current`, `kv_default`, `gpu_layers_current`, `history_turns_current`, `history_turns_default` (from `GET /admin/models`); `POST /admin/model/profile`.
- Produces: user-facing controls; no other module depends on this.

- [ ] **Step 1: Add the KV + history controls to the drilldown**

In `src/manager_ui/app.js`, in the per-model drilldown render (right after the existing `ctxBox` block, before `wrap.append(ctxBox)` or just after it), build a profile box reusing the `el(...)` helper and the `ctxbox` styling classes:

```js
  // --- KV cache quantization ---
  const kvBox = el("div", "ctxbox");
  kvBox.append(el("div", "ctxtitle", "KV cache"));
  const kvSel = el("select", "kvselect");
  [["q8", "Q8 · metade da RAM (padrão)"], ["f16", "F16 · máxima qualidade"], ["q4", "Q4 · menor RAM"]]
    .forEach(([val, label]) => {
      const opt = el("option", "", label);
      opt.value = val;
      if (val === m.kv_current) opt.selected = true;
      kvSel.append(opt);
    });
  kvBox.append(kvSel);
  kvBox.append(el("div", "ctxhint", `recomendado: ${m.kv_default}`));
  wrap.append(kvBox);

  // --- History window (turns) ---
  const histBox = el("div", "ctxbox");
  histBox.append(el("div", "ctxtitle", "Histórico (turnos)"));
  const histInput = el("input", "ctxinput");
  histInput.type = "number";
  histInput.min = 0;
  histInput.placeholder = "todos";
  if (m.history_turns_current != null) histInput.value = m.history_turns_current;
  histBox.append(histInput);
  histBox.append(el("div", "ctxhint", "0 ou vazio = manter tudo"));
  wrap.append(histBox);
```

- [ ] **Step 2: Add the collapsed “Avançado” gpu_layers control**

```js
  // --- Advanced: GPU layers (offload) ---
  const adv = el("details", "advbox");
  adv.append(el("summary", "advsummary", "Avançado"));
  const gpuInput = el("input", "ctxinput");
  gpuInput.type = "number";
  gpuInput.min = 0;
  gpuInput.placeholder = "todas";
  if (m.gpu_layers_current != null) gpuInput.value = m.gpu_layers_current;
  adv.append(el("div", "ctxtitle", "Camadas na GPU"));
  adv.append(gpuInput);
  adv.append(el("div", "ctxhint", "vazio = todas na GPU. Menos = tira pressão da Metal, porém mais lento."));
  wrap.append(adv);
```

- [ ] **Step 3: Add a “Salvar perfil” button that POSTs the profile**

After the controls, add one save button that collects the three values and posts them (ctx keeps its existing dedicated control + `saveCtx`):

```js
  const profActions = el("div", "actions");
  const saveProfBtn = el("button", "btn", "Salvar perfil");
  saveProfBtn.onclick = () => saveProfile(m, {
    kv_type: kvSel.value,
    history_turns: histInput.value === "" ? 0 : Number(histInput.value),
    gpu_layers: gpuInput.value === "" ? 4294967295 : Number(gpuInput.value), // u32::MAX = clear
  }, wrap);
  profActions.append(saveProfBtn);
  wrap.append(profActions);
```

Add the `saveProfile` function next to `saveCtx`:

```js
async function saveProfile(m, fields, wrap) {
  let res;
  try {
    res = await api("POST", "/admin/model/profile", { repo: m.repo, file: m.file, ...fields });
  } catch (e) {
    toast(`Falha ao salvar perfil: ${e.message || e}`);
    return;
  }
  // Active model reloads at the new profile — reuse the switch progress UX,
  // exactly as saveCtx does.
  handleReloadResponse(res, wrap); // use whatever saveCtx calls after a successful POST
}
```

Match `saveProfile`'s post-success handling to whatever `saveCtx` does (toast + reload/progress). If `saveCtx` inlines that logic, extract the shared tail into a helper and call it from both.

- [ ] **Step 4: Style the new controls**

In `src/manager_ui/style.css`, add rules mirroring `.ctxinput` / `.ctxbox`:

```css
.kvselect {
  width: 100%;
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid var(--border, #333);
  background: var(--bg, #1a1a1a);
  color: inherit;
}
.advbox { margin-top: 10px; }
.advsummary { cursor: pointer; opacity: 0.8; }
```

(Use the variable names / palette already present in `style.css`; if none, match the literal colors used by `.ctxbox`.)

- [ ] **Step 5: Manual verification**

Run: `cargo run --release -- --port 31415` and open the tray → model manager window. On a model's drilldown:
- Change KV to `q4`, set history to `3`, save → toast + (if active) reload.
- Reopen the drilldown → KV shows `q4`, history shows `3` (persisted).
- Restart the binary → values still present (persisted in `settings.json`).
- Under “Avançado”, set GPU layers to a small number on a large model, save, confirm it loads (slower) and the log shows `gpu_layers=Some(n)`.

- [ ] **Step 6: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -m "feat(ui): per-model KV, history, and gpu_layers controls

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** §1 data model → T1; §2 engine wiring → T4 (gpu_layers) + T5 (per-model kv on reload); §3 resolution → T3 + T5; §4 catalog → T2; §5 fit → T2 (per-entry kv; gpu_layers advisory, no verdict change per Non-goals); §6 request shaping → T6 + T7; §7 endpoints → T8; §8 UI → T9. All sections mapped.
- **KvType naming:** `crate::config::KvType` throughout; `KvCacheType` is the llama type, mapped only in `engine_llama` and the `kv_type_to_llama` helper (T5).
- **Signature consistency:** `LlamaEngine::load` gains `gpu_layers: Option<u32>` before `total_ram_mb` (T4), and both call sites in `lib.rs` are updated in the same feature branch (T5) — the crate is only guaranteed to compile again at the end of T5, called out explicitly in T4 Step 4.
- **`catalog_view`** gains a trailing `profile_override` closure; every caller (tests + `server.rs` catalog handler + `lib.rs`) must pass it. T2 updates the test callers; the production caller in `server.rs` (`handle_models_catalog`, ~L514) must also pass `|r, f| crate::settings::load_model_profile(&crate::settings::model_ctx_key(r, f))` — handled in T2 Step 6, committed with `src/server.rs` in T2 Step 8.
