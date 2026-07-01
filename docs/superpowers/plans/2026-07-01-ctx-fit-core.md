# Context-Fit Core + Load-Time Ctx Auto-Clamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the largest context a model fits in the device memory budget and clamp the requested `ctx_len` down to it at load time, so a too-large model (Phi-4 14B at ctx 32768 on 16 GB) runs instead of OOMing.

**Architecture:** A pure `src/fit.rs` module does the KV-cache/context math. The llama worker thread, after the model opens, reads the model's real dimensions, computes the fitting max context, clamps the requested ctx (or refuses if it can't fit at the minimum), and reports the effective ctx back so the router's window matches reality.

**Tech Stack:** Rust, llama-cpp-2 (model metadata accessors), sysinfo (total RAM), tracing.

## Global Constraints

- Build/test command prefix: `MISTRALRS_METAL_PRECOMPILE=0` on every `cargo` invocation.
- `fit.rs` is **pure** — no llama/ggml calls, no I/O, no globals. It takes numbers and returns numbers so it is fully unit-testable. Its own `KvKind` enum decouples it from `llama_cpp_2::KvCacheType`.
- Constants (exact): `MIN_CTX = 2048`, `DEFAULT_SMALL_CTX = 8192`, `GLOBAL_MAX_CTX = 32768`, `COMPUTE_HEADROOM_MB = 1024`, `METAL_BUDGET_PCT = 78`, `CPU_BUDGET_PCT = 65`.
- Bytes per KV element: `F16 = 2.0`, `Q8 = 1.0625`, `Q4 = 0.5625`.
- `kv_bytes_per_token = 2 * n_layer * n_head_kv * head_dim * bytes_per_elem`, rounded to `u64`. `head_dim = n_embd / n_head`.
- Budget refinement vs the spec: the spec suggested querying the ggml GPU device's `memory_total`. This plan instead uses a **RAM fraction** (`total_ram_mb * METAL_BUDGET_PCT/100`) — it is pure/testable, deterministic, and on this Apple unified-memory target ~78% of RAM ≈ the observed Metal working-set cap (16384 MB → 12780 ≈ the logged 12713 MB). No ggml device query.
- A clamp only ever *lowers* ctx, never raises it. If the request already fits, nothing changes and no warning fires.
- Never auto-pick more than `GLOBAL_MAX_CTX` even if more fits.

---

### Task 1: pure `fit` module

**Files:**
- Create: `src/fit.rs`
- Modify: `src/lib.rs` (add `pub mod fit;` alongside the other module declarations)
- Test: inline `#[cfg(test)]` in `src/fit.rs`

**Interfaces:**
- Produces:
  - `pub enum KvKind { F16, Q8, Q4 }`
  - `pub fn kv_bytes_per_token(n_layer: u32, n_head_kv: u32, head_dim: u32, kv: KvKind) -> u64`
  - `pub fn est_kv_bytes_per_token(params_b: f32, kv: KvKind) -> u64`
  - `pub fn max_ctx_fit(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> u32`
  - `pub struct CtxBounds { pub min: u32, pub default: u32, pub max: u32 }`
  - `pub fn ctx_bounds(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> CtxBounds`
  - `pub fn device_budget_mb(total_ram_mb: u64, gpu_present: bool) -> u32`

- [ ] **Step 1: Write the module with failing tests (bodies stubbed to `unimplemented!()`)**

Create `src/fit.rs`:

```rust
//! Pure context-fit math: how much memory a model's KV cache costs at a given
//! context length, and the largest context that fits a memory budget. No I/O,
//! no llama/ggml calls — just numbers in, numbers out, so it is fully testable.

/// KV-cache element storage, decoupled from `llama_cpp_2::KvCacheType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvKind {
    F16,
    Q8,
    Q4,
}

/// Bytes per stored KV element for each quantization (llama.cpp sizes).
fn bytes_per_elem(kv: KvKind) -> f64 {
    match kv {
        KvKind::F16 => 2.0,
        KvKind::Q8 => 1.0625, // q8_0: 34 bytes / 32 elems
        KvKind::Q4 => 0.5625, // q4_0: 18 bytes / 32 elems
    }
}

pub const MIN_CTX: u32 = 2048;
pub const DEFAULT_SMALL_CTX: u32 = 8192;
pub const GLOBAL_MAX_CTX: u32 = 32768;
pub const COMPUTE_HEADROOM_MB: u32 = 1024;
pub const METAL_BUDGET_PCT: u64 = 78;
pub const CPU_BUDGET_PCT: u64 = 65;

/// KV-cache bytes per token, exact from the model's dimensions.
/// `2` covers both the K and V caches.
pub fn kv_bytes_per_token(n_layer: u32, n_head_kv: u32, head_dim: u32, kv: KvKind) -> u64 {
    let elems = 2.0 * n_layer as f64 * n_head_kv as f64 * head_dim as f64;
    (elems * bytes_per_elem(kv)).round() as u64
}

/// Coarse KV bytes/token from parameter count alone, for catalog entries whose
/// GGUF metadata is not yet available (pre-download). Tuned so a typical dense
/// transformer lands within ~2x of the exact value. `ELEMS_PER_TOKEN_PER_B`
/// (~6700) is the empirical `2*n_layer*n_head_kv*head_dim / params_b` for common
/// models (Qwen 3B ≈ 6144, Phi-4 14B ≈ 7314).
pub fn est_kv_bytes_per_token(params_b: f32, kv: KvKind) -> u64 {
    const ELEMS_PER_TOKEN_PER_B: f64 = 6700.0;
    let elems = ELEMS_PER_TOKEN_PER_B * params_b as f64;
    (elems * bytes_per_elem(kv)).round() as u64
}

/// Largest context (multiple of 256, capped at `n_ctx_train` and `GLOBAL_MAX_CTX`)
/// whose `weights + KV(ctx) + COMPUTE_HEADROOM` fits in `budget_mb`. Returns 0 if
/// even `MIN_CTX` does not fit.
pub fn max_ctx_fit(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> u32 {
    let overhead = weights_mb.saturating_add(COMPUTE_HEADROOM_MB);
    let avail_mb = budget_mb.saturating_sub(overhead);
    if avail_mb == 0 || kv_per_token_bytes == 0 {
        return 0;
    }
    let avail_bytes = avail_mb as u64 * 1024 * 1024;
    let by_mem = (avail_bytes / kv_per_token_bytes) as u64;
    let cap = by_mem.min(n_ctx_train as u64).min(GLOBAL_MAX_CTX as u64);
    let rounded = (cap / 256) * 256; // floor to a multiple of 256
    if rounded < MIN_CTX as u64 {
        0
    } else {
        rounded as u32
    }
}

/// The three ctx values surfaced for a model. `max == 0` means "won't fit even
/// at MIN_CTX" — the caller refuses the load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtxBounds {
    pub min: u32,
    pub default: u32,
    pub max: u32,
}

/// `max = max_ctx_fit(...)`; `min = MIN_CTX`; `default = min(DEFAULT_SMALL_CTX, max)`.
/// When the model won't fit, `default` and `max` are both 0.
pub fn ctx_bounds(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> CtxBounds {
    let max = max_ctx_fit(weights_mb, kv_per_token_bytes, budget_mb, n_ctx_train);
    if max == 0 {
        return CtxBounds { min: MIN_CTX, default: 0, max: 0 };
    }
    CtxBounds { min: MIN_CTX, default: DEFAULT_SMALL_CTX.min(max), max }
}

/// Usable memory budget in MB: `METAL_BUDGET_PCT` of RAM when a GPU is present
/// (Apple unified memory ≈ the Metal working-set cap), else `CPU_BUDGET_PCT`.
pub fn device_budget_mb(total_ram_mb: u64, gpu_present: bool) -> u32 {
    let pct = if gpu_present { METAL_BUDGET_PCT } else { CPU_BUDGET_PCT };
    (total_ram_mb * pct / 100) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_bytes_per_token_phi4_q8() {
        // Phi-4: n_layer 40, n_head_kv 10, head_dim 128, Q8.
        // 2*40*10*128 = 102400 elems * 1.0625 = 108800 bytes.
        assert_eq!(kv_bytes_per_token(40, 10, 128, KvKind::Q8), 108800);
    }

    #[test]
    fn kv_bytes_scales_with_kv_kind() {
        let f16 = kv_bytes_per_token(40, 10, 128, KvKind::F16);
        let q8 = kv_bytes_per_token(40, 10, 128, KvKind::Q8);
        let q4 = kv_bytes_per_token(40, 10, 128, KvKind::Q4);
        assert!(f16 > q8 && q8 > q4);
        assert_eq!(f16, 204800); // 102400 * 2.0
    }

    #[test]
    fn est_kv_within_2x_of_exact() {
        // Phi-4 exact 108800; heuristic from 14B should be within 2x.
        let est = est_kv_bytes_per_token(14.0, KvKind::Q8) as f64;
        let exact = 108800.0;
        assert!(est > exact / 2.0 && est < exact * 2.0, "est={est} exact={exact}");
    }

    #[test]
    fn max_ctx_fit_caps_at_global_max_for_small_model() {
        // 3B (~2000 MB weights), tiny KV, big budget → capped at GLOBAL_MAX_CTX.
        let kv = kv_bytes_per_token(36, 2, 128, KvKind::Q8); // Qwen 3B ~19584 B/tok
        let m = max_ctx_fit(2000, kv, 12780, 131072);
        assert_eq!(m, GLOBAL_MAX_CTX);
    }

    #[test]
    fn max_ctx_fit_phi4_16gb_caps_at_ctx_train() {
        // Phi-4 on 16GB Metal: weights ~8634, budget 12780, trained ctx 16384.
        // Memory would allow ~30k, but n_ctx_train (16384) binds first — and
        // that is well under the requested 32768, so the load will clamp.
        let kv = kv_bytes_per_token(40, 10, 128, KvKind::Q8); // 108800 B/tok
        let m = max_ctx_fit(8634, kv, 12780, 16384);
        assert_eq!(m, 16384);
    }

    #[test]
    fn max_ctx_fit_memory_bound_below_ctx_train() {
        // Same big model but a large trained ctx → MEMORY binds, not n_ctx_train.
        // avail = 12780 - (8634+1024) = 3122 MB → 3122MiB/108800B ≈ 30088 tokens
        // → floored to a multiple of 256 (29952), and below GLOBAL_MAX_CTX.
        let kv = kv_bytes_per_token(40, 10, 128, KvKind::Q8);
        let m = max_ctx_fit(8634, kv, 12780, 131072);
        assert!(m < GLOBAL_MAX_CTX && m >= 29000, "got {m}");
        assert_eq!(m % 256, 0);
    }

    #[test]
    fn max_ctx_fit_zero_when_weights_exceed_budget() {
        assert_eq!(max_ctx_fit(13000, 108800, 12780, 16384), 0);
    }

    #[test]
    fn ctx_bounds_default_is_small_and_within_max() {
        let kv = kv_bytes_per_token(36, 2, 128, KvKind::Q8);
        let b = ctx_bounds(2000, kv, 12780, 131072);
        assert_eq!(b.min, MIN_CTX);
        assert_eq!(b.max, GLOBAL_MAX_CTX);
        assert_eq!(b.default, DEFAULT_SMALL_CTX); // 8192 < 32768
    }

    #[test]
    fn ctx_bounds_wont_fit_reports_zero() {
        let b = ctx_bounds(13000, 108800, 12780, 16384);
        assert_eq!(b.max, 0);
        assert_eq!(b.default, 0);
        assert_eq!(b.min, MIN_CTX);
    }

    #[test]
    fn device_budget_gpu_vs_cpu() {
        assert_eq!(device_budget_mb(16384, true), 12779);  // 78%
        assert_eq!(device_budget_mb(16384, false), 10649); // 65%
    }
}
```

- [ ] **Step 2: Register the module**

In `src/lib.rs`, add `pub mod fit;` next to the other `pub mod` lines (e.g. after `pub mod engine_llama;`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib fit::`
Expected: compile error / failures because bodies are stubbed. (If you wrote the real bodies directly from the Global Constraints, skip to Step 4 — the code above is the reference implementation.)

- [ ] **Step 4: Ensure the implementations above are in place, run tests to verify they pass**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib fit::`
Expected: PASS (10 tests). Output pristine.

- [ ] **Step 5: Commit**

```bash
git add src/fit.rs src/lib.rs
git commit -m "feat(fit): pure context-fit math (KV sizing, max_ctx_fit, ctx_bounds)"
```

---

### Task 2: load-time ctx clamp in the llama worker

**Files:**
- Modify: `src/engine_llama.rs` (`LlamaEngine` struct + `load` + `worker_thread`)
- Test: none automated (Metal-runtime); manual integration verification in Step 6

**Interfaces:**
- Consumes: `crate::fit::{KvKind, kv_bytes_per_token, ctx_bounds, device_budget_mb}` (Task 1).
- Produces (for Task 3):
  - `LlamaEngine::load(model_id, gguf_files, ctx_len, kv_cache_type, kv_cache_dir, total_ram_mb: u64) -> Result<Self>` — **new trailing `total_ram_mb` param**.
  - `LlamaEngine::ctx_window(&self) -> usize` — the effective (post-clamp) context.

- [ ] **Step 1: Add the effective-ctx field + accessor to `LlamaEngine`**

Find the `LlamaEngine` struct (holds `tx`). Add a field and accessor:

```rust
pub struct LlamaEngine {
    tx: std_mpsc::SyncSender<Job>,
    ctx_window: usize,
}

impl LlamaEngine {
    /// The context length the worker actually created (after any fit clamp).
    pub fn ctx_window(&self) -> usize {
        self.ctx_window
    }
}
```

(Adjust the existing `LlamaEngine { tx }` construction at the end of `load` to `LlamaEngine { tx, ctx_window }` — see Step 3.)

- [ ] **Step 2: Change `load` — add `total_ram_mb`, split provenance, receive effective ctx**

In `LlamaEngine::load`:
1. Add the trailing param `total_ram_mb: u64` to the signature.
2. Change the oneshot channel type to carry the effective ctx:
   `let (load_tx, load_rx) = tokio::sync::oneshot::channel::<Result<u32>>();`
3. Replace the `provenance_prefix` construction (the `format!("{}-{}-{}-pf{}", model_tag, kv_tag, ctx_len, PROMPT_FORMAT_VERSION)`) with a **base without ctx** — the worker appends the effective ctx after clamping:
   `let provenance_base = format!("{}-{}-pf{}", model_tag, kv_tag, PROMPT_FORMAT_VERSION);`
4. Pass `total_ram_mb` and `provenance_base` into `worker_thread` (see Step 4 signature).
5. Await the effective ctx and build the struct with it:

```rust
let effective_ctx = load_rx
    .await
    .context("worker thread dropped load channel without signalling")??;

Ok(LlamaEngine { tx, ctx_window: effective_ctx as usize })
```

- [ ] **Step 3: Update `worker_thread` — signature, clamp, effective provenance**

Change the `worker_thread` signature to take `total_ram_mb` and `provenance_base` (renamed from `provenance_prefix`) and the new `load_tx` type:

```rust
fn worker_thread(
    model_path: PathBuf,
    ctx_len: u32,
    kv_cache_type: KvCacheType,
    provenance_base: String,
    kv_cache_dir: Option<PathBuf>,
    total_ram_mb: u64,
    rx: std_mpsc::Receiver<Job>,
    load_tx: tokio::sync::oneshot::Sender<Result<u32>>,
) {
```

After `LlamaModel::load_from_file` succeeds (the `let model = match ... { Ok(m) => m, Err(e) => { let _ = load_tx.send(Err(e)); return; } };` block) and **before** the `make_ctx_params` call, insert the clamp:

```rust
    // --- Fit the requested context to the device memory budget ---
    // Compute the largest context that fits (weights + KV + compute headroom)
    // and clamp the request down to it, instead of OOMing at decode time.
    let n_head = model.n_head();
    let head_dim = if n_head > 0 { model.n_embd() as u32 / n_head } else { 0 };
    let kv_kind = match kv_cache_type {
        KvCacheType::Q8_0 => crate::fit::KvKind::Q8,
        KvCacheType::Q4_0 => crate::fit::KvKind::Q4,
        _ => crate::fit::KvKind::F16,
    };
    let kv_per_token = crate::fit::kv_bytes_per_token(
        model.n_layer(),
        model.n_head_kv(),
        head_dim,
        kv_kind,
    );
    let weights_mb = std::fs::metadata(&model_path)
        .map(|m| (m.len() / (1024 * 1024)) as u32)
        .unwrap_or(0);
    // n_gpu_layers is u32::MAX (all layers on GPU) → Metal budget applies.
    let budget_mb = crate::fit::device_budget_mb(total_ram_mb, true);
    let bounds = crate::fit::ctx_bounds(weights_mb, kv_per_token, budget_mb, model.n_ctx_train());
    if bounds.max == 0 {
        let _ = load_tx.send(Err(anyhow::anyhow!(
            "model needs more memory than is available even at the minimum context \
             (budget {budget_mb} MB, weights {weights_mb} MB): pick a smaller model or free RAM"
        )));
        return;
    }
    let effective_ctx = ctx_len.clamp(bounds.min, bounds.max);
    if effective_ctx < ctx_len {
        tracing::warn!(
            target: "localllm::llama",
            "ctx {ctx_len} → {effective_ctx} to fit memory budget \
             (budget {budget_mb} MB, weights {weights_mb} MB, KV {kv_per_token} B/tok)",
        );
        crate::usage::notify(
            "localllm — context reduced",
            &format!("Context reduced to {effective_ctx} tokens to fit available memory."),
        );
    }
    let provenance_prefix = format!("{provenance_base}-ctx{effective_ctx}");
    let ctx_len = effective_ctx; // shadow: all downstream uses the fitted value
```

The existing code below already uses `ctx_len` for `make_ctx_params(ctx_len, kv_cache_type)` and the tracing line — the shadow makes them use the effective value automatically. The existing `provenance_prefix` variable is now produced here (delete the old parameter usage; it is built locally).

Finally, change the load-success signal from `load_tx.send(Ok(()))` to send the effective ctx:

```rust
    if load_tx.send(Ok(ctx_len)).is_err() {
        return;
    }
```

- [ ] **Step 4: Update the internal `worker_thread` call in `load` to pass the new args**

The `std::thread::spawn(move || { worker_thread(path, ctx_len_u32, kv_cache_type, provenance_prefix, kv_cache_dir, rx, load_tx); });` becomes:

```rust
std::thread::spawn(move || {
    worker_thread(path, ctx_len_u32, kv_cache_type, provenance_base, kv_cache_dir, total_ram_mb, rx, load_tx);
});
```

- [ ] **Step 5: Build**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo build`
Expected: compile errors at the two `LlamaEngine::load(...)` call sites in `src/lib.rs` (missing `total_ram_mb`) — those are fixed in Task 3. To build Task 2 in isolation, temporarily pass `0` at both call sites, or proceed to Task 3 and build together. Note this in your report.

- [ ] **Step 6: Manual integration verification (Metal runtime)**

After Task 3 wires the call sites, run the debug binary against Phi-4 at the default ctx and confirm the clamp + no OOM:

```bash
LOCALLLM_LOG=/tmp/phi-clamp.log MISTRALRS_METAL_PRECOMPILE=0 ./target/debug/localllm \
  --model-id bartowski/phi-4-GGUF --gguf-file phi-4-Q4_K_M.gguf \
  --ctx-len 32768 --port 31520 --no-kv-persist &
# wait for "listening", then:
curl -s http://127.0.0.1:31520/v1/messages -H 'content-type: application/json' \
  -d '{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'
grep -E "ctx 32768 →|route:|Insufficient Memory" /tmp/phi-clamp.log
```
Expected: a `ctx 32768 → 16384` warn line (Phi-4's trained context is 16384, which binds before memory), a **successful** JSON reply (no `Insufficient Memory`, no `ctx.decode failed`), and the `route:` line showing `ctx_window=16384`. Record the output in your report.

- [ ] **Step 7: Commit**

```bash
git add src/engine_llama.rs
git commit -m "feat(engine): clamp ctx to the device memory budget at load (fixes Phi-4 OOM)"
```

---

### Task 3: wire total RAM in + effective ctx to the router window

**Files:**
- Modify: `src/lib.rs` (`run_server_with_ready_policy_token`: move RAM query up, pass to both loads, set router window from effective ctx)
- Test: existing suite compiles + passes; manual verify from Task 2 Step 6

**Interfaces:**
- Consumes: `LlamaEngine::load(.., total_ram_mb)` and `LlamaEngine::ctx_window()` (Task 2).

- [ ] **Step 1: Move the `total_ram_mb` computation above the initial engine load**

Cut the `let total_ram_mb = { use sysinfo::System; ... };` block (currently after the manager is built) and paste it **before** the `let engine = match cfg.backend { ... }` initial-load match. Keep the `if total_ram_mb == 0 { tracing::warn!(...) }` right after it.

- [ ] **Step 2: Thread `total_ram_mb` into the initial load and capture the effective ctx**

Rewrite the `Backend::Llama` arm of the initial-load match so it binds the engine, reads its effective ctx, then wraps it — and set a shared `effective_ctx` for both backends:

```rust
let (engine, effective_ctx): (Arc<dyn Generator>, usize) = match cfg.backend {
    Backend::Llama => {
        let kv_cache_type = cfg.llama_kv_cache_type();
        let kv_cache_dir = cfg.resolved_kv_cache_dir();
        tracing::info!("KV cache type: --kv-type={:?}", cfg.kv_type);
        tracing::info!(
            "KV persist dir: {:?} (no-persist={})",
            kv_cache_dir,
            cfg.no_kv_persist
        );
        let llama = LlamaEngine::load(
            &cfg.model_id,
            &cfg.gguf_files,
            cfg.ctx_len,
            kv_cache_type,
            kv_cache_dir,
            total_ram_mb,
        )
        .await?;
        let eff = llama.ctx_window();
        (Arc::new(llama) as Arc<dyn Generator>, eff)
    }
    Backend::Mistralrs => (
        Arc::new(Engine::load(&cfg.engine_config()).await?) as Arc<dyn Generator>,
        cfg.ctx_len,
    ),
};
```

(The `let engine = ...;` binding is replaced by the tuple `let (engine, effective_ctx) = ...;`.)

- [ ] **Step 3: Thread `total_ram_mb` into the switch builder closure**

Before the `let builder: EngineBuilder = Box::new(move |spec| { ... })`, capture the RAM:

```rust
let b_total_ram_mb = total_ram_mb;
```

Inside the closure, pass it to the switched `LlamaEngine::load`:

```rust
let engine =
    LlamaEngine::load(&spec.repo, &[spec.file], b_ctx_len, b_kv_type, kv_dir, b_total_ram_mb).await?;
```

- [ ] **Step 4: Pass the effective ctx as the router's local window**

In the `router(...)` call, replace the `cfg.ctx_len` argument (the `local_ctx_window` position) with `effective_ctx`:

```rust
let app = router(
    manager,
    cfg.model_id.clone(),
    policy,
    effective_ctx,      // was cfg.ctx_len — now the fitted window
    usage,
    cfg.cloud_token_alert,
    admin_token.clone(),
    total_ram_mb,
);
```

- [ ] **Step 5: Build + full suite + clippy**

Run:
```
MISTRALRS_METAL_PRECOMPILE=0 cargo build
MISTRALRS_METAL_PRECOMPILE=0 cargo test
MISTRALRS_METAL_PRECOMPILE=0 cargo clippy --all-targets
```
Expected: build clean; all tests pass (the fit unit tests plus the existing suite; no behavior change for the existing integration tests, which use `router_for_test*` and never hit `LlamaEngine::load`); no new clippy warnings in `fit`/`engine_llama`/`lib` (pre-existing SSE `to_string` / tray warnings remain).

- [ ] **Step 6: Run the Task 2 Step 6 manual verification now that call sites compile**

Follow Task 2 Step 6. Confirm the Phi-4 clamp + successful reply + clamped `route:` window. Record output.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs
git commit -m "feat(server): pass total RAM to load, use fitted ctx as the router window"
```

---

## Notes for the executor
- The clamp in Task 2 applies to **every** llama load — the initial one and every hot-swap through the builder closure — so a swap to a too-large model is also fixed (it clamps instead of OOMing). Propagating a *swapped* model's effective ctx into `AppState.local_ctx_window` (which is fixed at router-build time) is **out of scope** here and belongs with sub-project 2's per-model ctx work; note it, don't build it.
- `LlamaEngine::load` is `async` and its two call sites are both in `src/lib.rs`. There are no other callers (the smoke test uses its own path). If you find another caller, stop and report it.
- Do not change the default `cfg.ctx_len` (32768). It is now a *ceiling request* that the loader fits to the device, not a guarantee.
