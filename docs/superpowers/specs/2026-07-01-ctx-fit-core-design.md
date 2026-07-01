# Context-fit core + load-time ctx auto-clamp — design

**Date:** 2026-07-01
**Status:** approved (design)
**Sub-project:** 1 of 3 in the "ctx-aware model fit" feature. Delivers the actual
fix for the Phi-4 crash. Sub-project 2 (per-model ctx persistence + KV-aware
catalog) and 3 (Model Manager page) build on the pure `fit` module defined here.

## Why

Loading Phi-4 14B (8.4 GB weights) at the default `ctx_len` 32768 on a 16 GB Mac
allocates weights + Q8 KV cache (~3.3 GB) + compute buffer past the Metal
working-set budget (`recommendedMaxWorkingSetSize` ≈ 12.7 GB). Decode then fails
with `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` — every
request errors (and, before the recovery fix, bricked the worker). At ctx 4096
the same model runs fine.

The root problem: nothing accounts for the **KV-cache cost of the chosen
context** when picking `ctx_len`. The fix is to compute, from the model's real
dimensions and the device's memory budget, the largest context that fits, and
**clamp the requested ctx down to it at load time** instead of OOMing.

This sub-project builds the pure fit math and wires the load-time clamp. It does
not add per-model persistence or UI (sub-projects 2/3).

## Scope

### In scope
- New pure module `src/fit.rs`: KV-cache sizing (exact + heuristic), device
  budget, and `ctx_bounds` → `{ min, default, max }`.
- Load-time clamp in the llama worker: after the model opens, reduce the
  requested `ctx_len` to the fitting maximum; log a `WARN` and fire a tray
  notification when reduced; refuse the load with a clear error if the model
  cannot fit even at the minimum context.
- Propagate the **effective** (post-clamp) ctx so the router's
  `local_ctx_window` and the KV-state provenance key reflect what actually
  loaded, not what was requested.

### Out of scope (sub-projects 2/3)
- Persisting a per-model ctx override; the Model Manager page; the KV-aware
  catalog fit verdict and recommendation. (The `fit` module exposes the
  functions those will call, but nothing here reads/writes settings or changes
  `catalog_view`.)
- CPU-only precise budgeting beyond the RAM-fraction fallback.

## Architecture

```
LlamaModel opens (worker_thread)
      │  n_layer, n_head_kv, n_embd, n_ctx_train  (llama metadata)
      │  device budget (ggml GPU memory_total, else RAM fraction)
      ▼
fit::ctx_bounds(model dims, weights_mb, kv_type, budget) → { min, default, max }
      │
      ▼
effective_ctx = requested_ctx.clamp(min ..= max)   (WARN + notify if reduced;
                                                     error if max < min)
      ▼
make_ctx_params(effective_ctx, kv_type) → LlamaContext
      ▼
effective_ctx reported back → AppState.local_ctx_window + provenance key
```

### Module `src/fit.rs` (pure, unit-tested)

Bytes-per-element by KV type (llama.cpp quant sizes):
- `F16` → 2.0, `Q8_0` → 1.0625 (34 B / 32 elems), `Q4_0` → 0.5625 (18 B / 32).

```rust
/// KV-cache bytes per token for one model, exact from its dimensions.
/// = 2 (K and V) × n_layer × n_head_kv × head_dim × bytes_per_elem(kv_type).
pub fn kv_bytes_per_token(n_layer: u32, n_head_kv: u32, head_dim: u32, kv: KvKind) -> u64;

/// Approximate KV bytes/token from parameter count alone, for catalog entries
/// whose GGUF metadata is not available (pre-download). Coarse, labelled an
/// estimate by callers. Tuned so Q8 ≈ 0.007 MB/token per billion params.
pub fn est_kv_bytes_per_token(params_b: f32, kv: KvKind) -> u64;

/// Largest ctx (rounded down to a multiple of 256, capped at n_ctx_train and
/// GLOBAL_MAX_CTX) whose weights + KV(ctx) + COMPUTE_HEADROOM_MB fit in budget.
/// Returns 0 if even MIN_CTX does not fit.
pub fn max_ctx_fit(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> u32;

/// The three ctx values surfaced for a model.
pub struct CtxBounds { pub min: u32, pub default: u32, pub max: u32 }

/// min = MIN_CTX; max = max_ctx_fit(...); default = min(DEFAULT_SMALL_CTX, max).
/// If max < MIN_CTX, `max` is returned as 0 (caller treats as "won't fit").
pub fn ctx_bounds(weights_mb: u32, kv_per_token_bytes: u64, budget_mb: u32, n_ctx_train: u32) -> CtxBounds;

/// Device memory budget in MB: the ggml GPU device's memory_total (on Metal this
/// is the working-set cap ~= recommendedMaxWorkingSetSize) minus COMPUTE_HEADROOM_MB;
/// if no GPU device, total_ram_mb × CPU_BUDGET_FRAC.
pub fn device_budget_mb(total_ram_mb: u64) -> u32;
```

Constants (in the module, documented):
- `MIN_CTX = 2048`
- `DEFAULT_SMALL_CTX = 8192` — the small, safe per-model default (a large ctx is
  not "better"; it only costs KV RAM — so default conservatively and let the
  user raise it in sub-project 3).
- `GLOBAL_MAX_CTX = 32768` — we never auto-pick more than this even if it fits.
- `COMPUTE_HEADROOM_MB = 1024` — reserve for the decode compute graph (measured
  ~824 MB at n_ubatch 2048 for a 14B; round up).
- `CPU_BUDGET_FRAC = 0.65` — fallback budget fraction when there is no GPU
  device (matches the catalog's existing conservative fraction).

`head_dim = n_embd / n_head`. `weights_mb` = the GGUF file size in MB (resident
weight footprint ≈ file size for a memory-mapped Q4 GGUF).

### Load-time clamp (`src/engine_llama.rs::worker_thread`)

After `LlamaModel::load_from_file` succeeds and before `make_ctx_params`:

1. Read `n_layer`, `n_head_kv`, `n_embd`, `n_head` (→ `head_dim`), `n_ctx_train`
   from the model; `weights_mb` from the file size; `budget_mb` from
   `fit::device_budget_mb(total_ram_mb)`.
2. `bounds = fit::ctx_bounds(weights_mb, kv_bytes_per_token(...), budget_mb, n_ctx_train)`.
3. If `bounds.max == 0` (or `< MIN_CTX`): send `Err` on `load_tx` with a clear
   message ("model needs more memory than is available even at the minimum
   context; pick a smaller model") — the manager surfaces it (503/notify).
4. `effective_ctx = requested_ctx.clamp(bounds.min, bounds.max)`.
5. If `effective_ctx < requested_ctx`: `tracing::warn!` (e.g.
   `"ctx 32768 → 24576 to fit Metal budget (12713 MB)"`) and fire the tray
   notification used elsewhere (best-effort; test-gated like other notifies).
6. Build the context with `effective_ctx`; use it for `provenance_prefix` and
   for the value reported back so the router sees the real window.

`total_ram_mb` is already available to the load path (sysinfo, used by the
catalog). Thread it into `LlamaEngine::load`/`worker_thread`.

### Effective-ctx propagation

`worker_thread` currently signals `load_tx: oneshot::Sender<Result<()>>`. Change
the success payload to carry the effective ctx: `Result<u32>` (or a small
`LoadOk { effective_ctx: u32 }`). `LlamaEngine::load` returns it; the caller
(`lib.rs`) sets `AppState.local_ctx_window = effective_ctx` instead of the
requested `config.ctx_len`. This keeps difficulty/gate math honest after a
clamp.

## Error handling
- Model cannot fit at `MIN_CTX` → `Err` on load (clear message), no context
  created, no OOM. The manager reports load failure as it already does.
- Device query returns no GPU device (pure CPU) → RAM-fraction budget; the clamp
  still applies.
- A clamp never *raises* ctx above the request; it only lowers it. If the
  request already fits, nothing changes and no warning fires.

## Testing
Pure unit tests in `fit.rs` (no model, no GPU):
- `kv_bytes_per_token`: exact value for known dims (e.g. Phi-4: n_layer 40,
  n_head_kv 10, head_dim 128, Q8 → ~108800 B/token) within rounding.
- `max_ctx_fit`: monotonic — a bigger budget yields a ≥ ctx; capped at
  `n_ctx_train` and `GLOBAL_MAX_CTX`; returns 0 when weights alone exceed
  budget − headroom.
- `ctx_bounds`: `default = min(DEFAULT_SMALL_CTX, max)`; `min = MIN_CTX`;
  Phi-4-on-16GB scenario yields `max` in the ~24–28k range (fits) and 32768
  does **not** (would need > budget); a 3B model yields `max` ≥ 32768 → capped
  at `GLOBAL_MAX_CTX`, `default` 8192.
- `est_kv_bytes_per_token`: within ~2× of the exact value for a known model
  (heuristic is coarse by design).
- `device_budget_mb`: with an injected total_ram, the CPU fallback returns
  `ram × 0.65`. (The GPU/Metal path is validated by the integration run below,
  not a unit test.)

Integration verification (manual, recorded — Metal runtime can't be unit
tested): load Phi-4 with `--ctx-len 32768`; confirm the log shows the clamp
(`ctx 32768 → NNNNN`), the model **serves a request without OOM**, and the
router's `route:` line shows the clamped `ctx_window`.

## Files touched
- `src/fit.rs` — **new** (pure math + constants + unit tests).
- `src/lib.rs` — `mod fit;`; set `AppState.local_ctx_window` from the effective
  ctx returned by the load path.
- `src/engine_llama.rs` — clamp in `worker_thread`; thread `total_ram_mb` into
  `LlamaEngine::load`/`worker_thread`; change `load_tx` payload to carry the
  effective ctx; use effective ctx for `make_ctx_params` and `provenance_prefix`.
- `src/config.rs` — none required (default `ctx_len` stays 32768; it is now a
  *ceiling request*, not a guarantee).

## Follow-up (sub-projects 2 & 3)
- **2:** persist a per-model ctx override (settings), compute `ctx_bounds` for
  catalog display using `est_kv_bytes_per_token` (pre-download) / exact
  (downloaded), make `catalog_view` fit + recommendation KV-aware, add an admin
  endpoint to set per-model ctx.
- **3:** Model Manager page shows per model the `{min, default, max}` and an
  editable ctx input bounded to `[min, max]`, saving via the sub-project 2
  endpoint.
