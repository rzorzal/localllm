# Per-model ctx + KV-aware catalog — design

**Date:** 2026-07-01
**Status:** approved (design)
**Sub-project:** 2 of 3 in the "ctx-aware model fit" feature. Builds on
sub-project 1's pure `src/fit.rs` (KV sizing, `ctx_bounds`, `device_budget_mb`,
`est_kv_bytes_per_token`) and its load-time clamp. Sub-project 3 (the Model
Manager page) consumes the fields and endpoint defined here.

## Why

Sub-project 1 fits the *loaded* context to the device. This sub-project makes the
context **per-model and user-controllable**, and makes the catalog's fit verdict
and recommendation **account for the KV cache** at each model's context — today
`catalog_view` estimates RAM as `size_mb * 1.2`, ignoring the KV cost, so it
green-lights models that then clamp or are tight. The user can also set a
per-model context that persists and takes effect immediately.

## Decisions (locked)

- **Default stays 32768 as a ceiling.** A model with no user override requests
  `cfg.ctx_len` (32768) and sub-project 1 clamps it to what fits. The small
  per-model default is a *suggestion* shown on the page; it only takes effect
  when the user saves it as an override.
- **Saving a ctx for the active model reloads it immediately** via the existing
  switch machinery (drain → rebuild at the new ctx), returning 202.
- **Override storage is opt-in and persisted**, keyed by `"repo/file"`.

## Scope

### In scope
- `settings.rs`: a persisted `model_ctx: BTreeMap<String, u32>` map (per-model
  override), with load/save/clear that preserve the other settings sections.
- `catalog.rs`: a curated `ctx_train` field on `CatalogEntry`; a KV-aware
  `catalog_view` that includes KV in the RAM estimate at each model's effective
  context and exposes `ctx_min/ctx_default/ctx_max/ctx_current` per model.
- `server.rs`: `POST /admin/model/ctx` (token-guarded) to set/clear a per-model
  ctx, reloading the active model when it is the target; load-path resolution of
  the requested ctx from the override.
- `AppState`: carry the requested-ctx ceiling and the KV kind so the catalog
  endpoint and the load path can compute bounds.

### Out of scope (sub-project 3)
- The Model Manager page UI (shows the new fields + the bounded ctx input, calls
  the endpoint). This sub-project is backend + data only.
- Reading exact `n_ctx_train` from a downloaded GGUF's metadata — the catalog
  uses the curated `ctx_train` for display; the load path already uses the exact
  value from the loaded model (sub-project 1).

## Data model

### `settings.rs`
```rust
// Added to the on-disk Settings struct, #[serde(default)]:
model_ctx: BTreeMap<String, u32>,   // key = "{repo}/{file}", value = ctx override

pub fn model_ctx_key(repo: &str, file: &str) -> String;      // "{repo}/{file}"
pub fn load_model_ctx(key: &str) -> Option<u32>;             // None = no override
pub fn save_model_ctx(key: &str, ctx: u32) -> anyhow::Result<()>;
pub fn clear_model_ctx(key: &str) -> anyhow::Result<()>;
```
`save_model_ctx`/`clear_model_ctx` do read-modify-write of the whole `Settings`
so `profile` and `integrations` are preserved (same pattern as `save_profile`).

### `catalog.rs`
`CatalogEntry` gains `pub ctx_train: u32` — the model's trained context window,
curated per entry (verify each against its model card). Known values:
Qwen2.5 = 32768, Qwen3 = 32768, Llama 3.2/3.1 = 131072, Gemma 2 = 8192,
Gemma 3 = as documented, Phi-3.5 = 131072, Phi-4 (14B) = 16384,
Phi-4-mini = 131072, Mistral = 32768. (Exact values pinned in the plan.)

## KV-aware `catalog_view`

New signature:
```rust
pub fn catalog_view(
    entries: &[CatalogEntry],
    total_ram_mb: u64,
    requested_ctx_ceiling: u32,          // cfg.ctx_len (default 32768)
    kv: crate::fit::KvKind,
    active: Option<&ModelSpec>,
    is_downloaded: impl Fn(&str, &str) -> bool,
    ctx_override: impl Fn(&str, &str) -> Option<u32>,
) -> Vec<FamilyView>
```

Per entry:
- `budget_mb = fit::device_budget_mb(total_ram_mb, true)`
- `kv_per_token = fit::est_kv_bytes_per_token(params_b, kv)` (heuristic; the
  catalog shows not-yet-downloaded models)
- `bounds = fit::ctx_bounds(size_mb, kv_per_token, budget_mb, ctx_train)`
- `ctx_current = ctx_override(repo,file).unwrap_or(min(requested_ctx_ceiling, bounds.max))`
  — the context this model would actually load at.
- `est_ram_mb = size_mb + kv_mb(ctx_current) + COMPUTE_HEADROOM_MB`, where
  `kv_mb(c) = kv_per_token * c / (1024*1024)`. (Replaces `size_mb * 1.2`.)
- `fit`: `Fits` if `est_ram_mb <= budget` (65%/78% as today via the existing
  budget/tight ceilings — see note), `Tight` if `<= tight_ceiling`, else
  `WontFit`. When `bounds.max == 0` → `WontFit`.
- `recommended`: unchanged rule (largest `params_b` among `Fits`, else smallest),
  now over the KV-aware verdict.

`ModelView` gains: `pub ctx_min: u32, pub ctx_default: u32, pub ctx_max: u32,
pub ctx_current: u32` (serialized snake_case like the rest).

**Budget note:** the existing `catalog_view` uses `budget = ram*65%` and
`tight = ram*85%` for the *verdict*. Keep those two thresholds for the
Fits/Tight/WontFit classification (they are the display comfort bands). Use
`fit::device_budget_mb` (78%) only inside `ctx_bounds` for the *max ctx* math.
This keeps the verdict conservative while the ctx ceiling tracks the real Metal
cap. (The two are intentionally different: one is "comfortable to run", the other
is "physically fits".)

## Endpoint + load-path

### `POST /admin/model/ctx` (token-guarded)
Body `{ "repo": "...", "file": "...", "ctx": <u32> }`. `ctx == 0` clears the
override (revert to default).
- Reject (`400`) if `repo` empty or `file` not a bare filename (reuse
  `is_safe_model_file`), or the model is not in `CATALOG`.
- Look up the entry, compute `bounds = ctx_bounds(size_mb, est_kv_bytes_per_token(params_b, kv), device_budget_mb(ram), ctx_train)`.
- If `ctx != 0` and `ctx` outside `[bounds.min, bounds.max]` → `400` with the
  allowed range in the message.
- Persist: `ctx == 0` → `clear_model_ctx(key)`; else `save_model_ctx(key, ctx)`.
- If the target is the **active** model → `manager.start_switch(same_spec)` to
  reload at the new ctx; return `202` (`{"reloading": true}`). Handle
  `AlreadySwitching` → `409`. Otherwise return `200` (`{"saved": true}`).

### Load-path override resolution
The requested ctx for a model becomes `override ?? cfg.ctx_len`:
- Initial load (`run_server_with_ready_policy_token`): before `LlamaEngine::load`,
  `let requested = settings::load_model_ctx(&model_ctx_key(&cfg.model_id, &cfg.gguf_files[0])).unwrap_or(cfg.ctx_len);` and pass `requested` instead of `cfg.ctx_len`.
- Switch builder closure: resolve `settings::load_model_ctx(&model_ctx_key(&spec.repo, &spec.file)).unwrap_or(b_ctx_len)` at build time so a reload picks up the saved override.
Sub-project 1's clamp still fits `requested` to the device.

### `AppState` additions
`pub requested_ctx_ceiling: u32` (= `cfg.ctx_len`) and `pub kv_kind: crate::fit::KvKind` (mapped from `cfg.kv_type`), set at `router(...)` construction, so `handle_models_catalog` and the new ctx endpoint can compute bounds. The catalog handler passes these + a `ctx_override` closure (`settings::load_model_ctx`) into `catalog_view`.

## Error handling
- Out-of-range or unknown-model ctx set → `400`, no persistence.
- `AlreadySwitching` on an active-model reload → `409` (client retries).
- A bad settings file remains non-fatal (existing load-or-default contract);
  `model_ctx` defaults to empty.
- Persistence is atomic (the settings save path; already routed through
  `atomic_write` after sub-project A's fix).

## Testing
Pure `catalog_view` unit tests (injected RAM/ctx-ceiling/override/downloaded):
- KV-aware est: a model's `est_ram_mb` grows with `ctx_current`; Phi-4 14B is
  `Fits` at its clamped 16384 (weights + KV(16384) < budget), while a same-size
  model forced to a large `ctx_current` is `Tight`/`WontFit`.
- `ctx_current` = override when set, else `min(ceiling, max)`.
- `ctx_min/default/max` populated from `ctx_bounds` (default = `min(8192, max)`).
- recommendation flips correctly when KV pushes the largest model over budget.
- `ctx_train` caps `ctx_max` (a 32768-ceiling model with `ctx_train` 8192 shows
  `ctx_max` 8192).
`settings.rs`: `model_ctx` round-trip; `save_model_ctx` preserves `profile` and
`integrations`; `clear_model_ctx` removes only that key; missing → `None`.
`tests/http.rs`: `POST /admin/model/ctx` → `400` for out-of-range, `400` for
unknown model, `200` for a valid non-active set, and (with the test manager whose
active model matches) `202`/`409` on the active-model reload path; `401` without
the token.

## Files touched
- `src/settings.rs` — `model_ctx` map + load/save/clear + key helper + tests.
- `src/catalog.rs` — `ctx_train` field; KV-aware `catalog_view`; new `ModelView`
  fields; tests.
- `src/server.rs` — `POST /admin/model/ctx` handler + route; `AppState`
  `requested_ctx_ceiling` + `kv_kind`; catalog handler passes the new args.
- `src/lib.rs` — resolve the per-model override for the initial load and the
  switch builder; set the new `AppState` fields at `router(...)`.
- `tests/http.rs` — endpoint tests.

## Follow-up (sub-project 3)
Model Manager page: per model show size, KV-aware fit, and `ctx_current` with a
bounded input (`min`..`max`, suggesting `ctx_default`); on save, `POST
/admin/model/ctx`; reflect the `202` reload (show "reloading…"). Consumes only
the fields and endpoint defined here.
