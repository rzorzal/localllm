# Per-model execution profile (sub-project 1)

**Date:** 2026-07-01
**Status:** Approved design, ready for planning
**Scope:** Sub-project 1 of 3. Load-time knobs + history window.

## Context

The model manager already lets users override **context length** per model
(`settings.model_ctx`, surfaced in the catalog drilldown, applied on load). The
user wants to expand this into a general **per-model execution profile** so each
model carries its own tuning, with a recommended default per model.

Motivation traces back to AirLLM (running oversized models by trading speed for
memory). On this project's platform — Apple unified memory via llama.cpp/Metal —
the literal AirLLM lever (layer-by-layer disk streaming) is a poor fit: it yields
15–30 min/response and llama.cpp's mmap already streams weights better. The
adjacent, genuinely useful levers are: KV-cache quantization, partial GPU offload
(`n_gpu_layers`), and — for controlling what reaches the model — history
truncation. Those become per-model, user-visible knobs.

This is **sub-project 1**. Two follow-ups are already scoped and out of scope
here:

- **Sub-2 — quant tier:** choose the GGUF quant variant (Q4/Q3/Q2/IQ) per model.
  Requires restructuring `CatalogEntry` to hold multiple variants + variant
  download. Deferred.
- **Sub-3 — tool filter:** discover the client's tool set on first request, let
  the user enable/disable tools via checkboxes, persist **per client**
  (`claude-code`, `codex` — reusing `ClientPrior`), gated behind a KV rebuild.
  Deferred.

## Goals

Expose four per-model knobs, each with a per-model recommended default and the
existing precedence (**saved profile → catalog recommendation → global CLI
default**):

1. `ctx` — context length (already exists; folded into the new profile).
2. `kv_type` — KV-cache quantization (F16/Q8/Q4). Today a global CLI flag only.
3. `gpu_layers` — number of layers on the GPU (`None` = all). Today hardcoded
   `u32::MAX`. Advisory-only in the fit verdict for this sub-project.
4. `history_turns` — how many recent conversation *turns* of history to send to
   the model (request-time truncation). `None` = keep all.

## Non-goals

- Modeling `gpu_layers` precisely in the fit verdict. On unified memory, CPU
  offload barely reduces *total* RAM; it only shifts weight out of the Metal
  working-set cap (78%). Sub-1 exposes it as a manual advanced knob with advisory
  text; the fit verdict stays conservative (all-GPU estimate). Refine later if
  warranted.
- Quant-variant selection (sub-2).
- Tool filtering (sub-3).

## Architecture

### 1. Data model — `settings.rs`

Replace the single-purpose `model_ctx` map with a per-model profile map.

```rust
/// Per-model execution profile. Every field is optional: `None` means "fall
/// back to the catalog recommendation, then the global CLI default".
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_type: Option<KvType>,       // imported from config.rs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,       // None = u32::MAX (all layers on GPU)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_turns: Option<u32>,     // None = keep all turns
}
```

`Settings` gains `model_profiles: BTreeMap<String, ExecProfile>` (keyed by
`"{repo}/{file}"`, the existing `model_ctx_key`), all `#[serde(default)]`.

**Migration.** On load, fold any legacy `model_ctx` entries into
`model_profiles[key].ctx` when the profile has no `ctx` yet, then drop the legacy
field from what we write back. Keep the `model_ctx` field in the struct (with
`#[serde(default)]`) only long enough to read old files; new writes use
`model_profiles`.

**API.** Add:
- `load_model_profile(key) -> ExecProfile`
- `save_model_profile(key, &ExecProfile)` (preserves the rest of settings)
- `clear_model_profile(key)`

Keep `load_model_ctx` / `save_model_ctx` / `clear_model_ctx` as thin wrappers over
`profile.ctx` so existing call sites (catalog_view, the ctx endpoint) don't break.

`KvType` currently lives in `config.rs` with a clap derive. `settings.rs` imports
it. No move needed; if the dependency direction is awkward, lift `KvType` into a
small shared module — decide during planning.

### 2. Engine wiring — `engine.rs` / `engine_llama.rs`

- `EngineConfig` gains `gpu_layers: Option<u32>`. The two hardcoded
  `LlamaModelParams::default().with_n_gpu_layers(u32::MAX)` sites become
  `.with_n_gpu_layers(gpu_layers.unwrap_or(u32::MAX))`.
- **Per-model `kv_type` on reload.** Today `kv_cache_type` is captured once at
  worker spawn from the global `Config`. Model switch / reload must resolve the
  target model's `ExecProfile` and thread the resolved `ctx` + `kv_type` +
  `gpu_layers` into the worker respawn. This is the most delicate change — the
  worker thread captures these at spawn, so the reload path (in the server's
  model-switch handler + `engine` reload) must pass the resolved values, not the
  startup ones.

### 3. Resolution (precedence)

A single helper resolves the effective profile at load time. For each field:
**saved profile value → catalog recommended value → global CLI default**. Mirrors
today's ctx behavior (`ctx_override → requested_ctx_ceiling.min(bounds.max)`).

### 4. Catalog recommendations — `catalog.rs`

`CatalogEntry` gains optional recommendation fields, set only where they differ
from the global default (most stay `None`):

```rust
pub rec_kv: Option<KvType>,
pub rec_gpu_layers: Option<u32>,
pub rec_history_turns: Option<u32>,
```

`ctx` recommendation already exists via `ctx_bounds().default`.

`ModelView` gains: `kv_current`, `kv_default`, `gpu_layers_current`,
`history_turns_current`, `history_turns_default`. `catalog_view` resolves `kv`
**per entry** (from the saved profile, falling back to `rec_kv`, then the global
`kv`) instead of taking one global `KvKind` for all rows — this changes the KV
figure used in each row's RAM estimate.

### 5. Fit math — `fit.rs`

- The per-token KV cost already varies with `KvKind`; feeding a per-model `kv`
  into the estimate is the only change on the fit side.
- `gpu_layers` does **not** alter the fit verdict in sub-1 (see Non-goals). The
  UI shows advisory text ("reduz pressão Metal, mais lento") next to the knob.

### 6. Request shaping — `api/common.rs`

Both API adapters (OpenAI, Anthropic) normalize to the internal
`{ messages, tools }` request. History truncation is applied there, after
normalization, before the engine call:

- Unit = **conversation turns**, not raw messages. A turn begins at a `user`
  message and includes the assistant/tool messages up to the next `user`.
- Keep the last `history_turns` turns. **Always** keep the leading `system`
  message. Never split a `tool_use` from its matching `tool_result`, and respect
  the first-message role rules of each format (don't emit a truncated array that
  starts mid-tool-chain).
- `None` (or unset) = no truncation.
- This is request-time only — no KV rebuild. It does interact with prefix reuse
  (a sliding window shifts the mid-prefix each turn, reducing reuse after the cut
  point); documented, accepted.

### 7. Endpoints — `server.rs`

- New `POST /admin/model/profile` with optional fields
  `{ repo, file, ctx?, kv_type?, gpu_layers?, history_turns? }`. Sets the provided
  subset; validates `ctx` against `ctx_bounds` exactly as the current ctx endpoint
  does; clears a field when its documented sentinel is sent (e.g. `ctx = 0`).
  Reloads the model if it's the active one, reusing the existing reload+progress
  flow.
- Keep `POST /admin/model/ctx` as a back-compat wrapper delegating to the profile
  path.
- `GET /admin/models` (catalog) now surfaces the new `ModelView` fields.

### 8. UI — `manager_ui` (drilldown panel)

Reuse the existing `ctxbox` pattern in the per-model drilldown:

- **KV cache**: a select (F16 / Q8 / Q4) with the recommended option marked.
- **History**: a numeric input for `history_turns` (turns), with the recommended
  default shown.
- **Avançado** (collapsed): `gpu_layers` numeric input + advisory text.
- A single `saveProfile()` posts the changed subset to `/admin/model/profile`,
  reusing the reload/progress UX when the active model changes.

Consistent with the Firestore-style drilldown of the model-picker window.

## Data flow

```
request → API adapter → common{messages,tools}
        → history truncation (history_turns)      [request-time]
        → engine (ctx, kv_type, gpu_layers)      [load-time, from resolved profile]

catalog_view → per-entry kv → RAM estimate + fit verdict
UI drilldown → POST /admin/model/profile → settings.model_profiles → reload
```

## Error handling

- Bad/absent settings file → defaults (existing behavior; never blocks startup).
- Invalid `ctx` (outside bounds) → 4xx from the endpoint, same as today.
- `gpu_layers` beyond the model's layer count → llama.cpp clamps at load; the
  input is a raw integer, not bounded by the catalog (layer count is unknown
  pre-download).
- History truncation that would break a tool-use/tool-result pair → the truncator
  extends the window to keep the pair intact rather than emitting a malformed
  array.

## Testing

- **settings:** `model_ctx` → `model_profiles` migration round-trip; profile
  save/load/clear preserves the rest of settings; back-compat wrappers.
- **fit/catalog:** `catalog_view` with per-entry `kv`; recommended-field
  resolution and precedence.
- **config/engine:** `gpu_layers` maps to `with_n_gpu_layers`; `None` → `u32::MAX`.
- **request shaping:** turn-based truncation keeps `system`, keeps the last N
  turns, never splits a tool pair; `None` = passthrough.
- **endpoint:** `/admin/model/profile` sets a subset, validates ctx bounds, clears
  via sentinel; ctx wrapper still works.

## Open questions for planning

- Exact home of `KvType` (leave in `config.rs` vs. lift to a shared module) once
  the `settings → config` import direction is examined.
- Whether the reload path already threads enough state to pass a resolved profile,
  or needs a small refactor to carry `{ctx, kv_type, gpu_layers}` together.
