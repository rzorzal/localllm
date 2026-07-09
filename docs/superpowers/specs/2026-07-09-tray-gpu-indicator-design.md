# Tray GPU indicator

**Date:** 2026-07-09
**Status:** approved (design)
**Component:** `src/tray.rs` (new info line + pure helper), reads `src/settings.rs` + `src/profile.rs` + `src/catalog.rs`.

## Problem

The user could not tell from the app whether the local model is running on the
GPU (Metal) or CPU, and suspected weak/slow generation meant it was not using
the GPU. There is no indicator. (In fact the default is all-layers-on-Metal; the
observed slowness was cold-prefill, since fixed.)

## Goal

Show, in the menu-bar tray, an at-a-glance indicator of whether the active local
model is offloaded to the GPU, and how many layers, updating when the model is
hot-swapped.

## Non-goals

- No runtime-actual offload measurement (not parsing llama.cpp load logs / model
  metadata for the real offloaded count). The indicator is config-derived — the
  resolved `gpu_layers` the engine was built with. (Chosen for cost; honest on
  macOS, see Truth level.)
- No dashboard/Config surface — tray only (that is what the user asked for).
- No change to how `gpu_layers` is resolved or to the engine.

## Truth level (config-derived)

The label reflects `profile::resolve(saved, catalog, …).gpu_layers`
(= `saved.gpu_layers.or(catalog.rec_gpu_layers)`), which is exactly the value the
engine is loaded with (`lib.rs` resolves the same way before building the
engine). On macOS the llama-cpp-sys build always compiles the Metal backend, and
a Metal allocation failure errors at load (handled by model-load resilience)
rather than silently falling back to CPU — so on this platform the config-derived
label honestly reflects reality. This app is macOS-only (menu-bar tray); the
platform assumption is acceptable and noted here.

## Components

### Pure helper `gpu_menu_label`

`fn gpu_menu_label(gpu_layers: Option<u32>) -> String` in `src/tray.rs`:

| Input | Output |
|-------|--------|
| `None` | `🟢 GPU: all layers` |
| `Some(0)` | `⚪ CPU only` |
| `Some(n)` where `n > 0` | `🟢 GPU: {n} layers` |

`None` maps to "all layers" because a `None` resolved value makes the engine use
`with_n_gpu_layers(u32::MAX)` (full offload). The emoji prefix gives the
at-a-glance GPU/CPU read; the `Label: value` shape matches the existing info
block (`Model:`, `Context:`, `KV cache:`, `Backend:`).

### Tray line + live update

- In the menu build (`StartCause::Init`), add a `gpu_line` `MenuItem` immediately
  after `backend_line`, disabled/informational, initial text
  `gpu_menu_label(<active model's resolved gpu_layers at startup>)`. Keep its
  handle: `gpu_handle: Option<MenuItem>` (mirrors `model_handle`).
- In the status poll, inside the existing `if last_model.as_ref() != Some(&st.current)`
  branch (the model hot-swap trigger), also recompute the GPU label for
  `st.current` and `gpu_handle.set_text(...)`, guarded by a `last_gpu: Option<String>`
  dedupe (mirrors `last_status`/`last_model`). Updating in the same branch means
  the GPU line refreshes exactly when the active model changes.

### Data path (resolving the active model's `gpu_layers`)

Given `st.current` (a `ModelSpec` with `.repo`, `.file`):

```
key      = settings::model_ctx_key(&repo, &file)
saved    = settings::load_model_profile(&key)
catalog  = catalog::CATALOG.iter().find(|e| e.repo == repo && e.file == file)
resolved = profile::resolve(&saved, catalog, /*global_ctx*/ 0, /*global_kv*/ <any>)
gpu      = resolved.gpu_layers
label    = gpu_menu_label(gpu)
```

`resolved.gpu_layers` does not depend on `global_ctx` / `global_kv` (only `ctx`
and `kv_type` do), so inert placeholder values are passed for those. This mirrors
`lib.rs`'s existing profile resolution. A small private helper
`active_gpu_layers(repo: &str, file: &str) -> Option<u32>` in `tray.rs` wraps the
four lines above so both the startup build and the poll call one function.

## Error handling

- No new fallible paths. `load_model_profile` already returns a default profile
  for unknown keys; a missing catalog entry yields `None` from `resolve` (→
  "all layers"), which is the correct default.
- No panics, no I/O beyond the existing settings read the poll already performs
  for other lines (`load_integrations`, `load_profile`).

## Testing

- Unit tests for `gpu_menu_label`: `None → "🟢 GPU: all layers"`,
  `Some(0) → "⚪ CPU only"`, `Some(24) → "🟢 GPU: 24 layers"`.
- Manual: launch → tray shows `🟢 GPU: all layers`. Set a model's profile
  `gpu_layers = 0` (Config → model exec profile), hot-swap to it → the line flips
  to `⚪ CPU only` without an app restart. Set `gpu_layers = 20` → `🟢 GPU: 20 layers`.

## Rollout

Backend/tray only; rebuild the `.app` bundle + restart. Manual verification as
above.
