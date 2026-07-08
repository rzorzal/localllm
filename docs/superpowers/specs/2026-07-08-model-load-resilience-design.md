# Model-load resilience: bind-first + background load with retry + notify

**Date:** 2026-07-08
**Status:** approved (design)
**Related:** [[tray-dies-on-model-load-failure]] "L2". Resilience theme, alongside routing sub-2.

## Problem

At boot, `run_server_with_ready_policy_token` (`src/lib.rs:191`) does `LlamaEngine::load(...).await?` BEFORE binding the HTTP listener (`src/lib.rs:290`). A load failure (bad `active_model`, HuggingFace error, OOM) therefore returns early — the server never binds, and the manager webview loads a blank page (all Config/Dashboard pages white). The user has no in-app way to recover. A poisoned `active_model={"repo":"r2","file":"f2"}` reproduced exactly this: "Server failed — ensure_model failed" + blank UI.

## Goals

- The HTTP server binds and serves the management UI even when no model is loaded, so the user can recover in-app (switch to a working model).
- The initial model load runs in the background with a few retries; on final failure, notify (native desktop notification) and keep serving.
- While no local engine is loaded, requests route to cloud if credentials are present (never-error), else return a clean 503.

## Non-goals

- Changing the hot-swap/switch path itself (reused as-is).
- Configurable retry counts/timeouts in v1 (sensible hardcoded defaults).
- Recovering a mid-generation engine panic (already handled by `recover_context`).

## Architecture

Bind before load. Build the `ModelManager` in an empty "loading" state (its `engine` is already `ArcSwapOption`, so `None` is representable), bind + serve immediately, then spawn the initial load using the existing `builder` (download + `LlamaEngine::load` + status/progress). Requests that need the local engine check whether one is loaded and fall back to cloud-or-503 while it is absent.

## Components

### `ModelManager` (`src/model_manager.rs`)
- `new_loading(current: ModelSpec, builder: EngineBuilder) -> Arc<Self>`: like `new` but `engine: None`, `switching: true`, `phase: loading`, `progress: 0`. The manager reports "loading" via its existing `status()`.
- `has_engine(&self) -> bool`: `self.engine.load_full().is_some()`.
- Reuse `start_switch(spec)` for the initial load — it already downloads, builds via `builder`, swaps the engine in, and updates `switching`/`phase`/`progress`/`errored`/`error`. The boot triggers `start_switch(initial_spec)`.

### Boot (`src/lib.rs`)
- Remove the pre-bind `LlamaEngine::load(...).await?` (lines ~190–203); build the manager with `new_loading(initial_spec, builder)`.
- Bind + `axum::serve` as today (the router/app is built from the manager).
- After bind (and after flipping `ready`), spawn the initial load with retry:
  ```
  spawn:
    for attempt in 1..=MAX_ATTEMPTS (3):
      manager.start_switch(initial_spec) and await its completion
      if success -> break
      else -> sleep backoff[attempt] (5s, 15s, 30s)
    if still failed:
      usage::notify("localllm", "modelo não carregou: <reason> — abra Config → Models para escolher outro")
      (manager stays errored; server keeps serving)
  ```
  `start_switch` is fire-and-forget (spawns a worker, returns immediately), so "await its completion" = poll `manager.is_switching()` every 500 ms until it is false, then read `manager.is_errored()` / `manager.has_engine()` to decide success vs retry.

### Request paths (`src/server.rs`)
- Before dispatching to the local engine (in `route_decision`/the handlers), if `!manager.has_engine()`:
  - if the request carries cloud creds AND the profile allows cloud → force `Decision::Cloud` with a reason `LocalUnavailable` (reuse the reverse-proxy path).
  - else → return `503` with `{"error":"modelo carregando, tente em instantes"}`.
- When `has_engine()` is true, behavior is unchanged.

### Notification
- Reuse `crate::usage::notify(title, body)` (native, already used for cloud-degraded).

### Tray
- No change required: the tray already polls `manager.status()` and renders loading/errored labels (L1). Confirm the errored reason surfaces once the retries are exhausted.

## Data flow

boot → build empty manager → bind + serve (UI live) → spawn initial load (retry×3) →
  success: engine swapped in, requests serve locally as normal.
  failure: notify + errored; requests route cloud-or-503; user opens Config → Models → switch → fresh load.

## Error handling

- Initial load failure never blocks bind (it runs in a spawned task).
- No-engine requests: cloud when possible, else clean 503 — never a hang or crash.
- `usage::notify` is best-effort.
- Existing `check_admin`, hot-swap, and `recover_context` paths unchanged.

## Testing

- `ModelManager::new_loading` → `has_engine()` false, `status()` reports loading; after a (mock) successful build the engine is present and `has_engine()` true. (Use the existing test-double `Generator` + builder pattern in `model_manager.rs` tests.)
- Routing/handler decision: no local engine + cloud creds → cloud; no engine + no creds → 503. Unit-test the pure branch that chooses cloud-vs-503 given `(has_engine, cloud_available)`.
- Retry: a pure `backoff_for(attempt)` / attempt-count helper tested for the 3-attempt schedule.
- Real model load = integration (needs a model) → not unit-tested; manual verification with a bad `active_model` (server binds, UI loads, notify fires, switch recovers).

## Rollout

Backend + embedded frontend unaffected (UI already tolerant). Rebuild the `.app` bundle + restart. Manual test: set `active_model` to a bogus repo/file, launch → confirm the tray/UI come up (not blank), a notification fires after retries, and Config → Models can switch to a real model to recover.
