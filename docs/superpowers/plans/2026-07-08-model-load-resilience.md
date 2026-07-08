# Model-Load Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind and serve the management UI even when the model fails to load; run the initial load in the background with retries; on final failure notify and keep serving; requests with no local engine route to cloud when possible, else a clean 503.

**Architecture:** `ModelManager` gains an empty "loading" constructor and an awaitable `try_initial_load` (its `engine` is already `ArcSwapOption`). `lib.rs` builds the manager empty, binds + serves, then spawns a retry loop that loads the model. Server request paths override the routing decision to `Cloud(LocalUnavailable)` (creds present) or `NoModel` → 503 (no creds) while no engine is loaded.

**Tech Stack:** Rust (tokio, arc_swap, axum), `crate::usage::notify` for native notifications.

## Global Constraints

- The HTTP listener must bind regardless of model-load success (load moves to a spawned background task).
- No local engine + cloud creds + cloud allowed → route to cloud (`Cloud(LocalUnavailable)`). No local engine + no cloud → `503 {"error":"modelo carregando, tente em instantes"}`.
- Initial load retries: 3 attempts, backoff 5 s / 15 s / 30 s. On final failure: `crate::usage::notify("localllm — modelo não carregou", <reason>)` and leave the manager errored; keep serving.
- Reuse existing machinery: `builder`, `status()`, `is_switching()`, `is_errored()`, the hot-swap `start_switch` (for user-driven recovery). Do NOT change `start_switch`/`run_switch`.
- `decide` in `route/mod.rs` stays pure; the no-engine override lives in `server.rs` (runtime state).

---

### Task 1: `ModelManager` — empty-loading state + awaitable initial load

**Files:**
- Modify: `src/model_manager.rs` (`new_loading`, `has_engine`, `try_initial_load`, tests)

**Interfaces:**
- Produces: `ModelManager::new_loading(current: ModelSpec, builder: EngineBuilder) -> Arc<Self>`, `ModelManager::has_engine(&self) -> bool`, `async ModelManager::try_initial_load(self: &Arc<Self>, spec: ModelSpec) -> bool` (true = engine loaded).

- [ ] **Step 1: Write failing tests**

In the `model_manager.rs` tests module (reuse helpers `marker_gen`, `spec`, `counting_builder`, `req`, `ContentPart`):

```rust
#[tokio::test]
async fn new_loading_has_no_engine_until_initial_load() {
    let calls = Arc::new(TestAtomicUsize::new(0));
    let m = ModelManager::new_loading(spec("r", "m"), counting_builder(calls.clone(), None));
    assert!(!m.has_engine());
    assert_eq!(m.status().state, "switching"); // loading
    let ok = m.try_initial_load(spec("r", "m")).await;
    assert!(ok);
    assert!(m.has_engine());
    assert_eq!(m.status().state, "ready");
    let out = m.generate(req()).await.unwrap();
    assert!(matches!(&out.content[0], ContentPart::Text(_)));
}

#[tokio::test]
async fn initial_load_failure_leaves_no_engine_and_errored() {
    let builder: EngineBuilder = Box::new(|_spec| Box::pin(async { Err(anyhow::anyhow!("boom")) }));
    let m = ModelManager::new_loading(spec("bad", "x"), builder);
    let ok = m.try_initial_load(spec("bad", "x")).await;
    assert!(!ok);
    assert!(!m.has_engine());
    assert!(m.is_errored());
    assert!(m.status().error.unwrap().contains("boom"));
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test -p localllm model_manager::tests::new_loading model_manager::tests::initial_load`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement the three methods**

Add to `impl ModelManager` (near `new`):

```rust
    /// Build a manager with NO engine yet, in a "loading" state. The caller runs
    /// `try_initial_load` (typically after binding the server) to populate it.
    pub fn new_loading(current: ModelSpec, builder: EngineBuilder) -> Arc<Self> {
        Arc::new(Self {
            engine: ArcSwapOption::from(None),
            current: Mutex::new(current),
            target: Mutex::new(None),
            switching: AtomicBool::new(true),
            errored: AtomicBool::new(false),
            inflight: Arc::new(AtomicUsize::new(0)),
            phase: AtomicU8::new(PHASE_LOADING),
            progress: AtomicU8::new(0),
            error: Mutex::new(None),
            builder,
        })
    }

    /// True when a local engine is loaded and ready to serve.
    pub fn has_engine(&self) -> bool {
        self.engine.load_full().is_some()
    }

    /// Load the initial engine via the builder (no drain, no restore — there is
    /// no previous engine). Returns true on success. On failure, sets errored +
    /// error and leaves the engine absent. Clears `switching` either way.
    pub async fn try_initial_load(self: &Arc<Self>, spec: ModelSpec) -> bool {
        self.switching.store(true, Ordering::SeqCst);
        self.errored.store(false, Ordering::SeqCst);
        *self.error.lock().unwrap() = None;
        self.set_phase(PHASE_LOADING);
        *self.target.lock().unwrap() = Some(spec.clone());
        let result = (self.builder)(spec.clone()).await;
        match result {
            Ok(engine) => {
                self.engine.store(Some(Arc::new(engine)));
                *self.current.lock().unwrap() = spec;
                *self.target.lock().unwrap() = None;
                self.progress.store(100, Ordering::SeqCst);
                self.set_phase(PHASE_IDLE);
                self.switching.store(false, Ordering::SeqCst);
                true
            }
            Err(e) => {
                *self.error.lock().unwrap() = Some(format!("model load failed: {e}"));
                self.errored.store(true, Ordering::SeqCst);
                self.set_phase(PHASE_IDLE);
                *self.target.lock().unwrap() = None;
                self.switching.store(false, Ordering::SeqCst);
                false
            }
        }
    }
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cargo test -p localllm model_manager`
Expected: the two new tests pass; existing manager tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/model_manager.rs
git commit -m "feat(manager): new_loading + has_engine + try_initial_load (empty-engine state)"
```

---

### Task 2: No-engine routing override → cloud or 503

**Files:**
- Modify: `src/route/mod.rs` (`RouteReason::LocalUnavailable`, `Decision::NoModel`)
- Modify: `src/server.rs` (`no_engine_decision` helper + wire into the 3 handlers)
- Test: inline in `src/server.rs`

**Interfaces:**
- Consumes: `ModelManager::has_engine` (Task 1)
- Produces: `RouteReason::LocalUnavailable`, `Decision::NoModel`, `server::no_engine_decision(has_engine: bool, cloud_available: bool) -> Option<crate::route::Decision>`

- [ ] **Step 1: Add the enum variants**

In `src/route/mod.rs`, add to `RouteReason` (after `ColdPrefill`):

```rust
    /// No local engine loaded (initial load in progress / failed) — served from
    /// cloud instead.
    LocalUnavailable,
```

Add to `Decision` (after `LocalNoCreds`):

```rust
    /// No local engine AND cloud not possible → the caller returns 503.
    NoModel,
```

- [ ] **Step 2: Write the failing helper test**

In `src/server.rs` tests module:

```rust
#[test]
fn no_engine_decision_prefers_cloud_then_503() {
    use crate::route::{Decision, RouteReason};
    // engine present → no override
    assert_eq!(super::no_engine_decision(true, true), None);
    assert_eq!(super::no_engine_decision(true, false), None);
    // no engine + cloud available → cloud
    assert_eq!(super::no_engine_decision(false, true), Some(Decision::Cloud(RouteReason::LocalUnavailable)));
    // no engine + no cloud → NoModel (→ 503)
    assert_eq!(super::no_engine_decision(false, false), Some(Decision::NoModel));
}
```

- [ ] **Step 3: Run test, verify it fails**

Run: `cargo test -p localllm no_engine_decision`
Expected: FAIL — not defined / variants missing.

- [ ] **Step 4: Implement the helper**

In `src/server.rs` (near `route_decision`):

```rust
/// Override the routing decision when no local engine is loaded: prefer cloud if
/// available, else signal a 503 via `NoModel`. Returns None when an engine is
/// present (normal routing applies).
pub(crate) fn no_engine_decision(has_engine: bool, cloud_available: bool) -> Option<crate::route::Decision> {
    if has_engine {
        return None;
    }
    Some(if cloud_available {
        crate::route::Decision::Cloud(crate::route::RouteReason::LocalUnavailable)
    } else {
        crate::route::Decision::NoModel
    })
}
```

- [ ] **Step 5: Wire into each handler's guard**

Each handler has a guard block near its top: `if state.manager.is_switching() { return 503 "model switching" }` (openai ~2081, anthropic ~2331, responses ~2483 — locate the three by that string). Immediately AFTER that guard in each handler, add:

```rust
    // No local engine (initial load in progress / failed): cloud if possible, else 503.
    if !state.manager.has_engine() {
        let cloud_available = state.policy.read().unwrap_or_else(|e| e.into_inner()).allow_cloud
            && crate::cloud::has_forwardable_creds(&headers);
        match no_engine_decision(false, cloud_available) {
            Some(crate::route::Decision::NoModel) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"error": "modelo carregando, tente em instantes"})),
                )
                    .into_response();
            }
            Some(dec) => { /* fall through to serve via cloud below */ let _forced = dec; }
            None => {}
        }
    }
```

Then, where each handler computes its `decision` (from `route_decision`), OR right before the `match decision` that serves, force the cloud path when no engine + cloud available. The minimal, correct wiring: replace the handler's `let (decision, prompt_tokens) = route_decision(...)` result so that when `!has_engine && cloud_available`, `decision` becomes `Cloud(LocalUnavailable)`. Concretely, right after `route_decision` returns, add:

```rust
    let (decision, prompt_tokens) = {
        let (d, pt) = route_decision(/* existing args */).await;
        match no_engine_decision(state.manager.has_engine(), /* cloud_available computed above */ cloud_available) {
            Some(forced) => (forced, pt),
            None => (d, pt),
        }
    };
```

If a handler doesn't already bind `cloud_available`, compute it once (as in Step 5's guard) and reuse. Verify `crate::cloud::has_forwardable_creds(&headers) -> bool` exists; if the real credential-check helper has a different name, use it (grep `has_forwardable_creds`/`forwardable`/`api-key` in `src/cloud.rs`/`src/server.rs`) and note it. The `Cloud(LocalUnavailable)` decision flows through the existing `Decision::Cloud(reason) => { … reverse-proxy … }` arm unchanged.

- [ ] **Step 6: Add the `NoModel` arm where handlers match `Decision`**

In each handler's `match decision { … }` (the one that dispatches Cloud/Local/LocalNoCreds/LocalThenCascade), add:

```rust
        crate::route::Decision::NoModel => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "modelo carregando, tente em instantes"})),
            )
                .into_response();
        }
```

(This makes the `match` exhaustive after adding the variant. If the guard in Step 5 already returned 503 for the no-cloud case, this arm is the belt-and-suspenders exhaustiveness handler — keep it.)

- [ ] **Step 7: Build + test**

Run: `cargo test -p localllm no_engine_decision && cargo build -p localllm`
Expected: helper test passes; clean build (all `match decision` sites exhaustive with `NoModel`). Fix any non-exhaustive-match errors by adding the `NoModel` arm there too.

- [ ] **Step 8: Commit**

```bash
git add src/route/mod.rs src/server.rs
git commit -m "feat(server): no-engine requests route to cloud or clean 503"
```

---

### Task 3: Boot — bind before load + background retry loop

**Files:**
- Modify: `src/lib.rs` (boot flow), add `backoff_secs` helper + test

**Interfaces:**
- Consumes: `ModelManager::new_loading`, `try_initial_load`, `has_engine` (Task 1)
- Produces: `crate::backoff_secs(attempt: u32) -> u64` (or module-local), the bind-before-load boot flow.

- [ ] **Step 1: Write the backoff test**

Add to `src/lib.rs` (a small tests module if none, else inline):

```rust
#[cfg(test)]
mod resilience_tests {
    #[test]
    fn backoff_schedule_is_5_15_30() {
        assert_eq!(super::backoff_secs(1), 5);
        assert_eq!(super::backoff_secs(2), 15);
        assert_eq!(super::backoff_secs(3), 30);
        assert_eq!(super::backoff_secs(9), 30); // caps
    }
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cargo test -p localllm backoff_schedule`
Expected: FAIL — not defined.

- [ ] **Step 3: Add `backoff_secs`**

In `src/lib.rs`:

```rust
/// Retry backoff for the initial model load: 5 s, 15 s, then 30 s (capped).
pub fn backoff_secs(attempt: u32) -> u64 {
    match attempt {
        1 => 5,
        2 => 15,
        _ => 30,
    }
}
```

- [ ] **Step 4: Replace the pre-bind load with an empty manager**

In `run_server_with_ready_policy_token` (`src/lib.rs`), the block at ~lines 190–203 does `let llama = LlamaEngine::load(...).await?; ... (Arc::new(llama) as Arc<dyn Generator>, eff)`. Replace the eager load so the manager starts EMPTY. Concretely: keep everything that computes `initial_spec`, `builder`, etc., but build the manager with `new_loading` instead of `new`, and do not `.await?` a `LlamaEngine::load` before bind.

- Delete the eager `LlamaEngine::load(...).await?` and the `(Arc::new(llama)..., eff)` tuple.
- `eff` (effective ctx window) was read from the loaded engine for the router build. Source it without a live engine: use the resolved `r.ctx` (the `resolve_load_params` result already computed at ~line 180) as the effective window for the router config. Read how `eff` is consumed downstream; pass `r.ctx as usize` (or the config `ctx_len`) in its place.
- Build the manager: `let manager = ModelManager::new_loading(initial_spec.clone(), builder);` (keep `manager_slot` wiring).

- [ ] **Step 5: Spawn the retry loop after bind**

After the listener binds and `ready` is flipped (~lines 290–292), before/around `axum::serve`, spawn the initial load:

```rust
    {
        let mgr = manager.clone();
        let spec = initial_spec.clone();
        tokio::spawn(async move {
            const MAX_ATTEMPTS: u32 = 3;
            for attempt in 1..=MAX_ATTEMPTS {
                if mgr.try_initial_load(spec.clone()).await {
                    tracing::info!(target: "localllm", "initial model loaded (attempt {attempt})");
                    return;
                }
                if attempt < MAX_ATTEMPTS {
                    let secs = backoff_secs(attempt);
                    tracing::warn!(target: "localllm", "model load attempt {attempt} failed; retry in {secs}s");
                    tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                }
            }
            let reason = mgr.status().error.unwrap_or_else(|| "unknown".into());
            tracing::error!(target: "localllm", "initial model load failed after {MAX_ATTEMPTS} attempts: {reason}");
            crate::usage::notify(
                "localllm — modelo não carregou",
                &format!("{reason} — abra Config → Models para escolher outro"),
            );
        });
    }
```

Ensure `manager` and `initial_spec` are still in scope at the spawn point and are cloneable (`ModelSpec` derives Clone; `manager` is an `Arc`). Keep `axum::serve(listener, app).await?` as the last statement so the server runs.

- [ ] **Step 6: Build + tests**

Run: `cargo test -p localllm backoff_schedule && cargo build -p localllm && cargo test -p localllm`
Expected: backoff test passes; clean build; full suite green.

- [ ] **Step 7: Commit**

```bash
git add src/lib.rs
git commit -m "feat(boot): bind before load; background initial-load retry + notify"
```

---

### Task 4: Build + verify

**Files:** none (build + manual verify)

- [ ] **Step 1: Full build + tests**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean, all pass.

- [ ] **Step 2: Rebuild bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS`.

- [ ] **Step 3: Manual — bad-model recovery**

- Temporarily set `active_model` to a bogus repo/file (e.g. `python3 -c` edit of `~/Library/Application Support/localllm/settings.json`, backing it up first).
- Launch the app. Expected: the tray comes up and Config/Dashboard pages LOAD (not blank); after retries a desktop notification "modelo não carregou …" fires; while no engine, a client request either goes to cloud (if creds) or gets a 503.
- Open Config → Models and switch to a real model → it loads and serves. Restore the real `active_model`.

- [ ] **Step 4: Commit any tweaks**

```bash
git add -A && git commit -m "chore: model-load resilience verification tweaks"
```

---

## Self-Review

**Spec coverage:**
- Bind before load; UI live on load failure → Task 3 (new_loading + bind + spawn). ✓
- Background initial load with retry (3×, 5/15/30s) + notify on final failure → Task 3 (`backoff_secs`, spawn loop, `usage::notify`). ✓
- No local engine → cloud if creds else 503 → Task 2 (`no_engine_decision`, handler wiring, `NoModel` 503). ✓
- Reuse hot-swap for recovery; don't change start_switch → Tasks reuse `start_switch`; `try_initial_load` is separate. ✓
- Empty-engine state representable → Task 1 (`new_loading`, `has_engine`). ✓
- Tray already reflects status → no code; verified in Task 4. ✓

**Placeholder scan:** Task 2 Step 5 names the credential-check helper to grep (`has_forwardable_creds`) with a fallback instruction to use the real name — a concrete verify step, not a placeholder. Task 3 Step 4 points at the exact lines to replace and how to source `eff`. No TBD/handle-edge-cases.

**Type consistency:** `new_loading`/`has_engine`/`try_initial_load` (Task 1) consumed in Tasks 2–3. `RouteReason::LocalUnavailable` + `Decision::NoModel` (Task 2) used in handler arms + helper. `backoff_secs(u32)->u64` (Task 3). `no_engine_decision(bool,bool)->Option<Decision>` consistent between helper + test.
