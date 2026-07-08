# Cold-Prefill-Aware Routing Implementation Plan (sub-1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route a request to cloud (and warm the local KV in the background) when the local model would need an expensive COLD prefill, so the client never waits minutes on a cold first turn; keep warm turns local; log warm/cold to the dashboard.

**Architecture:** The local engine tracks its KV-resident prefix (`cached_tokens`). It exposes (a) a cheap `estimate_cold_tokens(req)` query (tokenize + longest-common-prefix vs `cached_tokens`, no generation) and (b) a `prefill(req)` op (decode the prompt into the KV, no sampling). The pure `decide` function gains inputs `cold_tokens` + `prefill_tok_s`; when `cold_tokens / prefill_tok_s` exceeds a configurable gate and cloud is available, it returns `Cloud(ColdPrefill)`. The server measures those inputs, on `ColdPrefill` fires a background `prefill`, and logs warm/cold fields to the route-log for the dashboard.

**Tech Stack:** Rust (axum, tokio, std mpsc worker thread, llama-cpp-2), vanilla JS/CSS dashboard.

## Global Constraints

- `decide` in `src/route/mod.rs` stays a PURE function of `Signals` + `RoutingPolicy` — no I/O — so it is unit-testable. All measurement happens in the server and is passed in via `Signals`.
- The existing difficulty score, `capability_adjustment`, `ctx_gate_frac` overflow gate, and cascade behavior are UNCHANGED. This adds one new independent decision branch ordered AFTER the context gate and cloud-possible check, BEFORE the difficulty branch.
- Background prefill and the estimate query are best-effort: a failure NEVER affects the served response and never fails a request.
- The local engine context is single-threaded (one worker thread, FIFO `std_mpsc` channel). New jobs (`EstimateCold`, `Prefill`) queue on that channel; no preemption.
- New `RouteEntry` fields are optional with `#[serde(default, skip_serializing_if = "Option::is_none")]` (legacy log lines parse).
- Default `cold_prefill_gate_secs = 360.0` (patient with local; only monstrous cold prefills escalate). Config UI must note the client request timeout (sub-3) should exceed this.
- Route-log path `~/Library/Application Support/localllm/routing-log.jsonl`; never conflate with app-log `/tmp/localllm.log`.

---

### Task 1: Routing core — `ColdPrefill` branch (pure, TDD)

**Files:**
- Modify: `src/route/mod.rs` (`Signals`, `RouteReason`, `decide`, tests)
- Modify: `src/route/policy.rs` (`RoutingPolicy.cold_prefill_gate_secs` + 4 profile literals)

**Interfaces:**
- Produces: `Signals { …, cold_tokens: usize, prefill_tok_s: f64 }`
- Produces: `RouteReason::ColdPrefill`
- Produces: `RoutingPolicy.cold_prefill_gate_secs: f64`
- Produces: `decide` returns `Decision::Cloud(RouteReason::ColdPrefill)` for cold+big+cloud-available

- [ ] **Step 1: Add the policy field**

In `src/route/policy.rs`, add to `RoutingPolicy` (after `allow_cloud`):

```rust
    /// Escalate to cloud + background-warm local when the estimated COLD
    /// prefill would exceed this many seconds. Large = patient with local.
    pub cold_prefill_gate_secs: f64,
```

Add `cold_prefill_gate_secs: 360.0,` to ALL FOUR profile literals (SaveTokens, Balanced, MaxLocal/whatever the third is, MaxQuality) in the `ALL`/profile constructor near lines 55-78. Use `360.0` for every profile (the gate is latency-based, not quality-based).

- [ ] **Step 2: Write the failing routing tests**

In `src/route/mod.rs` tests module, add (adjust the `sig` helper or build `Signals` literally with the two new fields):

```rust
#[test]
fn cold_big_prefill_with_cloud_routes_coldprefill() {
    let mut p = pol(); // Balanced-like, allow_cloud=true, cascade=true
    p.cold_prefill_gate_secs = 8.0;
    let s = Signals {
        prompt_tokens: 24000, local_ctx_window: 32768, n_tools: 32, n_messages: 3,
        last_turn_tokens: 40, has_cloud_creds: true, local_capability_b: 8.0,
        cold_tokens: 19000, prefill_tok_s: 200.0, // 19000/200 = 95s > 8s
    };
    assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::ColdPrefill));
}

#[test]
fn warm_request_stays_local_despite_big_prompt() {
    let mut p = pol();
    p.cold_prefill_gate_secs = 8.0;
    let s = Signals {
        prompt_tokens: 24000, local_ctx_window: 32768, n_tools: 32, n_messages: 5,
        last_turn_tokens: 40, has_cloud_creds: true, local_capability_b: 8.0,
        cold_tokens: 300, prefill_tok_s: 200.0, // 300/200 = 1.5s < 8s → not cold-escalated
    };
    // low difficulty + under gate → cascade/local, NOT ColdPrefill
    assert!(matches!(decide(&s, &p), Decision::LocalThenCascade | Decision::Local));
}

#[test]
fn cold_big_prefill_without_cloud_stays_local() {
    let mut p = pol();
    p.cold_prefill_gate_secs = 8.0;
    p.allow_cloud = true;
    let s = Signals {
        prompt_tokens: 24000, local_ctx_window: 32768, n_tools: 32, n_messages: 3,
        last_turn_tokens: 40, has_cloud_creds: false, local_capability_b: 8.0,
        cold_tokens: 19000, prefill_tok_s: 200.0,
    };
    // no creds → cloud impossible → must NOT be ColdPrefill
    assert!(!matches!(decide(&s, &p), Decision::Cloud(RouteReason::ColdPrefill)));
}

#[test]
fn zero_prefill_tok_s_does_not_panic() {
    let mut p = pol();
    p.cold_prefill_gate_secs = 8.0;
    let s = Signals {
        prompt_tokens: 24000, local_ctx_window: 32768, n_tools: 32, n_messages: 3,
        last_turn_tokens: 40, has_cloud_creds: true, local_capability_b: 8.0,
        cold_tokens: 19000, prefill_tok_s: 0.0,
    };
    let _ = decide(&s, &p); // must not divide-by-zero panic
}
```

Add the two new fields to any existing `sig(...)` test helper's `Signals` construction so the module still compiles (`cold_tokens: 0, prefill_tok_s: 200.0`).

- [ ] **Step 3: Run tests, verify they fail**

Run: `cargo test -p localllm cold_ -- --nocapture` and `warm_request_stays_local`
Expected: FAIL — `no field cold_tokens on Signals` / `no variant ColdPrefill`.

- [ ] **Step 4: Add the Signals fields + RouteReason variant**

In `src/route/mod.rs`, add to `Signals` (after `local_capability_b`):

```rust
    /// Tokens NOT already in the local KV cache — the cold prefill this request
    /// would incur (0 when fully warm). Measured by the server via the engine.
    pub cold_tokens: usize,
    /// Measured local prefill speed (tokens/sec); server passes a fallback when
    /// no sample exists. Used to estimate cold-prefill seconds.
    pub prefill_tok_s: f64,
```

Add to `RouteReason` (after `Difficulty`):

```rust
    /// Local would need an expensive cold KV prefill; served from cloud while a
    /// background local prefill warms the cache for the next turn.
    ColdPrefill,
```

- [ ] **Step 5: Add the decision branch**

In `decide` (`src/route/mod.rs`), insert AFTER the cloud-possible check (the `if !p.allow_cloud || !s.has_cloud_creds` block that returns `Local`) and BEFORE the difficulty branch:

```rust
    // 2b. Cold-prefill gate: a large uncached prefill would make the local
    // model take too long. Serve from cloud now; the caller warms local in the
    // background. Reaching here means cloud is allowed AND creds are present.
    let prefill_secs = if s.prefill_tok_s > 0.0 {
        s.cold_tokens as f64 / s.prefill_tok_s
    } else {
        f64::INFINITY
    };
    if prefill_secs > p.cold_prefill_gate_secs {
        return Decision::Cloud(RouteReason::ColdPrefill);
    }
```

- [ ] **Step 6: Run tests, verify pass**

Run: `cargo test -p localllm -- routing` (or the four test names)
Expected: PASS. Also run `cargo test -p localllm` — existing routing tests still green (they build `Signals` via the updated helper).

- [ ] **Step 7: Commit**

```bash
git add src/route/mod.rs src/route/policy.rs
git commit -m "feat(route): ColdPrefill decision branch (pure, gated on estimated prefill secs)"
```

---

### Task 2: Engine — publish `PrefixState`, add `estimate_cold_tokens` + `prefill` jobs

**Files:**
- Modify: `src/engine_llama.rs` (`Job` enum, worker loop, `LlamaEngine` struct + methods, tests)

**Interfaces:**
- Produces: `pub struct PrefixState { pub len: usize }` (KV-resident token count)
- Produces: `LlamaEngine::prefix_len(&self) -> usize`
- Produces: `Job::EstimateCold { req: ChatRequest, reply: oneshot::Sender<usize> }`
- Produces: `Job::Prefill { req: ChatRequest, reply: oneshot::Sender<Result<()>> }`
- Produces: `LlamaEngine::estimate_cold_tokens(&self, req) -> usize` and `prefill(&self, req) -> Result<()>` (async wrappers over the jobs)

Context for the implementer: the worker loop is at `src/engine_llama.rs:498`, jobs are `Job::Generate`/`Job::Stream`, and `cached_tokens: Vec<LlamaToken>` (line 449) is the KV-resident prefix. `run_decode_loop` already tokenizes the prompt and computes the longest-common-prefix vs `cached_tokens` before decoding. Read `run_decode_loop` and factor out two reusable helpers rather than duplicating.

- [ ] **Step 1: Factor out prompt tokenization + common-prefix helpers**

Read `run_decode_loop`. Extract (without changing its behavior) two free functions it can call:

```rust
/// Render + tokenize `req` into the model's prompt token ids.
fn tokenize_prompt(model: &LlamaModel, req: &ChatRequest) -> Result<Vec<LlamaToken>>;

/// Length of the longest shared prefix of two token slices.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}
```

Have `run_decode_loop` call these so the logic exists in exactly one place.

- [ ] **Step 2: Add a `common_prefix_len` unit test**

```rust
#[test]
fn common_prefix_len_counts_shared_leading_tokens() {
    let a = vec![LlamaToken(1), LlamaToken(2), LlamaToken(3)];
    let b = vec![LlamaToken(1), LlamaToken(2), LlamaToken(9)];
    assert_eq!(common_prefix_len(&a, &b), 2);
    assert_eq!(common_prefix_len(&a, &[]), 0);
}
```

Run: `cargo test -p localllm common_prefix_len` → PASS.

- [ ] **Step 3: Add the `Job` variants**

In the `Job` enum (near line 105) add:

```rust
    /// Cheap query: how many prompt tokens are NOT in the current KV cache.
    EstimateCold {
        req: ChatRequest,
        reply: tokio::sync::oneshot::Sender<usize>,
    },
    /// Decode the prompt into the KV cache WITHOUT sampling, to warm the prefix.
    Prefill {
        req: ChatRequest,
        reply: tokio::sync::oneshot::Sender<Result<()>>,
    },
```

- [ ] **Step 4: Handle the new jobs in the worker loop**

In `while let Ok(job) = rx.recv()` (line 498), add arms:

```rust
            Job::EstimateCold { req, reply } => {
                let cold = match tokenize_prompt(&model, &req) {
                    Ok(toks) => toks.len().saturating_sub(common_prefix_len(&cached_tokens, &toks)),
                    Err(_) => req_estimate_fallback(&req), // approx: full prompt is cold
                };
                let _ = reply.send(cold);
            }
            Job::Prefill { req, reply } => {
                // Reuse the decode loop but stop after prefill: pass a sink
                // callback that returns false immediately so no tokens are
                // sampled; run_decode_loop still updates `cached_tokens` with
                // the prompt prefix. Wrap in catch_unwind + recover_context like
                // Generate. See Generate arm (line 500) for the exact pattern.
                let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_decode_loop(&model, &mut ctx, &mut cached_tokens, &req,
                        kv_cache_dir.as_deref(), &provenance_prefix, |_piece| false)
                }));
                let out = match panic_result {
                    Ok(r) => r.map(|_| ()),
                    Err(_) => {
                        recover_context(&model, backend, &mut ctx, &mut cached_tokens, ctx_len, kv_cache_type);
                        Err(anyhow::anyhow!("prefill panicked"))
                    }
                };
                let _ = reply.send(out);
            }
```

Add a tiny helper `fn req_estimate_fallback(req: &ChatRequest) -> usize` returning `crate::route::estimate_prompt_tokens(req)` (treat the whole prompt as cold when tokenization failed).

Verify `run_decode_loop`'s callback contract: returning `false` from the piece callback must stop sampling after prefill without erroring. If its signature differs, adapt the closure; if it cannot stop cleanly after prefill, add a `prefill_only: bool` parameter to `run_decode_loop` that skips the sampling loop after the prompt decode, and thread `false` through the Generate/Stream callers. Report NEEDS_CONTEXT if neither is feasible from the code.

- [ ] **Step 5: Publish `PrefixState` after each generating job + add accessor**

Add a shared field to `LlamaEngine`:

```rust
    prefix_len: std::sync::Arc<std::sync::atomic::AtomicUsize>,
```

Pass a clone into the worker thread. After each `Generate`, `Stream`, and `Prefill` job finishes (and after the disk warm-start sets `cached_tokens`, line ~472), store the length:

```rust
    prefix_len.store(cached_tokens.len(), std::sync::atomic::Ordering::Relaxed);
```

Add the accessor + async job wrappers on `LlamaEngine`:

```rust
    pub fn prefix_len(&self) -> usize {
        self.prefix_len.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub async fn estimate_cold_tokens(&self, req: ChatRequest) -> usize {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.tx.send(Job::EstimateCold { req, reply: tx }).is_err() { return usize::MAX; }
        rx.await.unwrap_or(usize::MAX)
    }
    pub async fn prefill(&self, req: ChatRequest) -> anyhow::Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx.send(Job::Prefill { req, reply: tx })
            .map_err(|_| anyhow::anyhow!("engine gone"))?;
        rx.await.unwrap_or_else(|_| Err(anyhow::anyhow!("engine dropped reply")))
    }
```

(`self.tx` is `std_mpsc::SyncSender<Job>`, line 127 — `.send` is blocking; that's fine, it's a fast enqueue.)

- [ ] **Step 6: Engine integration test — prefill warms the prefix**

Add a test that loads the tiny test model already used by other engine tests (find an existing engine test for the load pattern), runs `prefill(req)` on a prompt, then asserts a following `estimate_cold_tokens(same_req)` returns ~0 (fully warm), and `prefix_len() > 0`.

```rust
#[tokio::test]
async fn prefill_warms_prefix_so_next_estimate_is_low() {
    let eng = load_test_engine().await; // reuse existing helper/pattern
    let req = simple_req("hello world this is a warm-up prompt");
    eng.prefill(req.clone()).await.unwrap();
    assert!(eng.prefix_len() > 0);
    assert!(eng.estimate_cold_tokens(req).await <= 1);
}
```

If no reusable test-model helper exists, gate this test behind the same `#[cfg]`/env the other engine tests use; if engine tests are not run in CI without a model, mark this test `#[ignore]` with a comment and note it in the report.

- [ ] **Step 7: Run tests**

Run: `cargo test -p localllm engine` (or the specific names)
Expected: `common_prefix_len` PASS; the prefill test PASS (or ignored per Step 6).

- [ ] **Step 8: Commit**

```bash
git add src/engine_llama.rs
git commit -m "feat(engine): PrefixState + estimate_cold_tokens + prefill-only job"
```

---

### Task 3: Manager trait — expose estimate + prefill + prefix_len

**Files:**
- Modify: `src/model_manager.rs` (trait the server calls + real + test impls)

**Interfaces:**
- Produces: trait methods `estimate_cold_tokens(&self, req) -> usize`, `prefill(&self, req)`, `prefix_len(&self) -> usize` on the engine trait the `ModelManager` exposes to the server.

Context: `ModelManager` (line 69) fronts the active `LlamaEngine` and defines `generate`/`generate_stream` (lines 271, 287). There is a test double at lines 447-494. Mirror those for the three new methods.

- [ ] **Step 1: Add the three methods to the manager**

Add to `ModelManager`'s impl, delegating to the active engine (follow how `generate` reaches the engine at line 271):

```rust
    pub fn prefix_len(&self) -> usize {
        // active engine's prefix length, 0 when switching/none (mirror generate's guard)
        self.with_active(|e| e.prefix_len()).unwrap_or(0)
    }
    pub async fn estimate_cold_tokens(&self, req: ChatRequest) -> usize {
        match self.active_engine() { Some(e) => e.estimate_cold_tokens(req).await, None => usize::MAX }
    }
    pub async fn prefill(&self, req: ChatRequest) -> anyhow::Result<()> {
        match self.active_engine() { Some(e) => e.prefill(req).await, None => Ok(()) }
    }
```

Use whatever accessor `generate` already uses to reach the active engine (read lines 271-300; replicate that guard exactly — the pseudocode `with_active`/`active_engine` names above must be replaced with the real accessor). If switching, `prefix_len` returns 0 (safe: server treats as fully cold) and `estimate_cold_tokens` returns `usize::MAX` (safe: escalates).

- [ ] **Step 2: Update the test double (lines 447-494)**

Add matching methods to the test impl: `prefix_len` returns a field the test can set (default 0), `estimate_cold_tokens` returns a field (default 0), `prefill` returns `Ok(())`. Keep them trivial — they exist so `server.rs` compiles and its tests run.

- [ ] **Step 3: Build**

Run: `cargo build -p localllm && cargo test -p localllm model_manager`
Expected: clean build; manager tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/model_manager.rs
git commit -m "feat(manager): expose prefix_len, estimate_cold_tokens, prefill"
```

---

### Task 4: Route-log fields + dashboard display

**Files:**
- Modify: `src/route_log.rs` (`RouteEntry` fields + a round-trip test)
- Modify: `src/manager_ui/app.js` (`scoreExplainNode` — show warm/cold)

**Interfaces:**
- Produces: `RouteEntry.cold_tokens: Option<u64>`, `prefill_secs_est: Option<f64>`, `was_cold: Option<bool>`, `bg_prefill_fired: Option<bool>`

- [ ] **Step 1: Add the fields to `RouteEntry`**

In `src/route_log.rs`, in `RouteEntry` (after `local_model`, line ~55):

```rust
    /// Uncached tokens the local model would have to prefill (cold prefill size).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold_tokens: Option<u64>,
    /// Estimated cold-prefill seconds at decision time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_secs_est: Option<f64>,
    /// True when the KV cache missed at decision time (a cold prefill).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub was_cold: Option<bool>,
    /// True when a background local prefill was fired to warm the next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bg_prefill_fired: Option<bool>,
```

- [ ] **Step 2: Round-trip test**

```rust
#[test]
fn route_entry_round_trips_cold_prefill_fields() {
    let e = RouteEntry { ts: 1, rid: "z".into(), dest: "cloud".into(),
        cold_tokens: Some(19000), prefill_secs_est: Some(95.0),
        was_cold: Some(true), bg_prefill_fired: Some(true), ..Default::default() };
    let s = serde_json::to_string(&e).unwrap();
    let back: RouteEntry = serde_json::from_str(&s).unwrap();
    assert_eq!(back.cold_tokens, Some(19000));
    assert_eq!(back.was_cold, Some(true));
    assert_eq!(back.bg_prefill_fired, Some(true));
}
```

Run: `cargo test -p localllm route_entry_round_trips_cold` → after adding fields, PASS.

- [ ] **Step 3: Show warm/cold in the score popover**

In `src/manager_ui/app.js` `scoreExplainNode(e)`, before the final `why` block, add (use `textContent`-safe `el`; numeric interpolation is safe):

```javascript
  if (e.was_cold != null) {
    line(`cache local: <b>${e.was_cold ? "frio (miss)" : "quente"}</b>`);
    if (e.cold_tokens != null) {
      line(`prefill frio ≈ ${fmtNum(e.cold_tokens)} tok${e.prefill_secs_est != null ? ` (~${e.prefill_secs_est.toFixed(0)}s)` : ""}`);
    }
    if (e.bg_prefill_fired) line(`→ cloud agora; aquecendo local em background`);
  }
```

Run: `node --check src/manager_ui/app.js` → exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/route_log.rs src/manager_ui/app.js
git commit -m "feat(dashboard): log + show warm/cold prefill state per decision"
```

---

### Task 5: Server wiring — measure, decide, background-prefill, log

**Files:**
- Modify: `src/server.rs` (the decision point ~line 263-329 in `route_decision`/equivalent; add cold measurement, ColdPrefill handling, log fields)

**Interfaces:**
- Consumes: `manager.prefix_len()`, `manager.estimate_cold_tokens(req)`, `manager.prefill(req)` (Task 3); `Signals.cold_tokens`/`prefill_tok_s` and `RouteReason::ColdPrefill` (Task 1); `RouteEntry` cold fields (Task 4); `policy.cold_prefill_gate_secs` (Task 1).

Context: the decision code builds `Signals`, calls `crate::route::decide`, logs a `RouteEntry` (append at ~line 312), and the caller reverse-proxies to cloud on a `Cloud(_)` decision. Read lines ~250-370.

- [ ] **Step 1: Measure cold_tokens + prefill_tok_s and fill Signals**

Before constructing `Signals`, add:

```rust
    // Cold-prefill estimate: how many prompt tokens are NOT in the local KV
    // cache, and the local model's measured prefill speed.
    let cold_tokens = state.manager.estimate_cold_tokens(internal.clone()).await;
    let was_cold = cold_tokens > COLD_TOKENS_WARM_MARGIN; // e.g. const = 512
    let prefill_tok_s = crate::route_log::local_prefill_tok_s().unwrap_or(DEFAULT_PREFILL_TOK_S);
```

Add module consts near the top of the fn's module: `const COLD_TOKENS_WARM_MARGIN: usize = 512;` and `const DEFAULT_PREFILL_TOK_S: f64 = 150.0;` (conservative M-series 8B prefill floor). Set `Signals.cold_tokens = cold_tokens` and `Signals.prefill_tok_s = prefill_tok_s` in the existing `Signals { … }` literal.

Add `local_prefill_tok_s()` to `src/route_log.rs`: read the log, compute local prefill tok/s from recorded outcomes (prompt_tokens / ttft in seconds averaged over recent local requests); return `Option<f64>` (None when no sample). Keep it best-effort and cheap (reuse `read_all` + the same recency window the latency rollup uses). Add a small unit test that a synthetic local outcome yields a positive rate.

- [ ] **Step 2: On ColdPrefill, fire a background local prefill**

After `decide` returns and you know the `Decision`, add:

```rust
    let bg_prefill_fired = if matches!(decision, crate::route::Decision::Cloud(crate::route::RouteReason::ColdPrefill)) {
        let mgr = state.manager.clone();
        let warm_req = internal.clone();
        tokio::spawn(async move { let _ = mgr.prefill(warm_req).await; });
        true
    } else {
        false
    };
```

(`internal` is the parsed `ChatRequest`; confirm it is `Clone` — `ChatRequest` derives Clone. `state.manager` is an `Arc`.)

- [ ] **Step 3: Log the new fields**

In the `RouteEntry { … }` appended at ~line 312, add:

```rust
        cold_tokens: Some(cold_tokens as u64),
        prefill_secs_est: Some(if prefill_tok_s > 0.0 { cold_tokens as f64 / prefill_tok_s } else { f64::INFINITY }),
        was_cold: Some(was_cold),
        bg_prefill_fired: Some(bg_prefill_fired),
```

Guard `prefill_secs_est` against serializing infinity: if `!secs.is_finite()`, store `None`. Inline:

```rust
        prefill_secs_est: {
            let secs = if prefill_tok_s > 0.0 { cold_tokens as f64 / prefill_tok_s } else { f64::INFINITY };
            secs.is_finite().then_some(secs)
        },
```

- [ ] **Step 4: Update the routing log line (observability)**

The existing `tracing::info!` route line (line ~266) — append `cold_tok={cold_tokens} was_cold={was_cold}` to the fields so `/tmp/localllm.log` shows the new signal.

- [ ] **Step 5: Build + test**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean build, all green (server tests use the Task 3 test double; add trailing fields to any `Signals` literal in server tests if present).

- [ ] **Step 6: Commit**

```bash
git add src/server.rs src/route_log.rs
git commit -m "feat(server): measure cold prefill, route ColdPrefill, warm local in background, log"
```

---

### Task 6: Config knob — `cold_prefill_gate_secs` endpoint + Config UI

**Files:**
- Modify: `src/server.rs` (admin GET/POST `/admin/cold-prefill-gate`, mirroring `/admin/threshold` at lines 809-810 + `handle_threshold_get/set`)
- Modify: `src/settings.rs` (persist the gate, mirroring how the Balanced threshold persists)
- Modify: `src/manager_ui/app.js` (`renderConfig` — add a gate control like `renderThresholdConfig`)

**Interfaces:**
- Produces: `GET/POST /admin/cold-prefill-gate` → `{ "secs": <f64> }`

- [ ] **Step 1: Persist + load the gate**

Read how the Balanced threshold is persisted (`grep -n threshold src/settings.rs`). Add a parallel `load_cold_prefill_gate() -> f64` (default `360.0`) and `save_cold_prefill_gate(secs: f64)`. Apply the loaded value into the active `RoutingPolicy.cold_prefill_gate_secs` wherever the profile/threshold is applied at startup and on change (mirror the threshold's application path).

- [ ] **Step 2: Add the admin handlers**

Mirror `handle_threshold_get`/`handle_threshold_set` (find them near line 809). Add:

```rust
async fn handle_cold_gate_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    Json(json!({ "secs": crate::settings::load_cold_prefill_gate() })).into_response()
}
async fn handle_cold_gate_set(State(state): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<serde_json::Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) { return resp; }
    let secs = body.get("secs").and_then(|v| v.as_f64()).unwrap_or(360.0).clamp(1.0, 3600.0);
    crate::settings::save_cold_prefill_gate(secs);
    // apply into the live policy (mirror threshold's apply)
    Json(json!({ "secs": secs })).into_response()
}
```

Register the route next to `/admin/threshold`:

```rust
        .route("/admin/cold-prefill-gate", get(handle_cold_gate_get).post(handle_cold_gate_set))
```

- [ ] **Step 3: Add a settings round-trip test**

Mirror an existing settings test: save `120.0`, load, assert `120.0`; default (unset) loads `360.0`. Use the settings test-isolation pattern already in the file (`LOCALLLM_SETTINGS` temp override).

Run: `cargo test -p localllm cold_prefill_gate` → PASS.

- [ ] **Step 4: Config UI control**

In `src/manager_ui/app.js`, mirror `renderThresholdConfig` (a panel with a numeric input) as `renderColdGateConfig(container)` hitting `/admin/cold-prefill-gate`, labelled "Limite de prefill frio (s)" with sub-text: "Acima disso, o 1º request de contexto grande vai pra cloud e o local aquece em background. O timeout do seu cliente deve ser maior que este valor." Call it from `renderConfig` alongside `renderThresholdConfig(shell)`.

Run: `node --check src/manager_ui/app.js` → exit 0.

- [ ] **Step 5: Build + test**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean, green.

- [ ] **Step 6: Commit**

```bash
git add src/server.rs src/settings.rs src/manager_ui/app.js
git commit -m "feat(config): cold-prefill gate endpoint + Config page control"
```

---

### Task 7: Build + verify

**Files:** none (build + manual verify)

- [ ] **Step 1: Full build + tests**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean, all pass.

- [ ] **Step 2: Rebuild bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS`.

- [ ] **Step 3: Manual check**

- Config page shows the "Limite de prefill frio (s)" control; set it low (e.g. 5s) to force escalation.
- Start a fresh conversation with a large context from a client → the first request routes cloud with `ColdPrefill`; the score popover shows "cache local: frio" + "aquecendo local em background".
- Send a follow-up turn → routes local (warm); popover shows "cache local: quente".

- [ ] **Step 4: Commit any tweaks**

```bash
git add -A && git commit -m "chore: cold-prefill routing verification tweaks"
```

---

## Self-Review

**Spec coverage:**
- Engine exposes cached-prefix state → Task 2 (`prefix_len`/`PrefixState`). ✓
- Router estimates cold_tokens → Task 2 `estimate_cold_tokens` + Task 5 measurement. ✓
- Estimate prefill time, gate, ColdPrefill branch → Task 1. ✓
- Cold+cloud → Cloud + background prefill → Task 5 (spawn `prefill`). ✓
- Warm → local; under-gate → local → Task 1 branch order (falls through). ✓
- Log warm/cold + cold_tokens + prefill est + bg flag → Task 4 (fields) + Task 5 (populate). ✓
- Dashboard shows warm/cold → Task 4 (scoreExplainNode). ✓
- Config gate knob, default 360, UI note about client timeout → Task 6. ✓
- Best-effort background prefill; pure `decide`; single-thread FIFO; div-by-zero guard → Tasks 1, 2, 5 + Global Constraints. ✓
- Backward-compat serde on new fields → Task 4. ✓

**Placeholder scan:** The engine internals (`run_decode_loop` reuse, the real active-engine accessor in the manager) point at concrete source locations to read and factor, with exact new-surface signatures given — no invented internals. Two conditional escalation paths (prefill callback vs `prefill_only` param; ignored engine test when no model) are spelled out with the decision rule and a NEEDS_CONTEXT escape. No "TBD/TODO/handle edge cases" left.

**Type consistency:** `cold_tokens: usize` (Signals) → `Some(cold_tokens as u64)` (RouteEntry). `prefill_tok_s: f64`. `RouteReason::ColdPrefill`, `cold_prefill_gate_secs: f64`, `estimate_cold_tokens(req)->usize`, `prefill(req)->Result<()>`, `prefix_len()->usize` used identically across tasks. Endpoint `/admin/cold-prefill-gate` → `{ secs }` consistent between Task 6 handler and UI.
