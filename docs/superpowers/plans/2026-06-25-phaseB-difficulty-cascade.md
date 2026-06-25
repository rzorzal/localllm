# Phase B — Difficulty Score + Cascade Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route in-window requests to cloud when a cheap pre-generation difficulty score exceeds the active profile's threshold, and otherwise serve locally with a cascade fallback that escalates a weak local result to cloud.

**Architecture:** Extend the pure `route::decide` with a documented heuristic `difficulty_score(Signals) -> f64` and full branch logic (context gate → cloud-impossible → difficulty threshold → cascade-or-local). Add a pure `route::is_weak_result` predicate. In the HTTP handlers, when the decision is `LocalThenCascade`, run the local model on the buffered/non-stream paths and, if the result is weak (truncated by length) or generation failed, reverse-proxy the original request to cloud via the existing `cloud::forward`.

**Tech Stack:** Rust, axum 0.7, reqwest 0.12, tokio. Dev: wiremock.

## Global Constraints

- Builds on Phase A (branch `feat/model-router`): `route::{Signals, Decision, RouteReason, RoutingPolicy, Profile, decide, estimate_prompt_tokens}`, `cloud::{Provider, forward}`, and `AppState { policy, local_ctx_window, gen, model_id }` already exist.
- **Pure routing logic stays I/O-free** in `src/route/`. Only the handler task touches HTTP.
- **Cascade is for buffered/non-stream paths only.** The incremental-stream path (`stream == true && n_tools == 0`) must NOT cascade — tokens are already on the wire. There it behaves as plain local.
- Difficulty score is a **transparent, documented heuristic** in `[0,1]`; a learned router is out of scope (YAGNI).
- Profile knobs are fixed (from Phase A): SaveTokens(threshold 0.9, cascade true), Balanced(0.6, true), MaxQuality(0.2, cascade false), LocalOnly(allow_cloud false). MaxQuality achieves cloud-first via its low threshold (no separate forced-profile branch).
- Credentials: cascade reuses the same client credential headers; never log or store credential values.
- TDD: failing test first; complete code in every step; commit per task when green; test output pristine.

---

### Task 1: Difficulty score + full `decide` logic

**Files:**
- Modify: `src/route/mod.rs` (add `difficulty_score`; replace the Phase-A `decide` body; update one existing unit test; add new unit tests)

**Interfaces:**
- Consumes: `Signals`, `RoutingPolicy`, `Decision`, `RouteReason` (Phase A).
- Produces: `fn difficulty_score(s: &Signals) -> f64`; a `decide` that can now return `Decision::Cloud(RouteReason::Difficulty)` and `Decision::LocalThenCascade`.

- [ ] **Step 1: Update the now-stale Phase-A test and add the new failing tests**

In `src/route/mod.rs`, inside the existing `#[cfg(test)] mod tests`, REPLACE the existing `under_window_stays_local` test with this (its expectation changes: an in-window request with creds under a cascade profile now yields `LocalThenCascade`, not `Local`):

```rust
    #[test]
    fn under_window_with_creds_and_cascade_returns_local_then_cascade() {
        // SaveTokens has cascade=true and a high threshold; a tiny in-window
        // request scores well below it, so it serves local with cascade fallback.
        assert_eq!(
            decide(&sig(100, true, 1000), &pol()),
            Decision::LocalThenCascade
        );
    }
```

Then ADD these tests to the same module:

```rust
    #[test]
    fn under_window_without_creds_is_plain_local() {
        // No credential to forward → cannot cascade → plain Local.
        assert_eq!(decide(&sig(100, false, 1000), &pol()), Decision::Local);
    }

    #[test]
    fn high_difficulty_in_window_routes_cloud() {
        // Balanced threshold 0.6; a nearly-full window pushes the score over it.
        let p = Profile::Balanced.policy();
        // 850/1000 = 0.85 ctx fill (under the 0.9 ctx_gate, so NOT overflow),
        // score = 0.6*0.85 = 0.51 from ctx alone; add tools to clear 0.6.
        let s = Signals {
            prompt_tokens: 850,
            local_ctx_window: 1000,
            n_tools: 12,
            n_messages: 1,
            has_cloud_creds: true,
        };
        assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn max_quality_routes_nontrivial_to_cloud() {
        // MaxQuality threshold 0.2; a modest request clears it.
        let p = Profile::MaxQuality.policy();
        let s = Signals {
            prompt_tokens: 400,
            local_ctx_window: 1000,
            n_tools: 0,
            n_messages: 1,
            has_cloud_creds: true,
        };
        // ctx_fill 0.4 → 0.6*0.4 = 0.24 > 0.2 → cloud.
        assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn max_quality_keeps_trivial_local_no_cascade() {
        // Trivial request scores under 0.2; MaxQuality has cascade=false → plain Local.
        let p = Profile::MaxQuality.policy();
        assert_eq!(decide(&sig(50, true, 1000), &p), Decision::Local);
    }

    #[test]
    fn difficulty_score_is_low_for_trivial_and_high_for_full() {
        let trivial = Signals {
            prompt_tokens: 10, local_ctx_window: 1000, n_tools: 0,
            n_messages: 1, has_cloud_creds: true,
        };
        let full = Signals {
            prompt_tokens: 1000, local_ctx_window: 1000, n_tools: 12,
            n_messages: 20, has_cloud_creds: true,
        };
        assert!(difficulty_score(&trivial) < 0.1);
        assert!(difficulty_score(&full) > 0.95);
        // monotonic in tool count
        let more_tools = Signals { n_tools: 6, ..trivial };
        assert!(difficulty_score(&more_tools) > difficulty_score(&trivial));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib route:: 2>&1 | tail -25`
Expected: FAIL — `difficulty_score` undefined and the new `decide` outcomes not yet produced.

- [ ] **Step 3: Add `difficulty_score` and rewrite `decide`**

In `src/route/mod.rs`, ADD the scoring function (place it just above `decide`):

```rust
/// Cheap pre-generation difficulty score in `[0.0, 1.0]`. Higher means the
/// request is harder/bigger and more likely to need the cloud model.
///
/// Transparent heuristic blend (weights sum to 1.0); context fill dominates:
/// - context fill: how full the local window is (prompt_tokens / window)
/// - tool load: tool count, saturating at 12 (agentic complexity)
/// - depth: message count, saturating at 20 (long multi-turn is harder)
///
/// A learned router could replace this later; the `decide` interface is unchanged.
pub fn difficulty_score(s: &Signals) -> f64 {
    let ctx_fill = if s.local_ctx_window == 0 {
        1.0
    } else {
        (s.prompt_tokens as f64 / s.local_ctx_window as f64).min(1.0)
    };
    let tool_load = (s.n_tools as f64 / 12.0).min(1.0);
    let depth = (s.n_messages as f64 / 20.0).min(1.0);
    0.6 * ctx_fill + 0.25 * tool_load + 0.15 * depth
}
```

Then REPLACE the entire existing `decide` function body with:

```rust
/// Decide where a request runs from cheap signals and the active policy.
///
/// Order: (1) hard context gate, (2) cloud-impossible shortcut, (3) difficulty
/// score vs the profile threshold, (4) in-window low-score → cascade or local.
pub fn decide(s: &Signals, p: &RoutingPolicy) -> Decision {
    // 1. Hard context gate: prompt too big for the local window.
    let over_window =
        s.prompt_tokens as f64 > s.local_ctx_window as f64 * p.ctx_gate_frac;
    if over_window {
        if p.allow_cloud && s.has_cloud_creds {
            return Decision::Cloud(RouteReason::ContextOverflow);
        }
        return Decision::LocalNoCreds;
    }

    // 2. Cloud impossible (profile forbids it or no credential to forward) → local.
    if !p.allow_cloud || !s.has_cloud_creds {
        return Decision::Local;
    }

    // 3. Difficulty above the profile threshold → cloud now.
    //    (MaxQuality's low threshold makes this cloud-first; trivial requests
    //     still score below it and stay local.)
    if difficulty_score(s) > p.escalation_threshold {
        return Decision::Cloud(RouteReason::Difficulty);
    }

    // 4. In-window, below threshold: local, with cascade fallback when enabled.
    if p.cascade {
        Decision::LocalThenCascade
    } else {
        Decision::Local
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib route:: 2>&1 | tail -25`
Expected: PASS — all route unit tests (the updated test plus the five new ones, and the pre-existing context-gate/estimate tests).

- [ ] **Step 5: Run the full suite to confirm no integration regressions**

Run: `cargo test 2>&1 | tail -20`
Expected: PASS. The handler still treats `LocalThenCascade` via its catch-all arm (falls through to local) until Task 2, and no integration test sends an in-window request WITH a credential, so behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/route/mod.rs
git commit -m "feat(route): difficulty score + full decide (threshold + cascade decision)"
```

---

### Task 2: Cascade fallback in the handlers

**Files:**
- Modify: `src/route/mod.rs` (add `is_weak_result`)
- Modify: `src/server.rs` (capture decision, add cascade helper, wire into both handlers' buffered-stream + non-stream branches)
- Modify: `src/lib.rs` (add `FakeGenWeak` + `router_for_test_with`)
- Test: `tests/http.rs` (cascade escalation + no-cascade integration tests); `src/route/mod.rs` (`is_weak_result` unit tests)

**Interfaces:**
- Consumes: `route::Decision::LocalThenCascade`, `cloud::{Provider, forward}`, `crate::api::common::{ChatResult, FinishReason, ContentPart}`.
- Produces:
  - `fn is_weak_result(r: &ChatResult) -> bool` (in `route`)
  - `async fn cascade_or_result(want_cascade: bool, gen_result: anyhow::Result<ChatResult>, provider: crate::cloud::Provider, headers: &HeaderMap, raw: Bytes, rid: &str, api: &str) -> Result<ChatResult, axum::response::Response>` (in `server`)
  - test helpers `FakeGenWeak`, `router_for_test_with(gen, policy, window)` (in `lib`)

- [ ] **Step 1: Add the `is_weak_result` predicate with failing tests**

In `src/route/mod.rs`, add to the `#[cfg(test)] mod tests` module:

```rust
    #[test]
    fn weak_result_is_only_length_truncation() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let mk = |fr| ChatResult {
            content: vec![ContentPart::Text("x".into())],
            finish_reason: fr,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        assert!(is_weak_result(&mk(FinishReason::Length)));
        assert!(!is_weak_result(&mk(FinishReason::Stop)));
        assert!(!is_weak_result(&mk(FinishReason::ToolCalls)));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib route::tests::weak_result 2>&1 | tail -10`
Expected: FAIL — `is_weak_result` undefined.

- [ ] **Step 3: Implement `is_weak_result`**

In `src/route/mod.rs`, add near `decide` (use the already-imported `ChatRequest`; add the `ChatResult`/`FinishReason` path inline):

```rust
/// Whether a local result is "weak" enough to escalate to cloud. A length
/// truncation means the local model hit its token budget mid-answer — a strong
/// signal it under-served the request. (Logprob/judge confidence is future work.)
pub fn is_weak_result(r: &crate::api::common::ChatResult) -> bool {
    r.finish_reason == crate::api::common::FinishReason::Length
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib route::tests::weak_result 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Add test support (`FakeGenWeak`, `router_for_test_with`) in `src/lib.rs`**

In `src/lib.rs`, after the existing `FakeGen` impl, add a weak generator that always reports a length-truncated text answer:

```rust
/// A fake generator that always returns a length-truncated text answer.
/// Used to exercise the cascade path (weak local result → escalate to cloud).
pub struct FakeGenWeak;

#[async_trait::async_trait]
impl Generator for FakeGenWeak {
    async fn generate(&self, _req: crate::api::common::ChatRequest) -> anyhow::Result<ChatResult> {
        Ok(ChatResult {
            content: vec![ContentPart::Text("local-weak-answer".to_string())],
            finish_reason: FinishReason::Length,
            prompt_tokens: 5,
            completion_tokens: 5,
        })
    }

    async fn generate_stream(
        &self,
        _req: crate::api::common::ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
            Ok(StreamDelta { text: Some("local-weak-answer".into()), done: false, finish_reason: None }),
            Ok(StreamDelta { text: None, done: true, finish_reason: Some(FinishReason::Length) }),
        ];
        Ok(Box::pin(futures::stream::iter(deltas)))
    }
}
```

Then REPLACE the existing `router_for_test` function with a parameterized builder plus a thin default wrapper:

```rust
/// Build a test router with a custom generator, routing policy, and local
/// context window. Lets tests drive specific routing/cascade decisions.
pub fn router_for_test_with(
    gen: Arc<dyn Generator>,
    policy: crate::route::RoutingPolicy,
    local_ctx_window: usize,
) -> Router {
    let policy = Arc::new(std::sync::RwLock::new(policy));
    crate::server::router(gen, "test-model".to_string(), policy, local_ctx_window)
}

/// Build a test router wired to `FakeGen`, the default (SaveTokens) policy, and
/// a small context window so the ctx gate is testable.
pub fn router_for_test() -> Router {
    router_for_test_with(
        Arc::new(FakeGen),
        crate::route::Profile::default().policy(),
        1000,
    )
}
```

- [ ] **Step 6: Write the failing cascade integration tests**

Append to `tests/http.rs`:

```rust
/// In-window request with a credential under a cascade profile (SaveTokens):
/// the local model returns a length-truncated (weak) result, so the router
/// escalates to the cloud upstream and returns the cloud response.
#[tokio::test]
async fn weak_local_with_cascade_escalates_to_cloud() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"routed":"cloud"}"#))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    // Small in-window prompt → decide() = LocalThenCascade (SaveTokens, creds present).
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGenWeak),
        localllm::route::Profile::SaveTokens.policy(),
        1000,
    );
    let resp =
        localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test")
            .await;
    assert_eq!(resp["routed"], "cloud");

    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// Trivial in-window request with a credential under MaxQuality (cascade=false,
/// but trivial score stays under the 0.2 threshold → plain Local). A weak local
/// result must NOT escalate; the local answer is returned as-is.
#[tokio::test]
async fn weak_local_without_cascade_stays_local() {
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGenWeak),
        localllm::route::Profile::MaxQuality.policy(),
        1000,
    );
    let resp =
        localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test")
            .await;
    // Anthropic maps FinishReason::Length → stop_reason "max_tokens"; the local
    // weak text is returned (no cloud escalation).
    assert_eq!(resp["content"][0]["text"], "local-weak-answer");
    assert_eq!(resp["stop_reason"], "max_tokens");
}
```

- [ ] **Step 7: Run to verify the cascade test fails**

Run: `cargo test --test http weak_local_with_cascade_escalates_to_cloud 2>&1 | tail -20`
Expected: FAIL — handler does not yet escalate; it returns the local weak answer (`max_tokens`/text) instead of `{"routed":"cloud"}`.

- [ ] **Step 8: Add the `cascade_or_result` helper in `src/server.rs`**

In `src/server.rs`, after the `route_decision` helper, add:

```rust
/// Bridge a local generation result into the cascade decision.
///
/// - `Ok(result)` and (cascade off OR result is strong) → returns `Ok(result)`;
///   the caller proceeds with the local result as before.
/// - `Ok(result)` weak (length-truncated) AND `want_cascade` → escalates: returns
///   `Err(cloud forward response)` which the caller returns directly.
/// - `Err(gen error)` → if `want_cascade`, escalate to cloud; otherwise return
///   `Err(500)`.
///
/// Consumes `raw` (the original request bytes) because escalation reverse-proxies
/// it. Only ever called on the buffered/non-stream paths — never mid-stream.
async fn cascade_or_result(
    want_cascade: bool,
    gen_result: anyhow::Result<ChatResult>,
    provider: crate::cloud::Provider,
    headers: &HeaderMap,
    raw: Bytes,
    rid: &str,
    api: &str,
) -> Result<ChatResult, axum::response::Response> {
    use axum::response::IntoResponse;
    match gen_result {
        Ok(result) => {
            if want_cascade && crate::route::is_weak_result(&result) {
                tracing::info!(target: "localllm::req", "{rid} [{api}] cascade: weak local (length) → escalating to cloud");
                Err(crate::cloud::forward(provider, headers, raw).await)
            } else {
                Ok(result)
            }
        }
        Err(e) => {
            if want_cascade {
                tracing::warn!(target: "localllm::req", "{rid} [{api}] cascade: local generate failed ({e}) → escalating to cloud");
                Err(crate::cloud::forward(provider, headers, raw).await)
            } else {
                tracing::error!(target: "localllm::req", "{rid} [{api}] 500 generate: {e}");
                Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response())
            }
        }
    }
}
```

- [ ] **Step 9: Wire the decision capture + cascade into `handle_anth_messages`**

In `src/server.rs`, in `handle_anth_messages`, REPLACE the routing-decision match block (the `match route_decision(&state, &internal, &headers) { ... }`) with a version that captures whether to cascade:

```rust
    // --- Routing decision ---
    let want_cascade = match route_decision(&state, &internal, &headers) {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [anthropic] route=cloud reason={reason:?}");
            return crate::cloud::forward(crate::cloud::Provider::Anthropic, &headers, raw).await;
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] route=local (cloud wanted but no creds/disallowed)");
            false
        }
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };
```

Then in the **buffered-stream** branch (`if stream_flag && n_tools > 0`), REPLACE the `let result = match state.gen.generate(internal).await { ... };` block with:

```rust
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::Anthropic,
            &headers,
            raw,
            &rid,
            "anthropic",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
```

And in the **non-stream** branch (the final `else` of this handler), REPLACE its `let result = match state.gen.generate(internal).await { ... };` block with the same call but WITHOUT moving `raw` twice — since the buffered-stream and non-stream branches are mutually exclusive, each may move `raw`. Use:

```rust
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::Anthropic,
            &headers,
            raw,
            &rid,
            "anthropic",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
```

> Note: the incremental-stream branch (`if stream_flag` with no tools) is left UNCHANGED — it must not cascade.

- [ ] **Step 10: Wire the same into `handle_oai_chat`**

In `src/server.rs`, in `handle_oai_chat`, REPLACE its routing-decision match block with:

```rust
    // --- Routing decision ---
    let want_cascade = match route_decision(&state, &internal, &headers) {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [openai] route=cloud reason={reason:?}");
            return crate::cloud::forward(crate::cloud::Provider::OpenAI, &headers, raw).await;
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] route=local (cloud wanted but no creds/disallowed)");
            false
        }
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };
```

Then in this handler's **buffered-stream** branch (`if stream_flag && n_tools > 0`) and its **non-stream** branch (final `else`), REPLACE each `let result = match state.gen.generate(internal).await { ... };` block with:

```rust
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::OpenAI,
            &headers,
            raw,
            &rid,
            "openai",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
```

> The incremental-stream branch stays UNCHANGED (no cascade mid-stream).

- [ ] **Step 11: Run the cascade tests, then the full suite + clippy**

Run: `cargo test --test http 2>&1 | tail -30`
Expected: PASS — `weak_local_with_cascade_escalates_to_cloud` returns `{"routed":"cloud"}`; `weak_local_without_cascade_stays_local` returns the local `local-weak-answer`/`max_tokens`; all pre-existing handler/stream/routing tests still pass.

Run: `cargo test 2>&1 | tail -15 && cargo clippy --all-targets 2>&1 | grep -E "route/|cloud.rs|src/server.rs" | grep -- "-->" || echo "no clippy in changed code"`
Expected: full suite green; no clippy warnings in `route`/`server`/`cloud`.

- [ ] **Step 12: Commit**

```bash
git add src/route/mod.rs src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): cascade weak local results to cloud (buffered/non-stream paths)"
```

---

## Phase B Acceptance

- In-window requests with a credential route to cloud when `difficulty_score > threshold` (reason `Difficulty`); otherwise serve local with cascade when the profile enables it.
- A weak (length-truncated) local result on the buffered/non-stream path escalates to the cloud upstream and returns the cloud response; a non-cascade profile returns the local result unchanged.
- The incremental-stream path never cascades.
- `difficulty_score` and the full `decide` logic are unit-tested across all four profiles; `is_weak_result` is unit-tested.
- `cargo test` and `cargo clippy --all-targets` are green; no new clippy warnings in changed code.

## Self-Review

- **Spec coverage (Phase B scope):** difficulty score (Task 1 `difficulty_score`), threshold routing + MaxQuality cloud-first (Task 1 `decide`), cascade decision `LocalThenCascade` (Task 1), weakness predicate (Task 2 `is_weak_result`), cascade escalation on buffered/non-stream with incremental-stream excluded (Task 2 handlers). Tray selector and usage/degrade remain Phases C/D.
- **Placeholder scan:** none — every step shows complete code or an exact command.
- **Type consistency:** `difficulty_score(&Signals)->f64`, `is_weak_result(&ChatResult)->bool`, `cascade_or_result(...)->Result<ChatResult, Response>`, `router_for_test_with(Arc<dyn Generator>, RoutingPolicy, usize)`, `FakeGenWeak` all referenced consistently across tasks. `route_decision` (Phase A) and `cloud::forward(Provider,&HeaderMap,Bytes)` (Phase A) reused unchanged.
- **Known carry-over:** the Phase-A test `under_window_stays_local` is intentionally replaced in Task 1 Step 1 (its stub expectation no longer holds).
