# Phase A — Routing Core + Cloud Reverse-Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route each request local-vs-cloud by a hard context-size gate; when cloud is chosen, byte-faithfully reverse-proxy the original request to the provider using the client's own credentials.

**Architecture:** A pure `route::decide(Signals, RoutingPolicy) -> Decision` function (no I/O) computes the routing choice in the HTTP handler. `Local`/`LocalNoCreds` use the unchanged engine path; `Cloud` streams the raw request bytes to `api.anthropic.com`/`api.openai.com` via `cloud::forward` and relays the raw response. Decision logic stays testable in isolation; the proxy I/O stays at the HTTP layer.

**Tech Stack:** Rust, axum 0.7, reqwest 0.12 (rustls, stream), tokio. Dev: wiremock for a mock upstream.

## Global Constraints

- **Single self-contained binary.** No external processes, no new runtime deps beyond crates already in `Cargo.toml` (reqwest is already present). New dev-only dep `wiremock` is allowed.
- **macOS / Apple Silicon** target; Metal backend default. Do not touch engine/Metal code in this phase.
- **Provider inferred from endpoint:** `/v1/messages` → Anthropic, `/v1/chat/completions` → OpenAI. Never guess.
- **Credentials reused verbatim:** forward the client's incoming auth headers unchanged. Never store or log credential values.
- **Local path behavior unchanged** when the decision is `Local`.
- Default profile is **SaveTokens** (`ctx_gate_frac = 0.95`, `cascade = true`, `allow_cloud = true`). Difficulty scoring and cascade are **out of scope for Phase A** (Phase B); `decide` returns `Local` for any in-window request this phase.
- TDD: write the failing test first, every code step shows complete code, commit after each green task.

---

### Task 1: Routing policy & profiles (`route::policy`)

**Files:**
- Create: `src/route/mod.rs`
- Create: `src/route/policy.rs`
- Modify: `src/lib.rs:1-9` (add `pub mod route;`)

**Interfaces:**
- Produces:
  - `enum Profile { SaveTokens, Balanced, MaxQuality, LocalOnly }`
  - `struct RoutingPolicy { escalation_threshold: f64, cascade: bool, ctx_gate_frac: f64, allow_cloud: bool }`
  - `impl Profile { fn policy(&self) -> RoutingPolicy }`
  - `impl Default for Profile { fn default() -> Self { Profile::SaveTokens } }`

- [ ] **Step 1: Register the module**

In `src/lib.rs`, add after `pub mod engine_llama;` (line 5):

```rust
pub mod route;
```

Create `src/route/mod.rs`:

```rust
//! Local-vs-cloud routing: a pure decision function plus its policy knobs.
//!
//! The decision logic (`decide`) performs no I/O so it is unit-testable in
//! isolation. The HTTP layer turns a `Decision` into either the local engine
//! path or a cloud reverse-proxy.

pub mod policy;

pub use policy::{Profile, RoutingPolicy};
```

- [ ] **Step 2: Write the failing test**

Create `src/route/policy.rs`:

```rust
//! Routing profiles and their tunable knobs.
//!
//! A `Profile` is the user-facing choice (set from the tray in a later phase);
//! it maps to a `RoutingPolicy` of concrete knobs consumed by `route::decide`.

/// Concrete routing knobs derived from a `Profile`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingPolicy {
    /// Difficulty-score cutoff above which a request goes to cloud (Phase B).
    pub escalation_threshold: f64,
    /// Whether a weak local result may escalate to cloud (Phase B).
    pub cascade: bool,
    /// Fraction of the local context window above which a prompt is too big
    /// for local and must go to cloud.
    pub ctx_gate_frac: f64,
    /// Whether cloud routing is permitted at all.
    pub allow_cloud: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_tokens_is_local_biased_and_allows_cloud() {
        let p = Profile::SaveTokens.policy();
        assert!(p.allow_cloud);
        assert!(p.cascade);
        assert!((p.escalation_threshold - 0.9).abs() < 1e-9);
        assert!((p.ctx_gate_frac - 0.95).abs() < 1e-9);
    }

    #[test]
    fn local_only_forbids_cloud() {
        let p = Profile::LocalOnly.policy();
        assert!(!p.allow_cloud);
        assert!(!p.cascade);
    }

    #[test]
    fn max_quality_has_low_threshold_no_cascade() {
        let p = Profile::MaxQuality.policy();
        assert!(p.allow_cloud);
        assert!(!p.cascade);
        assert!((p.escalation_threshold - 0.2).abs() < 1e-9);
    }

    #[test]
    fn default_profile_is_save_tokens() {
        assert_eq!(Profile::default(), Profile::SaveTokens);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib route::policy 2>&1 | head -30`
Expected: FAIL — `Profile` not found / does not compile.

- [ ] **Step 4: Write minimal implementation**

Add to the top of `src/route/policy.rs` (above the `#[cfg(test)]` block):

```rust
/// User-facing routing choice. Set from the tray in a later phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Local-first; cloud only when local truly cannot serve. Cheapest.
    SaveTokens,
    /// Local for easy/medium, cloud for hard/big-context.
    Balanced,
    /// Cloud-first; local only for trivial calls. Best quality, most tokens.
    MaxQuality,
    /// Never route to cloud. Pure local, zero tokens.
    LocalOnly,
}

impl Default for Profile {
    fn default() -> Self {
        Profile::SaveTokens
    }
}

impl Profile {
    /// Map a profile to its concrete routing knobs.
    pub fn policy(&self) -> RoutingPolicy {
        match self {
            Profile::SaveTokens => RoutingPolicy {
                escalation_threshold: 0.9,
                cascade: true,
                ctx_gate_frac: 0.95,
                allow_cloud: true,
            },
            Profile::Balanced => RoutingPolicy {
                escalation_threshold: 0.6,
                cascade: true,
                ctx_gate_frac: 0.9,
                allow_cloud: true,
            },
            Profile::MaxQuality => RoutingPolicy {
                escalation_threshold: 0.2,
                cascade: false,
                ctx_gate_frac: 0.75,
                allow_cloud: true,
            },
            Profile::LocalOnly => RoutingPolicy {
                escalation_threshold: 1.0,
                cascade: false,
                ctx_gate_frac: 1.0,
                allow_cloud: false,
            },
        }
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib route::policy 2>&1 | tail -15`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/route/mod.rs src/route/policy.rs
git commit -m "feat(route): routing profiles and policy knobs"
```

---

### Task 2: Decision function & token estimation (`route::decide`)

**Files:**
- Modify: `src/route/mod.rs` (add `Signals`, `Decision`, `RouteReason`, `decide`, `estimate_prompt_tokens`)
- Test: same file (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `RoutingPolicy` from Task 1; `crate::api::common::ChatRequest`.
- Produces:
  - `struct Signals { prompt_tokens: usize, local_ctx_window: usize, n_tools: usize, n_messages: usize, has_cloud_creds: bool }`
  - `enum RouteReason { ContextOverflow, Difficulty, Profile }`
  - `enum Decision { Local, LocalThenCascade, Cloud(RouteReason), LocalNoCreds }`
  - `fn decide(s: &Signals, p: &RoutingPolicy) -> Decision`
  - `fn estimate_prompt_tokens(req: &ChatRequest) -> usize`

- [ ] **Step 1: Write the failing test**

Append to `src/route/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::{ChatMessage, ChatRequest, Role};

    fn pol() -> RoutingPolicy {
        Profile::SaveTokens.policy()
    }

    fn sig(prompt_tokens: usize, has_creds: bool, window: usize) -> Signals {
        Signals {
            prompt_tokens,
            local_ctx_window: window,
            n_tools: 0,
            n_messages: 1,
            has_cloud_creds: has_creds,
        }
    }

    #[test]
    fn under_window_stays_local() {
        assert_eq!(decide(&sig(100, true, 1000), &pol()), Decision::Local);
    }

    #[test]
    fn over_window_with_creds_goes_cloud() {
        // 0.95 * 1000 = 950; 980 > 950 → overflow.
        assert_eq!(
            decide(&sig(980, true, 1000), &pol()),
            Decision::Cloud(RouteReason::ContextOverflow)
        );
    }

    #[test]
    fn over_window_without_creds_stays_local_no_creds() {
        assert_eq!(decide(&sig(980, false, 1000), &pol()), Decision::LocalNoCreds);
    }

    #[test]
    fn local_only_never_routes_cloud_even_on_overflow() {
        let p = Profile::LocalOnly.policy();
        assert_eq!(decide(&sig(5000, true, 1000), &p), Decision::LocalNoCreds);
    }

    #[test]
    fn estimate_tokens_counts_message_text() {
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                text: Some("a".repeat(400)), // 400 chars / 4 = 100 tokens
                tool_calls: vec![],
                tool_result: None,
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
            model: "m".into(),
        };
        assert_eq!(estimate_prompt_tokens(&req), 100);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib route:: 2>&1 | head -30`
Expected: FAIL — `Signals`/`decide`/`estimate_prompt_tokens` not found.

- [ ] **Step 3: Write minimal implementation**

In `src/route/mod.rs`, after the `pub use` line and before the test module, add:

```rust
use crate::api::common::ChatRequest;

/// Cheap, pre-generation signals used to decide where a request runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signals {
    /// Estimated prompt size in tokens.
    pub prompt_tokens: usize,
    /// Local model's usable context window (config `ctx_len`).
    pub local_ctx_window: usize,
    pub n_tools: usize,
    pub n_messages: usize,
    /// Whether the incoming request carries cloud credentials we can forward.
    pub has_cloud_creds: bool,
}

/// Why a request was sent to cloud.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteReason {
    /// Prompt exceeds the local context window.
    ContextOverflow,
    /// Difficulty score above the profile threshold (Phase B).
    Difficulty,
    /// Profile forces cloud (e.g. MaxQuality, Phase B).
    Profile,
}

/// The routing outcome for a single request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Serve locally, no escalation.
    Local,
    /// Serve locally; escalate to cloud if the result is weak (Phase B,
    /// buffered paths only).
    LocalThenCascade,
    /// Skip local; reverse-proxy to the provider now.
    Cloud(RouteReason),
    /// Cloud was wanted but no usable credentials / cloud disallowed; serve
    /// local and warn (local will reject an over-window prompt cleanly).
    LocalNoCreds,
}

/// Decide where a request runs from cheap signals and the active policy.
///
/// Phase A implements only the hard context-size gate. Difficulty scoring and
/// cascade are added in Phase B; in-window requests therefore return `Local`.
pub fn decide(s: &Signals, p: &RoutingPolicy) -> Decision {
    let over_window =
        s.prompt_tokens as f64 > s.local_ctx_window as f64 * p.ctx_gate_frac;
    if over_window {
        if p.allow_cloud && s.has_cloud_creds {
            return Decision::Cloud(RouteReason::ContextOverflow);
        }
        return Decision::LocalNoCreds;
    }
    Decision::Local
}

/// Estimate the prompt token count with a cheap `chars / 4` heuristic over all
/// message text, tool-call arguments, tool-result content, and tool schemas.
/// Conservative and good enough for the context gate; an exact local tokenizer
/// can replace this later if the boundary proves too coarse.
pub fn estimate_prompt_tokens(req: &ChatRequest) -> usize {
    let mut chars = 0usize;
    for m in &req.messages {
        if let Some(t) = &m.text {
            chars += t.len();
        }
        for c in &m.tool_calls {
            chars += c.name.len() + c.arguments.len();
        }
        if let Some(tr) = &m.tool_result {
            chars += tr.content.len();
        }
    }
    for t in &req.tools {
        chars += t.name.len() + t.description.len() + t.parameters.to_string().len();
    }
    chars / 4
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib route:: 2>&1 | tail -15`
Expected: PASS (9 tests total across Task 1 + Task 2).

- [ ] **Step 5: Commit**

```bash
git add src/route/mod.rs
git commit -m "feat(route): context-gate decision fn + token estimation"
```

---

### Task 3: Cloud reverse-proxy (`cloud::forward`)

**Files:**
- Create: `src/cloud.rs`
- Modify: `src/lib.rs` (add `pub mod cloud;`)
- Modify: `Cargo.toml` (add `wiremock` under `[dev-dependencies]`)
- Test: `src/cloud.rs` (`#[cfg(test)]`, async)

**Interfaces:**
- Produces:
  - `enum Provider { Anthropic, OpenAI }`
  - `impl Provider { fn base_url(&self) -> String; fn path(&self) -> &'static str }`
  - `async fn forward(provider: Provider, headers: &axum::http::HeaderMap, body: axum::body::Bytes) -> axum::response::Response`

- [ ] **Step 1: Add the dev-dependency and module**

In `Cargo.toml`, add a `[dev-dependencies]` section (create it if absent, after the main `[dependencies]` block):

```toml
[dev-dependencies]
wiremock = "0.6"
```

In `src/lib.rs`, add after `pub mod config;`:

```rust
pub mod cloud;
```

- [ ] **Step 2: Write the failing test**

Create `src/cloud.rs`:

```rust
//! Byte-faithful reverse-proxy to a cloud provider.
//!
//! Forwards the client's original request bytes and credential headers to the
//! upstream API and relays the response unchanged, so tool-calls and streaming
//! framing are preserved exactly. The credential is the client's own; we never
//! store or log it.

use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::response::Response;

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn forwards_body_and_relays_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(r#"{"ok":true}"#),
            )
            .mount(&server)
            .await;

        // Point the Anthropic base URL at the mock server.
        std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-test".parse().unwrap());
        let body = Bytes::from_static(br#"{"model":"claude","messages":[]}"#);

        let resp = forward(Provider::Anthropic, &headers, body).await;
        assert_eq!(resp.status(), 200);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], br#"{"ok":true}"#);

        std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
    }

    #[tokio::test]
    async fn upstream_unreachable_returns_502() {
        std::env::set_var("LOCALLLM_OPENAI_BASE", "http://127.0.0.1:1"); // nothing listening
        let resp = forward(Provider::OpenAI, &HeaderMap::new(), Bytes::new()).await;
        assert_eq!(resp.status(), 502);
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib cloud:: 2>&1 | head -30`
Expected: FAIL — `Provider`/`forward` not found.

- [ ] **Step 4: Write minimal implementation**

In `src/cloud.rs`, above the `#[cfg(test)]` block (after the `use` lines), add:

```rust
/// Cloud provider, inferred from the endpoint the client hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAI,
}

impl Provider {
    /// Upstream base URL. Overridable via env for tests.
    pub fn base_url(&self) -> String {
        match self {
            Provider::Anthropic => std::env::var("LOCALLLM_ANTHROPIC_BASE")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            Provider::OpenAI => std::env::var("LOCALLLM_OPENAI_BASE")
                .unwrap_or_else(|_| "https://api.openai.com".to_string()),
        }
    }

    /// Upstream request path for this provider's chat endpoint.
    pub fn path(&self) -> &'static str {
        match self {
            Provider::Anthropic => "/v1/messages",
            Provider::OpenAI => "/v1/chat/completions",
        }
    }
}

/// Hop-by-hop / connection headers that must not be forwarded verbatim;
/// reqwest sets correct values itself.
fn is_skipped_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host" | "content-length" | "connection" | "accept-encoding" | "transfer-encoding"
    )
}

/// Reverse-proxy `body` to `provider`, forwarding the client's headers
/// (including its credential) unchanged, and relay the streamed response.
/// On any transport error, return HTTP 502 (graceful local fallback arrives in
/// Phase D).
pub async fn forward(provider: Provider, headers: &HeaderMap, body: Bytes) -> Response {
    let url = format!("{}{}", provider.base_url(), provider.path());

    let mut fwd = HeaderMap::new();
    for (name, value) in headers.iter() {
        if !is_skipped_header(name.as_str()) {
            fwd.insert(name.clone(), value.clone());
        }
    }

    let client = reqwest::Client::new();
    let upstream = match client.post(&url).headers(fwd).body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "localllm::req", "cloud forward error: {e}");
            return Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"error": format!("cloud forward failed: {e}")})
                        .to_string(),
                ))
                .unwrap();
        }
    };

    let status = upstream.status();
    let ctype = upstream
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/json".parse().unwrap());

    let stream = upstream.bytes_stream();
    let mut builder = Response::builder().status(status);
    builder
        .headers_mut()
        .unwrap()
        .insert("content-type", ctype);
    builder
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib cloud:: 2>&1 | tail -15`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/cloud.rs
git commit -m "feat(cloud): byte-faithful reverse-proxy to provider upstream"
```

---

### Task 4: Wire policy + ctx window into `AppState` and `router`

**Files:**
- Modify: `src/server.rs:64-102` (`AppState`, `router`)
- Modify: `src/lib.rs:62` (`run_server_with_ready` call to `router`) and `src/lib.rs:120-122` (`router_for_test`)

**Interfaces:**
- Consumes: `RoutingPolicy` (Task 1).
- Produces: `AppState { gen, model_id, policy: std::sync::Arc<std::sync::RwLock<RoutingPolicy>>, local_ctx_window: usize }`; `router(gen, model_id, policy, local_ctx_window)`.

> No new behavior yet — this task only threads the state so Task 5 can read it. Tests still pass with the existing assertions.

- [ ] **Step 1: Extend `AppState` and `router` signature**

In `src/server.rs`, replace the `AppState` struct (lines 65-71) with:

```rust
/// Shared state injected into every handler via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    /// The inference backend (real Engine or FakeGen in tests).
    pub gen: Arc<dyn Generator>,
    /// The model identifier reported by `GET /v1/models` and used in responses.
    pub model_id: String,
    /// Active routing policy, shared with the tray (writer) — read per request.
    pub policy: Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    /// Local model's usable context window (config `ctx_len`), for the ctx gate.
    pub local_ctx_window: usize,
}
```

Replace the `router` function (lines 94-102) with:

```rust
/// Build the axum Router with all four endpoints wired to the given generator,
/// the configured model id, the shared routing policy, and the local context
/// window used by the context gate.
pub fn router(
    gen: Arc<dyn Generator>,
    model_id: String,
    policy: Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    local_ctx_window: usize,
) -> Router {
    let state = Arc::new(AppState {
        gen,
        model_id,
        policy,
        local_ctx_window,
    });
    Router::new()
        .route("/v1/chat/completions", post(handle_oai_chat))
        .route("/v1/messages", post(handle_anth_messages))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .with_state(state)
}
```

- [ ] **Step 2: Update the two call sites**

In `src/lib.rs`, replace line 62 (`let app = router(engine, cfg.model_id.clone());`) with:

```rust
    let policy = std::sync::Arc::new(std::sync::RwLock::new(
        crate::route::Profile::default().policy(),
    ));
    let app = router(engine, cfg.model_id.clone(), policy, cfg.ctx_len);
```

In `src/lib.rs`, replace `router_for_test` (lines 120-122) with:

```rust
/// Build a test router wired to `FakeGen`, a fixed model id, the default
/// (SaveTokens) policy, and a small context window so the ctx gate is testable.
pub fn router_for_test() -> Router {
    let policy = Arc::new(std::sync::RwLock::new(
        crate::route::Profile::default().policy(),
    ));
    crate::server::router(Arc::new(FakeGen), "test-model".to_string(), policy, 1000)
}
```

- [ ] **Step 3: Run the existing suite to verify nothing broke**

Run: `cargo test 2>&1 | tail -25`
Expected: PASS — all existing `tests/http.rs` tests and lib tests still green (handlers don't yet read the new state, so behavior is unchanged).

- [ ] **Step 4: Commit**

```bash
git add src/server.rs src/lib.rs
git commit -m "feat(server): thread routing policy + ctx window through AppState"
```

---

### Task 5: Route in handlers — raw-bytes extraction + cloud branch

**Files:**
- Modify: `src/server.rs:108-130` (`handle_oai_chat` signature + decision) and `src/server.rs:232-256` (`handle_anth_messages` signature + decision)
- Modify: `src/lib.rs` (add `axum_test_request_status_with_header` helper)
- Test: `tests/http.rs` (cloud-route + local-route integration tests)

**Interfaces:**
- Consumes: `route::{decide, estimate_prompt_tokens, Signals, Decision}`, `cloud::{forward, Provider}`, `AppState.policy`, `AppState.local_ctx_window`.
- Produces: handlers that parse from raw `Bytes`, decide, and either run the existing local path or reverse-proxy to cloud.

- [ ] **Step 1: Add a header-aware test helper**

In `src/lib.rs`, after `axum_test_request_status` (line 177), add:

```rust
/// POST with a JSON body AND a custom header; return the HTTP status code.
/// Used to exercise routing decisions that depend on a credential header.
pub async fn axum_test_request_status_with_header(
    app: Router,
    path: &str,
    body: &str,
    header_name: &str,
    header_value: &str,
) -> u16 {
    use axum::body::Body;
    use tower::ServiceExt;

    let request = axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header(header_name, header_value)
        .body(Body::from(body.to_owned()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    response.status().as_u16()
}
```

- [ ] **Step 2: Write the failing integration tests**

Append to `tests/http.rs`:

```rust
/// A request that fits the local window is served locally (FakeGen tool call),
/// even with a credential present.
#[tokio::test]
async fn small_request_with_key_stays_local() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}],
        "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}]}"#;
    let resp = localllm::axum_test_request(app, "/v1/messages", body).await;
    // FakeGen always returns a tool_use → proves the local path ran.
    assert_eq!(resp["stop_reason"], "tool_use");
}

/// A request that overflows the (tiny, 1000-token test) window with a credential
/// present is reverse-proxied to the mock upstream.
#[tokio::test]
async fn overflow_request_routes_to_cloud() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"routed":"cloud"}"#))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    let big = "x".repeat(8000); // ~2000 est. tokens > 0.95 * 1000
    let body = format!(
        r#"{{"model":"claude","max_tokens":256,"messages":[{{"role":"user","content":"{big}"}}]}}"#
    );
    let resp = localllm::axum_test_request_with_header(
        localllm::router_for_test(),
        "/v1/messages",
        &body,
        "x-api-key",
        "sk-test",
    )
    .await;
    assert_eq!(resp["routed"], "cloud");

    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// Overflow WITHOUT a credential must NOT hit cloud; it falls through to the
/// local path (which here returns FakeGen's tool_use rather than erroring,
/// because FakeGen ignores prompt size).
#[tokio::test]
async fn overflow_request_without_key_stays_local() {
    let big = "x".repeat(8000);
    let body = format!(
        r#"{{"model":"claude","max_tokens":256,"messages":[{{"role":"user","content":"{big}"}}]}}"#
    );
    let resp = localllm::axum_test_request(localllm::router_for_test(), "/v1/messages", &body).await;
    assert_eq!(resp["stop_reason"], "tool_use");
}
```

Add the JSON-returning header helper to `src/lib.rs` (after the status helper from Step 1):

```rust
/// POST with a JSON body AND a custom header; return the parsed JSON response.
pub async fn axum_test_request_with_header(
    app: Router,
    path: &str,
    body: &str,
    header_name: &str,
    header_value: &str,
) -> serde_json::Value {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let request = axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header(header_name, header_value)
        .body(Body::from(body.to_owned()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --test http overflow_request_routes_to_cloud 2>&1 | tail -20`
Expected: FAIL — `axum_test_request_with_header` may compile, but the handler still ignores routing so the overflow test gets FakeGen's `tool_use` instead of `{"routed":"cloud"}`.

- [ ] **Step 4: Add a shared routing helper in `server.rs`**

In `src/server.rs`, after `request_summary` (line 40), add:

```rust
/// Compute the routing decision for an already-parsed internal request.
/// Returns the decision plus whether a forwardable credential header is present.
fn route_decision(
    state: &AppState,
    internal: &ChatRequest,
    headers: &axum::http::HeaderMap,
) -> crate::route::Decision {
    let has_cloud_creds =
        headers.contains_key("x-api-key") || headers.contains_key("authorization");
    let signals = crate::route::Signals {
        prompt_tokens: crate::route::estimate_prompt_tokens(internal),
        local_ctx_window: state.local_ctx_window,
        n_tools: internal.tools.len(),
        n_messages: internal.messages.len(),
        has_cloud_creds,
    };
    let policy = *state.policy.read().unwrap();
    crate::route::decide(&signals, &policy)
}
```

Add the import at the top of `src/server.rs` (near the other `use crate::api::common::` line, ~line 26):

```rust
use axum::body::Bytes;
use axum::http::HeaderMap;
```

(Keep the existing `use crate::api::common::{ChatRequest, ChatResult, StreamDelta};` — `ChatRequest` is now also referenced by `route_decision`.)

- [ ] **Step 5: Convert `handle_anth_messages` to raw extraction + cloud branch**

In `src/server.rs`, replace the signature and parse block of `handle_anth_messages` (lines 232-252) with:

```rust
async fn handle_anth_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;

    let rid = new_request_id();

    let req: AnthRequest = match serde_json::from_slice(&raw) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] 400 bad json: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response();
        }
    };
    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::anthropic::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] 400 bad request: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response();
        }
    };

    // --- Routing decision (Phase A: context-size gate only) ---
    match route_decision(&state, &internal, &headers) {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [anthropic] route=cloud reason={reason:?}");
            return crate::cloud::forward(crate::cloud::Provider::Anthropic, &headers, raw).await;
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] route=local (cloud wanted but no creds/disallowed)");
        }
        _ => {}
    }
```

This replaces only the header of the function up to and including the old
`let internal = ... ;` block. The remainder of the function (from
`let (n_msgs, n_tools) = request_summary(&internal);` onward) is unchanged.

- [ ] **Step 6: Convert `handle_oai_chat` the same way**

In `src/server.rs`, replace the signature and parse block of `handle_oai_chat` (lines 109-130) with:

```rust
async fn handle_oai_chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;
    use uuid::Uuid;

    let rid = new_request_id();

    let req: OaiChatRequest = match serde_json::from_slice(&raw) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] 400 bad json: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response();
        }
    };
    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::openai::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] 400 bad request: {e}");
            return (StatusCode::BAD_REQUEST, Json(json!({"error": e}))).into_response();
        }
    };

    // --- Routing decision (Phase A: context-size gate only) ---
    match route_decision(&state, &internal, &headers) {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [openai] route=cloud reason={reason:?}");
            return crate::cloud::forward(crate::cloud::Provider::OpenAI, &headers, raw).await;
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] route=local (cloud wanted but no creds/disallowed)");
        }
        _ => {}
    }
```

The remainder (from `let (n_msgs, n_tools) = request_summary(&internal);` onward) is unchanged. Note: `Uuid` is still used later in the function, so keep its `use`.

- [ ] **Step 7: Run the routing tests to verify they pass**

Run: `cargo test --test http 2>&1 | tail -25`
Expected: PASS — including `small_request_with_key_stays_local`, `overflow_request_routes_to_cloud`, `overflow_request_without_key_stays_local`, and all pre-existing handler tests (the old `axum_test_request*` helpers send no credential header, so they stay local and are unaffected).

- [ ] **Step 8: Full build + suite + clippy**

Run: `cargo test 2>&1 | tail -15 && cargo clippy --all-targets 2>&1 | tail -15`
Expected: all tests PASS; no new clippy errors.

- [ ] **Step 9: Commit**

```bash
git add src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): route over-window requests to cloud reverse-proxy"
```

---

## Phase A Acceptance

- An over-window request **with** a credential is reverse-proxied to the provider; the response is relayed with the provider's native shape (status, content-type, body bytes) — verified against a wiremock upstream.
- An over-window request **without** a credential, and any in-window request, is served locally; existing tool-call + streaming tests are unchanged.
- The routing decision logic is fully unit-tested in `src/route/` with no HTTP or model dependency.
- Provider is selected by endpoint; credentials are forwarded verbatim and never logged.
- `cargo test` and `cargo clippy --all-targets` are green.

## Self-Review

- **Spec coverage (Phase A scope):** ctx gate (Task 2), reverse-proxy with reused creds (Task 3), policy plumbing (Task 1, 4), endpoint→provider + raw extraction (Task 5). Difficulty score, cascade, tray, persistence, usage/alerts are explicitly later phases — not in this plan.
- **Placeholder scan:** none — every step shows complete code or an exact command.
- **Type consistency:** `RoutingPolicy`, `Signals`, `Decision`, `RouteReason`, `Provider`, `forward`, `decide`, `estimate_prompt_tokens`, `route_decision`, and the `router(gen, model_id, policy, local_ctx_window)` signature match across Tasks 1–5 and both call sites.
