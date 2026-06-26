# Phase D — Usage Tracking + Graceful Degrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a cloud route fails (auth/quota/5xx/offline) fall back to the local model where possible and notify the user once; track cloud usage per session and alert once when it crosses a threshold.

**Architecture:** `cloud::forward` returns a `ForwardOutcome` (`Relayed(Response)` or `Degrade(DegradeReason)`). A `usage` module holds atomic session counters plus one-shot notification gates and a runtime-gated macOS notifier (off in tests, enabled only by the real server). Handlers: on an upfront cloud route, a degrade serves local instead (when the prompt fits the window; else a clean error) and records/alerts usage on success; on a cascade escalation, a cloud failure returns the already-computed local result.

**Tech Stack:** Rust, axum 0.7, reqwest 0.12, tokio, std atomics; macOS notifications via `osascript`. Dev: wiremock.

## Global Constraints

- Builds on Phases A–C (branch `feat/model-router`): `cloud::{Provider, forward}`, `route::{decide, RouteReason, is_weak_result}`, `server::{route_decision, cascade_or_result, AppState, router}`, `lib::run_server_with_ready_and_policy`, `settings`, tray.
- **Degrade triggers** (all → fall back to local when possible + one-shot notify): upstream `401/403` (Auth), `429` (Quota), `5xx` (ServerError), and any transport/network error (Offline).
- **Degrade-to-local rule:** if the prompt fits the local window, serve local; if the cloud route was taken because of context overflow (cannot fit), return a clean error instead. A failed **cascade** escalation returns the already-computed local result.
- **Notifications are one-shot:** a degrade notifies once and re-arms only after a later cloud success; the high-usage alert fires once per session. Notifications are a no-op unless the real server enabled them (so `cargo test` never triggers `osascript`).
- Credentials are never logged or stored (unchanged from Phase A).
- `bytes::Bytes::clone` is O(1) (shared buffer) — cloning `raw` for the cloud call to keep the original for a local fallback is cheap and intended.
- Default high-usage threshold: **200_000** cloud prompt tokens/session, overridable via `--cloud-token-alert`.
- TDD for testable units (usage atomics, forward classification, degrade wiring via wiremock). The `osascript` side-effect itself is not unit-tested. Commit per task when green; pristine output.

---

### Task 1: `usage` module — counters, one-shot gates, gated notifier

**Files:**
- Create: `src/usage.rs`
- Modify: `src/lib.rs` (`pub mod usage;`)
- Test: `src/usage.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces:
  - `usage::Usage` with `new()`, `record_cloud_call(&self, prompt_tokens: usize, threshold: usize) -> bool` (returns `true` exactly once, when the cumulative cloud prompt tokens first reach/exceed `threshold`), `note_degrade(&self) -> bool` (returns `true` only on transition into the degraded state — re-arms after `note_success`), `note_success(&self)` (clears the degraded gate), `totals(&self) -> (u64, u64)` (calls, prompt_tokens).
  - `usage::enable_notifications()` — flips the process-global notifier on (called by the real server).
  - `usage::notify(title: &str, body: &str)` — shows a macOS notification when enabled; no-op otherwise.

- [ ] **Step 1: Register the module and write failing tests**

In `src/lib.rs`, add after `pub mod settings;`:

```rust
pub mod usage;
```

Create `src/usage.rs`:

```rust
//! Per-session cloud usage counters and one-shot user notifications.
//!
//! Counters are atomic so the shared `Usage` can be read/updated from any
//! request without a lock. Notifications are one-shot (degrade re-arms only
//! after a later success) and are a no-op unless the real server has called
//! [`enable_notifications`], so test runs never spawn `osascript`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Why a cloud request degraded to local. Carried into the user notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeReason {
    Auth,
    Quota,
    ServerError,
    Offline,
}

impl DegradeReason {
    /// Short user-facing description for the notification body.
    pub fn message(&self) -> &'static str {
        match self {
            DegradeReason::Auth => "Cloud auth failed — serving locally.",
            DegradeReason::Quota => "Cloud quota exceeded — serving locally.",
            DegradeReason::ServerError => "Cloud error — serving locally.",
            DegradeReason::Offline => "No internet — serving locally.",
        }
    }
}

/// Atomic per-session cloud usage + one-shot notification gates.
#[derive(Debug, Default)]
pub struct Usage {
    calls: AtomicU64,
    prompt_tokens: AtomicU64,
    /// True while in a degraded run (cloud failing); gates one-shot degrade alert.
    degraded: AtomicBool,
    /// True once the high-usage alert has fired this session.
    high_alerted: AtomicBool,
}

impl Usage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one successful cloud call of `prompt_tokens`. Returns `true` exactly
    /// once — the call that first brings cumulative prompt tokens to/over
    /// `threshold` — so the caller fires the high-usage alert a single time.
    pub fn record_cloud_call(&self, prompt_tokens: usize, threshold: usize) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let total = self.prompt_tokens.fetch_add(prompt_tokens as u64, Ordering::SeqCst)
            + prompt_tokens as u64;
        if total >= threshold as u64
            && self
                .high_alerted
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            return true;
        }
        false
    }

    /// Mark a degrade. Returns `true` only on the transition into the degraded
    /// state, so the degrade notification fires once until a success re-arms it.
    pub fn note_degrade(&self) -> bool {
        self.degraded
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Clear the degrade gate after a successful cloud call so a later failure
    /// notifies again.
    pub fn note_success(&self) {
        self.degraded.store(false, Ordering::SeqCst);
    }

    /// (calls, prompt_tokens) so far this session.
    pub fn totals(&self) -> (u64, u64) {
        (
            self.calls.load(Ordering::SeqCst),
            self.prompt_tokens.load(Ordering::SeqCst),
        )
    }
}

static NOTIFY_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable real user notifications. Called once by the running server; left off in
/// tests so `osascript` never spawns.
pub fn enable_notifications() {
    NOTIFY_ENABLED.store(true, Ordering::SeqCst);
}

/// Show a desktop notification when enabled; otherwise a no-op (also logs).
pub fn notify(title: &str, body: &str) {
    tracing::info!(target: "localllm::req", "notify: {title} — {body}");
    if !NOTIFY_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        // Best-effort; ignore failure. Quote-escape to avoid breaking the script.
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            body.replace('"', "'"),
            title.replace('"', "'"),
        );
        let _ = std::process::Command::new("osascript")
            .arg("-e")
            .arg(script)
            .spawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_usage_alert_fires_exactly_once_on_crossing() {
        let u = Usage::new();
        // threshold 100; first call 60 → under, no alert
        assert!(!u.record_cloud_call(60, 100));
        // second call 60 → total 120 ≥ 100 → alert once
        assert!(u.record_cloud_call(60, 100));
        // further calls never re-alert
        assert!(!u.record_cloud_call(60, 100));
        let (calls, toks) = u.totals();
        assert_eq!(calls, 3);
        assert_eq!(toks, 180);
    }

    #[test]
    fn degrade_is_one_shot_until_success_rearms() {
        let u = Usage::new();
        assert!(u.note_degrade()); // first degrade → notify
        assert!(!u.note_degrade()); // still degraded → no repeat
        u.note_success(); // cloud recovered
        assert!(u.note_degrade()); // degrade again → notify again
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib usage:: 2>&1 | head -20`
Expected: FAIL — `usage` module/items not found.

- [ ] **Step 3: (code above is the implementation) Run to verify they pass**

The module body in Step 1 is the implementation. Run:
Run: `cargo test --lib usage:: 2>&1 | tail -12`
Expected: PASS (2 tests).

- [ ] **Step 4: Full suite + clippy**

Run: `cargo test 2>&1 | tail -8 && cargo clippy --lib 2>&1 | grep -E "src/usage.rs" | grep -- "-->" || echo "no clippy in usage.rs"`
Expected: full suite green; no clippy in `usage.rs`.

- [ ] **Step 5: Commit**

```bash
git add src/usage.rs src/lib.rs
git commit -m "feat(usage): session cloud counters + one-shot gates + gated notifier"
```

---

### Task 2: `cloud::forward` returns `ForwardOutcome` + degrade classification

**Files:**
- Modify: `src/cloud.rs` (return type, classification, `degrade_error`)
- Modify: `src/server.rs` (update the call sites to keep compiling, behavior preserved)
- Test: `src/cloud.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `usage::DegradeReason`.
- Produces:
  - `cloud::ForwardOutcome { Relayed(axum::response::Response), Degrade(crate::usage::DegradeReason) }`
  - `cloud::forward(provider, headers, body) -> ForwardOutcome` (signature return type changed)
  - `cloud::degrade_error(reason: crate::usage::DegradeReason) -> axum::response::Response` (502 JSON, no creds)

- [ ] **Step 1: Write the failing classification tests**

In `src/cloud.rs`, REPLACE the existing `upstream_unreachable_returns_502` test and ADD classification tests, so the test module asserts the new outcome type:

```rust
    #[tokio::test]
    async fn unreachable_upstream_degrades_offline() {
        std::env::set_var("LOCALLLM_OPENAI_BASE", "http://127.0.0.1:1"); // nothing listening
        let outcome = forward(Provider::OpenAI, &HeaderMap::new(), Bytes::new()).await;
        assert!(matches!(
            outcome,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Offline)
        ));
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
    }

    async fn outcome_for_status(status: u16) -> ForwardOutcome {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;
        std::env::set_var("LOCALLLM_OPENAI_BASE", server.uri());
        let out = forward(Provider::OpenAI, &HeaderMap::new(), Bytes::new()).await;
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
        out
    }

    #[tokio::test]
    async fn status_401_degrades_auth() {
        assert!(matches!(
            outcome_for_status(401).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Auth)
        ));
    }

    #[tokio::test]
    async fn status_429_degrades_quota() {
        assert!(matches!(
            outcome_for_status(429).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Quota)
        ));
    }

    #[tokio::test]
    async fn status_500_degrades_server_error() {
        assert!(matches!(
            outcome_for_status(500).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::ServerError)
        ));
    }
```

Also update the existing `forwards_body_and_relays_response` test to unwrap the new type — replace its `let resp = forward(...).await; assert_eq!(resp.status(), 200);` block with:

```rust
        let outcome = forward(Provider::Anthropic, &headers, body).await;
        let resp = match outcome {
            ForwardOutcome::Relayed(r) => r,
            ForwardOutcome::Degrade(d) => panic!("expected relay, got degrade {d:?}"),
        };
        assert_eq!(resp.status(), 200);
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib cloud:: 2>&1 | head -20`
Expected: FAIL — `ForwardOutcome` undefined; `forward` still returns `Response`.

- [ ] **Step 3: Change `forward` to return `ForwardOutcome` and classify**

In `src/cloud.rs`, add the outcome type (after the `Provider` impl):

```rust
/// Result of a reverse-proxy attempt.
pub enum ForwardOutcome {
    /// Upstream produced a client-relevant response (relay it verbatim).
    Relayed(Response),
    /// Upstream failed in a way that should degrade to local + notify.
    Degrade(crate::usage::DegradeReason),
}

/// A clean 502 to return to the client when degrade-to-local is impossible
/// (e.g. the prompt overflows the local window). Carries no credential/internal
/// detail beyond the reason category.
pub fn degrade_error(reason: crate::usage::DegradeReason) -> Response {
    Response::builder()
        .status(502)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::json!({"error": reason.message()}).to_string(),
        ))
        .unwrap()
}
```

Then REPLACE the `forward` function body so it returns `ForwardOutcome` and classifies failures:

```rust
/// Reverse-proxy `body` to `provider`, forwarding the client's headers
/// (including its credential) unchanged. Returns `Relayed` with the streamed
/// response on success, or `Degrade(reason)` on a failure the caller should
/// handle by falling back to local. Credentials are never logged.
pub async fn forward(provider: Provider, headers: &HeaderMap, body: Bytes) -> ForwardOutcome {
    use crate::usage::DegradeReason;
    let url = format!("{}{}", provider.base_url(), provider.path());

    let mut fwd = HeaderMap::new();
    for (name, value) in headers.iter() {
        if !is_skipped_header(name.as_str()) {
            fwd.insert(name.clone(), value.clone());
        }
    }

    let upstream = match shared_client().post(&url).headers(fwd).body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "localllm::req", "cloud forward transport error: {e}");
            return ForwardOutcome::Degrade(DegradeReason::Offline);
        }
    };

    let status = upstream.status();
    if let Some(reason) = match status.as_u16() {
        401 | 403 => Some(DegradeReason::Auth),
        429 => Some(DegradeReason::Quota),
        s if s >= 500 => Some(DegradeReason::ServerError),
        _ => None,
    } {
        tracing::warn!(target: "localllm::req", "cloud upstream status {status} → degrade {reason:?}");
        return ForwardOutcome::Degrade(reason);
    }

    let mut resp_headers = HeaderMap::new();
    for (name, value) in upstream.headers().iter() {
        if !is_skipped_header(name.as_str()) {
            resp_headers.insert(name.clone(), value.clone());
        }
    }
    let stream = upstream.bytes_stream();
    let mut builder = Response::builder().status(status);
    *builder.headers_mut().unwrap() = resp_headers;
    ForwardOutcome::Relayed(builder.body(axum::body::Body::from_stream(stream)).unwrap())
}
```

- [ ] **Step 4: Keep the handler call sites compiling (behavior preserved)**

The four `forward(...)` call sites in `src/server.rs` now receive a `ForwardOutcome`. For THIS task, preserve current behavior (no degrade-to-local yet): on `Degrade`, return the clean error. Task 3 replaces these with full degrade handling.

In each handler's decision match `Cloud(reason)` arm, the current line is:

```rust
            return crate::cloud::forward(crate::cloud::Provider::OpenAI, &headers, raw).await;
```

REPLACE each (OpenAI in `handle_oai_chat`, Anthropic in `handle_anth_messages`) with:

```rust
            return match crate::cloud::forward(crate::cloud::Provider::OpenAI, &headers, raw).await {
                crate::cloud::ForwardOutcome::Relayed(resp) => resp,
                crate::cloud::ForwardOutcome::Degrade(d) => crate::cloud::degrade_error(d),
            };
```

(Use `Provider::Anthropic` in the anthropic handler.)

In `cascade_or_result`, the two `Err(crate::cloud::forward(...).await)` lines become:

```rust
                match crate::cloud::forward(provider, headers, raw).await {
                    crate::cloud::ForwardOutcome::Relayed(resp) => Err(resp),
                    crate::cloud::ForwardOutcome::Degrade(d) => Err(crate::cloud::degrade_error(d)),
                }
```

> Note: `cascade_or_result` returns `Result<ChatResult, Response>`; both arms above produce `Err(Response)`, so the surrounding `match gen_result` arms still type-check. The `raw: Bytes` is consumed by `forward` exactly once per call as before.

- [ ] **Step 5: Run cloud tests + full suite**

Run: `cargo test --lib cloud:: 2>&1 | tail -15`
Expected: PASS — relay test + four degrade-classification tests.

Run: `cargo test 2>&1 | tail -10`
Expected: full suite green; the existing handler/cascade integration tests still pass (Cloud routes that succeed relay; the overflow-to-cloud test still gets a relayed 200 from its mock).

- [ ] **Step 6: Commit**

```bash
git add src/cloud.rs src/server.rs
git commit -m "feat(cloud): forward returns ForwardOutcome (relay vs classified degrade)"
```

---

### Task 3: Wire degrade-to-local + usage/alerts into handlers

**Files:**
- Modify: `src/server.rs` (`AppState`, `router`, `route_decision` to expose the estimate, both handler Cloud arms, `cascade_or_result`)
- Modify: `src/config.rs` (`--cloud-token-alert`)
- Modify: `src/lib.rs` (build `Usage`, pass to `router`, `enable_notifications()`, `router_for_test_with` defaults)
- Test: `tests/http.rs` (degrade-to-local + cascade-degrade integration tests)

**Interfaces:**
- Consumes: `usage::{Usage, DegradeReason, notify, enable_notifications}`, `cloud::{forward, ForwardOutcome, degrade_error}`, `route::{RouteReason, estimate_prompt_tokens}`.
- Produces:
  - `AppState { …, usage: std::sync::Arc<crate::usage::Usage>, cloud_token_alert: usize }`
  - `router(gen, model_id, policy, local_ctx_window, usage, cloud_token_alert)`
  - `Config.cloud_token_alert: usize` (`--cloud-token-alert`, default 200_000)
  - updated `cascade_or_result(..., usage: &crate::usage::Usage, est_prompt_tokens: usize, cloud_token_alert: usize, ...)`

- [ ] **Step 1: Add the CLI flag with a parse test**

In `src/config.rs`, add to `Config` (after `profile`):

```rust
    /// Session cloud prompt-token total at which a one-shot "high usage" alert
    /// fires (suggesting the Save-tokens profile). Default 200_000.
    #[arg(long, default_value_t = 200_000)]
    pub cloud_token_alert: usize,
```

Add a test to `src/config.rs` tests:

```rust
    #[test]
    fn cloud_token_alert_defaults_and_parses() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.cloud_token_alert, 200_000);
        let c = Config::parse_from(["localllm", "--cloud-token-alert", "50000"]);
        assert_eq!(c.cloud_token_alert, 50_000);
    }
```

- [ ] **Step 2: Thread `usage` + `cloud_token_alert` through `AppState`/`router`/lib + enable notifications**

In `src/server.rs`, add to `AppState`:

```rust
    /// Per-session cloud usage counters + notification gates.
    pub usage: std::sync::Arc<crate::usage::Usage>,
    /// Session cloud-token total that triggers the one-shot high-usage alert.
    pub cloud_token_alert: usize,
```

Extend `router` signature and the `AppState { … }` it builds:

```rust
pub fn router(
    gen: Arc<dyn Generator>,
    model_id: String,
    policy: Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    local_ctx_window: usize,
    usage: Arc<crate::usage::Usage>,
    cloud_token_alert: usize,
) -> Router {
    let state = Arc::new(AppState {
        gen,
        model_id,
        policy,
        local_ctx_window,
        usage,
        cloud_token_alert,
    });
    // … routes unchanged …
```

In `src/lib.rs`, in `run_server_with_ready_and_policy`, before building the router add:

```rust
    crate::usage::enable_notifications();
    let usage = std::sync::Arc::new(crate::usage::Usage::new());
```

and change the `router(engine, cfg.model_id.clone(), policy, cfg.ctx_len)` call to:

```rust
    let app = router(engine, cfg.model_id.clone(), policy, cfg.ctx_len, usage, cfg.cloud_token_alert);
```

In `src/lib.rs`, update `router_for_test_with` to construct a fresh `Usage` and a default alert threshold so tests never enable notifications:

```rust
pub fn router_for_test_with(
    gen: Arc<dyn Generator>,
    policy: crate::route::RoutingPolicy,
    local_ctx_window: usize,
) -> Router {
    let policy = Arc::new(std::sync::RwLock::new(policy));
    let usage = Arc::new(crate::usage::Usage::new());
    crate::server::router(gen, "test-model".to_string(), policy, local_ctx_window, usage, 200_000)
}
```

- [ ] **Step 3: Expose the prompt-token estimate from `route_decision`**

In `src/server.rs`, change `route_decision` to also return the estimate so the handler can record usage without recomputing. Replace its signature/return:

```rust
/// Returns the routing decision plus the estimated prompt token count (so the
/// caller can record cloud usage without recomputing).
fn route_decision(
    state: &AppState,
    internal: &ChatRequest,
    headers: &axum::http::HeaderMap,
) -> (crate::route::Decision, usize) {
    let has_cloud_creds =
        headers.contains_key("x-api-key") || headers.contains_key("authorization");
    let prompt_tokens = crate::route::estimate_prompt_tokens(internal);
    let signals = crate::route::Signals {
        prompt_tokens,
        local_ctx_window: state.local_ctx_window,
        n_tools: internal.tools.len(),
        n_messages: internal.messages.len(),
        has_cloud_creds,
    };
    let policy = *state.policy.read().unwrap();
    (crate::route::decide(&signals, &policy), prompt_tokens)
}
```

- [ ] **Step 4: Add a usage-recording helper and degrade handling in `handle_anth_messages`**

In `src/server.rs`, add a helper near `cascade_or_result`:

```rust
/// Record a successful cloud call and fire the one-shot high-usage alert if the
/// session just crossed the threshold. Also clears the degrade gate so a later
/// failure notifies again.
fn record_cloud_success(state: &AppState, est_prompt_tokens: usize) {
    state.usage.note_success();
    if state
        .usage
        .record_cloud_call(est_prompt_tokens, state.cloud_token_alert)
    {
        crate::usage::notify(
            "localllm — high cloud usage",
            "High cloud token use this session — consider the Save tokens profile.",
        );
    }
}

/// Handle a degrade signal: notify once, then decide whether local can serve.
/// Returns `Some(response)` when the caller must return it (cannot fall back to
/// local because the prompt overflows the window); `None` when the caller should
/// proceed to the local path.
fn handle_degrade(
    state: &AppState,
    reason: crate::usage::DegradeReason,
    route_reason: crate::route::RouteReason,
) -> Option<axum::response::Response> {
    if state.usage.note_degrade() {
        crate::usage::notify("localllm — cloud degraded", reason.message());
    }
    if route_reason == crate::route::RouteReason::ContextOverflow {
        // Prompt cannot fit the local window → no local fallback.
        Some(crate::cloud::degrade_error(reason))
    } else {
        None
    }
}
```

Then in `handle_anth_messages`, REPLACE the decision match (currently capturing `want_cascade`) with one that records the estimate and handles degrade-to-local:

```rust
    let (decision, est_prompt_tokens) = route_decision(&state, &internal, &headers);
    let want_cascade = match decision {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [anthropic] route=cloud reason={reason:?}");
            match crate::cloud::forward(crate::cloud::Provider::Anthropic, &headers, raw.clone()).await {
                crate::cloud::ForwardOutcome::Relayed(resp) => {
                    record_cloud_success(&state, est_prompt_tokens);
                    return resp;
                }
                crate::cloud::ForwardOutcome::Degrade(d) => {
                    if let Some(resp) = handle_degrade(&state, d, reason) {
                        return resp;
                    }
                    tracing::warn!(target: "localllm::req", "{rid} [anthropic] cloud degraded ({d:?}) → serving local");
                    false
                }
            }
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [anthropic] route=local (cloud wanted but no creds/disallowed)");
            false
        }
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };
```

> `raw.clone()` is an O(1) `Bytes` refcount bump; the original `raw` stays owned so the local-path branches (and `cascade_or_result`) can still use it.

- [ ] **Step 5: Apply the identical change to `handle_oai_chat`**

In `src/server.rs`, make the same replacement in `handle_oai_chat`, using `Provider::OpenAI` and the `[openai]` log tag:

```rust
    let (decision, est_prompt_tokens) = route_decision(&state, &internal, &headers);
    let want_cascade = match decision {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [openai] route=cloud reason={reason:?}");
            match crate::cloud::forward(crate::cloud::Provider::OpenAI, &headers, raw.clone()).await {
                crate::cloud::ForwardOutcome::Relayed(resp) => {
                    record_cloud_success(&state, est_prompt_tokens);
                    return resp;
                }
                crate::cloud::ForwardOutcome::Degrade(d) => {
                    if let Some(resp) = handle_degrade(&state, d, reason) {
                        return resp;
                    }
                    tracing::warn!(target: "localllm::req", "{rid} [openai] cloud degraded ({d:?}) → serving local");
                    false
                }
            }
        }
        crate::route::Decision::LocalNoCreds => {
            tracing::warn!(target: "localllm::req", "{rid} [openai] route=local (cloud wanted but no creds/disallowed)");
            false
        }
        crate::route::Decision::LocalThenCascade => true,
        crate::route::Decision::Local => false,
    };
```

- [ ] **Step 6: Make `cascade_or_result` degrade to the local result + record success**

In `src/server.rs`, REPLACE `cascade_or_result` so a cascade escalation records success on relay and returns the already-computed local result on degrade:

```rust
async fn cascade_or_result(
    want_cascade: bool,
    gen_result: anyhow::Result<ChatResult>,
    provider: crate::cloud::Provider,
    headers: &HeaderMap,
    raw: Bytes,
    state: &AppState,
    est_prompt_tokens: usize,
    rid: &str,
    api: &str,
) -> Result<ChatResult, axum::response::Response> {
    use axum::response::IntoResponse;
    match gen_result {
        Ok(result) => {
            if want_cascade && crate::route::is_weak_result(&result) {
                tracing::info!(target: "localllm::req", "{rid} [{api}] cascade: weak local (length) → escalating to cloud");
                match crate::cloud::forward(provider, headers, raw).await {
                    crate::cloud::ForwardOutcome::Relayed(resp) => {
                        record_cloud_success(state, est_prompt_tokens);
                        Err(resp)
                    }
                    crate::cloud::ForwardOutcome::Degrade(d) => {
                        if state.usage.note_degrade() {
                            crate::usage::notify("localllm — cloud degraded", d.message());
                        }
                        tracing::warn!(target: "localllm::req", "{rid} [{api}] cascade cloud degraded ({d:?}) → keeping local result");
                        Ok(result) // serve the (truncated) local answer we already have
                    }
                }
            } else {
                Ok(result)
            }
        }
        Err(e) => {
            if want_cascade {
                tracing::warn!(target: "localllm::req", "{rid} [{api}] cascade: local generate failed ({e}) → escalating to cloud");
                match crate::cloud::forward(provider, headers, raw).await {
                    crate::cloud::ForwardOutcome::Relayed(resp) => {
                        record_cloud_success(state, est_prompt_tokens);
                        Err(resp)
                    }
                    crate::cloud::ForwardOutcome::Degrade(d) => {
                        if state.usage.note_degrade() {
                            crate::usage::notify("localllm — cloud degraded", d.message());
                        }
                        Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("local generation failed and cloud degraded ({d:?})")}))).into_response())
                    }
                }
            } else {
                tracing::error!(target: "localllm::req", "{rid} [{api}] 500 generate: {e}");
                Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response())
            }
        }
    }
}
```

Update the FOUR `cascade_or_result(...)` call sites (two per handler) to pass `&state` and `est_prompt_tokens` instead of the old `&headers, raw, &rid, api` ordering. Each call becomes:

```rust
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::Anthropic, // or OpenAI in handle_oai_chat
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "anthropic", // or "openai"
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
```

- [ ] **Step 7: Write the degrade integration tests**

Append to `tests/http.rs`:

```rust
/// An in-window difficulty/cloud route whose upstream returns 429 degrades to the
/// local model instead of erroring (FakeGen tool_use proves local ran).
#[tokio::test]
async fn cloud_quota_degrades_to_local() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    // MaxQuality + a modest in-window prompt → Cloud(Difficulty); 429 → degrade local.
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"a fairly involved question about routing"}],
        "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGen),
        localllm::route::Profile::MaxQuality.policy(),
        1000,
    );
    let resp = localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test").await;
    assert_eq!(resp["stop_reason"], "tool_use"); // FakeGen → local served

    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// An over-window (context-overflow) cloud route whose upstream is unreachable
/// cannot fall back to local; it returns a clean 502.
#[tokio::test]
async fn overflow_cloud_offline_returns_clean_error() {
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", "http://127.0.0.1:1");
    let big = "x".repeat(8000); // > 0.95 * 1000 window → ContextOverflow
    let body = format!(
        r#"{{"model":"claude","max_tokens":256,"messages":[{{"role":"user","content":"{big}"}}]}}"#
    );
    let status = localllm::axum_test_request_status_with_header(
        localllm::router_for_test_with(
            std::sync::Arc::new(localllm::FakeGen),
            localllm::route::Profile::SaveTokens.policy(),
            1000,
        ),
        "/v1/messages",
        &body,
        "x-api-key",
        "sk-test",
    )
    .await;
    assert_eq!(status, 502);
    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// A weak local cascade whose cloud escalation fails (offline) keeps the local
/// (truncated) answer instead of erroring.
#[tokio::test]
async fn cascade_cloud_offline_keeps_local_answer() {
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", "http://127.0.0.1:1");
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGenWeak),
        localllm::route::Profile::SaveTokens.policy(),
        1000,
    );
    let resp = localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test").await;
    assert_eq!(resp["content"][0]["text"], "local-weak-answer"); // local kept
    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}
```

- [ ] **Step 8: Run the new tests, full suite, build, clippy**

Run: `cargo test --test http 2>&1 | tail -30`
Expected: PASS — `cloud_quota_degrades_to_local`, `overflow_cloud_offline_returns_clean_error`, `cascade_cloud_offline_keeps_local_answer`, plus all pre-existing tests (the earlier `overflow_request_routes_to_cloud` still relays a 200 from its mock).

Run: `cargo build 2>&1 | tail -8 && cargo test 2>&1 | tail -8`
Expected: clean build; full suite green.

Run: `cargo clippy --all-targets 2>&1 | grep -E "src/usage.rs|src/cloud.rs|src/server.rs|src/config.rs" | grep -- "-->" || echo "no clippy in changed files"`
Expected: no clippy warnings in changed files (pre-existing `tray.rs:30` and the SSE `to_string` lines are out of scope).

- [ ] **Step 9: Commit**

```bash
git add src/server.rs src/config.rs src/lib.rs tests/http.rs
git commit -m "feat(server): degrade cloud failures to local + session usage alerts"
```

---

## Phase D Acceptance

- A cloud route that fails with 401/403/429/5xx/offline degrades to the local model when the prompt fits the window, and returns a clean 502 when it cannot (context overflow); the user is notified once per degrade episode.
- A failed cascade escalation returns the already-computed local result instead of an error.
- Cloud usage (calls + estimated prompt tokens) is tracked per session; a one-shot alert fires when the session crosses `--cloud-token-alert` (default 200_000).
- Notifications never fire during `cargo test` (gated off unless the real server enabled them).
- `cargo test`, `cargo build`, and `cargo clippy --all-targets` are clean (no new warnings in changed files).

## Self-Review

- **Spec coverage (Phase D scope):** three degrade triggers + classification (Task 2), degrade-to-local with the overflow exception (Task 3 `handle_degrade`), cascade keeps local on degrade (Task 3 `cascade_or_result`), session usage counters (Task 1), high-usage one-shot alert (Task 1 + Task 3 `record_cloud_success`), one-shot degrade notify re-armed on success (Task 1 + handlers), gated notifier so tests stay silent (Task 1).
- **Placeholder scan:** none — every step shows complete code or an exact command.
- **Type consistency:** `ForwardOutcome`/`DegradeReason`, `forward(...) -> ForwardOutcome`, `degrade_error(DegradeReason)`, `Usage::{record_cloud_call, note_degrade, note_success, totals}`, `notify`/`enable_notifications`, `route_decision -> (Decision, usize)`, `router(... usage, cloud_token_alert)`, `cascade_or_result(... &AppState, est_prompt_tokens ...)` are consistent across tasks. `Bytes::clone` used intentionally (O(1)).
- **Carry-over fixed:** the Phase-B minor "cascade discards a usable local answer on cloud failure" is resolved by Task 3's `cascade_or_result` degrade arm.
- **Known deferrals still open after Phase D:** atomic settings write; buffered-stream cascade test; `RouteReason::Profile` wire-or-drop; per-token cloud usage (we track estimated prompt tokens only, not parsed completion tokens — sufficient for the alert heuristic).
