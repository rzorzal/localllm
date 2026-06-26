//! HTTP router and handlers.
//!
//! Exposes four endpoints:
//!   POST /v1/chat/completions  — OpenAI Chat Completions API (streaming + non-streaming)
//!   POST /v1/messages          — Anthropic Messages API (streaming + non-streaming)
//!   GET  /v1/models            — OpenAI model list (reports the configured model id)
//!   GET  /health               — liveness probe
//!
//! When `stream: true` is set in the request body, the handler returns a
//! `text/event-stream` SSE response built from the engine's `generate_stream`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
};
use futures::stream::BoxStream;
use serde_json::json;

use axum::body::Bytes;
use axum::http::HeaderMap;

use crate::api::common::{ChatRequest, ChatResult, StreamDelta};
use crate::api::openai::{OaiChatRequest, OaiModelInfo, OaiModelList};
use crate::api::anthropic::AnthRequest;

/// Constant-time byte comparison: false on length mismatch, otherwise XOR-
/// accumulate so the timing does not depend on where the first difference is.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Resolve the admin token: the CLI value, else a fresh random 32-hex string.
pub fn resolve_admin_token(cli: Option<String>) -> String {
    cli.unwrap_or_else(|| {
        let u = uuid::Uuid::new_v4();
        // 32 hex chars (no dashes)
        u.simple().to_string()
    })
}

/// Best-effort: write the admin token to `<config-dir>/localllm/admin-token`
/// with 0600 perms so the tray/window/CLI can read it. Logs only that it wrote
/// the file, never the value.
pub fn write_admin_token_file(token: &str) {
    let Some(dir) = dirs::config_dir() else { return };
    let path = dir.join("localllm").join("admin-token");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&path, token).is_ok() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        tracing::info!("admin token written to {}", path.display());
    }
}

/// Generate a short per-request id (e.g. `req-1a2b3c4d`) used to correlate the
/// start/finish log lines of a single prompt when several run concurrently.
fn new_request_id() -> String {
    let u = uuid::Uuid::new_v4();
    format!("req-{}", &u.simple().to_string()[..8])
}

/// Summarize an internal request for the start log line.
fn request_summary(req: &ChatRequest) -> (usize, usize) {
    (req.messages.len(), req.tools.len())
}

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

/// Bridge a local generation result into the cascade decision.
///
/// - `Ok(result)` and (cascade off OR result is strong) → returns `Ok(result)`;
///   the caller proceeds with the local result as before.
/// - `Ok(result)` weak (length-truncated) AND `want_cascade` → escalates: on
///   Relayed records success and returns `Err(cloud response)`; on Degrade keeps
///   the already-computed local result.
/// - `Err(gen error)` → if `want_cascade`, escalate to cloud; otherwise return
///   `Err(500)`.
///
/// Consumes `raw` (the original request bytes) because escalation reverse-proxies
/// it. Only ever called on the buffered/non-stream paths — never mid-stream.
#[allow(clippy::too_many_arguments)]
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

// ---------------------------------------------------------------------------
// Generator trait — decouples HTTP handlers from the real Engine
// ---------------------------------------------------------------------------

/// Async-trait wrapper so handlers can hold `Arc<dyn Generator>` and tests
/// can inject a `FakeGen` without loading the real model.
#[async_trait::async_trait]
pub trait Generator: Send + Sync {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult>;

    /// Stream inference. Returns a boxed stream of `StreamDelta` so the trait
    /// remains object-safe (`impl Stream` is not dyn-compatible).
    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>>;
}

// ---------------------------------------------------------------------------
// Router state — bundles the generator with the configured model id
// ---------------------------------------------------------------------------

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
    /// Per-session cloud usage counters + notification gates.
    pub usage: std::sync::Arc<crate::usage::Usage>,
    /// Session cloud-token total that triggers the one-shot high-usage alert.
    pub cloud_token_alert: usize,
}

// Implement Generator for Engine by delegating to its inherent methods.
#[async_trait::async_trait]
impl Generator for crate::engine::Engine {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult> {
        crate::engine::Engine::generate(self, req).await
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        crate::engine::Engine::generate_stream(self, req).await
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum Router with all four endpoints wired to the given generator,
/// the configured model id, the shared routing policy, the local context
/// window used by the context gate, and the session usage tracker.
pub fn router(
    gen: Arc<dyn Generator>,
    model_id: String,
    policy: Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    local_ctx_window: usize,
    usage: std::sync::Arc<crate::usage::Usage>,
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
    // Large prompts must reach the routing layer to be forwarded to cloud;
    // 64 MB ≈ ~16 M chars, giving ample headroom for over-window requests.
    // axum's default is 2 MB, which would reject them with 413 before routing.
    Router::new()
        .route("/v1/chat/completions", post(handle_oai_chat))
        .route("/v1/messages", post(handle_anth_messages))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions
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

    // --- Routing decision ---
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

    let (n_msgs, n_tools) = request_summary(&internal);
    tracing::info!(target: "localllm::req", "{rid} [openai] start: model={model} msgs={n_msgs} tools={n_tools} stream={stream_flag}");
    let started_at = Instant::now();

    if stream_flag && n_tools > 0 {
        // Buffered streaming: tool-bearing requests are generated fully (reusing
        // the tested non-streaming tool-call path) then replayed as SSE chunks.
        // Incremental streaming cannot frame tool_calls safely.
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::OpenAI,
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "openai",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 { result.completion_tokens as f64 / secs } else { 0.0 };
        tracing::info!(target: "localllm::req", "{rid} [openai] done (buffered stream): finish={:?} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.completion_tokens);
        let mut lines = crate::api::openai::stream_chunks_from_result(&result, &id, &model);
        lines.push("[DONE]".to_string());
        let sse_stream = futures::stream::iter(
            lines.into_iter().map(|l| {
                let data = l.strip_prefix("data: ").unwrap_or(&l).to_string();
                Ok::<Event, Infallible>(Event::default().data(data))
            }),
        );
        return Sse::new(sse_stream).into_response();
    }

    if stream_flag {
        // Streaming path: return SSE
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let id_clone = id.clone();
        let model_clone = model.clone();
        let rid_stream = rid.clone();

        let delta_stream = match state.gen.generate_stream(internal).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(target: "localllm::req", "{rid} [openai] 500 stream init: {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                ).into_response();
            }
        };

        // Thread the `started` flag so the first chunk includes "role":"assistant"
        // per the OpenAI streaming spec.
        let sse_stream = delta_stream
            .scan(false, move |started, result| {
                let was_started = *started;
                let event: Result<Event, Infallible> = match result {
                    Ok(delta) => {
                        if delta.done {
                            let secs = started_at.elapsed().as_secs_f64();
                            tracing::info!(target: "localllm::req", "{rid_stream} [openai] done (stream): {secs:.1}s");
                        }
                        let line = crate::api::openai::stream_chunk(&delta, &id_clone, &model_clone, was_started);
                        *started = true;
                        // line is "data: {json}", strip the "data: " prefix for Event::default().data()
                        let data = line.strip_prefix("data: ").unwrap_or(&line);
                        Ok(Event::default().data(data.to_string()))
                    }
                    Err(e) => {
                        tracing::error!(target: "localllm::req", "{rid_stream} [openai] stream error: {e}");
                        // Emit an error event; client will see it.
                        Ok(Event::default().data(format!("[ERROR] {e}")))
                    }
                };
                futures::future::ready(Some(event))
            })
            // Append the [DONE] sentinel after all deltas
            .chain(futures::stream::once(async {
                Ok::<Event, Infallible>(Event::default().data("[DONE]"))
            }));

        Sse::new(sse_stream).into_response()
    } else {
        // Non-streaming path
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::OpenAI,
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "openai",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 { result.completion_tokens as f64 / secs } else { 0.0 };
        tracing::info!(target: "localllm::req", "{rid} [openai] done: finish={:?} prompt_tok={} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.prompt_tokens, result.completion_tokens);
        let resp = crate::api::openai::from_internal(result, &model);
        Json(serde_json::to_value(resp).unwrap()).into_response()
    }
}

/// POST /v1/messages
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

    // --- Routing decision ---
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

    let (n_msgs, n_tools) = request_summary(&internal);
    tracing::info!(target: "localllm::req", "{rid} [anthropic] start: model={model} msgs={n_msgs} tools={n_tools} stream={stream_flag}");
    let started_at = Instant::now();

    if stream_flag && n_tools > 0 {
        // Buffered streaming: tool-bearing requests (e.g. every Claude Code turn)
        // are generated fully then replayed as a correct tool_use/text SSE
        // sequence. Incremental streaming cannot frame tool_use blocks safely.
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::Anthropic,
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "anthropic",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 { result.completion_tokens as f64 / secs } else { 0.0 };
        tracing::info!(target: "localllm::req", "{rid} [anthropic] done (buffered stream): finish={:?} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.completion_tokens);
        let events = crate::api::anthropic::stream_events_from_result(&result, &model);
        let sse_stream = futures::stream::iter(events.into_iter().map(|e_str| {
            let mut lines = e_str.splitn(2, '\n');
            let event_type = lines.next().unwrap_or("").strip_prefix("event: ").unwrap_or("").to_string();
            let data = lines.next().unwrap_or("").strip_prefix("data: ").unwrap_or("").to_string();
            Ok::<Event, Infallible>(Event::default().event(event_type).data(data))
        }));
        return Sse::new(sse_stream).into_response();
    }

    if stream_flag {
        // Streaming path: return Anthropic SSE protocol
        let rid_stream = rid.clone();
        let delta_stream = match state.gen.generate_stream(internal).await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(target: "localllm::req", "{rid} [anthropic] 500 stream init: {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                ).into_response();
            }
        };

        // Collect all events, tracking whether the first chunk was sent.
        // We use scan to thread `started` state through the stream.
        let sse_stream = delta_stream
            .scan(false, move |started, result| {
                let was_started = *started;
                let events: Vec<Result<Event, Infallible>> = match result {
                    Ok(delta) => {
                        if delta.done {
                            let secs = started_at.elapsed().as_secs_f64();
                            tracing::info!(target: "localllm::req", "{rid_stream} [anthropic] done (stream): {secs:.1}s");
                        }
                        let event_strs = crate::api::anthropic::stream_events(&delta, was_started);
                        *started = true;
                        event_strs
                            .into_iter()
                            .map(|e_str| {
                                // Each event string is "event: <type>\ndata: {json}"
                                // Parse it into axum's Event type.
                                let mut lines = e_str.splitn(2, '\n');
                                let event_line = lines.next().unwrap_or("");
                                let data_line = lines.next().unwrap_or("");
                                let event_type = event_line.strip_prefix("event: ").unwrap_or("");
                                let data = data_line.strip_prefix("data: ").unwrap_or("");
                                Ok(Event::default().event(event_type).data(data.to_string()))
                            })
                            .collect()
                    }
                    Err(e) => {
                        tracing::error!(target: "localllm::req", "{rid_stream} [anthropic] stream error: {e}");
                        vec![Ok(Event::default().data(format!("[ERROR] {e}")))]
                    }
                };
                futures::future::ready(Some(futures::stream::iter(events)))
            })
            .flatten();

        Sse::new(sse_stream).into_response()
    } else {
        // Non-streaming path
        let result = match cascade_or_result(
            want_cascade,
            state.gen.generate(internal).await,
            crate::cloud::Provider::Anthropic,
            &headers,
            raw,
            &state,
            est_prompt_tokens,
            &rid,
            "anthropic",
        ).await {
            Ok(r) => r,
            Err(resp) => return resp,
        };
        let secs = started_at.elapsed().as_secs_f64();
        let tps = if secs > 0.0 { result.completion_tokens as f64 / secs } else { 0.0 };
        tracing::info!(target: "localllm::req", "{rid} [anthropic] done: finish={:?} prompt_tok={} completion_tok={} {secs:.1}s {tps:.1} tok/s", result.finish_reason, result.prompt_tokens, result.completion_tokens);
        let resp = crate::api::anthropic::from_internal(result, &model);
        Json(serde_json::to_value(resp).unwrap()).into_response()
    }
}

/// GET /v1/models — reports the configured model id.
async fn handle_models(State(state): State<Arc<AppState>>) -> Json<OaiModelList> {
    Json(OaiModelList {
        object: "list".to_string(),
        data: vec![OaiModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            owned_by: "localllm".to_string(),
        }],
    })
}

/// GET /health
async fn handle_health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

#[cfg(test)]
mod tests {
    #[test]
    fn constant_time_eq_matches_and_rejects() {
        assert!(super::constant_time_eq(b"abc", b"abc"));
        assert!(!super::constant_time_eq(b"abc", b"abd"));
        assert!(!super::constant_time_eq(b"abc", b"abcd")); // length mismatch
        assert!(!super::constant_time_eq(b"", b"x"));
    }

    #[test]
    fn resolve_admin_token_uses_cli_else_random() {
        assert_eq!(super::resolve_admin_token(Some("z".into())), "z");
        let r = super::resolve_admin_token(None);
        assert_eq!(r.len(), 32); // random 32-hex
        assert_ne!(super::resolve_admin_token(None), r); // different each call
    }
}
