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
                match crate::cloud::forward(provider, headers, raw).await {
                    crate::cloud::ForwardOutcome::Relayed(resp) => Err(resp),
                    crate::cloud::ForwardOutcome::Degrade(d) => Err(crate::cloud::degrade_error(d)),
                }
            } else {
                Ok(result)
            }
        }
        Err(e) => {
            if want_cascade {
                tracing::warn!(target: "localllm::req", "{rid} [{api}] cascade: local generate failed ({e}) → escalating to cloud");
                match crate::cloud::forward(provider, headers, raw).await {
                    crate::cloud::ForwardOutcome::Relayed(resp) => Err(resp),
                    crate::cloud::ForwardOutcome::Degrade(d) => Err(crate::cloud::degrade_error(d)),
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
    let want_cascade = match route_decision(&state, &internal, &headers) {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [openai] route=cloud reason={reason:?}");
            return match crate::cloud::forward(crate::cloud::Provider::OpenAI, &headers, raw).await {
                crate::cloud::ForwardOutcome::Relayed(resp) => resp,
                crate::cloud::ForwardOutcome::Degrade(d) => crate::cloud::degrade_error(d),
            };
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
    let want_cascade = match route_decision(&state, &internal, &headers) {
        crate::route::Decision::Cloud(reason) => {
            tracing::info!(target: "localllm::req", "{rid} [anthropic] route=cloud reason={reason:?}");
            return match crate::cloud::forward(crate::cloud::Provider::Anthropic, &headers, raw).await {
                crate::cloud::ForwardOutcome::Relayed(resp) => resp,
                crate::cloud::ForwardOutcome::Degrade(d) => crate::cloud::degrade_error(d),
            };
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
