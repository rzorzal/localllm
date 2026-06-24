//! HTTP router and handlers for Task 6/8.
//!
//! Exposes four endpoints:
//!   POST /v1/chat/completions  — OpenAI Chat Completions API (streaming + non-streaming)
//!   POST /v1/messages          — Anthropic Messages API (streaming + non-streaming)
//!   GET  /v1/models            — OpenAI model list (static)
//!   GET  /health               — liveness probe
//!
//! When `stream: true` is set in the request body, the handler returns a
//! `text/event-stream` SSE response built from the engine's `generate_stream`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
};
use futures::stream::BoxStream;
use serde_json::json;

use crate::api::common::{ChatRequest, ChatResult, StreamDelta};
use crate::api::openai::{OaiChatRequest, OaiModelInfo, OaiModelList};
use crate::api::anthropic::AnthRequest;

/// The fixed model id returned by GET /v1/models.
/// Task 7 will make this configurable; a const is sufficient here.
pub const MODEL_ID: &str = "qwen2.5-7b-instruct";

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

/// Build the axum Router with all four endpoints wired to the given generator.
pub fn router(state: Arc<dyn Generator>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_oai_chat))
        .route("/v1/messages", post(handle_anth_messages))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions
async fn handle_oai_chat(
    State(gen): State<Arc<dyn Generator>>,
    Json(req): Json<OaiChatRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;
    use uuid::Uuid;

    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::openai::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e})),
            ).into_response();
        }
    };

    if stream_flag {
        // Streaming path: return SSE
        let id = format!("chatcmpl-{}", Uuid::new_v4());
        let id_clone = id.clone();
        let model_clone = model.clone();

        let delta_stream = match gen.generate_stream(internal).await {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                ).into_response();
            }
        };

        let sse_stream = delta_stream
            .map(move |result| -> Result<Event, Infallible> {
                match result {
                    Ok(delta) => {
                        let line = crate::api::openai::stream_chunk(&delta, &id_clone, &model_clone);
                        // line is "data: {json}", strip the "data: " prefix for Event::default().data()
                        let data = line.strip_prefix("data: ").unwrap_or(&line);
                        Ok(Event::default().data(data.to_string()))
                    }
                    Err(e) => {
                        // Emit an error event and keep stream alive; client will see it.
                        Ok(Event::default().data(format!("[ERROR] {e}")))
                    }
                }
            })
            // Append the [DONE] sentinel after all deltas
            .chain(futures::stream::once(async {
                Ok::<Event, Infallible>(Event::default().data("[DONE]"))
            }));

        Sse::new(sse_stream).into_response()
    } else {
        // Non-streaming path (unchanged from Task 6)
        let result = match gen.generate(internal).await {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                ).into_response();
            }
        };
        let resp = crate::api::openai::from_internal(result, &model);
        Json(serde_json::to_value(resp).unwrap()).into_response()
    }
}

/// POST /v1/messages
async fn handle_anth_messages(
    State(gen): State<Arc<dyn Generator>>,
    Json(req): Json<AnthRequest>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;

    let model = req.model.clone();
    let stream_flag = req.stream.unwrap_or(false);

    let internal = match crate::api::anthropic::to_internal(req) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": e})),
            ).into_response();
        }
    };

    if stream_flag {
        // Streaming path: return Anthropic SSE protocol
        let delta_stream = match gen.generate_stream(internal).await {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                ).into_response();
            }
        };

        // Collect all events, tracking whether the first chunk was sent.
        // We use scan to thread `started` state through the stream.
        let sse_stream = delta_stream
            .scan(false, |started, result| {
                let was_started = *started;
                let events: Vec<Result<Event, Infallible>> = match result {
                    Ok(delta) => {
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
                        vec![Ok(Event::default().data(format!("[ERROR] {e}")))]
                    }
                };
                futures::future::ready(Some(futures::stream::iter(events)))
            })
            .flatten();

        Sse::new(sse_stream).into_response()
    } else {
        // Non-streaming path (unchanged from Task 6)
        let result = match gen.generate(internal).await {
            Ok(r) => r,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": e.to_string()})),
                ).into_response();
            }
        };
        let resp = crate::api::anthropic::from_internal(result, &model);
        Json(serde_json::to_value(resp).unwrap()).into_response()
    }
}

/// GET /v1/models
async fn handle_models() -> Json<OaiModelList> {
    Json(OaiModelList {
        object: "list".to_string(),
        data: vec![OaiModelInfo {
            id: MODEL_ID.to_string(),
            object: "model".to_string(),
            owned_by: "localllm".to_string(),
        }],
    })
}

/// GET /health
async fn handle_health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}
