//! HTTP router and handlers for Task 6.
//!
//! Exposes four endpoints:
//!   POST /v1/chat/completions  — OpenAI Chat Completions API
//!   POST /v1/messages          — Anthropic Messages API
//!   GET  /v1/models            — OpenAI model list (static)
//!   GET  /health               — liveness probe
//!
//! All handlers are non-streaming. Streaming is Task 8.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use serde_json::json;

use crate::api::common::{ChatRequest, ChatResult};
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
}

// Implement Generator for Engine by delegating to its inherent method.
#[async_trait::async_trait]
impl Generator for crate::engine::Engine {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult> {
        crate::engine::Engine::generate(self, req).await
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
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let model = req.model.clone();
    let internal = crate::api::openai::to_internal(req).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e})),
        )
    })?;

    let result = gen.generate(internal).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    let resp = crate::api::openai::from_internal(result, &model);
    Ok(Json(serde_json::to_value(resp).unwrap()))
}

/// POST /v1/messages
async fn handle_anth_messages(
    State(gen): State<Arc<dyn Generator>>,
    Json(req): Json<AnthRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let model = req.model.clone();
    let internal = crate::api::anthropic::to_internal(req).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e})),
        )
    })?;

    let result = gen.generate(internal).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    let resp = crate::api::anthropic::from_internal(result, &model);
    Ok(Json(serde_json::to_value(resp).unwrap()))
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
