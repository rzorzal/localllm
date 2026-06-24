pub mod api;
pub mod config;
pub mod engine;
pub mod server;

// ---------------------------------------------------------------------------
// Test helpers (always public so integration tests in tests/ can import them)
// ---------------------------------------------------------------------------

use std::sync::Arc;

use axum::Router;
use futures::stream::BoxStream;

use crate::api::common::{ChatResult, ContentPart, FinishReason, StreamDelta, ToolCall};
use crate::server::Generator;

/// A fake generator that always returns a fixed tool call to `get_weather`.
/// Used by integration tests so the real model is never loaded.
pub struct FakeGen;

#[async_trait::async_trait]
impl Generator for FakeGen {
    async fn generate(&self, _req: crate::api::common::ChatRequest) -> anyhow::Result<ChatResult> {
        Ok(ChatResult {
            content: vec![ContentPart::Call(ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Recife"}"#.to_string(),
            })],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 10,
            completion_tokens: 5,
        })
    }

    async fn generate_stream(
        &self,
        _req: crate::api::common::ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        // Return a small fixed stream: two text deltas then a done delta.
        let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
            Ok(StreamDelta { text: Some("Hello".into()), done: false, finish_reason: None }),
            Ok(StreamDelta { text: Some(" world".into()), done: false, finish_reason: None }),
            Ok(StreamDelta { text: None, done: true, finish_reason: Some(FinishReason::Stop) }),
        ];
        Ok(Box::pin(futures::stream::iter(deltas)))
    }
}

/// Build a test router wired to `FakeGen`.
pub fn router_for_test() -> Router {
    crate::server::router(Arc::new(FakeGen))
}

/// Send a POST with a JSON body to `path` on `app` and return the parsed
/// response body as a `serde_json::Value`.
///
/// Uses axum/tower's `oneshot` so no real TCP listener is needed.
pub async fn axum_test_request(app: Router, path: &str, body: &str) -> serde_json::Value {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let request = axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// Send a GET request to `path` on `app` and return the parsed response body.
pub async fn axum_test_get(app: Router, path: &str) -> serde_json::Value {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let request = axum::http::Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// Send a POST with a JSON body to `path` on `app` and return the raw
/// response body as a `String`. Used for SSE streaming tests where the body
/// is not a single JSON object.
pub async fn axum_test_request_raw(app: Router, path: &str, body: &str) -> String {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let request = axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}
