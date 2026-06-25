pub mod api;
pub mod cloud;
pub mod config;
pub mod download;
pub mod engine;
pub mod engine_llama;
pub mod route;
pub mod server;
pub mod tscg;
#[cfg(target_os = "macos")]
pub mod tray;

// ---------------------------------------------------------------------------
// Shared server startup logic
// ---------------------------------------------------------------------------

/// Build the engine + router, bind, and serve.
///
/// Callable from both the headless `main()` path and the tray-mode background
/// thread. Never returns on success (the axum serve future runs forever).
pub async fn run_server(cfg: crate::config::Config) -> anyhow::Result<()> {
    run_server_with_ready(cfg, None).await
}

/// Like [`run_server`] but flips `ready` to `true` once the listener is bound
/// (server actually serving). Used by the tray to show a live "running" status
/// only after the model has loaded and the port is up.
pub async fn run_server_with_ready(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> anyhow::Result<()> {
    use std::sync::Arc;
    use crate::config::Backend;
    use crate::engine::Engine;
    use crate::engine_llama::LlamaEngine;
    use crate::server::{router, Generator};

    tracing::info!("loading model {} (backend={:?})…", cfg.model_id, cfg.backend);

    let engine: Arc<dyn Generator> = match cfg.backend {
        Backend::Llama => {
            let kv_cache_type = cfg.llama_kv_cache_type();
            let kv_cache_dir = cfg.resolved_kv_cache_dir();
            tracing::info!("KV cache type: --kv-type={:?}", cfg.kv_type);
            tracing::info!(
                "KV persist dir: {:?} (no-persist={})",
                kv_cache_dir,
                cfg.no_kv_persist
            );
            Arc::new(
                LlamaEngine::load(
                    &cfg.model_id,
                    &cfg.gguf_files,
                    cfg.ctx_len,
                    kv_cache_type,
                    kv_cache_dir,
                )
                .await?,
            )
        }
        Backend::Mistralrs => Arc::new(Engine::load(&cfg.engine_config()).await?),
    };

    let app = router(engine, cfg.model_id.clone());
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
    tracing::info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    if let Some(flag) = ready {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    axum::serve(listener, app).await?;
    Ok(())
}

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

/// Build a test router wired to `FakeGen` and a fixed model id.
pub fn router_for_test() -> Router {
    crate::server::router(Arc::new(FakeGen), "test-model".to_string())
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

/// Send a POST with a JSON body to `path` on `app` and return the HTTP status
/// code as a `u16`. Used to assert error-path responses without parsing the body.
pub async fn axum_test_request_status(app: Router, path: &str, body: &str) -> u16 {
    use axum::body::Body;
    use tower::ServiceExt;

    let request = axum::http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    response.status().as_u16()
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
