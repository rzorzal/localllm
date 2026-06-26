pub mod api;
pub mod catalog;
pub mod cloud;
pub mod config;
pub mod download;
pub mod engine;
pub mod engine_llama;
pub mod model_manager;
pub mod route;
pub mod settings;
pub mod usage;
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
    use std::sync::{Arc, RwLock};
    let profile = crate::settings::resolve_profile(cfg.profile);
    tracing::info!("routing profile: {profile:?}");
    let policy = Arc::new(RwLock::new(profile.policy()));
    run_server_with_ready_and_policy(cfg, ready, policy).await
}

/// Like [`run_server_with_ready`] but takes an externally-owned routing policy
/// so a caller (the tray) can share the same `Arc<RwLock<RoutingPolicy>>` and
/// mutate it live while the server reads it per request.
pub async fn run_server_with_ready_and_policy(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    policy: std::sync::Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
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

    crate::usage::enable_notifications();
    let usage = std::sync::Arc::new(crate::usage::Usage::new());

    // Wrap the initial engine in a hot-swappable ModelManager. The builder
    // downloads (reporting progress to the manager via a Weak slot to avoid a
    // reference cycle) then loads a LlamaEngine. A switch always builds a
    // LlamaEngine (switching backends is out of scope).
    use crate::model_manager::{EngineBuilder, ModelManager, ModelSpec};
    let initial_spec = ModelSpec {
        repo: cfg.model_id.clone(),
        file: cfg.gguf_files[0].clone(),
    };
    let b_ctx_len = cfg.ctx_len;
    let b_kv_type = cfg.llama_kv_cache_type();
    let b_kv_dir = cfg.resolved_kv_cache_dir();
    let manager_slot: Arc<std::sync::OnceLock<std::sync::Weak<ModelManager>>> =
        Arc::new(std::sync::OnceLock::new());
    let slot_for_builder = manager_slot.clone();
    let builder: EngineBuilder = Box::new(move |spec: ModelSpec| {
        let kv_dir = b_kv_dir.clone();
        let slot = slot_for_builder.clone();
        Box::pin(async move {
            let progress_target = slot.get().and_then(|w| w.upgrade());
            crate::download::ensure_model_with_progress(
                &spec.repo,
                std::slice::from_ref(&spec.file),
                |done, total| {
                    if let Some(m) = &progress_target {
                        let pct = match total {
                            Some(t) if t > 0 => (done * 100 / t) as u8,
                            _ => 0,
                        };
                        m.set_progress(pct);
                    }
                },
            )
            .await?;
            let engine =
                LlamaEngine::load(&spec.repo, &[spec.file], b_ctx_len, b_kv_type, kv_dir).await?;
            Ok(Arc::new(engine) as Arc<dyn Generator>)
        })
    });
    let manager = ModelManager::new(engine, initial_spec, builder);
    let _ = manager_slot.set(Arc::downgrade(&manager));

    let admin_token = crate::server::resolve_admin_token(cfg.admin_token.clone());
    crate::server::write_admin_token_file(&admin_token);

    let app = router(
        manager,
        cfg.model_id.clone(),
        policy,
        cfg.ctx_len,
        usage,
        cfg.cloud_token_alert,
        Arc::from(admin_token),
    );
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
    tracing::info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    if let Some(flag) = ready {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    axum::serve(listener, app).await?;
    Ok(())
}

/// Test/diagnostic generator whose responses echo a fixed tag, so a test can
/// tell which engine instance is currently serving (used by hot-swap tests).
pub mod test_support {
    use crate::api::common::{ChatResult, ContentPart, FinishReason, StreamDelta};
    use crate::server::Generator;
    use futures::stream::BoxStream;

    pub struct TaggedGen(pub &'static str);

    #[async_trait::async_trait]
    impl Generator for TaggedGen {
        async fn generate(&self, _req: crate::api::common::ChatRequest) -> anyhow::Result<ChatResult> {
            Ok(ChatResult {
                content: vec![ContentPart::Text(self.0.to_string())],
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }
        async fn generate_stream(
            &self,
            _req: crate::api::common::ChatRequest,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
            let tag = self.0.to_string();
            let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
                Ok(StreamDelta { text: Some(tag), done: false, finish_reason: None }),
                Ok(StreamDelta { text: None, done: true, finish_reason: Some(FinishReason::Stop) }),
            ];
            Ok(Box::pin(futures::stream::iter(deltas)))
        }
    }
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

/// Build a test router with a custom generator, routing policy, and local
/// context window. Lets tests drive specific routing/cascade decisions.
pub fn router_for_test_with(
    gen: Arc<dyn Generator>,
    policy: crate::route::RoutingPolicy,
    local_ctx_window: usize,
) -> Router {
    use crate::model_manager::{EngineBuilder, ModelManager, ModelSpec};
    let policy = Arc::new(std::sync::RwLock::new(policy));
    let usage = Arc::new(crate::usage::Usage::new());
    // Test builder: an instant TaggedGen("switched") — no download/Metal/network.
    let builder: EngineBuilder = Box::new(|_spec| {
        Box::pin(async {
            Ok(Arc::new(crate::test_support::TaggedGen("switched")) as Arc<dyn Generator>)
        })
    });
    let manager = ModelManager::new(
        gen,
        ModelSpec { repo: "test".into(), file: "test".into() },
        builder,
    );
    crate::server::router(
        manager,
        "test-model".to_string(),
        policy,
        local_ctx_window,
        usage,
        200_000,
        Arc::from("test-token"),
    )
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

/// GET `path` on `app` → HTTP status code (for auth-gate tests).
pub async fn axum_test_get_status(app: Router, path: &str) -> u16 {
    use axum::body::Body;
    use tower::ServiceExt;
    let request = axum::http::Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    app.oneshot(request).await.unwrap().status().as_u16()
}

/// GET `path` on `app` with a header → parsed JSON body.
pub async fn axum_test_get_with_header(
    app: Router,
    path: &str,
    hname: &str,
    hval: &str,
) -> serde_json::Value {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let request = axum::http::Request::builder()
        .method("GET")
        .uri(path)
        .header(hname, hval)
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
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
