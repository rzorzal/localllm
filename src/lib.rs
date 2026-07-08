// Backend selection guard. On Linux/Windows a build must pick an inference
// backend explicitly; without one the native engines would fail to link with
// an inscrutable error. macOS is exempt: its target dependency table
// (Cargo.toml) always supplies Metal, so a plain `cargo build` needs no flag.
#[cfg(all(not(target_os = "macos"), not(feature = "cuda"), not(feature = "cpu")))]
compile_error!(
    "No inference backend selected. On Linux/Windows build with \
     `--features cuda` or `--features cpu`. macOS enables Metal automatically."
);

pub mod api;
pub mod breaker;
pub mod budget;
pub mod catalog;
pub mod catalog_variants;
pub mod cloud;
pub mod config;
pub mod download;
pub mod engine_llama;
pub mod fit;
pub mod history_select;
pub mod integrations;
pub mod launch;
pub mod model_manager;
pub mod pricing;
pub mod profile;
pub mod route;
pub mod route_log;
pub mod server;
pub mod settings;
pub mod tray;
pub mod tscg;
pub mod usage;

// ---------------------------------------------------------------------------
// Per-model profile resolution helpers
// ---------------------------------------------------------------------------

/// Resolve a model's effective load parameters (ctx, kv_type, gpu_layers) from
/// its saved profile, the catalog recommendation, and the global CLI defaults.
fn resolve_load_params(
    repo: &str,
    file: &str,
    global_ctx: u32,
    global_kv: crate::config::KvType,
) -> crate::profile::Resolved {
    let key = crate::settings::model_ctx_key(repo, file);
    let saved = crate::settings::load_model_profile(&key);
    let catalog = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == repo && e.file == file);
    crate::profile::resolve(&saved, catalog, global_ctx, global_kv)
}

/// Resolve the GGUF file list for a model + quant: the variant's files if the
/// quant is available, else the model's default `file`.
fn resolve_variant_files(repo: &str, file: &str, quant: &str) -> Vec<String> {
    if let Some(entry) = crate::catalog::CATALOG
        .iter()
        .find(|e| e.repo == repo && e.file == file)
    {
        if let Some(files) = crate::catalog::files_for_quant(entry, quant) {
            return files;
        }
    }
    vec![file.to_string()]
}

/// The effective quant for a switch: an explicit spec quant wins over the
/// profile-resolved quant.
fn effective_quant(spec_quant: Option<String>, resolved_quant: String) -> String {
    spec_quant.unwrap_or(resolved_quant)
}

/// Map `KvType` to the llama-cpp-2 KV cache type.
fn kv_type_to_llama(t: crate::config::KvType) -> llama_cpp_2::context::params::KvCacheType {
    use llama_cpp_2::context::params::KvCacheType;
    match t {
        crate::config::KvType::Q8 => KvCacheType::Q8_0,
        crate::config::KvType::Q4 => KvCacheType::Q4_0,
        crate::config::KvType::F16 => KvCacheType::F16,
    }
}

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
    let policy = Arc::new(RwLock::new(crate::settings::resolve_policy(profile)));
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
    let admin_token =
        std::sync::Arc::from(crate::server::resolve_admin_token(cfg.admin_token.clone()));
    let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
    run_server_with_ready_policy_token(cfg, ready, policy, admin_token, None, breaker).await
}

/// Like [`run_server_with_ready_and_policy`] but takes an externally-resolved
/// admin token (so the tray shares the same token with its Model Manager window).
///
/// `manager_out`, when supplied, is filled with the live `Arc<ModelManager>`
/// once built, so a caller (the tray) can read the current model after a
/// hot-swap and keep its menu in sync.
pub async fn run_server_with_ready_policy_token(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    policy: std::sync::Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    admin_token: std::sync::Arc<str>,
    manager_out: Option<
        std::sync::Arc<std::sync::OnceLock<std::sync::Arc<crate::model_manager::ModelManager>>>,
    >,
    breaker: std::sync::Arc<crate::breaker::CircuitBreaker>,
) -> anyhow::Result<()> {
    use crate::engine_llama::LlamaEngine;
    use crate::server::{router, Generator};
    use std::sync::Arc;

    tracing::info!(
        "loading model {} (backend={:?})…",
        cfg.model_id,
        cfg.backend
    );

    let total_ram_mb = {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        sys.total_memory() / (1024 * 1024) // bytes → MB
    };
    if total_ram_mb == 0 {
        tracing::warn!("sysinfo reported 0 total RAM; catalog fit verdicts will use budget=0");
    }

    // Rolling retention: drop routing-log lines older than ~30 days at boot,
    // and cap the plain-text app log so neither grows unbounded.
    crate::route_log::prune_file(crate::route_log::now_secs(), 30 * 24 * 3600);
    crate::route_log::cap_lines_file(20_000);
    crate::route_log::rotate_app_log(crate::route_log::now_secs(), 7 * 24 * 3600);

    // Single backend: embedded llama.cpp. (The mistralrs backend was removed —
    // it was an unused alternate that also blocked the CUDA build.)
    let (engine, effective_ctx): (Arc<dyn Generator>, usize) = {
        let kv_cache_dir = cfg.resolved_kv_cache_dir();
        tracing::info!("KV cache type: --kv-type={:?}", cfg.kv_type);
        tracing::info!(
            "KV persist dir: {:?} (no-persist={})",
            kv_cache_dir,
            cfg.no_kv_persist
        );
        let r = resolve_load_params(
            &cfg.model_id,
            &cfg.gguf_files[0],
            cfg.ctx_len as u32,
            cfg.kv_type.clone(),
        );
        tracing::info!(
            "resolved load params: ctx={} kv={:?} gpu_layers={:?}",
            r.ctx,
            r.kv_type,
            r.gpu_layers
        );
        // Startup: use CLI --gguf-file directly; saved-quant variant resolution applies on switch, not here.
        let llama = LlamaEngine::load(
            &cfg.model_id,
            &cfg.gguf_files,
            r.ctx as usize,
            kv_type_to_llama(r.kv_type),
            kv_cache_dir,
            r.gpu_layers,
            total_ram_mb,
        )
        .await?;
        let eff = llama.ctx_window();
        (Arc::new(llama) as Arc<dyn Generator>, eff)
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
        quant: None,
    };
    let b_ctx_len = cfg.ctx_len as u32;
    let b_kv_type = cfg.kv_type.clone(); // crate::config::KvType
    let b_kv_dir = cfg.resolved_kv_cache_dir();
    let b_total_ram_mb = total_ram_mb;
    let manager_slot: Arc<std::sync::OnceLock<std::sync::Weak<ModelManager>>> =
        Arc::new(std::sync::OnceLock::new());
    let slot_for_builder = manager_slot.clone();
    let builder: EngineBuilder = Box::new(move |spec: ModelSpec| {
        let kv_dir = b_kv_dir.clone();
        let kv_type = b_kv_type.clone(); // clone per-invocation before moving into async block
        let slot = slot_for_builder.clone();
        Box::pin(async move {
            let r = resolve_load_params(&spec.repo, &spec.file, b_ctx_len, kv_type);
            // Prefer an explicit quant on the switch spec; else the saved-profile-resolved quant.
            let quant = effective_quant(spec.quant.clone(), r.quant.clone());
            // On switch, load the resolved quant variant's file list.
            // The startup path keeps &cfg.gguf_files unchanged (CLI is authoritative).
            let files = resolve_variant_files(&spec.repo, &spec.file, &quant);
            let progress_target = slot.get().and_then(|w| w.upgrade());
            crate::download::ensure_model_with_progress(&spec.repo, &files, |done, total| {
                if let Some(m) = &progress_target {
                    let pct = match total {
                        Some(t) if t > 0 => (done * 100 / t) as u8,
                        _ => 0,
                    };
                    m.set_progress(pct);
                }
            })
            .await?;
            let engine = LlamaEngine::load(
                &spec.repo,
                &files,
                r.ctx as usize,
                kv_type_to_llama(r.kv_type),
                kv_dir,
                r.gpu_layers,
                b_total_ram_mb,
            )
            .await?;
            Ok(Arc::new(engine) as Arc<dyn Generator>)
        })
    });
    let manager = ModelManager::new(engine, initial_spec, builder);
    let _ = manager_slot.set(Arc::downgrade(&manager));
    // Hand the live manager to the tray (if any) so it can poll the active model.
    if let Some(slot) = manager_out {
        let _ = slot.set(manager.clone());
    }

    crate::server::write_admin_token_file(&admin_token);

    let app = router(
        manager,
        cfg.model_id.clone(),
        policy,
        effective_ctx,
        usage,
        cfg.cloud_token_alert,
        admin_token.clone(),
        total_ram_mb,
        cfg.ctx_len as u32,
        match cfg.kv_type {
            crate::config::KvType::Q8 => crate::fit::KvKind::Q8,
            crate::config::KvType::Q4 => crate::fit::KvKind::Q4,
            crate::config::KvType::F16 => crate::fit::KvKind::F16,
        },
        cfg.port,
        breaker,
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
        async fn generate(
            &self,
            _req: crate::api::common::ChatRequest,
        ) -> anyhow::Result<ChatResult> {
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
                Ok(StreamDelta {
                    text: Some(tag),
                    done: false,
                    finish_reason: None,
                }),
                Ok(StreamDelta {
                    text: None,
                    done: true,
                    finish_reason: Some(FinishReason::Stop),
                }),
            ];
            Ok(Box::pin(futures::stream::iter(deltas)))
        }
        /// Test generators represent a warm engine; return 0 to avoid spurious
        /// ColdPrefill escalations in unit/integration tests.
        async fn estimate_cold_tokens(&self, _req: crate::api::common::ChatRequest) -> usize {
            0
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
            Ok(StreamDelta {
                text: Some("Hello".into()),
                done: false,
                finish_reason: None,
            }),
            Ok(StreamDelta {
                text: Some(" world".into()),
                done: false,
                finish_reason: None,
            }),
            Ok(StreamDelta {
                text: None,
                done: true,
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        Ok(Box::pin(futures::stream::iter(deltas)))
    }

    /// Test generators have a warm cache; override to prevent spurious ColdPrefill escalations.
    async fn estimate_cold_tokens(&self, _req: crate::api::common::ChatRequest) -> usize {
        0
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

    /// Test generators have a warm cache; override to prevent spurious ColdPrefill escalations.
    async fn estimate_cold_tokens(&self, _req: crate::api::common::ChatRequest) -> usize {
        0
    }

    async fn generate_stream(
        &self,
        _req: crate::api::common::ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
            Ok(StreamDelta {
                text: Some("local-weak-answer".into()),
                done: false,
                finish_reason: None,
            }),
            Ok(StreamDelta {
                text: None,
                done: true,
                finish_reason: Some(FinishReason::Length),
            }),
        ];
        Ok(Box::pin(futures::stream::iter(deltas)))
    }
}

/// A generator that records the messages it receives (for assertion in tests).
pub struct RecordingGen(pub Arc<std::sync::Mutex<Vec<crate::api::common::ChatMessage>>>);

#[async_trait::async_trait]
impl Generator for RecordingGen {
    async fn generate(&self, req: crate::api::common::ChatRequest) -> anyhow::Result<ChatResult> {
        *self.0.lock().unwrap() = req.messages.clone();
        Ok(ChatResult {
            content: vec![ContentPart::Text("ok".into())],
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
    }

    async fn generate_stream(
        &self,
        req: crate::api::common::ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        *self.0.lock().unwrap() = req.messages.clone();
        let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
            Ok(StreamDelta {
                text: Some("ok".into()),
                done: false,
                finish_reason: None,
            }),
            Ok(StreamDelta {
                text: None,
                done: true,
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        Ok(Box::pin(futures::stream::iter(deltas)))
    }

    async fn estimate_cold_tokens(&self, _req: crate::api::common::ChatRequest) -> usize {
        0
    }
}

/// A generator that records the tool names it receives (for assertion in tests).
pub struct ToolRecordingGen(pub Arc<std::sync::Mutex<Vec<String>>>);

#[async_trait::async_trait]
impl Generator for ToolRecordingGen {
    async fn generate(&self, req: crate::api::common::ChatRequest) -> anyhow::Result<ChatResult> {
        *self.0.lock().unwrap() = req.tools.iter().map(|t| t.name.clone()).collect();
        Ok(ChatResult {
            content: vec![ContentPart::Text("ok".into())],
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        })
    }

    async fn generate_stream(
        &self,
        req: crate::api::common::ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        *self.0.lock().unwrap() = req.tools.iter().map(|t| t.name.clone()).collect();
        let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
            Ok(StreamDelta {
                text: Some("ok".into()),
                done: false,
                finish_reason: None,
            }),
            Ok(StreamDelta {
                text: None,
                done: true,
                finish_reason: Some(FinishReason::Stop),
            }),
        ];
        Ok(Box::pin(futures::stream::iter(deltas)))
    }

    async fn estimate_cold_tokens(&self, _req: crate::api::common::ChatRequest) -> usize {
        0
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
        ModelSpec {
            repo: "test".into(),
            file: "test".into(),
            quant: None,
        },
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
        16384,
        32768,
        crate::fit::KvKind::Q8,
        31415,
        Arc::new(crate::breaker::CircuitBreaker::new()),
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

/// GET `path` → (status, content-type, body string).
pub async fn axum_test_get_full(app: Router, path: &str) -> (u16, String, String) {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let request = axum::http::Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let ctype = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, ctype, String::from_utf8_lossy(&bytes).into_owned())
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

/// DELETE with a JSON body + header → HTTP status code.
pub async fn axum_test_delete_status_with_header(
    app: Router,
    path: &str,
    body: &str,
    hname: &str,
    hval: &str,
) -> u16 {
    use axum::body::Body;
    use tower::ServiceExt;
    let request = axum::http::Request::builder()
        .method("DELETE")
        .uri(path)
        .header("content-type", "application/json")
        .header(hname, hval)
        .body(Body::from(body.to_owned()))
        .unwrap();
    app.oneshot(request).await.unwrap().status().as_u16()
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

#[cfg(test)]
mod tests {
    #[test]
    fn effective_quant_prefers_spec_over_resolved() {
        assert_eq!(
            crate::effective_quant(Some("Q8_0".into()), "Q4_K_M".into()),
            "Q8_0"
        );
        assert_eq!(crate::effective_quant(None, "Q4_K_M".into()), "Q4_K_M");
    }

    #[test]
    fn resolve_variant_files_prefers_quant_then_falls_back() {
        // Absent repo → no catalog entry → falls back to [file] for any quant.
        let files = crate::resolve_variant_files("q/absent", "model-q4_k_m.gguf", "Q4_K_M");
        assert_eq!(files, vec!["model-q4_k_m.gguf".to_string()]);
        let fb = crate::resolve_variant_files("q/absent", "model-q4_k_m.gguf", "Q9_NOPE");
        assert_eq!(fb, vec!["model-q4_k_m.gguf".to_string()]); // fallback to spec.file
    }
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
