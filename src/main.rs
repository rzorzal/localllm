use std::sync::Arc;

use clap::Parser;
use localllm::{config::{Backend, Config}, engine::Engine, engine_llama::LlamaEngine, server::router};
use localllm::server::Generator;

#[derive(clap::Parser, Debug)]
struct MainArgs {
    #[clap(flatten)]
    config: Config,

    /// Run the LlamaEngine smoke test (load model, call get_weather tool) and exit.
    #[arg(long, default_value_t = false)]
    llama_smoke: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("localllm=info".parse()?)
                .add_directive("mistralrs_core=info".parse()?),
        )
        .init();

    let args = MainArgs::parse();

    if args.llama_smoke {
        return localllm::engine_llama::run_smoke_test().await;
    }

    let cfg = args.config;
    tracing::info!("loading model {} (backend={:?})…", cfg.model_id, cfg.backend);

    let engine: Arc<dyn Generator> = match cfg.backend {
        Backend::Llama => {
            let kv_cache_type = cfg.llama_kv_cache_type();
            tracing::info!("KV cache type: --kv-type={:?}", cfg.kv_type);
            Arc::new(LlamaEngine::load(&cfg.model_id, &cfg.gguf_files, cfg.ctx_len, kv_cache_type).await?)
        }
        Backend::Mistralrs => Arc::new(Engine::load(&cfg.engine_config()).await?),
    };
    let app = router(engine, cfg.model_id.clone());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
    tracing::info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
