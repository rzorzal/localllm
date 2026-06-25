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

    /// Debug: load the model, build the prompt for the given Anthropic request
    /// JSON file, print it to stdout, and exit (no inference). For inspecting
    /// what actually gets sent to the model.
    #[arg(long)]
    dump_prompt: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Respect RUST_LOG when set; otherwise default to info for our crate.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("localllm=info,mistralrs_core=info"));
    // Logs go to stderr so stdout stays clean (e.g. --dump-prompt output).
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();

    let args = MainArgs::parse();

    if let Some(path) = &args.dump_prompt {
        let cfg = &args.config;
        let prompt = localllm::engine_llama::dump_prompt(&cfg.model_id, &cfg.gguf_files, path).await?;
        println!("{prompt}");
        return Ok(());
    }

    if args.llama_smoke {
        return localllm::engine_llama::run_smoke_test().await;
    }

    let cfg = args.config;
    tracing::info!("loading model {} (backend={:?})…", cfg.model_id, cfg.backend);

    let engine: Arc<dyn Generator> = match cfg.backend {
        Backend::Llama => {
            let kv_cache_type = cfg.llama_kv_cache_type();
            let kv_cache_dir = cfg.resolved_kv_cache_dir();
            tracing::info!("KV cache type: --kv-type={:?}", cfg.kv_type);
            tracing::info!("KV persist dir: {:?} (no-persist={})", kv_cache_dir, cfg.no_kv_persist);
            Arc::new(LlamaEngine::load(&cfg.model_id, &cfg.gguf_files, cfg.ctx_len, kv_cache_type, kv_cache_dir).await?)
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
