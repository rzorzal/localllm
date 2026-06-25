use std::sync::Arc;

use clap::Parser;
use localllm::{config::Config, engine::Engine, server::router};
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
    tracing::info!("loading model {} (force_cpu={})…", cfg.model_id, cfg.force_cpu);

    let engine: Arc<dyn Generator> = Arc::new(Engine::load(&cfg.engine_config()).await?);
    let app = router(engine, cfg.model_id.clone());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
    tracing::info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
