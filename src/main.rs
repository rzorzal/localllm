use std::sync::Arc;

use clap::Parser;
use localllm::{config::Config, engine::Engine, server::router};
use localllm::server::Generator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("localllm=info".parse()?)
                .add_directive("mistralrs_core=info".parse()?),
        )
        .init();

    let cfg = Config::parse();
    tracing::info!("loading model {} (force_cpu={})…", cfg.model_id, cfg.force_cpu);

    let engine: Arc<dyn Generator> = Arc::new(Engine::load(&cfg.engine_config()).await?);
    let app = router(engine, cfg.model_id.clone());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], cfg.port));
    tracing::info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
