use clap::Parser;

#[derive(clap::Parser, Debug)]
struct MainArgs {
    #[clap(flatten)]
    config: localllm::config::Config,

    /// Run the LlamaEngine smoke test (load model, call get_weather tool) and exit.
    #[arg(long, default_value_t = false)]
    llama_smoke: bool,

    /// Debug: load the model, build the prompt for the given Anthropic request
    /// JSON file, print it to stdout, and exit (no inference). For inspecting
    /// what actually gets sent to the model.
    #[arg(long)]
    dump_prompt: Option<std::path::PathBuf>,

    /// Run as a macOS menu-bar (tray) application: no Dock icon, no terminal
    /// window. The server starts in a background thread; the main thread drives
    /// the AppKit event loop and displays a status-bar icon with a menu
    /// showing the server URL and a "Quit" item.
    ///
    /// When running inside a .app bundle (LSUIElement=true), pass this flag
    /// via the launcher script so the app behaves as a background agent.
    ///
    /// NOTE: Only functional on macOS; on other platforms it is accepted but
    /// silently falls back to headless mode.
    #[arg(long, default_value_t = false)]
    tray: bool,
}

fn main() -> anyhow::Result<()> {
    // Respect RUST_LOG when set; otherwise default to info for our crate.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("localllm=info,mistralrs_core=info"));
    // Logs go to stderr so stdout stays clean (e.g. --dump-prompt output).
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();

    let args = MainArgs::parse();

    // --- Short-circuit flags that never start the server ---

    if let Some(path) = &args.dump_prompt {
        let cfg = &args.config;
        let prompt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(localllm::engine_llama::dump_prompt(&cfg.model_id, &cfg.gguf_files, path))?;
        println!("{prompt}");
        return Ok(());
    }

    if args.llama_smoke {
        return tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(localllm::engine_llama::run_smoke_test());
    }

    // --- Tray mode (macOS only): server on background thread,
    //     AppKit event loop on the main thread. ---
    #[cfg(target_os = "macos")]
    if args.tray {
        // run_tray() is `-> !` (exits via process::exit on Quit).
        localllm::tray::macos::run_tray(args.config);
    }

    // --- Headless mode (default) ---
    // Also the fallback on non-macOS even if --tray was passed.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(localllm::run_server(args.config))
}
