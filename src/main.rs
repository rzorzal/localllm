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

    /// Debug/build: render the app icon to a PNG at the given path and exit.
    /// Used by scripts/build-app.sh to generate the .icns. (macOS only.)
    #[arg(long)]
    export_icon: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    use tracing_subscriber::fmt::writer::MakeWriterExt;

    // Respect RUST_LOG when set; otherwise default to info for our crate.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("localllm=info,mistralrs_core=info"));

    // Always mirror logs to a file so the tray app (which has no terminal) can
    // be inspected. Path overridable via LOCALLLM_LOG; default /tmp/localllm.log.
    let log_path = std::env::var("LOCALLLM_LOG")
        .unwrap_or_else(|_| "/tmp/localllm.log".to_string());
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path);

    // Logs go to stderr (clean: stdout reserved for --dump-prompt) AND the file.
    match log_file {
        Ok(f) => {
            let file_writer = move || f.try_clone().expect("clone log file handle");
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(std::io::stderr.and(file_writer))
                .init();
            eprintln!("logging to {log_path}");
        }
        Err(_) => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .init();
        }
    }

    let args = MainArgs::parse();

    // --- Short-circuit flags that never start the server ---

    #[cfg(target_os = "macos")]
    if let Some(path) = &args.export_icon {
        let size: u32 = 1024;
        let rgba = localllm::tray::macos::render_icon_rgba(size);
        let file = std::fs::File::create(path)?;
        let w = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, size, size);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&rgba)?;
        eprintln!("wrote icon PNG to {}", path.display());
        return Ok(());
    }

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
    // Enable tray mode if --tray is passed OR we were launched from a .app
    // bundle (launchd sets __CFBundleIdentifier). Launching as the bundle's
    // direct executable (not via a shell wrapper) is required for the
    // NSStatusItem to register with the WindowServer.
    #[cfg(target_os = "macos")]
    {
        let from_bundle = std::env::var_os("__CFBundleIdentifier").is_some();
        if args.tray || from_bundle {
            // run_tray() is `-> !` (exits via process::exit on Quit).
            localllm::tray::macos::run_tray(args.config);
        }
    }

    // --- Headless mode (default) ---
    // Also the fallback on non-macOS even if --tray was passed.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(localllm::run_server(args.config))
}
