/// macOS menu-bar tray integration.
///
/// This module is only compiled on macOS and only active when the binary is
/// launched with `--tray`.
///
/// ## Threading model
///
/// On macOS, AppKit (and therefore tray-icon which uses AppKit via objc2)
/// **must** run on the main thread. We therefore:
///
///   1. Spawn the axum server on a dedicated `std::thread` that builds its own
///      `tokio::Runtime` and calls `localllm::run_server()` to completion.
///   2. Build the tray icon and menu on the first event after the event loop
///      starts (`NewEvents(StartCause::Init)`) per tray-icon docs.
///   3. Run `event_loop.run(...)` on the main thread (blocks forever).
///      On "Quit", call `std::process::exit(0)`.
///
/// ## Menu events on macOS
///
/// On macOS, tray-icon fires menu events through its own crossbeam channel
/// (`MenuEvent::receiver()`), NOT as variants of the tao `Event` enum.
/// We poll this channel on `MainEventsCleared` and `ResumeTimeReached` ticks
/// via a short `WaitUntil` timeout so the event loop stays responsive.
///
/// ## Status text strategy
///
/// The status MenuItem is set to "localllm — running on http://127.0.0.1:PORT"
/// immediately (static). Port is known from `Config::port` at launch time, so
/// no cross-thread update is needed. If the server fails to bind, it logs the
/// error; the tray still shows the expected URL (acceptable for a local tool).

#[cfg(target_os = "macos")]
pub mod macos {
    use std::time::{Duration, Instant};

    use tray_icon::{
        TrayIconBuilder,
        menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
        Icon,
    };
    use tao::event::{Event, StartCause};
    use tao::event_loop::{ControlFlow, EventLoopBuilder};

    use crate::config::Config;

    // -----------------------------------------------------------------------
    // Icon generation
    // -----------------------------------------------------------------------

    /// Generate a 32×32 RGBA icon as a filled rounded square in a teal/accent
    /// colour. No external asset files required.
    fn make_icon() -> Icon {
        const SIZE: u32 = 32;
        const R: i32 = 6; // corner radius
        let cx = SIZE as i32 / 2;
        let cy = SIZE as i32 / 2;
        let hw = SIZE as i32 / 2 - 1; // half-width (rect goes -(hw)..(hw) from center)

        // Teal accent colour: #2DD4BF (r=45, g=212, b=191, a=255)
        const FR: u8 = 45;
        const FG: u8 = 212;
        const FB: u8 = 191;

        let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];

        for y in 0..SIZE as i32 {
            for x in 0..SIZE as i32 {
                let dx = (x - cx).abs();
                let dy = (y - cy).abs();
                // Is this pixel inside the rounded rectangle?
                let inside = if dx <= hw - R && dy <= hw {
                    true
                } else if dy <= hw - R && dx <= hw {
                    true
                } else {
                    // Check corner circle
                    let cdx = dx - (hw - R);
                    let cdy = dy - (hw - R);
                    cdx >= 0 && cdy >= 0 && cdx * cdx + cdy * cdy <= R * R
                };
                let i = ((y * SIZE as i32 + x) * 4) as usize;
                if inside {
                    rgba[i] = FR;
                    rgba[i + 1] = FG;
                    rgba[i + 2] = FB;
                    rgba[i + 3] = 255;
                }
                // else transparent (already 0)
            }
        }

        Icon::from_rgba(rgba, SIZE, SIZE).expect("failed to build tray icon")
    }

    // -----------------------------------------------------------------------
    // Tray entry point
    // -----------------------------------------------------------------------

    /// Run the server in a background thread and drive the macOS event loop +
    /// tray icon on the main thread.
    ///
    /// This function is `-> !` — it blocks forever (exits via
    /// `std::process::exit(0)` on Quit, or the OS kills the process).
    pub fn run_tray(cfg: Config) -> ! {
        let port = cfg.port;
        let url = format!("http://127.0.0.1:{port}");

        // Capture display info BEFORE moving cfg into the server thread.
        let info_model = cfg.model_id.clone();
        let info_ctx = cfg.ctx_len;
        let info_kv = format!("{:?}", cfg.kv_type);
        let info_backend = format!("{:?}", cfg.backend);

        // ---- Spawn server on a background thread with its own tokio runtime ----
        std::thread::Builder::new()
            .name("localllm-server".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Runtime::new()
                    .expect("failed to build server tokio runtime");
                if let Err(e) = rt.block_on(crate::run_server(cfg)) {
                    tracing::error!("server exited with error: {e:#}");
                    std::process::exit(1);
                }
            })
            .expect("failed to spawn server thread");

        // ---- Build tao event loop on the main thread ----
        let event_loop = EventLoopBuilder::<()>::new().build();

        // The TrayIcon MUST stay alive for the icon to remain visible.
        // We keep it in a Vec (rather than Option) to avoid the Rust
        // "value never read" lint on assignment while still preventing drop.
        let mut _tray_keeper: Vec<tray_icon::TrayIcon> = Vec::new();
        let url_for_tray = url.clone();
        // Short model name (drop the "Owner/" prefix) for a tidy menu line.
        let model_short = info_model.rsplit('/').next().unwrap_or(&info_model).to_string();
        // Set in Init; compared against menu events so only Quit exits.
        let mut quit_id: Option<tray_icon::menu::MenuId> = None;

        // Poll interval for the menu-event channel.
        let poll_interval = Duration::from_millis(100);

        event_loop.run(move |event, _, control_flow| {
            // Wake up periodically to poll the menu-event channel.
            *control_flow = ControlFlow::WaitUntil(Instant::now() + poll_interval);

            match event {
                Event::NewEvents(StartCause::Init) => {
                    // Safe to create the tray icon now that NSApp has started.
                    let icon = make_icon();

                    let menu = Menu::new();
                    // Title + status, then info lines (all disabled/informational),
                    // a separator, and the clickable Quit item.
                    let title = MenuItem::new("localllm — local LLM server", false, None);
                    let status = MenuItem::new("● Running", false, None);
                    let url_line = MenuItem::new(format!("URL:    {url_for_tray}"), false, None);
                    let model_line = MenuItem::new(format!("Model:  {model_short}"), false, None);
                    let ctx_line = MenuItem::new(format!("Context: {info_ctx} tokens"), false, None);
                    let kv_line = MenuItem::new(format!("KV cache: {info_kv}"), false, None);
                    let backend_line = MenuItem::new(format!("Backend: {info_backend}"), false, None);
                    let separator = PredefinedMenuItem::separator();
                    let quit_item = MenuItem::new("Quit localllm", true, None);
                    quit_id = Some(quit_item.id().clone());

                    menu.append(&title).expect("append title");
                    menu.append(&status).expect("append status");
                    menu.append(&PredefinedMenuItem::separator()).expect("sep");
                    menu.append(&url_line).expect("append url");
                    menu.append(&model_line).expect("append model");
                    menu.append(&ctx_line).expect("append ctx");
                    menu.append(&kv_line).expect("append kv");
                    menu.append(&backend_line).expect("append backend");
                    menu.append(&separator).expect("append separator");
                    menu.append(&quit_item).expect("append quit item");

                    _tray_keeper.push(
                        TrayIconBuilder::new()
                            .with_menu(Box::new(menu))
                            .with_tooltip(format!("localllm — {url_for_tray}"))
                            .with_icon(icon)
                            .build()
                            .expect("failed to create tray icon"),
                    );

                    tracing::info!("tray icon created; server loading in background…");
                }

                // Poll the tray-icon menu-event channel on every wake-up.
                // On macOS tray-icon fires events through crossbeam, not via
                // tao Event enum variants, so we must poll explicitly.
                Event::MainEventsCleared
                | Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                    while let Ok(menu_event) = MenuEvent::receiver().try_recv() {
                        if quit_id.as_ref() == Some(&menu_event.id) {
                            tracing::info!("quit requested via tray menu — shutting down");
                            std::process::exit(0);
                        }
                    }
                }

                _ => {}
            }
        })
    }
}
