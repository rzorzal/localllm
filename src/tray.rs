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
    use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};

    use crate::config::Config;

    // -----------------------------------------------------------------------
    // Icon generation
    // -----------------------------------------------------------------------

    /// Copy text to the macOS clipboard via `pbcopy`.
    fn copy_to_clipboard(text: &str) {
        use std::io::Write;
        use std::process::{Command, Stdio};
        if let Ok(mut child) = Command::new("pbcopy").stdin(Stdio::piped()).spawn() {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
            tracing::info!("copied to clipboard: {text}");
        }
    }

    /// Open the log file in the user's editor: Zed if installed, else the
    /// default text app via `open -t`.
    fn open_log_file(path: &str) {
        use std::process::Command;
        let zed = Command::new("zed").arg(path).spawn();
        if zed.is_err() {
            let _ = Command::new("open").arg("-t").arg(path).spawn();
        }
        tracing::info!("opening log file {path}");
    }

    /// Render the app icon as `size`×`size` RGBA: a rounded square with a
    /// teal→indigo diagonal gradient and a white minimalist chat-bubble glyph
    /// with a spark. Shared by the tray (32px) and the .app icon export.
    pub fn render_icon_rgba(size: u32) -> Vec<u8> {
        let s = size as f32;
        let r = s * 0.22; // corner radius
        let mut rgba = vec![0u8; (size * size * 4) as usize];

        // gradient endpoints: teal #2DD4BF → indigo #6366F1
        let (r0, g0, b0) = (45.0f32, 212.0, 191.0);
        let (r1, g1, b1) = (99.0f32, 102.0, 241.0);

        // chat bubble geometry (centered, with a tail)
        let bw = s * 0.52; // bubble width
        let bh = s * 0.40; // bubble height
        let bx = (s - bw) / 2.0;
        let by = s * 0.26;
        let br = s * 0.12; // bubble corner radius

        let inside_round = |px: f32, py: f32, x: f32, y: f32, w: f32, h: f32, rad: f32| -> bool {
            let dx = (px - (x + rad)).max(0.0).max((x + w - rad) - px).max(0.0);
            // compute distance to the rounded rect
            let cx = px.clamp(x + rad, x + w - rad);
            let cy = py.clamp(y + rad, y + h - rad);
            let _ = dx;
            let ddx = px - cx;
            let ddy = py - cy;
            px >= x && px <= x + w && py >= y && py <= y + h
                && ddx * ddx + ddy * ddy <= rad * rad + 0.5
        };

        for y in 0..size {
            for x in 0..size {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                let i = ((y * size + x) * 4) as usize;

                // rounded-square mask for the badge
                let in_badge = inside_round(px, py, 0.0, 0.0, s, s, r);
                if !in_badge {
                    continue; // transparent
                }
                // diagonal gradient factor 0..1
                let t = ((px + py) / (2.0 * s)).clamp(0.0, 1.0);
                let mut cr = r0 + (r1 - r0) * t;
                let mut cg = g0 + (g1 - g0) * t;
                let mut cb = b0 + (b1 - b0) * t;

                // white chat bubble
                let in_bubble = inside_round(px, py, bx, by, bw, bh, br);
                // bubble tail (small triangle at bottom-left of bubble)
                let tail_x = bx + bw * 0.28;
                let tail_y = by + bh;
                let in_tail = py >= tail_y - 0.5
                    && py <= tail_y + s * 0.12
                    && px >= tail_x
                    && (px - tail_x) <= (tail_y + s * 0.12 - py);
                if in_bubble || in_tail {
                    cr = 255.0; cg = 255.0; cb = 255.0;
                }
                // spark: three dots inside the bubble (suggesting chat/AI)
                let dot_y = by + bh * 0.5;
                for k in 0..3 {
                    let dx = bx + bw * (0.30 + 0.20 * k as f32);
                    let dd = (px - dx) * (px - dx) + (py - dot_y) * (py - dot_y);
                    if dd <= (s * 0.035) * (s * 0.035) {
                        // dots tinted with the gradient colour for contrast on white
                        cr = r0 + (r1 - r0) * t;
                        cg = g0 + (g1 - g0) * t;
                        cb = b0 + (b1 - b0) * t;
                    }
                }

                rgba[i] = cr as u8;
                rgba[i + 1] = cg as u8;
                rgba[i + 2] = cb as u8;
                rgba[i + 3] = 255;
            }
        }
        rgba
    }

    fn make_icon() -> Icon {
        const SIZE: u32 = 32;
        Icon::from_rgba(render_icon_rgba(SIZE), SIZE, SIZE).expect("failed to build tray icon")
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

        // Shared flag flipped to true once the server is actually serving.
        let ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready_for_server = ready.clone();

        // ---- Spawn server on a background thread with its own tokio runtime ----
        std::thread::Builder::new()
            .name("localllm-server".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Runtime::new()
                    .expect("failed to build server tokio runtime");
                if let Err(e) = rt.block_on(crate::run_server_with_ready(cfg, Some(ready_for_server))) {
                    tracing::error!("server exited with error: {e:#}");
                    std::process::exit(1);
                }
            })
            .expect("failed to spawn server thread");

        // ---- Build tao event loop on the main thread ----
        // ActivationPolicy::Accessory = menu-bar agent: NO Dock icon, and the
        // status-bar item shows reliably. Without this, the app launches as a
        // regular app (Dock icon) and the tray item often fails to appear.
        let mut event_loop = EventLoopBuilder::<()>::new().build();
        event_loop.set_activation_policy(ActivationPolicy::Accessory);

        // The TrayIcon MUST stay alive for the icon to remain visible.
        // We keep it in a Vec (rather than Option) to avoid the Rust
        // "value never read" lint on assignment while still preventing drop.
        let mut _tray_keeper: Vec<tray_icon::TrayIcon> = Vec::new();
        let url_for_tray = url.clone();
        // Short model name (drop the "Owner/" prefix) for a tidy menu line.
        let model_short = info_model.rsplit('/').next().unwrap_or(&info_model).to_string();
        // Set in Init; compared against menu events to dispatch clicks.
        let mut quit_id: Option<tray_icon::menu::MenuId> = None;
        let mut logs_id: Option<tray_icon::menu::MenuId> = None;
        let mut url_id: Option<tray_icon::menu::MenuId> = None;
        // Kept so we can flip the status text to "running" once the server is up.
        let mut status_handle: Option<MenuItem> = None;
        let mut shown_running = false;
        let log_path = std::env::var("LOCALLLM_LOG")
            .unwrap_or_else(|_| "/tmp/localllm.log".to_string());

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
                    // Starts as loading (yellow); flips to green "Running" once
                    // the server thread signals it is actually serving.
                    let status = MenuItem::new("🟡 Loading model…", false, None);
                    status_handle = Some(status.clone());
                    // URL is clickable → copies to clipboard.
                    let url_line = MenuItem::new(format!("URL:    {url_for_tray}  (click to copy)"), true, None);
                    url_id = Some(url_line.id().clone());
                    let model_line = MenuItem::new(format!("Model:  {model_short}"), false, None);
                    let ctx_line = MenuItem::new(format!("Context: {info_ctx} tokens"), false, None);
                    let kv_line = MenuItem::new(format!("KV cache: {info_kv}"), false, None);
                    let backend_line = MenuItem::new(format!("Backend: {info_backend}"), false, None);
                    let separator = PredefinedMenuItem::separator();
                    let logs_item = MenuItem::new("Open Logs", true, None);
                    logs_id = Some(logs_item.id().clone());
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
                    menu.append(&logs_item).expect("append logs item");
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
                    // Flip status to green once the server is actually serving.
                    if !shown_running
                        && ready.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        if let Some(s) = &status_handle {
                            s.set_text("🟢 Running");
                        }
                        shown_running = true;
                    }

                    while let Ok(menu_event) = MenuEvent::receiver().try_recv() {
                        if quit_id.as_ref() == Some(&menu_event.id) {
                            tracing::info!("quit requested via tray menu — shutting down");
                            std::process::exit(0);
                        } else if logs_id.as_ref() == Some(&menu_event.id) {
                            open_log_file(&log_path);
                        } else if url_id.as_ref() == Some(&menu_event.id) {
                            copy_to_clipboard(&url_for_tray);
                        }
                    }
                }

                _ => {}
            }
        })
    }
}
