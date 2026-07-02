//! Cross-platform menu-bar tray integration.
//!
//! Active when the binary is launched with `--tray` (or from a macOS .app
//! bundle). Compiles on all platforms; only macOS has a working runtime today.
//!
//! ## Threading model
//!
//! On macOS, AppKit (and therefore tray-icon which uses AppKit via objc2)
//! **must** run on the main thread. We therefore:
//!
//!   1. Spawn the axum server on a dedicated `std::thread` that builds its own
//!      `tokio::Runtime` and calls `localllm::run_server_with_ready_policy_token`
//!      to completion.
//!   2. Build the tray icon and menu on the first event after the event loop
//!      starts (`NewEvents(StartCause::Init)`) per tray-icon docs.
//!   3. Run `event_loop.run(...)` on the main thread (blocks forever).
//!      On "Quit", call `std::process::exit(0)`.
//!
//! ## Menu events on macOS
//!
//! On macOS, tray-icon fires menu events through its own crossbeam channel
//! (`MenuEvent::receiver()`), NOT as variants of the tao `Event` enum.
//! We poll this channel on `MainEventsCleared` and `ResumeTimeReached` ticks
//! via a short `WaitUntil` timeout so the event loop stays responsive.
//!
//! ## Status text strategy
//!
//! The status MenuItem is set to "localllm — running on http://127.0.0.1:PORT"
//! immediately (static). Port is known from `Config::port` at launch time, so
//! no cross-thread update is needed. If the server fails to bind, it logs the
//! error; the tray still shows the expected URL (acceptable for a local tool).

use std::time::{Duration, Instant};

use tray_icon::{
    TrayIconBuilder,
    menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu},
    Icon,
};
use std::sync::{Arc, RwLock};
use crate::route::{Profile, RoutingPolicy};
use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder};

use crate::config::Config;

// ---------------------------------------------------------------------------
// Platform abstraction: clipboard + open
// ---------------------------------------------------------------------------

mod platform {
    /// Copy text to the OS clipboard, best-effort.
    pub fn copy_to_clipboard(text: &str) {
        use std::io::Write;
        use std::process::{Command, Stdio};
        #[cfg(target_os = "macos")]
        let mut cmd = Command::new("pbcopy");
        #[cfg(target_os = "windows")]
        let mut cmd = Command::new("clip");
        #[cfg(all(unix, not(target_os = "macos")))]
        let mut cmd = {
            // prefer wl-copy (Wayland), else xclip (X11)
            if Command::new("wl-copy")
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok()
            {
                Command::new("wl-copy")
            } else {
                let mut c = Command::new("xclip");
                c.args(["-selection", "clipboard"]);
                c
            }
        };
        if let Ok(mut child) = cmd.stdin(Stdio::piped()).spawn() {
            if let Some(si) = child.stdin.as_mut() {
                let _ = si.write_all(text.as_bytes());
            }
            let _ = child.wait();
        }
    }

    /// Open a file/path in the OS default handler, best-effort.
    pub fn open_path(path: &str) {
        use std::process::Command;
        #[cfg(target_os = "macos")]
        {
            let _ = Command::new("open").arg("-t").arg(path).spawn();
        }
        #[cfg(target_os = "windows")]
        {
            let _ = Command::new("cmd").args(["/C", "start", "", path]).spawn();
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            let _ = Command::new("xdg-open").arg(path).spawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Icon generation
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Model Manager webview window
// ---------------------------------------------------------------------------

mod window {
    //! The Model Manager webview window (tao + wry). Single-instance, opened from
    //! the tray. Cross-platform (macOS WKWebView / Linux WebKitGTK / Windows WebView2).
    pub(crate) struct ModelManagerWindow {
        pub window: tao::window::Window,
        webview: wry::WebView,
    }

    impl ModelManagerWindow {
        /// Build the window + webview pointed at the local /manager SPA, injecting
        /// the admin token in-memory.
        pub fn open<T: 'static>(
            target: &tao::event_loop::EventLoopWindowTarget<T>,
            port: u16,
            admin_token: &str,
        ) -> anyhow::Result<Self> {
            use tao::dpi::LogicalSize;
            use tao::window::WindowBuilder;
            let window = WindowBuilder::new()
                .with_title("localllm \u{2014} Config")
                .with_inner_size(LogicalSize::new(900.0_f64, 640.0_f64))
                .build(target)?;
            let url = format!("http://127.0.0.1:{port}/manager");
            let webview = wry::WebViewBuilder::new(&window)
                .with_url(&url)
                .with_initialization_script(&crate::server::manager_init_script(admin_token))
                .build()?;
            Ok(Self { window, webview })
        }

        /// Navigate the SPA to a hash route (e.g. "#/dashboard") without a reload.
        pub fn navigate(&self, route: &str) {
            let js = format!("location.hash = {:?};", route);
            let _ = self.webview.evaluate_script(&js);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::model_menu_label;
    use crate::model_manager::ModelSpec;

    #[test]
    fn label_strips_repo_owner_and_appends_file() {
        let spec = ModelSpec {
            repo: "Qwen/Qwen2.5-7B-Instruct-GGUF".into(),
            file: "qwen2.5-7b-instruct-q4_k_m.gguf".into(),
            quant: None,
        };
        assert_eq!(
            model_menu_label(&spec),
            "Model:  Qwen2.5-7B-Instruct-GGUF / qwen2.5-7b-instruct-q4_k_m.gguf"
        );
    }

    #[test]
    fn label_handles_repo_without_owner() {
        let spec = ModelSpec { repo: "local".into(), file: "m.gguf".into(), quant: None };
        assert_eq!(model_menu_label(&spec), "Model:  local / m.gguf");
    }

    use super::status_menu_label;
    use crate::model_manager::{SwitchPhase, SwitchStatus};

    fn status(state: &str, phase: SwitchPhase, progress: u8) -> SwitchStatus {
        SwitchStatus {
            state: state.into(),
            current: ModelSpec { repo: "r".into(), file: "f".into(), quant: None },
            target: None,
            phase,
            progress,
            error: None,
        }
    }

    #[test]
    fn status_ready_is_running() {
        assert_eq!(status_menu_label(&status("ready", SwitchPhase::Idle, 0)), "🟢 Running");
    }

    #[test]
    fn status_switching_shows_phase_and_percent() {
        assert_eq!(
            status_menu_label(&status("switching", SwitchPhase::Downloading, 42)),
            "🟡 Switching… downloading 42%"
        );
    }

    #[test]
    fn status_error_is_failed() {
        assert_eq!(status_menu_label(&status("error", SwitchPhase::Idle, 0)), "🔴 Switch failed");
    }
}

/// Format the tray "Model:" line from the active model spec. Mirrors the
/// Manager window header (`repo-basename / file`) so the two never disagree.
fn model_menu_label(spec: &crate::model_manager::ModelSpec) -> String {
    let short = spec.repo.rsplit('/').next().unwrap_or(&spec.repo);
    format!("Model:  {short} / {}", spec.file)
}

/// Format the tray status line from live manager state: green Running when idle,
/// yellow Switching with phase+percent mid-swap, red on a failed switch.
fn status_menu_label(s: &crate::model_manager::SwitchStatus) -> String {
    use crate::model_manager::SwitchPhase;
    match s.state.as_str() {
        "switching" => {
            let phase = match s.phase {
                SwitchPhase::Draining => "draining",
                SwitchPhase::Downloading => "downloading",
                SwitchPhase::Loading => "loading",
                SwitchPhase::Idle => "switching",
            };
            format!("🟡 Switching… {phase} {}%", s.progress)
        }
        "error" => "🔴 Switch failed".to_string(),
        _ => "🟢 Running".to_string(),
    }
}

/// Format the tray "Wired:" sub-line from persisted integration state.
fn wired_label(state: &crate::settings::IntegrationState) -> String {
    if !state.enabled {
        return "Apps: direct to provider".to_string();
    }
    if state.priors.is_empty() {
        return "Wired: none (no client configs found)".to_string();
    }
    // Map known ids to display names for the line.
    let names: Vec<&str> = state
        .priors
        .keys()
        .map(|id| match id.as_str() {
            "claude-code" => "Claude Code",
            "codex" => "Codex",
            other => other,
        })
        .collect();
    format!("Wired: {}", names.join(", "))
}

// ---------------------------------------------------------------------------
// Tray entry point
// ---------------------------------------------------------------------------

/// Run the server in a background thread and drive the event loop +
/// tray icon on the main thread.
///
/// This function is `-> !` — it blocks forever (exits via
/// `std::process::exit(0)` on Quit, or the OS kills the process).
pub fn run_tray(cfg: Config, admin_token: std::sync::Arc<str>) -> ! {
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

    // Resolve the startup profile (CLI > saved > default) and build the
    // policy the menu and server share. The menu mutates it live.
    let initial_profile = crate::settings::resolve_profile(cfg.profile);
    let policy: Arc<RwLock<RoutingPolicy>> =
        Arc::new(RwLock::new(initial_profile.policy()));
    let policy_for_server = policy.clone();

    // Clone the token for the server thread; keep the original for Task 3's window.
    let admin_token_for_server = admin_token.clone();

    // Shared slot the server fills with the live ModelManager once built, so the
    // tray can poll the active model and keep its "Model:" line in sync after a
    // hot-swap (the line is otherwise static from launch config).
    let manager_slot: Arc<std::sync::OnceLock<Arc<crate::model_manager::ModelManager>>> =
        Arc::new(std::sync::OnceLock::new());
    let manager_slot_for_server = manager_slot.clone();

    // ---- Spawn server on a background thread with its own tokio runtime ----
    std::thread::Builder::new()
        .name("localllm-server".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Runtime::new()
                .expect("failed to build server tokio runtime");
            if let Err(e) = rt.block_on(crate::run_server_with_ready_policy_token(
                cfg,
                Some(ready_for_server),
                policy_for_server,
                admin_token_for_server,
                Some(manager_slot_for_server),
            )) {
                tracing::error!("server exited with error: {e:#}");
                std::process::exit(1);
            }
        })
        .expect("failed to spawn server thread");

    // ---- Build tao event loop on the main thread ----
    let mut event_loop = EventLoopBuilder::<()>::new().build();
    // On macOS: ActivationPolicy::Accessory = menu-bar agent (no Dock icon).
    // Without this, the app launches as a regular app and the tray item often
    // fails to appear.
    #[cfg(target_os = "macos")]
    {
        use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
        event_loop.set_activation_policy(ActivationPolicy::Accessory);
    }

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
    // Kept so we can live-update the status text from the server/switch state.
    // last_status avoids redundant set_text on every poll tick.
    let mut status_handle: Option<MenuItem> = None;
    let mut last_status: Option<String> = None;
    // Kept so we can live-update the "Model:" line after a hot-swap; last_model
    // avoids redundant set_text on every poll tick.
    let mut model_handle: Option<MenuItem> = None;
    let mut last_model: Option<crate::model_manager::ModelSpec> = None;
    // Routing submenu state: (menu id, profile, check item) for each profile,
    // populated in Init and used to dispatch clicks + re-check on change.
    let mut routing_items: Vec<(tray_icon::menu::MenuId, Profile, CheckMenuItem)> = Vec::new();
    let mut routing_status: Option<MenuItem> = None;
    let log_path = std::env::var("LOCALLLM_LOG")
        .unwrap_or_else(|_| "/tmp/localllm.log".to_string());

    // Config window state: None until first open, then single-instance. The
    // Config submenu items each open/show it and navigate to their SPA route.
    // admin_token (function param) is captured by the closure for window injection.
    let mut config_models_id: Option<tray_icon::menu::MenuId> = None;
    let mut config_tools_id: Option<tray_icon::menu::MenuId> = None;
    let mut config_dash_id: Option<tray_icon::menu::MenuId> = None;
    let mut manager_window: Option<window::ModelManagerWindow> = None;

    // Read-only wired sub-line: reflects the integration state toggled from the
    // Config page. last_wired avoids redundant set_text on every poll tick.
    let mut wired_handle: Option<MenuItem> = None;
    let mut last_wired: Option<String> = None;

    // Poll interval for the menu-event channel.
    let poll_interval = Duration::from_millis(100);

    event_loop.run(move |event, target, control_flow| {
        // Wake up periodically to poll the menu-event channel.
        *control_flow = ControlFlow::WaitUntil(Instant::now() + poll_interval);

        match event {
            Event::NewEvents(StartCause::Init) => {
                // Safe to create the tray icon now that the event loop has started.
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
                model_handle = Some(model_line.clone());
                let ctx_line = MenuItem::new(format!("Context: {info_ctx} tokens"), false, None);
                let kv_line = MenuItem::new(format!("KV cache: {info_kv}"), false, None);
                let backend_line = MenuItem::new(format!("Backend: {info_backend}"), false, None);
                let logs_item = MenuItem::new("Open Logs", true, None);
                logs_id = Some(logs_item.id().clone());
                let config_submenu = Submenu::new("Config", true);
                let cfg_models = MenuItem::new("Models", true, None);
                let cfg_tools = MenuItem::new("Tools", true, None);
                let cfg_dash = MenuItem::new("Dashboard", true, None);
                config_models_id = Some(cfg_models.id().clone());
                config_tools_id = Some(cfg_tools.id().clone());
                config_dash_id = Some(cfg_dash.id().clone());
                config_submenu.append(&cfg_models).expect("append config models");
                config_submenu.append(&cfg_tools).expect("append config tools");
                config_submenu.append(&cfg_dash).expect("append config dashboard");
                let quit_item = MenuItem::new("Quit localllm", true, None);
                quit_id = Some(quit_item.id().clone());

                // Routing profile selector (live; persists on click).
                let routing_submenu = Submenu::new("Routing", true);
                for p in Profile::ALL {
                    let item = CheckMenuItem::new(
                        p.label(),
                        true,
                        p == initial_profile,
                        None,
                    );
                    routing_items.push((item.id().clone(), p, item.clone()));
                    routing_submenu.append(&item).expect("append routing item");
                }
                let routing_line = MenuItem::new(
                    format!("Routing: {}", initial_profile.label()),
                    false,
                    None,
                );
                routing_status = Some(routing_line.clone());

                let init_state = crate::settings::load_integrations();
                let wired_line = MenuItem::new(
                    wired_label(&init_state),
                    false,
                    None,
                );
                wired_handle = Some(wired_line.clone());

                menu.append(&title).expect("append title");
                menu.append(&status).expect("append status");
                menu.append(&PredefinedMenuItem::separator()).expect("sep");
                menu.append(&url_line).expect("append url");
                menu.append(&model_line).expect("append model");
                menu.append(&ctx_line).expect("append ctx");
                menu.append(&kv_line).expect("append kv");
                menu.append(&backend_line).expect("append backend");
                menu.append(&PredefinedMenuItem::separator()).expect("sep2");
                menu.append(&routing_line).expect("append routing line");
                menu.append(&routing_submenu).expect("append routing submenu");
                menu.append(&PredefinedMenuItem::separator()).expect("append separator");
                menu.append(&wired_line).expect("append wired line");
                menu.append(&PredefinedMenuItem::separator()).expect("append separator2");
                menu.append(&logs_item).expect("append logs item");
                menu.append(&config_submenu).expect("append config submenu");
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
                // Until the server is bound, keep the launch-time "Loading model…"
                // line. Once ready, drive both the status and the model line from
                // the live manager so startup, hot-swaps, and switch failures all
                // show in the tray.
                if ready.load(std::sync::atomic::Ordering::SeqCst) {
                    if let Some(mgr) = manager_slot.get() {
                        let st = mgr.status();
                        let label = status_menu_label(&st);
                        if last_status.as_deref() != Some(label.as_str()) {
                            if let Some(s) = &status_handle {
                                s.set_text(&label);
                            }
                            last_status = Some(label);
                        }
                        if last_model.as_ref() != Some(&st.current) {
                            if let Some(m) = &model_handle {
                                m.set_text(model_menu_label(&st.current));
                            }
                            last_model = Some(st.current);
                        }
                    }
                }

                // Keep the read-only wired line in sync with the Config-page toggle.
                {
                    let st = crate::settings::load_integrations();
                    let label = wired_label(&st);
                    if last_wired.as_deref() != Some(label.as_str()) {
                        if let Some(line) = &wired_handle {
                            line.set_text(&label);
                        }
                        last_wired = Some(label);
                    }
                }

                while let Ok(menu_event) = MenuEvent::receiver().try_recv() {
                    if quit_id.as_ref() == Some(&menu_event.id) {
                        tracing::info!("quit requested via tray menu — unwiring integrations");
                        let st = crate::settings::load_integrations();
                        if st.enabled {
                            let injectors = crate::integrations::injectors_default();
                            let summary = crate::integrations::disable_all(&st.priors, &injectors);
                            let failed: std::collections::HashSet<&String> =
                                summary.failed.iter().map(|(id, _)| id).collect();
                            let mut new_state = st.clone();
                            new_state.priors.retain(|id, _| failed.contains(id));
                            new_state.enabled = !new_state.priors.is_empty();
                            let _ = crate::settings::save_integrations(&new_state);
                        }
                        std::process::exit(0);
                    } else if logs_id.as_ref() == Some(&menu_event.id) {
                        platform::open_path(&log_path);
                        tracing::info!("opening log file {log_path}");
                    } else if url_id.as_ref() == Some(&menu_event.id) {
                        platform::copy_to_clipboard(&url_for_tray);
                        tracing::info!("copied to clipboard: {url_for_tray}");
                    } else if let Some((_, profile, _)) =
                        routing_items.iter().find(|(id, _, _)| id == &menu_event.id)
                    {
                        let chosen = *profile;
                        crate::route::policy::apply_profile(&policy, chosen);
                        if let Err(e) = crate::settings::save_profile(chosen) {
                            tracing::warn!("failed to persist routing profile: {e}");
                        }
                        // Re-check exactly the chosen item; update the status line.
                        for (_, p, item) in &routing_items {
                            item.set_checked(*p == chosen);
                        }
                        if let Some(s) = &routing_status {
                            s.set_text(format!("Routing: {}", chosen.label()));
                        }
                        tracing::info!("routing profile set via tray: {chosen:?}");
                    } else if config_models_id.as_ref() == Some(&menu_event.id)
                        || config_tools_id.as_ref() == Some(&menu_event.id)
                        || config_dash_id.as_ref() == Some(&menu_event.id)
                    {
                        let route = if config_models_id.as_ref() == Some(&menu_event.id) {
                            "#/models"
                        } else if config_tools_id.as_ref() == Some(&menu_event.id) {
                            "#/tools"
                        } else {
                            "#/dashboard"
                        };
                        match &manager_window {
                            Some(w) => {
                                w.window.set_visible(true);
                                w.window.set_focus();
                                w.navigate(route);
                            }
                            None => match window::ModelManagerWindow::open(target, port, &admin_token) {
                                Ok(w) => {
                                    w.navigate(route);
                                    manager_window = Some(w);
                                }
                                Err(e) => tracing::error!(
                                    "Config window failed: {e} (needs WebView2/WebKitGTK)"
                                ),
                            },
                        }
                    }
                }
            }

            // Hide (don't destroy) the Config window on close.
            Event::WindowEvent {
                event: tao::event::WindowEvent::CloseRequested,
                window_id,
                ..
            } => {
                if let Some(w) = &manager_window {
                    if w.window.id() == window_id {
                        w.window.set_visible(false);
                    }
                }
            }

            // Hide the Config window when it loses focus (blur) to save memory.
            Event::WindowEvent {
                event: tao::event::WindowEvent::Focused(false),
                window_id,
                ..
            } => {
                if let Some(w) = &manager_window {
                    if w.window.id() == window_id {
                        w.window.set_visible(false);
                    }
                }
            }

            _ => {}
        }
    })
}
