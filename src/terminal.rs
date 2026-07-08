//! Terminal-app abstraction for the tray "Launch … via LocalLLM" items (macOS).

use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalApp {
    Terminal, // Apple Terminal (always present on macOS)
    ITerm,
    Warp,
    Wave,
}

/// Preference order for auto-default: Apple Terminal last-resort but always available.
pub const PREFERENCE: [TerminalApp; 4] = [
    TerminalApp::Terminal,
    TerminalApp::ITerm,
    TerminalApp::Warp,
    TerminalApp::Wave,
];

impl TerminalApp {
    pub fn id(self) -> &'static str {
        match self {
            TerminalApp::Terminal => "terminal",
            TerminalApp::ITerm => "iterm",
            TerminalApp::Warp => "warp",
            TerminalApp::Wave => "wave",
        }
    }
    pub fn from_id(s: &str) -> Option<Self> {
        PREFERENCE.into_iter().find(|a| a.id() == s)
    }
    /// The .app bundle name under /Applications.
    pub fn app_name(self) -> &'static str {
        match self {
            TerminalApp::Terminal => "Terminal",
            TerminalApp::ITerm => "iTerm",
            TerminalApp::Warp => "Warp",
            TerminalApp::Wave => "Wave",
        }
    }
    fn is_present(self) -> bool {
        let name = self.app_name();
        Path::new(&format!("/Applications/{name}.app")).exists()
            || dirs::home_dir()
                .map(|h| h.join("Applications").join(format!("{name}.app")).exists())
                .unwrap_or(false)
    }
}

/// First present app in PREFERENCE order; Apple Terminal always qualifies.
pub fn pick_default(present: &[TerminalApp]) -> TerminalApp {
    PREFERENCE
        .into_iter()
        .find(|a| present.contains(a))
        .unwrap_or(TerminalApp::Terminal)
}

/// Installed terminals in preference order.
pub fn installed() -> Vec<TerminalApp> {
    PREFERENCE.into_iter().filter(|a| a.is_present()).collect()
}

/// Open a new terminal window in `dir` running `command`. Best-effort.
pub fn open(app: TerminalApp, dir: &Path, command: &str) -> std::io::Result<()> {
    let dir = dir.display();
    match app {
        // Scriptable: run the command directly in a new window.
        TerminalApp::Terminal => {
            let script =
                format!("tell application \"Terminal\" to do script \"cd {dir} && {command}\"");
            Command::new("osascript")
                .arg("-e")
                .arg(script)
                .status()
                .map(|_| ())
        }
        TerminalApp::ITerm => {
            let script = format!(
                "tell application \"iTerm\"\ncreate window with default profile\ntell current session of current window to write text \"cd {dir} && {command}\"\nend tell"
            );
            Command::new("osascript")
                .arg("-e")
                .arg(script)
                .status()
                .map(|_| ())
        }
        // Warp/Wave: no stable `do script`. Open the app at the folder; the user
        // runs the wrapper (which is on PATH). Documented limitation.
        TerminalApp::Warp | TerminalApp::Wave => Command::new("open")
            .arg("-a")
            .arg(app.app_name())
            .arg(dir.to_string())
            .status()
            .map(|_| ()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_round_trips() {
        for a in PREFERENCE {
            assert_eq!(TerminalApp::from_id(a.id()), Some(a));
        }
        assert_eq!(TerminalApp::from_id("nope"), None);
    }

    #[test]
    fn pick_default_prefers_first_present_in_order() {
        let present = vec![TerminalApp::Warp, TerminalApp::Terminal];
        assert_eq!(pick_default(&present), TerminalApp::Terminal);
        assert_eq!(pick_default(&[TerminalApp::Warp]), TerminalApp::Warp);
        assert_eq!(pick_default(&[]), TerminalApp::Terminal);
    }
}
