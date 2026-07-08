//! `localllm claude|codex`: run the client in the current directory wired to the
//! local proxy with raised timeouts, so no manual env export is needed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Default proxy port. MUST match `Config`'s `--port` default (31415).
pub const DEFAULT_PORT: u16 = 31415;

const TIMEOUT_MS: &str = "1200000"; // 20 min; covers slow cold-prefill on the local model

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Client {
    Claude,
    Codex,
}

impl Client {
    fn bin(self) -> &'static str {
        match self {
            Client::Claude => "claude",
            Client::Codex => "codex",
        }
    }
}

/// If the first positional arg is `claude`/`codex`, return the client and the
/// remaining passthrough args. Otherwise None (normal server/tray mode).
pub fn detect(args: &[String]) -> Option<(Client, Vec<String>)> {
    let first = args.get(1)?;
    let client = match first.as_str() {
        "claude" => Client::Claude,
        "codex" => Client::Codex,
        _ => return None,
    };
    Some((client, args[2..].to_vec()))
}

/// Proxy env to inject for a client. Pure — unit-tested.
pub fn env_for(client: Client, port: u16) -> Vec<(String, String)> {
    match client {
        Client::Claude => vec![
            (
                "ANTHROPIC_BASE_URL".into(),
                format!("http://127.0.0.1:{port}"),
            ),
            ("API_TIMEOUT_MS".into(), TIMEOUT_MS.into()),
            ("CLAUDE_STREAM_IDLE_TIMEOUT_MS".into(), TIMEOUT_MS.into()),
        ],
        Client::Codex => vec![(
            "OPENAI_BASE_URL".into(),
            format!("http://127.0.0.1:{port}/v1"),
        )],
    }
}

/// Exec the client with proxy env in the current directory. Replaces this
/// process on unix so tty/signals/exit-code pass through.
pub fn run(client: Client, passthrough: Vec<String>) -> ! {
    let mut cmd = Command::new(client.bin());
    cmd.args(&passthrough);
    for (k, v) in env_for(client, DEFAULT_PORT) {
        cmd.env(k, v);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec(); // only returns on failure
        eprintln!(
            "localllm: failed to launch `{}`: {err}. Is it installed and on PATH?",
            client.bin()
        );
        std::process::exit(127);
    }
    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(s) => std::process::exit(s.code().unwrap_or(1)),
            Err(e) => {
                eprintln!(
                    "localllm: failed to launch `{}`: {e}. Is it installed and on PATH?",
                    client.bin()
                );
                std::process::exit(127);
            }
        }
    }
}

/// Pick the CLI install dir: first writable candidate, else the home fallback.
pub fn cli_target_dir(
    candidates: &[&Path],
    home_local_bin: &Path,
    is_writable: impl Fn(&Path) -> bool,
) -> PathBuf {
    for c in candidates {
        if is_writable(c) {
            return c.to_path_buf();
        }
    }
    home_local_bin.to_path_buf()
}

/// (Re)symlink `localllm` → the running binary into a PATH dir, so `localllm
/// claude` works from the user's terminal and always matches this app version.
/// Best-effort — never blocks boot.
pub fn install_cli() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let home_local_bin = dirs::home_dir()
        .map(|h| h.join(".local/bin"))
        .unwrap_or_else(|| PathBuf::from("/usr/local/bin"));
    let candidates = [Path::new("/usr/local/bin"), Path::new("/opt/homebrew/bin")];
    let is_writable = |p: &Path| {
        p.exists()
            && std::fs::metadata(p)
                .map(|m| !m.permissions().readonly())
                .unwrap_or(false)
            && {
                let t = p.join(".localllm-wtest");
                let ok = std::fs::write(&t, b"").is_ok();
                let _ = std::fs::remove_file(&t);
                ok
            }
    };
    let dir = cli_target_dir(&candidates, &home_local_bin, is_writable);
    let _ = std::fs::create_dir_all(&dir);
    let link = dir.join("localllm");
    // Idempotent: skip if already pointing at the current exe.
    if std::fs::read_link(&link).ok().as_deref() == Some(exe.as_path()) {
        return;
    }
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    let res = std::os::unix::fs::symlink(&exe, &link);
    #[cfg(not(unix))]
    let res: std::io::Result<()> = std::fs::copy(&exe, &link).map(|_| ());
    match res {
        Ok(_) => {
            tracing::info!(target: "localllm", "CLI installed: {} -> {}", link.display(), exe.display());
            if !std::env::var("PATH")
                .unwrap_or_default()
                .split(':')
                .any(|p| Path::new(p) == dir)
            {
                tracing::warn!(target: "localllm", "{} is not on PATH — add it to use `localllm claude`", dir.display());
            }
        }
        Err(e) => tracing::warn!(target: "localllm", "CLI install skipped: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_target_dir_prefers_first_writable_then_home() {
        use std::path::{Path, PathBuf};
        let usr = Path::new("/usr/local/bin");
        let brew = Path::new("/opt/homebrew/bin");
        let home = PathBuf::from("/Users/x/.local/bin");
        assert_eq!(
            cli_target_dir(&[usr, brew], &home, |p| p == usr),
            usr.to_path_buf()
        );
        assert_eq!(
            cli_target_dir(&[usr, brew], &home, |p| p == brew),
            brew.to_path_buf()
        );
        assert_eq!(cli_target_dir(&[usr, brew], &home, |_| false), home);
    }

    #[test]
    fn detect_claude_and_passthrough() {
        let args = vec!["localllm".into(), "claude".into(), "--resume".into()];
        let (c, rest) = detect(&args).unwrap();
        assert_eq!(c, Client::Claude);
        assert_eq!(rest, vec!["--resume".to_string()]);
    }

    #[test]
    fn detect_codex() {
        let args = vec!["localllm".into(), "codex".into()];
        assert_eq!(detect(&args).unwrap().0, Client::Codex);
    }

    #[test]
    fn detect_none_for_server_mode() {
        let args = vec!["localllm".into(), "--port".into(), "31415".into()];
        assert!(detect(&args).is_none());
    }

    #[test]
    fn env_for_claude_sets_base_url_and_timeouts() {
        let env = env_for(Client::Claude, 31415);
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:31415"));
        assert_eq!(get("API_TIMEOUT_MS"), Some("1200000"));
        assert_eq!(get("CLAUDE_STREAM_IDLE_TIMEOUT_MS"), Some("1200000"));
    }

    #[test]
    fn env_for_codex_sets_openai_base_url() {
        let env = env_for(Client::Codex, 31415);
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("OPENAI_BASE_URL"), Some("http://127.0.0.1:31415/v1"));
    }
}
