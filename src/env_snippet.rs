//! Builds the inline env-var snippet the tray copies to the clipboard, so a
//! user can prepend it to their own `claude`/`codex` invocation to point the
//! client at the local server. Format matches the OS this binary runs on.

/// Client request timeout (ms). 20 min — covers a slow cold-prefill on the
/// local model so the client does not give up mid-request.
const TIMEOUT_MS: &str = "1200000";

/// The four (key, value) env pairs, port interpolated into the base URLs.
fn pairs(port: u16) -> [(&'static str, String); 4] {
    [
        ("ANTHROPIC_BASE_URL", format!("http://127.0.0.1:{port}")),
        ("API_TIMEOUT_MS", TIMEOUT_MS.to_string()),
        ("CLAUDE_STREAM_IDLE_TIMEOUT_MS", TIMEOUT_MS.to_string()),
        ("OPENAI_BASE_URL", format!("http://127.0.0.1:{port}/v1")),
    ]
}

/// Format the snippet for the given OS. `windows` selects the PowerShell
/// `$env:K="v";` form (space-joined); otherwise the POSIX inline `K=v` prefix
/// (space-joined, no `export`, no trailing command — the user appends it).
fn format_snippet(port: u16, windows: bool) -> String {
    let pairs = pairs(port);
    if windows {
        let parts: Vec<String> = pairs
            .iter()
            .map(|(k, v)| format!("$env:{k}=\"{v}\";"))
            .collect();
        let mut result = parts.join(" ");
        // Remove trailing semicolon from the last assignment
        if result.ends_with(';') {
            result.pop();
        }
        result
    } else {
        let parts: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect();
        parts.join(" ")
    }
}

/// Build the snippet for the OS this binary was compiled for (the tray runs on
/// the user's own machine, so the compile-time target OS is the user's OS).
pub fn build(port: u16) -> String {
    format_snippet(port, cfg!(target_os = "windows"))
}

#[cfg(test)]
mod tests {
    use super::format_snippet;

    #[test]
    fn unix_is_inline_space_joined_no_export() {
        assert_eq!(
            format_snippet(31415, false),
            "ANTHROPIC_BASE_URL=http://127.0.0.1:31415 \
API_TIMEOUT_MS=1200000 \
CLAUDE_STREAM_IDLE_TIMEOUT_MS=1200000 \
OPENAI_BASE_URL=http://127.0.0.1:31415/v1"
        );
    }

    #[test]
    fn windows_is_powershell_env_assignments() {
        assert_eq!(
            format_snippet(31415, true),
            "$env:ANTHROPIC_BASE_URL=\"http://127.0.0.1:31415\"; \
$env:API_TIMEOUT_MS=\"1200000\"; \
$env:CLAUDE_STREAM_IDLE_TIMEOUT_MS=\"1200000\"; \
$env:OPENAI_BASE_URL=\"http://127.0.0.1:31415/v1\""
        );
    }

    #[test]
    fn port_is_interpolated_into_both_base_urls() {
        let out = format_snippet(9000, false);
        assert!(out.contains("ANTHROPIC_BASE_URL=http://127.0.0.1:9000 "));
        assert!(out.contains("OPENAI_BASE_URL=http://127.0.0.1:9000/v1"));
    }
}
