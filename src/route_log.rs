//! Persistent per-request routing history (JSONL) with rolling retention.
//!
//! One JSON object per line at `<config-dir>/localllm/routing-log.jsonl`
//! (override with `LOCALLLM_ROUTE_LOG`). Writes are best-effort — a failure
//! never fails a request. Old lines are pruned on boot (see `prune_file`).

use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RouteEntry {
    /// Unix seconds.
    pub ts: i64,
    /// Client surface: "anthropic" | "openai" | "openai-responses".
    pub surface: String,
    /// "local" | "cloud".
    pub dest: String,
    /// RouteReason as a short string when dest == "cloud"; else None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub score: f64,
    pub prompt_tok: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tok: Option<u64>,
}

/// Resolve the log path. `LOCALLLM_ROUTE_LOG` (full file path) wins.
pub fn log_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LOCALLLM_ROUTE_LOG") {
        return Some(PathBuf::from(p));
    }
    dirs::config_dir().map(|d| d.join("localllm").join("routing-log.jsonl"))
}

/// Append one entry as a JSON line. Best-effort; never panics.
pub fn append(entry: &RouteEntry) {
    let Some(path) = log_path() else { return };
    let Ok(line) = serde_json::to_string(entry) else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Read every entry, skipping malformed lines. Empty when the file is absent.
pub fn read_all() -> Vec<RouteEntry> {
    let Some(path) = log_path() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    text.lines()
        .filter_map(|l| serde_json::from_str::<RouteEntry>(l).ok())
        .collect()
}

/// Pure retention filter: keep entries with `ts >= now - max_age_secs`.
pub fn prune(entries: &[RouteEntry], now: i64, max_age_secs: i64) -> Vec<RouteEntry> {
    let cutoff = now - max_age_secs;
    entries.iter().filter(|e| e.ts >= cutoff).cloned().collect()
}

/// Read, prune, and atomically rewrite the log. Best-effort.
pub fn prune_file(now: i64, max_age_secs: i64) {
    let Some(path) = log_path() else { return };
    let kept = prune(&read_all(), now, max_age_secs);
    let mut buf = String::new();
    for e in &kept {
        if let Ok(l) = serde_json::to_string(e) {
            buf.push_str(&l);
            buf.push('\n');
        }
    }
    let _ = crate::integrations::atomic_write(&path, buf.as_bytes());
}

/// Current unix seconds.
pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(ts: i64, dest: &str) -> RouteEntry {
        RouteEntry {
            ts,
            surface: "openai".into(),
            dest: dest.into(),
            reason: None,
            score: 0.1,
            prompt_tok: 100,
            completion_tok: Some(20),
        }
    }

    #[test]
    fn prune_drops_entries_older_than_max_age() {
        let now = 1_000_000i64;
        let month = 30 * 24 * 3600;
        let kept = prune(&[entry(now - month - 1, "local"), entry(now - 10, "cloud")], now, month);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].dest, "cloud");
    }

    #[test]
    fn prune_keeps_everything_when_all_fresh() {
        let now = 1_000_000i64;
        let kept = prune(&[entry(now - 5, "local"), entry(now - 6, "cloud")], now, 100);
        assert_eq!(kept.len(), 2);
    }
}
