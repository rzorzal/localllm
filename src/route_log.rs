//! Persistent per-request routing history (JSONL) with rolling retention.
//!
//! One JSON object per line at `<config-dir>/localllm/routing-log.jsonl`
//! (override with `LOCALLLM_ROUTE_LOG`). Writes are best-effort — a failure
//! never fails a request. Old lines are pruned on boot (see `prune_file`).

use std::path::PathBuf;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    // --- score breakdown (optional; populated at the decision point) so the
    // dashboard can explain WHY a request scored as it did. ---
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_turn_tok: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_messages: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx_window: Option<u64>,
    /// Capability-adjusted escalation threshold the score was compared against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    /// Active local model parameter size in billions (0 = unknown).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_b: Option<f64>,
    /// Truncated latest-turn prompt text (the ask only, not full history).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_snippet: Option<String>,
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

/// Best-effort app-log rotation: cap the plain-text app log at a line budget so
/// it cannot grow unbounded. `LOCALLLM_LOG` (default `/tmp/localllm.log`).
/// `_now`/`_max_age_secs` are accepted for symmetry with `prune_file`, but
/// line-count capping is the robust mechanism (the app log has no guaranteed
/// machine-parsable per-line timestamp).
pub fn rotate_app_log(_now: i64, _max_age_secs: i64) {
    const MAX_LINES: usize = 50_000;
    let path = std::env::var("LOCALLLM_LOG").unwrap_or_else(|_| "/tmp/localllm.log".to_string());
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= MAX_LINES {
        return;
    }
    let tail = lines[lines.len() - MAX_LINES..].join("\n");
    let _ = crate::integrations::atomic_write(
        std::path::Path::new(&path),
        format!("{tail}\n").as_bytes(),
    );
}

/// Delete all routing history (the "clear dashboard data" action). Best-effort:
/// truncates the file to empty. Returns true if the file existed.
pub fn clear() -> bool {
    let Some(path) = log_path() else { return false };
    let existed = path.exists();
    let _ = crate::integrations::atomic_write(&path, b"");
    existed
}

/// Current unix seconds.
pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Bucket {
    pub local_count: u64,
    pub cloud_count: u64,
    /// Σ(prompt+completion) of local requests — tokens NOT sent to the provider.
    pub tokens_saved: u64,
    /// Σ(prompt+completion) of ALL requests — the hypothetical all-cloud cost.
    pub tokens_if_all_cloud: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dashboard {
    pub hour: Bucket,
    pub day: Bucket,
    pub month: Bucket,
    /// Newest-first, capped at `recent_n`.
    pub recent: Vec<RouteEntry>,
}

const HOUR: i64 = 3600;
const DAY: i64 = 86_400;
const MONTH: i64 = 2_592_000; // 30 days

fn accumulate(bucket: &mut Bucket, e: &RouteEntry) {
    let toks = e.prompt_tok + e.completion_tok.unwrap_or(0);
    bucket.tokens_if_all_cloud += toks;
    if e.dest == "local" {
        bucket.local_count += 1;
        bucket.tokens_saved += toks;
    } else {
        bucket.cloud_count += 1;
    }
}

/// Build the dashboard rollups: rolling hour/day/month buckets plus the newest
/// `recent_n` entries (newest first).
pub fn build_dashboard(entries: &[RouteEntry], now: i64, recent_n: usize) -> Dashboard {
    let (mut hour, mut day, mut month) = (Bucket::default(), Bucket::default(), Bucket::default());
    for e in entries {
        let age = now - e.ts;
        if age <= MONTH {
            accumulate(&mut month, e);
        }
        if age <= DAY {
            accumulate(&mut day, e);
        }
        if age <= HOUR {
            accumulate(&mut hour, e);
        }
    }
    let mut recent: Vec<RouteEntry> = entries.to_vec();
    recent.sort_by(|a, b| b.ts.cmp(&a.ts));
    recent.truncate(recent_n);
    Dashboard { hour, day, month, recent }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(ts: i64, dest: &str) -> RouteEntry {
        RouteEntry {
            ts,
            surface: "openai".into(),
            dest: dest.into(),
            score: 0.1,
            prompt_tok: 100,
            completion_tok: Some(20),
            ..Default::default()
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

    fn e(ts: i64, dest: &str, p: u64, c: u64) -> RouteEntry {
        RouteEntry {
            ts,
            surface: "openai".into(),
            dest: dest.into(),
            score: 0.1,
            prompt_tok: p,
            completion_tok: Some(c),
            ..Default::default()
        }
    }

    #[test]
    fn dashboard_buckets_and_token_math() {
        let now = 10_000_000i64;
        let entries = vec![
            e(now - 10, "local", 100, 20),      // in hour/day/month
            e(now - 7200, "cloud", 200, 50),    // in day/month, not hour
            e(now - 200_000, "local", 300, 30), // in month only
        ];
        let d = build_dashboard(&entries, now, 10);
        // hour: only the first local entry
        assert_eq!(d.hour.local_count, 1);
        assert_eq!(d.hour.cloud_count, 0);
        assert_eq!(d.hour.tokens_saved, 120);
        assert_eq!(d.hour.tokens_if_all_cloud, 120);
        // day: local(120) + cloud(250)
        assert_eq!(d.day.local_count, 1);
        assert_eq!(d.day.cloud_count, 1);
        assert_eq!(d.day.tokens_saved, 120); // only local counts as saved
        assert_eq!(d.day.tokens_if_all_cloud, 370); // all requests
        // month: all three
        assert_eq!(d.month.tokens_saved, 120 + 330); // two local
        assert_eq!(d.month.tokens_if_all_cloud, 120 + 250 + 330);
        // recent newest-first
        assert_eq!(d.recent.len(), 3);
        assert_eq!(d.recent[0].ts, now - 10);
    }

    #[test]
    fn dashboard_recent_is_capped() {
        let now = 100i64;
        let entries: Vec<RouteEntry> = (0..20).map(|i| e(now - i, "local", 1, 1)).collect();
        let d = build_dashboard(&entries, now, 5);
        assert_eq!(d.recent.len(), 5);
    }

    #[test]
    fn clear_empties_the_log() {
        let dir = std::env::temp_dir().join(format!("localllm-rl-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);
        append(&entry(now_secs(), "local"));
        assert!(!read_all().is_empty());
        assert!(clear());
        assert!(read_all().is_empty());
        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
