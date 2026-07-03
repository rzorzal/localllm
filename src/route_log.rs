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
    /// Correlation id shared with the request's OutcomeEntry (e.g. "req-1a2b3c4d").
    #[serde(default)]
    pub rid: String,
    /// Cloud model the client asked for (the one that WOULD have served). Used
    /// to price $ saved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// When this request fell back to local due to a provider degrade, the reason
    /// (Quota/Auth/ServerError/Offline). None otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degrade_reason: Option<String>,
}

/// Post-generation outcome for a request, correlated to its RouteEntry by `rid`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OutcomeEntry {
    pub rid: String,
    pub ts: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tok: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_ms: Option<u64>,
    /// What this LOCAL request would have cost on cloud (0.0 for cloud requests).
    #[serde(default)]
    pub cost_saved_usd: f64,
}

/// A parsed log line: a routing decision or a post-generation outcome.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind")]
pub enum LogLine {
    #[serde(rename = "d")]
    Decision(RouteEntry),
    #[serde(rename = "o")]
    Outcome(OutcomeEntry),
}

impl LogLine {
    pub fn ts(&self) -> i64 {
        match self {
            LogLine::Decision(d) => d.ts,
            LogLine::Outcome(o) => o.ts,
        }
    }
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
    append_line(&LogLine::Decision(entry.clone()));
}

/// Append one outcome as a JSON line. Best-effort; never panics.
pub fn append_outcome(entry: &OutcomeEntry) {
    append_line(&LogLine::Outcome(entry.clone()));
}

fn append_line(line: &LogLine) {
    let Some(path) = log_path() else { return };
    let Ok(text) = serde_json::to_string(line) else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{text}");
    }
}

/// Read every line: new tagged lines, or legacy untagged decision lines.
pub fn read_all() -> Vec<LogLine> {
    let Some(path) = log_path() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    text.lines()
        .filter_map(|l| {
            serde_json::from_str::<LogLine>(l).ok().or_else(|| {
                serde_json::from_str::<RouteEntry>(l).ok().map(LogLine::Decision)
            })
        })
        .collect()
}

/// Pure retention filter: keep entries with `ts >= now - max_age_secs`.
pub fn prune(entries: &[LogLine], now: i64, max_age_secs: i64) -> Vec<LogLine> {
    let cutoff = now - max_age_secs;
    entries.iter().filter(|e| e.ts() >= cutoff).cloned().collect()
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
    /// Σ cost_saved_usd of local requests in this window.
    pub cost_saved_usd: f64,
}

/// Average latency for one route (local or cloud) over the log.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RouteLatency {
    pub avg_ttft_ms: u64,
    pub avg_tok_s: f64,
    pub n: u64,
}

/// A contiguous window where requests fell back to local, one reason.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FallbackWindow {
    /// "budget" | "provider"
    pub kind: String,
    /// human reason (e.g. "Quota", "BudgetExceeded")
    pub reason: String,
    pub start_ts: i64,
    pub end_ts: i64,
    pub count: u64,
}

/// A recent decision joined to its outcome, for the dashboard table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecentRow {
    #[serde(flatten)]
    pub entry: RouteEntry,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tok: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ms: Option<u64>,
    pub cost_saved_usd: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Dashboard {
    pub hour: Bucket,
    pub day: Bucket,
    pub month: Bucket,
    pub local_latency: RouteLatency,
    pub cloud_latency: RouteLatency,
    pub windows: Vec<FallbackWindow>,
    /// Newest-first, capped at `recent_n`.
    pub recent: Vec<RecentRow>,
}

const HOUR: i64 = 3600;
const DAY: i64 = 86_400;
const MONTH: i64 = 2_592_000; // 30 days

fn accumulate(bucket: &mut Bucket, d: &RouteEntry, o: Option<&OutcomeEntry>) {
    // Completion tokens: outcome by rid first, then the legacy RouteEntry field.
    let completion = o.and_then(|o| o.completion_tok).or(d.completion_tok).unwrap_or(0);
    let toks = d.prompt_tok + completion;
    bucket.tokens_if_all_cloud += toks;
    if d.dest == "local" {
        bucket.local_count += 1;
        bucket.tokens_saved += toks;
        bucket.cost_saved_usd += o.map(|o| o.cost_saved_usd).unwrap_or(0.0);
    } else {
        bucket.cloud_count += 1;
    }
}

/// Build the dashboard rollups: rolling hour/day/month buckets (with $ saved),
/// per-route latency, contiguous fallback windows, plus the newest `recent_n`
/// decisions joined to their outcomes. Accepts the full `Vec<LogLine>`.
pub fn build_dashboard(entries: &[LogLine], now: i64, recent_n: usize) -> Dashboard {
    use std::collections::HashMap;
    let mut decisions: Vec<&RouteEntry> = Vec::new();
    let mut outcomes: HashMap<&str, &OutcomeEntry> = HashMap::new();
    for l in entries {
        match l {
            LogLine::Decision(d) => decisions.push(d),
            LogLine::Outcome(o) => { outcomes.insert(o.rid.as_str(), o); }
        }
    }
    let (mut hour, mut day, mut month) = (Bucket::default(), Bucket::default(), Bucket::default());
    let (mut l_ttft, mut l_tps, mut l_n) = (0u64, 0f64, 0u64);
    let (mut c_ttft, mut c_tps, mut c_n) = (0u64, 0f64, 0u64);
    for d in &decisions {
        let o = outcomes.get(d.rid.as_str()).copied();
        let age = now - d.ts;
        if age <= MONTH { accumulate(&mut month, d, o); }
        if age <= DAY { accumulate(&mut day, d, o); }
        if age <= HOUR { accumulate(&mut hour, d, o); }
        if let Some(o) = o {
            let tps = match (o.completion_tok, o.gen_ms) {
                (Some(ct), Some(ms)) if ms > 0 => ct as f64 / (ms as f64 / 1000.0),
                _ => 0.0,
            };
            if d.dest == "local" {
                if let Some(t) = o.ttft_ms { l_ttft += t; }
                l_tps += tps; l_n += 1;
            } else {
                if let Some(t) = o.ttft_ms { c_ttft += t; }
                c_tps += tps; c_n += 1;
            }
        }
    }
    let latency = |ttft: u64, tps: f64, n: u64| RouteLatency {
        avg_ttft_ms: if n > 0 { ttft / n } else { 0 },
        avg_tok_s: if n > 0 { tps / n as f64 } else { 0.0 },
        n,
    };
    // Fallback windows: group contiguous same-reason local decisions.
    let mut flagged: Vec<&RouteEntry> = decisions.iter().filter(|d| {
        d.degrade_reason.is_some() || d.reason.as_deref() == Some("BudgetExceeded")
    }).copied().collect();
    flagged.sort_by_key(|d| d.ts);
    let mut windows: Vec<FallbackWindow> = Vec::new();
    const GAP: i64 = 120; // seconds; same-reason events within this gap merge
    for d in flagged {
        let (kind, reason) = if let Some(r) = &d.degrade_reason {
            ("provider", r.clone())
        } else {
            ("budget", "BudgetExceeded".to_string())
        };
        match windows.last_mut() {
            Some(w) if w.kind == kind && w.reason == reason && d.ts - w.end_ts <= GAP => {
                w.end_ts = d.ts; w.count += 1;
            }
            _ => windows.push(FallbackWindow {
                kind: kind.to_string(), reason, start_ts: d.ts, end_ts: d.ts, count: 1,
            }),
        }
    }
    // Recent rows: newest-first decisions joined to their outcome.
    let mut recent: Vec<RecentRow> = decisions.iter().map(|d| {
        let o = outcomes.get(d.rid.as_str()).copied();
        RecentRow {
            entry: (*d).clone(),
            completion_tok: o.and_then(|o| o.completion_tok),
            ttft_ms: o.and_then(|o| o.ttft_ms),
            gen_ms: o.and_then(|o| o.gen_ms),
            cost_saved_usd: o.map(|o| o.cost_saved_usd).unwrap_or(0.0),
        }
    }).collect();
    recent.sort_by(|a, b| b.entry.ts.cmp(&a.entry.ts));
    recent.truncate(recent_n);
    Dashboard {
        hour, day, month,
        local_latency: latency(l_ttft, l_tps, l_n),
        cloud_latency: latency(c_ttft, c_tps, c_n),
        windows,
        recent,
    }
}

/// Serialises tests (in any module) that mutate the process-global
/// LOCALLLM_ROUTE_LOG env var, so parallel runs don't clobber each other's path.
#[cfg(test)]
pub(crate) static ROUTE_LOG_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        let kept = prune(
            &[
                LogLine::Decision(entry(now - month - 1, "local")),
                LogLine::Decision(entry(now - 10, "cloud")),
            ],
            now,
            month,
        );
        assert_eq!(kept.len(), 1);
        assert!(matches!(&kept[0], LogLine::Decision(d) if d.dest == "cloud"));
    }

    #[test]
    fn prune_keeps_everything_when_all_fresh() {
        let now = 1_000_000i64;
        let kept = prune(
            &[
                LogLine::Decision(entry(now - 5, "local")),
                LogLine::Decision(entry(now - 6, "cloud")),
            ],
            now,
            100,
        );
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
    fn dashboard_buckets_cost_and_join() {
        let now = 10_000_000i64;
        let mut lines = vec![
            LogLine::Decision(RouteEntry { ts: now - 10, rid: "a".into(), surface: "openai".into(),
                dest: "local".into(), prompt_tok: 100, ..Default::default() }),
            LogLine::Decision(RouteEntry { ts: now - 20, rid: "b".into(), surface: "openai".into(),
                dest: "cloud".into(), prompt_tok: 200, ..Default::default() }),
        ];
        lines.push(LogLine::Outcome(OutcomeEntry { rid: "a".into(), ts: now - 9,
            completion_tok: Some(20), ttft_ms: Some(30), gen_ms: Some(100), cost_saved_usd: 0.42 }));
        lines.push(LogLine::Outcome(OutcomeEntry { rid: "b".into(), ts: now - 19,
            completion_tok: Some(50), ttft_ms: Some(80), gen_ms: Some(500), cost_saved_usd: 0.0 }));
        let d = build_dashboard(&lines, now, 10);
        // hour bucket: local 120 saved, all-cloud 120+250
        assert_eq!(d.hour.tokens_saved, 120);
        assert_eq!(d.hour.tokens_if_all_cloud, 370);
        assert!((d.hour.cost_saved_usd - 0.42).abs() < 1e-9);
        // local latency: 20 tok / 0.1s = 200 tok/s, ttft 30
        assert_eq!(d.local_latency.n, 1);
        assert_eq!(d.local_latency.avg_ttft_ms, 30);
        assert!((d.local_latency.avg_tok_s - 200.0).abs() < 1.0);
        // recent carries the joined completion_tok
        assert_eq!(d.recent.len(), 2);
        let row_a = d.recent.iter().find(|r| r.entry.rid == "a").unwrap();
        assert_eq!(row_a.completion_tok, Some(20));
    }

    #[test]
    fn dashboard_groups_fallback_windows() {
        let now = 1_000_000i64;
        let lines = vec![
            LogLine::Decision(RouteEntry { ts: now - 300, rid: "1".into(), surface: "openai".into(),
                dest: "local".into(), degrade_reason: Some("Quota".into()), ..Default::default() }),
            LogLine::Decision(RouteEntry { ts: now - 290, rid: "2".into(), surface: "openai".into(),
                dest: "local".into(), degrade_reason: Some("Quota".into()), ..Default::default() }),
            LogLine::Decision(RouteEntry { ts: now - 100, rid: "3".into(), surface: "openai".into(),
                dest: "local".into(), reason: Some("BudgetExceeded".into()), ..Default::default() }),
        ];
        let d = build_dashboard(&lines, now, 10);
        assert_eq!(d.windows.len(), 2);
        let prov = d.windows.iter().find(|w| w.kind == "provider").unwrap();
        assert_eq!(prov.count, 2);
        assert_eq!(prov.reason, "Quota");
        let bud = d.windows.iter().find(|w| w.kind == "budget").unwrap();
        assert_eq!(bud.count, 1);
    }

    #[test]
    fn dashboard_recent_is_capped() {
        let now = 100i64;
        let entries: Vec<LogLine> = (0..20).map(|i| LogLine::Decision(e(now - i, "local", 1, 1))).collect();
        let d = build_dashboard(&entries, now, 5);
        assert_eq!(d.recent.len(), 5);
    }

    #[test]
    fn clear_empties_the_log() {
        let _guard = ROUTE_LOG_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    #[test]
    fn read_all_parses_new_and_legacy_lines() {
        let _guard = ROUTE_LOG_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-ll-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);
        // legacy decision line (no "kind")
        let legacy = r#"{"ts":10,"surface":"openai","dest":"local","score":0.1,"prompt_tok":5}"#;
        std::fs::write(&path, format!("{legacy}\n")).unwrap();
        // new decision + outcome via the API
        append(&RouteEntry { ts: 20, rid: "req-1".into(), surface: "openai".into(),
            dest: "local".into(), prompt_tok: 7, ..Default::default() });
        append_outcome(&OutcomeEntry { rid: "req-1".into(), ts: 21, completion_tok: Some(9),
            ttft_ms: Some(30), gen_ms: Some(100), cost_saved_usd: 0.5 });
        let lines = read_all();
        assert_eq!(lines.len(), 3);
        assert!(matches!(lines[0], LogLine::Decision(ref d) if d.prompt_tok == 5));
        assert!(matches!(lines[2], LogLine::Outcome(ref o) if o.completion_tok == Some(9)));
        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
