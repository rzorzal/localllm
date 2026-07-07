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
    /// Compact key (model_ctx_key(repo,file)) of the LOCAL model active when
    /// this decision was made. Set on every decision; None on legacy lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_model: Option<String>,
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

/// A routing-quality signal about an earlier decision, correlated by `rid`.
/// Signals: "cascade" (local escalated to cloud), "reask" (user re-sent a very
/// similar prompt shortly after a local answer), "truncated" (local hit the
/// length limit un-escalated), "cloud_trivial" (cloud produced a trivial reply).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FeedbackEntry {
    /// The rid of the DECISION being judged (for "reask": the PREVIOUS request).
    pub rid: String,
    pub ts: i64,
    pub signal: String,
}

/// A parsed log line: a routing decision or a post-generation outcome.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind")]
pub enum LogLine {
    #[serde(rename = "d")]
    Decision(RouteEntry),
    #[serde(rename = "o")]
    Outcome(OutcomeEntry),
    #[serde(rename = "f")]
    Feedback(FeedbackEntry),
}

impl LogLine {
    pub fn ts(&self) -> i64 {
        match self {
            LogLine::Decision(d) => d.ts,
            LogLine::Outcome(o) => o.ts,
            LogLine::Feedback(f) => f.ts,
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

/// Append one feedback signal as a JSON line. Best-effort; never panics.
pub fn append_feedback(entry: &FeedbackEntry) {
    append_line(&LogLine::Feedback(entry.clone()));
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

/// Rollup of quality signals over the dashboard window (30d).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FeedbackStats {
    pub local_total: u64,
    /// Locals with >=1 of: cascade | truncated | reask.
    pub local_flagged: u64,
    pub cloud_total: u64,
    /// Clouds flagged cloud_trivial.
    pub cloud_trivial: u64,
}

/// Data-driven Balanced-threshold suggestion (measurement only; never applied).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThresholdSuggestion {
    /// "lower" | "raise"
    pub direction: String,
    /// Suggested threshold, clamped to [SUGGEST_CLAMP_MIN, SUGGEST_CLAMP_MAX].
    pub suggested: f64,
    /// Human sentence for the dashboard banner (pt-BR, matches the UI language).
    pub why: String,
}

/// Per-local-model effective-capability estimate (informational; not routed).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelCapability {
    /// Compact model key (model_ctx_key).
    pub model: String,
    pub local_total: u64,
    pub flagged_rate: f64,
    /// Nominal size (billions) from params_b_for_key.
    pub nominal_b: f64,
    /// Estimated effective size (billions) after inverting the capability slope.
    pub effective_b: f64,
}

/// Effective-capability tunables — named so a later phase can calibrate them.
pub const CAP_MIN_SAMPLE: u64 = 30;
pub const CAP_BASELINE_FLAG_RATE: f64 = 0.10;
pub const CAP_SCALE: f64 = 0.5;
/// Must equal the slope in `route::capability_adjustment` (0.03 threshold / 1B).
pub const CAP_SLOPE: f64 = 0.03;

/// Suggestion tunables — named so Fase D2 can calibrate them.
pub const SUGGEST_WINDOW_SECS: i64 = 7 * 86_400;
pub const SUGGEST_MIN_LOCAL_SAMPLE: u64 = 20;
pub const SUGGEST_LOWER_FLAGGED_RATE: f64 = 0.15;
pub const SUGGEST_RAISE_TRIVIAL_RATE: f64 = 0.30;
pub const SUGGEST_CLAMP_MIN: f64 = 0.20;
pub const SUGGEST_CLAMP_MAX: f64 = 0.80;

/// p-th percentile (0.0..=1.0) of a non-empty slice; interpolation-free
/// (nearest-rank). Returns None on empty input.
fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() { return None; }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted.get(idx).copied()
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
    /// Quality signals recorded against this decision (may be empty).
    #[serde(default)]
    pub feedback: Vec<String>,
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
    pub feedback: FeedbackStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<ThresholdSuggestion>,
    #[serde(default)]
    pub model_capabilities: Vec<ModelCapability>,
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
pub fn build_dashboard(entries: &[LogLine], now: i64, recent_n: usize, balanced: bool) -> Dashboard {
    use std::collections::HashMap;
    let mut decisions: Vec<&RouteEntry> = Vec::new();
    let mut outcomes: HashMap<&str, &OutcomeEntry> = HashMap::new();
    let mut feedback: HashMap<&str, Vec<String>> = HashMap::new();
    for l in entries {
        match l {
            LogLine::Decision(d) => decisions.push(d),
            LogLine::Outcome(o) => { outcomes.insert(o.rid.as_str(), o); }
            LogLine::Feedback(f) => feedback.entry(f.rid.as_str()).or_default().push(f.signal.clone()),
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
        d.degrade_reason.is_some()
            || d.reason.as_deref() == Some("BudgetExceeded")
            || d.reason.as_deref() == Some("CloudDown")
    }).copied().collect();
    flagged.sort_by_key(|d| d.ts);
    let mut windows: Vec<FallbackWindow> = Vec::new();
    const GAP: i64 = 120; // seconds; same-reason events within this gap merge
    for d in flagged {
        let (kind, reason) = if let Some(r) = &d.degrade_reason {
            ("provider", r.clone())
        } else if d.reason.as_deref() == Some("CloudDown") {
            // Breaker-forced local (cloud in cooldown) — a provider outage window.
            ("provider", "CloudDown".to_string())
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
            feedback: feedback.get(d.rid.as_str()).cloned().unwrap_or_default(),
        }
    }).collect();
    recent.sort_by(|a, b| b.entry.ts.cmp(&a.entry.ts));
    recent.truncate(recent_n);
    // Quality-signal rollup (whole retained log) + threshold suggestion (7d).
    const NEG_LOCAL: [&str; 3] = ["cascade", "truncated", "reask"];
    let mut stats = FeedbackStats::default();
    let mut lower_scores: Vec<f64> = Vec::new(); // flagged locals, 7d
    let mut raise_scores: Vec<f64> = Vec::new(); // trivial clouds, 7d
    let mut w_local_total = 0u64;
    let mut w_local_flagged = 0u64;
    let mut w_cloud_total = 0u64;
    let mut w_cloud_trivial = 0u64;
    let window_start = now - SUGGEST_WINDOW_SECS;
    for d in &decisions {
        let sigs = feedback.get(d.rid.as_str());
        let in_window = d.ts >= window_start;
        if d.dest == "local" {
            stats.local_total += 1;
            let flagged = sigs.map(|s| s.iter().any(|x| NEG_LOCAL.contains(&x.as_str()))).unwrap_or(false);
            if flagged { stats.local_flagged += 1; }
            if in_window {
                w_local_total += 1;
                if flagged { w_local_flagged += 1; lower_scores.push(d.score); }
            }
        } else {
            stats.cloud_total += 1;
            let trivial = sigs.map(|s| s.iter().any(|x| x == "cloud_trivial")).unwrap_or(false);
            if trivial { stats.cloud_trivial += 1; }
            if in_window {
                w_cloud_total += 1;
                if trivial { w_cloud_trivial += 1; raise_scores.push(d.score); }
            }
        }
    }
    let suggestion = if balanced && w_local_total >= SUGGEST_MIN_LOCAL_SAMPLE {
        let flagged_rate = w_local_flagged as f64 / w_local_total as f64;
        let trivial_rate = if w_cloud_total > 0 { w_cloud_trivial as f64 / w_cloud_total as f64 } else { 0.0 };
        if flagged_rate > SUGGEST_LOWER_FLAGGED_RATE {
            lower_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percentile(&lower_scores, 0.25).map(|p| {
                let v = p.clamp(SUGGEST_CLAMP_MIN, SUGGEST_CLAMP_MAX);
                ThresholdSuggestion {
                    direction: "lower".into(),
                    suggested: v,
                    why: format!(
                        "{:.0}% dos pedidos locais precisaram da cloud — considere baixar o limiar para {:.0}%",
                        flagged_rate * 100.0, v * 100.0),
                }
            })
        } else if trivial_rate > SUGGEST_RAISE_TRIVIAL_RATE {
            raise_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            percentile(&raise_scores, 0.75).map(|p| {
                let v = p.clamp(SUGGEST_CLAMP_MIN, SUGGEST_CLAMP_MAX);
                ThresholdSuggestion {
                    direction: "raise".into(),
                    suggested: v,
                    why: format!(
                        "{:.0}% dos pedidos na cloud foram triviais — considere subir o limiar para {:.0}%",
                        trivial_rate * 100.0, v * 100.0),
                }
            })
        } else { None }
    } else { None };
    // Per-model effective-capability estimate (informational).
    use std::collections::BTreeMap;
    let mut per_model: BTreeMap<&str, (u64, u64)> = BTreeMap::new(); // model → (total, flagged)
    for d in &decisions {
        if d.dest != "local" { continue; }
        let Some(m) = d.local_model.as_deref() else { continue; };
        let flagged = feedback
            .get(d.rid.as_str())
            .map(|s| s.iter().any(|x| NEG_LOCAL.contains(&x.as_str())))
            .unwrap_or(false);
        let e = per_model.entry(m).or_insert((0, 0));
        e.0 += 1;
        if flagged { e.1 += 1; }
    }
    let mut model_capabilities: Vec<ModelCapability> = per_model
        .into_iter()
        .filter(|(_, (total, _))| *total >= CAP_MIN_SAMPLE)
        .filter_map(|(model, (total, flagged))| {
            let nominal_b = crate::catalog::params_b_for_key(model) as f64;
            if nominal_b <= 0.0 { return None; }
            let flagged_rate = flagged as f64 / total as f64;
            let excess = flagged_rate - CAP_BASELINE_FLAG_RATE;
            let delta_b = -(excess * CAP_SCALE) / CAP_SLOPE;
            let effective_b = (nominal_b + delta_b).clamp(0.5, nominal_b * 1.5);
            Some(ModelCapability {
                model: model.to_string(),
                local_total: total,
                flagged_rate,
                nominal_b,
                effective_b,
            })
        })
        .collect();
    model_capabilities.sort_by(|a, b| b.local_total.cmp(&a.local_total));
    Dashboard {
        hour, day, month,
        local_latency: latency(l_ttft, l_tps, l_n),
        cloud_latency: latency(c_ttft, c_tps, c_n),
        windows,
        recent,
        feedback: stats,
        suggestion,
        model_capabilities,
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
        let d = build_dashboard(&lines, now, 10, true);
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
            LogLine::Decision(RouteEntry { ts: now - 50, rid: "4".into(), surface: "openai".into(),
                dest: "local".into(), reason: Some("CloudDown".into()), ..Default::default() }),
        ];
        let d = build_dashboard(&lines, now, 10, true);
        assert_eq!(d.windows.len(), 3);
        let prov = d.windows.iter().find(|w| w.reason == "Quota").unwrap();
        assert_eq!(prov.kind, "provider");
        assert_eq!(prov.count, 2);
        let cd = d.windows.iter().find(|w| w.reason == "CloudDown").unwrap();
        assert_eq!(cd.kind, "provider");
        assert_eq!(cd.count, 1);
        let bud = d.windows.iter().find(|w| w.kind == "budget").unwrap();
        assert_eq!(bud.count, 1);
    }

    #[test]
    fn dashboard_recent_is_capped() {
        let now = 100i64;
        let entries: Vec<LogLine> = (0..20).map(|i| LogLine::Decision(e(now - i, "local", 1, 1))).collect();
        let d = build_dashboard(&entries, now, 5, true);
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

    #[test]
    fn feedback_lines_parse_and_join_into_recent_rows() {
        let now = 1_000i64;
        let lines = vec![
            LogLine::Decision(RouteEntry { ts: now, rid: "a".into(), surface: "openai".into(),
                dest: "local".into(), ..Default::default() }),
            LogLine::Feedback(FeedbackEntry { rid: "a".into(), ts: now + 1, signal: "cascade".into() }),
            LogLine::Feedback(FeedbackEntry { rid: "a".into(), ts: now + 2, signal: "reask".into() }),
            LogLine::Decision(RouteEntry { ts: now, rid: "b".into(), surface: "openai".into(),
                dest: "local".into(), ..Default::default() }),
        ];
        let d = build_dashboard(&lines, now, 10, true);
        let row_a = d.recent.iter().find(|r| r.entry.rid == "a").unwrap();
        assert_eq!(row_a.feedback, vec!["cascade".to_string(), "reask".to_string()]);
        let row_b = d.recent.iter().find(|r| r.entry.rid == "b").unwrap();
        assert!(row_b.feedback.is_empty());
    }

    #[test]
    fn feedback_line_round_trips_through_serde() {
        let f = LogLine::Feedback(FeedbackEntry { rid: "x".into(), ts: 5, signal: "truncated".into() });
        let s = serde_json::to_string(&f).unwrap();
        assert!(s.contains("\"kind\":\"f\""), "got: {s}");
        let back: LogLine = serde_json::from_str(&s).unwrap();
        match back {
            LogLine::Feedback(fe) => { assert_eq!(fe.rid, "x"); assert_eq!(fe.signal, "truncated"); }
            other => panic!("expected Feedback, got {other:?}"),
        }
    }

    #[test]
    fn feedback_stats_count_flagged_and_trivial() {
        let now = 1_000_000i64;
        let mk = |rid: &str, dest: &str| LogLine::Decision(RouteEntry {
            ts: now - 10, rid: rid.into(), surface: "openai".into(),
            dest: dest.into(), score: 0.3, ..Default::default() });
        let fb = |rid: &str, sig: &str| LogLine::Feedback(FeedbackEntry {
            rid: rid.into(), ts: now - 9, signal: sig.into() });
        let lines = vec![
            mk("l1", "local"), fb("l1", "cascade"),
            mk("l2", "local"), fb("l2", "truncated"), fb("l2", "reask"), // flagged once, not twice
            mk("l3", "local"),
            mk("c1", "cloud"), fb("c1", "cloud_trivial"),
            mk("c2", "cloud"),
        ];
        let d = build_dashboard(&lines, now, 10, true);
        assert_eq!(d.feedback.local_total, 3);
        assert_eq!(d.feedback.local_flagged, 2);
        assert_eq!(d.feedback.cloud_total, 2);
        assert_eq!(d.feedback.cloud_trivial, 1);
    }

    #[test]
    fn suggestion_lower_triggers_on_flagged_locals() {
        let now = 1_000_000i64;
        let mut lines = Vec::new();
        // 20 local decisions in the 7d window; 4 flagged (20% > 15%).
        for i in 0..20 {
            let rid = format!("l{i}");
            // Flagged locals get scores 0.30/0.32/0.34/0.36 → p25 = 0.315-ish.
            let score = if i < 4 { 0.30 + 0.02 * i as f64 } else { 0.10 };
            lines.push(LogLine::Decision(RouteEntry { ts: now - 100, rid: rid.clone(),
                surface: "openai".into(), dest: "local".into(), score, ..Default::default() }));
            if i < 4 {
                lines.push(LogLine::Feedback(FeedbackEntry { rid, ts: now - 99, signal: "cascade".into() }));
            }
        }
        let d = build_dashboard(&lines, now, 10, true);
        let s = d.suggestion.expect("expected a suggestion");
        assert_eq!(s.direction, "lower");
        assert!(s.suggested >= 0.20 && s.suggested <= 0.80);
        assert!(s.suggested <= 0.36, "should sit within the flagged score range, got {}", s.suggested);
    }

    #[test]
    fn suggestion_raise_triggers_on_trivial_clouds() {
        let now = 1_000_000i64;
        let mut lines = Vec::new();
        // 20 healthy locals (sample floor) + 10 clouds of which 4 trivial (40% > 30%).
        for i in 0..20 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 100, rid: format!("l{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.1, ..Default::default() }));
        }
        for i in 0..10 {
            let rid = format!("c{i}");
            let score = if i < 4 { 0.50 + 0.02 * i as f64 } else { 0.90 };
            lines.push(LogLine::Decision(RouteEntry { ts: now - 100, rid: rid.clone(),
                surface: "openai".into(), dest: "cloud".into(), score, ..Default::default() }));
            if i < 4 {
                lines.push(LogLine::Feedback(FeedbackEntry { rid, ts: now - 99, signal: "cloud_trivial".into() }));
            }
        }
        let d = build_dashboard(&lines, now, 10, true);
        let s = d.suggestion.expect("expected a suggestion");
        assert_eq!(s.direction, "raise");
    }

    #[test]
    fn suggestion_none_when_sample_small_or_not_balanced() {
        let now = 1_000_000i64;
        // Only 5 locals, all flagged — sample below 20 → None.
        let mut lines = Vec::new();
        for i in 0..5 {
            let rid = format!("l{i}");
            lines.push(LogLine::Decision(RouteEntry { ts: now - 100, rid: rid.clone(),
                surface: "openai".into(), dest: "local".into(), score: 0.3, ..Default::default() }));
            lines.push(LogLine::Feedback(FeedbackEntry { rid, ts: now - 99, signal: "cascade".into() }));
        }
        assert!(build_dashboard(&lines, now, 10, true).suggestion.is_none());
        // Enough sample but balanced=false → None.
        for i in 5..25 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 100, rid: format!("l{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.1, ..Default::default() }));
        }
        assert!(build_dashboard(&lines, now, 10, false).suggestion.is_none());
    }

    #[test]
    fn route_entry_local_model_round_trips_and_defaults() {
        let e = RouteEntry { ts: 1, rid: "r".into(), surface: "openai".into(),
            dest: "local".into(), local_model: Some("Owner/Repo/file.gguf".into()),
            ..Default::default() };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"local_model\":\"Owner/Repo/file.gguf\""), "got {s}");
        let back: RouteEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back.local_model.as_deref(), Some("Owner/Repo/file.gguf"));
        // Legacy line without the field → None.
        let legacy: RouteEntry = serde_json::from_str(
            r#"{"ts":1,"surface":"openai","dest":"local","score":0.1,"prompt_tok":5,"rid":"r"}"#
        ).unwrap();
        assert_eq!(legacy.local_model, None);
    }

    #[test]
    fn suggestion_ignores_signals_older_than_7_days() {
        let now = 1_000_000_000i64;
        let mut lines = Vec::new();
        // 20 flagged locals, but 8 days old → outside the window → None.
        for i in 0..20 {
            let rid = format!("l{i}");
            let ts = now - 8 * 86_400;
            lines.push(LogLine::Decision(RouteEntry { ts, rid: rid.clone(),
                surface: "openai".into(), dest: "local".into(), score: 0.3, ..Default::default() }));
            lines.push(LogLine::Feedback(FeedbackEntry { rid, ts: ts + 1, signal: "cascade".into() }));
        }
        assert!(build_dashboard(&lines, now, 10, true).suggestion.is_none());
    }

    #[test]
    fn model_capability_estimates_and_filters() {
        let now = 1_000_000i64;
        // Model key that scans to 3.0 B nominal (params_b_for_key: "*-3b*").
        let m = "Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf";
        let mut lines = Vec::new();
        // 40 locals for model m; 12 flagged (30% flag rate).
        for i in 0..40 {
            let rid = format!("m{i}");
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: rid.clone(),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some(m.into()), ..Default::default() }));
            if i < 12 {
                lines.push(LogLine::Feedback(FeedbackEntry { rid, ts: now - 9, signal: "cascade".into() }));
            }
        }
        // A second model with too few locals → omitted.
        for i in 0..5 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: format!("s{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some("foo/tiny-1b.gguf".into()), ..Default::default() }));
        }
        let d = build_dashboard(&lines, now, 10, true);
        assert_eq!(d.model_capabilities.len(), 1, "only the 40-local model qualifies");
        let mc = &d.model_capabilities[0];
        assert_eq!(mc.model, m);
        assert_eq!(mc.local_total, 40);
        assert!((mc.flagged_rate - 0.30).abs() < 1e-9);
        assert!((mc.nominal_b - 3.0).abs() < 1e-9);
        // effective_b: excess=0.30-0.10=0.20; delta=-(0.20*0.5)/0.03 = -3.333..;
        // 3.0 + (-3.333) = -0.333 → clamp low 0.5.
        assert!((mc.effective_b - 0.5).abs() < 1e-9, "got {}", mc.effective_b);
    }

    #[test]
    fn model_capability_high_flag_clamps_low_and_healthy_above_nominal() {
        let now = 1_000_000i64;
        let m = "Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf";
        let mut lines = Vec::new();
        // 40 locals, ZERO flagged → excess = -0.10; delta = +(0.10*0.5)/0.03 = +1.667;
        // effective = 3.0 + 1.667 = 4.667 (< clamp high 3.0*1.5=4.5) → clamps to 4.5.
        for i in 0..40 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: format!("h{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some(m.into()), ..Default::default() }));
        }
        let d = build_dashboard(&lines, now, 10, true);
        let mc = &d.model_capabilities[0];
        assert!((mc.flagged_rate - 0.0).abs() < 1e-9);
        assert!((mc.effective_b - 4.5).abs() < 1e-9, "got {}", mc.effective_b);
    }

    #[test]
    fn model_capability_skips_unknown_nominal() {
        let now = 1_000_000i64;
        let mut lines = Vec::new();
        // 40 locals but the key has no size token and isn't in catalog → nominal 0 → skip.
        for i in 0..40 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: format!("u{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some("foo/mystery-model.gguf".into()), ..Default::default() }));
        }
        let d = build_dashboard(&lines, now, 10, true);
        assert!(d.model_capabilities.is_empty());
    }
}
