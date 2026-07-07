# Fase D1 — Routing Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-request routing-quality signals (cascade, truncated, cloud_trivial, reask) as a third route-log line kind, meter cloud relay responses, and surface an accuracy card, per-row markers, and a data-driven threshold suggestion on the dashboard.

**Architecture:** `route_log.rs` gains `LogLine::Feedback` ("f") plus aggregation (`FeedbackStats`, `ThresholdSuggestion`) inside `build_dashboard`. `cloud.rs` gains an optional `RelayMeter` that wraps the relayed body stream to record cloud outcomes (TTFT, est. completion) and emit `cloud_trivial`. `server.rs` emits `cascade`/`truncated` at existing sites and detects `reask` via an in-memory per-surface prompt buffer consulted in `route_decision`. The dashboard UI renders the three new pieces.

**Tech Stack:** Rust (axum 0.7, serde, futures), vanilla-JS manager UI. No new crates.

## Global Constraints

- Signal strings are exactly: `"cascade"`, `"reask"`, `"truncated"`, `"cloud_trivial"`.
- Feedback line serde tag is `"f"` (alongside existing `"d"`/`"o"`).
- Re-ask detector: Jaccard > 0.6, window < 120 s, buffer cap 8 per surface, only LOCAL decisions enter the buffer, first match wins, one reask line max per new request.
- `cloud_trivial`: estimated completion tokens < 40 (`bytes / 4`).
- Suggestion: 7-day window, minimum 20 local decisions, lower-trigger at flagged-rate > 0.15, raise-trigger at trivial-rate > 0.30 AND flagged-rate <= 0.15, lower wins if both, suggested value = p25 (lower) / p75 (raise) of the relevant scores, clamped to [0.20, 0.80], only when the active profile is Balanced.
- All tunables live as named consts in `route_log.rs` (aggregation) / `server.rs` (detector) / `cloud.rs` (trivial cutoff) so D2 can tune them.
- The re-ask buffer is in-memory only (lost on restart — by design).
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
  ```
- Build machine is slow (cold link ~6 min, warm ~10-13 s). Run cargo in the background or with long timeouts.
- Branch: `feat/fase-d1` off `feat/fase-c`.

---

### Task 1: `LogLine::Feedback` line kind + join into `RecentRow`

**Files:**
- Modify: `src/route_log.rs` (structs ~line 54-87, append fns ~97-105, `RecentRow` ~221-233, `build_dashboard` join ~328-338, tests)

**Interfaces:**
- Consumes: existing `LogLine` tagged enum ("d"/"o"), `append_line`, `read_all`, `RecentRow`, `build_dashboard`.
- Produces (later tasks rely on these exact names):
  - `pub struct FeedbackEntry { pub rid: String, pub ts: i64, pub signal: String }`
  - `LogLine::Feedback(FeedbackEntry)` with serde tag `"f"`
  - `pub fn append_feedback(entry: &FeedbackEntry)`
  - `RecentRow.feedback: Vec<String>` (serialized; empty vec omitted is NOT required — always serialize)

- [ ] **Step 1: Write the failing tests**

In `src/route_log.rs` `#[cfg(test)] mod tests`, add:

```rust
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
        let d = build_dashboard(&lines, now, 10);
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
```

Note: `build_dashboard` still takes 3 args in this task (the 4th comes in Task 2).

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib feedback_lines_parse -- --nocapture`
Expected: FAIL to compile — `FeedbackEntry` / `LogLine::Feedback` / `.feedback` not defined.

- [ ] **Step 3: Implement**

a) After the `OutcomeEntry` struct (~line 68), add:

```rust
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
```

b) Extend `LogLine`:

```rust
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
```

And extend `LogLine::ts()`:

```rust
    pub fn ts(&self) -> i64 {
        match self {
            LogLine::Decision(d) => d.ts,
            LogLine::Outcome(o) => o.ts,
            LogLine::Feedback(f) => f.ts,
        }
    }
```

c) After `append_outcome` (~line 105), add:

```rust
/// Append one feedback signal as a JSON line. Best-effort; never panics.
pub fn append_feedback(entry: &FeedbackEntry) {
    append_line(&LogLine::Feedback(entry.clone()));
}
```

d) `RecentRow` gains the field (after `cost_saved_usd`):

```rust
    pub cost_saved_usd: f64,
    /// Quality signals recorded against this decision (may be empty).
    #[serde(default)]
    pub feedback: Vec<String>,
}
```

e) In `build_dashboard`: collect feedback per rid and join. In the initial
scan loop (`for l in entries { match l { ... } }`), add an arm and a map:

```rust
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
```

(The existing match may use a different second-arm body — keep its behavior;
only ADD the Feedback arm and the `feedback` map. If the match is non-exhaustive
after adding the variant, the compiler will point at every site to fix; extend
each with a `Feedback` arm that does the sensible minimal thing — e.g. the CSV
export in `server.rs` skips or emits a "f" row consistent with its header. For
the export specifically: emit `kind` "f" with rid/ts/signal in the `reason`
column and other columns empty.)

In the `recent` construction, populate the new field:

```rust
        RecentRow {
            entry: (*d).clone(),
            completion_tok: o.and_then(|o| o.completion_tok),
            ttft_ms: o.and_then(|o| o.ttft_ms),
            gen_ms: o.and_then(|o| o.gen_ms),
            cost_saved_usd: o.map(|o| o.cost_saved_usd).unwrap_or(0.0),
            feedback: feedback.get(d.rid.as_str()).cloned().unwrap_or_default(),
        }
```

- [ ] **Step 4: Fix non-exhaustive-match fallout across the crate**

Run: `cargo build 2>&1 | grep -E "^error" | head -20`
Fix every `match` over `LogLine` that the new variant broke (known candidates:
`server.rs` export_csv ~line 985-1000, `budget.rs::seed_from_log`, metrics).
For `seed_from_log` and metrics: `LogLine::Feedback(_) => {}` (ignore).
For export_csv: emit a row with `kind` "f", the rid, ts, and the signal in the
`reason` column, remaining columns empty strings.

- [ ] **Step 5: Run tests to verify pass**

Run: `cargo test --lib route_log -- --nocapture`
Expected: PASS, including the two new tests.

Run: `cargo build`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add src/route_log.rs src/server.rs src/budget.rs
git commit -F - <<'EOF'
feat(feedback): LogLine::Feedback line kind joined into recent rows

Third route-log line kind "f" (rid, ts, signal) with append_feedback;
RecentRow gains feedback: Vec<String> joined by rid in build_dashboard.
CSV export emits f rows; budget/metrics scans ignore them.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 2: `FeedbackStats` + `ThresholdSuggestion` aggregation

**Files:**
- Modify: `src/route_log.rs` (Dashboard struct ~235-245, `build_dashboard` ~268, new consts + percentile helper, tests)
- Modify: `src/server.rs` (`handle_dashboard` ~991-1002 — pass the new arg)

**Interfaces:**
- Consumes from Task 1: the `feedback: HashMap<&str, Vec<String>>` map inside `build_dashboard`.
- Produces:
  - `pub struct FeedbackStats { pub local_total: u64, pub local_flagged: u64, pub cloud_total: u64, pub cloud_trivial: u64 }`
  - `pub struct ThresholdSuggestion { pub direction: String, pub suggested: f64, pub why: String }`
  - `build_dashboard(entries: &[LogLine], now: i64, recent_n: usize, balanced: bool) -> Dashboard` (4th param NEW)
  - `Dashboard.feedback: FeedbackStats` and `Dashboard.suggestion: Option<ThresholdSuggestion>`

- [ ] **Step 1: Write the failing tests**

```rust
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
```

Also update the THREE existing `build_dashboard(` calls in route_log tests
(`dashboard_groups_fallback_windows`, `dashboard_recent_is_capped`, and the
Task-1 test `feedback_lines_parse_and_join_into_recent_rows`) to pass a 4th
arg `true`.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib suggestion_ -- --nocapture`
Expected: FAIL to compile — 4th param / `FeedbackStats` / `suggestion` missing.

- [ ] **Step 3: Implement**

a) New public types + consts near `RouteLatency` (~line 200):

```rust
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
```

b) `Dashboard` gains:

```rust
    pub recent: Vec<RecentRow>,
    pub feedback: FeedbackStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<ThresholdSuggestion>,
}
```

c) `build_dashboard` signature + logic:

```rust
pub fn build_dashboard(entries: &[LogLine], now: i64, recent_n: usize, balanced: bool) -> Dashboard {
```

After the existing rollups (before constructing `Dashboard`), compute stats
over the FULL decisions list (30d file retention bounds it) and the suggestion
over the 7d subset:

```rust
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
```

And include `feedback: stats, suggestion,` in the returned `Dashboard`.

d) `src/server.rs` `handle_dashboard` passes the profile flag:

```rust
    let entries = crate::route_log::read_all();
    let balanced = crate::settings::load_profile() == crate::route::Profile::Balanced;
    let dash = crate::route_log::build_dashboard(&entries, crate::route_log::now_secs(), 50, balanced);
    Json(dash).into_response()
```

(Verify `load_profile()`'s exact return type at implementation time — it may
return `Profile` directly or an Option; adapt the comparison accordingly.)

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test --lib route_log -- --nocapture`
Expected: PASS (all new + existing).

Run: `cargo build`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/route_log.rs src/server.rs
git commit -F - <<'EOF'
feat(feedback): accuracy stats + threshold suggestion in dashboard

FeedbackStats rollup (local flagged / cloud trivial) over the retained
log and a 7d ThresholdSuggestion (p25/p75 of offending scores, clamped
0.20-0.80, Balanced-only, 20-local minimum). build_dashboard gains a
`balanced` flag passed from the active profile.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 3: Cloud relay metering (`RelayMeter`) + `cloud_trivial`

**Files:**
- Modify: `src/cloud.rs` (forward signature ~72, stream wrap ~113-116, new MeteredStream, tests ~119+)
- Modify: `src/server.rs` (5 `cloud::forward(` call sites: 2 in `cascade_or_result` ~473/493, 3 relay sites ~1492/1686/1796 — line numbers drift; find by `crate::cloud::forward(`)

**Interfaces:**
- Consumes: `crate::route_log::{append_outcome, append_feedback, OutcomeEntry, FeedbackEntry, now_secs}`, `crate::route::estimate_text_tokens` semantics (bytes/4 — here applied to byte counts directly).
- Produces:
  - `pub struct RelayMeter { pub rid: String }`
  - `pub async fn forward(provider, upstream_path, headers, body, meter: Option<RelayMeter>) -> ForwardOutcome` (5th param NEW)
  - `pub const CLOUD_TRIVIAL_MAX_TOK: u64 = 40;`

- [ ] **Step 1: Write the failing test**

In `src/cloud.rs` `mod tests` (wiremock is already a dev-dep — the module has
`MockServer` tests; also note the existing tests call `forward(...)` with 4
args and must gain a `None` 5th arg in this task):

```rust
    #[tokio::test]
    async fn metered_relay_records_outcome_and_trivial_flag() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-meter-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);

        let server = MockServer::start().await;
        // Tiny body: 20 bytes → est 5 tokens < 40 → trivial.
        Mock::given(method("POST")).and(path_regex(".*"))
            .respond_with(ResponseTemplate::new(200).set_body_string("01234567890123456789"))
            .mount(&server).await;
        let provider = test_provider(&server.uri()); // reuse the module's existing helper for pointing Provider at the mock; if none exists, follow the pattern of the existing forward tests
        let out = forward(provider, "/v1/messages", &HeaderMap::new(), Bytes::from("{}"),
            Some(RelayMeter { rid: "rm1".into() })).await;
        let resp = match out { ForwardOutcome::Relayed(r) => r, _ => panic!("expected relay") };
        // Drain the body so the metered stream completes.
        let _ = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();

        let lines = crate::route_log::read_all();
        let outcome = lines.iter().find_map(|l| match l {
            crate::route_log::LogLine::Outcome(o) if o.rid == "rm1" => Some(o.clone()), _ => None });
        let outcome = outcome.expect("expected a cloud outcome line");
        assert_eq!(outcome.completion_tok, Some(5)); // 20 bytes / 4
        assert!(outcome.ttft_ms.is_some());
        let trivial = lines.iter().any(|l| matches!(l,
            crate::route_log::LogLine::Feedback(f) if f.rid == "rm1" && f.signal == "cloud_trivial"));
        assert!(trivial, "expected cloud_trivial feedback line");

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

IMPORTANT adaptation note for the implementer: look at how the EXISTING cloud
tests construct a `Provider` against `server.uri()` (there is an established
pattern in this module — e.g. an env-based base-url override or a test
Provider variant). Use that same mechanism; do not invent a new one. If the
existing tests use `path("/v1/messages")` matchers, mirror them instead of
`path_regex`.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib metered_relay -- --nocapture`
Expected: FAIL to compile (forward has no 5th param; RelayMeter undefined).

- [ ] **Step 3: Implement the meter**

In `src/cloud.rs`:

```rust
/// Estimated-completion cutoff (tokens ≈ bytes/4) below which a cloud reply is
/// flagged "cloud_trivial". Named so Fase D2 can tune it. Note the estimate is
/// inflated by JSON/SSE framing, biasing AGAINST false trivial flags.
pub const CLOUD_TRIVIAL_MAX_TOK: u64 = 40;

/// Attach to `forward` to record the relayed response as a cloud OutcomeEntry
/// (TTFT + estimated completion tokens) and emit "cloud_trivial" feedback.
pub struct RelayMeter {
    pub rid: String,
}

/// Body-stream wrapper that counts bytes and, when the upstream stream ends,
/// writes the outcome + feedback lines. If the client disconnects mid-stream
/// the wrapper is dropped without reaching the end → no outcome (accepted).
struct MeteredStream<S> {
    inner: S,
    rid: String,
    started: std::time::Instant,
    first_chunk_ms: Option<u64>,
    bytes: u64,
    done: bool,
}

impl<S, E> futures::Stream for MeteredStream<S>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = &mut *self;
        match std::pin::Pin::new(&mut this.inner).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                if this.first_chunk_ms.is_none() {
                    this.first_chunk_ms = Some(this.started.elapsed().as_millis() as u64);
                }
                this.bytes += chunk.len() as u64;
                std::task::Poll::Ready(Some(Ok(chunk)))
            }
            std::task::Poll::Ready(None) => {
                if !this.done {
                    this.done = true;
                    let est_tok = this.bytes / 4;
                    crate::route_log::append_outcome(&crate::route_log::OutcomeEntry {
                        rid: this.rid.clone(),
                        ts: crate::route_log::now_secs(),
                        completion_tok: Some(est_tok),
                        ttft_ms: this.first_chunk_ms,
                        gen_ms: Some(this.started.elapsed().as_millis() as u64),
                        cost_saved_usd: 0.0,
                    });
                    if est_tok < CLOUD_TRIVIAL_MAX_TOK {
                        crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                            rid: this.rid.clone(),
                            ts: crate::route_log::now_secs(),
                            signal: "cloud_trivial".to_string(),
                        });
                    }
                }
                std::task::Poll::Ready(None)
            }
            other => other,
        }
    }
}
```

Change `forward`'s signature and the Relayed construction:

```rust
pub async fn forward(
    provider: Provider,
    upstream_path: &str,
    headers: &HeaderMap,
    body: Bytes,
    meter: Option<RelayMeter>,
) -> ForwardOutcome {
```

```rust
    let stream = upstream.bytes_stream();
    let mut builder = Response::builder().status(status);
    *builder.headers_mut().unwrap() = resp_headers;
    match meter {
        Some(m) => {
            let metered = MeteredStream {
                inner: stream,
                rid: m.rid,
                started: std::time::Instant::now(),
                first_chunk_ms: None,
                bytes: 0,
                done: false,
            };
            ForwardOutcome::Relayed(builder.body(axum::body::Body::from_stream(metered)).unwrap())
        }
        None => ForwardOutcome::Relayed(builder.body(axum::body::Body::from_stream(stream)).unwrap()),
    }
```

(TTFT here is measured from response-header receipt, not request start — name
the limitation in a comment; the decision-to-first-byte gap is captured well
enough for dashboard purposes.)

- [ ] **Step 4: Update all forward call sites**

a) The 3 relay sites in `server.rs` (search `crate::cloud::forward(`; they look
like `crate::cloud::forward(crate::cloud::Provider::OpenAI, "/v1/chat/completions", &headers, raw.clone()).await`):
append `Some(crate::cloud::RelayMeter { rid: rid.to_string() })` as the 5th arg.
Use the rid variable in scope at each site (`rid`).

b) The 2 cascade sites in `cascade_or_result` (~473/493): the function has
`rid: &str` in scope; append `Some(crate::cloud::RelayMeter { rid: rid.to_string() })`.

c) The existing cloud.rs tests calling `forward(...)`: append `None`.

- [ ] **Step 5: Run tests to verify pass**

Run: `cargo test --lib cloud -- --nocapture`
Expected: PASS including `metered_relay_records_outcome_and_trivial_flag`.

Run: `cargo build`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add src/cloud.rs src/server.rs
git commit -F - <<'EOF'
feat(feedback): meter cloud relays; emit cloud_trivial + real cloud outcomes

forward() optionally wraps the relayed body in a byte-counting stream
that records a cloud OutcomeEntry (TTFT, est completion = bytes/4) and a
cloud_trivial feedback line (<40 tok) when the stream completes. All
five forward sites pass a RelayMeter; cloud latency on the dashboard now
gets real data.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 4: `cascade` + `truncated` emission in server.rs

**Files:**
- Modify: `src/server.rs` (`record_outcome` ~258-280 gains a finish param; its 6 local call sites; the 2 cascade arms in `cascade_or_result`; the record_outcome unit test ~2119-2121; new wiring tests)

**Interfaces:**
- Consumes: `crate::route_log::{append_feedback, FeedbackEntry}`, `crate::api::common::FinishReason`, Task 1's line kind.
- Produces: `record_outcome(rid, dest, model, prompt_tok, completion_tok, ttft_ms, gen_ms, finish: Option<crate::api::common::FinishReason>)` (8th param NEW).

- [ ] **Step 1: Write the failing tests**

In `src/server.rs` `mod tests`:

```rust
    #[test]
    fn record_outcome_emits_truncated_for_local_length() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-tr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);

        use crate::api::common::FinishReason;
        super::record_outcome("t1", "local", None, 10, Some(10), None, None, Some(FinishReason::Length));
        super::record_outcome("t2", "local", None, 10, Some(10), None, None, Some(FinishReason::Stop));
        super::record_outcome("t3", "cloud", None, 10, Some(10), None, None, Some(FinishReason::Length));

        let lines = crate::route_log::read_all();
        let has_trunc = |rid: &str| lines.iter().any(|l| matches!(l,
            crate::route_log::LogLine::Feedback(f) if f.rid == rid && f.signal == "truncated"));
        assert!(has_trunc("t1"), "local Length must emit truncated");
        assert!(!has_trunc("t2"), "local Stop must not");
        assert!(!has_trunc("t3"), "cloud Length must not (cloud is not judged here)");

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib record_outcome_emits_truncated -- --nocapture`
Expected: FAIL to compile — record_outcome takes 7 args.

- [ ] **Step 3: Implement**

a) `record_outcome` gains the param and the emission:

```rust
fn record_outcome(
    rid: &str,
    dest: &str,
    model: Option<&str>,
    prompt_tok: u64,
    completion_tok: Option<u64>,
    ttft_ms: Option<u64>,
    gen_ms: Option<u64>,
    finish: Option<crate::api::common::FinishReason>,
) {
    // ...existing body unchanged...
    // Weak-local signal: a local answer that hit the length cap and was served
    // as-is (cascade escalations never reach this path with Length — they
    // escalate instead).
    if dest == "local" && finish == Some(crate::api::common::FinishReason::Length) {
        crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
            rid: rid.to_string(),
            ts: crate::route_log::now_secs(),
            signal: "truncated".to_string(),
        });
    }
}
```

(`FinishReason` derives `PartialEq` — verify; if not, add `PartialEq` to its
derive list in `src/api/common.rs`. It currently derives Debug/Clone/PartialEq
per the struct definitions around it — confirm.)

b) Update all 6 local call sites (search `record_outcome(&rid`): the 5
non-stream sites have `result: ChatResult` in scope → pass
`Some(result.finish_reason.clone())` (FinishReason is small; if it derives
`Copy`, drop the clone). The OpenAI stream site (inside the `scan` closure,
`delta.done` branch) has the final `delta` in scope → pass
`delta.finish_reason.clone()` (it is already an `Option<FinishReason>` — pass
it directly, no `Some(...)` wrap).

c) The existing unit test `record_outcome_saves_cost_only_for_local` calls with
7 args → append `None`.

d) Cascade signal — in `cascade_or_result`, BOTH arms that escalate to cloud
(the ones calling `crate::cloud::forward`), immediately before the `match
crate::cloud::forward(...)` call, add:

```rust
                crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                    rid: rid.to_string(),
                    ts: crate::route_log::now_secs(),
                    signal: "cascade".to_string(),
                });
```

Emit BEFORE the forward (the judgment "local was too weak" holds regardless of
whether the escalation then succeeds or degrades).

Add a wiring test:

```rust
    #[tokio::test]
    async fn cascade_escalation_emits_cascade_feedback() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-cf-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &path);

        // Weak local result (finish=Length) + cascade wanted → escalates; the
        // unreachable upstream degrades, but the cascade signal must be logged.
        let weak = Ok(crate::api::common::ChatResult {
            content: vec![crate::api::common::ContentPart::Text("x".into())],
            finish_reason: crate::api::common::FinishReason::Length,
            prompt_tokens: 1,
            completion_tokens: 1,
        });
        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        let state = wiring_state(breaker);
        let _ = super::cascade_or_result(
            true, weak, crate::cloud::Provider::OpenAI, "/v1/chat/completions",
            &axum::http::HeaderMap::new(), bytes::Bytes::from("{}"), &state, 1, "rc1", "openai",
        ).await;
        let lines = crate::route_log::read_all();
        assert!(lines.iter().any(|l| matches!(l,
            crate::route_log::LogLine::Feedback(f) if f.rid == "rc1" && f.signal == "cascade")));

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

(Reuses Task-C's `wiring_state` helper already in `mod tests`. `is_weak_result`
must treat finish=Length as weak — it does; that is its definition. The forward
to the real OpenAI URL will fail as Offline in the test environment → Degrade →
result kept; the cascade line is already written. If the test environment
somehow reaches a network, the mock-free forward still exercises the same
pre-forward emission. `bytes::Bytes` — use the `Bytes` type already imported in
server.rs.)

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test --lib record_outcome_emits_truncated cascade_escalation_emits -- --nocapture`
Expected: PASS.

Run: `cargo build`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs src/api/common.rs
git commit -F - <<'EOF'
feat(feedback): emit cascade + truncated signals

record_outcome carries the finish reason and logs "truncated" for local
Length completions; both cascade escalation arms log "cascade" before
forwarding. Wiring tests cover local/cloud/Stop/Length matrix and the
cascade path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 5: Re-ask detector (in-memory buffer)

**Files:**
- Modify: `src/history_select.rs` (make `tokenize` pub(crate), line ~23)
- Modify: `src/server.rs` (AppState field ~537-570 + both test-state constructors `make_seeded_test_router` and `wiring_state` + `router()` AppState literal; detector fns + `route_decision` wiring; tests)
- Modify: `src/lib.rs` (`router_for_test_with` — only if it builds AppState directly; it calls `server::router()` which owns the literal, so likely no change)

**Interfaces:**
- Consumes: `crate::history_select::tokenize` (make `pub(crate) fn tokenize(text: &str) -> Vec<String>`), Task 1's `append_feedback`.
- Produces:
  - `pub struct RecentPrompt { pub rid: String, pub ts: i64, pub tokens: std::collections::BTreeSet<String> }`
  - `AppState.recent_prompts: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, std::collections::VecDeque<RecentPrompt>>>>`
  - `fn detect_reask(buf: &std::collections::VecDeque<RecentPrompt>, tokens: &std::collections::BTreeSet<String>, now: i64) -> Option<String>` (returns the matched PREVIOUS rid)
  - `const REASK_JACCARD: f64 = 0.6; const REASK_WINDOW_SECS: i64 = 120; const REASK_BUFFER_CAP: usize = 8;`

- [ ] **Step 1: Make the tokenizer reusable**

In `src/history_select.rs` line ~23: change `fn tokenize` to
`pub(crate) fn tokenize`. (Its rules — split non-alphanumeric, drop empties,
lowercase — match the spec except the "drop tokens < 2 chars" filter, which the
detector applies itself; do NOT change the tokenizer's behavior.)

- [ ] **Step 2: Write the failing detector tests**

In `src/server.rs` `mod tests`:

```rust
    fn tokset(s: &str) -> std::collections::BTreeSet<String> {
        crate::history_select::tokenize(s).into_iter().filter(|t| t.len() >= 2).collect()
    }

    #[test]
    fn detect_reask_matches_similar_recent_prompt() {
        use std::collections::VecDeque;
        let mut buf: VecDeque<super::RecentPrompt> = VecDeque::new();
        buf.push_front(super::RecentPrompt { rid: "old1".into(), ts: 1000,
            tokens: tokset("como faço deploy do serviço no kubernetes") });
        // Near-identical re-ask 30s later → match.
        let t = tokset("como faço deploy do serviço no kubernetes agora");
        assert_eq!(super::detect_reask(&buf, &t, 1030), Some("old1".to_string()));
        // Different topic → no match.
        let t2 = tokset("escreva um poema sobre gatos persas");
        assert_eq!(super::detect_reask(&buf, &t2, 1030), None);
        // Same prompt but 3 minutes later → outside window.
        assert_eq!(super::detect_reask(&buf, &t, 1000 + 181), None);
        // Empty token set never matches.
        let empty = std::collections::BTreeSet::new();
        assert_eq!(super::detect_reask(&buf, &empty, 1030), None);
    }

    #[test]
    fn detect_reask_first_match_wins() {
        use std::collections::VecDeque;
        let mut buf: VecDeque<super::RecentPrompt> = VecDeque::new();
        // Newest first: both similar; the front (newest) must win.
        buf.push_front(super::RecentPrompt { rid: "older".into(), ts: 990,
            tokens: tokset("erro de compilação no módulo de rede") });
        buf.push_front(super::RecentPrompt { rid: "newer".into(), ts: 1000,
            tokens: tokset("erro de compilação no módulo de rede") });
        let t = tokset("erro de compilação no módulo de rede ainda");
        assert_eq!(super::detect_reask(&buf, &t, 1010), Some("newer".to_string()));
    }
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test --lib detect_reask -- --nocapture`
Expected: FAIL to compile.

- [ ] **Step 4: Implement the detector + AppState field**

In `src/server.rs`, near `route_decision`:

```rust
/// Re-ask detector tunables — named so Fase D2 can calibrate them.
const REASK_JACCARD: f64 = 0.6;
const REASK_WINDOW_SECS: i64 = 120;
const REASK_BUFFER_CAP: usize = 8;

/// One recent LOCAL decision's prompt fingerprint for re-ask detection.
#[derive(Debug, Clone)]
pub struct RecentPrompt {
    pub rid: String,
    pub ts: i64,
    pub tokens: std::collections::BTreeSet<String>,
}

/// Scan newest-first for a prior prompt whose token set is Jaccard-similar
/// (> REASK_JACCARD) within REASK_WINDOW_SECS. Returns the matched rid.
fn detect_reask(
    buf: &std::collections::VecDeque<RecentPrompt>,
    tokens: &std::collections::BTreeSet<String>,
    now: i64,
) -> Option<String> {
    if tokens.is_empty() {
        return None;
    }
    for old in buf {
        if now - old.ts >= REASK_WINDOW_SECS || old.tokens.is_empty() {
            continue;
        }
        let inter = tokens.intersection(&old.tokens).count() as f64;
        let union = tokens.union(&old.tokens).count() as f64;
        if union > 0.0 && inter / union > REASK_JACCARD {
            return Some(old.rid.clone());
        }
    }
    None
}
```

AppState gains (after `breaker`):

```rust
    /// Per-surface ring of recent LOCAL decisions (rid, ts, prompt token set)
    /// for re-ask detection. In-memory; cap REASK_BUFFER_CAP per surface.
    pub recent_prompts: std::sync::Arc<std::sync::Mutex<
        std::collections::BTreeMap<String, std::collections::VecDeque<RecentPrompt>>>>,
```

Initialize in ALL AppState literals (router()'s, `make_seeded_test_router`'s,
`wiring_state`'s):

```rust
        recent_prompts: std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )),
```

- [ ] **Step 5: Wire into `route_decision`**

At the end of `route_decision`, right BEFORE the final `(decision, prompt_tokens)`
return (the RouteEntry has already been appended), add:

```rust
    // Re-ask detection: judge the PREVIOUS local decision if this request is a
    // near-duplicate sent shortly after it, then remember this request when it
    // itself routes local. Uses the latest user turn only.
    {
        let last_user_text = internal
            .messages
            .iter()
            .rev()
            .find(|m| m.role == crate::api::common::Role::User)
            .and_then(|m| m.text.clone())
            .unwrap_or_default();
        let tokens: std::collections::BTreeSet<String> =
            crate::history_select::tokenize(&last_user_text)
                .into_iter()
                .filter(|t| t.len() >= 2)
                .collect();
        let mut map = state.recent_prompts.lock().unwrap_or_else(|e| e.into_inner());
        let buf = map.entry(surface.to_string()).or_default();
        if let Some(prev_rid) = detect_reask(buf, &tokens, now) {
            crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                rid: prev_rid,
                ts: now,
                signal: "reask".to_string(),
            });
        }
        if matches!(decision, crate::route::Decision::Local | crate::route::Decision::LocalThenCascade) {
            buf.push_front(RecentPrompt { rid: rid.to_string(), ts: now, tokens });
            buf.truncate(REASK_BUFFER_CAP);
        }
    }
```

(`now` is the i64 binding from the Fase C fix, already in scope. VERIFY the
real `Decision` variant names in `src/route/mod.rs` — the local variants are
`Local` and, if it exists, the cascade-local variant (`LocalThenCascade` or
similar). Any decision that SERVES LOCALLY first must be pushed; plain
`Decision::Cloud(_)` must not. Adjust the `matches!` to the real enum.)

Add a wiring test:

```rust
    #[test]
    fn reask_buffer_only_keeps_local_decisions_and_caps() {
        use std::collections::VecDeque;
        let mut buf: VecDeque<super::RecentPrompt> = VecDeque::new();
        for i in 0..12 {
            buf.push_front(super::RecentPrompt { rid: format!("r{i}"), ts: 1000 + i,
                tokens: tokset(&format!("prompt número {i} totalmente diferente dos outros assunto{i}")) });
            buf.truncate(super::REASK_BUFFER_CAP);
        }
        assert_eq!(buf.len(), super::REASK_BUFFER_CAP);
        assert_eq!(buf.front().unwrap().rid, "r11"); // newest kept
    }
```

- [ ] **Step 6: Run tests + build**

Run: `cargo test --lib detect_reask reask_buffer -- --nocapture`
Expected: PASS.

Run: `cargo build`
Expected: success.

- [ ] **Step 7: Commit**

```bash
git add src/server.rs src/history_select.rs src/lib.rs
git commit -F - <<'EOF'
feat(feedback): in-memory re-ask detector

Per-surface ring buffer (cap 8) of recent local prompts; a near-duplicate
(Jaccard > 0.6) within 120s judges the previous local decision with a
"reask" feedback line. Reuses the history_select tokenizer.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 6: Dashboard UI — accuracy card, table column, suggestion banner

**Files:**
- Modify: `src/manager_ui/app.js` (`renderDashboard` — card after the latency card; `renderDecisionsTable` — new column; banner after the breaker strip)
- Modify: `src/manager_ui/style.css`

**Interfaces:**
- Consumes: dashboard JSON now carries `feedback: {local_total, local_flagged, cloud_total, cloud_trivial}`, `suggestion: {direction, suggested, why} | absent`, and each `recent[i].feedback: string[]`. Existing helpers `el(tag, cls, html)`, `toast`.
- Produces: UI only.

- [ ] **Step 1: Accuracy card**

In `renderDashboard`, right after the latency card (`wrap.append(lat);`), add:

```javascript
  // Routing accuracy from feedback signals
  const fb = d.feedback || { local_total: 0, local_flagged: 0, cloud_total: 0, cloud_trivial: 0 };
  const acc = el("div", "dash-card");
  acc.append(el("div", "dash-card-head", "ACERTO DO ROUTING"));
  const pct = (num, den) => den > 0 ? Math.round(100 * (1 - num / den)) + "%" : "—";
  acc.append(el("div", "dash-card-alt",
    `local: ${pct(fb.local_flagged, fb.local_total)} ok (${fb.local_total} pedidos)`));
  acc.append(el("div", "dash-card-alt",
    `cloud: ${pct(fb.cloud_trivial, fb.cloud_total)} aproveitada (${fb.cloud_total} pedidos)`));
  wrap.append(acc);
```

- [ ] **Step 2: Suggestion banner**

Right after the breaker strip (`wrap.append(brk); paintBreaker(brk);` block), add:

```javascript
  // Data-driven threshold suggestion (Balanced only; measurement, not applied)
  if (d.suggestion) {
    const sug = el("div", "sug-banner");
    const sugText = el("span", "sug-text");
    sugText.textContent = d.suggestion.why;
    sug.append(sugText);
    const go = el("button", "btn", "Abrir Config");
    go.onclick = () => { location.hash = "#/config"; };
    sug.append(go);
    wrap.append(sug);
  }
```

- [ ] **Step 3: Table column**

In `renderDecisionsTable`, add a header cell `""` (narrow) as the FIRST column
and, per row, a marker cell:

```javascript
    const fbCell = el("td", "fb-cell");
    const signals = r.feedback || [];
    if (signals.length === 0) {
      fbCell.textContent = "✓";
      fbCell.classList.add("fb-ok");
    } else {
      fbCell.textContent = "⚠";
      fbCell.classList.add("fb-warn");
      fbCell.title = signals.join(", ");
    }
    tr.append(fbCell);
```

ADAPT to the table's real construction: read `renderDecisionsTable` first —
it may build rows with `el(...)` appends or an HTML string. Follow its
existing pattern exactly (if it builds `<tr>` HTML strings, add a `<td>`
with the same classes; signals content is our own enum strings, safe).
Keep column order consistent between header and rows.

- [ ] **Step 4: CSS**

Append to `style.css`:

```css
/* Feedback markers in the decisions table */
.fb-cell { text-align: center; width: 28px; }
.fb-ok { color: var(--green); }
.fb-warn { color: var(--amber); cursor: help; }

/* Threshold suggestion banner */
.sug-banner { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  margin: 4px 0 12px; padding: 10px 14px; border-radius: 10px;
  background: var(--panel); border: 1px solid var(--amber); }
.sug-text { font-weight: 600; }
```

- [ ] **Step 5: Build gate + commit**

Run: `cargo build`
Expected: success (assets are string-embedded).

Manual check deferred to the user's end-to-end pass (per standing instruction).

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -F - <<'EOF'
feat(ui): routing accuracy card, feedback column, threshold suggestion banner

Dashboard shows local/cloud accuracy from feedback signals, a per-row
check/warn marker with signal tooltip, and the 7d threshold suggestion
banner linking to Config.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

## Final Verification (after all tasks)

- [ ] `cargo test --lib` (background) → all green (was 228 before D1).
- [ ] `cargo test --test http -- --test-threads=1` → 47 green (pre-existing parallel flake; single-thread is the accepted mode).
- [ ] `cargo build` clean.

## Self-Review Notes

- **Spec coverage:** line kind (T1), stats+suggestion (T2), cloud metering +
  cloud_trivial (T3), cascade+truncated (T4), reask (T5), UI (T6). Spec's
  "Open Risks" constants all appear as named consts.
- **Type consistency:** `FeedbackEntry{rid,ts,signal}`, `append_feedback`,
  `FeedbackStats`, `ThresholdSuggestion{direction,suggested,why}`,
  `build_dashboard(..., balanced: bool)`, `RelayMeter{rid}`,
  `forward(..., meter: Option<RelayMeter>)`, `record_outcome(..., finish:
  Option<FinishReason>)`, `RecentPrompt{rid,ts,tokens}`, `detect_reask(buf,
  tokens, now) -> Option<String>` — used identically across tasks.
- **Known verification points for implementers:** `Decision` local-variant
  names (T5), `load_profile()` return type (T2), cloud test Provider-mock
  pattern (T3), `FinishReason` PartialEq derive (T4), decisions-table row
  construction pattern (T6).
