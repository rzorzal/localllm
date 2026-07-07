# Fase D1 — Routing Feedback Loop (Design Spec)

**Date:** 2026-07-07
**Branch base:** `feat/fase-c` (D1 builds on Fase A observability + Fase C branch stack)
**Status:** Approved for planning

## Goal

Record, per request, whether the routing decision turned out to be good — and
surface that as a dashboard accuracy score, per-row feedback markers, and a
data-driven threshold suggestion. D1 only measures and suggests; automatic
tuning is Fase D2.

## Context

Fase D was decomposed into four sub-projects: **D1 feedback loop (this spec)**
→ D2 auto-calibrate capability_b → D3 multi-local-model tiering → D4 semantic
routing. D1 first because it produces the ground-truth data the others consume.

Existing infrastructure this builds on:
- `route_log.rs` two-line JSONL model: `LogLine::Decision(RouteEntry)` /
  `LogLine::Outcome(OutcomeEntry)` correlated by `rid`, with legacy fallback.
- `cascade_or_result` already detects a weak local result (`is_weak_result`,
  finish=length) and escalates to cloud.
- `record_outcome` runs at every post-generation done-site with
  `completion_tok` and dest.
- Dashboard (`build_dashboard`) aggregates 30d and renders period cards,
  fallback windows, and the recent-decisions table.

## Non-Goals

- No automatic threshold changes (D2).
- No embeddings/semantic similarity — the re-ask detector uses cheap token
  overlap only (D4 owns embeddings).
- No persistence of the re-ask buffer across restarts (statistical signal;
  losing a 2-minute window on restart is irrelevant).

## 1. Data model — `LogLine::Feedback` ("f")

`route_log.rs` gains a third line kind:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeedbackEntry {
    /// The rid of the DECISION being judged (for "reask", the PREVIOUS
    /// request's rid — not the new request that triggered detection).
    pub rid: String,
    pub ts: i64,
    /// "cascade" | "reask" | "truncated" | "cloud_trivial"
    pub signal: String,
}
```

`LogLine` gains `Feedback(FeedbackEntry)` with serde tag `"f"` (alongside
`"d"`/`"o"`). `read_all()` legacy fallback unchanged. Multiple signals per rid
are allowed (separate lines). New `append_feedback(&FeedbackEntry)` mirrors
`append_outcome`.

## 2. Signal detection — four signals

| Signal | Meaning | Where detected |
|---|---|---|
| `cascade` | Local answer was weak; cascade escalated to cloud. Strong evidence score was underestimated. | Both escalate-to-cloud arms of `cascade_or_result` in `server.rs` (the arms that call `cloud::forward` after a weak/failed local result). |
| `truncated` | Local generation hit `FinishReason::Length` WITHOUT cascade escalation — weak local answer served as-is. (A local ERROR without cascade returns 500 and records no outcome; out of scope.) | Local done-sites that call `record_outcome`: when dest=="local" and the generation's finish reason is Length. |
| `cloud_trivial` | Cloud decision produced a trivial completion (`completion_tok < 40`) — local likely sufficient. Weak, statistical signal. | Inside `record_outcome` when dest=="cloud" and `completion_tok < 40`. |
| `reask` | User re-sent a very similar prompt shortly after a local answer — dissatisfaction heuristic. | `route_decision`, via the in-memory recent-prompts buffer (below). |

### Re-ask detector (approach A — in-memory buffer)

`AppState` gains:

```rust
/// Per-surface ring of recent LOCAL decisions for re-ask detection:
/// (rid, ts, token set of the last user turn). Cap 8 entries per surface.
pub recent_prompts: std::sync::Arc<std::sync::Mutex<
    std::collections::BTreeMap<String, std::collections::VecDeque<RecentPrompt>>>>,
```

`RecentPrompt { rid: String, ts: i64, tokens: std::collections::BTreeSet<String> }`.

Detection in `route_decision`, after computing the last-turn text:
1. Tokenize the latest user turn: lowercase, split on non-alphanumeric,
   drop tokens shorter than 2 chars (reuse `history_select`'s tokenizer if it
   is reusable as-is; otherwise a local helper with these exact rules).
2. Against the newest-first entries for this surface: if
   `jaccard(new, old) > 0.6` AND `new_ts - old.ts < 120` seconds →
   `append_feedback(FeedbackEntry { rid: old.rid, ts: now, signal: "reask" })`.
   First match wins (do not emit multiple reask lines for one new request).
3. Push the new request into the buffer ONLY if its decision is local
   (only local answers can be "re-asked because local was weak");
   evict oldest beyond cap 8.

Detection runs regardless of the new request's own destination (a re-ask that
routes to cloud still judges the previous local decision). The detector is a
pure function over injected `now` for testability.

## 3. Dashboard aggregation + UI

`build_dashboard` (existing 30d scan) additionally computes:

```rust
pub struct FeedbackStats {
    pub local_total: u64,
    pub local_flagged: u64,   // locals with >=1 of: cascade|truncated|reask
    pub cloud_total: u64,
    pub cloud_trivial: u64,   // clouds flagged cloud_trivial
}
```

- `RecentRow` gains `feedback: Vec<String>` — the signals joined by rid.
- Dashboard JSON gains `feedback: FeedbackStats` and
  `suggestion: Option<ThresholdSuggestion>` (section 4).

UI (`app.js` dashboard):
- New card "Acerto do routing": `local NN% ok · cloud NN% aproveitada`
  (`local_ok = 1 - flagged/total`, `cloud_util = 1 - trivial/total`; guard
  division by zero → show "—").
- Recent-decisions table: new narrow column; ✓ (no signals) or ⚠ with a
  tooltip listing the signals (e.g. "cascade, reask").

## 4. Threshold suggestion

Computed in Rust during `build_dashboard` over a **7-day** window (subset of
the 30d scan), only when the ACTIVE profile is Balanced (threshold only
affects Balanced), with a minimum sample of **20 local decisions** in the
window:

```rust
pub struct ThresholdSuggestion {
    pub direction: String,   // "lower" | "raise"
    pub suggested: f64,      // clamped to [0.20, 0.80]
    pub why: String,         // human sentence for the banner
}
```

- **Lower** (route more to cloud): if `local_flagged / local_total > 0.15` →
  `suggested = p25 of the difficulty scores of the flagged local decisions`
  (so most of them would have crossed to cloud), clamped.
- **Raise** (route more to local): if `cloud_trivial / cloud_total > 0.30`
  AND `local_flagged / local_total <= 0.15` →
  `suggested = p75 of the scores of the trivial cloud decisions`, clamped.
- Neither condition → `None`. Lower wins if both trigger (bad local answers
  cost the user more than wasted cloud tokens).

UI: banner on the dashboard, e.g. "20% dos pedidos locais precisaram da cloud
— considere baixar o limiar para 38%". Links to the existing Config threshold
control; nothing is auto-applied.

## Files Touched

- Modify: `src/route_log.rs` (FeedbackEntry, LogLine::Feedback, append_feedback,
  FeedbackStats, ThresholdSuggestion, suggestion + stats in build_dashboard,
  RecentRow.feedback)
- Modify: `src/server.rs` (AppState.recent_prompts, reask detection in
  route_decision, cascade/truncated/cloud_trivial emission, dashboard JSON)
- Modify: `src/manager_ui/app.js` (accuracy card, table column, banner)
- Modify: `src/manager_ui/style.css` (card/column/banner styles)
- Possibly reuse: `src/history_select.rs` tokenizer

## Testing

- **route_log unit:** parse "f" lines (+ legacy retrocompat), multiple signals
  per rid, FeedbackStats aggregation, RecentRow.feedback join, suggestion
  scenarios: lower-trigger, raise-trigger, both-trigger (lower wins),
  insufficient sample (None), non-Balanced profile (None), clamping at 0.20/0.80.
- **re-ask detector unit:** Jaccard threshold boundary, 120s window boundary,
  cap-8 eviction, first-match-wins, cloud decisions not pushed to buffer,
  injected time.
- **server wiring:** cascade arms emit "cascade"; local Length emits
  "truncated"; cloud completion <40 emits "cloud_trivial" (minimal-AppState
  pattern from Fase C).

## Open Risks

- Jaccard 0.6 / 120s / cap 8 / trivial<40 / 15% / 30% / 7d / 20-sample are
  starting constants — all live as named consts so D2 can tune them.
- `record_outcome` currently lacks finish-reason info at some done-sites; the
  plan must thread the finish reason (or a `was_truncated: bool`) into the
  emission point without changing `OutcomeEntry`'s wire format.
