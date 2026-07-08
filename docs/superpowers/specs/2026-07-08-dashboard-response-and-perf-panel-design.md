# Dashboard: response text in recent decisions + performance panel

**Date:** 2026-07-08
**Status:** approved (design)

## Problem

Two dashboard shortcomings:

1. **Recent decisions** show the prompt (expandable row) but not the model's
   response. The route log stores `completion_tok` (a count) but never the
   generated text, so there is nothing to display.
2. The **latency** and **quality** cards are two separate stacked cards with
   metrics written as running sentences — hard to read and scan.

## Goals

- Show the full model response next to the prompt in the expandable decision row.
- Replace the two separate latency/quality cards with a single, scannable
  two-column "Desempenho" panel.

## Non-goals

- Routing/scoring changes (large-context requests going local and timing out)
  are a separate, tracked issue — out of scope here.
- No redaction/PII filtering of stored responses (user chose to store full text).

## Storage model

Two files, do not conflate:

- **App-log** `/tmp/localllm.log` — human-readable tracing text. Not touched.
- **Route-log** `~/Library/Application Support/localllm/routing-log.jsonl`
  (override `LOCALLLM_ROUTE_LOG`) — one JSON object per line, the dashboard's
  data source. All changes below are here.

## Part A — response text in recent decisions

### Backend (`src/route_log.rs`, `src/server.rs`, `src/cloud.rs`)

- Add `output_text: Option<String>` to `OutcomeEntry`. Full text, no length cap
  (user decision). Serialized into `routing-log.jsonl`.
- `record_outcome(...)` gains an `output_text: Option<String>` parameter; every
  call site passes the generated response:
  - non-streaming local + cloud: the completed body is already in hand.
  - buffered-stream path: already accumulates the full text — pass it.
  - passthrough SSE path: add a collector that tees streamed chunks into a
    `String` so the full response is available at the `record_outcome` call.
- `build_dashboard` already joins the outcome to its `RecentEntry`; surface the
  joined `output_text` on `RecentEntry` so the frontend receives it.

### Route-log rotation (new, `src/route_log.rs`)

Full responses grow `routing-log.jsonl` unbounded. Mirror the existing app-log
rotation: cap the route-log at a line budget (and/or byte budget), trimming
oldest lines best-effort on append. Keep the newest N entries so the dashboard's
recent view and rollups stay intact.

### Frontend (`src/manager_ui/app.js`)

In `renderDecisionsTable`, inside the expandable detail row, after the existing
"Prompt (último turno)" box, add a **"Resposta"** section: a scrollable box
reusing the `prompt-box` style. Render via `textContent` (never innerHTML) to
avoid injection from stored model text. Show a muted "sem resposta registrada"
when `output_text` is absent (older rows, cloud-degraded, etc.).

## Part B — "Desempenho" panel (latency + quality side by side)

Replace the two separate cards (`LATÊNCIA (30d)` and `QUALIDADE (observado)`)
in `renderDashboard` with one `.perf-panel` laid out as two columns:

```
DESEMPENHO
┌─ LATÊNCIA ─────────┬─ QUALIDADE ────────┐
│ Local              │ Local              │
│   TTFT   320 ms    │   problemas  12%   │
│   vel    48 tok/s  │   8/64             │
│ Cloud              │ Cloud              │
│   TTFT   90 ms     │   trivial     4%   │
│   vel   120 tok/s  │   2/50             │
└────────────────────┴────────────────────┘
```

- Left **Latência**: Local and Cloud groups, each a label↔value stack —
  `TTFT <ms> ms`, `<tok/s> tok/s`, sample count `(n)`.
- Right **Qualidade**: Local (`problemas <pct>`, `<flagged>/<total>`) and Cloud
  (`trivial <pct>`, `<trivial>/<total>`).
- Data sources unchanged: `d.local_latency`, `d.cloud_latency`, `d.feedback`.
- New CSS in `src/manager_ui/style.css`: `.perf-panel` as a 2-column grid that
  collapses to a single column at narrow width (existing responsive breakpoints).

## Error handling

- All route-log writes stay best-effort (never fail a request), as today.
- Missing `output_text` / zero-sample latency / zero-total quality render an
  explicit em-dash or empty-state string — no throw, no blank panel.
- Rotation failures are best-effort and silent (same as app-log rotation).

## Testing

- `route_log.rs` unit tests: `OutcomeEntry` round-trips `output_text`;
  `build_dashboard` surfaces `output_text` on the matching `RecentEntry`;
  rotation trims to the cap while preserving newest entries.
- `server.rs`: `record_outcome` persists `output_text` for a local and a cloud
  outcome (extend existing `record_outcome_*` tests).
- Frontend: manual — expand a decision row, confirm the response box renders and
  scrolls; confirm the Desempenho panel is two columns and collapses when narrow.

## Rollout

Frontend is embedded via `include_str!`; the running app is the `.app` bundle.
After implementing, rebuild with `scripts/build-app.sh --fast`, kill the running
bundle process, and relaunch to verify.
