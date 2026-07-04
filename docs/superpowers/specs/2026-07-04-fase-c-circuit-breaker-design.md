# Fase C — Cloud Circuit Breaker (Design Spec)

**Date:** 2026-07-04
**Branch base:** `feat/fase-a` (Fase C builds on Fase A budget/degrade infrastructure)
**Status:** Approved for planning

## Goal

Stop hammering a down cloud provider on every request. When cloud fails, serve
local instantly during a cooldown, then probe cloud on a rate-limited schedule
to auto-detect recovery. Give the user a manual override (tray + dashboard) to
retry cloud immediately.

## Motivation

Fase A already degrades a failed cloud call to local and notifies the user once
(`Usage::note_degrade`). But it retries cloud on *every* subsequent request, so
during an outage each request eats a full cloud timeout / 429 before falling
back to local. A circuit breaker rate-limits those retries: skip cloud during a
cooldown, then let a single probe test recovery.

## Non-Goals

- Per-surface or per-provider breakers. One **global** breaker for all cloud
  (matches the "global for everything" routing decision).
- Persistence across restarts. Breaker state is in-memory only, like `Usage`;
  a restart starts Closed.
- Active/background health pings. Probes are passive — a real cloud-bound
  request becomes the probe. No extra cloud calls, no wasted spend.

## State Machine

```
Closed ──fail──────────────> Open{until, backoff, reason}
Open ──cooldown elapsed────> HalfOpen        (next cloud-bound request = probe)
HalfOpen ──probe success───> Closed          (notify "cloud recovered")
HalfOpen ──probe failure───> Open{backoff*2, capped}
<any> ──manual reset───────> Closed
```

- **Closed:** normal. Cloud allowed.
- **Open{until, backoff, reason}:** cloud skipped until `until`. `backoff` is the
  current cooldown length (seconds). `reason` is the `DegradeReason` that tripped.
- **HalfOpen{reason}:** cooldown elapsed; the next cloud-bound request is allowed
  through as a single probe. Its outcome decides Closed vs Open.

### Backoff schedule

- First trip (Closed→Open): `backoff = 30s`.
- Each failed probe (HalfOpen→Open): `backoff = min(backoff * 2, 300)`.
  Sequence: 30 → 60 → 120 → 240 → 300 (cap).
- Success (→Closed) resets `backoff` to the 30s base.

### Trip reasons

All four `DegradeReason` variants trip the breaker: `Auth`, `Quota`,
`ServerError`, `Offline`. Auth won't self-heal on a probe, but probes on a bad
key are harmless, and the tray "Retry cloud now" button lets the user force a
retry the moment they fix credentials.

## Components

### `src/breaker.rs` (new module)

Global breaker, held as `AppState.breaker: Arc<CircuitBreaker>` wrapping a
`Mutex<BreakerState>`. All time is injected (`now: u64` seconds, matching
`route_log::now_secs()`) so tests are deterministic.

```rust
pub const BASE_COOLDOWN_SECS: u64 = 30;
pub const MAX_COOLDOWN_SECS: u64 = 300;

enum BreakerState {
    Closed,
    Open { until: u64, backoff: u64, reason: crate::usage::DegradeReason },
    HalfOpen { reason: crate::usage::DegradeReason },
}

pub enum Gate {
    /// Cloud may proceed (Closed, or Open cooldown elapsed → this request is the probe).
    Allow,
    /// Cloud blocked; serve local. `next_probe_in` = seconds until probe allowed.
    Block { reason: crate::usage::DegradeReason, next_probe_in: u64 },
}

pub struct BreakerSnapshot {
    pub state: &'static str,          // "closed" | "open" | "half-open"
    pub reason: Option<&'static str>, // DegradeReason label, None when Closed
    pub next_probe_secs: Option<u64>, // seconds until probe, None unless Open
}

pub struct CircuitBreaker { /* Mutex<BreakerState> */ }

impl CircuitBreaker {
    pub fn new() -> Self;                                   // Closed

    /// Consult before a cloud-bound request. In Open with elapsed cooldown,
    /// transitions to HalfOpen and returns Allow (this request is the probe).
    pub fn gate(&self, now: u64) -> Gate;

    /// A cloud attempt failed. Closed→Open(base); HalfOpen→Open(backoff*2 capped);
    /// Open stays Open (keeps existing until/backoff).
    pub fn on_failure(&self, now: u64, reason: crate::usage::DegradeReason);

    /// A cloud attempt succeeded. →Closed, reset backoff. Returns true iff the
    /// prior state was Open or HalfOpen (so the caller notifies "recovered").
    pub fn on_success(&self, now: u64) -> bool;

    /// Manual force-close (tray / dashboard). →Closed.
    pub fn reset(&self, now: u64);

    /// Read-only view for the dashboard endpoint.
    pub fn snapshot(&self, now: u64) -> BreakerSnapshot;
}
```

**Label mapping** reuses `server::degrade_reason_label` semantics
("Auth"/"Quota"/"ServerError"/"Offline"). The label helper is moved to (or
re-exported from) a place both `breaker` and `server` can call, to avoid
duplication.

### `route_decision` gate (src/server.rs, after the `budget_forced` block ~184)

Precedence: **budget → breaker** (both force local; budget wins if both apply).
The breaker gate is only consulted when the raw decision is cloud-bound and not
already budget-forced, so a probe is only spent on a real cloud attempt:

```rust
// after computing budget_forced and the budget-adjusted `decision`
let breaker_block = if !budget_forced {
    if let crate::route::Decision::Cloud(_) = decision {
        match state.breaker.gate(crate::route_log::now_secs()) {
            crate::breaker::Gate::Block { reason, .. } => Some(reason),
            crate::breaker::Gate::Allow => None,
        }
    } else { None }
} else { None };
let decision = if breaker_block.is_some() {
    crate::route::Decision::Local
} else { decision };
```

Route-log `reason` for a breaker-forced local is `"CloudDown"` (parallel to
`"BudgetExceeded"`). The dest/reason match arm gains this case. The dashboard's
existing fallback-window grouping (Fase A, GAP=120s) already groups contiguous
degrade rows and now absorbs `"CloudDown"` rows too.

### Trip wiring (degrade sites)

At each site that currently calls `state.usage.note_degrade()`, also call
`state.breaker.on_failure(now, reason)`:

- `handle_degrade` (~344)
- cascade degrade arms (~479, ~499)
- stream degrade arm (~1497)

`now = crate::route_log::now_secs()`. The `reason` is the `DegradeReason`
already in scope at each site.

### Recover wiring (`record_cloud_success`, ~317)

```rust
fn record_cloud_success(state: &AppState, est_prompt_tokens: usize) {
    state.usage.note_success();
    if state.breaker.on_success(crate::route_log::now_secs()) {
        crate::usage::notify("localllm — cloud recovered", "Cloud back — resuming.");
    }
    // ...existing budget accrual...
}
```

### Tray "Retry cloud now" (src/tray.rs)

Mirror the `policy` Arc idiom: tray constructs `Arc<CircuitBreaker>` and threads
it through `run_server_with_ready_policy_token` into `AppState`. A new tray menu
item "↻ Retry cloud now" calls `breaker.reset(now)` directly in-process (no HTTP).
Placed near the routing/status lines in the tray menu.

### `/admin/breaker` endpoints (src/server.rs)

- `GET /admin/breaker` → `BreakerSnapshot` as JSON (admin-token guarded, like
  other `/admin/*` GETs). For the dashboard widget.
- `POST /admin/breaker/reset` → `breaker.reset(now)`, returns 200. For the
  dashboard reset button (browser can't call the shared Arc directly).

### Dashboard widget (src/manager_ui/app.js + style.css)

Small status widget on the dashboard: current state (Closed / Open / Half-open),
reason, next-probe countdown (from `next_probe_secs`), and a "Retry cloud now"
button that POSTs `/admin/breaker/reset` then refreshes. Styled consistently
with the Fase A dashboard cards.

## Threading / Construction

`Arc<CircuitBreaker>` is created once in `tray::run_tray` (like `policy`),
cloned into the server entry (`run_server_with_ready_policy_token` signature
gains the breaker arg), and stored in `AppState.breaker`. The test router
constructor (`router()` / `router_for_test_with`) constructs a fresh
`CircuitBreaker::new()`.

## Testing

**`src/breaker.rs` unit tests** (deterministic via injected `now`):
- Closed → Open on first failure; `gate` blocks with `next_probe_in ≈ 30`.
- Cooldown elapsed → `gate` transitions to HalfOpen and returns `Allow`.
- HalfOpen + `on_failure` → Open with doubled backoff (60), capped at 300 after
  enough failures.
- HalfOpen + `on_success` → Closed, backoff reset to 30; `on_success` returns
  `true` when recovering, `false` when already Closed.
- Manual `reset` from Open/HalfOpen → Closed, `gate` returns `Allow`.
- `snapshot` reports correct labels and countdown in each state.

**Integration** (server tests, reuse existing env locks):
- `route_decision` forces `Local` with reason `"CloudDown"` when the breaker is
  Open and cooldown not elapsed, even when the raw decision is Cloud.
- When cooldown has elapsed, a cloud-bound request is allowed through (probe).
- Budget precedence: when both budget and breaker would force local, route-log
  reason is `"BudgetExceeded"` (budget wins).

## Files Touched

- Create: `src/breaker.rs`
- Modify: `src/lib.rs` (`pub mod breaker;`)
- Modify: `src/server.rs` (gate, trip/recover wiring, endpoints, AppState field,
  entry signature, route-log reason arm, label helper relocation)
- Modify: `src/tray.rs` (breaker Arc construction + threading, menu item)
- Modify: `src/manager_ui/app.js` (widget + reset button + poll)
- Modify: `src/manager_ui/style.css` (widget styling)
