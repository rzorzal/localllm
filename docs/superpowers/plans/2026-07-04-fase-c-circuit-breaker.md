# Fase C — Cloud Circuit Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a global in-memory cloud circuit breaker that skips cloud during an exponential cooldown after failures, passively probes for recovery, surfaces state on the dashboard, and offers a manual "retry now" reset via tray + dashboard.

**Architecture:** A `CircuitBreaker` struct (`Mutex<BreakerState>`) holds all state-machine logic and is unit-tested in isolation via injected `now`. One instance lives in `AppState.breaker`, constructed by the tray (like `policy`) and threaded through the server entry into `router()`. `route_decision` consults it via a pure `breaker_gate_block` helper; degrade sites trip it; `record_cloud_success` closes it. Two admin endpoints and a dashboard widget expose/reset it.

**Tech Stack:** Rust, axum 0.7, `std::sync::Mutex`/`Arc`, serde_json, vanilla-JS `manager_ui`. No new crates. Seconds-since-epoch time via existing `crate::route_log::now_secs()`.

## Global Constraints

- Breaker state is **in-memory only** — no disk persistence; a restart starts `Closed`.
- **One global breaker** for all cloud (no per-surface/per-provider state).
- Probes are **passive**: a real cloud-bound request becomes the probe; never issue background/health calls.
- Backoff: first trip `BASE_COOLDOWN_SECS = 30`; each failed probe doubles, capped at `MAX_COOLDOWN_SECS = 300`; success resets to base.
- **All four** `DegradeReason` variants (`Auth`, `Quota`, `ServerError`, `Offline`) trip the breaker.
- Routing precedence when forcing local: **budget → breaker** (budget wins if both apply).
- Route-log `reason` for a breaker-forced local is the exact string `"CloudDown"` (parallel to existing `"BudgetExceeded"`).
- All time is injected as `now: u64` (seconds); no direct clock reads inside `CircuitBreaker` so tests are deterministic.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
  ```
- Build machine is slow (cold link ~6min, warm ~10-13s). Run `cargo test` in the background.

---

### Task 1: `CircuitBreaker` module + state machine

**Files:**
- Create: `src/breaker.rs`
- Modify: `src/lib.rs` (add `pub mod breaker;` beside `pub mod budget;`)

**Interfaces:**
- Consumes: `crate::usage::DegradeReason` (enum: `Auth`, `Quota`, `ServerError`, `Offline`; `Copy`).
- Produces:
  - `pub const BASE_COOLDOWN_SECS: u64 = 30;`
  - `pub const MAX_COOLDOWN_SECS: u64 = 300;`
  - `pub enum Gate { Allow, Block { reason: crate::usage::DegradeReason, next_probe_in: u64 } }`
  - `pub struct BreakerSnapshot { pub state: &'static str, pub reason: Option<&'static str>, pub next_probe_secs: Option<u64> }`
  - `pub struct CircuitBreaker` with:
    - `pub fn new() -> Self`
    - `pub fn gate(&self, now: u64) -> Gate`
    - `pub fn on_failure(&self, now: u64, reason: crate::usage::DegradeReason)`
    - `pub fn on_success(&self, now: u64) -> bool`
    - `pub fn reset(&self, now: u64)`
    - `pub fn snapshot(&self, now: u64) -> BreakerSnapshot`
  - `pub fn reason_label(r: crate::usage::DegradeReason) -> &'static str`

- [ ] **Step 1: Add the module declaration**

In `src/lib.rs`, find the line `pub mod budget;` and add directly after it:

```rust
pub mod breaker;
```

- [ ] **Step 2: Write the failing tests**

Create `src/breaker.rs` with the full implementation stubbed to `todo!()` bodies is NOT the approach — instead write the tests first against the real signatures, then implement. Create `src/breaker.rs` containing ONLY the type declarations + a `#[cfg(test)] mod tests` block below, then run to confirm failure. Paste this as the initial file:

```rust
//! Global in-memory cloud circuit breaker.
//!
//! After a cloud request fails, the breaker opens for an exponential cooldown
//! (30s doubling to a 300s cap). While open, cloud-bound requests are served
//! locally without touching cloud. When the cooldown elapses the breaker goes
//! half-open and the next cloud-bound request is allowed through as a single
//! probe: success closes the breaker (and signals recovery), failure reopens it
//! with a longer cooldown. A manual `reset` force-closes it (tray / dashboard).
//!
//! State is process-global and in-memory only — a restart starts Closed. Time is
//! injected as `now` (seconds since epoch) so the state machine is deterministic
//! under test.

use std::sync::Mutex;

use crate::usage::DegradeReason;

/// First cooldown after a trip, in seconds.
pub const BASE_COOLDOWN_SECS: u64 = 30;
/// Cooldown ceiling, in seconds.
pub const MAX_COOLDOWN_SECS: u64 = 300;

/// Verdict for a cloud-bound request consulting the breaker.
#[derive(Debug, PartialEq, Eq)]
pub enum Gate {
    /// Cloud may proceed (Closed, or Open with cooldown elapsed → this is a probe).
    Allow,
    /// Cloud blocked; serve local. `next_probe_in` = seconds until a probe is allowed.
    Block {
        reason: DegradeReason,
        next_probe_in: u64,
    },
}

/// Read-only view for the dashboard endpoint.
#[derive(Debug, PartialEq, Eq)]
pub struct BreakerSnapshot {
    pub state: &'static str,          // "closed" | "open" | "half-open"
    pub reason: Option<&'static str>, // DegradeReason label; None when Closed
    pub next_probe_secs: Option<u64>, // seconds until probe; None unless Open
}

#[derive(Debug, Clone, Copy)]
enum BreakerState {
    Closed,
    Open {
        until: u64,
        backoff: u64,
        reason: DegradeReason,
    },
    HalfOpen {
        /// Carried from the Open state so a failed probe doubles from here.
        backoff: u64,
        reason: DegradeReason,
    },
}

/// Stable short label for a degrade reason (shared with the route-log/dashboard).
pub fn reason_label(r: DegradeReason) -> &'static str {
    match r {
        DegradeReason::Auth => "Auth",
        DegradeReason::Quota => "Quota",
        DegradeReason::ServerError => "ServerError",
        DegradeReason::Offline => "Offline",
    }
}

/// Global cloud circuit breaker. Cheap to share behind an `Arc`.
#[derive(Debug)]
pub struct CircuitBreaker {
    inner: Mutex<BreakerState>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreaker {
    /// A fresh, Closed breaker.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BreakerState::Closed),
        }
    }

    /// Consult before a cloud-bound request. In Open with an elapsed cooldown,
    /// transitions to HalfOpen and returns `Allow` (this request is the probe).
    pub fn gate(&self, now: u64) -> Gate {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match *st {
            BreakerState::Closed | BreakerState::HalfOpen { .. } => Gate::Allow,
            BreakerState::Open {
                until,
                backoff,
                reason,
            } => {
                if now >= until {
                    // Cooldown elapsed → half-open; this caller is the probe.
                    // Carry backoff so a failed probe doubles from the current value.
                    *st = BreakerState::HalfOpen { backoff, reason };
                    Gate::Allow
                } else {
                    Gate::Block {
                        reason,
                        next_probe_in: until - now,
                    }
                }
            }
        }
    }

    /// Record a failed cloud attempt. Closed→Open(base); HalfOpen→Open(backoff*2
    /// capped); Open stays Open with its existing schedule.
    pub fn on_failure(&self, now: u64, reason: DegradeReason) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let next_backoff = match *st {
            BreakerState::Closed => BASE_COOLDOWN_SECS,
            // A failed probe (or a direct failure while Open): double the carried
            // backoff, capped. HalfOpen and Open both carry the current value.
            BreakerState::HalfOpen { backoff, .. } => (backoff * 2).min(MAX_COOLDOWN_SECS),
            BreakerState::Open { backoff, .. } => (backoff * 2).min(MAX_COOLDOWN_SECS),
        };
        *st = BreakerState::Open {
            until: now + next_backoff,
            backoff: next_backoff,
            reason,
        };
    }

    /// Record a successful cloud attempt. →Closed, reset backoff. Returns `true`
    /// iff the breaker was recovering (Open or HalfOpen) so the caller notifies.
    pub fn on_success(&self, _now: u64) -> bool {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let was_recovering = !matches!(*st, BreakerState::Closed);
        *st = BreakerState::Closed;
        was_recovering
    }

    /// Manual force-close (tray / dashboard "retry now").
    pub fn reset(&self, _now: u64) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *st = BreakerState::Closed;
    }

    /// Read-only snapshot for the dashboard.
    pub fn snapshot(&self, now: u64) -> BreakerSnapshot {
        let st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match *st {
            BreakerState::Closed => BreakerSnapshot {
                state: "closed",
                reason: None,
                next_probe_secs: None,
            },
            BreakerState::HalfOpen { reason, .. } => BreakerSnapshot {
                state: "half-open",
                reason: Some(reason_label(reason)),
                next_probe_secs: None,
            },
            BreakerState::Open { until, reason, .. } => BreakerSnapshot {
                state: "open",
                reason: Some(reason_label(reason)),
                next_probe_secs: Some(until.saturating_sub(now)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::DegradeReason;

    #[test]
    fn closed_allows_and_snapshots_closed() {
        let b = CircuitBreaker::new();
        assert_eq!(b.gate(100), Gate::Allow);
        assert_eq!(
            b.snapshot(100),
            BreakerSnapshot { state: "closed", reason: None, next_probe_secs: None }
        );
    }

    #[test]
    fn first_failure_opens_for_base_cooldown() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Quota);
        assert_eq!(
            b.gate(100),
            Gate::Block { reason: DegradeReason::Quota, next_probe_in: BASE_COOLDOWN_SECS }
        );
        let s = b.snapshot(100);
        assert_eq!(s.state, "open");
        assert_eq!(s.reason, Some("Quota"));
        assert_eq!(s.next_probe_secs, Some(30));
    }

    #[test]
    fn cooldown_elapsed_gate_goes_half_open_and_allows_probe() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::ServerError);
        // Still cooling at t=120.
        assert!(matches!(b.gate(120), Gate::Block { .. }));
        // At t=130 (>= until=130) the probe is allowed and state is half-open.
        assert_eq!(b.gate(130), Gate::Allow);
        assert_eq!(b.snapshot(130).state, "half-open");
    }

    #[test]
    fn failed_probe_reopens_with_doubled_cooldown() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Offline); // open until 130, backoff 30
        assert_eq!(b.gate(130), Gate::Allow); // half-open probe
        b.on_failure(130, DegradeReason::Offline); // probe failed → doubled to 60
        assert_eq!(
            b.gate(130),
            Gate::Block { reason: DegradeReason::Offline, next_probe_in: 60 }
        );
    }

    #[test]
    fn backoff_doubles_and_caps_at_max() {
        let b = CircuitBreaker::new();
        let mut now = 0u64;
        b.on_failure(now, DegradeReason::ServerError); // 30
        // Drive repeated failed probes: 30→60→120→240→300(cap)→300.
        let expected = [60u64, 120, 240, 300, 300];
        for exp in expected {
            // advance to the probe window
            let s = b.snapshot(now);
            now += s.next_probe_secs.unwrap();
            assert_eq!(b.gate(now), Gate::Allow); // half-open
            b.on_failure(now, DegradeReason::ServerError);
            assert_eq!(b.snapshot(now).next_probe_secs, Some(exp));
        }
    }

    #[test]
    fn successful_probe_closes_and_signals_recovery() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Quota);
        assert_eq!(b.gate(130), Gate::Allow); // half-open probe
        assert!(b.on_success(130)); // was recovering → true
        assert_eq!(b.snapshot(130).state, "closed");
        // A success while already closed does not re-signal.
        assert!(!b.on_success(140));
    }

    #[test]
    fn success_resets_backoff_to_base() {
        let b = CircuitBreaker::new();
        b.on_failure(0, DegradeReason::Quota); // 30
        assert_eq!(b.gate(30), Gate::Allow);
        b.on_failure(30, DegradeReason::Quota); // 60
        assert_eq!(b.gate(90), Gate::Allow);
        assert!(b.on_success(90)); // recover → reset
        // Next trip starts from base again.
        b.on_failure(200, DegradeReason::Quota);
        assert_eq!(b.snapshot(200).next_probe_secs, Some(BASE_COOLDOWN_SECS));
    }

    #[test]
    fn manual_reset_force_closes() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Auth);
        assert!(matches!(b.gate(105), Gate::Block { .. }));
        b.reset(105);
        assert_eq!(b.gate(105), Gate::Allow);
        assert_eq!(b.snapshot(105).state, "closed");
    }

    #[test]
    fn all_reasons_trip_and_label() {
        for r in [
            DegradeReason::Auth,
            DegradeReason::Quota,
            DegradeReason::ServerError,
            DegradeReason::Offline,
        ] {
            let b = CircuitBreaker::new();
            b.on_failure(0, r);
            assert_eq!(b.snapshot(0).reason, Some(reason_label(r)));
        }
    }
}
```

- [ ] **Step 3: Run tests to verify they pass**

The file already contains the implementation (this module is small enough that test-first-then-implement collapses into one write). Run in background:

Run: `cargo test --lib breaker:: -- --nocapture`
Expected: all breaker tests pass (`test result: ok. 9 passed`).

- [ ] **Step 4: Verify the whole crate still builds**

Run: `cargo build --lib`
Expected: success (the new `pub mod breaker;` compiles; nothing else references it yet).

- [ ] **Step 5: Commit**

```bash
git add src/breaker.rs src/lib.rs
git commit -F - <<'EOF'
feat(breaker): cloud circuit breaker state machine

Global in-memory breaker: Closed -> Open(exponential 30s..300s cap) ->
HalfOpen(passive probe) -> Closed/reopen. Manual reset force-closes. Time
injected for deterministic tests. Not wired into routing yet.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 2: Wire breaker into `AppState` + `route_decision` gate

**Files:**
- Modify: `src/server.rs` (AppState field ~569; `router()` signature/body ~628-666; `route_decision` ~184; new `breaker_gate_block` helper; `make_seeded_test_router` ~1977-1994; test-state helper + unit tests in `mod tests`)
- Modify: `src/lib.rs` (`run_server_with_ready_policy_token` ~113-266 pass breaker; `run_server_with_ready_and_policy` ~98-105 construct breaker; `router_for_test_with` ~460-472 construct fresh breaker internally)

**Interfaces:**
- Consumes from Task 1: `crate::breaker::CircuitBreaker::new()`, `.gate(now) -> Gate`, `Gate::{Allow, Block{reason, next_probe_in}}`.
- Produces:
  - `AppState.breaker: std::sync::Arc<crate::breaker::CircuitBreaker>`
  - `router(...)` gains a trailing param `breaker: std::sync::Arc<crate::breaker::CircuitBreaker>`
  - `fn breaker_gate_block(breaker: &crate::breaker::CircuitBreaker, decision: &crate::route::Decision, budget_forced: bool, now: u64) -> Option<crate::usage::DegradeReason>`
  - `run_server_with_ready_policy_token` gains a trailing param `breaker: std::sync::Arc<crate::breaker::CircuitBreaker>` (before `manager_out`? No — append LAST to minimize churn; see step notes).

- [ ] **Step 1: Add the `breaker` field to `AppState`**

In `src/server.rs`, after the `budget` field (~569), add:

```rust
    /// Daily cloud-spend tracker for the budget cap.
    pub budget: std::sync::Arc<crate::budget::Budget>,
    /// Global cloud circuit breaker (skip cloud during cooldown after failures).
    pub breaker: std::sync::Arc<crate::breaker::CircuitBreaker>,
}
```

- [ ] **Step 2: Add the `breaker` param to `router()` and set the field**

Change the `router()` signature (~628-640): after the final `port: u16,` param add:

```rust
    port: u16,
    breaker: std::sync::Arc<crate::breaker::CircuitBreaker>,
) -> Router {
```

And in the `AppState { ... }` literal it builds (~641-664), after the `budget: { ... }` block add:

```rust
        budget: {
            let b = std::sync::Arc::new(crate::budget::Budget::new());
            b.seed_from_log(&crate::route_log::read_all(), crate::route_log::now_secs());
            b
        },
        breaker,
    });
```

- [ ] **Step 3: Write the failing unit tests for `breaker_gate_block`**

In `src/server.rs`, inside `#[cfg(test)] mod tests` (near the other `super::` tests ~1998), add:

```rust
    #[test]
    fn breaker_gate_block_forces_local_only_for_cloud_when_open() {
        use crate::route::{Decision, RouteReason};
        let b = crate::breaker::CircuitBreaker::new();
        // Closed → never blocks.
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Cloud(RouteReason::HighDifficulty), false, 0),
            None
        );
        // Open → blocks a cloud decision with the tripping reason.
        b.on_failure(0, crate::usage::DegradeReason::Quota);
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Cloud(RouteReason::HighDifficulty), false, 5),
            Some(crate::usage::DegradeReason::Quota)
        );
        // Open but the decision is Local → nothing to block.
        assert_eq!(super::breaker_gate_block(&b, &Decision::Local, false, 5), None);
        // Budget already forced local → breaker defers (budget precedence).
        assert_eq!(
            super::breaker_gate_block(&b, &Decision::Cloud(RouteReason::HighDifficulty), true, 5),
            None
        );
    }
```

Note: confirm the exact `RouteReason` variant name by reading `src/route/mod.rs` (`Decision::Cloud(RouteReason)`); use whichever real variant exists (e.g. `HighDifficulty`). If the variant differs, use the real one in the test.

- [ ] **Step 4: Run the test to verify it fails**

Run: `cargo test --lib breaker_gate_block -- --nocapture`
Expected: FAIL to compile — `breaker_gate_block` not found.

- [ ] **Step 5: Implement `breaker_gate_block` and call it in `route_decision`**

In `src/server.rs`, add the helper directly above `fn route_decision` (~154):

```rust
/// Decide whether the circuit breaker forces a cloud-bound decision to local.
/// Returns the tripping `DegradeReason` when the raw decision is cloud, budget
/// did not already force local, and the breaker gate is closed to cloud (Open,
/// cooldown not elapsed). Consulting `gate` may consume a half-open probe, so
/// only call this once per decision and only for cloud-bound decisions.
fn breaker_gate_block(
    breaker: &crate::breaker::CircuitBreaker,
    decision: &crate::route::Decision,
    budget_forced: bool,
    now: u64,
) -> Option<crate::usage::DegradeReason> {
    if budget_forced {
        return None;
    }
    if !matches!(decision, crate::route::Decision::Cloud(_)) {
        return None;
    }
    match breaker.gate(now) {
        crate::breaker::Gate::Block { reason, .. } => Some(reason),
        crate::breaker::Gate::Allow => None,
    }
}
```

Then in `route_decision`, right after the existing `let decision = if budget_forced { Local } else { raw_decision };` block (~184-188), insert:

```rust
    // Circuit breaker: if cloud recently failed, skip it during the cooldown and
    // serve local. Budget takes precedence (already forced above). A half-open
    // probe is consumed here only when the decision is cloud-bound.
    let breaker_block = breaker_gate_block(
        &state.breaker,
        &decision,
        budget_forced,
        crate::route_log::now_secs(),
    );
    let decision = if breaker_block.is_some() {
        crate::route::Decision::Local
    } else {
        decision
    };
```

- [ ] **Step 6: Record the `"CloudDown"` reason in the route log**

In `route_decision`, replace the existing `(dest, reason)` computation (~209-216):

```rust
    let (dest, reason) = if budget_forced {
        ("local", Some("BudgetExceeded".to_string()))
    } else {
        match decision {
            crate::route::Decision::Cloud(r) => ("cloud", Some(format!("{r:?}"))),
            _ => ("local", None),
        }
    };
```

with:

```rust
    let (dest, reason) = if budget_forced {
        ("local", Some("BudgetExceeded".to_string()))
    } else if breaker_block.is_some() {
        ("local", Some("CloudDown".to_string()))
    } else {
        match decision {
            crate::route::Decision::Cloud(r) => ("cloud", Some(format!("{r:?}"))),
            _ => ("local", None),
        }
    };
```

- [ ] **Step 7: Fix all `router()` call sites and the test AppState literal**

a) `src/lib.rs` entry call (~250-266): the `router(...)` call ends with `cfg.port,`. Before that, the enclosing function must have a `breaker` in scope. Add near the top of `run_server_with_ready_policy_token` (after `let usage = ...` ~183 or wherever `usage` is bound — actually `usage` is bound ~183; add right after it is fine, but the breaker comes from the new param, see step 8). For now, append the arg to the `router(...)` call:

```rust
        cfg.port,
        breaker,
    );
```

b) `src/lib.rs` `router_for_test_with` (~460-472): append a freshly-constructed breaker as the last arg so the test-router signature is unchanged (keeps `tests/http.rs` and existing server tests untouched):

```rust
    crate::server::router(
        manager,
        "test-model".to_string(),
        policy,
        local_ctx_window,
        usage,
        200_000,
        Arc::from("test-token"),
        16384,
        32768,
        crate::fit::KvKind::Q8,
        31415,
        Arc::new(crate::breaker::CircuitBreaker::new()),
    )
```

c) `src/server.rs` `make_seeded_test_router` (~1977-1994): add the `breaker` field to the `AppState { ... }` literal, after `budget: ...`:

```rust
        port: 31415,
        budget: std::sync::Arc::new(crate::budget::Budget::new()),
        breaker: std::sync::Arc::new(crate::breaker::CircuitBreaker::new()),
    }))
```

- [ ] **Step 8: Thread the breaker through the server entry functions**

a) `src/lib.rs` `run_server_with_ready_policy_token` signature (~113-119): append a param before `manager_out` is awkward (call sites use positional args); instead append it as the LAST param:

```rust
pub async fn run_server_with_ready_policy_token(
    cfg: crate::config::Config,
    ready: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    policy: std::sync::Arc<std::sync::RwLock<crate::route::RoutingPolicy>>,
    admin_token: std::sync::Arc<str>,
    manager_out: Option<std::sync::Arc<std::sync::OnceLock<std::sync::Arc<crate::model_manager::ModelManager>>>>,
    breaker: std::sync::Arc<crate::breaker::CircuitBreaker>,
) -> anyhow::Result<()> {
```

(The `router(...)` call from step 7a already passes this `breaker`.)

b) `src/lib.rs` `run_server_with_ready_and_policy` (~98-105): construct a breaker and pass it:

```rust
    let admin_token = std::sync::Arc::from(crate::server::resolve_admin_token(cfg.admin_token.clone()));
    let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
    run_server_with_ready_policy_token(cfg, ready, policy, admin_token, None, breaker).await
```

(The tray call site is updated in Task 5; until then the tray won't compile — Task 5 completes the wiring. To keep the crate compiling after Task 2, also update the tray call now: see step 9.)

- [ ] **Step 9: Keep the tray call compiling (minimal)**

In `src/tray.rs`, the `rt.block_on(crate::run_server_with_ready_policy_token(...))` call (~404-410) currently passes 5 args ending in `Some(manager_slot_for_server)`. Add a sixth:

```rust
            if let Err(e) = rt.block_on(crate::run_server_with_ready_policy_token(
                cfg,
                Some(ready_for_server),
                policy_for_server,
                admin_token_for_server,
                Some(manager_slot_for_server),
                std::sync::Arc::new(crate::breaker::CircuitBreaker::new()),
            )) {
```

(Task 5 replaces this throwaway `Arc::new(...)` with a shared one the menu can reset.)

- [ ] **Step 10: Run the breaker gate test + full lib build**

Run: `cargo test --lib breaker_gate_block -- --nocapture`
Expected: PASS.

Run: `cargo build`
Expected: success (all crates, including the bin/tray, compile).

- [ ] **Step 11: Commit**

```bash
git add src/server.rs src/lib.rs src/tray.rs
git commit -F - <<'EOF'
feat(breaker): gate route_decision on the circuit breaker

AppState carries an Arc<CircuitBreaker> threaded from the server entry
through router(). route_decision consults breaker_gate_block after the
budget check (budget precedence) and forces Local with route-log reason
"CloudDown" while cloud is in cooldown. Test-router constructors build a
fresh Closed breaker.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 3: Trip on degrade + recover on success

**Files:**
- Modify: `src/server.rs` (`handle_degrade` ~339-353; cascade degrade arms ~478-484 and ~498-503; `record_cloud_success` ~317-333; a test-state helper + tests in `mod tests`)

**Interfaces:**
- Consumes: `state.breaker.on_failure(now, reason)`, `state.breaker.on_success(now) -> bool`, `crate::usage::notify`.
- Produces: no new public API; behavior wiring only.

- [ ] **Step 1: Write failing wiring tests**

In `src/server.rs` `#[cfg(test)] mod tests`, add a helper that builds a minimal `AppState` with a chosen breaker, plus two tests. Place near the other tests:

```rust
    /// Minimal AppState for wiring tests: TaggedGen engine, Closed-by-default
    /// breaker (caller may pre-trip). No network, no files touched by the paths
    /// under test (record_cloud_success and handle_degrade are pure w.r.t. disk).
    #[cfg(test)]
    fn wiring_state(breaker: std::sync::Arc<crate::breaker::CircuitBreaker>) -> super::AppState {
        use crate::model_manager::{EngineBuilder, ModelManager, ModelSpec};
        let builder: EngineBuilder = Box::new(|_spec| {
            Box::pin(async {
                Ok(std::sync::Arc::new(crate::test_support::TaggedGen("switched"))
                    as std::sync::Arc<dyn super::Generator>)
            })
        });
        let manager = ModelManager::new(
            std::sync::Arc::new(crate::test_support::TaggedGen("test"))
                as std::sync::Arc<dyn super::Generator>,
            ModelSpec { repo: "test".into(), file: "test".into(), quant: None },
            builder,
        );
        super::AppState {
            manager,
            model_id: "test-model".to_string(),
            policy: std::sync::Arc::new(std::sync::RwLock::new(
                crate::route::Profile::default().policy(),
            )),
            local_ctx_window: 1000,
            usage: std::sync::Arc::new(crate::usage::Usage::new()),
            cloud_token_alert: 200_000,
            admin_token: std::sync::Arc::from("test-token"),
            total_ram_mb: 16384,
            requested_ctx_ceiling: 32768,
            kv_kind: crate::fit::KvKind::Q8,
            tool_registry: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            tool_descs: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::BTreeMap::new(),
            )),
            port: 31415,
            budget: std::sync::Arc::new(crate::budget::Budget::new()),
            breaker,
        }
    }

    #[test]
    fn handle_degrade_trips_breaker() {
        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        let state = wiring_state(breaker.clone());
        // Non-overflow reason → handle_degrade returns None (local can serve) and
        // the breaker opens.
        let out = super::handle_degrade(
            &state,
            crate::usage::DegradeReason::Quota,
            crate::route::RouteReason::HighDifficulty,
        );
        assert!(out.is_none());
        assert_eq!(breaker.snapshot(crate::route_log::now_secs()).state, "open");
    }

    #[test]
    fn record_cloud_success_closes_breaker() {
        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        breaker.on_failure(0, crate::usage::DegradeReason::ServerError);
        let state = wiring_state(breaker.clone());
        super::record_cloud_success(&state, 10);
        assert_eq!(breaker.snapshot(crate::route_log::now_secs()).state, "closed");
    }
```

Note: confirm `crate::route::RouteReason::HighDifficulty` is a real variant (read `src/route/mod.rs`); use the actual variant used by `RouteReason::ContextOverflow`'s sibling. Any non-`ContextOverflow` variant works because `handle_degrade` only special-cases `ContextOverflow`.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib handle_degrade_trips_breaker record_cloud_success_closes_breaker -- --nocapture`
Expected: FAIL — assertions on breaker state fail (wiring not added yet).

- [ ] **Step 3: Trip the breaker in `handle_degrade`**

In `src/server.rs` `handle_degrade` (~344), after the one-shot notify, add the trip:

```rust
    if state.usage.note_degrade() {
        crate::usage::notify("localllm — cloud degraded", reason.message());
    }
    state
        .breaker
        .on_failure(crate::route_log::now_secs(), reason);
```

- [ ] **Step 4: Trip the breaker at the two cascade degrade arms**

In `cascade_or_result`, both `ForwardOutcome::Degrade(d)` arms (~478 and ~498) currently do `if state.usage.note_degrade() { notify }`. After each `note_degrade` block, add:

```rust
                        if state.usage.note_degrade() {
                            crate::usage::notify("localllm — cloud degraded", d.message());
                        }
                        state.breaker.on_failure(crate::route_log::now_secs(), d);
```

Apply to BOTH arms (the `Ok(result)` weak-cascade arm ~479 and the `Err(e)` failed-local arm ~499).

- [ ] **Step 5: Close the breaker + notify recovery in `record_cloud_success`**

In `src/server.rs` `record_cloud_success` (~317), after `state.usage.note_success();` add:

```rust
fn record_cloud_success(state: &AppState, est_prompt_tokens: usize) {
    state.usage.note_success();
    if state.breaker.on_success(crate::route_log::now_secs()) {
        crate::usage::notify("localllm — cloud recovered", "Cloud back — resuming.");
    }
    // Charge the budget with a prompt-only estimate...
```

(Leave the rest of the function unchanged.)

- [ ] **Step 6: Run the wiring tests + full build**

Run: `cargo test --lib handle_degrade_trips_breaker record_cloud_success_closes_breaker -- --nocapture`
Expected: PASS.

Run: `cargo build`
Expected: success.

- [ ] **Step 7: Commit**

```bash
git add src/server.rs
git commit -F - <<'EOF'
feat(breaker): trip on degrade, close + notify on recovery

handle_degrade and both cascade degrade arms call breaker.on_failure;
record_cloud_success calls breaker.on_success and, on the recovering
edge, notifies "cloud recovered". Wiring tests build a minimal AppState.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 4: `/admin/breaker` GET + POST reset endpoints

**Files:**
- Modify: `src/server.rs` (route registration ~614-615; two handlers near `handle_budget_get`; tests in `mod tests`)

**Interfaces:**
- Consumes: `state.breaker.snapshot(now)`, `state.breaker.reset(now)`, existing `check_admin(&headers, &state)`.
- Produces:
  - `GET /admin/breaker` → `{"state","reason","next_probe_secs"}`
  - `POST /admin/breaker/reset` → `{"ok":true,"state":"closed"}`

- [ ] **Step 1: Register the routes**

In `build_router_inner` (~614), after the `/admin/budget` route line, add:

```rust
        .route("/admin/budget", get(handle_budget_get).post(handle_budget_set))
        .route("/admin/breaker", get(handle_breaker_get))
        .route("/admin/breaker/reset", post(handle_breaker_reset))
```

- [ ] **Step 2: Write the failing endpoint tests**

In `src/server.rs` `#[cfg(test)] mod tests`, add (reuse the `make_seeded_test_router`/`axum_test_*` helpers pattern already present; use the crate test helpers for GET/POST with the `x-admin-token` header):

```rust
    #[tokio::test]
    async fn breaker_get_requires_token() {
        let app = super::make_seeded_test_router(std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )));
        let code = crate::axum_test_get_status(app, "/admin/breaker").await;
        assert_eq!(code, 401);
    }

    #[tokio::test]
    async fn breaker_get_reports_closed_by_default() {
        let app = super::make_seeded_test_router(std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeMap::new(),
        )));
        let body = crate::axum_test_get_with_header(
            app, "/admin/breaker", "x-admin-token", "test-token",
        ).await;
        assert_eq!(body["state"], "closed");
    }
```

Note: confirm the exact name/signature of the crate test helper that does an authenticated GET returning parsed JSON (`axum_test_get_with_header` at `src/lib.rs:512`). If its signature differs, adapt the call.

- [ ] **Step 3: Run to verify failure**

Run: `cargo test --lib breaker_get_requires_token breaker_get_reports_closed_by_default -- --nocapture`
Expected: FAIL — handlers/routes not defined.

- [ ] **Step 4: Implement the handlers**

In `src/server.rs`, add near `handle_budget_get` (search for `async fn handle_budget_get`):

```rust
/// GET /admin/breaker — current circuit-breaker status (token-guarded).
async fn handle_breaker_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    let s = state.breaker.snapshot(crate::route_log::now_secs());
    Json(json!({
        "state": s.state,
        "reason": s.reason,
        "next_probe_secs": s.next_probe_secs,
    }))
    .into_response()
}

/// POST /admin/breaker/reset — manually force-close the breaker (retry cloud now).
async fn handle_breaker_reset(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(resp) = check_admin(&headers, &state) {
        return resp;
    }
    state.breaker.reset(crate::route_log::now_secs());
    Json(json!({ "ok": true, "state": "closed" })).into_response()
}
```

- [ ] **Step 5: Run the endpoint tests + build**

Run: `cargo test --lib breaker_get_requires_token breaker_get_reports_closed_by_default -- --nocapture`
Expected: PASS.

Run: `cargo build`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add src/server.rs
git commit -F - <<'EOF'
feat(breaker): GET /admin/breaker + POST /admin/breaker/reset

Token-guarded status snapshot for the dashboard widget and a manual
force-close for the "retry cloud now" button. Tests cover the auth gate
and the default-closed snapshot.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 5: Tray — shared breaker Arc + "Retry cloud now" menu item

**Files:**
- Modify: `src/tray.rs` (`run_tray` breaker construction ~384-410; menu item creation ~512-547; menu-event handler ~613-665)

**Interfaces:**
- Consumes: `crate::breaker::CircuitBreaker::new()`, `.reset(now)`, `crate::run_server_with_ready_policy_token(...)` (now takes a trailing `breaker` arg).
- Produces: the tray owns the same `Arc<CircuitBreaker>` the server uses; a menu click force-closes it.

- [ ] **Step 1: Construct a shared breaker and pass it to the server**

In `src/tray.rs` `run_tray`, near the `policy` construction (~384-386), add:

```rust
    let policy_for_server = policy.clone();

    // Shared circuit breaker: the server reads/trips it per request; the tray
    // "Retry cloud now" item force-closes it. Same Arc on both sides.
    let breaker: Arc<crate::breaker::CircuitBreaker> =
        Arc::new(crate::breaker::CircuitBreaker::new());
    let breaker_for_server = breaker.clone();
```

Then replace the throwaway `std::sync::Arc::new(crate::breaker::CircuitBreaker::new())` argument added in Task 2 step 9 with `breaker_for_server`:

```rust
            if let Err(e) = rt.block_on(crate::run_server_with_ready_policy_token(
                cfg,
                Some(ready_for_server),
                policy_for_server,
                admin_token_for_server,
                Some(manager_slot_for_server),
                breaker_for_server,
            )) {
```

- [ ] **Step 2: Declare the menu-item id holder**

Near the other `Option<tray_icon::menu::MenuId>` declarations (~436-438), add:

```rust
    let mut retry_cloud_id: Option<tray_icon::menu::MenuId> = None;
```

- [ ] **Step 3: Create the menu item and append it**

In the Init block where items are built (~512-547), after `let quit_item = ...; quit_id = Some(...)` and before appending items, create:

```rust
                let retry_cloud_item = MenuItem::new("↻  Retry cloud now", true, None);
                retry_cloud_id = Some(retry_cloud_item.id().clone());
```

Then append it into the menu right after the `routing_line` (so it sits near the routing status), e.g. after `menu.append(&routing_line).expect("append routing line");` (~541):

```rust
                menu.append(&routing_line).expect("append routing line");
                menu.append(&retry_cloud_item).expect("append retry cloud item");
```

- [ ] **Step 4: Handle the click**

In the menu-event `while let Ok(menu_event) = ...` loop (~613-665), add a branch (e.g. after the `logs_id` branch):

```rust
                    } else if retry_cloud_id.as_ref() == Some(&menu_event.id) {
                        breaker.reset(crate::route_log::now_secs());
                        tracing::info!("circuit breaker manually reset via tray");
```

(`breaker` is captured by the event-loop closure; it is `Clone` via `Arc` but the closure only needs a shared borrow — it is moved into the `event_loop.run` closure like the other captured values. If a borrow-checker error arises because `breaker` is also used above, clone it into a `breaker_for_menu = breaker.clone();` before the `event_loop.run(...)` closure and use that inside.)

- [ ] **Step 5: Build (tray has no unit tests — compilation is the gate)**

Run: `cargo build`
Expected: success. The tray now shares one breaker with the server.

- [ ] **Step 6: Commit**

```bash
git add src/tray.rs
git commit -F - <<'EOF'
feat(tray): share circuit breaker + "Retry cloud now" menu item

The tray constructs the Arc<CircuitBreaker> and threads it into the
server (like the routing policy), then force-closes it on demand from a
new tray menu item.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 6: Dashboard breaker widget

**Files:**
- Modify: `src/manager_ui/app.js` (`renderDashboard` ~755-804; add a widget after the toolbar `bar`)
- Modify: `src/manager_ui/style.css` (widget styles)

**Interfaces:**
- Consumes: `api("GET", "/admin/breaker")`, `api("POST", "/admin/breaker/reset")`, existing `el`, `toast`, `fmtDateTime` helpers, `TOKEN`.
- Produces: a live breaker status strip with a reset button on the dashboard.

- [ ] **Step 1: Add the widget builder + call it**

In `src/manager_ui/app.js`, inside `renderDashboard`, after `wrap.append(bar);` (~804) insert a call:

```javascript
  wrap.append(bar);

  // Circuit-breaker status strip (live cloud availability).
  const brk = el("div", "brk-strip");
  wrap.append(brk);
  paintBreaker(brk);
```

Then add this function near `renderDashboard` (top-level, e.g. right after the `renderDashboard` function definition closes):

```javascript
// Fetch + render the circuit-breaker status strip. Re-fetches on reset.
async function paintBreaker(host) {
  host.innerHTML = "";
  let b;
  try { b = await api("GET", "/admin/breaker"); }
  catch (e) { host.append(el("div", "brk-line", "breaker: " + e.message)); return; }

  const state = b.state || "closed";
  const dot = el("span", "brk-dot brk-" + state);
  const label = state === "closed" ? "Cloud disponível"
    : state === "half-open" ? "Testando cloud (probe)…"
    : `Cloud indisponível (${b.reason || "erro"})`;
  const line = el("div", "brk-line");
  line.append(dot);
  line.append(el("span", "brk-text", label));
  if (state === "open" && typeof b.next_probe_secs === "number") {
    line.append(el("span", "brk-eta", `· nova tentativa em ${b.next_probe_secs}s`));
  }
  host.append(line);

  if (state !== "closed") {
    const btn = el("button", "btn", "↻ Tentar cloud agora");
    btn.onclick = async () => {
      btn.disabled = true;
      try { await api("POST", "/admin/breaker/reset"); toast("Breaker resetado"); }
      catch (e) { toast(e.message, true); }
      paintBreaker(host);
    };
    host.append(btn);
  }
}
```

- [ ] **Step 2: Add widget styles**

In `src/manager_ui/style.css`, append:

```css
/* Circuit-breaker status strip on the dashboard */
.brk-strip { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  margin: 4px 0 12px; padding: 10px 14px; border-radius: 10px;
  background: var(--panel, #1b1f24); border: 1px solid rgba(255,255,255,.06); }
.brk-line { display: flex; align-items: center; gap: 8px; }
.brk-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.brk-closed { background: #33c06a; }
.brk-open { background: #e5484d; }
.brk-half-open { background: #f5a524; }
.brk-text { font-weight: 600; }
.brk-eta { opacity: .7; font-size: .9em; }
```

Note: reuse the existing CSS variable names actually present in `style.css` (check `--panel` etc.; if the file uses different variable names, substitute the real ones).

- [ ] **Step 3: Manual visual verification**

There is no JS test harness. Verify by building and loading the dashboard:

Run: `cargo build`
Expected: success (the embedded assets are string-included; a build confirms no Rust breakage).

Manual (deferred to the user's end-to-end test): open Config → Dashboard, confirm the strip shows "Cloud disponível" when healthy, and shows the reset button + countdown when the breaker is open.

- [ ] **Step 4: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -F - <<'EOF'
feat(ui): dashboard circuit-breaker status widget

Live strip on the dashboard shows cloud availability (closed/half-open/
open), the next-probe countdown, and a "retry cloud now" button that
POSTs /admin/breaker/reset.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

## Final Verification (after all tasks)

- [ ] Full lib suite in background: `cargo test --lib` → expect all green (Fase A left it at 212+; Task 1/2/3/4 add ~15 tests).
- [ ] Integration tests single-threaded (known pre-existing parallel flake in `tests/http.rs`): `cargo test --test http -- --test-threads=1` → expect all green.
- [ ] `cargo build` (bin + tray) clean.

## Self-Review Notes

- **Spec coverage:** state machine + backoff (Task 1); route gate + precedence + `"CloudDown"` (Task 2); trip/recover + recovery notify (Task 3); endpoints (Task 4); tray reset + shared Arc (Task 5); dashboard widget (Task 6). All spec sections mapped.
- **Type consistency:** `CircuitBreaker`, `Gate::{Allow, Block{reason, next_probe_in}}`, `BreakerSnapshot{state, reason, next_probe_secs}`, `reason_label`, `breaker_gate_block` used identically across tasks. `router()` and `run_server_with_ready_policy_token` each gain exactly one trailing `breaker` param; test-router constructors build their own.
- **Known risk:** `RouteReason` variant names in Task 2/3 tests must be confirmed against `src/route/mod.rs` before writing (only `ContextOverflow` is behaviorally special in `handle_degrade`).
