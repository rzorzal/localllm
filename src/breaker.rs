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
    ///
    /// While HalfOpen every caller gets `Allow`, so under concurrency several
    /// simultaneous probes may hit cloud within one half-open window. This is
    /// intentional: favoring self-healing convergence over a strict single
    /// probe means a dropped probe can never wedge the breaker (no reaper).
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
    /// capped); already Open → no-op (keeps its existing schedule).
    pub fn on_failure(&self, now: u64, reason: DegradeReason) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let next_backoff = match *st {
            BreakerState::Closed => BASE_COOLDOWN_SECS,
            // A failed probe: double the carried backoff, capped.
            BreakerState::HalfOpen { backoff, .. } => (backoff * 2).min(MAX_COOLDOWN_SECS),
            // Already Open: keep the existing schedule. A stray failure while
            // Open (e.g. an ungated cascade escalation) must not push `until`
            // out, or the breaker could starve of probes during sustained
            // cascade traffic. The half-open probe owns backoff escalation.
            BreakerState::Open { .. } => return,
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

    /// Passive-read advance for the dashboard poll: if `Open` with an elapsed
    /// cooldown, transition to `HalfOpen` (carrying `backoff` + `reason`, as
    /// `gate` does) so recovery happens without needing a cloud request, then
    /// return the snapshot of the resulting state. Unlike `snapshot` this may
    /// MUTATE; unlike `gate` it grants no `Allow` to a caller. `on_failure`
    /// still owns backoff escalation, so a failed probe after this doubles the
    /// cooldown rather than resetting it.
    pub fn poll(&self, now: u64) -> BreakerSnapshot {
        {
            let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let BreakerState::Open {
                until,
                backoff,
                reason,
            } = *st
            {
                if now >= until {
                    *st = BreakerState::HalfOpen { backoff, reason };
                }
            }
        } // drop the lock before snapshot re-acquires it
        self.snapshot(now)
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
            BreakerSnapshot {
                state: "closed",
                reason: None,
                next_probe_secs: None
            }
        );
    }

    #[test]
    fn first_failure_opens_for_base_cooldown() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Quota);
        assert_eq!(
            b.gate(100),
            Gate::Block {
                reason: DegradeReason::Quota,
                next_probe_in: BASE_COOLDOWN_SECS
            }
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
            Gate::Block {
                reason: DegradeReason::Offline,
                next_probe_in: 60
            }
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
    fn failure_while_open_keeps_schedule() {
        let b = CircuitBreaker::new();
        b.on_failure(0, DegradeReason::Quota); // Open, until=30, backoff=30
        let before = b.snapshot(10);
        assert_eq!(before.next_probe_secs, Some(20)); // 30 - 10
                                                      // A stray failure while still Open (no gate/half-open first) is a no-op.
        b.on_failure(10, DegradeReason::ServerError);
        let after = b.snapshot(10);
        assert_eq!(after.next_probe_secs, Some(20)); // unchanged: until still 30
        assert_eq!(after.reason, Some("Quota")); // original reason retained
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

    #[test]
    fn poll_elapsed_open_transitions_to_half_open() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Quota); // Open until 130, backoff 30
        // Not elapsed: poll leaves it Open with the correct countdown.
        let s = b.poll(120);
        assert_eq!(s.state, "open");
        assert_eq!(s.next_probe_secs, Some(10));
        // Elapsed: poll transitions to half-open and reports the reason.
        let s = b.poll(130);
        assert_eq!(s.state, "half-open");
        assert_eq!(s.reason, Some("Quota"));
        // It is a REAL transition, not just a reported label: the next gate
        // returns Allow (the probe), which only happens from Closed/HalfOpen.
        assert_eq!(b.gate(130), Gate::Allow);
    }

    #[test]
    fn poll_is_noop_on_closed_and_half_open() {
        let b = CircuitBreaker::new();
        assert_eq!(b.poll(0).state, "closed"); // closed → no-op
        b.on_failure(0, DegradeReason::Offline); // Open until 30
        assert_eq!(b.gate(30), Gate::Allow); // → half-open via gate
        assert_eq!(b.poll(30).state, "half-open"); // no-op
        assert_eq!(b.poll(999).state, "half-open"); // still half-open, no-op
    }

    #[test]
    fn poll_preserves_backoff_escalation() {
        let b = CircuitBreaker::new();
        b.on_failure(0, DegradeReason::Quota); // backoff 30, Open until 30
        let s = b.poll(30); // half-open via poll (not gate)
        assert_eq!(s.state, "half-open");
        b.on_failure(30, DegradeReason::Quota); // failed probe → doubled to 60
        // Backoff escalated, NOT reset to base 30.
        assert_eq!(b.snapshot(30).next_probe_secs, Some(60));
    }
}
