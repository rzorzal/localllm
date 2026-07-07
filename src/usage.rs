//! Per-session cloud usage counters and one-shot user notifications.
//!
//! Counters are atomic so the shared `Usage` can be read/updated from any
//! request without a lock. Notifications are one-shot (degrade re-arms only
//! after a later success) and are a no-op unless the real server has called
//! [`enable_notifications`], so test runs never raise a desktop notification.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Why a cloud request degraded to local. Carried into the user notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeReason {
    Auth,
    Quota,
    ServerError,
    Offline,
}

impl DegradeReason {
    /// Short user-facing description for the notification body.
    pub fn message(&self) -> &'static str {
        match self {
            DegradeReason::Auth => "Cloud auth failed — serving locally.",
            DegradeReason::Quota => "Cloud quota exceeded — serving locally.",
            DegradeReason::ServerError => "Cloud error — serving locally.",
            DegradeReason::Offline => "No internet — serving locally.",
        }
    }
}

/// Atomic per-session cloud usage + one-shot notification gates.
#[derive(Debug, Default)]
pub struct Usage {
    calls: AtomicU64,
    prompt_tokens: AtomicU64,
    /// True while in a degraded run (cloud failing); gates one-shot degrade alert.
    degraded: AtomicBool,
    /// True once the high-usage alert has fired this session.
    high_alerted: AtomicBool,
}

impl Usage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one successful cloud call of `prompt_tokens`. Returns `true` exactly
    /// once — the call that first brings cumulative prompt tokens to/over
    /// `threshold` — so the caller fires the high-usage alert a single time.
    pub fn record_cloud_call(&self, prompt_tokens: usize, threshold: usize) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let total = self
            .prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::SeqCst)
            + prompt_tokens as u64;
        if total >= threshold as u64
            && self
                .high_alerted
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            return true;
        }
        false
    }

    /// Mark a degrade. Returns `true` only on the transition into the degraded
    /// state, so the degrade notification fires once until a success re-arms it.
    pub fn note_degrade(&self) -> bool {
        self.degraded
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Clear the degrade gate after a successful cloud call so a later failure
    /// notifies again.
    pub fn note_success(&self) {
        self.degraded.store(false, Ordering::SeqCst);
    }

    /// (calls, prompt_tokens) so far this session.
    pub fn totals(&self) -> (u64, u64) {
        (
            self.calls.load(Ordering::SeqCst),
            self.prompt_tokens.load(Ordering::SeqCst),
        )
    }
}

static NOTIFY_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable real user notifications. Called once by the running server; left off in
/// tests so `osascript` never spawns.
pub fn enable_notifications() {
    NOTIFY_ENABLED.store(true, Ordering::SeqCst);
}

/// Show a cross-platform desktop notification when enabled; otherwise a no-op
/// (always logs). Uses `notify-rust`: libnotify/D-Bus on Linux, toast on
/// Windows, and the native Notification Center on macOS — which picks up the
/// `localllm.app` bundle icon when launched from the bundle. Best-effort:
/// failures (no D-Bus session, no bundle, etc.) are ignored.
pub fn notify(title: &str, body: &str) {
    tracing::info!(target: "localllm::req", "notify: {title} — {body}");
    if !NOTIFY_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    let _ = notify_rust::Notification::new()
        .summary(title)
        .body(body)
        .show();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_usage_alert_fires_exactly_once_on_crossing() {
        let u = Usage::new();
        // threshold 100; first call 60 → under, no alert
        assert!(!u.record_cloud_call(60, 100));
        // second call 60 → total 120 ≥ 100 → alert once
        assert!(u.record_cloud_call(60, 100));
        // further calls never re-alert
        assert!(!u.record_cloud_call(60, 100));
        let (calls, toks) = u.totals();
        assert_eq!(calls, 3);
        assert_eq!(toks, 180);
    }

    #[test]
    fn degrade_is_one_shot_until_success_rearms() {
        let u = Usage::new();
        assert!(u.note_degrade()); // first degrade → notify
        assert!(!u.note_degrade()); // still degraded → no repeat
        u.note_success(); // cloud recovered
        assert!(u.note_degrade()); // degrade again → notify again
    }
}
