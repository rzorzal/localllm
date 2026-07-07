//! Daily cloud-spend tracker for the budget cap. Day = UTC day (no tz dep).

use std::sync::Mutex;

/// UTC day bucket for a unix-seconds timestamp.
pub fn day_index(ts: i64) -> i64 {
    ts.div_euclid(86_400)
}

#[derive(Debug, Default)]
struct DaySpend {
    day: i64,
    spent: f64,
}

#[derive(Debug, Default)]
pub struct Budget {
    inner: Mutex<DaySpend>,
}

impl Budget {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add cost to today's total, resetting first if the UTC day rolled over.
    pub fn note_cloud_cost_at(&self, usd: f64, now: i64) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let today = day_index(now);
        if g.day != today {
            g.day = today;
            g.spent = 0.0;
        }
        g.spent += usd;
    }

    /// Convenience: cost at the current wall clock.
    pub fn note_cloud_cost(&self, usd: f64) {
        self.note_cloud_cost_at(usd, crate::route_log::now_secs());
    }

    /// Today's spend (0 if the stored day is stale).
    pub fn spent_today(&self, now: i64) -> f64 {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if g.day == day_index(now) {
            g.spent
        } else {
            0.0
        }
    }

    /// Whether today's spend has reached `limit`. `limit <= 0` means disabled.
    pub fn is_over(&self, now: i64, limit: f64) -> bool {
        limit > 0.0 && self.spent_today(now) >= limit
    }

    /// Seed today's spend by summing today's cloud costs from the log. Cloud
    /// cost = price_for(model).cost(prompt, completion) using the joined outcome.
    pub fn seed_from_log(&self, entries: &[crate::route_log::LogLine], now: i64) {
        use crate::route_log::LogLine;
        use std::collections::HashMap;
        let mut outcomes: HashMap<&str, u64> = HashMap::new();
        for l in entries {
            if let LogLine::Outcome(o) = l {
                outcomes.insert(o.rid.as_str(), o.completion_tok.unwrap_or(0));
            }
        }
        let today = day_index(now);
        let mut total = 0.0;
        for l in entries {
            if let LogLine::Decision(d) = l {
                if d.dest == "cloud" && day_index(d.ts) == today {
                    let completion = outcomes.get(d.rid.as_str()).copied().unwrap_or(0);
                    let price = crate::pricing::price_for(d.model.as_deref().unwrap_or(""));
                    total += price.cost(d.prompt_tok, completion);
                }
            }
        }
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.day = today;
        g.spent = total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_log::{LogLine, OutcomeEntry, RouteEntry};

    #[test]
    fn accumulates_and_resets_on_new_day() {
        let b = Budget::new();
        let t0 = 100_000i64; // some day
        b.note_cloud_cost_at(1.5, t0);
        b.note_cloud_cost_at(1.0, t0 + 10);
        assert!((b.spent_today(t0 + 20) - 2.5).abs() < 1e-9);
        // next day → resets
        let t1 = t0 + 86_400;
        assert!((b.spent_today(t1) - 0.0).abs() < 1e-9);
        assert!(b.is_over(t0 + 20, 2.0));
        assert!(!b.is_over(t0 + 20, 3.0));
        assert!(!b.is_over(t0 + 20, 0.0)); // 0 limit = disabled
    }

    #[test]
    fn seeds_today_cloud_cost_from_log() {
        let b = Budget::new();
        let now = 200_000i64;
        // one cloud decision today with a cost; one local (ignored); one old cloud
        let lines = vec![
            LogLine::Decision(RouteEntry {
                ts: now - 10,
                rid: "c1".into(),
                dest: "cloud".into(),
                prompt_tok: 1_000_000,
                model: Some("claude-sonnet-4-6".into()),
                ..Default::default()
            }),
            LogLine::Outcome(OutcomeEntry {
                rid: "c1".into(),
                ts: now - 9,
                completion_tok: Some(0),
                ..Default::default()
            }),
            LogLine::Decision(RouteEntry {
                ts: now - 86_400 - 10,
                rid: "old".into(),
                dest: "cloud".into(),
                prompt_tok: 1_000_000,
                model: Some("claude-sonnet-4-6".into()),
                ..Default::default()
            }),
        ];
        b.seed_from_log(&lines, now);
        // c1: 1M prompt @ sonnet $3 = $3.0 (completion 0)
        assert!((b.spent_today(now) - 3.0).abs() < 1e-6);
    }
}
