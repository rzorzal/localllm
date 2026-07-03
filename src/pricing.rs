//! Static, code-maintained cloud model pricing ($/1M tokens). No user editing.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    pub in_per_1m: f64,
    pub out_per_1m: f64,
}

impl Price {
    /// USD cost for a call of the given token counts.
    pub fn cost(&self, prompt_tok: u64, completion_tok: u64) -> f64 {
        (prompt_tok as f64 * self.in_per_1m + completion_tok as f64 * self.out_per_1m) / 1_000_000.0
    }
}

/// Estimated price for an unrecognized model (mid-tier). Documented estimate.
pub const FALLBACK: Price = Price { in_per_1m: 3.0, out_per_1m: 15.0 };

/// (substring of the model id, $/1M input, $/1M output). First match wins, so
/// order more-specific patterns before broader ones. Estimates as of 2026-07;
/// update as providers change pricing.
const MODEL_PRICES: &[(&str, f64, f64)] = &[
    ("claude-opus-4", 15.0, 75.0),
    ("claude-sonnet-4", 3.0, 15.0),
    ("claude-haiku-4", 1.0, 5.0),
    ("claude-3-5-haiku", 0.8, 4.0),
    ("claude-3-opus", 15.0, 75.0),
    ("gpt-5", 1.25, 10.0),
    ("gpt-4o-mini", 0.15, 0.6),
    ("gpt-4o", 2.5, 10.0),
    ("gpt-4.1", 2.0, 8.0),
    ("o3", 2.0, 8.0),
    ("o1", 15.0, 60.0),
];

/// Price for a model id, matched case-insensitively by substring; `FALLBACK`
/// when nothing matches.
pub fn price_for(model_id: &str) -> Price {
    let m = model_id.to_lowercase();
    for (pat, in_p, out_p) in MODEL_PRICES {
        if m.contains(pat) {
            return Price { in_per_1m: *in_p, out_per_1m: *out_p };
        }
    }
    FALLBACK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_match_and_unknown_falls_back() {
        assert_eq!(price_for("claude-opus-4-8"), Price { in_per_1m: 15.0, out_per_1m: 75.0 });
        assert_eq!(price_for("claude-sonnet-4-6"), Price { in_per_1m: 3.0, out_per_1m: 15.0 });
        assert_eq!(price_for("gpt-5"), Price { in_per_1m: 1.25, out_per_1m: 10.0 });
        // unknown → fallback
        assert_eq!(price_for("some-random-model"), FALLBACK);
    }

    #[test]
    fn cost_arithmetic() {
        let p = Price { in_per_1m: 3.0, out_per_1m: 15.0 };
        // 1M prompt @3 + 1M completion @15 = 18
        assert!((p.cost(1_000_000, 1_000_000) - 18.0).abs() < 1e-9);
    }
}
