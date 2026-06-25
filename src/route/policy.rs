//! Routing profiles and their tunable knobs.
//!
//! A `Profile` is the user-facing choice (set from the tray in a later phase);
//! it maps to a `RoutingPolicy` of concrete knobs consumed by `route::decide`.

/// User-facing routing choice. Set from the tray in a later phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Local-first; cloud only when local truly cannot serve. Cheapest.
    SaveTokens,
    /// Local for easy/medium, cloud for hard/big-context.
    Balanced,
    /// Cloud-first; local only for trivial calls. Best quality, most tokens.
    MaxQuality,
    /// Never route to cloud. Pure local, zero tokens.
    LocalOnly,
}

impl Default for Profile {
    fn default() -> Self {
        Profile::SaveTokens
    }
}

impl Profile {
    /// Map a profile to its concrete routing knobs.
    pub fn policy(&self) -> RoutingPolicy {
        match self {
            Profile::SaveTokens => RoutingPolicy {
                escalation_threshold: 0.9,
                cascade: true,
                ctx_gate_frac: 0.95,
                allow_cloud: true,
            },
            Profile::Balanced => RoutingPolicy {
                escalation_threshold: 0.6,
                cascade: true,
                ctx_gate_frac: 0.9,
                allow_cloud: true,
            },
            Profile::MaxQuality => RoutingPolicy {
                escalation_threshold: 0.2,
                cascade: false,
                ctx_gate_frac: 0.75,
                allow_cloud: true,
            },
            Profile::LocalOnly => RoutingPolicy {
                escalation_threshold: 1.0,
                cascade: false,
                ctx_gate_frac: 1.0,
                allow_cloud: false,
            },
        }
    }
}

/// Concrete routing knobs derived from a `Profile`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingPolicy {
    /// Difficulty-score cutoff above which a request goes to cloud (Phase B).
    pub escalation_threshold: f64,
    /// Whether a weak local result may escalate to cloud (Phase B).
    pub cascade: bool,
    /// Fraction of the local context window above which a prompt is too big
    /// for local and must go to cloud.
    pub ctx_gate_frac: f64,
    /// Whether cloud routing is permitted at all.
    pub allow_cloud: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_tokens_is_local_biased_and_allows_cloud() {
        let p = Profile::SaveTokens.policy();
        assert!(p.allow_cloud);
        assert!(p.cascade);
        assert!((p.escalation_threshold - 0.9).abs() < 1e-9);
        assert!((p.ctx_gate_frac - 0.95).abs() < 1e-9);
    }

    #[test]
    fn local_only_forbids_cloud() {
        let p = Profile::LocalOnly.policy();
        assert!(!p.allow_cloud);
        assert!(!p.cascade);
    }

    #[test]
    fn max_quality_has_low_threshold_no_cascade() {
        let p = Profile::MaxQuality.policy();
        assert!(p.allow_cloud);
        assert!(!p.cascade);
        assert!((p.escalation_threshold - 0.2).abs() < 1e-9);
    }

    #[test]
    fn default_profile_is_save_tokens() {
        assert_eq!(Profile::default(), Profile::SaveTokens);
    }
}
