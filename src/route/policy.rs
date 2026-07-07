//! Routing profiles and their tunable knobs.
//!
//! A `Profile` is the user-facing choice (set from the tray in a later phase);
//! it maps to a `RoutingPolicy` of concrete knobs consumed by `route::decide`.

/// User-facing routing choice. Selected from the tray; persisted in settings.
/// Defaults to `SaveTokens` (the token-thrift, local-first profile).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    /// Local-first; cloud only when local truly cannot serve. Cheapest.
    #[default]
    SaveTokens,
    /// Local for easy/medium, cloud for hard/big-context.
    Balanced,
    /// Cloud-first; local only for trivial calls. Best quality, most tokens.
    MaxQuality,
    /// Never route to cloud. Pure local, zero tokens.
    LocalOnly,
}

impl Profile {
    /// All profiles in the order they appear in the tray menu.
    pub const ALL: [Profile; 4] = [
        Profile::SaveTokens,
        Profile::Balanced,
        Profile::MaxQuality,
        Profile::LocalOnly,
    ];

    /// Human-readable menu/status label.
    pub fn label(&self) -> &'static str {
        match self {
            Profile::SaveTokens => "Save tokens (local-first)",
            Profile::Balanced => "Balanced (smart)",
            Profile::MaxQuality => "Max quality (cloud-first)",
            Profile::LocalOnly => "Local only (offline)",
        }
    }

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
                // Lowered 0.6 → 0.45 so medium-difficulty turns escalate to cloud
                // more readily (Balanced was staying too local on real workloads).
                escalation_threshold: 0.45,
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

/// Write `p`'s knobs into the shared policy under the lock. Used by the tray to
/// switch routing live without a restart.
pub fn apply_profile(lock: &std::sync::RwLock<RoutingPolicy>, p: Profile) {
    *lock.write().unwrap() = p.policy();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_profile_in_menu_order() {
        assert_eq!(
            Profile::ALL,
            [
                Profile::SaveTokens,
                Profile::Balanced,
                Profile::MaxQuality,
                Profile::LocalOnly
            ]
        );
    }

    #[test]
    fn labels_are_human_readable() {
        assert_eq!(Profile::SaveTokens.label(), "Save tokens (local-first)");
        assert_eq!(Profile::LocalOnly.label(), "Local only (offline)");
    }

    #[test]
    fn apply_profile_updates_policy_live() {
        let lock = std::sync::RwLock::new(Profile::SaveTokens.policy());
        apply_profile(&lock, Profile::MaxQuality);
        assert_eq!(*lock.read().unwrap(), Profile::MaxQuality.policy());
    }

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
