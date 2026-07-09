//! Persisted user settings: routing profile and client-integration toggle state.
//!
//! Stored as JSON at `<config-dir>/localllm/settings.json` (e.g.
//! `~/Library/Application Support/localllm/settings.json` on macOS). The path
//! is overridable via the `LOCALLLM_SETTINGS` env var (full file path), used by
//! tests and power users.

use std::path::PathBuf;

use crate::route::Profile;

/// Per-model execution profile. Every field optional: `None` means "fall back
/// to the catalog recommendation, then the global CLI default".
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_type: Option<crate::config::KvType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}

/// The last successfully-activated model, restored at boot unless the CLI
/// explicitly overrides `--model-id`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ActiveModel {
    pub repo: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}

/// On-disk settings shape. New fields must be `#[serde(default)]` so older
/// files (which only had `profile`) still load.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    profile: Profile,
    /// Legacy per-model ctx map. Read-only for migration; new writes go to
    /// `model_profiles`. Kept so old files still load.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    model_ctx: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    model_profiles: std::collections::BTreeMap<String, ExecProfile>,
    #[serde(default)]
    tool_filters: std::collections::BTreeMap<String, Vec<String>>,
    /// Per-surface last-seen discovered tool names, persisted so the Tools view
    /// shows them immediately after a restart (before any new request refreshes).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    tool_seen: std::collections::BTreeMap<String, Vec<String>>,
    /// Per-surface tool descriptions (name -> desc), persisted so the Tools view
    /// can expand descriptions right after a restart, before any new request.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    tool_descs: std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_model: Option<ActiveModel>,
    /// Global toggle: when true, history trimming selects turns by relevance
    /// (BM25+MMR) instead of pure recency. Default false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    smart_history: bool,
    /// User override for the Balanced profile's difficulty cutoff (`[0.0, 1.0]`).
    /// `None` keeps the built-in default. Only affects the Balanced profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    balanced_threshold: Option<f64>,
    /// User override for the cold-prefill gate (`[1.0, 3600.0]` seconds).
    /// `None` keeps the built-in default (360 s) for all profiles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cold_prefill_gate_secs: Option<f64>,
    /// Whether the daily cloud-spend budget cap is enforced.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    budget_enabled: bool,
    /// Daily cloud-spend cap in USD. 0 = no cap.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    budget_daily_usd: f64,
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

/// Resolve the settings file path. `LOCALLLM_SETTINGS` (full file path) wins;
/// otherwise `<config-dir>/localllm/settings.json`. Returns `None` only if no
/// config directory can be determined and no override is set.
pub fn settings_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LOCALLLM_SETTINGS") {
        return Some(PathBuf::from(p));
    }
    dirs::config_dir().map(|d| d.join("localllm").join("settings.json"))
}

/// Load the full settings object, or defaults if the file is absent/unreadable/
/// malformed. Never fails — a bad settings file must not stop startup.
fn load_settings() -> Settings {
    let mut s = match settings_path()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .and_then(|t| serde_json::from_str::<Settings>(&t).ok())
    {
        Some(s) => s,
        None => Settings::default(),
    };
    // Migrate any legacy model_ctx entries into model_profiles.ctx.
    for (k, ctx) in std::mem::take(&mut s.model_ctx) {
        s.model_profiles
            .entry(k)
            .or_default()
            .ctx
            .get_or_insert(ctx);
    }
    s
}

/// Persist the full settings object, creating the parent directory if needed.
fn save_settings(s: &Settings) -> anyhow::Result<()> {
    let path =
        settings_path().ok_or_else(|| anyhow::anyhow!("no settings path (no config dir)"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(s)?;
    crate::fsutil::atomic_write(&path, json.as_bytes())?;
    Ok(())
}

/// Load the saved routing profile, or the default if absent/unreadable/malformed.
pub fn load_profile() -> Profile {
    load_settings().profile
}

/// Resolve the effective startup profile: an explicit CLI choice wins; else the
/// saved setting (or the default if none/unreadable).
pub fn resolve_profile(cli: Option<Profile>) -> Profile {
    cli.unwrap_or_else(load_profile)
}

/// Persist the chosen routing profile, preserving the integrations block.
pub fn save_profile(p: Profile) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.profile = p;
    save_settings(&s)
}

/// The Balanced profile's difficulty cutoff — the user override if set, else the
/// built-in default. Always clamped to `[0.0, 1.0]`.
pub fn load_balanced_threshold() -> f64 {
    load_settings()
        .balanced_threshold
        .unwrap_or(
            crate::route::Profile::Balanced
                .policy()
                .escalation_threshold,
        )
        .clamp(0.0, 1.0)
}

/// Persist the Balanced difficulty cutoff (clamped to `[0.0, 1.0]`), preserving
/// the rest of the settings file.
pub fn save_balanced_threshold(t: f64) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.balanced_threshold = Some(t.clamp(0.0, 1.0));
    save_settings(&s)
}

/// The cold-prefill gate — the user override if set, else the built-in default
/// (360 s). Always clamped to `[1.0, 3600.0]`.
pub fn load_cold_prefill_gate() -> f64 {
    load_settings()
        .cold_prefill_gate_secs
        .unwrap_or(360.0)
        .clamp(1.0, 3600.0)
}

/// Persist the cold-prefill gate (clamped to `[1.0, 3600.0]`), preserving
/// the rest of the settings file.
pub fn save_cold_prefill_gate(secs: f64) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.cold_prefill_gate_secs = Some(secs.clamp(1.0, 3600.0));
    save_settings(&s)
}

/// Load (enabled, daily_usd) for the budget cap.
pub fn load_budget() -> (bool, f64) {
    let s = load_settings();
    (s.budget_enabled, s.budget_daily_usd)
}

/// Persist the budget cap config, preserving the rest.
pub fn save_budget(enabled: bool, daily_usd: f64) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.budget_enabled = enabled;
    s.budget_daily_usd = daily_usd.max(0.0);
    save_settings(&s)
}

/// The concrete policy for a profile, applying any user overrides. The Balanced
/// profile's difficulty cutoff comes from [`load_balanced_threshold`]; the
/// cold-prefill gate applies to all profiles; every other knob is unchanged.
pub fn resolve_policy(p: Profile) -> crate::route::RoutingPolicy {
    let mut pol = p.policy();
    if p == Profile::Balanced {
        pol.escalation_threshold = load_balanced_threshold();
    }
    pol.cold_prefill_gate_secs = load_cold_prefill_gate();
    pol
}

/// Load the last-activated model, if any was saved.
pub fn load_active_model() -> Option<ActiveModel> {
    load_settings().active_model
}

/// Persist the last-activated model (called on a successful switch).
pub fn save_active_model(repo: &str, file: &str, quant: Option<&str>) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.active_model = Some(ActiveModel {
        repo: repo.to_string(),
        file: file.to_string(),
        quant: quant.map(|q| q.to_string()),
    });
    save_settings(&s)
}

/// Resolve the boot model: an explicit CLI `--model-id` (i.e. one that differs
/// from the compiled default) wins; otherwise the saved model; otherwise the
/// default. Split-model file lists from the saved entry are not restored
/// (single-file only) — a saved model always resolves to `[file]`.
pub fn resolve_active_model(
    cli_model: &str,
    cli_files: &[String],
    default_model: &str,
    default_file: &str,
    saved: Option<ActiveModel>,
) -> (String, Vec<String>) {
    if cli_model != default_model {
        return (cli_model.to_string(), cli_files.to_vec());
    }
    match saved {
        Some(a) => (a.repo, vec![a.file]),
        None => (default_model.to_string(), vec![default_file.to_string()]),
    }
}

/// Settings key for a model's per-model ctx override: `"{repo}/{file}"`.
pub fn model_ctx_key(repo: &str, file: &str) -> String {
    format!("{repo}/{file}")
}

/// Load a model's full execution profile, or defaults if unset.
pub fn load_model_profile(key: &str) -> ExecProfile {
    load_settings()
        .model_profiles
        .get(key)
        .cloned()
        .unwrap_or_default()
}

/// Persist a model's execution profile, preserving the rest of settings.
pub fn save_model_profile(key: &str, p: &ExecProfile) -> anyhow::Result<()> {
    let mut s = load_settings();
    if *p == ExecProfile::default() {
        s.model_profiles.remove(key);
    } else {
        s.model_profiles.insert(key.to_string(), p.clone());
    }
    save_settings(&s)
}

/// Remove a model's execution profile, preserving the rest of settings.
pub fn clear_model_profile(key: &str) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.model_profiles.remove(key);
    save_settings(&s)
}

// --- ctx back-compat wrappers, now backed by the profile's ctx field ---

/// Load a model's persisted ctx override, or `None` if unset.
pub fn load_model_ctx(key: &str) -> Option<u32> {
    load_model_profile(key).ctx
}

/// Persist a model's ctx override, preserving the rest of settings.
pub fn save_model_ctx(key: &str, ctx: u32) -> anyhow::Result<()> {
    let mut p = load_model_profile(key);
    p.ctx = Some(ctx);
    save_model_profile(key, &p)
}

/// Remove a model's ctx override, preserving the rest of settings.
pub fn clear_model_ctx(key: &str) -> anyhow::Result<()> {
    let mut p = load_model_profile(key);
    p.ctx = None;
    save_model_profile(key, &p)
}

/// Load a surface's disabled-tool blocklist, or an empty list if unset.
pub fn load_tool_filter(surface: &str) -> Vec<String> {
    load_settings()
        .tool_filters
        .get(surface)
        .cloned()
        .unwrap_or_default()
}

/// Persist a surface's blocklist (empty list clears it), preserving the rest.
pub fn save_tool_filter(surface: &str, disabled: &[String]) -> anyhow::Result<()> {
    let mut s = load_settings();
    if disabled.is_empty() {
        s.tool_filters.remove(surface);
    } else {
        s.tool_filters
            .insert(surface.to_string(), disabled.to_vec());
    }
    save_settings(&s)
}

/// Load the global smart-history toggle (default false = recency truncation).
pub fn load_smart_history() -> bool {
    load_settings().smart_history
}

/// Persist the global smart-history toggle, preserving the rest of settings.
pub fn save_smart_history(on: bool) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.smart_history = on;
    save_settings(&s)
}

/// Load a surface's persisted last-seen discovered tool names.
pub fn load_tool_seen(surface: &str) -> Vec<String> {
    load_settings()
        .tool_seen
        .get(surface)
        .cloned()
        .unwrap_or_default()
}

/// Persist a surface's last-seen tool names (empty clears it), preserving the rest.
pub fn save_tool_seen(surface: &str, seen: &[String]) -> anyhow::Result<()> {
    let mut s = load_settings();
    if seen.is_empty() {
        s.tool_seen.remove(surface);
    } else {
        s.tool_seen.insert(surface.to_string(), seen.to_vec());
    }
    save_settings(&s)
}

/// Load a surface's persisted tool descriptions (name -> desc).
pub fn load_tool_descs(surface: &str) -> std::collections::BTreeMap<String, String> {
    load_settings()
        .tool_descs
        .get(surface)
        .cloned()
        .unwrap_or_default()
}

/// Persist a surface's tool descriptions (empty clears it), preserving the rest.
pub fn save_tool_descs(
    surface: &str,
    descs: &std::collections::BTreeMap<String, String>,
) -> anyhow::Result<()> {
    let mut s = load_settings();
    if descs.is_empty() {
        s.tool_descs.remove(surface);
    } else {
        s.tool_descs.insert(surface.to_string(), descs.clone());
    }
    save_settings(&s)
}

/// Serialises tests (in any module) that mutate the process-global
/// LOCALLLM_SETTINGS env var, so parallel runs don't clobber each other.
#[cfg(test)]
pub(crate) static SETTINGS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_seen_round_trips_and_clears_and_preserves_filter() {
        with_temp_settings(|| {
            save_tool_filter("claude-code", &["Bash".to_string()]).unwrap();
            save_tool_seen("claude-code", &["Bash".to_string(), "Read".to_string()]).unwrap();
            assert_eq!(
                load_tool_seen("claude-code"),
                vec!["Bash".to_string(), "Read".to_string()]
            );
            // seen persistence must not disturb the disabled filter
            assert_eq!(load_tool_filter("claude-code"), vec!["Bash".to_string()]);
            // empty clears
            save_tool_seen("claude-code", &[]).unwrap();
            assert!(load_tool_seen("claude-code").is_empty());
            assert_eq!(load_tool_filter("claude-code"), vec!["Bash".to_string()]);
        });
    }

    #[test]
    fn tool_descs_round_trip_and_clear_and_preserve_seen() {
        with_temp_settings(|| {
            save_tool_seen("claude-code", &["Bash".to_string()]).unwrap();
            let mut d = std::collections::BTreeMap::new();
            d.insert("Bash".to_string(), "Run a shell command".to_string());
            save_tool_descs("claude-code", &d).unwrap();
            assert_eq!(load_tool_descs("claude-code"), d);
            // desc persistence must not disturb the seen set
            assert_eq!(load_tool_seen("claude-code"), vec!["Bash".to_string()]);
            // empty clears
            save_tool_descs("claude-code", &std::collections::BTreeMap::new()).unwrap();
            assert!(load_tool_descs("claude-code").is_empty());
            assert_eq!(load_tool_seen("claude-code"), vec!["Bash".to_string()]);
        });
    }

    #[test]
    fn budget_round_trips_and_preserves_profile() {
        with_temp_settings(|| {
            save_profile(Profile::Balanced).unwrap();
            save_budget(true, 5.0).unwrap();
            assert_eq!(load_budget(), (true, 5.0));
            assert_eq!(load_profile(), Profile::Balanced);
            save_budget(false, 0.0).unwrap();
            assert_eq!(load_budget(), (false, 0.0));
        });
    }

    #[test]
    fn balanced_threshold_round_trips_clamps_and_resolves() {
        with_temp_settings(|| {
            // Default = the profile's built-in threshold.
            let default_t = Profile::Balanced.policy().escalation_threshold;
            assert!((load_balanced_threshold() - default_t).abs() < 1e-9);
            // Save + read back; out-of-range clamps to [0,1].
            save_balanced_threshold(0.7).unwrap();
            assert!((load_balanced_threshold() - 0.7).abs() < 1e-9);
            save_balanced_threshold(1.5).unwrap();
            assert!((load_balanced_threshold() - 1.0).abs() < 1e-9);
            // resolve_policy applies the override only to Balanced.
            assert!((resolve_policy(Profile::Balanced).escalation_threshold - 1.0).abs() < 1e-9);
            assert_eq!(
                resolve_policy(Profile::MaxQuality).escalation_threshold,
                Profile::MaxQuality.policy().escalation_threshold
            );
        });
    }

    #[test]
    fn cold_prefill_gate_round_trips_clamps_and_resolves() {
        with_temp_settings(|| {
            // Default = 360.0 (unset).
            assert!((load_cold_prefill_gate() - 360.0).abs() < 1e-9);
            // Save a specific value and read it back.
            save_cold_prefill_gate(120.0).unwrap();
            assert!((load_cold_prefill_gate() - 120.0).abs() < 1e-9);
            // Out-of-range values clamp to [1, 3600].
            save_cold_prefill_gate(0.0).unwrap();
            assert!((load_cold_prefill_gate() - 1.0).abs() < 1e-9);
            save_cold_prefill_gate(9999.0).unwrap();
            assert!((load_cold_prefill_gate() - 3600.0).abs() < 1e-9);
            // resolve_policy applies the override to all profiles.
            save_cold_prefill_gate(120.0).unwrap();
            assert!((resolve_policy(Profile::SaveTokens).cold_prefill_gate_secs - 120.0).abs() < 1e-9);
            assert!((resolve_policy(Profile::Balanced).cold_prefill_gate_secs - 120.0).abs() < 1e-9);
            assert!((resolve_policy(Profile::MaxQuality).cold_prefill_gate_secs - 120.0).abs() < 1e-9);
            assert!((resolve_policy(Profile::LocalOnly).cold_prefill_gate_secs - 120.0).abs() < 1e-9);
        });
    }

    fn with_temp_settings<F: FnOnce()>(f: F) {
        let _guard = SETTINGS_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::env::set_var("LOCALLLM_SETTINGS", &path);
        f();
        std::env::remove_var("LOCALLLM_SETTINGS");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_then_load_round_trips() {
        with_temp_settings(|| {
            save_profile(Profile::Balanced).unwrap();
            assert_eq!(load_profile(), Profile::Balanced);
        });
    }

    #[test]
    fn load_missing_file_returns_default() {
        with_temp_settings(|| {
            // no save → file does not exist
            assert_eq!(load_profile(), Profile::default());
        });
    }

    #[test]
    fn load_corrupt_file_returns_default() {
        with_temp_settings(|| {
            std::fs::write(settings_path().unwrap(), b"not json {{{").unwrap();
            assert_eq!(load_profile(), Profile::default());
        });
    }

    #[test]
    fn save_overwrites_previous() {
        with_temp_settings(|| {
            save_profile(Profile::MaxQuality).unwrap();
            save_profile(Profile::LocalOnly).unwrap();
            assert_eq!(load_profile(), Profile::LocalOnly);
        });
    }

    #[test]
    fn resolve_prefers_cli_then_saved_then_default() {
        with_temp_settings(|| {
            // nothing saved, no CLI → default
            assert_eq!(resolve_profile(None), Profile::default());
            // saved setting, no CLI → saved
            save_profile(Profile::MaxQuality).unwrap();
            assert_eq!(resolve_profile(None), Profile::MaxQuality);
            // CLI overrides saved
            assert_eq!(
                resolve_profile(Some(Profile::LocalOnly)),
                Profile::LocalOnly
            );
        });
    }

    #[test]
    fn model_ctx_round_trip() {
        with_temp_settings(|| {
            let k = model_ctx_key("bartowski/phi-4-GGUF", "phi-4-Q4_K_M.gguf");
            assert_eq!(k, "bartowski/phi-4-GGUF/phi-4-Q4_K_M.gguf");
            assert_eq!(load_model_ctx(&k), None);
            save_model_ctx(&k, 16384).unwrap();
            assert_eq!(load_model_ctx(&k), Some(16384));
        });
    }

    #[test]
    fn saving_model_ctx_preserves_profile() {
        with_temp_settings(|| {
            save_profile(Profile::MaxQuality).unwrap();
            save_model_ctx("r/f", 8192).unwrap();
            assert_eq!(load_profile(), Profile::MaxQuality);
            assert_eq!(load_model_ctx("r/f"), Some(8192));
        });
    }

    #[test]
    fn clear_model_ctx_removes_only_that_key() {
        with_temp_settings(|| {
            save_model_ctx("a/1", 4096).unwrap();
            save_model_ctx("b/2", 8192).unwrap();
            clear_model_ctx("a/1").unwrap();
            assert_eq!(load_model_ctx("a/1"), None);
            assert_eq!(load_model_ctx("b/2"), Some(8192));
        });
    }

    #[test]
    fn model_profile_round_trips() {
        with_temp_settings(|| {
            let k = model_ctx_key("r", "f");
            assert_eq!(load_model_profile(&k), ExecProfile::default());
            let p = ExecProfile {
                ctx: Some(8192),
                kv_type: Some(crate::config::KvType::Q4),
                gpu_layers: Some(20),
                history_turns: Some(3),
                quant: None,
            };
            save_model_profile(&k, &p).unwrap();
            assert_eq!(load_model_profile(&k), p);
        });
    }

    #[test]
    fn legacy_model_ctx_migrates_into_profile() {
        with_temp_settings(|| {
            // Write an old-shape settings file that only has model_ctx.
            let path = settings_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, r#"{"model_ctx":{"r/f":16384}}"#).unwrap();
            // Reading the profile should surface the migrated ctx.
            assert_eq!(load_model_profile("r/f").ctx, Some(16384));
            // And the ctx back-compat wrapper still sees it.
            assert_eq!(load_model_ctx("r/f"), Some(16384));
        });
    }

    #[test]
    fn ctx_wrapper_writes_through_to_profile() {
        with_temp_settings(|| {
            save_model_ctx("r/f", 4096).unwrap();
            assert_eq!(load_model_profile("r/f").ctx, Some(4096));
            clear_model_ctx("r/f").unwrap();
            assert_eq!(load_model_profile("r/f").ctx, None);
        });
    }

    #[test]
    fn exec_profile_quant_round_trips() {
        with_temp_settings(|| {
            let k = model_ctx_key("r", "f");
            let p = ExecProfile {
                quant: Some("Q5_K_M".into()),
                ..Default::default()
            };
            save_model_profile(&k, &p).unwrap();
            assert_eq!(load_model_profile(&k).quant, Some("Q5_K_M".to_string()));
        });
    }

    #[test]
    fn tool_filter_round_trips_and_clears() {
        with_temp_settings(|| {
            assert!(load_tool_filter("anthropic").is_empty());
            save_tool_filter("anthropic", &["Read".to_string(), "Glob".to_string()]).unwrap();
            assert_eq!(
                load_tool_filter("anthropic"),
                vec!["Read".to_string(), "Glob".to_string()]
            );
            // other surface unaffected
            assert!(load_tool_filter("openai").is_empty());
            // empty list clears
            save_tool_filter("anthropic", &[]).unwrap();
            assert!(load_tool_filter("anthropic").is_empty());
        });
    }

    #[test]
    fn saving_tool_filter_preserves_profile() {
        with_temp_settings(|| {
            save_profile(Profile::MaxQuality).unwrap();
            save_tool_filter("openai", &["Foo".to_string()]).unwrap();
            assert_eq!(load_profile(), Profile::MaxQuality);
            assert_eq!(load_tool_filter("openai"), vec!["Foo".to_string()]);
        });
    }

    #[test]
    fn active_model_round_trips() {
        with_temp_settings(|| {
            assert!(load_active_model().is_none());
            save_active_model("Qwen/Qwen2.5-7B-Instruct-GGUF", "q7.gguf", Some("Q4_K_M")).unwrap();
            let a = load_active_model().unwrap();
            assert_eq!(a.repo, "Qwen/Qwen2.5-7B-Instruct-GGUF");
            assert_eq!(a.file, "q7.gguf");
            assert_eq!(a.quant.as_deref(), Some("Q4_K_M"));
        });
    }

    #[test]
    fn resolve_active_model_prefers_explicit_cli() {
        // CLI differs from default → CLI wins even if a model is saved.
        let saved = Some(ActiveModel {
            repo: "saved/repo".into(),
            file: "s.gguf".into(),
            quant: None,
        });
        let (m, f) = resolve_active_model(
            "cli/repo",
            &["c.gguf".to_string()],
            "default/repo",
            "d.gguf",
            saved,
        );
        assert_eq!(m, "cli/repo");
        assert_eq!(f, vec!["c.gguf".to_string()]);
    }

    #[test]
    fn resolve_active_model_uses_saved_when_cli_is_default() {
        let saved = Some(ActiveModel {
            repo: "saved/repo".into(),
            file: "s.gguf".into(),
            quant: None,
        });
        let (m, f) = resolve_active_model(
            "default/repo",
            &["d.gguf".to_string()],
            "default/repo",
            "d.gguf",
            saved,
        );
        assert_eq!(m, "saved/repo");
        assert_eq!(f, vec!["s.gguf".to_string()]);
    }

    #[test]
    fn resolve_active_model_falls_back_to_default_when_nothing_saved() {
        let (m, f) = resolve_active_model(
            "default/repo",
            &["d.gguf".to_string()],
            "default/repo",
            "d.gguf",
            None,
        );
        assert_eq!(m, "default/repo");
        assert_eq!(f, vec!["d.gguf".to_string()]);
    }
}
