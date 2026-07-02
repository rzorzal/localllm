//! Persisted user settings: routing profile and client-integration toggle state.
//!
//! Stored as JSON at `<config-dir>/localllm/settings.json` (e.g.
//! `~/Library/Application Support/localllm/settings.json` on macOS). The path
//! is overridable via the `LOCALLLM_SETTINGS` env var (full file path), used by
//! tests and power users.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::integrations::ClientPrior;
use crate::route::Profile;

/// Persisted toggle state for client integrations.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IntegrationState {
    pub enabled: bool,
    pub priors: BTreeMap<String, ClientPrior>,
}

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
}

/// On-disk settings shape. New fields must be `#[serde(default)]` so older
/// files (which only had `profile`) still load.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    profile: Profile,
    #[serde(default)]
    integrations: IntegrationState,
    /// Legacy per-model ctx map. Read-only for migration; new writes go to
    /// `model_profiles`. Kept so old files still load.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    model_ctx: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    model_profiles: std::collections::BTreeMap<String, ExecProfile>,
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
        s.model_profiles.entry(k).or_default().ctx.get_or_insert(ctx);
    }
    s
}

/// Persist the full settings object, creating the parent directory if needed.
fn save_settings(s: &Settings) -> anyhow::Result<()> {
    let path = settings_path()
        .ok_or_else(|| anyhow::anyhow!("no settings path (no config dir)"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(s)?;
    crate::integrations::atomic_write(&path, json.as_bytes())?;
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

/// Load the persisted client-integration toggle state.
pub fn load_integrations() -> IntegrationState {
    load_settings().integrations
}

/// Persist the client-integration toggle state, preserving the profile.
pub fn save_integrations(state: &IntegrationState) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.integrations = state.clone();
    save_settings(&s)
}

/// Settings key for a model's per-model ctx override: `"{repo}/{file}"`.
pub fn model_ctx_key(repo: &str, file: &str) -> String {
    format!("{repo}/{file}")
}

/// Load a model's full execution profile, or defaults if unset.
pub fn load_model_profile(key: &str) -> ExecProfile {
    load_settings().model_profiles.get(key).cloned().unwrap_or_default()
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

#[cfg(test)]
mod tests {
    use super::*;

    // Mutex to serialise tests that mutate the LOCALLLM_SETTINGS env var.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_temp_settings<F: FnOnce()>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
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
            assert_eq!(resolve_profile(Some(Profile::LocalOnly)), Profile::LocalOnly);
        });
    }

    #[test]
    fn integrations_round_trip() {
        with_temp_settings(|| {
            use crate::integrations::ClientPrior;
            let mut priors = std::collections::BTreeMap::new();
            let mut keys = std::collections::BTreeMap::new();
            keys.insert("env.ANTHROPIC_BASE_URL".to_string(), None);
            priors.insert("claude-code".to_string(), ClientPrior { keys });
            let state = IntegrationState { enabled: true, priors };
            save_integrations(&state).unwrap();
            assert_eq!(load_integrations(), state);
        });
    }

    #[test]
    fn saving_profile_preserves_integrations() {
        with_temp_settings(|| {
            let state = IntegrationState { enabled: true, priors: Default::default() };
            save_integrations(&state).unwrap();
            save_profile(Profile::MaxQuality).unwrap();
            // profile saved, integrations untouched
            assert_eq!(load_profile(), Profile::MaxQuality);
            assert!(load_integrations().enabled);
        });
    }

    #[test]
    fn saving_integrations_preserves_profile() {
        with_temp_settings(|| {
            save_profile(Profile::LocalOnly).unwrap();
            let state = IntegrationState { enabled: true, priors: Default::default() };
            save_integrations(&state).unwrap();
            assert_eq!(load_profile(), Profile::LocalOnly);
        });
    }

    #[test]
    fn load_integrations_missing_returns_default() {
        with_temp_settings(|| {
            assert_eq!(load_integrations(), IntegrationState::default());
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
    fn saving_model_ctx_preserves_profile_and_integrations() {
        with_temp_settings(|| {
            save_profile(Profile::MaxQuality).unwrap();
            let state = IntegrationState { enabled: true, priors: Default::default() };
            save_integrations(&state).unwrap();
            save_model_ctx("r/f", 8192).unwrap();
            assert_eq!(load_profile(), Profile::MaxQuality);
            assert!(load_integrations().enabled);
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
}
