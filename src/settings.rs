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

/// On-disk settings shape. New fields must be `#[serde(default)]` so older
/// files (which only had `profile`) still load.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    profile: Profile,
    #[serde(default)]
    integrations: IntegrationState,
    #[serde(default)]
    model_ctx: std::collections::BTreeMap<String, u32>,
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
    let Some(path) = settings_path() else {
        return Settings::default();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Settings::default();
    };
    serde_json::from_str::<Settings>(&text).unwrap_or_default()
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

/// Load a model's persisted ctx override, or `None` if unset.
pub fn load_model_ctx(key: &str) -> Option<u32> {
    load_settings().model_ctx.get(key).copied()
}

/// Persist a model's ctx override, preserving the rest of settings.
pub fn save_model_ctx(key: &str, ctx: u32) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.model_ctx.insert(key.to_string(), ctx);
    save_settings(&s)
}

/// Remove a model's ctx override, preserving the rest of settings.
pub fn clear_model_ctx(key: &str) -> anyhow::Result<()> {
    let mut s = load_settings();
    s.model_ctx.remove(key);
    save_settings(&s)
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
}
