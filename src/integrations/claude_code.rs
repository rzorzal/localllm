//! Claude Code injector: edits `<home>/.claude/settings.json` (JSON).

use std::path::PathBuf;

use serde_json::{json, Map, Value};

use super::{atomic_write, ClientInjector, ClientPrior};

const BASE_URL_KEY: &str = "env.ANTHROPIC_BASE_URL";
const TOOL_SEARCH_KEY: &str = "env.ENABLE_TOOL_SEARCH";

pub struct ClaudeCode {
    base: PathBuf,
}

impl ClaudeCode {
    /// Construct against an explicit home/base dir (tests pass a temp dir).
    pub fn with_base(base: PathBuf) -> Self {
        Self { base }
    }
    /// Construct against the real home dir.
    pub fn default_home() -> Self {
        Self {
            base: dirs::home_dir().unwrap_or_default(),
        }
    }
    fn path(&self) -> PathBuf {
        self.base.join(".claude").join("settings.json")
    }
}

impl Default for ClaudeCode {
    fn default() -> Self {
        Self::default_home()
    }
}

/// Read the settings file into a JSON object, or an empty object if absent.
/// Errors if the file exists but is not a JSON object.
fn read_object(path: &std::path::Path) -> anyhow::Result<Map<String, Value>> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| anyhow::anyhow!("malformed {}: {e}", path.display()))?;
            match v {
                Value::Object(m) => Ok(m),
                _ => anyhow::bail!("{} is not a JSON object", path.display()),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Map::new()),
        Err(e) => Err(e.into()),
    }
}

impl ClientInjector for ClaudeCode {
    fn id(&self) -> &'static str {
        "claude-code"
    }
    fn display_name(&self) -> &'static str {
        "Claude Code"
    }
    fn detect(&self) -> bool {
        self.base.join(".claude").is_dir()
    }

    fn enable(&self, port: u16) -> anyhow::Result<ClientPrior> {
        let path = self.path();
        let mut root = read_object(&path)?;
        let env = root
            .entry("env")
            .or_insert_with(|| Value::Object(Map::new()));
        let env = env
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("env is not an object in {}", path.display()))?;

        let mut prior = ClientPrior::default();
        prior.keys.insert(
            BASE_URL_KEY.to_string(),
            env.get("ANTHROPIC_BASE_URL").cloned(),
        );
        prior.keys.insert(
            TOOL_SEARCH_KEY.to_string(),
            env.get("ENABLE_TOOL_SEARCH").cloned(),
        );

        env.insert(
            "ANTHROPIC_BASE_URL".to_string(),
            json!(format!("http://127.0.0.1:{port}")),
        );
        env.insert("ENABLE_TOOL_SEARCH".to_string(), json!("1"));

        let text = serde_json::to_string_pretty(&Value::Object(root))?;
        atomic_write(&path, text.as_bytes())?;
        Ok(prior)
    }

    fn disable(&self, prior: &ClientPrior) -> anyhow::Result<()> {
        let path = self.path();
        let mut root = read_object(&path)?;
        if root.is_empty() {
            return Ok(());
        }
        if let Some(env) = root.get_mut("env").and_then(|v| v.as_object_mut()) {
            for (full_key, name) in [
                (BASE_URL_KEY, "ANTHROPIC_BASE_URL"),
                (TOOL_SEARCH_KEY, "ENABLE_TOOL_SEARCH"),
            ] {
                match prior.keys.get(full_key) {
                    Some(Some(v)) => {
                        env.insert(name.to_string(), v.clone());
                    }
                    Some(None) | None => {
                        env.remove(name);
                    }
                }
            }
            // Tidy up an env object we created and emptied.
            if env.is_empty() {
                root.remove("env");
            }
        }
        let text = serde_json::to_string_pretty(&Value::Object(root))?;
        atomic_write(&path, text.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home() -> PathBuf {
        let d = std::env::temp_dir().join(format!("llm-cc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join(".claude")).unwrap();
        d
    }

    fn read(path: &std::path::Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn enable_on_absent_file_writes_keys_and_records_absent_prior() {
        let home = temp_home();
        let cc = ClaudeCode::with_base(home.clone());
        let prior = cc.enable(31415).unwrap();
        let v = read(&cc.path());
        assert_eq!(v["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:31415");
        assert_eq!(v["env"]["ENABLE_TOOL_SEARCH"], "1");
        assert_eq!(prior.keys[BASE_URL_KEY], None);
        assert_eq!(prior.keys[TOOL_SEARCH_KEY], None);
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn enable_records_preexisting_value_and_disable_restores_it() {
        let home = temp_home();
        let cc = ClaudeCode::with_base(home.clone());
        std::fs::write(
            cc.path(),
            r#"{"env":{"ANTHROPIC_BASE_URL":"https://api.anthropic.com","FOO":"bar"}}"#,
        )
        .unwrap();
        let prior = cc.enable(31415).unwrap();
        assert_eq!(
            prior.keys[BASE_URL_KEY],
            Some(json!("https://api.anthropic.com"))
        );
        assert_eq!(prior.keys[TOOL_SEARCH_KEY], None);
        // our keys are live now
        assert_eq!(
            read(&cc.path())["env"]["ANTHROPIC_BASE_URL"],
            "http://127.0.0.1:31415"
        );

        cc.disable(&prior).unwrap();
        let v = read(&cc.path());
        // restored prior value, our tool-search key removed, unrelated key preserved
        assert_eq!(v["env"]["ANTHROPIC_BASE_URL"], "https://api.anthropic.com");
        assert!(v["env"].get("ENABLE_TOOL_SEARCH").is_none());
        assert_eq!(v["env"]["FOO"], "bar");
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn disable_removes_added_keys_when_prior_absent() {
        let home = temp_home();
        let cc = ClaudeCode::with_base(home.clone());
        let prior = cc.enable(31415).unwrap();
        cc.disable(&prior).unwrap();
        let v = read(&cc.path());
        // env was created by us and is now empty → removed entirely
        assert!(v.get("env").is_none());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn malformed_json_errors_not_panics() {
        let home = temp_home();
        let cc = ClaudeCode::with_base(home.clone());
        std::fs::write(cc.path(), b"not json {{{").unwrap();
        assert!(cc.enable(31415).is_err());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn detect_true_only_when_dir_present() {
        let home = temp_home();
        assert!(ClaudeCode::with_base(home.clone()).detect());
        let empty = std::env::temp_dir().join(format!("llm-none-{}", uuid::Uuid::new_v4()));
        assert!(!ClaudeCode::with_base(empty).detect());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn disable_on_absent_file_creates_no_file() {
        let home = temp_home();
        let cc = ClaudeCode::with_base(home.clone());
        let prior = ClientPrior::default();
        cc.disable(&prior).unwrap();
        assert!(!cc.path().exists());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn non_object_json_errors() {
        let home = temp_home();
        let cc = ClaudeCode::with_base(home.clone());
        std::fs::write(cc.path(), b"[]").unwrap();
        assert!(cc.enable(31415).is_err());
        std::fs::remove_dir_all(&home).ok();
    }
}
