//! Codex injector: edits `<home>/.codex/config.toml` (TOML), format-preserving.

use std::path::PathBuf;

use serde_json::{json, Value};
use toml_edit::{value, DocumentMut};

use super::{atomic_write, ClientInjector, ClientPrior};

const PROVIDER_KEY: &str = "model_provider";

pub struct Codex {
    base: PathBuf,
}

impl Codex {
    pub fn with_base(base: PathBuf) -> Self {
        Self { base }
    }
    pub fn default_home() -> Self {
        Self { base: dirs::home_dir().unwrap_or_default() }
    }
    fn path(&self) -> PathBuf {
        self.base.join(".codex").join("config.toml")
    }
}

impl Default for Codex {
    fn default() -> Self {
        Self::default_home()
    }
}

/// Parse the config file into a TOML document, or an empty document if absent.
/// Errors if the file exists but is not valid TOML.
fn read_doc(path: &std::path::Path) -> anyhow::Result<DocumentMut> {
    match std::fs::read_to_string(path) {
        Ok(text) => text
            .parse::<DocumentMut>()
            .map_err(|e| anyhow::anyhow!("malformed {}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DocumentMut::new()),
        Err(e) => Err(e.into()),
    }
}

impl ClientInjector for Codex {
    fn id(&self) -> &'static str {
        "codex"
    }
    fn display_name(&self) -> &'static str {
        "Codex"
    }
    fn detect(&self) -> bool {
        self.base.join(".codex").is_dir()
    }

    fn enable(&self, port: u16) -> anyhow::Result<ClientPrior> {
        let path = self.path();
        let mut doc = read_doc(&path)?;

        // Record prior model_provider (string, if any).
        let prior_provider = doc
            .get(PROVIDER_KEY)
            .and_then(|i| i.as_str())
            .map(|s| Value::String(s.to_string()));
        let mut prior = ClientPrior::default();
        prior.keys.insert(PROVIDER_KEY.to_string(), prior_provider);

        doc[PROVIDER_KEY] = value("localllm");
        let p = &mut doc["model_providers"]["localllm"];
        p["name"] = value("localllm");
        p["base_url"] = value(format!("http://127.0.0.1:{port}/v1"));
        p["wire_api"] = value("responses");
        p["env_key"] = value("OPENAI_API_KEY");

        atomic_write(&path, doc.to_string().as_bytes())?;
        Ok(prior)
    }

    fn disable(&self, prior: &ClientPrior) -> anyhow::Result<()> {
        let path = self.path();
        let mut doc = read_doc(&path)?;
        if doc.to_string().trim().is_empty() {
            return Ok(());
        }

        match prior.keys.get(PROVIDER_KEY) {
            Some(Some(Value::String(s))) => doc[PROVIDER_KEY] = value(s.clone()),
            _ => {
                doc.remove(PROVIDER_KEY);
            }
        }

        // Remove our provider table; drop the parent table if it became empty.
        // Handle both regular tables and inline tables (which result from serialization).
        let should_remove_providers = {
            if let Some(item) = doc.get_mut("model_providers") {
                let is_empty = if let Some(providers) = item.as_table_mut() {
                    // Regular table
                    providers.remove("localllm");
                    providers.is_empty()
                } else if let Some(providers) = item.as_inline_table_mut() {
                    // Inline table (from serialization)
                    providers.remove("localllm");
                    providers.is_empty()
                } else {
                    false
                };
                is_empty
            } else {
                false
            }
        };
        if should_remove_providers {
            doc.remove("model_providers");
        }

        atomic_write(&path, doc.to_string().as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home() -> PathBuf {
        let d = std::env::temp_dir().join(format!("llm-cx-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join(".codex")).unwrap();
        d
    }

    #[test]
    fn enable_adds_provider_block_and_sets_selector() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        let prior = cx.enable(31415).unwrap();
        let doc = std::fs::read_to_string(cx.path()).unwrap().parse::<DocumentMut>().unwrap();
        assert_eq!(doc["model_provider"].as_str(), Some("localllm"));
        let p = &doc["model_providers"]["localllm"];
        assert_eq!(p["base_url"].as_str(), Some("http://127.0.0.1:31415/v1"));
        assert_eq!(p["wire_api"].as_str(), Some("responses"));
        assert_eq!(p["env_key"].as_str(), Some("OPENAI_API_KEY"));
        assert_eq!(prior.keys["model_provider"], None);
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn enable_preserves_model_and_comments_then_disable_reverts() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        std::fs::write(
            cx.path(),
            "# my config\nmodel = \"gpt-5-codex\"\nmodel_provider = \"openai\"\n",
        )
        .unwrap();
        let prior = cx.enable(31415).unwrap();
        assert_eq!(prior.keys["model_provider"], Some(json!("openai")));
        let text = std::fs::read_to_string(cx.path()).unwrap();
        // model untouched, comment preserved, provider switched
        assert!(text.contains("model = \"gpt-5-codex\""));
        assert!(text.contains("# my config"));
        assert!(text.contains("model_provider = \"localllm\""));

        cx.disable(&prior).unwrap();
        let doc = std::fs::read_to_string(cx.path()).unwrap().parse::<DocumentMut>().unwrap();
        // provider restored, our table gone, model still present
        assert_eq!(doc["model_provider"].as_str(), Some("openai"));
        assert!(doc.get("model_providers").is_none());
        assert_eq!(doc["model"].as_str(), Some("gpt-5-codex"));
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn disable_with_absent_prior_removes_selector() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        let prior = cx.enable(31415).unwrap(); // file had no model_provider
        cx.disable(&prior).unwrap();
        let doc = std::fs::read_to_string(cx.path()).unwrap().parse::<DocumentMut>().unwrap();
        assert!(doc.get("model_provider").is_none());
        assert!(doc.get("model_providers").is_none());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn malformed_toml_errors_not_panics() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        std::fs::write(cx.path(), b"this is = = not toml").unwrap();
        assert!(cx.enable(31415).is_err());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn disable_absent_file_is_noop_creates_no_file() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        // config.toml does not exist
        assert!(!cx.path().exists());

        // Create a dummy prior (empty keys)
        let prior = ClientPrior::default();

        // disable should be a no-op
        cx.disable(&prior).unwrap();

        // file should still not exist
        assert!(!cx.path().exists());
        std::fs::remove_dir_all(&home).ok();
    }
}
