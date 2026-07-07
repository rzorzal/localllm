//! Codex injector: edits `<home>/.codex/config.toml` (TOML), format-preserving.

use std::path::PathBuf;

#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use toml_edit::{value, DocumentMut, Item, Table};

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
        Self {
            base: dirs::home_dir().unwrap_or_default(),
        }
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

        // Build explicit standard table for model_providers.localllm, inserting
        // into the existing parent table so sibling providers are preserved.
        let mut localllm = Table::new();
        localllm["name"] = value("localllm");
        localllm["base_url"] = value(format!("http://127.0.0.1:{port}/v1"));
        localllm["wire_api"] = value("responses");
        localllm["env_key"] = value("OPENAI_API_KEY");
        let providers = doc
            .entry("model_providers")
            .or_insert(Item::Table(Table::new()))
            .as_table_mut()
            .ok_or_else(|| {
                anyhow::anyhow!("model_providers is not a table in {}", path.display())
            })?;
        providers.insert("localllm", Item::Table(localllm));

        atomic_write(&path, doc.to_string().as_bytes())?;
        Ok(prior)
    }

    fn disable(&self, prior: &ClientPrior) -> anyhow::Result<()> {
        let path = self.path();
        let mut doc = read_doc(&path)?;
        if doc.is_empty() {
            return Ok(());
        }

        match prior.keys.get(PROVIDER_KEY) {
            Some(Some(Value::String(s))) => doc[PROVIDER_KEY] = value(s.clone()),
            _ => {
                doc.remove(PROVIDER_KEY);
            }
        }

        // Remove our provider table; drop the parent table if it became empty.
        if let Some(providers) = doc.get_mut("model_providers").and_then(Item::as_table_mut) {
            providers.remove("localllm");
            if providers.is_empty() {
                doc.remove("model_providers");
            }
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
        let text = std::fs::read_to_string(cx.path()).unwrap();
        assert!(
            text.contains("[model_providers.localllm]"),
            "expected standard table, got:\n{text}"
        );
        let doc = text.parse::<DocumentMut>().unwrap();
        assert_eq!(doc["model_provider"].as_str(), Some("localllm"));
        let p = &doc["model_providers"]["localllm"];
        assert_eq!(p["name"].as_str(), Some("localllm"));
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
        let doc = std::fs::read_to_string(cx.path())
            .unwrap()
            .parse::<DocumentMut>()
            .unwrap();
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
        let doc = std::fs::read_to_string(cx.path())
            .unwrap()
            .parse::<DocumentMut>()
            .unwrap();
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
    fn disable_with_malformed_toml_errors() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        std::fs::write(cx.path(), b"this is = = not toml").unwrap();
        let prior = ClientPrior::default();
        assert!(cx.disable(&prior).is_err());
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn enable_preserves_sibling_providers_and_disable_restores() {
        let home = temp_home();
        let cx = Codex::with_base(home.clone());
        // Config already has model_provider = "openai" and a [model_providers.openai] block.
        std::fs::write(
            cx.path(),
            "model_provider = \"openai\"\n\n[model_providers.openai]\nname = \"openai\"\nbase_url = \"https://api.openai.com/v1\"\n",
        )
        .unwrap();
        let prior = cx.enable(31415).unwrap();
        let text = std::fs::read_to_string(cx.path()).unwrap();
        // Our block is present as a standard table header.
        assert!(
            text.contains("[model_providers.localllm]"),
            "localllm block missing:\n{text}"
        );
        // The sibling openai block and its field must survive.
        assert!(
            text.contains("[model_providers.openai]"),
            "openai block clobbered:\n{text}"
        );
        assert!(
            text.contains("base_url = \"https://api.openai.com/v1\""),
            "openai field clobbered:\n{text}"
        );
        assert_eq!(
            prior.keys["model_provider"],
            Some(serde_json::json!("openai"))
        );

        cx.disable(&prior).unwrap();
        let doc = std::fs::read_to_string(cx.path())
            .unwrap()
            .parse::<DocumentMut>()
            .unwrap();
        // model_provider restored to "openai".
        assert_eq!(doc["model_provider"].as_str(), Some("openai"));
        // localllm entry gone.
        assert!(
            doc["model_providers"].get("localllm").is_none(),
            "localllm not removed after disable"
        );
        // openai sibling still present with its field.
        assert!(
            doc["model_providers"].get("openai").is_some(),
            "openai sibling removed after disable"
        );
        assert_eq!(
            doc["model_providers"]["openai"]["base_url"].as_str(),
            Some("https://api.openai.com/v1"),
            "openai field lost after disable"
        );
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
