# Tray Client-Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A checkable tray item that injects (and surgically reverts) localllm endpoint config into Claude Code and Codex, so they auto-route through the local server.

**Architecture:** A `ClientInjector` trait with two impls (Claude Code = JSON via serde_json; Codex = TOML via toml_edit). Each `enable(port)` records the prior value of every key it touches and writes ours atomically; `disable(prior)` restores prior-or-deletes. Orchestration enables/disables all detected clients and returns the recorded priors; `settings.rs` persists the toggle state + priors; `tray.rs` drives it from a `CheckMenuItem`.

**Tech Stack:** Rust, serde_json, toml_edit, dirs, anyhow, uuid (tests), tray-icon.

## Global Constraints

- Server bind/port: clients point at `http://127.0.0.1:<port>`; default port `31415` (from `config.rs`). Codex `base_url` is `http://127.0.0.1:<port>/v1`.
- Claude Code file: `<home>/.claude/settings.json` (JSON object). Keys written under top-level `env`: `ANTHROPIC_BASE_URL = "http://127.0.0.1:<port>"`, `ENABLE_TOOL_SEARCH = "1"`.
- Codex file: `<home>/.codex/config.toml`. Set top-level `model_provider = "localllm"`; add table `[model_providers.localllm]` with `name = "localllm"`, `base_url = "http://127.0.0.1:<port>/v1"`, `wire_api = "responses"`, `env_key = "OPENAI_API_KEY"`. **Never** touch top-level `model`.
- Revert is **surgical**: record each touched key's prior (None = was absent; Some(v) = had value v); on disable restore v or delete. Never mutate unrelated keys.
- All writes **atomic**: temp file in the same directory + rename; preserve the original file's permissions when it existed; no temp file left on the success path.
- Config paths are **injectable** (constructor takes a base/home dir) so tests use temp dirs — never the real home.
- Errors are `anyhow::Result`; a malformed existing config returns `Err`, never panics.
- No new build warnings; `cargo clippy` clean in changed code. Build/test command prefix: `MISTRALRS_METAL_PRECOMPILE=0`.

---

### Task 1: integrations module — shared types + atomic write

**Files:**
- Create: `src/integrations/mod.rs`
- Modify: `src/lib.rs:14` (add `pub mod integrations;` after `pub mod tray;`)
- Modify: `Cargo.toml:25` (add `toml_edit = "0.22"` near `dirs`)
- Test: inline `#[cfg(test)]` in `src/integrations/mod.rs`

**Interfaces:**
- Produces:
  - `pub trait ClientInjector { fn id(&self) -> &'static str; fn display_name(&self) -> &'static str; fn detect(&self) -> bool; fn enable(&self, port: u16) -> anyhow::Result<ClientPrior>; fn disable(&self, prior: &ClientPrior) -> anyhow::Result<()>; }`
  - `pub struct ClientPrior { pub keys: std::collections::BTreeMap<String, Option<serde_json::Value>> }` — derives `Debug, Clone, Default, PartialEq, Serialize, Deserialize`.
  - `pub fn atomic_write(path: &std::path::Path, contents: &[u8]) -> anyhow::Result<()>`

- [ ] **Step 1: Add the dependency**

In `Cargo.toml`, under the existing deps (near `dirs = "5"`), add:

```toml
toml_edit = "0.22"
```

- [ ] **Step 2: Register the module**

In `src/lib.rs`, after `pub mod tray;` add:

```rust
pub mod integrations;
```

- [ ] **Step 3: Write the failing tests**

Create `src/integrations/mod.rs` with the types and tests (implementation bodies stubbed to `todo!()` for `atomic_write`):

```rust
//! Pluggable client-config injection: point external CLIs (Claude Code, Codex)
//! at the local server, and surgically revert. Each injector records the prior
//! value of every key it writes so disable restores exactly what was there.

pub mod claude_code;
pub mod codex;

use std::collections::BTreeMap;
use std::path::Path;

/// Records, per key an injector touches, what was there before we wrote ours:
/// `None` = the key was absent; `Some(v)` = it held `v`. Stored verbatim in
/// `settings.rs` so a later `disable` can restore or delete each key.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClientPrior {
    pub keys: BTreeMap<String, Option<serde_json::Value>>,
}

/// A target CLI whose config we can point at localllm and later revert.
pub trait ClientInjector {
    /// Stable id used as the settings key (e.g. "claude-code").
    fn id(&self) -> &'static str;
    /// Human label for the tray "Wired:" line (e.g. "Claude Code").
    fn display_name(&self) -> &'static str;
    /// True when this client's config dir/file is present on disk.
    fn detect(&self) -> bool;
    /// Record prior values and write our keys. Returns the recorded prior.
    fn enable(&self, port: u16) -> anyhow::Result<ClientPrior>;
    /// Restore the prior values (delete keys that were absent).
    fn disable(&self, prior: &ClientPrior) -> anyhow::Result<()>;
}

/// Write `contents` to `path` atomically: a temp file in the same directory is
/// written, fsync-flushed, given the original file's permissions (when it
/// existed), then renamed over the target. No temp file remains on success.
pub fn atomic_write(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_prior_round_trips_through_json() {
        let mut keys = BTreeMap::new();
        keys.insert("env.ANTHROPIC_BASE_URL".to_string(), None);
        keys.insert(
            "model_provider".to_string(),
            Some(serde_json::Value::String("openai".into())),
        );
        let prior = ClientPrior { keys };
        let text = serde_json::to_string(&prior).unwrap();
        let back: ClientPrior = serde_json::from_str(&text).unwrap();
        assert_eq!(prior, back);
    }

    #[test]
    fn atomic_write_creates_file_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!("llm-aw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("config.json");
        atomic_write(&target, b"hello").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello");
        // no sibling temp files left behind
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name() != "config.json")
            .collect();
        assert!(leftovers.is_empty(), "temp file left: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn atomic_write_overwrites_existing() {
        let dir = std::env::temp_dir().join(format!("llm-aw-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("config.json");
        std::fs::write(&target, b"old").unwrap();
        atomic_write(&target, b"new").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        std::fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib integrations::tests`
Expected: the two `atomic_write` tests panic with `not yet implemented` (todo!), confirming they exercise the real function. (`client_prior_round_trips` passes already.)

Note: `claude_code` and `codex` submodules are declared but not yet created — to compile Task 1 in isolation, also create empty stubs `src/integrations/claude_code.rs` and `src/integrations/codex.rs` each containing only a `//! stub (Task 2/3)` line. Tasks 2 and 3 replace them.

- [ ] **Step 5: Implement `atomic_write`**

Replace the `todo!()` body:

```rust
pub fn atomic_write(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent dir: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".localllm-tmp-{}", uuid::Uuid::new_v4()));
    // Best-effort cleanup if anything below fails.
    let result = (|| -> anyhow::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents)?;
        f.sync_all()?;
        // Preserve the original file's permissions when it existed.
        #[cfg(unix)]
        if let Ok(meta) = std::fs::metadata(path) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(meta.permissions().mode()))?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib integrations::tests`
Expected: PASS (3 tests). Output pristine.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/integrations/
git commit -m "feat(integrations): module scaffold — ClientInjector trait, ClientPrior, atomic_write"
```

---

### Task 2: Claude Code injector (JSON)

**Files:**
- Create (replace stub): `src/integrations/claude_code.rs`
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Consumes: `ClientInjector`, `ClientPrior`, `atomic_write` from Task 1.
- Produces:
  - `pub struct ClaudeCode { base: std::path::PathBuf }`
  - `impl ClaudeCode { pub fn with_base(base: std::path::PathBuf) -> Self; pub fn default_home() -> Self }`
  - `impl Default for ClaudeCode` (uses `dirs::home_dir()`)
  - `impl ClientInjector for ClaudeCode` — id `"claude-code"`, display `"Claude Code"`.
  - Key names recorded in `ClientPrior`: `"env.ANTHROPIC_BASE_URL"`, `"env.ENABLE_TOOL_SEARCH"`.

- [ ] **Step 1: Write the failing tests**

Replace `src/integrations/claude_code.rs` with:

```rust
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
        Self { base: dirs::home_dir().unwrap_or_default() }
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
        prior.keys.insert(BASE_URL_KEY.to_string(), env.get("ANTHROPIC_BASE_URL").cloned());
        prior.keys.insert(TOOL_SEARCH_KEY.to_string(), env.get("ENABLE_TOOL_SEARCH").cloned());

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
        if !path.exists() {
            return Ok(());
        }
        let mut root = read_object(&path)?;
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
        assert_eq!(prior.keys[BASE_URL_KEY], Some(json!("https://api.anthropic.com")));
        assert_eq!(prior.keys[TOOL_SEARCH_KEY], None);
        // our keys are live now
        assert_eq!(read(&cc.path())["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:31415");

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
}
```

- [ ] **Step 2: Run the tests to verify they fail (RED), then pass (GREEN)**

The implementation is written alongside the tests above (this task is small enough that the impl and tests land together). Run:

`MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib integrations::claude_code`
Expected: PASS (5 tests). If you wrote the impl incrementally, first confirm a RED run with the impl bodies replaced by `todo!()`.

- [ ] **Step 3: Commit**

```bash
git add src/integrations/claude_code.rs
git commit -m "feat(integrations): Claude Code JSON injector (surgical enable/disable)"
```

---

### Task 3: Codex injector (TOML)

**Files:**
- Create (replace stub): `src/integrations/codex.rs`
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Consumes: `ClientInjector`, `ClientPrior`, `atomic_write` from Task 1.
- Produces:
  - `pub struct Codex { base: std::path::PathBuf }`
  - `impl Codex { pub fn with_base(base) -> Self; pub fn default_home() -> Self }`, `impl Default`.
  - `impl ClientInjector for Codex` — id `"codex"`, display `"Codex"`.
  - Key recorded in `ClientPrior`: `"model_provider"` (Some(String) or None). The `[model_providers.localllm]` table is owned by localllm and removed on disable.

- [ ] **Step 1: Write impl + failing tests**

Replace `src/integrations/codex.rs` with:

```rust
//! Codex injector: edits `<home>/.codex/config.toml` (TOML), format-preserving.

use std::path::PathBuf;

use serde_json::{json, Value};
use toml_edit::{value, DocumentMut, Item};

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
        if !path.exists() {
            return Ok(());
        }
        let mut doc = read_doc(&path)?;

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
}
```

- [ ] **Step 2: Run the tests**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib integrations::codex`
Expected: PASS (4 tests). Output pristine.

- [ ] **Step 3: Commit**

```bash
git add src/integrations/codex.rs
git commit -m "feat(integrations): Codex TOML injector (format-preserving enable/disable)"
```

---

### Task 4: settings persistence for the toggle state

**Files:**
- Modify: `src/settings.rs` (add `integrations` to `Settings`; refactor to whole-file load/save; add integration helpers)
- Test: extend the inline `#[cfg(test)]` module

**Interfaces:**
- Consumes: `crate::integrations::ClientPrior`.
- Produces:
  - `pub struct IntegrationState { pub enabled: bool, pub priors: std::collections::BTreeMap<String, ClientPrior> }` — derives `Debug, Clone, Default, PartialEq, Serialize, Deserialize`.
  - `pub fn load_integrations() -> IntegrationState`
  - `pub fn save_integrations(state: &IntegrationState) -> anyhow::Result<()>`
  - Existing `load_profile`/`save_profile`/`resolve_profile` keep working and **preserve** the integrations block across a profile save.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/settings.rs`:

```rust
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
```

- [ ] **Step 2: Run to verify failure**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib settings::tests::integrations_round_trip`
Expected: FAIL to compile (`IntegrationState`, `save_integrations`, `load_integrations` not found).

- [ ] **Step 3: Refactor `settings.rs` to whole-file load/save + integration helpers**

Replace the `Settings` struct and the load/save functions (lines ~12–60) with:

```rust
use std::collections::BTreeMap;

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
    std::fs::write(&path, json)?;
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
```

Also remove the now-unused `use std::path::PathBuf;` line only if it becomes unused — `settings_path` still returns `Option<PathBuf>`, so keep it. Keep `Profile` deriving `Default` (it already does).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib settings::`
Expected: PASS (all existing profile tests + 4 new integration tests).

- [ ] **Step 5: Commit**

```bash
git add src/settings.rs
git commit -m "feat(settings): persist client-integration toggle state, preserve profile on save"
```

---

### Task 5: orchestration + tray toggle wiring

**Files:**
- Modify: `src/integrations/mod.rs` (add orchestration: `injectors_default`, `enable_all`, `disable_all`, `EnableOutcome`, `WireSummary`)
- Modify: `src/tray.rs` (add the `CheckMenuItem`, click handler, wired-clients sub-line)
- Test: inline `#[cfg(test)]` in `src/integrations/mod.rs` (orchestration tests with a fake injector)

**Interfaces:**
- Consumes: `ClientInjector`, `ClientPrior` (Task 1); the two injectors (Tasks 2–3); `settings::{IntegrationState, load_integrations, save_integrations}` (Task 4).
- Produces:
  - `pub struct WireSummary { pub wired: Vec<String>, pub failed: Vec<(String, String)> }`
  - `pub struct EnableOutcome { pub priors: std::collections::BTreeMap<String, ClientPrior>, pub summary: WireSummary }`
  - `pub fn injectors_default() -> Vec<Box<dyn ClientInjector>>`
  - `pub fn enable_all(port: u16, injectors: &[Box<dyn ClientInjector>]) -> EnableOutcome`
  - `pub fn disable_all(priors: &std::collections::BTreeMap<String, ClientPrior>, injectors: &[Box<dyn ClientInjector>]) -> WireSummary`

- [ ] **Step 1: Write the failing orchestration tests**

Add to `src/integrations/mod.rs` (above the existing `#[cfg(test)]` block contents, inside `mod tests`):

```rust
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    // A fake injector that records calls and is always "detected" unless told not.
    struct Fake {
        id: &'static str,
        detected: bool,
        fail_enable: bool,
        log: std::rc::Rc<RefCell<Vec<String>>>,
    }
    impl ClientInjector for Fake {
        fn id(&self) -> &'static str { self.id }
        fn display_name(&self) -> &'static str { self.id }
        fn detect(&self) -> bool { self.detected }
        fn enable(&self, port: u16) -> anyhow::Result<ClientPrior> {
            self.log.borrow_mut().push(format!("enable:{}:{port}", self.id));
            if self.fail_enable {
                anyhow::bail!("boom");
            }
            let mut keys = BTreeMap::new();
            keys.insert("k".to_string(), None);
            Ok(ClientPrior { keys })
        }
        fn disable(&self, _prior: &ClientPrior) -> anyhow::Result<()> {
            self.log.borrow_mut().push(format!("disable:{}", self.id));
            Ok(())
        }
    }

    #[test]
    fn enable_all_skips_undetected_and_collects_failures() {
        let log = std::rc::Rc::new(RefCell::new(vec![]));
        let injectors: Vec<Box<dyn ClientInjector>> = vec![
            Box::new(Fake { id: "a", detected: true, fail_enable: false, log: log.clone() }),
            Box::new(Fake { id: "b", detected: false, fail_enable: false, log: log.clone() }),
            Box::new(Fake { id: "c", detected: true, fail_enable: true, log: log.clone() }),
        ];
        let outcome = enable_all(31415, &injectors);
        // a wired with a prior; b skipped; c detected-but-failed
        assert_eq!(outcome.summary.wired, vec!["a".to_string()]);
        assert_eq!(outcome.priors.keys().collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(outcome.summary.failed.len(), 1);
        assert_eq!(outcome.summary.failed[0].0, "c");
        // b's enable was never called
        assert!(!log.borrow().iter().any(|l| l == "enable:b:31415"));
    }

    #[test]
    fn disable_all_calls_disable_for_recorded_priors_only() {
        let log = std::rc::Rc::new(RefCell::new(vec![]));
        let injectors: Vec<Box<dyn ClientInjector>> = vec![
            Box::new(Fake { id: "a", detected: true, fail_enable: false, log: log.clone() }),
            Box::new(Fake { id: "b", detected: true, fail_enable: false, log: log.clone() }),
        ];
        let mut priors = BTreeMap::new();
        priors.insert("a".to_string(), ClientPrior::default());
        let summary = disable_all(&priors, &injectors);
        assert_eq!(summary.wired, vec!["a".to_string()]);
        assert!(log.borrow().iter().any(|l| l == "disable:a"));
        assert!(!log.borrow().iter().any(|l| l == "disable:b"));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib integrations::tests::enable_all_skips`
Expected: FAIL to compile (`enable_all`, `disable_all`, `EnableOutcome`, `WireSummary` not found).

- [ ] **Step 3: Implement orchestration**

Add to `src/integrations/mod.rs` (after `atomic_write`):

```rust
use claude_code::ClaudeCode;
use codex::Codex;

/// Summary of which clients were (un)wired and which failed.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct WireSummary {
    pub wired: Vec<String>,
    pub failed: Vec<(String, String)>,
}

/// Result of enabling: the priors to persist plus a human summary.
#[derive(Debug, Default)]
pub struct EnableOutcome {
    pub priors: BTreeMap<String, ClientPrior>,
    pub summary: WireSummary,
}

/// The built-in injectors, pointed at the real home dir.
pub fn injectors_default() -> Vec<Box<dyn ClientInjector>> {
    vec![Box::new(ClaudeCode::default()), Box::new(Codex::default())]
}

/// Enable every detected client; record each one's prior. Undetected clients
/// are skipped; an injector whose `enable` fails is reported in `failed` and
/// contributes no prior (nothing to revert later).
pub fn enable_all(port: u16, injectors: &[Box<dyn ClientInjector>]) -> EnableOutcome {
    let mut out = EnableOutcome::default();
    for inj in injectors {
        if !inj.detect() {
            continue;
        }
        match inj.enable(port) {
            Ok(prior) => {
                out.priors.insert(inj.id().to_string(), prior);
                out.summary.wired.push(inj.display_name().to_string());
            }
            Err(e) => out.summary.failed.push((inj.id().to_string(), e.to_string())),
        }
    }
    out
}

/// Disable every client that has a recorded prior, restoring its config.
pub fn disable_all(
    priors: &BTreeMap<String, ClientPrior>,
    injectors: &[Box<dyn ClientInjector>],
) -> WireSummary {
    let mut summary = WireSummary::default();
    for inj in injectors {
        let Some(prior) = priors.get(inj.id()) else {
            continue;
        };
        match inj.disable(prior) {
            Ok(()) => summary.wired.push(inj.display_name().to_string()),
            Err(e) => summary.failed.push((inj.id().to_string(), e.to_string())),
        }
    }
    summary
}
```

- [ ] **Step 4: Run the orchestration tests**

Run: `MISTRALRS_METAL_PRECOMPILE=0 cargo test --lib integrations::tests`
Expected: PASS (3 Task-1 + 2 orchestration tests).

- [ ] **Step 5: Wire the tray `CheckMenuItem`**

In `src/tray.rs`, following the existing Routing-submenu pattern:

1. Near the other menu items (after `routing_line`, around line 460–470), add the toggle item and its sub-line, reading initial state from settings:

```rust
let integrations_enabled = crate::settings::load_integrations().enabled;
let toggle_item = CheckMenuItem::new(
    "Route apps through localllm",
    true,           // enabled (clickable)
    integrations_enabled, // checked
    None,
);
let wired_line = MenuItem::new(
    wired_label(&crate::settings::load_integrations()),
    false,
    None,
);
```

2. Append them to the menu (with a separator) in the same place other items are appended.

3. Add a `wired_label` free function near the other label helpers in `tray.rs`:

```rust
fn wired_label(state: &crate::settings::IntegrationState) -> String {
    if !state.enabled {
        return "Apps: direct to provider".to_string();
    }
    if state.priors.is_empty() {
        return "Wired: none (no client configs found)".to_string();
    }
    // Map known ids to display names for the line.
    let names: Vec<&str> = state
        .priors
        .keys()
        .map(|id| match id.as_str() {
            "claude-code" => "Claude Code",
            "codex" => "Codex",
            other => other,
        })
        .collect();
    format!("Wired: {}", names.join(", "))
}
```

4. In the `MenuEvent` handler loop, add a branch for `toggle_item.id()`:

```rust
} else if event.id == toggle_item.id() {
    let injectors = crate::integrations::injectors_default();
    let mut state = crate::settings::load_integrations();
    if state.enabled {
        // turn OFF
        let _ = crate::integrations::disable_all(&state.priors, &injectors);
        state = crate::settings::IntegrationState::default();
    } else {
        // turn ON
        let outcome = crate::integrations::enable_all(server_port, &injectors);
        state = crate::settings::IntegrationState { enabled: true, priors: outcome.priors };
    }
    let _ = crate::settings::save_integrations(&state);
    toggle_item.set_checked(state.enabled);
    wired_line.set_text(wired_label(&state));
}
```

Use the existing port value the tray already knows (the server URL/port shown in `url_line`; thread it as `server_port: u16` exactly like the URL is threaded — reuse the same source rather than introducing a new one).

- [ ] **Step 6: Build + full suite + clippy**

Run:
```
MISTRALRS_METAL_PRECOMPILE=0 cargo build
MISTRALRS_METAL_PRECOMPILE=0 cargo test
MISTRALRS_METAL_PRECOMPILE=0 cargo clippy --all-targets
```
Expected: build clean; all tests pass; no new clippy warnings in `integrations`/`settings`/`tray` changed code (pre-existing SSE `to_string`, `map_or`, items-after-test-module warnings remain, untouched).

- [ ] **Step 7: Commit**

```bash
git add src/integrations/mod.rs src/tray.rs
git commit -m "feat(integrations,tray): toggle wiring — enable/disable all clients + tray CheckMenuItem"
```

---

## Notes for the executor
- The tray click handler (Task 5 Step 5) is GUI code that the existing test suite does not exercise (consistent with prior tray work). Verify it compiles and behaves by reasoning + the unit-tested orchestration beneath it; a manual smoke test (toggle on, inspect `~/.claude/settings.json` + `~/.codex/config.toml`, toggle off, confirm revert) is the acceptance check the controller runs after the branch is built.
- If threading `server_port` into the tray event loop turns out to require touching the tray's construction signature more than trivially, report it as DONE_WITH_CONCERNS rather than restructuring the tray — the controller will finish the wiring (as in prior sub-projects).
