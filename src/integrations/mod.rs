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
            std::fs::set_permissions(
                &tmp,
                std::fs::Permissions::from_mode(meta.permissions().mode()),
            )?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

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
            Err(e) => out
                .summary
                .failed
                .push((inj.id().to_string(), e.to_string())),
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

    use std::cell::RefCell;

    // A fake injector that records calls and is always "detected" unless told not.
    struct Fake {
        id: &'static str,
        detected: bool,
        fail_enable: bool,
        log: std::rc::Rc<RefCell<Vec<String>>>,
    }
    impl ClientInjector for Fake {
        fn id(&self) -> &'static str {
            self.id
        }
        fn display_name(&self) -> &'static str {
            self.id
        }
        fn detect(&self) -> bool {
            self.detected
        }
        fn enable(&self, port: u16) -> anyhow::Result<ClientPrior> {
            self.log
                .borrow_mut()
                .push(format!("enable:{}:{port}", self.id));
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
            Box::new(Fake {
                id: "a",
                detected: true,
                fail_enable: false,
                log: log.clone(),
            }),
            Box::new(Fake {
                id: "b",
                detected: false,
                fail_enable: false,
                log: log.clone(),
            }),
            Box::new(Fake {
                id: "c",
                detected: true,
                fail_enable: true,
                log: log.clone(),
            }),
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
            Box::new(Fake {
                id: "a",
                detected: true,
                fail_enable: false,
                log: log.clone(),
            }),
            Box::new(Fake {
                id: "b",
                detected: true,
                fail_enable: false,
                log: log.clone(),
            }),
        ];
        let mut priors = BTreeMap::new();
        priors.insert("a".to_string(), ClientPrior::default());
        let summary = disable_all(&priors, &injectors);
        assert_eq!(summary.wired, vec!["a".to_string()]);
        assert!(log.borrow().iter().any(|l| l == "disable:a"));
        assert!(!log.borrow().iter().any(|l| l == "disable:b"));
    }
}
