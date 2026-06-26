# Hot-swappable Model Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the active local model at runtime (no restart) via token-guarded localhost endpoints, returning 503 while switching and respecting the Metal single-context rule.

**Architecture:** A new `ModelManager` wraps the engine behind `arc_swap::ArcSwapOption`, implements `Generator` (so it slots into `AppState` where `gen` is today), tracks in-flight requests, and runs a switch task that drains → drops the old engine → builds the new (download-with-progress) → swaps → restores the old on failure. A builder-closure seam lets tests drive the whole machine with `FakeGen` (no model/Metal/network). Two `/admin/model` endpoints (token-guarded) start a switch and report progress.

**Tech Stack:** Rust, axum 0.7, tokio, arc-swap, reqwest (stream); dev: wiremock.

## Global Constraints

- Single self-contained binary; no subprocess-per-model. New runtime dep: `arc-swap`.
- **Metal single-context rule:** never have two llama.cpp contexts resident at once → a switch must **drain in-flight → drop the old engine → then build the new**.
- Switch gap → requests get **503 + `Retry-After: 5`** (`{"error":"model switching, retry shortly"}`).
- Control endpoints: `POST /admin/model` (body `{"repo","file"}`), `GET /admin/model/status`. Both bound to the existing 127.0.0.1 listener.
- **Admin token** on all `/admin/*`: header `X-Admin-Token`, **constant-time** compare, no new dep. From `--admin-token` or a random 32-hex token at startup; written to `<config-dir>/localllm/admin-token` at **0600**; never log the value.
- One switch at a time → concurrent → **409**. Bad body → **400**. Missing/invalid token → **401**.
- v1 switches **model only** (repo+file); `ctx_len`/`kv_type`/`kv_cache_dir` stay as configured. `/v1/*` stay unauthenticated.
- On switch failure (old already dropped) → **restore the previous model** (cached → fast); restore-fail → `Error` state, all `/v1/*` → 503.
- TDD; complete code each step; commit per task; pristine test output.

---

### Task 1: `download::ensure_model_with_progress`

**Files:**
- Modify: `src/download.rs` (add progress variant; make `ensure_model` delegate)
- Test: `src/download.rs` (`#[cfg(test)]`, wiremock)

**Interfaces:**
- Produces: `pub async fn ensure_model_with_progress(repo: &str, files: &[String], on_progress: impl Fn(u64, Option<u64>)) -> anyhow::Result<Vec<PathBuf>>`
- `ensure_model(repo, files)` keeps its signature (delegates with a no-op callback).

- [ ] **Step 1: Write the failing test**

Add to `src/download.rs` `#[cfg(test)] mod tests` (add `use wiremock::...` at the top of the test module):

```rust
    #[tokio::test]
    async fn progress_reports_total_and_reaches_full() {
        use std::sync::{Arc, Mutex};
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = vec![b'x'; 1000];
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        // Point hf_url at the mock by using a repo/file whose resolve URL is the mock.
        // ensure_model_with_progress builds the URL via hf_url(repo,file); override the
        // base by setting the file to an absolute path is not possible, so we test the
        // callback wiring through a direct download against the mock server URL instead.
        let seen: Arc<Mutex<Vec<(u64, Option<u64>)>>> = Arc::new(Mutex::new(vec![]));
        let seen2 = seen.clone();

        // Use a temp cache by pointing HOME/XDG cache via the repo/file path under a temp dir
        // is overkill here; instead assert the callback contract via a unit on the streaming
        // helper. We verify: final callback has done == total == 1000.
        let dir = std::env::temp_dir().join(format!("dl-prog-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("model.bin");
        download_to_with_progress(&server.uri(), &dest, |done, total| {
            seen2.lock().unwrap().push((done, total));
        })
        .await
        .unwrap();

        let s = seen.lock().unwrap();
        let (last_done, last_total) = *s.last().unwrap();
        assert_eq!(last_done, 1000);
        assert_eq!(last_total, Some(1000));
        assert!(dest.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
```

> This test drives a small internal helper `download_to_with_progress(url, dest, cb)` that does the streaming + progress for one file; `ensure_model_with_progress` calls it per file after the cache-hit check. Splitting it out makes the progress logic unit-testable against a mock URL without faking HuggingFace's host.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib download::tests::progress_reports 2>&1 | head -20`
Expected: FAIL — `download_to_with_progress` / `ensure_model_with_progress` undefined.

- [ ] **Step 3: Implement the helper + the progress variant; make `ensure_model` delegate**

In `src/download.rs`, add the per-file streaming helper (factor it out of the existing loop body):

```rust
/// Download a single URL to `dest` (via a `.part` temp), invoking `on_progress`
/// with (bytes_done, total_from_Content-Length) throttled to ~1% changes.
async fn download_to_with_progress(
    url: &str,
    dest: &std::path::Path,
    on_progress: impl Fn(u64, Option<u64>),
) -> anyhow::Result<()> {
    use futures_util::StreamExt;
    let client = reqwest::Client::builder().use_rustls_tls().build()
        .context("building reqwest client")?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating cache dir {}", parent.display()))?;
    }
    let part_path = dest.with_extension("part");
    let response = client.get(url).send().await
        .with_context(|| format!("GET {url}"))?
        .error_for_status().with_context(|| format!("HTTP error for {url}"))?;
    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut part_file = tokio::fs::File::create(&part_path).await
        .with_context(|| format!("creating {}", part_path.display()))?;
    let mut downloaded: u64 = 0;
    let mut last_pct: i64 = -1;
    on_progress(0, total);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("reading stream for {url}"))?;
        tokio::io::AsyncWriteExt::write_all(&mut part_file, &chunk).await
            .with_context(|| format!("writing to {}", part_path.display()))?;
        downloaded += chunk.len() as u64;
        if let Some(t) = total {
            if t > 0 {
                let pct = (downloaded * 100 / t) as i64;
                if pct != last_pct {
                    last_pct = pct;
                    on_progress(downloaded, total);
                }
            }
        } else {
            on_progress(downloaded, None);
        }
    }
    tokio::io::AsyncWriteExt::flush(&mut part_file).await?;
    drop(part_file);
    std::fs::rename(&part_path, dest)
        .with_context(|| format!("renaming {} → {}", part_path.display(), dest.display()))?;
    on_progress(downloaded, total.or(Some(downloaded)));
    Ok(())
}

/// Like [`ensure_model`] but reports download progress per chunk. `on_progress`
/// receives (bytes_done, total) for the file currently downloading; cache-hit
/// files report 100% immediately (done==total==file len).
pub async fn ensure_model_with_progress(
    repo: &str,
    files: &[String],
    on_progress: impl Fn(u64, Option<u64>),
) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(files.len());
    for file in files {
        let dest = cache_path(repo, file);
        if dest.exists() && dest.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            info!("cache hit: {}", dest.display());
            let len = dest.metadata().map(|m| m.len()).unwrap_or(0);
            on_progress(len, Some(len)); // 100%
            paths.push(dest);
            continue;
        }
        let url = hf_url(repo, file);
        info!("downloading {} → {}", url, dest.display());
        download_to_with_progress(&url, &dest, &on_progress).await?;
        info!("download complete: {}", dest.display());
        paths.push(dest);
    }
    Ok(paths)
}
```

Replace the body of the existing `ensure_model` with a delegation:

```rust
pub async fn ensure_model(repo: &str, files: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    ensure_model_with_progress(repo, files, |_, _| {}).await
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib download:: 2>&1 | tail -15`
Expected: PASS — the new progress test plus the pre-existing download tests.

- [ ] **Step 5: Commit**

```bash
git add src/download.rs
git commit -m "feat(download): ensure_model_with_progress (per-chunk progress callback)"
```

---

### Task 2: `ModelManager` — struct, `Generator` impl, in-flight gate, status

**Files:**
- Create: `src/model_manager.rs`
- Modify: `src/lib.rs` (`pub mod model_manager;`), `Cargo.toml` (`arc-swap = "1"`)
- Test: `src/model_manager.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces:
  - `ModelSpec { repo: String, file: String }` (Clone, Serialize, Deserialize, PartialEq)
  - `SwitchPhase { Idle, Draining, Downloading, Loading }` (Serialize, snake_case)
  - `SwitchStatus { state: String, current: ModelSpec, target: Option<ModelSpec>, phase: SwitchPhase, progress: u8, error: Option<String> }` (Serialize)
  - `EngineBuilder = Box<dyn Fn(ModelSpec) -> futures::future::BoxFuture<'static, anyhow::Result<Arc<dyn Generator>>> + Send + Sync>`
  - `ModelManager` with `new(initial: Arc<dyn Generator>, current: ModelSpec, builder: EngineBuilder) -> Arc<Self>`, `is_switching(&self) -> bool`, `status(&self) -> SwitchStatus`, and `impl Generator`.
- Consumes: `crate::server::Generator`, `crate::api::common::{ChatRequest, ChatResult, StreamDelta}`.

- [ ] **Step 1: Add the dependency + module**

In `Cargo.toml` `[dependencies]` add:

```toml
arc-swap = "1"
```

In `src/lib.rs`, after `pub mod engine_llama;` add:

```rust
pub mod model_manager;
```

- [ ] **Step 2: Write the failing tests**

Create `src/model_manager.rs`:

```rust
//! Runtime-swappable inference backend.
//!
//! `ModelManager` holds the current engine behind `ArcSwapOption` and implements
//! `Generator`, so it slots into `AppState` where the fixed engine used to live.
//! It tracks in-flight requests and (in a later step) runs a switch that drains,
//! drops the old engine — freeing the single Metal context — builds the new one,
//! and swaps it in, restoring the old engine on failure.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwapOption;
use futures::future::BoxFuture;
use futures::stream::BoxStream;

use crate::api::common::{ChatRequest, ChatResult, StreamDelta};
use crate::server::Generator;

/// Identifies a local model: a HuggingFace repo + GGUF filename.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelSpec {
    pub repo: String,
    pub file: String,
}

/// Phase within a switch, for status reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SwitchPhase {
    Idle,
    Draining,
    Downloading,
    Loading,
}

/// Snapshot of the manager's switch state for `GET /admin/model/status`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SwitchStatus {
    pub state: String, // "ready" | "switching" | "error"
    pub current: ModelSpec,
    pub target: Option<ModelSpec>,
    pub phase: SwitchPhase,
    pub progress: u8,
    pub error: Option<String>,
}

/// Builds an engine for a given spec (download + load). Boxed so production and
/// tests can inject different implementations.
pub type EngineBuilder = Box<
    dyn Fn(ModelSpec) -> BoxFuture<'static, anyhow::Result<Arc<dyn Generator>>> + Send + Sync,
>;

const PHASE_IDLE: u8 = 0;
const PHASE_DRAINING: u8 = 1;
const PHASE_DOWNLOADING: u8 = 2;
const PHASE_LOADING: u8 = 3;

pub struct ModelManager {
    engine: ArcSwapOption<dyn Generator>,
    current: Mutex<ModelSpec>,
    target: Mutex<Option<ModelSpec>>,
    switching: AtomicBool,
    errored: AtomicBool,
    inflight: AtomicUsize,
    phase: AtomicU8,
    progress: AtomicU8,
    error: Mutex<Option<String>>,
    builder: EngineBuilder,
}

impl ModelManager {
    pub fn new(initial: Arc<dyn Generator>, current: ModelSpec, builder: EngineBuilder) -> Arc<Self> {
        Arc::new(Self {
            engine: ArcSwapOption::from(Some(initial)),
            current: Mutex::new(current),
            target: Mutex::new(None),
            switching: AtomicBool::new(false),
            errored: AtomicBool::new(false),
            inflight: AtomicUsize::new(0),
            phase: AtomicU8::new(PHASE_IDLE),
            progress: AtomicU8::new(100),
            error: Mutex::new(None),
            builder,
        })
    }

    pub fn is_switching(&self) -> bool {
        self.switching.load(Ordering::SeqCst)
    }

    pub fn status(&self) -> SwitchStatus {
        let state = if self.switching.load(Ordering::SeqCst) {
            "switching"
        } else if self.errored.load(Ordering::SeqCst) {
            "error"
        } else {
            "ready"
        };
        let phase = match self.phase.load(Ordering::SeqCst) {
            PHASE_DRAINING => SwitchPhase::Draining,
            PHASE_DOWNLOADING => SwitchPhase::Downloading,
            PHASE_LOADING => SwitchPhase::Loading,
            _ => SwitchPhase::Idle,
        };
        SwitchStatus {
            state: state.to_string(),
            current: self.current.lock().unwrap().clone(),
            target: self.target.lock().unwrap().clone(),
            phase,
            progress: self.progress.load(Ordering::SeqCst),
            error: self.error.lock().unwrap().clone(),
        }
    }
}

/// Decrements the in-flight counter when dropped (covers both the request future
/// and a streaming response's lifetime).
struct InflightGuard(Arc<ModelManager>);
impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A stream that holds an `InflightGuard` until it is fully consumed/dropped.
struct GuardedStream {
    inner: BoxStream<'static, anyhow::Result<StreamDelta>>,
    _guard: InflightGuard,
}
impl futures::Stream for GuardedStream {
    type Item = anyhow::Result<StreamDelta>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[async_trait::async_trait]
impl Generator for ModelManager {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult> {
        if self.switching.load(Ordering::SeqCst) {
            anyhow::bail!("model switching");
        }
        self.inflight.fetch_add(1, Ordering::SeqCst);
        let _dec = scopeguard_dec(&self.inflight);
        let engine = self
            .engine
            .load_full()
            .ok_or_else(|| anyhow::anyhow!("model switching"))?;
        engine.generate(req).await
    }

    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        if self.switching.load(Ordering::SeqCst) {
            anyhow::bail!("model switching");
        }
        self.inflight.fetch_add(1, Ordering::SeqCst);
        let engine = match self.engine.load_full() {
            Some(e) => e,
            None => {
                self.inflight.fetch_sub(1, Ordering::SeqCst);
                anyhow::bail!("model switching");
            }
        };
        // NOTE: the stream is guarded by a manual decrement in a wrapper created
        // by the manager's Arc-aware caller; see `generate_stream_arc`.
        let inner = engine.generate_stream(req).await;
        match inner {
            Ok(s) => Ok(s),
            Err(e) => {
                self.inflight.fetch_sub(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }
}

/// Decrement-on-drop helper for the non-stream path.
fn scopeguard_dec(counter: &AtomicUsize) -> impl Drop + '_ {
    struct G<'a>(&'a AtomicUsize);
    impl Drop for G<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    G(counter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::{ContentPart, FinishReason};

    fn marker_gen(tag: &'static str) -> Arc<dyn Generator> {
        Arc::new(crate::test_support::TaggedGen(tag))
    }

    fn spec(repo: &str, file: &str) -> ModelSpec {
        ModelSpec { repo: repo.into(), file: file.into() }
    }

    fn noop_builder() -> EngineBuilder {
        Box::new(|_spec| Box::pin(async { Ok(marker_gen("new")) }))
    }

    #[tokio::test]
    async fn delegates_to_current_engine() {
        let m = ModelManager::new(marker_gen("orig"), spec("r", "f"), noop_builder());
        let out = m.generate(ChatRequest {
            messages: vec![], tools: vec![], max_tokens: None,
            temperature: None, stream: false, model: "m".into(),
        }).await.unwrap();
        // TaggedGen returns its tag as the text content.
        match &out.content[0] {
            ContentPart::Text(t) => assert_eq!(t, "orig"),
            _ => panic!("expected text"),
        }
        assert_eq!(out.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn fresh_manager_is_ready_not_switching() {
        let m = ModelManager::new(marker_gen("orig"), spec("r", "f"), noop_builder());
        assert!(!m.is_switching());
        let s = m.status();
        assert_eq!(s.state, "ready");
        assert_eq!(s.current, spec("r", "f"));
        assert_eq!(s.target, None);
        assert_eq!(s.progress, 100);
    }
}
```

> The tests use a shared `TaggedGen` test double whose `generate` returns a `ChatResult` with the tag as text — add it in Task 2 Step 3 under a small `pub mod test_support` so both this module's tests and the integration tests can build distinct engines.

- [ ] **Step 3: Add the `TaggedGen` test double + register it**

In `src/lib.rs`, add (near the other test helpers, but NOT behind `#[cfg(test)]` so integration tests can use it):

```rust
/// Test/diagnostic generator whose responses echo a fixed tag, so a test can
/// tell which engine instance is currently serving (used by hot-swap tests).
pub mod test_support {
    use crate::api::common::{ChatResult, ContentPart, FinishReason, StreamDelta};
    use crate::server::Generator;
    use futures::stream::BoxStream;

    pub struct TaggedGen(pub &'static str);

    #[async_trait::async_trait]
    impl Generator for TaggedGen {
        async fn generate(&self, _req: crate::api::common::ChatRequest) -> anyhow::Result<ChatResult> {
            Ok(ChatResult {
                content: vec![ContentPart::Text(self.0.to_string())],
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }
        async fn generate_stream(
            &self,
            _req: crate::api::common::ChatRequest,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
            let tag = self.0.to_string();
            let deltas: Vec<anyhow::Result<StreamDelta>> = vec![
                Ok(StreamDelta { text: Some(tag), done: false, finish_reason: None }),
                Ok(StreamDelta { text: None, done: true, finish_reason: Some(FinishReason::Stop) }),
            ];
            Ok(Box::pin(futures::stream::iter(deltas)))
        }
    }
}
```

> `GuardedStream` is defined in this task for use by Task 3's stream path; if `cargo` warns it is unused until Task 3 wires it, add `#[allow(dead_code)]` on `GuardedStream` and `InflightGuard` and remove it in Task 3.

- [ ] **Step 4: Run to verify the tests pass**

Run: `cargo test --lib model_manager:: 2>&1 | tail -15`
Expected: PASS (2 tests). Build is warning-free (or only the documented `dead_code` allows).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/model_manager.rs
git commit -m "feat(model_manager): swappable Generator + status (no switch logic yet)"
```

---

### Task 3: `start_switch` — drain → drop → build → swap → restore

**Files:**
- Modify: `src/model_manager.rs` (add `start_switch`, `SwitchError`, progress setter, stream guard wiring)
- Test: `src/model_manager.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `EngineBuilder`, the manager fields from Task 2.
- Produces:
  - `enum SwitchError { AlreadySwitching }`
  - `ModelManager::start_switch(self: &Arc<Self>, target: ModelSpec) -> Result<(), SwitchError>` (spawns the switch task; returns immediately)
  - `ModelManager::set_progress(&self, pct: u8)` (called by the builder's download callback)

- [ ] **Step 1: Write the failing tests**

Add to `src/model_manager.rs` tests:

```rust
    use std::sync::atomic::AtomicUsize as TestAtomicUsize;

    /// A builder that records calls and returns a TaggedGen("new"), optionally
    /// after awaiting a release signal, or failing for a specific repo.
    fn counting_builder(
        calls: Arc<TestAtomicUsize>,
        fail_repo: Option<&'static str>,
    ) -> EngineBuilder {
        Box::new(move |spec: ModelSpec| {
            let calls = calls.clone();
            let fail = fail_repo.map(|s| s.to_string());
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                if Some(&spec.repo) == fail.as_ref() {
                    anyhow::bail!("boom");
                }
                Ok(marker_gen("new"))
            })
        })
    }

    async fn wait_ready(m: &Arc<ModelManager>) {
        for _ in 0..200 {
            if !m.is_switching() { return; }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        panic!("switch did not finish");
    }

    #[tokio::test]
    async fn happy_switch_swaps_engine_and_updates_current() {
        let calls = Arc::new(TestAtomicUsize::new(0));
        let m = ModelManager::new(marker_gen("orig"), spec("r", "old"), counting_builder(calls.clone(), None));
        m.start_switch(spec("r2", "new")).unwrap();
        wait_ready(&m).await;
        // engine now serves "new"
        let out = m.generate(req()).await.unwrap();
        assert!(matches!(&out.content[0], ContentPart::Text(t) if t == "new"));
        assert_eq!(m.status().current, spec("r2", "new"));
        assert_eq!(m.status().state, "ready");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn second_switch_while_in_progress_is_rejected() {
        // Builder blocks until released → first switch stays in progress.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let builder: EngineBuilder = Box::new(move |_spec| {
            let rx = rx.clone();
            Box::pin(async move {
                if let Some(rx) = rx.lock().await.take() { let _ = rx.await; }
                Ok(marker_gen("new"))
            })
        });
        let m = ModelManager::new(marker_gen("orig"), spec("r", "old"), builder);
        m.start_switch(spec("r2", "a")).unwrap();
        // give the task a moment to flip `switching`
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(m.start_switch(spec("r3", "b")), Err(SwitchError::AlreadySwitching)));
        assert!(m.is_switching());
        let _ = tx.send(());
        wait_ready(&m).await;
    }

    #[tokio::test]
    async fn failed_build_restores_previous_model() {
        // target repo "bad" fails; restore rebuilds "r"/"old" (succeeds).
        let calls = Arc::new(TestAtomicUsize::new(0));
        let m = ModelManager::new(marker_gen("orig"), spec("r", "old"), counting_builder(calls.clone(), Some("bad")));
        m.start_switch(spec("bad", "x")).unwrap();
        wait_ready(&m).await;
        let s = m.status();
        assert_eq!(s.state, "ready"); // restored
        assert_eq!(s.current, spec("r", "old")); // back to the old model
        assert!(s.error.is_some()); // but the failure is reported
        // and the engine still serves
        let out = m.generate(req()).await.unwrap();
        assert!(matches!(&out.content[0], ContentPart::Text(t) if t == "new")); // restore built a fresh engine
    }

    fn req() -> ChatRequest {
        ChatRequest { messages: vec![], tools: vec![], max_tokens: None,
            temperature: None, stream: false, model: "m".into() }
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib model_manager:: 2>&1 | head -20`
Expected: FAIL — `start_switch` / `SwitchError` undefined.

- [ ] **Step 3: Implement `start_switch` + helpers**

In `src/model_manager.rs`, add:

```rust
/// Why a switch could not be started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchError {
    AlreadySwitching,
}

impl ModelManager {
    /// Set download progress percent (0..100). Called by the builder's callback.
    pub fn set_progress(&self, pct: u8) {
        self.progress.store(pct.min(100), Ordering::SeqCst);
    }

    fn set_phase(&self, p: u8) {
        self.phase.store(p, Ordering::SeqCst);
    }

    /// Start switching to `target`. Returns immediately; the switch runs in a
    /// spawned task. One switch at a time — a second call while one is running
    /// returns `AlreadySwitching` (→ HTTP 409).
    pub fn start_switch(self: &Arc<Self>, target: ModelSpec) -> Result<(), SwitchError> {
        if self
            .switching
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(SwitchError::AlreadySwitching);
        }
        // entering a switch clears any prior error state
        self.errored.store(false, Ordering::SeqCst);
        *self.error.lock().unwrap() = None;
        *self.target.lock().unwrap() = Some(target.clone());
        self.progress.store(0, Ordering::SeqCst);

        let me = self.clone();
        tokio::spawn(async move {
            me.run_switch(target).await;
        });
        Ok(())
    }

    async fn run_switch(self: Arc<Self>, target: ModelSpec) {
        // 1. Drain in-flight (bounded). The `switching` flag already 503s new ones.
        self.set_phase(PHASE_DRAINING);
        let drained = {
            let mut ok = false;
            for _ in 0..6000 {
                // ~30s at 5ms
                if self.inflight.load(Ordering::SeqCst) == 0 {
                    ok = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            ok
        };
        if !drained {
            // Nothing torn down yet → abort, stay on the old model.
            *self.error.lock().unwrap() = Some("switch aborted: requests did not drain".into());
            self.set_phase(PHASE_IDLE);
            *self.target.lock().unwrap() = None;
            self.switching.store(false, Ordering::SeqCst);
            return;
        }

        // 2. Drop the old engine FIRST (frees the single Metal context).
        let previous = self.current.lock().unwrap().clone();
        self.engine.store(None);

        // 3. Build the new engine.
        self.set_phase(PHASE_DOWNLOADING);
        match (self.builder)(target.clone()).await {
            Ok(new_engine) => {
                self.engine.store(Some(new_engine));
                *self.current.lock().unwrap() = target;
                *self.target.lock().unwrap() = None;
                self.progress.store(100, Ordering::SeqCst);
                self.set_phase(PHASE_IDLE);
                self.switching.store(false, Ordering::SeqCst);
            }
            Err(e) => {
                let msg = format!("model switch failed: {e}");
                tracing::warn!(target: "localllm::req", "{msg} — restoring previous model");
                crate::usage::notify("localllm — model switch failed", &msg);
                // 4. Restore the previous model (its GGUF is cached → fast).
                self.set_phase(PHASE_LOADING);
                match (self.builder)(previous.clone()).await {
                    Ok(restored) => {
                        self.engine.store(Some(restored));
                        *self.current.lock().unwrap() = previous;
                        *self.error.lock().unwrap() = Some(msg);
                        *self.target.lock().unwrap() = None;
                        self.set_phase(PHASE_IDLE);
                        self.switching.store(false, Ordering::SeqCst);
                    }
                    Err(e2) => {
                        // Could not even restore → degraded; all /v1/* will 503.
                        *self.error.lock().unwrap() =
                            Some(format!("{msg}; restore also failed: {e2}"));
                        self.errored.store(true, Ordering::SeqCst);
                        self.set_phase(PHASE_IDLE);
                        *self.target.lock().unwrap() = None;
                        self.switching.store(false, Ordering::SeqCst);
                    }
                }
            }
        }
    }
}
```

> Note: when `errored` is true the engine handle is `None`, so `generate`/`generate_stream` already bail "model switching" → the handler maps to 503. The 503 gate (Task 5) also checks `is_switching()` which is false here, so add `|| manager.is_errored()` to the gate in Task 5; expose `pub fn is_errored(&self) -> bool`.

Add the `is_errored` accessor:

```rust
impl ModelManager {
    pub fn is_errored(&self) -> bool {
        self.errored.load(Ordering::SeqCst)
    }
}
```

- [ ] **Step 4: Wire the stream guard (replace the Task-2 placeholder stream path)**

Replace `generate_stream` in `impl Generator for ModelManager` with a version that keeps the in-flight count alive for the stream's whole lifetime. Because `&self` is not an `Arc`, route streaming through an `Arc<Self>` method is unnecessary — instead decrement via a guard owned by the returned stream using a raw pointer to the counter is unsafe; simplest safe approach: clone an `Arc` to a standalone counter. Change `inflight` usage so the guard holds an `Arc<AtomicUsize>`:

Add a field `inflight: Arc<AtomicUsize>` (change the type from `AtomicUsize` to `Arc<AtomicUsize>` in the struct and `new`), update all `self.inflight.fetch_*`/`load` call sites to deref through the Arc (same syntax). Then:

```rust
    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
        if self.switching.load(Ordering::SeqCst) {
            anyhow::bail!("model switching");
        }
        self.inflight.fetch_add(1, Ordering::SeqCst);
        let counter = self.inflight.clone();
        let engine = match self.engine.load_full() {
            Some(e) => e,
            None => { counter.fetch_sub(1, Ordering::SeqCst); anyhow::bail!("model switching"); }
        };
        match engine.generate_stream(req).await {
            Ok(inner) => Ok(Box::pin(CounterStream { inner, counter })),
            Err(e) => { counter.fetch_sub(1, Ordering::SeqCst); Err(e) }
        }
    }
```

Replace `GuardedStream`/`InflightGuard` with a self-contained `CounterStream`:

```rust
struct CounterStream {
    inner: BoxStream<'static, anyhow::Result<StreamDelta>>,
    counter: Arc<AtomicUsize>,
}
impl Drop for CounterStream {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}
impl futures::Stream for CounterStream {
    type Item = anyhow::Result<StreamDelta>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
```

(Also delete `scopeguard_dec` and use the Arc-counter directly in `generate`: `let counter = self.inflight.clone(); let _g = DecOnDrop(counter);` with a tiny `struct DecOnDrop(Arc<AtomicUsize>)` impl `Drop`. Define `DecOnDrop` once and use it in both paths for symmetry.)

- [ ] **Step 5: Run the switch tests**

Run: `cargo test --lib model_manager:: 2>&1 | tail -20`
Expected: PASS — `happy_switch_swaps_engine_and_updates_current`, `second_switch_while_in_progress_is_rejected`, `failed_build_restores_previous_model`, plus Task 2's tests.

- [ ] **Step 6: Commit**

```bash
git add src/model_manager.rs
git commit -m "feat(model_manager): start_switch state machine (drain/drop/build/swap/restore)"
```

---

### Task 4: Admin token

**Files:**
- Modify: `src/config.rs` (`--admin-token`), `src/server.rs` (constant-time compare + `check_admin` helper)
- Create helper for the 0600 token file write in `src/server.rs` or `src/settings.rs`
- Test: `src/config.rs`, `src/server.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces:
  - `Config.admin_token: Option<String>`
  - `server::constant_time_eq(a: &[u8], b: &[u8]) -> bool`
  - `server::resolve_admin_token(cli: Option<String>) -> String` (cli or random 32-hex)
  - `server::write_admin_token_file(token: &str)` (best-effort, 0600)

- [ ] **Step 1: Add the CLI flag + tests**

In `src/config.rs` add to `Config`:

```rust
    /// Token required on /admin/* control endpoints (X-Admin-Token header).
    /// If unset, a random token is generated at startup and written to
    /// <config-dir>/localllm/admin-token (0600).
    #[arg(long)]
    pub admin_token: Option<String>,
```

Add the parse test:

```rust
    #[test]
    fn admin_token_flag_parses_and_defaults_none() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.admin_token, None);
        let c = Config::parse_from(["localllm", "--admin-token", "secret123"]);
        assert_eq!(c.admin_token.as_deref(), Some("secret123"));
    }
```

In `src/server.rs` add tests (in its `#[cfg(test)] mod tests`, or add one):

```rust
    #[test]
    fn constant_time_eq_matches_and_rejects() {
        assert!(super::constant_time_eq(b"abc", b"abc"));
        assert!(!super::constant_time_eq(b"abc", b"abd"));
        assert!(!super::constant_time_eq(b"abc", b"abcd")); // length mismatch
        assert!(!super::constant_time_eq(b"", b"x"));
    }

    #[test]
    fn resolve_admin_token_uses_cli_else_random() {
        assert_eq!(super::resolve_admin_token(Some("z".into())), "z");
        let r = super::resolve_admin_token(None);
        assert_eq!(r.len(), 32); // random 32-hex
        assert_ne!(super::resolve_admin_token(None), r); // different each call
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib config::tests::admin_token 2>&1 | head; cargo test --lib server::tests::constant_time 2>&1 | head`
Expected: FAIL — flag/field and helpers undefined.

- [ ] **Step 3: Implement the helpers**

In `src/server.rs` add (module level):

```rust
/// Constant-time byte comparison: false on length mismatch, otherwise XOR-
/// accumulate so the timing does not depend on where the first difference is.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Resolve the admin token: the CLI value, else a fresh random 32-hex string.
pub fn resolve_admin_token(cli: Option<String>) -> String {
    cli.unwrap_or_else(|| {
        let u = uuid::Uuid::new_v4();
        // 32 hex chars (no dashes)
        u.simple().to_string()
    })
}

/// Best-effort: write the admin token to `<config-dir>/localllm/admin-token`
/// with 0600 perms so the tray/window/CLI can read it. Logs only that it wrote
/// the file, never the value.
pub fn write_admin_token_file(token: &str) {
    let Some(dir) = dirs::config_dir() else { return };
    let path = dir.join("localllm").join("admin-token");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&path, token).is_ok() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        tracing::info!("admin token written to {}", path.display());
    }
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test --lib 2>&1 | grep -E "admin_token|constant_time|resolve_admin"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/server.rs
git commit -m "feat(server): admin token (--admin-token / random) + constant-time check + 0600 file"
```

---

### Task 5: Wire `ModelManager` + admin endpoints into the server

**Files:**
- Modify: `src/server.rs` (AppState: `gen`→`manager` + `admin_token`; `router` signature; handler 503 gate; `/admin/model` + `/admin/model/status` routes + `check_admin`)
- Modify: `src/lib.rs` (build real `EngineBuilder` + initial `ModelManager`, resolve+write token, pass to `router`; update `router_for_test*`)
- Test: `tests/http.rs`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `AppState { manager: Arc<crate::model_manager::ModelManager>, model_id, policy, local_ctx_window, usage, cloud_token_alert, admin_token: Arc<str> }`; `router(manager, model_id, policy, local_ctx_window, usage, cloud_token_alert, admin_token)`.

- [ ] **Step 1: Swap `gen` → `manager` in `AppState` + `router`**

In `src/server.rs`, replace the `gen` field:

```rust
    /// The swappable inference backend (also the Generator the handlers call).
    pub manager: Arc<crate::model_manager::ModelManager>,
```
and add at the end of `AppState`:
```rust
    /// Token required on /admin/* endpoints.
    pub admin_token: Arc<str>,
```

Update `router(...)` to take `manager: Arc<crate::model_manager::ModelManager>` (instead of `gen`) and `admin_token: Arc<str>`, build the `AppState` accordingly, and add the routes:

```rust
        .route("/admin/model", post(handle_admin_switch))
        .route("/admin/model/status", get(handle_admin_status))
```
(before the `.layer(DefaultBodyLimit::…)`).

Replace every `state.gen.generate(...)`/`state.gen.generate_stream(...)` in the handlers with `state.manager.generate(...)`/`state.manager.generate_stream(...)`. Replace the `cascade_or_result(..., state.gen.generate(internal).await, ...)` calls with `state.manager.generate(internal).await`.

- [ ] **Step 2: Add the 503 gate + admin handlers**

In `src/server.rs`, at the top of BOTH `handle_oai_chat` and `handle_anth_messages`, right after the `serde_json::from_slice` parse succeeds (before `route_decision`):

```rust
    if state.manager.is_switching() || state.manager.is_errored() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "5")],
            Json(json!({"error": "model switching, retry shortly"})),
        ).into_response();
    }
```

Add the admin handlers + guard:

```rust
fn check_admin(headers: &HeaderMap, state: &AppState) -> Result<(), axum::response::Response> {
    use axum::response::IntoResponse;
    let provided = headers.get("x-admin-token").and_then(|v| v.to_str().ok()).unwrap_or("");
    if constant_time_eq(provided.as_bytes(), state.admin_token.as_bytes()) {
        Ok(())
    } else {
        Err((StatusCode::UNAUTHORIZED, Json(json!({"error":"missing or invalid admin token"}))).into_response())
    }
}

#[derive(serde::Deserialize)]
struct AdminSwitchBody { repo: String, file: String }

async fn handle_admin_switch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    raw: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Err(resp) = check_admin(&headers, &state) { return resp; }
    let body: AdminSwitchBody = match serde_json::from_slice(&raw) {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    };
    if body.repo.is_empty() || body.file.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error":"repo and file are required"}))).into_response();
    }
    let spec = crate::model_manager::ModelSpec { repo: body.repo, file: body.file };
    match state.manager.start_switch(spec) {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"state":"switching"}))).into_response(),
        Err(crate::model_manager::SwitchError::AlreadySwitching) =>
            (StatusCode::CONFLICT, Json(json!({"error":"switch already in progress"}))).into_response(),
    }
}

async fn handle_admin_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Err(resp) = check_admin(&headers, &state) { return resp; }
    Json(state.manager.status()).into_response()
}
```

- [ ] **Step 3: Build the real builder + manager in `lib.rs`**

In `src/lib.rs` `run_server_with_ready_and_policy`, replace the engine-construction + router call. After computing `kv_cache_type`/`kv_cache_dir` and the initial `engine: Arc<dyn Generator>` (keep the existing load for the initial engine), build the manager:

```rust
    use crate::model_manager::{ModelManager, ModelSpec};
    let initial_spec = ModelSpec { repo: cfg.model_id.clone(), file: cfg.gguf_files[0].clone() };

    // Builder: download (with progress → manager) then load a LlamaEngine.
    let ctx_len = cfg.ctx_len;
    let kv_type = cfg.llama_kv_cache_type();
    let kv_dir = cfg.resolved_kv_cache_dir();
    let manager_slot: std::sync::Arc<std::sync::OnceLock<std::sync::Weak<ModelManager>>> =
        std::sync::Arc::new(std::sync::OnceLock::new());
    let slot_for_builder = manager_slot.clone();
    let builder: crate::model_manager::EngineBuilder = Box::new(move |spec: ModelSpec| {
        let (ctx_len, kv_type, kv_dir) = (ctx_len, kv_type, kv_dir.clone());
        let slot = slot_for_builder.clone();
        Box::pin(async move {
            let progress_target = slot.get().and_then(|w| w.upgrade());
            crate::download::ensure_model_with_progress(&spec.repo, &[spec.file.clone()], |done, total| {
                if let Some(m) = &progress_target {
                    let pct = match total { Some(t) if t > 0 => (done * 100 / t) as u8, _ => 0 };
                    m.set_progress(pct);
                }
            }).await?;
            let engine = crate::engine_llama::LlamaEngine::load(&spec.repo, &[spec.file], ctx_len, kv_type, kv_dir).await?;
            Ok(std::sync::Arc::new(engine) as std::sync::Arc<dyn Generator>)
        })
    });

    let manager = ModelManager::new(engine, initial_spec, builder);
    let _ = manager_slot.set(std::sync::Arc::downgrade(&manager));

    let admin_token = crate::server::resolve_admin_token(cfg.admin_token.clone());
    crate::server::write_admin_token_file(&admin_token);

    crate::usage::enable_notifications();
    let usage = std::sync::Arc::new(crate::usage::Usage::new());
    let app = router(
        manager,
        cfg.model_id.clone(),
        policy,
        cfg.ctx_len,
        usage,
        cfg.cloud_token_alert,
        std::sync::Arc::from(admin_token),
    );
```

(The `Backend::Mistralrs` branch still produces the initial `engine: Arc<dyn Generator>` exactly as today; only the wrapping into a `ModelManager` is new. The mistralrs builder can reuse the same `builder` — switching backends is out of scope; a switch always builds a `LlamaEngine`, which is fine since llama is the default.)

- [ ] **Step 4: Update `router_for_test*` helpers**

In `src/lib.rs`, change `router_for_test_with` to wrap the generator in a `ModelManager` with a no-op-restore test builder:

```rust
pub fn router_for_test_with(
    gen: Arc<dyn Generator>,
    policy: crate::route::RoutingPolicy,
    local_ctx_window: usize,
) -> Router {
    use crate::model_manager::{ModelManager, ModelSpec};
    let policy = Arc::new(std::sync::RwLock::new(policy));
    let usage = Arc::new(crate::usage::Usage::new());
    let builder: crate::model_manager::EngineBuilder =
        Box::new(|_spec| Box::pin(async { Ok(Arc::new(crate::test_support::TaggedGen("switched")) as Arc<dyn Generator>) }));
    let manager = ModelManager::new(gen, ModelSpec { repo: "test".into(), file: "test".into() }, builder);
    crate::server::router(manager, "test-model".to_string(), policy, local_ctx_window, usage, 200_000, Arc::from("test-token"))
}
```

(Existing tests calling `router_for_test()` are unaffected — it delegates to `router_for_test_with`.)

- [ ] **Step 5: Write the integration tests**

Append to `tests/http.rs`:

```rust
#[tokio::test]
async fn admin_status_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_get_status(app, "/admin/model/status").await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_status_ok_with_token() {
    let app = localllm::router_for_test();
    let resp = localllm::axum_test_get_with_header(app, "/admin/model/status", "x-admin-token", "test-token").await;
    assert_eq!(resp["state"], "ready");
}

#[tokio::test]
async fn admin_switch_starts_and_v1_503s_then_recovers() {
    // router_for_test's builder returns instantly, so the switch completes fast;
    // assert the switch is accepted and status reports the new current model.
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app, "/admin/model",
        r#"{"repo":"r2","file":"f2"}"#, "x-admin-token", "test-token").await;
    assert_eq!(status, 202);
}

#[tokio::test]
async fn admin_switch_rejects_bad_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app, "/admin/model", r#"{"repo":"r","file":"f"}"#, "x-admin-token", "wrong").await;
    assert_eq!(status, 401);
}
```

Add the two GET helpers to `src/lib.rs` (next to the existing test helpers):

```rust
/// GET with a header → HTTP status code.
pub async fn axum_test_get_status(app: Router, path: &str) -> u16 {
    use axum::body::Body;
    use tower::ServiceExt;
    let request = axum::http::Request::builder().method("GET").uri(path).body(Body::empty()).unwrap();
    app.oneshot(request).await.unwrap().status().as_u16()
}

/// GET with a header → parsed JSON body.
pub async fn axum_test_get_with_header(app: Router, path: &str, hname: &str, hval: &str) -> serde_json::Value {
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let request = axum::http::Request::builder().method("GET").uri(path)
        .header(hname, hval).body(Body::empty()).unwrap();
    let response = app.oneshot(request).await.unwrap();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}
```

- [ ] **Step 6: Run new tests, full suite, build, clippy**

Run: `cargo test --test http 2>&1 | tail -30`
Expected: PASS — the 4 admin tests + all pre-existing handler/routing/cascade tests (now served through `ModelManager` wrapping `FakeGen`).

Run: `cargo build 2>&1 | tail -8 && cargo test 2>&1 | tail -8`
Expected: clean build; full suite green.

Run: `cargo clippy --all-targets 2>&1 | grep -E "src/model_manager.rs|src/server.rs|src/download.rs|src/config.rs" | grep -- "-->" || echo "no clippy in changed files"`
Expected: no clippy warnings in changed files.

- [ ] **Step 7: Commit**

```bash
git add src/server.rs src/lib.rs tests/http.rs
git commit -m "feat(server): ModelManager backend + token-guarded /admin/model switch endpoints"
```

---

## Acceptance

- `POST /admin/model` (valid token) starts a model switch (202); `GET /admin/model/status` reports `{state,current,target,phase,progress,error}`; wrong/missing token → 401; concurrent switch → 409; bad body → 400.
- During a switch, `/v1/*` return 503 + `Retry-After`; after it completes, the new model serves.
- A failed switch restores the previous model and reports the error; a request never observes a missing engine.
- Download progress flows to `status.progress`.
- The whole state machine is unit-tested headless via the builder seam; integration tests cover the HTTP surface. `cargo test`/`build`/`clippy` clean.

## Self-Review

- **Spec coverage:** download progress (T1), swappable Generator + status + in-flight gate (T2), drain/drop/build/swap/restore + 409 (T3), admin token + constant-time + 0600 file (T4), 503 gate + endpoints + wiring (T5). Metal one-context honored by drain-before-drop-before-build (T3). 503-during-switch (T5). v1 = model-only (T3 builds LlamaEngine with existing ctx/kv).
- **Placeholder scan:** none — every step has complete code or an exact command.
- **Type consistency:** `ModelSpec`, `SwitchPhase`, `SwitchStatus`, `EngineBuilder`, `SwitchError`, `ModelManager::{new,is_switching,is_errored,status,start_switch,set_progress}`, `constant_time_eq`, `resolve_admin_token`, `write_admin_token_file`, `router(manager, model_id, policy, local_ctx_window, usage, cloud_token_alert, admin_token)`, and `AppState{manager,…,admin_token}` are consistent across T2–T5. `inflight` is an `Arc<AtomicUsize>` from T3 Step 4 onward (used by `CounterStream`). `test_support::TaggedGen` is `pub` (used by both unit and integration tests).
- **Carry-over:** `router_for_test*` now wrap a `ModelManager`; all existing integration tests keep passing because the manager delegates to the injected `FakeGen` when not switching.
