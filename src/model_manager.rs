//! Runtime-swappable inference backend.
//!
//! `ModelManager` holds the current engine behind `ArcSwapOption` and implements
//! `Generator`, so it slots into `AppState` where the fixed engine used to live.
//! It tracks in-flight requests and runs a switch that drains, drops the old engine
//! — freeing the single Metal context — builds the new one, and swaps it in,
//! restoring the old engine on failure.

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

/// Why a switch could not be started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchError {
    AlreadySwitching,
}

const PHASE_IDLE: u8 = 0;
const PHASE_DRAINING: u8 = 1;
const PHASE_DOWNLOADING: u8 = 2;
const PHASE_LOADING: u8 = 3;

// DEVIATION from brief: `ArcSwapOption<dyn Generator>` requires T: Sized but
// `dyn Generator` is !Sized. Wrapping the type parameter as
// `ArcSwapOption<Arc<dyn Generator>>` stores `Option<Arc<Arc<dyn Generator>>>`,
// which satisfies arc_swap's `RefCnt` bound (Arc<dyn Generator> is Sized).
pub struct ModelManager {
    engine: ArcSwapOption<Arc<dyn Generator>>,
    current: Mutex<ModelSpec>,
    target: Mutex<Option<ModelSpec>>,
    switching: AtomicBool,
    errored: AtomicBool,
    /// Shared with `DecOnDrop` (generate path) and `CounterStream` (stream path)
    /// so the guard can outlive `&self`.
    inflight: Arc<AtomicUsize>,
    phase: AtomicU8,
    progress: AtomicU8,
    error: Mutex<Option<String>>,
    builder: EngineBuilder,
}

impl ModelManager {
    pub fn new(initial: Arc<dyn Generator>, current: ModelSpec, builder: EngineBuilder) -> Arc<Self> {
        Arc::new(Self {
            engine: ArcSwapOption::from(Some(Arc::new(initial))),
            current: Mutex::new(current),
            target: Mutex::new(None),
            switching: AtomicBool::new(false),
            errored: AtomicBool::new(false),
            inflight: Arc::new(AtomicUsize::new(0)),
            phase: AtomicU8::new(PHASE_IDLE),
            progress: AtomicU8::new(100),
            error: Mutex::new(None),
            builder,
        })
    }

    pub fn is_switching(&self) -> bool {
        self.switching.load(Ordering::SeqCst)
    }

    pub fn is_errored(&self) -> bool {
        self.errored.load(Ordering::SeqCst)
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
        // Entering a switch clears any prior error state.
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
        // 1. Drain in-flight (bounded ~30 s). The `switching` flag already 503s new ones.
        self.set_phase(PHASE_DRAINING);
        let drained = {
            let mut ok = false;
            for _ in 0..6000 {
                // ~30 s at 5 ms
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
                // new_engine: Arc<dyn Generator>; the field is ArcSwapOption<Arc<dyn Generator>>,
                // so we wrap it in another Arc to satisfy the double-Arc storage.
                self.engine.store(Some(Arc::new(new_engine)));
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
                        self.engine.store(Some(Arc::new(restored)));
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

/// Decrement-on-drop: used in the non-stream path to decrement inflight.
struct DecOnDrop(Arc<AtomicUsize>);
impl Drop for DecOnDrop {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A stream that holds a cloned `Arc<AtomicUsize>` and decrements inflight on drop,
/// covering the full lifetime of a streaming response.
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

#[async_trait::async_trait]
impl Generator for ModelManager {
    async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult> {
        if self.switching.load(Ordering::SeqCst) {
            anyhow::bail!("model switching");
        }
        self.inflight.fetch_add(1, Ordering::SeqCst);
        let _g = DecOnDrop(self.inflight.clone());
        // load_full returns Option<Arc<Arc<dyn Generator>>>; deref the outer Arc.
        let engine: Arc<dyn Generator> = Arc::clone(
            &*self.engine.load_full().ok_or_else(|| anyhow::anyhow!("model switching"))?,
        );
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
        let counter = self.inflight.clone();
        // load_full returns Option<Arc<Arc<dyn Generator>>>; deref the outer Arc.
        let engine: Arc<dyn Generator> = match self.engine.load_full() {
            Some(e) => Arc::clone(&*e),
            None => {
                counter.fetch_sub(1, Ordering::SeqCst);
                anyhow::bail!("model switching");
            }
        };
        match engine.generate_stream(req).await {
            Ok(inner) => Ok(Box::pin(CounterStream { inner, counter })),
            Err(e) => {
                counter.fetch_sub(1, Ordering::SeqCst);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::{ContentPart, FinishReason};
    use std::sync::atomic::AtomicUsize as TestAtomicUsize;

    fn marker_gen(tag: &'static str) -> Arc<dyn Generator> {
        Arc::new(crate::test_support::TaggedGen(tag))
    }

    fn spec(repo: &str, file: &str) -> ModelSpec {
        ModelSpec { repo: repo.into(), file: file.into() }
    }

    fn noop_builder() -> EngineBuilder {
        Box::new(|_spec| Box::pin(async { Ok(marker_gen("new")) }))
    }

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
