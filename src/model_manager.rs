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
    #[serde(default)]
    pub quant: Option<String>,
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
pub type EngineBuilder =
    Box<dyn Fn(ModelSpec) -> BoxFuture<'static, anyhow::Result<Arc<dyn Generator>>> + Send + Sync>;

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
    pub fn new(
        initial: Arc<dyn Generator>,
        current: ModelSpec,
        builder: EngineBuilder,
    ) -> Arc<Self> {
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

    /// Build a manager with NO engine yet. The caller runs `try_initial_load`
    /// (typically after binding the server) to populate it. `switching` starts
    /// `false` so that `try_initial_load` can acquire the guard via
    /// `compare_exchange`; in-flight requests will 503 via the absent-engine
    /// check in `generate` until the engine is loaded.
    pub fn new_loading(current: ModelSpec, builder: EngineBuilder) -> Arc<Self> {
        Arc::new(Self {
            engine: ArcSwapOption::from(None),
            current: Mutex::new(current),
            target: Mutex::new(None),
            switching: AtomicBool::new(false),
            errored: AtomicBool::new(false),
            inflight: Arc::new(AtomicUsize::new(0)),
            phase: AtomicU8::new(PHASE_LOADING),
            progress: AtomicU8::new(0),
            error: Mutex::new(None),
            builder,
        })
    }

    /// True when a local engine is loaded and ready to serve.
    pub fn has_engine(&self) -> bool {
        self.engine.load_full().is_some()
    }

    /// Load the initial engine via the builder (no drain, no restore — there is
    /// no previous engine). Returns true on success. On failure, sets errored +
    /// error and leaves the engine absent. Clears `switching` either way.
    ///
    /// Returns `false` immediately (without touching `errored`/`error`/`engine`)
    /// if `switching` is already `true` — this means a concurrent `start_switch`
    /// has the guard and we must not race with it.
    pub async fn try_initial_load(self: &Arc<Self>, spec: ModelSpec) -> bool {
        // Acquire the switching guard. If already held (e.g. a concurrent
        // start_switch), treat this as contention, not a load error, and bail.
        if self
            .switching
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        self.errored.store(false, Ordering::SeqCst);
        *self.error.lock().unwrap() = None;
        self.set_phase(PHASE_LOADING);
        *self.target.lock().unwrap() = Some(spec.clone());
        let result = (self.builder)(spec.clone()).await;
        match result {
            Ok(engine) => {
                self.engine.store(Some(Arc::new(engine)));
                *self.current.lock().unwrap() = spec;
                *self.target.lock().unwrap() = None;
                self.progress.store(100, Ordering::SeqCst);
                self.set_phase(PHASE_IDLE);
                self.switching.store(false, Ordering::SeqCst);
                true
            }
            Err(e) => {
                *self.error.lock().unwrap() = Some(format!("model load failed: {e}"));
                self.errored.store(true, Ordering::SeqCst);
                self.set_phase(PHASE_IDLE);
                *self.target.lock().unwrap() = None;
                self.switching.store(false, Ordering::SeqCst);
                false
            }
        }
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

impl ModelManager {
    /// Number of tokens currently resident in the KV cache.
    /// Returns 0 if the engine is switching or unavailable (safe: server treats as fully cold).
    pub fn prefix_len(&self) -> usize {
        if self.switching.load(Ordering::SeqCst) {
            return 0;
        }
        match self.engine.load_full() {
            Some(e) => Arc::clone(&*e).prefix_len(),
            None => 0,
        }
    }

    /// Estimate how many tokens in `req`'s prompt are NOT currently in the KV cache.
    /// Returns `usize::MAX` if switching/unavailable (safe: escalates to cloud).
    pub async fn estimate_cold_tokens(&self, req: ChatRequest) -> usize {
        if self.switching.load(Ordering::SeqCst) {
            return usize::MAX;
        }
        match self.engine.load_full() {
            Some(e) => Arc::clone(&*e).estimate_cold_tokens(req).await,
            None => usize::MAX,
        }
    }

    /// Prefill the KV cache with `req`'s prompt without generating output.
    /// Returns `Ok(())` if switching/unavailable (no-op; safe).
    pub async fn prefill(&self, req: ChatRequest) -> anyhow::Result<()> {
        if self.switching.load(Ordering::SeqCst) {
            return Ok(());
        }
        match self.engine.load_full() {
            Some(e) => Arc::clone(&*e).prefill(req).await,
            None => Ok(()),
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
            &*self
                .engine
                .load_full()
                .ok_or_else(|| anyhow::anyhow!("model switching"))?,
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
        // Guard decrements on any early return OR a panic in the await below.
        // On success we `forget` it and hand the count to the CounterStream,
        // which decrements for the stream's whole lifetime (no double-decrement).
        let guard = DecOnDrop(counter.clone());
        // load_full returns Option<Arc<Arc<dyn Generator>>>; deref the outer Arc.
        let engine: Arc<dyn Generator> = match self.engine.load_full() {
            Some(e) => Arc::clone(&*e),
            None => anyhow::bail!("model switching"), // guard drops → decrement
        };
        match engine.generate_stream(req).await {
            Ok(inner) => {
                std::mem::forget(guard); // CounterStream now owns the decrement
                Ok(Box::pin(CounterStream { inner, counter }))
            }
            Err(e) => Err(e), // guard drops → decrement
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
        ModelSpec {
            repo: repo.into(),
            file: file.into(),
            quant: None,
        }
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
            if !m.is_switching() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        panic!("switch did not finish");
    }

    #[tokio::test]
    async fn happy_switch_swaps_engine_and_updates_current() {
        let calls = Arc::new(TestAtomicUsize::new(0));
        let m = ModelManager::new(
            marker_gen("orig"),
            spec("r", "old"),
            counting_builder(calls.clone(), None),
        );
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
                if let Some(rx) = rx.lock().await.take() {
                    let _ = rx.await;
                }
                Ok(marker_gen("new"))
            })
        });
        let m = ModelManager::new(marker_gen("orig"), spec("r", "old"), builder);
        m.start_switch(spec("r2", "a")).unwrap();
        // give the task a moment to flip `switching`
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(matches!(
            m.start_switch(spec("r3", "b")),
            Err(SwitchError::AlreadySwitching)
        ));
        assert!(m.is_switching());
        let _ = tx.send(());
        wait_ready(&m).await;
    }

    #[tokio::test]
    async fn failed_build_restores_previous_model() {
        // target repo "bad" fails; restore rebuilds "r"/"old" (succeeds).
        let calls = Arc::new(TestAtomicUsize::new(0));
        let m = ModelManager::new(
            marker_gen("orig"),
            spec("r", "old"),
            counting_builder(calls.clone(), Some("bad")),
        );
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
        ChatRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
            model: "m".into(),
        }
    }

    /// A generator whose `generate` blocks until a oneshot is fired — lets a test
    /// hold a request in-flight to exercise the switch drain.
    struct BlockingGen(tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>);
    #[async_trait::async_trait]
    impl Generator for BlockingGen {
        async fn generate(&self, _req: ChatRequest) -> anyhow::Result<ChatResult> {
            if let Some(rx) = self.0.lock().await.take() {
                let _ = rx.await;
            }
            Ok(ChatResult {
                content: vec![ContentPart::Text("blocked".into())],
                finish_reason: FinishReason::Stop,
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }
        async fn generate_stream(
            &self,
            _req: ChatRequest,
        ) -> anyhow::Result<BoxStream<'static, anyhow::Result<StreamDelta>>> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    /// Spec: while a switch is in progress, requests are rejected — the source of
    /// the HTTP 503 gate.
    #[tokio::test]
    async fn requests_bail_while_switching() {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let builder: EngineBuilder = Box::new(move |_spec| {
            let rx = rx.clone();
            Box::pin(async move {
                if let Some(rx) = rx.lock().await.take() {
                    let _ = rx.await;
                }
                Ok(marker_gen("new"))
            })
        });
        let m = ModelManager::new(marker_gen("orig"), spec("r", "old"), builder);
        m.start_switch(spec("r2", "new")).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(m.is_switching());
        let err = m.generate(req()).await.unwrap_err();
        assert!(err.to_string().contains("model switching"), "got: {err}");
        let _ = tx.send(());
        wait_ready(&m).await;
    }

    /// Spec: an in-flight request delays the switch (drain) until it completes.
    #[tokio::test]
    async fn drain_waits_for_inflight_then_switches() {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let blocking =
            Arc::new(BlockingGen(tokio::sync::Mutex::new(Some(rx)))) as Arc<dyn Generator>;
        let m = ModelManager::new(blocking, spec("r", "old"), noop_builder());
        // Hold a request in-flight.
        let m2 = m.clone();
        let handle = tokio::spawn(async move {
            let _ = m2.generate(req()).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        // Start a switch — it must wait (drain) while the request is in flight.
        m.start_switch(spec("r2", "new")).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(m.is_switching(), "switch should still be draining");
        assert_eq!(m.status().phase, SwitchPhase::Draining);
        // Release the in-flight request → drain completes → switch proceeds.
        let _ = tx.send(());
        handle.await.unwrap();
        wait_ready(&m).await;
        assert_eq!(m.status().current, spec("r2", "new"));
    }

    #[tokio::test]
    async fn delegates_to_current_engine() {
        let m = ModelManager::new(marker_gen("orig"), spec("r", "f"), noop_builder());
        let out = m
            .generate(ChatRequest {
                messages: vec![],
                tools: vec![],
                max_tokens: None,
                temperature: None,
                stream: false,
                model: "m".into(),
            })
            .await
            .unwrap();
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

    #[test]
    fn model_spec_quant_defaults_and_round_trips() {
        // old JSON without "quant" → None
        let old: ModelSpec = serde_json::from_str(r#"{"repo":"r","file":"f"}"#).unwrap();
        assert_eq!(old.quant, None);
        // round-trip with quant
        let s = ModelSpec {
            repo: "r".into(),
            file: "f".into(),
            quant: Some("Q6_K".into()),
        };
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<ModelSpec>(&j).unwrap(), s);
    }

    #[tokio::test]
    async fn new_loading_has_no_engine_until_initial_load() {
        let calls = Arc::new(TestAtomicUsize::new(0));
        let m = ModelManager::new_loading(spec("r", "m"), counting_builder(calls.clone(), None));
        assert!(!m.has_engine());
        // new_loading starts with switching=false so try_initial_load can acquire
        // the guard. The engine is absent until try_initial_load succeeds.
        assert!(!m.is_switching());
        let ok = m.try_initial_load(spec("r", "m")).await;
        assert!(ok);
        assert!(m.has_engine());
        assert_eq!(m.status().state, "ready");
        let out = m.generate(req()).await.unwrap();
        assert!(matches!(&out.content[0], ContentPart::Text(_)));
    }

    #[tokio::test]
    async fn initial_load_failure_leaves_no_engine_and_errored() {
        let builder: EngineBuilder = Box::new(|_spec| Box::pin(async { Err(anyhow::anyhow!("boom")) }));
        let m = ModelManager::new_loading(spec("bad", "x"), builder);
        let ok = m.try_initial_load(spec("bad", "x")).await;
        assert!(!ok);
        assert!(!m.has_engine());
        assert!(m.is_errored());
        assert!(m.status().error.unwrap().contains("boom"));
    }

    /// Race guard: if `switching` is already true when `try_initial_load` is
    /// called (simulating a concurrent `start_switch`), it must return `false`
    /// WITHOUT setting `errored`, setting `error`, or touching `engine`/`current`.
    ///
    /// Strategy: use `ModelManager::new` (has an engine, switching=false) and
    /// call `start_switch` with a blocking builder to hold `switching=true`, then
    /// call `try_initial_load` while the switch is in progress and assert it bails.
    #[tokio::test]
    async fn try_initial_load_is_no_op_when_switching_already_held() {
        let calls = Arc::new(TestAtomicUsize::new(0));
        // Use a blocking builder so start_switch holds switching=true indefinitely.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let rx = Arc::new(tokio::sync::Mutex::new(Some(rx)));
        let calls2 = calls.clone();
        let builder: EngineBuilder = Box::new(move |_spec| {
            let rx = rx.clone();
            let calls2 = calls2.clone();
            Box::pin(async move {
                calls2.fetch_add(1, Ordering::SeqCst);
                if let Some(rx) = rx.lock().await.take() {
                    let _ = rx.await;
                }
                Ok(marker_gen("switched"))
            })
        });
        let m = ModelManager::new(marker_gen("orig"), spec("r", "old"), builder);
        // Hold switching=true via start_switch.
        m.start_switch(spec("r2", "new")).unwrap();
        // Give the switch task a moment to flip switching.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(m.is_switching(), "start_switch must hold switching=true");

        // try_initial_load must bail without touching errored/error/engine.
        let ok = m.try_initial_load(spec("r", "old")).await;

        // Must return false (contention, not an error).
        assert!(!ok, "should return false when switching is already held");
        // Must NOT set errored (this is a contention bail-out, not a load failure).
        assert!(!m.is_errored(), "errored must remain false on contention bail");
        // The key invariant: builder was NOT called by try_initial_load.
        // start_switch may have already called the builder once (during drain/build),
        // but try_initial_load must not add an extra call.
        let builder_calls_after_bail = calls.load(Ordering::SeqCst);
        // Release the switch so the test can exit cleanly.
        let _ = tx.send(());
        wait_ready(&m).await;
        // After switch completes, exactly one extra builder call (from the switch itself).
        // try_initial_load must not have added another.
        assert_eq!(
            calls.load(Ordering::SeqCst),
            builder_calls_after_bail,
            "builder must not be invoked by try_initial_load on contention bail"
        );
    }
}
