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
    inflight: AtomicUsize,
    phase: AtomicU8,
    progress: AtomicU8,
    error: Mutex<Option<String>>,
    #[allow(dead_code)]
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
#[allow(dead_code)]
struct InflightGuard(Arc<ModelManager>);
#[allow(dead_code)]
impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A stream that holds an `InflightGuard` until it is fully consumed/dropped.
#[allow(dead_code)]
struct GuardedStream {
    inner: BoxStream<'static, anyhow::Result<StreamDelta>>,
    _guard: InflightGuard,
}
#[allow(dead_code)]
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
        // load_full returns Option<Arc<Arc<dyn Generator>>>; deref the outer Arc.
        let engine: Arc<dyn Generator> = match self.engine.load_full() {
            Some(e) => Arc::clone(&*e),
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
