//! Byte-faithful reverse-proxy to a cloud provider.
//!
//! Forwards the client's original request bytes and credential headers to the
//! upstream API and relays the response unchanged, so tool-calls and streaming
//! framing are preserved exactly. The credential is the client's own; we never
//! store or log it.

use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::response::Response;
use std::sync::OnceLock;

/// Estimated-completion cutoff (tokens ≈ bytes/4) below which a cloud reply is
/// flagged "cloud_trivial". Named so Fase D2 can tune it. Note the estimate is
/// inflated by JSON/SSE framing, biasing AGAINST false trivial flags.
pub const CLOUD_TRIVIAL_MAX_TOK: u64 = 40;

/// Attach to `forward` to record the relayed response as a cloud OutcomeEntry
/// (TTFT + estimated completion tokens) and emit "cloud_trivial" feedback.
pub struct RelayMeter {
    pub rid: String,
}

/// Body-stream wrapper that counts bytes and, when the upstream stream ends,
/// writes the outcome + feedback lines. If the client disconnects mid-stream
/// the wrapper is dropped without reaching the end → no outcome (accepted).
struct MeteredStream<S> {
    inner: S,
    rid: String,
    started: std::time::Instant,
    first_chunk_ms: Option<u64>,
    bytes: u64,
    done: bool,
}

impl<S, E> futures::Stream for MeteredStream<S>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = &mut *self;
        match std::pin::Pin::new(&mut this.inner).poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(chunk))) => {
                if this.first_chunk_ms.is_none() {
                    this.first_chunk_ms = Some(this.started.elapsed().as_millis() as u64);
                }
                this.bytes += chunk.len() as u64;
                std::task::Poll::Ready(Some(Ok(chunk)))
            }
            std::task::Poll::Ready(None) => {
                if !this.done {
                    this.done = true;
                    let est_tok = this.bytes / 4;
                    // Note: TTFT is measured from response-header receipt, not
                    // request start — the decision-to-first-byte gap is not
                    // captured. Adequate for dashboard latency comparison.
                    crate::route_log::append_outcome(&crate::route_log::OutcomeEntry {
                        rid: this.rid.clone(),
                        ts: crate::route_log::now_secs(),
                        completion_tok: Some(est_tok),
                        ttft_ms: this.first_chunk_ms,
                        gen_ms: Some(this.started.elapsed().as_millis() as u64),
                        cost_saved_usd: 0.0,
                    });
                    if est_tok < CLOUD_TRIVIAL_MAX_TOK {
                        crate::route_log::append_feedback(&crate::route_log::FeedbackEntry {
                            rid: this.rid.clone(),
                            ts: crate::route_log::now_secs(),
                            signal: "cloud_trivial".to_string(),
                        });
                    }
                }
                std::task::Poll::Ready(None)
            }
            other => other,
        }
    }
}

static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn shared_client() -> &'static reqwest::Client {
    CLIENT.get_or_init(reqwest::Client::new)
}

/// Cloud provider, inferred from the endpoint the client hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAI,
}

impl Provider {
    /// Upstream base URL. Overridable via env for tests.
    pub fn base_url(&self) -> String {
        match self {
            Provider::Anthropic => std::env::var("LOCALLLM_ANTHROPIC_BASE")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            Provider::OpenAI => std::env::var("LOCALLLM_OPENAI_BASE")
                .unwrap_or_else(|_| "https://api.openai.com".to_string()),
        }
    }
}

/// Hop-by-hop / connection headers that must not be forwarded verbatim;
/// reqwest sets correct values itself.
fn is_skipped_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host" | "content-length" | "connection" | "accept-encoding" | "transfer-encoding"
    )
}

/// Result of a reverse-proxy attempt.
pub enum ForwardOutcome {
    /// Upstream produced a client-relevant response (relay it verbatim).
    Relayed(Response),
    /// Upstream failed in a way that should degrade to local + notify.
    Degrade(crate::usage::DegradeReason),
}

/// A clean 502 to return to the client when degrade-to-local is impossible
/// (e.g. the prompt overflows the local window). Carries no credential/internal
/// detail beyond the reason category.
pub fn degrade_error(reason: crate::usage::DegradeReason) -> Response {
    Response::builder()
        .status(502)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            serde_json::json!({"error": reason.message()}).to_string(),
        ))
        .unwrap()
}

/// Reverse-proxy `body` to `provider`, forwarding the client's headers
/// (including its credential) unchanged. Returns `Relayed` with the streamed
/// response on success, or `Degrade(reason)` on a failure the caller should
/// handle by falling back to local. Credentials are never logged.
///
/// If `meter` is `Some`, the response body is wrapped in a `MeteredStream`
/// that records a cloud `OutcomeEntry` (TTFT, est completion = bytes/4) and
/// emits a `cloud_trivial` feedback line when the stream ends and est < 40 tok.
pub async fn forward(
    provider: Provider,
    upstream_path: &str,
    headers: &HeaderMap,
    body: Bytes,
    meter: Option<RelayMeter>,
) -> ForwardOutcome {
    use crate::usage::DegradeReason;
    let url = format!("{}{}", provider.base_url(), upstream_path);

    let mut fwd = HeaderMap::new();
    for (name, value) in headers.iter() {
        if !is_skipped_header(name.as_str()) {
            fwd.insert(name.clone(), value.clone());
        }
    }

    let upstream = match shared_client()
        .post(&url)
        .headers(fwd)
        .body(body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "localllm::req", "cloud forward transport error: {e}");
            return ForwardOutcome::Degrade(DegradeReason::Offline);
        }
    };

    let status = upstream.status();
    if let Some(reason) = match status.as_u16() {
        401 | 403 => Some(DegradeReason::Auth),
        429 => Some(DegradeReason::Quota),
        s if s >= 500 => Some(DegradeReason::ServerError),
        _ => None,
    } {
        tracing::warn!(target: "localllm::req", "cloud upstream status {status} → degrade {reason:?}");
        return ForwardOutcome::Degrade(reason);
    }

    let mut resp_headers = HeaderMap::new();
    for (name, value) in upstream.headers().iter() {
        if !is_skipped_header(name.as_str()) {
            resp_headers.insert(name.clone(), value.clone());
        }
    }
    let stream = upstream.bytes_stream();
    let mut builder = Response::builder().status(status);
    *builder.headers_mut().unwrap() = resp_headers;
    match meter {
        Some(m) => {
            let metered = MeteredStream {
                inner: stream,
                rid: m.rid,
                started: std::time::Instant::now(),
                first_chunk_ms: None,
                bytes: 0,
                done: false,
            };
            ForwardOutcome::Relayed(
                builder
                    .body(axum::body::Body::from_stream(metered))
                    .unwrap(),
            )
        }
        None => {
            ForwardOutcome::Relayed(builder.body(axum::body::Body::from_stream(stream)).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Serialize tests that touch process-global env vars (LOCALLLM_*_BASE) so
    /// concurrent test threads cannot interfere with each other's upstream URLs.
    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn forwards_body_and_relays_response() {
        let _guard = ENV_LOCK.lock().await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_string(r#"{"ok":true}"#),
            )
            .mount(&server)
            .await;

        // Point the Anthropic base URL at the mock server.
        std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-test".parse().unwrap());
        let body = Bytes::from_static(br#"{"model":"claude","messages":[]}"#);

        let outcome = forward(Provider::Anthropic, "/v1/messages", &headers, body, None).await;
        let resp = match outcome {
            ForwardOutcome::Relayed(r) => r,
            ForwardOutcome::Degrade(d) => panic!("expected relay, got degrade {d:?}"),
        };
        assert_eq!(resp.status(), 200);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], br#"{"ok":true}"#);

        std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
    }

    #[tokio::test]
    async fn unreachable_upstream_degrades_offline() {
        let _guard = ENV_LOCK.lock().await;
        std::env::set_var("LOCALLLM_OPENAI_BASE", "http://127.0.0.1:1"); // nothing listening
        let outcome = forward(
            Provider::OpenAI,
            "/v1/chat/completions",
            &HeaderMap::new(),
            Bytes::new(),
            None,
        )
        .await;
        assert!(matches!(
            outcome,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Offline)
        ));
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
    }

    async fn outcome_for_status(status: u16) -> ForwardOutcome {
        // Caller must hold ENV_LOCK before calling this to avoid racing on
        // LOCALLLM_OPENAI_BASE with other concurrent tests.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;
        std::env::set_var("LOCALLLM_OPENAI_BASE", server.uri());
        let out = forward(
            Provider::OpenAI,
            "/v1/chat/completions",
            &HeaderMap::new(),
            Bytes::new(),
            None,
        )
        .await;
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
        out
    }

    #[tokio::test]
    async fn metered_relay_records_outcome_and_trivial_flag() {
        // Acquire both locks in fixed order (route_log first) to avoid deadlock.
        let _rlog_guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _env_guard = ENV_LOCK.lock().await;

        let dir = std::env::temp_dir().join(format!("localllm-meter-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let log_file = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &log_file);

        let server = MockServer::start().await;
        // Tiny body: 20 bytes → est 5 tokens < 40 → trivial.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("01234567890123456789"))
            .mount(&server)
            .await;

        // Point Anthropic base at mock server (same pattern as existing tests).
        std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

        let out = forward(
            Provider::Anthropic,
            "/v1/messages",
            &HeaderMap::new(),
            Bytes::from("{}"),
            Some(RelayMeter { rid: "rm1".into() }),
        )
        .await;
        let resp = match out {
            ForwardOutcome::Relayed(r) => r,
            _ => panic!("expected relay"),
        };
        // Drain the body so the metered stream completes.
        let _ = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();

        let lines = crate::route_log::read_all();
        let outcome = lines.iter().find_map(|l| match l {
            crate::route_log::LogLine::Outcome(o) if o.rid == "rm1" => Some(o.clone()),
            _ => None,
        });
        let outcome = outcome.expect("expected a cloud outcome line");
        assert_eq!(outcome.completion_tok, Some(5)); // 20 bytes / 4
        assert!(outcome.ttft_ms.is_some());
        let trivial = lines.iter().any(|l| {
            matches!(l,
                crate::route_log::LogLine::Feedback(f) if f.rid == "rm1" && f.signal == "cloud_trivial")
        });
        assert!(trivial, "expected cloud_trivial feedback line");

        std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn status_401_degrades_auth() {
        let _guard = ENV_LOCK.lock().await;
        assert!(matches!(
            outcome_for_status(401).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Auth)
        ));
    }

    #[tokio::test]
    async fn status_429_degrades_quota() {
        let _guard = ENV_LOCK.lock().await;
        assert!(matches!(
            outcome_for_status(429).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Quota)
        ));
    }

    #[tokio::test]
    async fn status_500_degrades_server_error() {
        let _guard = ENV_LOCK.lock().await;
        assert!(matches!(
            outcome_for_status(500).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::ServerError)
        ));
    }
}
