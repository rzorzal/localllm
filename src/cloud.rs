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

    /// Upstream request path for this provider's chat endpoint.
    pub fn path(&self) -> &'static str {
        match self {
            Provider::Anthropic => "/v1/messages",
            Provider::OpenAI => "/v1/chat/completions",
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
pub async fn forward(provider: Provider, headers: &HeaderMap, body: Bytes) -> ForwardOutcome {
    use crate::usage::DegradeReason;
    let url = format!("{}{}", provider.base_url(), provider.path());

    let mut fwd = HeaderMap::new();
    for (name, value) in headers.iter() {
        if !is_skipped_header(name.as_str()) {
            fwd.insert(name.clone(), value.clone());
        }
    }

    let upstream = match shared_client().post(&url).headers(fwd).body(body).send().await {
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
    ForwardOutcome::Relayed(builder.body(axum::body::Body::from_stream(stream)).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn forwards_body_and_relays_response() {
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

        let outcome = forward(Provider::Anthropic, &headers, body).await;
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
        std::env::set_var("LOCALLLM_OPENAI_BASE", "http://127.0.0.1:1"); // nothing listening
        let outcome = forward(Provider::OpenAI, &HeaderMap::new(), Bytes::new()).await;
        assert!(matches!(
            outcome,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Offline)
        ));
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
    }

    async fn outcome_for_status(status: u16) -> ForwardOutcome {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;
        std::env::set_var("LOCALLLM_OPENAI_BASE", server.uri());
        let out = forward(Provider::OpenAI, &HeaderMap::new(), Bytes::new()).await;
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
        out
    }

    #[tokio::test]
    async fn status_401_degrades_auth() {
        assert!(matches!(
            outcome_for_status(401).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Auth)
        ));
    }

    #[tokio::test]
    async fn status_429_degrades_quota() {
        assert!(matches!(
            outcome_for_status(429).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::Quota)
        ));
    }

    #[tokio::test]
    async fn status_500_degrades_server_error() {
        assert!(matches!(
            outcome_for_status(500).await,
            ForwardOutcome::Degrade(crate::usage::DegradeReason::ServerError)
        ));
    }
}
