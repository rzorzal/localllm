//! Byte-faithful reverse-proxy to a cloud provider.
//!
//! Forwards the client's original request bytes and credential headers to the
//! upstream API and relays the response unchanged, so tool-calls and streaming
//! framing are preserved exactly. The credential is the client's own; we never
//! store or log it.

use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::response::Response;

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

/// Reverse-proxy `body` to `provider`, forwarding the client's headers
/// (including its credential) unchanged, and relay the streamed response.
/// On any transport error, return HTTP 502 (graceful local fallback arrives in
/// Phase D).
pub async fn forward(provider: Provider, headers: &HeaderMap, body: Bytes) -> Response {
    let url = format!("{}{}", provider.base_url(), provider.path());

    let mut fwd = HeaderMap::new();
    for (name, value) in headers.iter() {
        if !is_skipped_header(name.as_str()) {
            fwd.insert(name.clone(), value.clone());
        }
    }

    let client = reqwest::Client::new();
    let upstream = match client.post(&url).headers(fwd).body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "localllm::req", "cloud forward error: {e}");
            return Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"error": format!("cloud forward failed: {e}")})
                        .to_string(),
                ))
                .unwrap();
        }
    };

    let status = upstream.status();
    let ctype = upstream
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| "application/json".parse().unwrap());

    let stream = upstream.bytes_stream();
    let mut builder = Response::builder().status(status);
    builder
        .headers_mut()
        .unwrap()
        .insert("content-type", ctype);
    builder
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn forwards_body_and_relays_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
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

        let resp = forward(Provider::Anthropic, &headers, body).await;
        assert_eq!(resp.status(), 200);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&bytes[..], br#"{"ok":true}"#);

        std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
    }

    #[tokio::test]
    async fn upstream_unreachable_returns_502() {
        std::env::set_var("LOCALLLM_OPENAI_BASE", "http://127.0.0.1:1"); // nothing listening
        let resp = forward(Provider::OpenAI, &HeaderMap::new(), Bytes::new()).await;
        assert_eq!(resp.status(), 502);
        std::env::remove_var("LOCALLLM_OPENAI_BASE");
    }
}
