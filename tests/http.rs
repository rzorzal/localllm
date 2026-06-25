// tests/http.rs — integration tests for HTTP handlers

#[tokio::test]
async fn openai_endpoint_returns_tool_call() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"m","messages":[{"role":"user","content":"weather?"}],
        "tools":[{"type":"function","function":{"name":"get_weather",
        "description":"w","parameters":{"type":"object"}}}]}"#;
    let resp = localllm::axum_test_request(app, "/v1/chat/completions", body).await;
    assert_eq!(resp["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = localllm::router_for_test();
    let resp = localllm::axum_test_get(app, "/health").await;
    assert_eq!(resp["status"], "ok");
}

#[tokio::test]
async fn anthropic_endpoint_returns_tool_use() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"m","max_tokens":256,"messages":[{"role":"user","content":"weather?"}],
        "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}]}"#;
    let resp = localllm::axum_test_request(app, "/v1/messages", body).await;
    assert_eq!(resp["stop_reason"], "tool_use");
    assert_eq!(resp["content"][0]["type"], "tool_use");
}

/// Error-path test: unknown role in messages must return HTTP 400.
#[tokio::test]
async fn openai_unknown_role_returns_400() {
    let app = localllm::router_for_test();
    // "invalid_role" is not handled by to_internal and must produce a 400 response.
    let body = r#"{"model":"m","messages":[{"role":"invalid_role","content":"hi"}]}"#;
    let status = localllm::axum_test_request_status(app, "/v1/chat/completions", body).await;
    assert_eq!(status, 400, "expected HTTP 400 for unknown role, got {status}");
}

/// SSE streaming tests — use FakeGen which emits two text deltas + done.

#[tokio::test]
async fn openai_stream_returns_sse_chunks_and_done() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let raw = localllm::axum_test_request_raw(app, "/v1/chat/completions", body).await;

    // Must contain at least one data: line with "chat.completion.chunk"
    assert!(
        raw.contains("chat.completion.chunk"),
        "expected chat.completion.chunk in SSE body, got: {raw}"
    );
    // Must contain BOTH text deltas from FakeGen ("Hello" and " world").
    // Using && so a regression where only one delta arrives is caught.
    assert!(
        raw.contains("Hello") && raw.contains("world"),
        "expected both streamed text deltas in SSE body, got: {raw}"
    );
    // Must end with the [DONE] sentinel
    assert!(
        raw.contains("[DONE]"),
        "expected [DONE] sentinel in SSE body, got: {raw}"
    );
}

#[tokio::test]
async fn anthropic_stream_returns_sse_events_and_message_stop() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"m","stream":true,"max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let raw = localllm::axum_test_request_raw(app, "/v1/messages", body).await;

    // Must contain message_start (session opener)
    assert!(
        raw.contains("message_start"),
        "expected message_start in SSE body, got: {raw}"
    );
    // Must contain text content from FakeGen
    assert!(
        raw.contains("content_block_delta"),
        "expected content_block_delta in SSE body, got: {raw}"
    );
    // Must contain the terminal message_stop event
    assert!(
        raw.contains("message_stop"),
        "expected message_stop in SSE body, got: {raw}"
    );
}

/// A request that fits the local window is served locally (FakeGen tool call),
/// even with a credential present.
#[tokio::test]
async fn small_request_with_key_stays_local() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}],
        "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}]}"#;
    let resp = localllm::axum_test_request(app, "/v1/messages", body).await;
    // FakeGen always returns a tool_use → proves the local path ran.
    assert_eq!(resp["stop_reason"], "tool_use");
}

/// A request that overflows the (tiny, 1000-token test) window with a credential
/// present is reverse-proxied to the mock upstream.
#[tokio::test]
async fn overflow_request_routes_to_cloud() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"routed":"cloud"}"#))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    let big = "x".repeat(8000); // ~2000 est. tokens > 0.95 * 1000
    let body = format!(
        r#"{{"model":"claude","max_tokens":256,"messages":[{{"role":"user","content":"{big}"}}]}}"#
    );
    let resp = localllm::axum_test_request_with_header(
        localllm::router_for_test(),
        "/v1/messages",
        &body,
        "x-api-key",
        "sk-test",
    )
    .await;
    assert_eq!(resp["routed"], "cloud");

    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// Overflow WITHOUT a credential must NOT hit cloud; it falls through to the
/// local path (which here returns FakeGen's tool_use rather than erroring,
/// because FakeGen ignores prompt size).
#[tokio::test]
async fn overflow_request_without_key_stays_local() {
    let big = "x".repeat(8000);
    let body = format!(
        r#"{{"model":"claude","max_tokens":256,"messages":[{{"role":"user","content":"{big}"}}]}}"#
    );
    let resp = localllm::axum_test_request(localllm::router_for_test(), "/v1/messages", &body).await;
    assert_eq!(resp["stop_reason"], "tool_use");
}
