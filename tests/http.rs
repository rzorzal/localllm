// tests/http.rs — integration tests for Task 6/8 HTTP handlers

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

/// SSE streaming tests (Task 8) — use FakeGen which emits two text deltas + done.

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
    // Must contain the text content from FakeGen ("Hello" and " world")
    assert!(
        raw.contains("Hello") || raw.contains("world"),
        "expected streamed text content in SSE body, got: {raw}"
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
