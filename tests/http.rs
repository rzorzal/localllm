// tests/http.rs — integration tests for Task 6 HTTP handlers

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
