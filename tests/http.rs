// tests/http.rs — integration tests for HTTP handlers

/// Serialize all tests that read/write process-global env vars (LOCALLLM_*_BASE)
/// so they cannot race with each other and corrupt each other's cloud-target URL.
/// `static Mutex` is the lightest per-process lock available without extra crates.
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

    let _guard = ENV_LOCK.lock().await;
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

/// A body larger than 2 MB (axum's old default limit) must still reach the
/// routing layer and be forwarded to cloud when it overflows the local window.
/// This is a regression test for the DefaultBodyLimit raise: before the fix a
/// body >2 MB would be rejected with 413 before routing ran at all.
#[tokio::test]
async fn large_over_2mb_request_routes_to_cloud() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let _guard = ENV_LOCK.lock().await;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"routed":"cloud"}"#))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    // 3 MB of content — well above the 2 MB default limit axum would have
    // previously enforced, and ~750 000 est. tokens (far over the 1 000-token
    // test window), so routing must choose Cloud.
    let big = "x".repeat(3_000_000);
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

/// In-window request with a credential under a cascade profile (SaveTokens):
/// the local model returns a length-truncated (weak) result, so the router
/// escalates to the cloud upstream and returns the cloud response.
#[tokio::test]
async fn weak_local_with_cascade_escalates_to_cloud() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let _guard = ENV_LOCK.lock().await;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"routed":"cloud"}"#))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    // Small in-window prompt → decide() = LocalThenCascade (SaveTokens, creds present).
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGenWeak),
        localllm::route::Profile::SaveTokens.policy(),
        1000,
    );
    let resp =
        localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test")
            .await;
    assert_eq!(resp["routed"], "cloud");

    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// Trivial in-window request with a credential under MaxQuality (cascade=false,
/// but trivial score stays under the 0.2 threshold → plain Local). A weak local
/// result must NOT escalate; the local answer is returned as-is.
#[tokio::test]
async fn weak_local_without_cascade_stays_local() {
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGenWeak),
        localllm::route::Profile::MaxQuality.policy(),
        1000,
    );
    let resp =
        localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test")
            .await;
    // Anthropic maps FinishReason::Length → stop_reason "max_tokens"; the local
    // weak text is returned (no cloud escalation).
    assert_eq!(resp["content"][0]["text"], "local-weak-answer");
    assert_eq!(resp["stop_reason"], "max_tokens");
}

/// An in-window difficulty/cloud route whose upstream returns 429 degrades to the
/// local model instead of erroring (FakeGen tool_use proves local ran).
#[tokio::test]
async fn cloud_quota_degrades_to_local() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let _guard = ENV_LOCK.lock().await;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", server.uri());

    // MaxQuality + a modest in-window prompt → Cloud(Difficulty); 429 → degrade local.
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"a fairly involved question about routing"}],
        "tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGen),
        localllm::route::Profile::MaxQuality.policy(),
        1000,
    );
    let resp = localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test").await;
    assert_eq!(resp["stop_reason"], "tool_use"); // FakeGen → local served

    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

/// An over-window (context-overflow) cloud route whose upstream is unreachable
/// cannot fall back to local; it returns a clean 502.
#[tokio::test]
async fn overflow_cloud_offline_returns_clean_error() {
    let _guard = ENV_LOCK.lock().await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", "http://127.0.0.1:1");
    let big = "x".repeat(8000); // > 0.95 * 1000 window → ContextOverflow
    let body = format!(
        r#"{{"model":"claude","max_tokens":256,"messages":[{{"role":"user","content":"{big}"}}]}}"#
    );
    let status = localllm::axum_test_request_status_with_header(
        localllm::router_for_test_with(
            std::sync::Arc::new(localllm::FakeGen),
            localllm::route::Profile::SaveTokens.policy(),
            1000,
        ),
        "/v1/messages",
        &body,
        "x-api-key",
        "sk-test",
    )
    .await;
    assert_eq!(status, 502);
    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

// --- Model catalog endpoints (sub-project 2, task 3) ---

#[tokio::test]
async fn admin_models_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_get_status(app, "/admin/models").await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_models_returns_catalog_with_recommendation() {
    let app = localllm::router_for_test();
    let resp = localllm::axum_test_get_with_header(
        app, "/admin/models", "x-admin-token", "test-token",
    ).await;
    // grouped families, non-empty, with exactly one recommended across all
    let families = resp.as_array().expect("array of families");
    assert!(!families.is_empty());
    let rec_count: usize = families.iter()
        .flat_map(|f| f["models"].as_array().unwrap())
        .filter(|m| m["recommended"] == true)
        .count();
    assert_eq!(rec_count, 1);
}

#[tokio::test]
async fn admin_delete_rejects_bad_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_delete_status_with_header(
        app, "/admin/models", r#"{"repo":"r","file":"f"}"#, "x-admin-token", "wrong",
    ).await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_delete_in_use_model_is_409() {
    // router_for_test's ModelManager current spec is {repo:"test", file:"test"}.
    let app = localllm::router_for_test();
    let status = localllm::axum_test_delete_status_with_header(
        app, "/admin/models", r#"{"repo":"test","file":"test"}"#, "x-admin-token", "test-token",
    ).await;
    assert_eq!(status, 409);
}

/// A weak local cascade whose cloud escalation fails (offline) keeps the local
/// (truncated) answer instead of erroring.
#[tokio::test]
async fn cascade_cloud_offline_keeps_local_answer() {
    let _guard = ENV_LOCK.lock().await;
    std::env::set_var("LOCALLLM_ANTHROPIC_BASE", "http://127.0.0.1:1");
    let body = r#"{"model":"claude","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    let app = localllm::router_for_test_with(
        std::sync::Arc::new(localllm::FakeGenWeak),
        localllm::route::Profile::SaveTokens.policy(),
        1000,
    );
    let resp = localllm::axum_test_request_with_header(app, "/v1/messages", body, "x-api-key", "sk-test").await;
    assert_eq!(resp["content"][0]["text"], "local-weak-answer"); // local kept
    std::env::remove_var("LOCALLLM_ANTHROPIC_BASE");
}

// --- Hot-swap admin endpoints (sub-project 1) ---

#[tokio::test]
async fn admin_status_requires_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_get_status(app, "/admin/model/status").await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_status_ok_with_token() {
    let app = localllm::router_for_test();
    let resp = localllm::axum_test_get_with_header(
        app, "/admin/model/status", "x-admin-token", "test-token",
    ).await;
    assert_eq!(resp["state"], "ready");
}

#[tokio::test]
async fn admin_switch_accepts_with_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app, "/admin/model",
        r#"{"repo":"r2","file":"f2"}"#, "x-admin-token", "test-token",
    ).await;
    assert_eq!(status, 202);
}

#[tokio::test]
async fn admin_switch_rejects_bad_token() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app, "/admin/model",
        r#"{"repo":"r","file":"f"}"#, "x-admin-token", "wrong",
    ).await;
    assert_eq!(status, 401);
}

#[tokio::test]
async fn admin_switch_bad_body_is_400() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_request_status_with_header(
        app, "/admin/model",
        r#"{"repo":"r"}"#, "x-admin-token", "test-token",
    ).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn admin_delete_bad_body_is_400() {
    let app = localllm::router_for_test();
    let status = localllm::axum_test_delete_status_with_header(
        app, "/admin/models", r#"{"not":"valid"}"#, "x-admin-token", "test-token",
    ).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn admin_delete_path_traversal_file_is_400() {
    // A crafted file with path components must be rejected before touching the FS.
    let app = localllm::router_for_test();
    let status = localllm::axum_test_delete_status_with_header(
        app, "/admin/models",
        r#"{"repo":"x","file":"../../etc/passwd"}"#, "x-admin-token", "test-token",
    ).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn manager_page_served_no_auth() {
    let app = localllm::router_for_test();
    let (status, ctype, body) = localllm::axum_test_get_full(app, "/manager").await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("text/html"));
    assert!(body.contains("<html") || body.contains("<!doctype") || body.contains("<!DOCTYPE"));
}

#[tokio::test]
async fn manager_assets_served_with_types() {
    let app = localllm::router_for_test();
    let (s1, c1, _b1) = localllm::axum_test_get_full(app.clone(), "/manager/app.js").await;
    assert_eq!(s1, 200);
    assert!(c1.contains("javascript"));
    let (s2, c2, _b2) = localllm::axum_test_get_full(app, "/manager/style.css").await;
    assert_eq!(s2, 200);
    assert!(c2.contains("css"));
}

// --- OpenAI Responses API (sub-project 3, task 5) ---

#[tokio::test]
async fn responses_endpoint_returns_function_call() {
    let app = localllm::router_for_test();
    // FakeGen returns a tool call → output should carry a function_call item.
    let body = r#"{"model":"m","input":"weather?",
        "tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object"}}]}"#;
    let resp = localllm::axum_test_request(app, "/v1/responses", body).await;
    assert_eq!(resp["object"], "response");
    assert_eq!(resp["output"][0]["type"], "function_call");
    assert_eq!(resp["output"][0]["name"], "get_weather");
}

#[tokio::test]
async fn responses_unknown_role_returns_400() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"m","input":[{"type":"message","role":"frob","content":"hi"}]}"#;
    let status = localllm::axum_test_request_status(app, "/v1/responses", body).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn responses_stream_returns_event_sequence() {
    let app = localllm::router_for_test();
    let body = r#"{"model":"m","stream":true,"input":"hi"}"#;
    let raw = localllm::axum_test_request_raw(app, "/v1/responses", body).await;
    assert!(raw.contains("response.created"), "missing created: {raw}");
    assert!(raw.contains("response.completed"), "missing completed: {raw}");
    // no Chat-style DONE sentinel for Responses
    assert!(!raw.contains("[DONE]"), "responses stream must not emit [DONE]: {raw}");
}

#[tokio::test]
async fn responses_overflow_routes_to_cloud() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let _guard = ENV_LOCK.lock().await;
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"routed":"cloud"}"#))
        .mount(&server)
        .await;
    std::env::set_var("LOCALLLM_OPENAI_BASE", server.uri());

    let big = "x".repeat(8000); // overflow the tiny test window
    let body = format!(r#"{{"model":"m","input":"{big}"}}"#);
    let resp = localllm::axum_test_request_with_header(
        localllm::router_for_test(),
        "/v1/responses",
        &body,
        "authorization",
        "Bearer sk-test",
    )
    .await;
    assert_eq!(resp["routed"], "cloud");

    std::env::remove_var("LOCALLLM_OPENAI_BASE");
}
