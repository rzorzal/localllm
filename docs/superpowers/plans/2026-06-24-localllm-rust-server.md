# localllm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single Rust binary that runs Qwen2.5-7B-Instruct (GGUF Q4_K_M) locally via mistralrs and serves it over HTTP on localhost, speaking both the OpenAI Chat Completions API (for Codex) and the Anthropic Messages API (for Claude Code), with working tool calling/chaining.

**Architecture:** axum HTTP server. Pure-function translation layer converts OpenAI and Anthropic request/response shapes to/from an internal representation. An `engine` module wraps the mistralrs `Model` and is the only code that touches inference. HTTP handlers never see mistralrs types; the engine never sees HTTP types. They meet at the internal types in `api/common.rs`.

**Tech Stack:** Rust 1.89, mistralrs 0.8.x (feature `metal`), axum 0.7, tokio, serde/serde_json, anyhow, clap, tracing.

## Global Constraints

- Target: Apple M1 Pro, 16 GB unified RAM, macOS 26.5. Metal acceleration required.
- Bind only to `127.0.0.1`. Never `0.0.0.0`.
- Default model: `Qwen/Qwen2.5-7B-Instruct-GGUF`, quant Q4_K_M, auto-download from HuggingFace.
- Default port: `8080`. Default context length: `16384`.
- mistralrs exact API identifiers (`GgufModelBuilder`, `with_isq`/`with_auto_isq`, `IsqType::Q4K`/`IsqBits::Four`, `with_paged_attn`, request/response types) MUST be reconciled against the installed crate's `cargo doc` and `mistralrs/examples/` — Task 1 locks the exact names; later tasks use whatever Task 1 records.
- Any optimization (KV-cache quant, prefix caching) not exposed by the installed mistralrs version is documented as unavailable in code comments + README — never faked.
- `engine` module must not import `axum`; `api/*` modules must not import `mistralrs`.

---

### Task 1: Scaffold + lock mistralrs API

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs` (temporary spike)
- Create: `NOTES-mistralrs-api.md` (records verified identifiers)

**Interfaces:**
- Produces: `NOTES-mistralrs-api.md` documenting the exact builder type for GGUF, the ISQ method+enum, the paged-attn method, the chat request type, the response type, the tool-call type, and the streaming chunk type. Every later task reads this file before using mistralrs.

- [ ] **Step 1: Write `Cargo.toml`**

```toml
[package]
name = "localllm"
version = "0.1.0"
edition = "2021"

[dependencies]
mistralrs = { version = "0.8", features = ["metal"] }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
thiserror = "1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
futures = "0.3"
uuid = { version = "1", features = ["v4"] }

[profile.release]
lto = true
codegen-units = 1
```

- [ ] **Step 2: Generate and read the crate docs**

Run: `cargo doc -p mistralrs --no-deps` then open `target/doc/mistralrs/index.html`, and clone-inspect examples:
`cargo add mistralrs --features metal` already pulls source into `~/.cargo`. Inspect `~/.cargo/registry/src/*/mistralrs-*/examples/` (especially any `tools/` example) and `lib.rs`.

Expected: confirm real names. As of 0.8.x the likely-correct shapes are: `GgufModelBuilder::new(model_id, vec![gguf_files])`, `.with_isq(IsqType::Q4K)`, `.with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?`, `.with_logging()`, `.build().await?` → `Model`; requests via `RequestBuilder` + `TextMessages`/`TextMessageRole`; `Model::send_chat_request(req).await?` → `ChatCompletionResponse`; tools via `RequestBuilder::set_tools(Vec<Tool>)`; tool calls in `response.choices[0].message.tool_calls: Vec<ToolCallResponse>`. **Verify each; do not trust this list blindly.**

- [ ] **Step 3: Write a minimal spike in `src/main.rs`**

```rust
use mistralrs::{GgufModelBuilder, IsqType, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = GgufModelBuilder::new(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        vec!["qwen2.5-7b-instruct-q4_k_m.gguf".to_string()],
    )
    .with_isq(IsqType::Q4K)
    .with_logging()
    .build()
    .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "Reply with the single word: ok");
    let resp = model.send_chat_request(messages).await?;
    println!("{}", resp.choices[0].message.content.clone().unwrap_or_default());
    Ok(())
}
```

- [ ] **Step 4: Build and run the spike**

Run: `cargo run --release`
Expected: model downloads on first run (multi-GB, slow), then prints `ok` (or close). If identifiers are wrong, fix them against the docs from Step 2 and rerun. **The build must compile and produce a model response before moving on.**

- [ ] **Step 5: Record verified API in `NOTES-mistralrs-api.md`**

Write down the exact, compiling identifiers used (builder, ISQ method+enum value, paged-attn call, request type, response field path for content and for `tool_calls`, streaming chunk type, and whether `with_paged_attn` / KV-cache-quant / prefix-caching options exist). Note any optimization the version does NOT support.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/main.rs NOTES-mistralrs-api.md
git commit -m "chore: scaffold project and lock mistralrs 0.8 API"
```

---

### Task 2: Internal request/response types

**Files:**
- Create: `src/api/mod.rs`
- Create: `src/api/common.rs`
- Test: inline `#[cfg(test)]` in `src/api/common.rs`

**Interfaces:**
- Produces:
  - `Role` enum: `System | User | Assistant | Tool`.
  - `ToolCall { id: String, name: String, arguments: String }` (arguments = JSON string).
  - `ToolResult { tool_call_id: String, content: String }`.
  - `ContentPart` for assistant output: `Text(String)` | `Call(ToolCall)`.
  - `ChatMessage { role: Role, text: Option<String>, tool_calls: Vec<ToolCall>, tool_result: Option<ToolResult> }`.
  - `ToolSpec { name: String, description: String, parameters: serde_json::Value }` (parameters = JSON schema).
  - `ChatRequest { messages: Vec<ChatMessage>, tools: Vec<ToolSpec>, max_tokens: Option<usize>, temperature: Option<f64>, stream: bool, model: String }`.
  - `ChatResult { content: Vec<ContentPart>, finish_reason: FinishReason, prompt_tokens: usize, completion_tokens: usize }`.
  - `FinishReason` enum: `Stop | Length | ToolCalls`.

- [ ] **Step 1: Write a failing test for round-tripping a tool call through the internal types**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_call_holds_json_arguments() {
        let c = ToolCall { id: "c1".into(), name: "get_weather".into(),
            arguments: r#"{"location":"Recife"}"#.into() };
        let v: serde_json::Value = serde_json::from_str(&c.arguments).unwrap();
        assert_eq!(v["location"], "Recife");
    }
    #[test]
    fn finish_reason_tool_calls_is_distinct() {
        assert_ne!(FinishReason::ToolCalls, FinishReason::Stop);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib api::common`
Expected: FAIL — types not defined / module not declared.

- [ ] **Step 3: Implement the types**

In `src/api/mod.rs`:
```rust
pub mod common;
pub mod openai;
pub mod anthropic;
```
In `src/api/common.rs` define every type listed in Interfaces with `#[derive(Debug, Clone, PartialEq)]` (add `serde` derives only where serialized later; these internal types need none yet). Make `FinishReason` and `Role` `#[derive(... , PartialEq, Eq)]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib api::common`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/
git commit -m "feat: internal chat request/response types"
```

---

### Task 3: OpenAI translation layer

**Files:**
- Create: `src/api/openai.rs`
- Test: inline `#[cfg(test)]` in `src/api/openai.rs`

**Interfaces:**
- Consumes: all types from `api/common.rs` (Task 2).
- Produces:
  - serde types: `OaiChatRequest`, `OaiMessage`, `OaiTool`, `OaiToolCall`, `OaiChatResponse`, `OaiChoice`, `OaiResponseMessage`, `OaiUsage`, `OaiModelList`.
  - `fn to_internal(req: OaiChatRequest) -> Result<ChatRequest, String>`.
  - `fn from_internal(res: ChatResult, model: &str) -> OaiChatResponse`.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::*;

    #[test]
    fn parses_user_message_and_tool() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"get_weather",
            "description":"w","parameters":{"type":"object"}}}]}"#;
        let req: OaiChatRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages.len(), 1);
        assert_eq!(internal.messages[0].role, Role::User);
        assert_eq!(internal.tools[0].name, "get_weather");
    }

    #[test]
    fn renders_tool_call_response() {
        let res = ChatResult {
            content: vec![ContentPart::Call(ToolCall{ id:"c1".into(),
                name:"get_weather".into(), arguments:"{}".into()})],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 5, completion_tokens: 2 };
        let oai = from_internal(res, "m");
        assert_eq!(oai.choices[0].finish_reason, "tool_calls");
        assert_eq!(oai.choices[0].message.tool_calls.as_ref().unwrap()[0].function.name,
            "get_weather");
    }

    #[test]
    fn parses_tool_result_message() {
        let json = r#"{"model":"m","messages":[
            {"role":"tool","tool_call_id":"c1","content":"sunny"}]}"#;
        let req: OaiChatRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::Tool);
        assert_eq!(internal.messages[0].tool_result.as_ref().unwrap().content, "sunny");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib api::openai`
Expected: FAIL — types/functions undefined.

- [ ] **Step 3: Implement OpenAI types + `to_internal` + `from_internal`**

Define serde structs matching the OpenAI Chat Completions schema. `content` may be string or null; assistant messages may carry `tool_calls`; tool messages carry `tool_call_id` + `content`. In `from_internal`, emit `id: "chatcmpl-<uuid>"`, `object: "chat.completion"`, `created` epoch secs, `choices[0].finish_reason` mapped from `FinishReason` (`Stop→"stop"`, `Length→"length"`, `ToolCalls→"tool_calls"`), and `usage`. Map `ContentPart::Call` → `OaiToolCall { id, type:"function", function:{name, arguments} }`, `ContentPart::Text` → `message.content`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib api::openai`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/openai.rs
git commit -m "feat: OpenAI chat completions translation layer"
```

---

### Task 4: Anthropic translation layer

**Files:**
- Create: `src/api/anthropic.rs`
- Test: inline `#[cfg(test)]` in `src/api/anthropic.rs`

**Interfaces:**
- Consumes: all types from `api/common.rs` (Task 2).
- Produces:
  - serde types: `AnthRequest`, `AnthMessage`, `AnthContentBlock` (tagged enum: `text`, `tool_use`, `tool_result`), `AnthTool`, `AnthResponse`.
  - `fn to_internal(req: AnthRequest) -> Result<ChatRequest, String>` (hoists top-level `system` field into a `Role::System` message).
  - `fn from_internal(res: ChatResult, model: &str) -> AnthResponse`.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::*;

    #[test]
    fn hoists_system_and_parses_tool() {
        let json = r#"{"model":"m","max_tokens":256,"system":"be terse",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"name":"get_weather","description":"w",
            "input_schema":{"type":"object"}}]}"#;
        let req: AnthRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::System);
        assert_eq!(internal.messages[0].text.as_deref(), Some("be terse"));
        assert_eq!(internal.tools[0].name, "get_weather");
    }

    #[test]
    fn renders_tool_use_block() {
        let res = ChatResult {
            content: vec![ContentPart::Call(ToolCall{ id:"tu1".into(),
                name:"get_weather".into(), arguments:r#"{"location":"X"}"#.into()})],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 4, completion_tokens: 3 };
        let a = from_internal(res, "m");
        assert_eq!(a.stop_reason, "tool_use");
        match &a.content[0] {
            AnthContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "X");
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn parses_tool_result_block() {
        let json = r#"{"model":"m","max_tokens":10,"messages":[
            {"role":"user","content":[
              {"type":"tool_result","tool_use_id":"tu1","content":"sunny"}]}]}"#;
        let req: AnthRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::Tool);
        assert_eq!(internal.messages[0].tool_result.as_ref().unwrap().content, "sunny");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib api::anthropic`
Expected: FAIL — types/functions undefined.

- [ ] **Step 3: Implement Anthropic types + translation**

`AnthMessage.content` is either a string or an array of `AnthContentBlock` (use `#[serde(untagged)]` helper or a custom deserialize). `AnthContentBlock` is `#[serde(tag = "type", rename_all = "snake_case")]` with variants `Text { text }`, `ToolUse { id, name, input }`, `ToolResult { tool_use_id, content }`. In `to_internal`, a `tool_result` block becomes a `Role::Tool` message (map `tool_use_id` → `ToolResult.tool_call_id`). In `from_internal`, emit `id: "msg_<uuid>"`, `type:"message"`, `role:"assistant"`, `stop_reason` (`Stop→"end_turn"`, `Length→"max_tokens"`, `ToolCalls→"tool_use"`), `usage:{input_tokens, output_tokens}`. `ContentPart::Call` → `ToolUse { id, name, input: parse(arguments) }`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib api::anthropic`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/anthropic.rs
git commit -m "feat: Anthropic messages translation layer"
```

---

### Task 5: Engine — model load + generation + tool wiring

**Files:**
- Create: `src/engine.rs`
- Modify: `src/api/mod.rs` (no change needed; engine is top-level)
- Test: manual run (model inference is not a cheap unit test)

**Interfaces:**
- Consumes: `ChatRequest`, `ChatResult`, `ContentPart`, `ToolSpec`, `FinishReason`, `Role`, `ToolCall` from `api/common.rs`; verified mistralrs names from `NOTES-mistralrs-api.md`.
- Produces:
  - `struct Engine { model: mistralrs::Model }`.
  - `async fn Engine::load(cfg: &Config) -> anyhow::Result<Engine>`.
  - `async fn Engine::generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult>`.
  - `fn Engine::supports_kv_quant() -> bool` etc. as needed for honest reporting.
- Note: `Config` is defined in Task 7; for this task, give `load` the concrete fields it needs (`model_id: &str`, `gguf_file: &str`, `isq`, `ctx_len`, `paged_attn: bool`) via a small `EngineConfig` struct defined here, and have Task 7's `Config` build it.

- [ ] **Step 1: Define `EngineConfig` and `Engine::load`**

```rust
pub struct EngineConfig {
    pub model_id: String,
    pub gguf_file: String,
    pub ctx_len: usize,
    pub paged_attn: bool,
}
```
Implement `load` using the verified builder from Task 1. Apply ALL available optimizations: `.with_isq(IsqType::Q4K)`, `.with_paged_attn(...)` when `paged_attn` and the API exists, context length, `.with_logging()`. For each optimization the installed version lacks, add a `tracing::warn!` line and a code comment citing NOTES — do not fake it.

- [ ] **Step 2: Implement `generate`**

Translate `ChatRequest` → mistralrs `RequestBuilder`/`TextMessages`: map `Role` (Tool → the version's tool-result message role per NOTES), attach tools via the verified `set_tools` equivalent, set `max_tokens`/`temperature`. Call `send_chat_request`. Build `ChatResult`: if `message.tool_calls` non-empty → `content = Call(..)` each, `finish_reason = ToolCalls`; else `content = Text(message.content)`, map finish reason from the response. Fill token counts from the response usage.

- [ ] **Step 3: Add a temporary `main` smoke path**

Temporarily wire `src/main.rs` to: load the engine, send a `ChatRequest` with a `get_weather` ToolSpec and user message "What's the weather in Recife? Use the tool.", print whether a tool call came back.

Run: `cargo run --release`
Expected: prints a parsed tool call for `get_weather`. This proves end-to-end model→tool parsing before HTTP exists.

- [ ] **Step 4: Commit**

```bash
git add src/engine.rs src/main.rs NOTES-mistralrs-api.md
git commit -m "feat: mistralrs engine with optimizations and tool parsing"
```

---

### Task 6: HTTP handlers (non-streaming) + router

**Files:**
- Create: `src/server.rs`
- Modify: `src/api/openai.rs` (add `/v1/models` payload builder if not present)
- Test: `tests/http.rs` (integration test with a mock engine trait)

**Interfaces:**
- Consumes: `to_internal`/`from_internal` from both api modules; `Engine::generate`.
- Produces:
  - `trait Generator { async fn generate(&self, req: ChatRequest) -> anyhow::Result<ChatResult>; }` implemented by `Engine` (lets tests inject a fake).
  - `fn router(state: Arc<dyn Generator>) -> axum::Router`.
  - Handlers: `POST /v1/chat/completions`, `POST /v1/messages`, `GET /v1/models`, `GET /health`.

- [ ] **Step 1: Write a failing integration test with a fake generator**

```rust
// tests/http.rs
use localllm::{router_for_test, FakeGen};
// router_for_test builds the router with a FakeGen returning a fixed tool call.
#[tokio::test]
async fn openai_endpoint_returns_tool_call() {
    let app = router_for_test();
    let body = r#"{"model":"m","messages":[{"role":"user","content":"weather?"}],
        "tools":[{"type":"function","function":{"name":"get_weather",
        "description":"w","parameters":{"type":"object"}}}]}"#;
    let resp = axum_test_request(app, "/v1/chat/completions", body).await;
    assert_eq!(resp["choices"][0]["finish_reason"], "tool_calls");
}
```
(Expose `router_for_test`, `FakeGen`, and a small `axum_test_request` helper from `src/lib.rs`; `FakeGen::generate` returns a `ChatResult` with one `get_weather` `Call`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test http`
Expected: FAIL — `router_for_test`/symbols undefined.

- [ ] **Step 3: Add `src/lib.rs` exposing modules + test helpers; implement handlers**

Create `src/lib.rs` declaring `pub mod api; pub mod engine; pub mod server; pub mod config;` and the `Generator` trait + `FakeGen` (behind `#[cfg(any(test, feature="test-util"))]` or always-public for simplicity). Implement each handler: deserialize body → `to_internal` (400 on error) → `generate` (500 on error) → `from_internal` → JSON. `/health` → `{"status":"ok"}`. `/v1/models` → list containing the configured model id.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test http`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server.rs src/lib.rs src/api/openai.rs tests/http.rs
git commit -m "feat: HTTP router and non-streaming handlers for both APIs"
```

---

### Task 7: Config + CLI + real main wiring

**Files:**
- Create: `src/config.rs`
- Modify: `src/main.rs` (replace spike with real server boot)

**Interfaces:**
- Consumes: `EngineConfig` (Task 5), `router` (Task 6), `Engine::load`.
- Produces:
  - `struct Config` (clap `Parser`): `--port u16 [8080]`, `--model-id String [Qwen/Qwen2.5-7B-Instruct-GGUF]`, `--gguf-file String [qwen2.5-7b-instruct-q4_k_m.gguf]`, `--ctx-len usize [16384]`, `--no-paged-attn bool flag`.
  - `fn Config::engine_config(&self) -> EngineConfig`.

- [ ] **Step 1: Write a failing test for CLI defaults**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn defaults_are_localhost_8080_qwen() {
        let c = Config::parse_from(["localllm"]);
        assert_eq!(c.port, 8080);
        assert_eq!(c.ctx_len, 16384);
        assert!(c.model_id.contains("Qwen2.5-7B-Instruct"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config`
Expected: FAIL — `Config` undefined.

- [ ] **Step 3: Implement `Config` with clap derive + `engine_config`**

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib config`
Expected: PASS.

- [ ] **Step 5: Rewrite `src/main.rs`**

```rust
use std::sync::Arc;
use clap::Parser;
use localllm::{config::Config, engine::Engine, server::router};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("localllm=info".parse()?)).init();
    let cfg = Config::parse();
    tracing::info!("loading model {} ...", cfg.model_id);
    let engine = Arc::new(Engine::load(&cfg.engine_config()).await?);
    let app = router(engine);
    let addr = std::net::SocketAddr::from(([127,0,0,1], cfg.port));
    tracing::info!("listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

- [ ] **Step 6: Build and boot**

Run: `cargo run --release` then in another shell `curl -s localhost:8080/health`
Expected: model loads; `{"status":"ok"}`.

- [ ] **Step 7: Commit**

```bash
git add src/config.rs src/main.rs
git commit -m "feat: CLI config and real server boot"
```

---

### Task 8: Streaming (SSE) for both APIs

**Files:**
- Modify: `src/engine.rs` (add `generate_stream`)
- Modify: `src/server.rs` (branch on `stream: true`)
- Modify: `src/api/openai.rs`, `src/api/anthropic.rs` (chunk renderers)
- Test: inline tests for chunk renderers (pure functions)

**Interfaces:**
- Consumes: verified streaming chunk type from NOTES; `ChatRequest`.
- Produces:
  - `async fn Engine::generate_stream(&self, req) -> impl Stream<Item=anyhow::Result<StreamDelta>>` where `StreamDelta { text: Option<String>, done: bool, finish_reason: Option<FinishReason> }`.
  - `fn openai::stream_chunk(delta: &StreamDelta, id: &str, model: &str) -> String` (one `data: {..}` line).
  - `fn anthropic::stream_events(delta: &StreamDelta, ...) -> Vec<String>` (event lines).

- [ ] **Step 1: Write failing tests for chunk renderers**

```rust
// in openai.rs tests
#[test]
fn openai_chunk_has_delta_content() {
    let d = StreamDelta{ text: Some("hi".into()), done:false, finish_reason:None };
    let line = stream_chunk(&d, "chatcmpl-1", "m");
    assert!(line.starts_with("data: "));
    assert!(line.contains("\"content\":\"hi\""));
}
```
```rust
// in anthropic.rs tests
#[test]
fn anthropic_emits_content_block_delta() {
    let d = StreamDelta{ text: Some("hi".into()), done:false, finish_reason:None };
    let evs = stream_events(&d, /*started=*/true);
    assert!(evs.iter().any(|e| e.contains("content_block_delta")));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib api::`
Expected: FAIL — `stream_chunk`/`stream_events`/`StreamDelta` undefined.

- [ ] **Step 3: Implement `StreamDelta` (in common.rs), renderers, `generate_stream`, and SSE handler branches**

In handlers, when `stream` is true, return `axum::response::sse::Sse` from a stream that maps engine deltas through the renderer and ends with `data: [DONE]` (OpenAI) / `message_stop` (Anthropic). For tool-call streaming, the minimum viable behavior is to buffer the tool call and emit it in the final chunk; document this in a comment.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib api::`
Expected: PASS.

- [ ] **Step 5: Manual stream check**

Run server, then:
`curl -N -s localhost:8080/v1/chat/completions -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"count 1 to 5"}]}' -H 'content-type: application/json'`
Expected: incremental `data:` chunks then `data: [DONE]`.

- [ ] **Step 6: Commit**

```bash
git add src/engine.rs src/server.rs src/api/ src/api/common.rs
git commit -m "feat: SSE streaming for OpenAI and Anthropic endpoints"
```

---

### Task 9: Acceptance test scripts + README + measurement

**Files:**
- Create: `scripts/test_openai_tools.sh`
- Create: `scripts/test_anthropic_tools.sh`
- Create: `scripts/measure.sh`
- Create: `README.md`

**Interfaces:**
- Consumes: a running server on `localhost:8080`.
- Produces: executable acceptance scripts and documented results.

- [ ] **Step 1: Write `scripts/test_openai_tools.sh`**

Two-step chain: (1) POST a request with `get_weather` tool and a weather question; assert the response `finish_reason == "tool_calls"` and capture the tool call id + arguments with `jq`. (2) POST a follow-up including the original assistant tool_call message and a `role:"tool"` message with `content:"28°C, sunny"`; assert the final `content` mentions the weather. Exit non-zero on any assertion failure.

- [ ] **Step 2: Write `scripts/test_anthropic_tools.sh`**

Same two-step chain against `/v1/messages` using `tool_use`/`tool_result` blocks; assert `stop_reason == "tool_use"` then a final text answer.

- [ ] **Step 3: Write `scripts/measure.sh`**

Send a fixed prompt, time it, compute tokens/sec from the `usage` in the response; print peak RSS using `/usr/bin/time -l cargo run ...` guidance or `ps -o rss= -p <pid>` sampled during a request.

- [ ] **Step 4: Run the acceptance scripts against the running server**

Run: `cargo run --release &` then `bash scripts/test_openai_tools.sh && bash scripts/test_anthropic_tools.sh && bash scripts/measure.sh`
Expected: both tool scripts pass (chaining works on both APIs); measure prints RAM + tok/s.

- [ ] **Step 5: Write `README.md`**

Document: build (`cargo build --release --features metal`), run, the two API base URLs, how to point Codex (`OPENAI_BASE_URL=http://localhost:8080/v1`) and Claude Code (`ANTHROPIC_BASE_URL=http://localhost:8080`) at it, which optimizations are active vs unavailable (from NOTES), and the measured RAM/tok/s.

- [ ] **Step 6: Commit**

```bash
git add scripts/ README.md
git commit -m "test: tool-calling acceptance scripts, measurement, and README"
```

---

## Self-Review

**Spec coverage:**
- Both API surfaces → Tasks 3, 4, 6, 8. ✓
- mistralrs embedded + GGUF Q4 → Tasks 1, 5. ✓
- All optimizations (ISQ, Metal, PagedAttn, FlashAttn-via-Metal, KV-quant, mmap, prefix-cache, 16k ctx) → Task 5 applies them and documents any unavailable; ctx in Task 7. ✓
- Tool calling/chaining → Tasks 3/4 (mapping), 5 (parse), 9 (acceptance). ✓
- Streaming both → Task 8. ✓
- Error handling (400/500/download/RAM) → Task 6 handlers + Task 5 load. ✓
- Acceptance test (health, OAI, Anthropic, Codex pointing, measure) → Tasks 6, 9. ✓ (Claude Code live test documented in README; Codex live test in Task 9 Step 5.)

**Placeholder scan:** No "TBD"/"handle edge cases" left vague; the one deliberate deferral (exact mistralrs identifiers) is gated and resolved in Task 1, which every later task references. Tool-call streaming buffering is explicitly specified as MVP, not left open.

**Type consistency:** `ChatRequest`/`ChatResult`/`ContentPart`/`ToolCall`/`FinishReason`/`Role`/`ToolSpec`/`StreamDelta` defined in Task 2/8 and used consistently in Tasks 3-8. `Generator` trait (Task 6) implemented by `Engine` (Task 5 produces the inherent method; Task 6 wraps it in the trait). `EngineConfig` defined in Task 5, built by `Config` in Task 7. `to_internal`/`from_internal` names identical across both api modules.
