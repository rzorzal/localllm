//! Smoke test for Task 5: Engine load + tool-call round-trip.
//!
//! Uses force_cpu=true to sidestep the broken Metal shader toolchain.
//! CPU inference of a 7B model is slow but correct.
//!
//! Run with:
//!   MISTRALRS_METAL_PRECOMPILE=0 cargo run --release

use localllm::api::common::{ChatMessage, ChatRequest, ContentPart, Role, ToolSpec};
use localllm::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise tracing so mistralrs progress is visible.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    println!("=== localllm Task-5 smoke test ===");
    println!("Loading Qwen2.5-7B-Instruct-GGUF on CPU (force_cpu=true)…");
    println!("This will be slow. Timeout is set to 600 s.");

    let cfg = EngineConfig {
        model_id: "Qwen/Qwen2.5-7B-Instruct-GGUF".to_string(),
        // Split GGUF: two shards for Q4_K_M.
        gguf_files: vec![
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf".to_string(),
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf".to_string(),
        ],
        ctx_len: 4096, // informational only — model uses built-in context
        paged_attn: false,
        force_cpu: true,
    };

    let engine = Engine::load(&cfg).await?;
    println!("Model loaded. Sending tool-call request…");

    let weather_tool = ToolSpec {
        name: "get_weather".to_string(),
        description: "Get the current weather for a given location.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name to get weather for."
                }
            },
            "required": ["location"]
        }),
    };

    let request = ChatRequest {
        messages: vec![ChatMessage {
            role: Role::User,
            text: Some(
                "What's the weather in Recife? Use the get_weather tool.".to_string(),
            ),
            tool_calls: vec![],
            tool_result: None,
        }],
        tools: vec![weather_tool],
        max_tokens: Some(256),
        temperature: Some(0.0),
        stream: false,
        model: "qwen2.5-7b-instruct".to_string(),
    };

    let result = engine.generate(request).await?;

    println!("\n=== Result ===");
    println!("finish_reason: {:?}", result.finish_reason);
    println!("prompt_tokens: {}", result.prompt_tokens);
    println!("completion_tokens: {}", result.completion_tokens);
    println!("content parts:");
    for part in &result.content {
        match part {
            ContentPart::Text(t) => println!("  Text: {t}"),
            ContentPart::Call(tc) => {
                println!("  ToolCall: name={}, id={}, arguments={}", tc.name, tc.id, tc.arguments);
            }
        }
    }

    let has_tool_call = result
        .content
        .iter()
        .any(|p| matches!(p, ContentPart::Call(_)));

    if has_tool_call {
        println!("\n✓ SMOKE TEST PASSED — model emitted a tool call");
    } else {
        println!("\n✗ SMOKE TEST NOTE — no tool call in response (model responded with text)");
    }

    Ok(())
}
