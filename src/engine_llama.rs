//! llama.cpp inference engine via `llama-cpp-2`.
//!
//! # Architecture
//!
//! `LlamaEngine` wraps a `LlamaModel` (behind `Arc`) and a `Mutex<LlamaBackend>`.
//!
//! ## Why a per-call context?
//! `LlamaContext<'_>` borrows from `LlamaModel` and is therefore NOT `'static`.
//! It also does not implement `Send`. To satisfy the `Generator` trait
//! (`Send + Sync`, `async fn generate`) we create a fresh `LlamaContext` inside
//! each `generate` call. This avoids any lifetime or Send issues at the cost of
//! re-allocating the KV-cache per request — acceptable for a local single-user
//! server. A `Mutex<()>` serializes inference calls (only one at a time).
//!
//! ## Prompt building with tools
//! `llama-cpp-2::model::apply_chat_template` accepts only `[LlamaChatMessage]`
//! (role + content pairs) — there is no native tool schema parameter at the
//! high-level API. We therefore embed the tool specifications as JSON in the
//! system prompt using the Qwen2.5 documented tool format:
//!
//! ```text
//! # Tools
//! You may call one or more functions. Available tools (JSON schema):
//! <tools>
//! {"name":"...","description":"...","parameters":{...}}
//! </tools>
//! To call: <tool_call>{"name":"...","arguments":{...}}</tool_call>
//! ```
//!
//! For the overall ChatML framing we use the model's built-in chat template
//! (retrieved via `model.chat_template(None)`), which is Qwen2.5's standard
//! ChatML template. We pass the serialized tool section as part of the system
//! message content.
//!
//! ## Tool-call output parsing
//! Qwen2.5 emits tool calls as `<tool_call>{json}</tool_call>`. After
//! generation we scan for these blocks and parse each into a `ContentPart::Call`.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use futures::stream::BoxStream;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    context::params::LlamaContextParams,
    sampling::LlamaSampler,
};
use uuid::Uuid;

use crate::api::common::{
    ChatRequest, ChatResult, ContentPart, FinishReason, Role, StreamDelta, ToolCall,
};
use crate::server::Generator;

// ---------------------------------------------------------------------------
// LlamaEngine
// ---------------------------------------------------------------------------

/// llama.cpp inference engine. Implements `Generator` for use in the HTTP server.
pub struct LlamaEngine {
    /// The loaded model — `LlamaModel` is `Send + Sync` (unsafe impls in llama-cpp-2).
    model: Arc<LlamaModel>,
    /// The backend — must outlive the context. Wrapped in Arc so it can be
    /// moved into `spawn_blocking`.
    backend: Arc<LlamaBackend>,
    /// Serializes concurrent inference calls. llama.cpp's context is not
    /// thread-safe; the Mutex ensures only one generate runs at a time.
    _inference_lock: Arc<Mutex<()>>,
    /// Context window in tokens.
    ctx_len: u32,
}

// Safety: LlamaModel is Send+Sync (as declared in llama-cpp-2/src/model.rs).
// LlamaBackend holds a process-wide singleton — safe to share.
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Load a GGUF model from HuggingFace (or local cache) and return a
    /// ready-to-use `LlamaEngine`.
    ///
    /// `model_id` – HF repo ID, e.g. `"Qwen/Qwen2.5-3B-Instruct-GGUF"`.
    /// `gguf_files` – GGUF filename(s) within the repo.
    /// `ctx_len` – context window in tokens.
    pub async fn load(model_id: &str, gguf_files: &[String], ctx_len: usize) -> Result<Self> {
        // Download (or skip if cached) and get local paths.
        let paths: Vec<PathBuf> =
            crate::download::ensure_model(model_id, gguf_files)
                .await
                .context("ensure_model failed")?;

        let path = paths
            .into_iter()
            .next()
            .context("ensure_model returned no paths")?;

        // Load backend + model in a blocking thread — llama.cpp init blocks.
        let ctx_len_u32 = u32::try_from(ctx_len).unwrap_or(u32::MAX);

        let (backend, model) = tokio::task::spawn_blocking(move || -> Result<_> {
            let backend = LlamaBackend::init().context("LlamaBackend::init failed")?;

            let model_params = LlamaModelParams::default()
                .with_n_gpu_layers(u32::MAX); // offload all layers to Metal

            let model = LlamaModel::load_from_file(&backend, &path, &model_params)
                .context("LlamaModel::load_from_file failed")?;

            Ok((backend, model))
        })
        .await
        .context("spawn_blocking panicked")??;

        Ok(LlamaEngine {
            model: Arc::new(model),
            backend: Arc::new(backend),
            _inference_lock: Arc::new(Mutex::new(())),
            ctx_len: ctx_len_u32,
        })
    }

    /// Core inference routine, executed in a blocking thread.
    fn generate_blocking(
        model: Arc<LlamaModel>,
        backend: Arc<LlamaBackend>,
        req: ChatRequest,
        ctx_len: u32,
    ) -> Result<ChatResult> {
        // --- Build prompt string ---
        let prompt = build_prompt(&model, &req)?;
        tracing::debug!(target: "localllm::llama", "prompt ({} chars):\n{}", prompt.len(), &prompt[..prompt.len().min(400)]);

        // --- Tokenize ---
        let tokens = model
            .str_to_token(&prompt, AddBos::Never) // chat template already includes BOS
            .context("str_to_token failed")?;
        let prompt_len = tokens.len();
        tracing::debug!(target: "localllm::llama", "prompt tokens: {prompt_len}");

        // --- Create context ---
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(ctx_len));
        let mut ctx = model
            .new_context(&backend, ctx_params)
            .context("new_context failed")?;

        // --- Prefill: add entire prompt as one batch ---
        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens.max(512), 1);
        batch
            .add_sequence(&tokens, 0, false)
            .context("batch.add_sequence failed")?;
        ctx.decode(&mut batch).context("ctx.decode (prefill) failed")?;
        batch.clear();

        // --- Sampler ---
        let temperature = req.temperature.unwrap_or(0.7) as f32;
        let mut sampler = if temperature < 0.01 {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::dist(42),
            ])
        };

        // --- Generation loop ---
        let max_new = req.max_tokens.unwrap_or(512);
        let mut generated_tokens = 0usize;
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut last_idx = (n_tokens as i32) - 1; // logit index of last prefill token

        loop {
            let tok = sampler.sample(&ctx, last_idx);
            sampler.accept(tok);
            generated_tokens += 1;

            if model.is_eog_token(tok) {
                break;
            }

            let piece = model
                .token_to_piece(tok, &mut decoder, true, None)
                .unwrap_or_default();
            output.push_str(&piece);

            if generated_tokens >= max_new {
                break;
            }

            // Decode the single new token to get its logits.
            batch.clear();
            let pos = (n_tokens + generated_tokens - 1) as i32;
            batch
                .add(tok, pos, &[0_i32], true)
                .context("batch.add failed")?;
            ctx.decode(&mut batch).context("ctx.decode (step) failed")?;
            last_idx = 0; // single-token batch → index 0
        }

        tracing::debug!(target: "localllm::llama", "generated {generated_tokens} tokens: {:?}", &output[..output.len().min(200)]);

        // --- Parse tool calls ---
        let (content, finish_reason) = parse_output(output, generated_tokens >= max_new);

        Ok(ChatResult {
            content,
            finish_reason,
            prompt_tokens: prompt_len,
            completion_tokens: generated_tokens,
        })
    }
}

// ---------------------------------------------------------------------------
// Prompt builder
// ---------------------------------------------------------------------------

/// Build the complete prompt string from a `ChatRequest`.
///
/// Strategy:
/// 1. If the request has tools, embed them in the system message using
///    Qwen2.5's documented tool-call format.
/// 2. Collect all messages into `LlamaChatMessage` pairs.
/// 3. Apply the model's built-in chat template (ChatML for Qwen2.5) via
///    `model.apply_chat_template`, adding the assistant opening tag.
fn build_prompt(model: &LlamaModel, req: &ChatRequest) -> Result<String> {
    // --- Fetch the model's chat template ---
    let tmpl: LlamaChatTemplate = model
        .chat_template(None)
        .context("model has no chat template")?;

    // --- Build the tool preamble (Qwen2.5 format) ---
    let tool_section = if req.tools.is_empty() {
        String::new()
    } else {
        let mut tool_json_lines = String::new();
        for tool in &req.tools {
            let obj = serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            });
            tool_json_lines.push_str(&serde_json::to_string(&obj)?);
            tool_json_lines.push('\n');
        }
        format!(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query. \
Don't make assumptions about what values to plug into functions. \
For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags as follows:\n\
<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-dict>}}\n</tool_call>\n\n\
Here are the available tools:\n<tools>\n{tool_json_lines}</tools>"
        )
    };

    // --- Assemble LlamaChatMessages ---
    let mut chat_messages: Vec<LlamaChatMessage> = Vec::new();

    for msg in &req.messages {
        match &msg.role {
            Role::System => {
                let base = msg.text.clone().unwrap_or_default();
                let content = format!("{base}{tool_section}");
                chat_messages.push(
                    LlamaChatMessage::new("system".to_string(), content)
                        .context("invalid system message content")?,
                );
            }
            Role::User => {
                let content = msg.text.clone().unwrap_or_default();
                chat_messages.push(
                    LlamaChatMessage::new("user".to_string(), content)
                        .context("invalid user message content")?,
                );
            }
            Role::Assistant => {
                // Reconstruct the assistant message content, including any tool calls.
                let mut content = msg.text.clone().unwrap_or_default();
                for tc in &msg.tool_calls {
                    let args_val: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                    let tc_json = serde_json::json!({
                        "name": tc.name,
                        "arguments": args_val,
                    });
                    content.push_str(&format!(
                        "<tool_call>{}</tool_call>",
                        serde_json::to_string(&tc_json)?
                    ));
                }
                chat_messages.push(
                    LlamaChatMessage::new("assistant".to_string(), content)
                        .context("invalid assistant message content")?,
                );
            }
            Role::Tool => {
                // Tool result: render as a user message with <tool_response> tags
                // (Qwen2.5 convention).
                let result_content = if let Some(result) = &msg.tool_result {
                    result.content.clone()
                } else {
                    msg.text.clone().unwrap_or_default()
                };
                let content =
                    format!("<tool_response>\n{result_content}\n</tool_response>");
                // Qwen2.5 expects tool results under the "tool" role.
                chat_messages.push(
                    LlamaChatMessage::new("tool".to_string(), content)
                        .context("invalid tool message content")?,
                );
            }
        }
    }

    // If there are no messages, push a minimal user turn so the template
    // doesn't produce an empty / broken prompt.
    if chat_messages.is_empty() {
        chat_messages.push(
            LlamaChatMessage::new("user".to_string(), "Hello".to_string())
                .expect("static content"),
        );
    }

    // If there are tools and no system message, prepend one.
    if !req.tools.is_empty()
        && !req.messages.iter().any(|m| m.role == Role::System)
    {
        let system_msg =
            LlamaChatMessage::new("system".to_string(), format!("You are a helpful assistant.{tool_section}"))
                .context("invalid synthetic system message")?;
        chat_messages.insert(0, system_msg);
    }

    // --- Apply template (add_ass = true → leaves the opening `<|im_start|>assistant\n`) ---
    let prompt = model
        .apply_chat_template(&tmpl, &chat_messages, true)
        .context("apply_chat_template failed")?;

    Ok(prompt)
}

// ---------------------------------------------------------------------------
// Tool-call output parser
// ---------------------------------------------------------------------------

/// Parse the raw model output into `ContentPart`s.
///
/// Scans for `<tool_call>...</tool_call>` blocks. Each block is expected to
/// contain JSON with `"name"` and `"arguments"` fields.
fn parse_output(output: String, hit_max_tokens: bool) -> (Vec<ContentPart>, FinishReason) {
    // Remove a common trailing artifact: <|im_end|> and similar.
    let output = output
        .trim_end_matches("<|im_end|>")
        .trim_end_matches("<|endoftext|>")
        .trim()
        .to_string();

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut remaining = output.as_str();

    while let Some(start) = remaining.find("<tool_call>") {
        let after_open = &remaining[start + "<tool_call>".len()..];
        if let Some(end) = after_open.find("</tool_call>") {
            let json_str = after_open[..end].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                let name = val
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = match val.get("arguments") {
                    Some(args) => serde_json::to_string(args).unwrap_or_default(),
                    None => "{}".to_string(),
                };
                if !name.is_empty() {
                    tool_calls.push(ToolCall {
                        id: format!("call-{}", Uuid::new_v4().simple()),
                        name,
                        arguments,
                    });
                }
            }
            remaining = &after_open[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    if !tool_calls.is_empty() {
        let content = tool_calls.into_iter().map(ContentPart::Call).collect();
        return (content, FinishReason::ToolCalls);
    }

    // No tool calls — return the text as-is.
    let finish_reason = if hit_max_tokens {
        FinishReason::Length
    } else {
        FinishReason::Stop
    };
    (vec![ContentPart::Text(output)], finish_reason)
}

// ---------------------------------------------------------------------------
// Generator impl
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl Generator for LlamaEngine {
    /// Non-streaming inference. Runs the blocking llama.cpp generate loop
    /// in a `spawn_blocking` task so the async runtime is not stalled.
    async fn generate(&self, req: ChatRequest) -> Result<ChatResult> {
        let model = Arc::clone(&self.model);
        let backend = Arc::clone(&self.backend);
        let ctx_len = self.ctx_len;

        tokio::task::spawn_blocking(move || {
            Self::generate_blocking(model, backend, req, ctx_len)
        })
        .await
        .context("spawn_blocking panicked")?
    }

    /// Streaming inference — Task 4 will make this incremental.
    ///
    /// For now, we run a full non-streaming generation and emit the entire
    /// result as a single terminal `StreamDelta`. This is correct and safe;
    /// it avoids a `todo!()` or panic.
    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamDelta>>> {
        let result = self.generate(req).await?;

        // Collect text from all content parts.
        let text: Option<String> = {
            let parts: Vec<String> = result
                .content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text(t) => Some(t.clone()),
                    ContentPart::Call(_) => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(""))
            }
        };

        let finish_reason = result.finish_reason;
        let deltas: Vec<Result<StreamDelta>> = vec![Ok(StreamDelta {
            text,
            done: true,
            finish_reason: Some(finish_reason),
        })];

        Ok(Box::pin(futures::stream::iter(deltas)))
    }
}

// ---------------------------------------------------------------------------
// Smoke-test helper (called from main when --llama flag is passed)
// ---------------------------------------------------------------------------

/// Convenience wrapper to smoke-test LlamaEngine end-to-end.
///
/// Loads the default 3B model, sends a weather tool request, and prints the
/// result. Called from `main.rs` when `--llama-smoke` is passed.
pub async fn run_smoke_test() -> Result<()> {
    use crate::api::common::{ChatMessage, ToolSpec};

    println!("[smoke] Loading Qwen2.5-3B via LlamaEngine …");
    let engine = LlamaEngine::load(
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        &["qwen2.5-3b-instruct-q4_k_m.gguf".to_string()],
        4096,
    )
    .await?;

    println!("[smoke] Model loaded. Sending tool-call request …");
    let req = ChatRequest {
        messages: vec![ChatMessage {
            role: Role::User,
            text: Some(
                "What's the weather in Recife? Use the get_weather tool.".to_string(),
            ),
            tool_calls: vec![],
            tool_result: None,
        }],
        tools: vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "Get the current weather for a city.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name."
                    }
                },
                "required": ["location"]
            }),
        }],
        max_tokens: Some(256),
        temperature: Some(0.0),
        stream: false,
        model: "qwen2.5-3b".to_string(),
    };

    let result = engine.generate(req).await?;

    println!("[smoke] Result:");
    println!("  finish_reason = {:?}", result.finish_reason);
    println!("  prompt_tokens = {}", result.prompt_tokens);
    println!("  completion_tokens = {}", result.completion_tokens);
    for (i, part) in result.content.iter().enumerate() {
        match part {
            ContentPart::Text(t) => println!("  content[{i}] = Text({t:?})"),
            ContentPart::Call(tc) => {
                println!("  content[{i}] = Call(name={}, args={})", tc.name, tc.arguments)
            }
        }
    }

    let got_tool_call = result
        .content
        .iter()
        .any(|p| matches!(p, ContentPart::Call(_)));

    if got_tool_call {
        println!("[smoke] SUCCESS — tool call parsed.");
    } else {
        println!("[smoke] WARN — no tool call found in output. Check raw output above.");
    }

    Ok(())
}
