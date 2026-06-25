//! llama.cpp inference engine via `llama-cpp-2`.
//!
//! # Architecture — Persistent Single-Context Worker Thread
//!
//! ## The Metal multi-context problem
//! On Apple Silicon (Metal), `ggml` compiles its shader library when the first
//! `LlamaContext` is created. Creating a SECOND context in the same process
//! causes a Metal shader redefinition error:
//!   `NSError code 3: program_source: error: redefinition of 'as_bits' / 'fp8_e4m3_to_float'`
//! This makes every request after the first fail.
//!
//! ## Fix: one persistent context, owned by a dedicated OS thread
//! `LlamaContext<'_>` borrows `LlamaModel` and is `!Send`, so it cannot live
//! in an async struct directly. We solve this with a **dedicated OS thread**
//! that owns `LlamaBackend` + `LlamaModel` + ONE `LlamaContext` for the whole
//! process lifetime, and processes requests from a `std::sync::mpsc` channel.
//!
//! The async `LlamaEngine` holds only the channel sender (which IS `Send+Sync`).
//!
//! ## KV-cache prefix reuse (Task 5)
//! Instead of full-clearing the KV cache before every request, the worker keeps
//! track of which tokens are currently resident in the KV cache (`cached_tokens`).
//! For each new request:
//!   1. Compute the longest common prefix between `cached_tokens` and the new
//!      prompt tokens (capped at `new_tokens.len() - 1` so at least 1 token is
//!      always decoded to produce fresh logits).
//!   2. Trim the KV cache from position `common` onward via `clear_kv_cache_seq`.
//!   3. Prefill only `new_tokens[common..]`, starting at position `common`.
//!   4. After generation, save the prompt tokens (NOT the generated tail) as the
//!      new `cached_tokens` so the static prefix stays warm for the next turn.
//!
//! Claude Code sends a ~26k-token static system+tools prefix on every turn.
//! After the cold first request the prefix is resident, and each subsequent turn
//! only re-prefills the small unique tail — dramatically reducing TTFT.
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
//! ## Tool-call output parsing
//! Qwen2.5 emits tool calls as `<tool_call>{json}</tool_call>`. After
//! generation we scan for these blocks and parse each into a `ContentPart::Call`.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;

use anyhow::{Context, Result};
use futures::stream::BoxStream;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
    context::params::LlamaContextParams,
    sampling::LlamaSampler,
    token::LlamaToken,
};
use uuid::Uuid;

use crate::api::common::{
    ChatRequest, ChatResult, ContentPart, FinishReason, Role, StreamDelta, ToolCall,
};
use crate::server::Generator;

// ---------------------------------------------------------------------------
// Job enum — requests sent to the worker thread
// ---------------------------------------------------------------------------

/// A request sent from async callers to the persistent worker thread.
enum Job {
    /// Non-streaming: generate the full output and return via oneshot.
    Generate {
        req: ChatRequest,
        reply: tokio::sync::oneshot::Sender<Result<ChatResult>>,
    },
    /// Streaming: send one `StreamDelta` per decoded piece through the mpsc channel.
    Stream {
        req: ChatRequest,
        reply: tokio::sync::mpsc::Sender<Result<StreamDelta>>,
    },
}

// ---------------------------------------------------------------------------
// LlamaEngine
// ---------------------------------------------------------------------------

/// llama.cpp inference engine. Implements `Generator` for use in the HTTP server.
///
/// Internally, all inference runs on a dedicated OS thread that owns the single
/// `LlamaContext`. This struct holds only the channel sender used to dispatch
/// jobs to that thread.
pub struct LlamaEngine {
    /// Channel sender to the worker thread. `std_mpsc::SyncSender` is `Send+Sync`.
    tx: std_mpsc::SyncSender<Job>,
}

// SyncSender<Job> is Send+Sync (Job itself need not be Send because we only
// ever send it from async tasks, but SyncSender requires T: Send).
// Job contains ChatRequest (Send) and tokio channel senders (Send), so this is fine.
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Load a GGUF model from HuggingFace (or local cache), spawn the persistent
    /// worker thread, and return a ready-to-use `LlamaEngine`.
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

        let ctx_len_u32 = u32::try_from(ctx_len).unwrap_or(u32::MAX);

        // Channel for sending jobs to the worker thread.
        // Bound of 4: limits queue depth (we process one at a time anyway).
        let (tx, rx) = std_mpsc::sync_channel::<Job>(4);

        // Oneshot to receive the load result back from the worker thread.
        let (load_tx, load_rx) = tokio::sync::oneshot::channel::<Result<()>>();

        // Spawn the persistent worker thread. It owns backend + model + context.
        std::thread::spawn(move || {
            worker_thread(path, ctx_len_u32, rx, load_tx);
        });

        // Wait for the worker to signal successful model load (or an error).
        load_rx
            .await
            .context("worker thread dropped load channel without signalling")??;

        Ok(LlamaEngine { tx })
    }
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

/// The persistent worker thread. Owns `LlamaBackend`, `LlamaModel`, and ONE
/// `LlamaContext` for the entire process lifetime.
///
/// Signals model-load success/failure via `load_tx`, then loops processing `Job`s.
fn worker_thread(
    model_path: PathBuf,
    ctx_len: u32,
    rx: std_mpsc::Receiver<Job>,
    load_tx: tokio::sync::oneshot::Sender<Result<()>>,
) {
    // --- Init backend ---
    let backend = match LlamaBackend::init().context("LlamaBackend::init failed") {
        Ok(b) => b,
        Err(e) => { let _ = load_tx.send(Err(e)); return; }
    };

    // --- Load model ---
    let model_params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
    let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("LlamaModel::load_from_file failed")
    {
        Ok(m) => m,
        Err(e) => { let _ = load_tx.send(Err(e)); return; }
    };

    // --- Create the ONE persistent context ---
    // Set n_batch = ctx_len so large prompts (e.g. 26k-token Claude Code prefix)
    // can be prefilled in a single decode call without hitting the
    // "n_tokens_all <= cparams.n_batch" assertion. Without this, the default
    // n_batch of 512 rejects any prefill batch larger than 512 tokens.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_len))
        .with_n_batch(ctx_len);
    let mut ctx = match model.new_context(&backend, ctx_params)
        .context("new_context failed")
    {
        Ok(c) => c,
        Err(e) => { let _ = load_tx.send(Err(e)); return; }
    };

    // Signal successful load.
    if load_tx.send(Ok(())).is_err() {
        // Receiver already dropped; nothing to do.
        return;
    }

    // Tracks the prompt tokens currently resident in the KV cache.
    // Initially empty (cache is clean after context creation).
    // Single-threaded — no locking needed (only the worker touches this).
    let mut cached_tokens: Vec<LlamaToken> = Vec::new();

    // --- Job loop ---
    while let Ok(job) = rx.recv() {
        match job {
            Job::Generate { req, reply } => {
                let mut output = String::new();
                let result = run_decode_loop(
                    &model,
                    &mut ctx,
                    &mut cached_tokens,
                    &req,
                    |piece| { output.push_str(&piece); true },
                );
                let chat_result = result.map(|(prompt_len, gen_tokens, hit_max)| {
                    tracing::debug!(
                        target: "localllm::llama",
                        "generated {gen_tokens} tokens: {:?}",
                        &output[..output.len().min(200)]
                    );
                    let (content, finish_reason) = parse_output(output, hit_max);
                    ChatResult {
                        content,
                        finish_reason,
                        prompt_tokens: prompt_len,
                        completion_tokens: gen_tokens,
                    }
                });
                // On error, clear the cache state so the next request starts clean.
                if chat_result.is_err() {
                    cached_tokens.clear();
                    ctx.clear_kv_cache();
                }
                let _ = reply.send(chat_result);
            }

            Job::Stream { req, reply } => {
                let reply_err = reply.clone();
                let result = run_decode_loop(
                    &model,
                    &mut ctx,
                    &mut cached_tokens,
                    &req,
                    |piece| {
                        let delta = Ok(StreamDelta {
                            text: Some(piece),
                            done: false,
                            finish_reason: None,
                        });
                        // blocking_send blocks if channel is full, returns Err if receiver dropped.
                        reply.blocking_send(delta).is_ok()
                    },
                );
                // On error, clear the cache state so the next request starts clean.
                if result.is_err() {
                    cached_tokens.clear();
                    ctx.clear_kv_cache();
                }
                // Terminal delta.
                match result {
                    Ok((_, _, hit_max)) => {
                        let finish_reason = if hit_max { FinishReason::Length } else { FinishReason::Stop };
                        let _ = reply_err.blocking_send(Ok(StreamDelta {
                            text: None,
                            done: true,
                            finish_reason: Some(finish_reason),
                        }));
                    }
                    Err(e) => {
                        let _ = reply_err.blocking_send(Err(e));
                    }
                }
            }
        }
    }
    // rx disconnected → process is shutting down. Worker exits cleanly.
}

// ---------------------------------------------------------------------------
// Core decode loop (operates on existing context — no new_context call)
// ---------------------------------------------------------------------------

/// Core inference routine, executed on the persistent worker thread.
///
/// Implements longest-common-prefix KV reuse: only re-prefills the portion of
/// the new prompt that differs from what is already resident in the KV cache.
///
/// `cached_tokens` — the prompt tokens currently in the KV cache (empty on
/// first call). Updated to `new_tokens` (the full prompt) on success so the
/// static prefix stays warm for the next turn.
///
/// Returns the prompt token count, the generated token count, and whether the
/// generation hit the max-token limit.
///
/// `on_piece` is called for each decoded non-empty piece. Return `false` to abort.
fn run_decode_loop(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    cached_tokens: &mut Vec<LlamaToken>,
    req: &ChatRequest,
    mut on_piece: impl FnMut(String) -> bool,
) -> Result<(usize /*prompt_tokens*/, usize /*generated*/, bool /*hit_max*/)> {
    // --- Build prompt string ---
    let prompt = build_prompt(model, req)?;
    tracing::debug!(target: "localllm::llama", "prompt ({} chars):\n{}", prompt.len(), &prompt[..prompt.len().min(400)]);

    // --- Tokenize ---
    let new_tokens = model
        .str_to_token(&prompt, AddBos::Never) // chat template already includes BOS
        .context("str_to_token failed")?;
    let prompt_len = new_tokens.len();
    tracing::debug!(target: "localllm::llama", "prompt tokens: {prompt_len}");

    // --- Compute longest common prefix for KV reuse ---
    // Cap at new_tokens.len()-1 so there is always at least one token to decode
    // and produce fresh logits.
    let max_common = new_tokens.len().saturating_sub(1);
    let common = cached_tokens
        .iter()
        .zip(new_tokens.iter())
        .take(max_common)
        .take_while(|(a, b)| a == b)
        .count();

    tracing::info!(
        target: "localllm::llama",
        "prefix reuse: {common}/{total} tokens cached, decoding {} new",
        new_tokens.len() - common,
        total = new_tokens.len(),
    );

    // --- Trim KV cache from `common` onward ---
    // If common == 0, this is equivalent to a full clear.
    if common == 0 {
        ctx.clear_kv_cache();
    } else {
        // Remove KV entries from position `common` to end of sequence 0.
        ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None)
            .context("clear_kv_cache_seq failed")?;
    }

    // --- Prefill: only new_tokens[common..] starting at position `common` ---
    let tail = &new_tokens[common..];
    let n_tail = tail.len();
    // Batch capacity must be >= n_tail so all tail tokens fit in a single
    // decode call. We also ensure a minimum of 1 to avoid zero-capacity batches.
    let mut batch = LlamaBatch::new(n_tail.max(1), 1);
    // Add each tail token with its absolute position, logits=true only on last.
    for (i, &tok) in tail.iter().enumerate() {
        let pos = (common + i) as i32;
        let is_last = i == n_tail - 1;
        batch
            .add(tok, pos, &[0_i32], is_last)
            .context("batch.add (prefill) failed")?;
    }
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
    // Single stateful UTF-8 decoder reused across all tokens (multi-byte safe).
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // After prefill, logits live at index 0 of the single-batch decode (n_tail-1
    // relative, but we used a single batch so it's always the last entry).
    // For the first sample we use idx = n_tail-1 within the batch we just ran
    // (which had n_tail tokens); since we only set logits=true on the last token
    // of that batch, we sample at index n_tail-1.
    let mut last_idx = (n_tail as i32) - 1;

    loop {
        let tok = sampler.sample(ctx, last_idx);
        sampler.accept(tok);
        generated_tokens += 1;

        if model.is_eog_token(tok) {
            // Update cached_tokens to the prompt (not the generated tail).
            *cached_tokens = new_tokens;
            return Ok((prompt_len, generated_tokens, false));
        }

        let piece = model
            .token_to_piece(tok, &mut decoder, true, None)
            .unwrap_or_default();

        if !piece.is_empty() && !on_piece(piece) {
            // Caller signalled abort (e.g. channel closed).
            // Still update cached_tokens — prefix is still warm.
            *cached_tokens = new_tokens;
            return Ok((prompt_len, generated_tokens, false));
        }

        if generated_tokens >= max_new {
            *cached_tokens = new_tokens;
            return Ok((prompt_len, generated_tokens, true));
        }

        // Decode the single new token to get its logits.
        // Position is absolute: (prompt length) + (generated so far) - 1.
        batch.clear();
        let pos = (prompt_len + generated_tokens - 1) as i32;
        batch
            .add(tok, pos, &[0_i32], true)
            .context("batch.add failed")?;
        ctx.decode(&mut batch).context("ctx.decode (step) failed")?;
        last_idx = 0; // single-token batch → index 0
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
        // Use Tscg compact serialization (~57% smaller than verbose JSON).
        let compact_block = crate::tscg::compact_tools_block(&req.tools);
        tracing::debug!(
            target: "localllm::llama",
            "tool_section: {} tools, compact_block={} chars",
            req.tools.len(),
            compact_block.len(),
        );
        format!(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query. \
Don't make assumptions about what values to plug into functions. \
For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags as follows:\n\
<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-dict>}}\n</tool_call>\n\n\
Here are the available tools (compact schema — name(param:type! (desc)) — description):\n<tools>\n{compact_block}\n</tools>"
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
    /// Non-streaming inference. Sends a `Job::Generate` to the worker thread
    /// and awaits the result via a oneshot channel.
    async fn generate(&self, req: ChatRequest) -> Result<ChatResult> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Job::Generate { req, reply: reply_tx })
            .map_err(|_| anyhow::anyhow!("worker thread has stopped"))?;
        reply_rx
            .await
            .context("worker thread dropped reply channel")?
    }

    /// Streaming inference. Sends a `Job::Stream` to the worker thread and
    /// returns a `BoxStream` backed by the reply channel receiver.
    ///
    /// The worker thread sends one `StreamDelta` per decoded piece, then a
    /// terminal delta with `done=true` and `finish_reason` set.
    async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamDelta>>> {
        // Channel capacity: 32 deltas — enough to buffer brief bursts without
        // blocking the decode loop. The loop parks when full (blocking_send).
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamDelta>>(32);

        self.tx
            .send(Job::Stream { req, reply: tx })
            .map_err(|_| anyhow::anyhow!("worker thread has stopped"))?;

        // Wrap the channel receiver in a futures::Stream.
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// Smoke-test helper (called from main when --llama-smoke is passed)
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
