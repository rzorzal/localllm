//! Inference engine: the ONLY module that touches mistralrs.
//!
//! Loads a GGUF model and converts between the localllm internal types
//! (`api::common`) and mistralrs types. HTTP handlers call `Engine::generate`.
//! This module must NOT import axum.
//!
//! ## Optimizations applied
//! - Prefix caching: ON by default in GgufModelBuilder (16 sequences). Left enabled.
//! - Logging: `.with_logging()` — enabled.
//! - Paged attention: `.with_paged_attn(...)` guarded by `paged_attn_supported()`.
//!   NOTE: paged attention is inert on this machine because the Metal shader
//!   toolchain is broken (MISTRALRS_METAL_PRECOMPILE=0 is required to build).
//!   The call is skipped entirely when `force_cpu` is true, because paged
//!   attention requires CUDA or a working Metal runtime.
//!
//! ## Optimizations NOT available for GGUF models (documented, not faked)
//! - ISQ re-quantization: NOT supported on GgufModelBuilder. GGUF files are
//!   pre-quantized (this model is Q4_K_M). ISQ only applies to TextModelBuilder.
//!   (NOTES-mistralrs-api.md §2)
//! - Flash Attention: requires the `flash-attn` cargo feature, which requires CUDA.
//!   Not available on this Metal/CPU machine. (NOTES-mistralrs-api.md §9)
//! - KV-cache quantization: not exposed via GgufModelBuilder. There is no
//!   `with_kv_cache_dtype()` on this builder. (NOTES-mistralrs-api.md §9, §10)

use anyhow::{Context, Result};
use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, Function, GgufModelBuilder,
    PagedAttentionMetaBuilder, RequestBuilder, Response,
    TextMessageRole, Tool, ToolCallResponse, ToolChoice, ToolType, paged_attn_supported,
};

use crate::api::common::{
    ChatRequest, ChatResult, ContentPart, FinishReason, Role, StreamDelta, ToolCall, ToolSpec,
};

/// Configuration for loading the GGUF model.
///
/// `ctx_len` is carried here for Task 7's CLI but is currently informational
/// — GgufModelBuilder has no context-length setter. Context length is read from
/// GGUF metadata (Qwen2.5-7B is 32k). See NOTES-mistralrs-api.md §10 for the
/// complete method list.
pub struct EngineConfig {
    /// HuggingFace repo ID, e.g. "Qwen/Qwen2.5-7B-Instruct-GGUF".
    pub model_id: String,
    /// One or more GGUF shard filenames within the repo.
    pub gguf_files: Vec<String>,
    /// Informational: desired context length. Not applied — model uses built-in
    /// context from GGUF metadata. Retained for Task 7's CLI config struct.
    pub ctx_len: usize,
    /// Whether to request paged attention. Guarded by `paged_attn_supported()`
    /// at runtime; inert on this machine and skipped when `force_cpu` is true.
    pub paged_attn: bool,
    /// Force CPU inference. Useful for smoke-testing on machines where Metal
    /// shaders are broken. CPU inference of a 7B model is slow but correct.
    pub force_cpu: bool,
}

/// The inference engine. Holds a loaded mistralrs `Model` behind an Arc so it
/// can be shared into 'static streaming closures without tying the stream to
/// the Engine's lifetime.
pub struct Engine {
    model: std::sync::Arc<mistralrs::Model>,
}

impl Engine {
    /// Load the GGUF model from HuggingFace (or local cache) and return a
    /// ready-to-use `Engine`.
    pub async fn load(cfg: &EngineConfig) -> Result<Engine> {
        let mut builder = GgufModelBuilder::new(
            cfg.model_id.clone(),
            cfg.gguf_files.clone(),
        )
        // Explicit tokenizer repo for Qwen GGUF (recommended per NOTES §1).
        .with_tok_model_id("Qwen/Qwen2.5-7B-Instruct")
        // Logging (NOTES §10).
        .with_logging();

        // Force CPU when requested (e.g. smoke test on this machine).
        if cfg.force_cpu {
            builder = builder.with_force_cpu();
        }

        // Paged attention: guarded by runtime support check and skipped on CPU.
        // On this machine, Metal shader compilation is broken (needs
        // MISTRALRS_METAL_PRECOMPILE=0), so even if `paged_attn_supported()`
        // returns true the runtime will not actually use paged attention pages.
        // Skipped on force_cpu because paged attention is a GPU-only feature.
        // (NOTES-mistralrs-api.md §3)
        if cfg.paged_attn && !cfg.force_cpu && paged_attn_supported() {
            builder = builder
                .with_paged_attn(PagedAttentionMetaBuilder::default().build()?);
        }

        // Use a compact max_seq_len for the auto device-map memory estimator.
        // The default is 4096, which causes the KV-cache estimate to push the
        // model over the available RAM budget on this machine.  512 is enough
        // for the estimator; the model still generates at its full runtime
        // context length — this setting only governs how much KV-cache headroom
        // the estimator reserves when deciding which layers fit on which device.
        builder = builder.with_device_mapping(DeviceMapSetting::Auto(
            AutoDeviceMapParams::Text {
                max_seq_len: 512,
                max_batch_size: 1,
            },
        ));

        let model = builder.build().await.context("Failed to load GGUF model")?;
        Ok(Engine { model: std::sync::Arc::new(model) })
    }

    /// Build a `RequestBuilder` from a `ChatRequest`.
    ///
    /// Shared by `generate` (non-streaming) and `generate_stream` (streaming)
    /// so request construction is not duplicated.
    fn build_request(req: &ChatRequest) -> Result<RequestBuilder> {
        let mut builder = RequestBuilder::new();

        for msg in &req.messages {
            match msg.role {
                Role::System => {
                    let text = msg.text.clone().unwrap_or_default();
                    builder = builder.add_message(TextMessageRole::System, text);
                }
                Role::User => {
                    let text = msg.text.clone().unwrap_or_default();
                    builder = builder.add_message(TextMessageRole::User, text);
                }
                Role::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        // Assistant message that produced tool calls: reconstruct
                        // as an assistant message with tool_calls attached.
                        // This matches the round-trip pattern from the mistralrs
                        // tools example (examples/advanced/tools/main.rs:71-74).
                        let tool_call_responses: Vec<ToolCallResponse> = msg
                            .tool_calls
                            .iter()
                            .enumerate()
                            .map(|(i, tc)| ToolCallResponse {
                                index: i,
                                id: tc.id.clone(),
                                tp: mistralrs::ToolCallType::Function,
                                function: mistralrs::CalledFunction {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.clone(),
                                },
                            })
                            .collect();
                        let text = msg.text.clone().unwrap_or_default();
                        builder = builder.add_message_with_tool_call(
                            TextMessageRole::Assistant,
                            text,
                            tool_call_responses,
                        );
                    } else {
                        let text = msg.text.clone().unwrap_or_default();
                        builder = builder.add_message(TextMessageRole::Assistant, text);
                    }
                }
                Role::Tool => {
                    // Tool result message: feed back via add_tool_message.
                    // Matches the round-trip pattern from the tools example
                    // (examples/advanced/tools/main.rs:76-78).
                    if let Some(result) = &msg.tool_result {
                        builder = builder.add_tool_message(
                            result.content.clone(),
                            result.tool_call_id.clone(),
                        );
                    } else if let Some(text) = &msg.text {
                        // Fallback: no structured tool_result, use text as content.
                        builder = builder.add_message(TextMessageRole::Tool, text.clone());
                    }
                }
            }
        }

        // Attach tools if any are specified.
        if !req.tools.is_empty() {
            let tools: Result<Vec<Tool>> = req
                .tools
                .iter()
                .map(|spec: &ToolSpec| -> Result<Tool> {
                    Ok(Tool {
                        tp: ToolType::Function,
                        function: Function {
                            description: Some(spec.description.clone()),
                            name: spec.name.clone(),
                            parameters: Some(
                                serde_json::from_value(spec.parameters.clone())
                                    .context("Failed to parse tool parameters schema")?,
                            ),
                        },
                    })
                })
                .collect();
            builder = builder
                .set_tools(tools?)
                .set_tool_choice(ToolChoice::Auto);
        }

        // Apply sampling parameters.
        if let Some(temp) = req.temperature {
            builder = builder.set_sampler_temperature(temp);
        }
        if let Some(max_tok) = req.max_tokens {
            builder = builder.set_sampler_max_len(max_tok);
        }

        Ok(builder)
    }

    /// Run inference for a single chat turn (non-streaming).
    ///
    /// Translates `ChatRequest` → mistralrs `RequestBuilder`, calls
    /// `send_chat_request`, and converts the response back to `ChatResult`.
    pub async fn generate(&self, req: ChatRequest) -> Result<ChatResult> {
        let builder = Self::build_request(&req)?;

        // Send the request and await the (non-streaming) response.
        let resp = self
            .model
            .send_chat_request(builder)
            .await
            .context("Model inference failed")?;

        // --- Convert response to ChatResult ---
        let choice = resp
            .choices
            .into_iter()
            .next()
            .context("Model returned no choices")?;

        let finish_reason = match choice.finish_reason.as_str() {
            "tool_calls" => FinishReason::ToolCalls,
            "length" => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        let content: Vec<ContentPart> = if let Some(tool_calls) = choice.message.tool_calls {
            if !tool_calls.is_empty() {
                tool_calls
                    .into_iter()
                    .map(|tc| {
                        ContentPart::Call(ToolCall {
                            id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                        })
                    })
                    .collect()
            } else {
                vec![ContentPart::Text(
                    choice.message.content.unwrap_or_default(),
                )]
            }
        } else {
            vec![ContentPart::Text(
                choice.message.content.unwrap_or_default(),
            )]
        };

        // Determine finish_reason from content if model said "stop" but we have tool calls.
        let finish_reason = if content
            .iter()
            .any(|p| matches!(p, ContentPart::Call(_)))
        {
            FinishReason::ToolCalls
        } else {
            finish_reason
        };

        Ok(ChatResult {
            content,
            finish_reason,
            prompt_tokens: resp.usage.prompt_tokens as usize,
            completion_tokens: resp.usage.completion_tokens as usize,
        })
    }

    /// Stream inference for a single chat turn.
    ///
    /// Returns a `BoxStream` of `StreamDelta` items. Each non-terminal item
    /// carries incremental text. The terminal item has `done=true` and
    /// `finish_reason` set.
    ///
    /// ## Tool-call streaming (MVP)
    /// Tool-call deltas are not incrementally emitted. mistralrs delivers them
    /// in a final Response::Done chunk; if present, we emit a single done
    /// StreamDelta with `text=None` and `finish_reason=ToolCalls`. Clients that
    /// need incremental tool-call streaming should use the non-streaming path.
    ///
    /// ## Implementation note
    /// The mistralrs `Stream<'_>` type borrows from the `Model`. To produce a
    /// `'static` BoxStream, we spawn a Tokio task that owns an `Arc<Model>`
    /// clone, drives the mistralrs stream, and forwards `StreamDelta`s through a
    /// channel. The caller receives a stream backed by that channel.
    pub async fn generate_stream(
        &self,
        req: ChatRequest,
    ) -> Result<futures::stream::BoxStream<'static, Result<StreamDelta>>> {
        let builder = Self::build_request(&req)?;
        let model = std::sync::Arc::clone(&self.model);

        // Spawn a task that owns the Arc<Model> and drives the mistralrs stream.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamDelta>>(32);

        tokio::spawn(async move {
            let raw_stream_result = model
                .stream_chat_request(builder)
                .await;

            let mut raw_stream = match raw_stream_result {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("Failed to start streaming: {e}"))).await;
                    return;
                }
            };

            while let Some(resp) = raw_stream.next().await {
                let delta = match resp {
                    Response::Chunk(chunk) => {
                        let choice = chunk.choices.into_iter().next();
                        let delta_text = choice
                            .as_ref()
                            .and_then(|ch| ch.delta.content.clone());
                        let finish_str = choice
                            .as_ref()
                            .and_then(|ch| ch.finish_reason.clone());

                        let (done, finish_reason) = if let Some(ref reason) = finish_str {
                            let fr = match reason.as_str() {
                                "tool_calls" => FinishReason::ToolCalls,
                                "length" => FinishReason::Length,
                                _ => FinishReason::Stop,
                            };
                            (true, Some(fr))
                        } else {
                            (false, None)
                        };

                        Ok(StreamDelta { text: delta_text, done, finish_reason })
                    }
                    Response::Done(_) => {
                        // Response::Done is the non-streaming terminal variant and is NOT
                        // expected on the streaming path (see NOTES §8). This arm is a
                        // defensive guard: if mistralrs ever emits Done instead of a final
                        // Chunk on a streaming request, we still emit a clean done sentinel
                        // rather than silently dropping the stream.
                        let _ = tx.send(Ok(StreamDelta {
                            text: None,
                            done: true,
                            finish_reason: Some(FinishReason::Stop),
                        })).await;
                        break;
                    }
                    Response::ModelError(msg, _) => {
                        Err(anyhow::anyhow!("Model error during streaming: {msg}"))
                    }
                    _ => {
                        // Ignore other Response variants.
                        continue;
                    }
                };

                let is_err = delta.is_err();
                let _ = tx.send(delta).await;
                if is_err {
                    break;
                }
            }
        });

        // Wrap the channel receiver in a futures::Stream using unfold.
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        Ok(Box::pin(stream))
    }
}


