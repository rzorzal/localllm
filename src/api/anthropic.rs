// Anthropic HTTP translation layer (Task 4)

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::common::{
    ChatMessage, ChatRequest, ChatResult, ContentPart, FinishReason, Role, StreamDelta, ToolCall,
    ToolResult, ToolSpec,
};

// ---------------------------------------------------------------------------
// Helper: content field is either a plain string or an array of content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum StringOrBlocks {
    Str(String),
    Blocks(Vec<AnthContentBlock>),
}

impl Default for StringOrBlocks {
    fn default() -> Self {
        StringOrBlocks::Str(String::new())
    }
}

// ---------------------------------------------------------------------------
// Anthropic wire types
// ---------------------------------------------------------------------------

/// A content block in an Anthropic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// A single message in the Anthropic `messages` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthMessage {
    pub role: String,
    /// Content is either a plain string or an array of content blocks.
    #[serde(default)]
    pub(crate) content: StringOrBlocks,
}

/// A tool definition in the Anthropic request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Anthropic uses `input_schema` instead of OpenAI's `parameters`.
    pub input_schema: serde_json::Value,
}

/// Token usage in the Anthropic response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// The top-level Anthropic Messages API request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthRequest {
    pub model: String,
    pub messages: Vec<AnthMessage>,
    #[serde(default)]
    pub tools: Vec<AnthTool>,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

/// The top-level Anthropic Messages API response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthContentBlock>,
    pub stop_reason: String,
    pub usage: AnthUsage,
}

// ---------------------------------------------------------------------------
// Conversion: AnthRequest → ChatRequest
// ---------------------------------------------------------------------------

/// Convert an incoming Anthropic Messages request into the internal
/// `ChatRequest` representation.
pub fn to_internal(req: AnthRequest) -> Result<ChatRequest, String> {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // Hoist top-level `system` field as the first message
    if let Some(system_text) = req.system {
        messages.push(ChatMessage {
            role: Role::System,
            text: Some(system_text),
            tool_calls: vec![],
            tool_result: None,
        });
    }

    for msg in req.messages {
        let role_str = msg.role.as_str();

        match msg.content {
            StringOrBlocks::Str(text) => {
                let role = parse_role(role_str)?;
                messages.push(ChatMessage {
                    role,
                    text: if text.is_empty() { None } else { Some(text) },
                    tool_calls: vec![],
                    tool_result: None,
                });
            }
            StringOrBlocks::Blocks(blocks) => {
                // Each block may generate its own ChatMessage (e.g. tool_result)
                // or contribute to a single message (text + tool_use on assistant).
                // Strategy: group text+tool_use into one assistant message;
                // tool_result blocks each become their own Role::Tool message.

                let mut text_parts: Vec<String> = vec![];
                let mut tool_calls: Vec<ToolCall> = vec![];
                let mut tool_result_msgs: Vec<ChatMessage> = vec![];

                for block in blocks {
                    match block {
                        AnthContentBlock::Text { text } => {
                            text_parts.push(text);
                        }
                        AnthContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(ToolCall {
                                id,
                                name,
                                arguments: input.to_string(),
                            });
                        }
                        AnthContentBlock::ToolResult {
                            tool_use_id,
                            content,
                        } => {
                            tool_result_msgs.push(ChatMessage {
                                role: Role::Tool,
                                text: None,
                                tool_calls: vec![],
                                tool_result: Some(ToolResult {
                                    tool_call_id: tool_use_id,
                                    content,
                                }),
                            });
                        }
                    }
                }

                // If there are tool_result blocks, emit them directly
                if !tool_result_msgs.is_empty() {
                    messages.extend(tool_result_msgs);
                } else {
                    // text and/or tool_use blocks → one message with the parsed role
                    let role = parse_role(role_str)?;
                    let combined_text = if text_parts.is_empty() {
                        None
                    } else {
                        Some(text_parts.join(""))
                    };
                    messages.push(ChatMessage {
                        role,
                        text: combined_text,
                        tool_calls,
                        tool_result: None,
                    });
                }
            }
        }
    }

    let tools = req
        .tools
        .into_iter()
        .map(|t| ToolSpec {
            name: t.name,
            description: t.description,
            parameters: t.input_schema,
        })
        .collect();

    Ok(ChatRequest {
        messages,
        tools,
        max_tokens: Some(req.max_tokens),
        temperature: req.temperature,
        stream: req.stream.unwrap_or(false),
        model: req.model,
    })
}

// ---------------------------------------------------------------------------
// Streaming: SSE event renderer (Anthropic protocol)
// ---------------------------------------------------------------------------

/// Render a `StreamDelta` as one or more Anthropic SSE event strings.
///
/// Each returned string is one complete SSE event: `event: <type>\ndata: {json}\n`
/// (without the trailing blank line; the caller adds `\n` to delimit frames).
///
/// Protocol summary:
/// - First chunk (`!started`): prepend `message_start` + `content_block_start`
///   events so the client can initialize its state machine.
/// - Text chunks: emit a `content_block_delta` with `type:"text_delta"`.
/// - Terminal chunk (`done=true`): emit `content_block_stop` + `message_delta`
///   + `message_stop`.
///
/// Tool-call streaming is MVP: tool calls are not incrementally streamed; they
/// are expected to be emitted in the final delta chunk's text field (buffered
/// in the engine). Clients that need incremental tool-call deltas should use
/// the non-streaming path.
pub fn stream_events(delta: &StreamDelta, started: bool) -> Vec<String> {
    let mut events: Vec<String> = Vec::new();

    // On the very first chunk, send the session-opener events.
    if !started {
        let msg_start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "",
                "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        events.push(format!(
            "event: message_start\ndata: {}",
            serde_json::to_string(&msg_start).unwrap()
        ));

        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        events.push(format!(
            "event: content_block_start\ndata: {}",
            serde_json::to_string(&block_start).unwrap()
        ));
    }

    // Emit text delta if present.
    if let Some(ref text) = delta.text {
        let block_delta = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text}
        });
        events.push(format!(
            "event: content_block_delta\ndata: {}",
            serde_json::to_string(&block_delta).unwrap()
        ));
    }

    // On the terminal chunk, close out the stream.
    if delta.done {
        let block_stop = serde_json::json!({"type": "content_block_stop", "index": 0});
        events.push(format!(
            "event: content_block_stop\ndata: {}",
            serde_json::to_string(&block_stop).unwrap()
        ));

        let stop_reason = delta.finish_reason.as_ref().map(|r| match r {
            FinishReason::Stop => "end_turn",
            FinishReason::Length => "max_tokens",
            FinishReason::ToolCalls => "tool_use",
        }).unwrap_or("end_turn");

        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": 0}
        });
        events.push(format!(
            "event: message_delta\ndata: {}",
            serde_json::to_string(&msg_delta).unwrap()
        ));

        let msg_stop = serde_json::json!({"type": "message_stop"});
        events.push(format!(
            "event: message_stop\ndata: {}",
            serde_json::to_string(&msg_stop).unwrap()
        ));
    }

    events
}

fn parse_role(s: &str) -> Result<Role, String> {
    match s {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(format!("Unknown role: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Conversion: ChatResult → AnthResponse
// ---------------------------------------------------------------------------

/// Convert an internal `ChatResult` into an Anthropic Messages API response.
pub fn from_internal(res: ChatResult, model: &str) -> AnthResponse {
    let id = format!("msg_{}", Uuid::new_v4());

    let stop_reason = match res.finish_reason {
        FinishReason::Stop => "end_turn",
        FinishReason::Length => "max_tokens",
        FinishReason::ToolCalls => "tool_use",
    }
    .to_string();

    let content = res
        .content
        .into_iter()
        .map(|part| match part {
            ContentPart::Text(text) => AnthContentBlock::Text { text },
            ContentPart::Call(tc) => {
                let input = serde_json::from_str(&tc.arguments)
                    .unwrap_or(serde_json::Value::Null);
                AnthContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.name,
                    input,
                }
            }
        })
        .collect();

    AnthResponse {
        id,
        r#type: "message".to_string(),
        role: "assistant".to_string(),
        model: model.to_string(),
        content,
        stop_reason,
        usage: AnthUsage {
            input_tokens: res.prompt_tokens,
            output_tokens: res.completion_tokens,
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    // --- Streaming renderer tests ---

    #[test]
    fn anthropic_emits_content_block_delta() {
        let d = StreamDelta { text: Some("hi".into()), done: false, finish_reason: None };
        let evs = stream_events(&d, /*started=*/true);
        assert!(evs.iter().any(|e| e.contains("content_block_delta")));
    }

    #[test]
    fn anthropic_done_produces_message_stop() {
        let d = StreamDelta {
            text: None,
            done: true,
            finish_reason: Some(FinishReason::Stop),
        };
        let evs = stream_events(&d, /*started=*/true);
        assert!(evs.iter().any(|e| e.contains("message_stop")));
        assert!(evs.iter().any(|e| e.contains("content_block_stop")));
    }

    #[test]
    fn anthropic_first_chunk_includes_message_start() {
        let d = StreamDelta { text: Some("hello".into()), done: false, finish_reason: None };
        let evs = stream_events(&d, /*started=*/false);
        assert!(evs.iter().any(|e| e.contains("message_start")));
        assert!(evs.iter().any(|e| e.contains("content_block_start")));
        assert!(evs.iter().any(|e| e.contains("content_block_delta")));
    }
}
