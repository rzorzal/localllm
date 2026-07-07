// OpenAI HTTP translation layer

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::api::common::{
    ChatMessage, ChatRequest, ChatResult, ContentPart, FinishReason, Role, StreamDelta, ToolCall,
    ToolResult, ToolSpec,
};

// ---------------------------------------------------------------------------
// OpenAI wire types (Serialize + Deserialize — they cross the HTTP boundary)
// ---------------------------------------------------------------------------

/// The `function` sub-object inside a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// A tool entry in the request's `tools` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiTool {
    /// Always `"function"` for the current OpenAI spec.
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: OaiFunctionDef,
}

/// The `function` sub-object inside a tool-call (request or response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A tool-call entry — appears in assistant messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: OaiFunctionCall,
}

/// A message `content` value: a plain string, or an array of content parts
/// (newer OpenAI / Codex clients send the array form, e.g. text + image_url).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

impl OaiContent {
    /// Flatten to plain text: a bare string as-is, or the concatenation of the
    /// text parts (non-text parts like `image_url` are dropped).
    pub fn into_text(self) -> String {
        match self {
            OaiContent::Text(s) => s,
            OaiContent::Parts(parts) => parts
                .into_iter()
                .filter_map(|p| match p {
                    OaiContentPart::Text { text } => Some(text),
                    OaiContentPart::Other => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// A single part inside an array-form `content`. Only text is mapped; any other
/// part type (`image_url`, `input_audio`, …) is accepted and ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OaiContentPart {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

/// Flatten an optional `content` to optional plain text (empty → `None`).
fn content_to_text(content: Option<OaiContent>) -> Option<String> {
    let s = content.map(|c| c.into_text()).unwrap_or_default();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// A single message in the request `messages` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiMessage {
    pub role: String,
    /// Present for `user`, `system`/`developer`, and `tool` messages (may be
    /// null for assistant). String or array-of-parts form.
    #[serde(default)]
    pub content: Option<OaiContent>,
    /// `tool_call_id` is present only for `role:"tool"` messages.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// `tool_calls` is present only for `role:"assistant"` messages that invoke tools.
    #[serde(default)]
    pub tool_calls: Option<Vec<OaiToolCall>>,
}

/// The top-level Chat Completions request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiChatRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub tools: Option<Vec<OaiTool>>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// The assistant message inside a choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OaiToolCall>>,
}

/// A single choice in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiChoice {
    pub index: usize,
    pub message: OaiResponseMessage,
    pub finish_reason: String,
}

/// Token usage block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// The full Chat Completions response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OaiChoice>,
    pub usage: OaiUsage,
}

/// Response for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiModelList {
    pub object: String,
    pub data: Vec<OaiModelInfo>,
}

// ---------------------------------------------------------------------------
// Conversion: OaiChatRequest → ChatRequest
// ---------------------------------------------------------------------------

/// Convert an incoming OpenAI Chat Completions request into the internal
/// `ChatRequest` representation.
pub fn to_internal(req: OaiChatRequest) -> Result<ChatRequest, String> {
    let mut messages = Vec::with_capacity(req.messages.len());

    for msg in req.messages {
        let role = match msg.role.as_str() {
            // `developer` is the newer OpenAI name for the system role.
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(format!("Unknown role: {other}")),
        };

        let chat_msg = match role {
            Role::Tool => {
                // tool messages: tool_call_id + content are required
                let tool_call_id = msg
                    .tool_call_id
                    .ok_or_else(|| "tool message missing tool_call_id".to_string())?;
                let content = content_to_text(msg.content)
                    .ok_or_else(|| "tool message missing content".to_string())?;
                ChatMessage {
                    role,
                    text: None,
                    tool_calls: vec![],
                    tool_result: Some(ToolResult {
                        tool_call_id,
                        content,
                    }),
                }
            }
            Role::Assistant => {
                // assistant may carry tool_calls
                let tool_calls = msg
                    .tool_calls
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    })
                    .collect();
                ChatMessage {
                    role,
                    text: content_to_text(msg.content),
                    tool_calls,
                    tool_result: None,
                }
            }
            _ => ChatMessage {
                role,
                text: content_to_text(msg.content),
                tool_calls: vec![],
                tool_result: None,
            },
        };

        messages.push(chat_msg);
    }

    let tools = req
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| ToolSpec {
            name: t.function.name,
            description: t.function.description,
            parameters: t.function.parameters,
        })
        .collect();

    Ok(ChatRequest {
        messages,
        tools,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: req.stream.unwrap_or(false),
        model: req.model,
    })
}

// ---------------------------------------------------------------------------
// Conversion: ChatResult → OaiChatResponse
// ---------------------------------------------------------------------------

/// Convert an internal `ChatResult` into an OpenAI Chat Completions response.
pub fn from_internal(res: ChatResult, model: &str) -> OaiChatResponse {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let finish_reason = match res.finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
    .to_string();

    // Separate text parts and tool-call parts
    let mut text_parts: Vec<String> = vec![];
    let mut tool_calls: Vec<OaiToolCall> = vec![];

    for part in res.content {
        match part {
            ContentPart::Text(t) => text_parts.push(t),
            ContentPart::Call(tc) => tool_calls.push(OaiToolCall {
                id: tc.id,
                r#type: "function".to_string(),
                function: OaiFunctionCall {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            }),
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let tool_calls_opt = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let message = OaiResponseMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: tool_calls_opt,
    };

    let total_tokens = res.prompt_tokens + res.completion_tokens;

    OaiChatResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![OaiChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage: OaiUsage {
            prompt_tokens: res.prompt_tokens,
            completion_tokens: res.completion_tokens,
            total_tokens,
        },
    }
}

// ---------------------------------------------------------------------------
// Streaming: SSE chunk renderer
// ---------------------------------------------------------------------------

/// Render one `StreamDelta` as a single `data: {json}` SSE line (no trailing newline).
/// The caller is responsible for appending `\n\n` to form a valid SSE frame,
/// and for emitting `data: [DONE]\n\n` when the stream ends.
///
/// `started` indicates whether any chunk has been sent yet for this response.
/// When `started` is `false` (first chunk), `{"role":"assistant"}` is included
/// in the delta object per the OpenAI streaming spec. Set `started` to `true`
/// for every subsequent chunk.
pub fn stream_chunk(delta: &StreamDelta, id: &str, model: &str, started: bool) -> String {
    use crate::api::common::FinishReason;
    use std::time::{SystemTime, UNIX_EPOCH};

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let finish_reason_str: Option<&str> = delta.finish_reason.as_ref().map(|r| match r {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    });

    // Build the delta object: include "content" key ONLY when text is Some.
    // The OpenAI spec requires an empty object {} on the final (done) chunk
    // rather than {"content": null}.
    // On the first chunk (!started), include "role":"assistant" per spec.
    let mut delta_obj = serde_json::Map::new();
    if !started {
        delta_obj.insert(
            "role".to_string(),
            serde_json::Value::String("assistant".to_string()),
        );
    }
    if let Some(ref text) = delta.text {
        delta_obj.insert(
            "content".to_string(),
            serde_json::Value::String(text.clone()),
        );
    }

    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta_obj,
            "finish_reason": finish_reason_str
        }]
    });

    format!("data: {}", serde_json::to_string(&chunk).unwrap())
}

/// Render a COMPLETE `ChatResult` as a sequence of OpenAI streaming `data:`
/// lines (the caller appends the terminating `data: [DONE]`).
///
/// Used for "buffered streaming": when a streaming request carries tools, the
/// engine generates the full result non-streaming (reusing the tested tool-call
/// path) and we replay it here as SSE chunks. This guarantees correct
/// `tool_calls` framing that incremental streaming does not support.
pub fn stream_chunks_from_result(result: &ChatResult, id: &str, model: &str) -> Vec<String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // First chunk: role + content + any tool_calls.
    let mut delta_obj = serde_json::Map::new();
    delta_obj.insert("role".into(), "assistant".into());

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for (i, part) in result.content.iter().enumerate() {
        match part {
            ContentPart::Text(t) => text.push_str(t),
            ContentPart::Call(tc) => tool_calls.push(serde_json::json!({
                "index": i,
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments}
            })),
        }
    }
    if !text.is_empty() {
        delta_obj.insert("content".into(), serde_json::Value::String(text));
    }
    if !tool_calls.is_empty() {
        delta_obj.insert("tool_calls".into(), serde_json::Value::Array(tool_calls));
    }

    let finish = match result.finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    };

    let content_chunk = serde_json::json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta_obj, "finish_reason": null}]
    });
    let final_chunk = serde_json::json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
    });

    vec![
        format!("data: {}", serde_json::to_string(&content_chunk).unwrap()),
        format!("data: {}", serde_json::to_string(&final_chunk).unwrap()),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::*;

    #[test]
    fn buffered_stream_emits_tool_calls_then_finish() {
        let result = ChatResult {
            content: vec![ContentPart::Call(ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"location":"Recife"}"#.into(),
            })],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 5,
            completion_tokens: 3,
        };
        let lines = stream_chunks_from_result(&result, "chatcmpl-x", "m");
        assert_eq!(lines.len(), 2);
        // First chunk carries the tool call.
        assert!(lines[0].contains("\"tool_calls\""));
        assert!(lines[0].contains("get_weather"));
        assert!(lines[0].contains("Recife"));
        // Final chunk carries the finish reason.
        assert!(lines[1].contains("\"finish_reason\":\"tool_calls\""));
    }

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
            content: vec![ContentPart::Call(ToolCall {
                id: "c1".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            })],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 5,
            completion_tokens: 2,
        };
        let oai = from_internal(res, "m");
        assert_eq!(oai.choices[0].finish_reason, "tool_calls");
        assert_eq!(
            oai.choices[0].message.tool_calls.as_ref().unwrap()[0]
                .function
                .name,
            "get_weather"
        );
    }

    #[test]
    fn parses_array_content_and_ignores_non_text_parts() {
        // Codex/newer OpenAI shape: content is an array of parts (text + image).
        let json = r#"{"model":"m","messages":[
            {"role":"user","content":[
                {"type":"text","text":"describe this"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
            ]}
        ]}"#;
        let req: OaiChatRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::User);
        assert_eq!(internal.messages[0].text.as_deref(), Some("describe this"));
    }

    #[test]
    fn developer_role_maps_to_system() {
        let json = r#"{"model":"m","messages":[{"role":"developer","content":"be terse"}]}"#;
        let req: OaiChatRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::System);
        assert_eq!(internal.messages[0].text.as_deref(), Some("be terse"));
    }

    #[test]
    fn tool_message_with_array_content_flattens() {
        let json = r#"{"model":"m","messages":[
            {"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"sunny"}]}]}"#;
        let req: OaiChatRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(
            internal.messages[0].tool_result.as_ref().unwrap().content,
            "sunny"
        );
    }

    #[test]
    fn parses_tool_result_message() {
        let json = r#"{"model":"m","messages":[
            {"role":"tool","tool_call_id":"c1","content":"sunny"}]}"#;
        let req: OaiChatRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::Tool);
        assert_eq!(
            internal.messages[0].tool_result.as_ref().unwrap().content,
            "sunny"
        );
    }

    // --- Streaming renderer tests ---

    #[test]
    fn openai_chunk_has_delta_content() {
        // started=true: subsequent chunk — no role, has content
        let d = StreamDelta {
            text: Some("hi".into()),
            done: false,
            finish_reason: None,
        };
        let line = stream_chunk(&d, "chatcmpl-1", "m", true);
        assert!(line.starts_with("data: "));
        assert!(line.contains("\"content\":\"hi\""));
        // role should NOT appear on subsequent chunks
        let json_str = line.strip_prefix("data: ").unwrap();
        let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert!(
            !v["choices"][0]["delta"]
                .as_object()
                .unwrap()
                .contains_key("role"),
            "subsequent chunk must not carry role"
        );
    }

    #[test]
    fn openai_first_chunk_includes_role_assistant() {
        // started=false: first chunk — must include "role":"assistant" in delta
        let d = StreamDelta {
            text: Some("Hi".into()),
            done: false,
            finish_reason: None,
        };
        let line = stream_chunk(&d, "chatcmpl-0", "m", false);
        let json_str = line.strip_prefix("data: ").unwrap();
        let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(
            v["choices"][0]["delta"]["role"], "assistant",
            "first chunk delta must carry role:assistant"
        );
        assert_eq!(v["choices"][0]["delta"]["content"], "Hi");
    }

    #[test]
    fn openai_chunk_with_finish_reason_sets_finish_reason() {
        let d = StreamDelta {
            text: None,
            done: true,
            finish_reason: Some(FinishReason::Stop),
        };
        let line = stream_chunk(&d, "chatcmpl-2", "m", true);
        assert!(line.starts_with("data: "));
        let json_str = line.strip_prefix("data: ").unwrap();
        let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn openai_done_chunk_delta_omits_content_key() {
        // When text is None (terminal/done chunk), the OpenAI spec requires the
        // delta object to be {} — the "content" key must be ABSENT, not null.
        // started=true so role is also absent.
        let d = StreamDelta {
            text: None,
            done: true,
            finish_reason: Some(FinishReason::Stop),
        };
        let line = stream_chunk(&d, "chatcmpl-3", "m", true);
        let json_str = line.strip_prefix("data: ").unwrap();
        let v: serde_json::Value = serde_json::from_str(json_str).unwrap();
        // The delta object must be empty — no "content" key at all.
        let delta = &v["choices"][0]["delta"];
        assert!(
            delta
                .as_object()
                .map(|o| !o.contains_key("content"))
                .unwrap_or(false),
            "expected delta to be empty {{}}, got: {delta}"
        );
    }
}
