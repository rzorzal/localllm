// OpenAI Responses API translation layer (stateless subset for Codex).
use serde::Deserialize;
use crate::api::common::{ChatMessage, ChatRequest, Role, ToolCall, ToolResult, ToolSpec};

#[derive(Debug, Clone, Deserialize)]
pub struct RespRequest {
    pub model: String,
    #[serde(default)]
    pub instructions: Option<String>,
    pub input: RespInput,
    #[serde(default)]
    pub tools: Option<Vec<RespTool>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RespInput {
    Text(String),
    Items(Vec<RespItem>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum RespItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        #[serde(default)]
        content: Option<RespContent>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RespContent {
    Text(String),
    Parts(Vec<RespPart>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum RespPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RespTool {
    #[serde(rename = "type")]
    pub r#type: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

fn content_to_text(c: Option<RespContent>) -> Option<String> {
    let s = match c {
        None => String::new(),
        Some(RespContent::Text(s)) => s,
        Some(RespContent::Parts(parts)) => parts
            .into_iter()
            .filter_map(|p| match p {
                RespPart::InputText { text } | RespPart::OutputText { text } => Some(text),
                RespPart::Other => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    };
    if s.is_empty() { None } else { Some(s) }
}

pub fn to_internal(req: RespRequest) -> Result<ChatRequest, String> {
    let mut messages: Vec<ChatMessage> = Vec::new();

    if let Some(instr) = req.instructions.filter(|s| !s.is_empty()) {
        messages.push(ChatMessage {
            role: Role::System,
            text: Some(instr),
            tool_calls: vec![],
            tool_result: None,
        });
    }

    match req.input {
        RespInput::Text(s) => messages.push(ChatMessage {
            role: Role::User,
            text: if s.is_empty() { None } else { Some(s) },
            tool_calls: vec![],
            tool_result: None,
        }),
        RespInput::Items(items) => {
            for item in items {
                match item {
                    RespItem::Message { role, content } => {
                        let role = match role.as_str() {
                            "system" | "developer" => Role::System,
                            "user" => Role::User,
                            "assistant" => Role::Assistant,
                            "tool" => Role::Tool,
                            other => return Err(format!("Unknown role: {other}")),
                        };
                        messages.push(ChatMessage {
                            role,
                            text: content_to_text(content),
                            tool_calls: vec![],
                            tool_result: None,
                        });
                    }
                    RespItem::FunctionCall { call_id, name, arguments } => {
                        // Assistant's prior tool call, replayed in a stateless multi-turn
                        // tool loop. Preserve it so its function_call_output has a matching
                        // preceding call (orphaned tool results break chat templates).
                        messages.push(ChatMessage {
                            role: Role::Assistant,
                            text: None,
                            tool_calls: vec![ToolCall { id: call_id, name, arguments }],
                            tool_result: None,
                        });
                    }
                    RespItem::FunctionCallOutput { call_id, output } => {
                        messages.push(ChatMessage {
                            role: Role::Tool,
                            text: None,
                            tool_calls: vec![],
                            tool_result: Some(ToolResult { tool_call_id: call_id, content: output }),
                        });
                    }
                    RespItem::Other => {} // reasoning/image/etc — ignored
                }
            }
        }
    }

    let tools = req
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| ToolSpec { name: t.name, description: t.description, parameters: t.parameters })
        .collect();

    Ok(ChatRequest {
        messages,
        tools,
        max_tokens: req.max_output_tokens,
        temperature: req.temperature,
        stream: req.stream.unwrap_or(false),
        model: req.model,
    })
}

use crate::api::common::{ChatResult, ContentPart, FinishReason};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

pub fn from_internal(res: ChatResult, model: &str) -> serde_json::Value {
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

    let mut output = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    let mut calls: Vec<serde_json::Value> = Vec::new();

    for part in res.content {
        match part {
            ContentPart::Text(t) => text_parts.push(t),
            ContentPart::Call(tc) => calls.push(serde_json::json!({
                "type": "function_call",
                "id": format!("fc_{}", Uuid::new_v4()),
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "status": "completed"
            })),
        }
    }

    if !text_parts.is_empty() {
        output.push(serde_json::json!({
            "type": "message",
            "id": format!("msg_{}", Uuid::new_v4()),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text_parts.join(""), "annotations": []}]
        }));
    }
    output.extend(calls);

    let (status, incomplete) = match res.finish_reason {
        FinishReason::Length => (
            "incomplete",
            Some(serde_json::json!({"reason": "max_output_tokens"})),
        ),
        _ => ("completed", None),
    };

    let mut obj = serde_json::json!({
        "id": format!("resp_{}", Uuid::new_v4()),
        "object": "response",
        "created_at": created,
        "model": model,
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": res.prompt_tokens,
            "output_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens
        }
    });
    if let Some(d) = incomplete {
        obj["incomplete_details"] = d;
    }
    obj
}

pub fn stream_events_from_result(
    res: &ChatResult,
    resp_id: &str,
    model: &str,
) -> Vec<(&'static str, String)> {
    let mut seq = 0u64;
    let mut out: Vec<(&'static str, String)> = Vec::new();
    let mut push = |ty: &'static str, mut data: serde_json::Value| {
        data["type"] = serde_json::Value::String(ty.to_string());
        data["sequence_number"] = serde_json::json!(seq);
        seq += 1;
        out.push((ty, serde_json::to_string(&data).unwrap()));
    };

    let in_progress = serde_json::json!({
        "response": {"id": resp_id, "object": "response", "model": model, "status": "in_progress"}
    });
    push("response.created", in_progress);

    let mut output_index = 0u64;
    for part in &res.content {
        match part {
            ContentPart::Text(t) => {
                let item_id = format!("msg_{}", Uuid::new_v4());
                push("response.output_item.added", serde_json::json!({
                    "output_index": output_index,
                    "item": {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}
                }));
                push("response.content_part.added", serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []}
                }));
                push("response.output_text.delta", serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "content_index": 0, "delta": t
                }));
                push("response.output_text.done", serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "content_index": 0, "text": t
                }));
                push("response.content_part.done", serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "content_index": 0,
                    "part": {"type": "output_text", "text": t, "annotations": []}
                }));
                push("response.output_item.done", serde_json::json!({
                    "output_index": output_index,
                    "item": {"type": "message", "id": item_id, "role": "assistant", "status": "completed",
                             "content": [{"type": "output_text", "text": t, "annotations": []}]}
                }));
                output_index += 1;
            }
            ContentPart::Call(tc) => {
                let item_id = format!("fc_{}", Uuid::new_v4());
                push("response.output_item.added", serde_json::json!({
                    "output_index": output_index,
                    "item": {"type": "function_call", "id": item_id, "call_id": tc.id,
                             "name": tc.name, "arguments": "", "status": "in_progress"}
                }));
                push("response.function_call_arguments.delta", serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "delta": tc.arguments
                }));
                push("response.function_call_arguments.done", serde_json::json!({
                    "item_id": item_id, "output_index": output_index, "arguments": tc.arguments
                }));
                push("response.output_item.done", serde_json::json!({
                    "output_index": output_index,
                    "item": {"type": "function_call", "id": item_id, "call_id": tc.id,
                             "name": tc.name, "arguments": tc.arguments, "status": "completed"}
                }));
                output_index += 1;
            }
        }
    }

    // Reuse the non-streaming render for the terminal payload, but override the
    // freshly-minted id so it matches the resp_id announced in response.created.
    let mut resp_obj = from_internal(res.clone(), model);
    resp_obj["id"] = serde_json::Value::String(resp_id.to_string());
    let completed = serde_json::json!({ "response": resp_obj });
    push("response.completed", completed);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_input_becomes_single_user_message() {
        let json = r#"{"model":"m","input":"hi"}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages.len(), 1);
        assert_eq!(internal.messages[0].role, Role::User);
        assert_eq!(internal.messages[0].text.as_deref(), Some("hi"));
    }

    #[test]
    fn instructions_become_leading_system_message() {
        let json = r#"{"model":"m","instructions":"be terse","input":"hi"}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::System);
        assert_eq!(internal.messages[0].text.as_deref(), Some("be terse"));
        assert_eq!(internal.messages[1].role, Role::User);
    }

    #[test]
    fn array_input_message_and_function_call_output() {
        let json = r#"{"model":"m","input":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
            {"type":"function_call_output","call_id":"call_1","output":"sunny"}
        ]}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages[0].role, Role::User);
        assert_eq!(internal.messages[0].text.as_deref(), Some("weather?"));
        assert_eq!(internal.messages[1].role, Role::Tool);
        let tr = internal.messages[1].tool_result.as_ref().unwrap();
        assert_eq!(tr.tool_call_id, "call_1");
        assert_eq!(tr.content, "sunny");
    }

    #[test]
    fn assistant_function_call_item_becomes_assistant_tool_call() {
        // Codex replays the prior assistant function_call alongside its output in a
        // stateless tool loop; the call must be preserved before its result.
        let json = r#"{"model":"m","input":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
            {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Recife\"}"},
            {"type":"function_call_output","call_id":"call_1","output":"sunny"}
        ]}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages.len(), 3);
        assert_eq!(internal.messages[1].role, Role::Assistant);
        let tc = &internal.messages[1].tool_calls[0];
        assert_eq!(tc.id, "call_1");
        assert_eq!(tc.name, "get_weather");
        assert_eq!(tc.arguments, r#"{"city":"Recife"}"#);
        assert_eq!(internal.messages[2].role, Role::Tool);
    }

    #[test]
    fn flat_tools_map_to_toolspec() {
        let json = r#"{"model":"m","input":"hi",
            "tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object"}}]}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.tools[0].name, "get_weather");
    }

    #[test]
    fn unknown_message_role_errors() {
        let json = r#"{"model":"m","input":[{"type":"message","role":"frobnicate","content":"x"}]}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        assert!(to_internal(req).is_err());
    }

    #[test]
    fn unknown_item_type_is_ignored_not_error() {
        let json = r#"{"model":"m","input":[
            {"type":"reasoning","summary":[]},
            {"type":"message","role":"user","content":"hi"}
        ]}"#;
        let req: RespRequest = serde_json::from_str(json).unwrap();
        let internal = to_internal(req).unwrap();
        assert_eq!(internal.messages.len(), 1);
        assert_eq!(internal.messages[0].text.as_deref(), Some("hi"));
    }

    #[test]
    fn renders_output_text_message() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let res = ChatResult {
            content: vec![ContentPart::Text("hello".into())],
            finish_reason: FinishReason::Stop,
            prompt_tokens: 3,
            completion_tokens: 2,
        };
        let v = from_internal(res, "m");
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(v["output"][0]["content"][0]["text"], "hello");
        assert_eq!(v["usage"]["input_tokens"], 3);
        assert_eq!(v["usage"]["output_tokens"], 2);
        assert_eq!(v["usage"]["total_tokens"], 5);
    }

    #[test]
    fn renders_function_call_item() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason, ToolCall};
        let res = ChatResult {
            content: vec![ContentPart::Call(ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Recife"}"#.into(),
            })],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 5,
            completion_tokens: 4,
        };
        let v = from_internal(res, "m");
        let fc = &v["output"][0];
        assert_eq!(fc["type"], "function_call");
        assert_eq!(fc["call_id"], "call_1");
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["arguments"], r#"{"city":"Recife"}"#);
        assert_eq!(v["status"], "completed");
    }

    #[test]
    fn length_finish_marks_incomplete() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let res = ChatResult {
            content: vec![ContentPart::Text("partial".into())],
            finish_reason: FinishReason::Length,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let v = from_internal(res, "m");
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    }

    fn event_types(pairs: &[(&'static str, String)]) -> Vec<&'static str> {
        pairs.iter().map(|(t, _)| *t).collect()
    }

    #[test]
    fn text_stream_event_order() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let res = ChatResult {
            content: vec![ContentPart::Text("hi".into())],
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let ev = stream_events_from_result(&res, "resp_x", "m");
        assert_eq!(
            event_types(&ev),
            vec![
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        // the delta carries the text
        let delta = ev.iter().find(|(t, _)| *t == "response.output_text.delta").unwrap();
        assert!(delta.1.contains("\"delta\":\"hi\""));
        // monotonic sequence_number starting at 0
        let first: serde_json::Value = serde_json::from_str(&ev[0].1).unwrap();
        assert_eq!(first["sequence_number"], 0);
    }

    #[test]
    fn created_and_completed_share_response_id() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let res = ChatResult {
            content: vec![ContentPart::Text("hi".into())],
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let ev = stream_events_from_result(&res, "resp_fixed", "m");
        let created: serde_json::Value =
            serde_json::from_str(&ev.iter().find(|(t, _)| *t == "response.created").unwrap().1)
                .unwrap();
        let completed: serde_json::Value =
            serde_json::from_str(&ev.iter().find(|(t, _)| *t == "response.completed").unwrap().1)
                .unwrap();
        assert_eq!(created["response"]["id"], "resp_fixed");
        assert_eq!(completed["response"]["id"], "resp_fixed");
    }

    #[test]
    fn sequence_numbers_are_monotonic() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let res = ChatResult {
            content: vec![ContentPart::Text("hi".into())],
            finish_reason: FinishReason::Stop,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let ev = stream_events_from_result(&res, "resp_x", "m");
        for (i, (_, data)) in ev.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(data).unwrap();
            assert_eq!(v["sequence_number"], i as u64);
        }
    }

    #[test]
    fn tool_stream_emits_function_call_sequence() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason, ToolCall};
        let res = ChatResult {
            content: vec![ContentPart::Call(ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            })],
            finish_reason: FinishReason::ToolCalls,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let ev = stream_events_from_result(&res, "resp_x", "m");
        let types = event_types(&ev);
        assert_eq!(types.first(), Some(&"response.created"));
        assert!(types.contains(&"response.function_call_arguments.delta"));
        assert!(types.contains(&"response.function_call_arguments.done"));
        assert_eq!(types.last(), Some(&"response.completed"));
        // the function_call item appears via output_item.added
        let added = ev.iter().find(|(t, _)| *t == "response.output_item.added").unwrap();
        assert!(added.1.contains("get_weather"));
    }
}
