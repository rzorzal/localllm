// OpenAI Responses API translation layer (stateless subset for Codex).
use serde::Deserialize;
use crate::api::common::{ChatMessage, ChatRequest, Role, ToolResult, ToolSpec};

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
}
