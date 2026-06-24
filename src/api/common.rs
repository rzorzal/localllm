#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON string containing the arguments.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text(String),
    Call(ToolCall),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: Role,
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_result: Option<ToolResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON schema for the tool parameters.
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub stream: bool,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResult {
    pub content: Vec<ContentPart>,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_holds_json_arguments() {
        let c = ToolCall {
            id: "c1".into(),
            name: "get_weather".into(),
            arguments: r#"{"location":"Recife"}"#.into(),
        };
        let v: serde_json::Value = serde_json::from_str(&c.arguments).unwrap();
        assert_eq!(v["location"], "Recife");
    }

    #[test]
    fn finish_reason_tool_calls_is_distinct() {
        assert_ne!(FinishReason::ToolCalls, FinishReason::Stop);
    }
}
