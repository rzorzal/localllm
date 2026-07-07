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

/// A single incremental chunk emitted by the streaming engine.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamDelta {
    /// Incremental text content for this chunk (None if empty/tool-call chunk).
    pub text: Option<String>,
    /// True when this is the final chunk (finish_reason is set).
    pub done: bool,
    /// Finish reason carried on the terminal chunk.
    pub finish_reason: Option<FinishReason>,
}

/// Drop tools whose name is in `disabled` (case-sensitive exact match),
/// preserving the order of the rest. `disabled` empty = passthrough.
pub fn filter_tools(tools: Vec<ToolSpec>, disabled: &[String]) -> Vec<ToolSpec> {
    if disabled.is_empty() {
        return tools;
    }
    tools
        .into_iter()
        .filter(|t| !disabled.iter().any(|d| d == &t.name))
        .collect()
}

/// Keep only the last `keep_turns` conversation turns, plus all leading system
/// messages. A turn begins at a `Role::User` message and runs until the next
/// `Role::User`, so cutting on a user boundary never splits a `tool_use` from
/// its `tool_result` (both live inside the same turn). `None` = passthrough.
pub fn truncate_history(messages: Vec<ChatMessage>, keep_turns: Option<u32>) -> Vec<ChatMessage> {
    let Some(n) = keep_turns else { return messages };
    let n = n as usize;

    // Leading system messages are always kept.
    let lead_sys = messages
        .iter()
        .take_while(|m| m.role == Role::System)
        .count();

    if n == 0 {
        // Keep only leading system messages.
        return messages.into_iter().take(lead_sys).collect();
    }

    // Indices (in the full vec) where a turn starts.
    let user_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect();

    if user_starts.len() <= n {
        return messages; // nothing to drop
    }

    // Start of the first turn we keep.
    let cut = user_starts[user_starts.len() - n];

    let mut out = Vec::with_capacity(lead_sys + (messages.len() - cut));
    out.extend(messages.iter().take(lead_sys).cloned());
    out.extend(messages.iter().skip(cut).cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(role: Role, text: &str) -> ChatMessage {
        ChatMessage {
            role,
            text: Some(text.into()),
            tool_calls: vec![],
            tool_result: None,
        }
    }

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

    #[test]
    fn truncate_keeps_system_and_last_n_turns() {
        let msgs = vec![
            m(Role::System, "sys"),
            m(Role::User, "u1"),
            m(Role::Assistant, "a1"),
            m(Role::User, "u2"),
            m(Role::Assistant, "a2"),
        ];
        let out = truncate_history(msgs, Some(1));
        let texts: Vec<_> = out.iter().map(|x| x.text.clone().unwrap()).collect();
        assert_eq!(texts, vec!["sys", "u2", "a2"]);
    }

    #[test]
    fn truncate_never_splits_a_tool_turn() {
        // Turn 1 has a tool call + result; keeping 1 turn must keep turn 2 whole.
        let msgs = vec![
            m(Role::System, "sys"),
            m(Role::User, "u1"),
            ChatMessage {
                role: Role::Assistant,
                text: None,
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "t".into(),
                    arguments: "{}".into(),
                }],
                tool_result: None,
            },
            ChatMessage {
                role: Role::Tool,
                text: None,
                tool_calls: vec![],
                tool_result: Some(ToolResult {
                    tool_call_id: "c1".into(),
                    content: "ok".into(),
                }),
            },
            m(Role::Assistant, "a1"),
            m(Role::User, "u2"),
            m(Role::Assistant, "a2"),
        ];
        let out = truncate_history(msgs.clone(), Some(1));
        let roles: Vec<_> = out.iter().map(|x| x.role.clone()).collect();
        assert_eq!(roles, vec![Role::System, Role::User, Role::Assistant]);
        // Keeping 2 turns returns everything.
        assert_eq!(truncate_history(msgs.clone(), Some(2)).len(), msgs.len());
    }

    #[test]
    fn truncate_none_is_passthrough() {
        let msgs = vec![m(Role::System, "sys"), m(Role::User, "u1")];
        assert_eq!(truncate_history(msgs.clone(), None), msgs);
    }

    #[test]
    fn truncate_zero_turns_keeps_only_leading_system() {
        let msgs = vec![
            m(Role::System, "sys"),
            m(Role::User, "u1"),
            m(Role::Assistant, "a1"),
            m(Role::User, "u2"),
            m(Role::Assistant, "a2"),
        ];
        let out = truncate_history(msgs, Some(0));
        let texts: Vec<_> = out.iter().map(|x| x.text.clone().unwrap()).collect();
        assert_eq!(texts, vec!["sys"]);
    }

    #[test]
    fn truncate_zero_turns_no_system_is_empty() {
        let msgs = vec![m(Role::User, "u1"), m(Role::Assistant, "a1")];
        assert!(truncate_history(msgs, Some(0)).is_empty());
    }

    fn tool(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "d".into(),
            parameters: serde_json::json!({}),
        }
    }

    #[test]
    fn filter_tools_drops_named_and_preserves_order() {
        let tools = vec![tool("Bash"), tool("Read"), tool("Glob"), tool("Edit")];
        let out = filter_tools(tools, &["Read".to_string(), "Glob".to_string()]);
        let names: Vec<_> = out.iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, vec!["Bash".to_string(), "Edit".to_string()]);
    }

    #[test]
    fn filter_tools_empty_disabled_is_passthrough() {
        let tools = vec![tool("Bash"), tool("Read")];
        assert_eq!(filter_tools(tools.clone(), &[]), tools);
    }

    #[test]
    fn filter_tools_unknown_name_is_noop() {
        let tools = vec![tool("Bash")];
        assert_eq!(filter_tools(tools.clone(), &["Nope".to_string()]), tools);
    }
}
