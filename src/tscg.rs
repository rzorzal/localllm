//! # tscg — Tool-Schema Compact Grammar
//!
//! Compresses JSON Schema tool definitions into a compact textual form,
//! suitable for embedding in LLM prompts. Achieves ~40-60% size reduction
//! vs. pretty-printed JSON by dropping schema boilerplate while preserving
//! names, types, enums, required-ness, and non-empty descriptions.
//!
//! **Fidelity:** lossless for flat schemas; nested object/array structure is
//! summarised (name + container type preserved, deep shape omitted — a nested
//! object param becomes `obj`, dropping its inner properties, and `format`/
//! `const`/numeric-range keywords are not emitted).
//!
//! ## Grammar
//!
//! ```text
//! block     ::= (tool "\n")*
//! tool      ::= name "(" params ")" " — " description
//!             | name "()"                           (no description)
//! params    ::= param ("," SP param)*
//!             | ""                                  (no params)
//! param     ::= name ":" type req? enum_clause? desc_clause?
//! req       ::= "!"
//! enum_clause ::= "=enum[" value ("," value)* "]"
//! desc_clause ::= " (" description ")"
//! type      ::= "string" | "integer" | "number" | "boolean" | "null"
//!             | "array" | "array<" type ">"
//!             | "obj"
//!             | "any"          (unknown/missing type)
//! ```
//!
//! ## Boilerplate dropped (carries no selection signal)
//!
//! - Top-level `"type": "object"`, `"additionalProperties"`, `"$schema"`, `"title"`
//! - Empty or whitespace-only `"description"` fields
//!
//! ## Examples
//!
//! ```text
//! get_weather(location:string! (City and country), unit:string=enum[celsius,fahrenheit]) — Get current weather
//! read_file(path:string!, encoding:string=enum[utf-8,latin-1,base64] (File encoding)) — Read a file from disk
//! noop() — Does nothing
//! ```

use crate::api::common::ToolSpec;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compact a single [`ToolSpec`] into one line using the tscg grammar.
pub fn compact_tool(tool: &ToolSpec) -> String {
    let params = format_params(&tool.parameters);
    let desc = tool.description.trim();
    if desc.is_empty() {
        format!("{}({})", tool.name, params)
    } else {
        format!("{}({}) \u{2014} {}", tool.name, params, desc)
    }
}

/// Compact a slice of [`ToolSpec`]s, one tool per line.
pub fn compact_tools_block(tools: &[ToolSpec]) -> String {
    tools
        .iter()
        .map(compact_tool)
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn format_params(parameters: &serde_json::Value) -> String {
    let properties = match parameters.get("properties").and_then(|v| v.as_object()) {
        Some(p) => p,
        None => return String::new(),
    };

    if properties.is_empty() {
        return String::new();
    }

    let required: Vec<&str> = parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let parts: Vec<String> = properties
        .iter()
        .map(|(name, schema)| format_param(name, schema, &required))
        .collect();

    parts.join(", ")
}

fn format_param(name: &str, schema: &serde_json::Value, required: &[&str]) -> String {
    let type_str = infer_type(schema);
    let is_required = required.contains(&name);
    let req_marker = if is_required { "!" } else { "" };

    let enum_clause = schema
        .get("enum")
        .and_then(|v| v.as_array())
        .map(|arr| {
            let values: Vec<String> = arr
                .iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect();
            format!("=enum[{}]", values.join(","))
        })
        .unwrap_or_default();

    let desc = schema
        .get("description")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| format!(" ({})", s))
        .unwrap_or_default();

    format!("{}:{}{}{}{}", name, type_str, req_marker, enum_clause, desc)
}

/// Infer a compact type string from a JSON Schema value.
fn infer_type(schema: &serde_json::Value) -> String {
    match schema.get("type").and_then(|v| v.as_str()) {
        Some("string") => "string".to_string(),
        Some("integer") => "integer".to_string(),
        Some("number") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("null") => "null".to_string(),
        Some("object") => "obj".to_string(),
        Some("array") => {
            // Try to get items.type for array<T> annotation
            if let Some(item_type) = schema
                .get("items")
                .and_then(|i| i.get("type"))
                .and_then(|t| t.as_str())
            {
                format!("array<{}>", item_type)
            } else {
                "array".to_string()
            }
        }
        Some(_) => "any".to_string(),
        None => {
            // Implicitly an object if it has "properties"
            if schema.get("properties").is_some() {
                "obj".to_string()
            } else {
                "any".to_string()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_tool(name: &str, description: &str, parameters: serde_json::Value) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: single tool — required string param + enum param
    // -----------------------------------------------------------------------
    #[test]
    fn compact_tool_required_and_enum_params() {
        let tool = make_tool(
            "get_weather",
            "Get current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }),
        );

        let compact = compact_tool(&tool);
        let pretty = serde_json::to_string_pretty(&tool.parameters).unwrap();

        assert!(compact.contains("get_weather"), "missing tool name");
        assert!(compact.contains("location"), "missing location param");
        assert!(compact.contains("unit"), "missing unit param");
        assert!(compact.contains("celsius"), "missing enum value celsius");
        assert!(
            compact.contains("fahrenheit"),
            "missing enum value fahrenheit"
        );
        assert!(
            compact.contains("location:string!"),
            "missing required marker on location"
        );
        assert!(
            compact.len() < pretty.len(),
            "compact ({} chars) not shorter than pretty ({} chars)\ncompact: {}\npretty: {}",
            compact.len(),
            pretty.len(),
            compact,
            pretty
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: compact_tools_block puts each tool on its own line
    // -----------------------------------------------------------------------
    #[test]
    fn compact_tools_block_one_per_line() {
        let tools = vec![
            make_tool(
                "tool_a",
                "First tool",
                json!({"type":"object","properties":{"x":{"type":"string"}},"required":[]}),
            ),
            make_tool(
                "tool_b",
                "Second tool",
                json!({"type":"object","properties":{"y":{"type":"integer"}},"required":[]}),
            ),
            make_tool(
                "tool_c",
                "Third tool",
                json!({"type":"object","properties":{}}),
            ),
        ];

        let block = compact_tools_block(&tools);
        let line_count = block.lines().count();

        assert_eq!(
            line_count, 3,
            "expected 3 lines, got {}: {:?}",
            line_count, block
        );
        assert!(block.contains("tool_a"), "missing tool_a");
        assert!(block.contains("tool_b"), "missing tool_b");
        assert!(block.contains("tool_c"), "missing tool_c");
    }

    // -----------------------------------------------------------------------
    // Test 3: token-reduction — realistic 5-tool set <= 60% of pretty JSON
    // -----------------------------------------------------------------------
    #[test]
    fn compact_tools_block_token_reduction_5_tools() {
        let tools = vec![
            make_tool(
                "search_web",
                "Search the web for information",
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "query": { "type": "string", "description": "The search query string" },
                        "num_results": { "type": "integer", "description": "Number of results to return (default 10)" },
                        "region": { "type": "string", "enum": ["us", "uk", "eu", "au"], "description": "Geographic region filter" }
                    },
                    "required": ["query"]
                }),
            ),
            make_tool(
                "read_file",
                "Read the contents of a file from disk",
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute or relative path to the file" },
                        "encoding": { "type": "string", "enum": ["utf-8", "latin-1", "base64"], "description": "File encoding" },
                        "max_bytes": { "type": "integer", "description": "Maximum bytes to read (0 = unlimited)" }
                    },
                    "required": ["path"]
                }),
            ),
            make_tool(
                "execute_command",
                "Execute a shell command and return output",
                json!({
                    "type": "object",
                    "title": "ExecuteCommandInput",
                    "additionalProperties": false,
                    "properties": {
                        "command": { "type": "string", "description": "Shell command to execute" },
                        "working_dir": { "type": "string", "description": "Working directory for the command" },
                        "timeout_ms": { "type": "integer", "description": "Timeout in milliseconds" },
                        "shell": { "type": "string", "enum": ["bash", "sh", "zsh", "fish"], "description": "Shell interpreter to use" }
                    },
                    "required": ["command"]
                }),
            ),
            make_tool(
                "send_email",
                "Send an email message to one or more recipients",
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "to": { "type": "array", "description": "List of recipient email addresses" },
                        "subject": { "type": "string", "description": "Email subject line" },
                        "body": { "type": "string", "description": "Email body text" },
                        "cc": { "type": "array", "description": "Carbon copy recipients" },
                        "priority": { "type": "string", "enum": ["low", "normal", "high"], "description": "Email priority level" }
                    },
                    "required": ["to", "subject", "body"]
                }),
            ),
            make_tool(
                "create_database_record",
                "Insert a new record into a database table",
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "$schema": "http://json-schema.org/draft-07/schema",
                    "title": "CreateRecordInput",
                    "properties": {
                        "table": { "type": "string", "description": "Target database table name" },
                        "data": { "type": "object", "description": "Record data as key-value pairs" },
                        "conflict_strategy": { "type": "string", "enum": ["error", "ignore", "replace", "update"], "description": "What to do on unique constraint conflict" },
                        "return_id": { "type": "boolean", "description": "Whether to return the new record ID" }
                    },
                    "required": ["table", "data"]
                }),
            ),
        ];

        let pretty_block: String = tools
            .iter()
            .map(|t| serde_json::to_string_pretty(&t.parameters).unwrap())
            .collect::<Vec<_>>()
            .join("\n");

        let compact_block = compact_tools_block(&tools);
        let ratio = compact_block.len() as f64 / pretty_block.len() as f64;

        eprintln!(
            "Token reduction: compact={} chars, pretty={} chars, ratio={:.1}%",
            compact_block.len(),
            pretty_block.len(),
            ratio * 100.0
        );
        eprintln!("Compact block:\n{}", compact_block);

        assert!(
            ratio <= 0.60,
            "compact block is {:.1}% of pretty JSON (target <=60%)",
            ratio * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: robustness — missing/odd fields don't panic
    // -----------------------------------------------------------------------
    #[test]
    fn compact_tool_robust_missing_fields() {
        // No properties at all
        let tool_empty = make_tool("noop", "", json!({}));
        let result = compact_tool(&tool_empty);
        assert!(result.contains("noop"), "missing name in empty tool");
        assert!(result.contains("noop()"), "empty params should be ()");

        // Param with no type → "any"
        let tool_no_type = make_tool(
            "mystery",
            "Does something",
            json!({"type":"object","properties":{"arg":{"description":"some arg"}},"required":[]}),
        );
        let result2 = compact_tool(&tool_no_type);
        assert!(result2.contains("arg:any"), "untyped param should be 'any'");
    }
}
