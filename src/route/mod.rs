//! Local-vs-cloud routing: a pure decision function plus its policy knobs.
//!
//! The decision logic (`decide`) performs no I/O so it is unit-testable in
//! isolation. The HTTP layer turns a `Decision` into either the local engine
//! path or a cloud reverse-proxy.

pub mod policy;

pub use policy::{Profile, RoutingPolicy};

use crate::api::common::ChatRequest;

/// Cheap, pre-generation signals used to decide where a request runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signals {
    /// Estimated prompt size in tokens.
    pub prompt_tokens: usize,
    /// Local model's usable context window (config `ctx_len`).
    pub local_ctx_window: usize,
    pub n_tools: usize,
    pub n_messages: usize,
    /// Whether the incoming request carries cloud credentials we can forward.
    pub has_cloud_creds: bool,
}

/// Why a request was sent to cloud.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteReason {
    /// Prompt exceeds the local context window.
    ContextOverflow,
    /// Difficulty score above the profile threshold (Phase B).
    Difficulty,
    /// Profile forces cloud (e.g. MaxQuality, Phase B).
    Profile,
}

/// The routing outcome for a single request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Serve locally, no escalation.
    Local,
    /// Serve locally; escalate to cloud if the result is weak (Phase B,
    /// buffered paths only).
    LocalThenCascade,
    /// Skip local; reverse-proxy to the provider now.
    Cloud(RouteReason),
    /// Cloud was wanted but no usable credentials / cloud disallowed; serve
    /// local and warn (local will reject an over-window prompt cleanly).
    LocalNoCreds,
}

/// Decide where a request runs from cheap signals and the active policy.
///
/// Phase A implements only the hard context-size gate. Difficulty scoring and
/// cascade are added in Phase B; in-window requests therefore return `Local`.
pub fn decide(s: &Signals, p: &RoutingPolicy) -> Decision {
    // Phase A: gate on prompt_tokens only; request's max_tokens output reservation
    // is intentionally ignored here — conservative ctx_gate_frac + the over-window
    // backstop cover it; output-reservation accounting is deferred to Phase B.
    let over_window =
        s.prompt_tokens as f64 > s.local_ctx_window as f64 * p.ctx_gate_frac;
    if over_window {
        if p.allow_cloud && s.has_cloud_creds {
            return Decision::Cloud(RouteReason::ContextOverflow);
        }
        return Decision::LocalNoCreds;
    }
    Decision::Local
}

/// Estimate the prompt token count with a cheap `chars / 4` heuristic over all
/// message text, tool-call arguments, tool-result content, and tool schemas.
/// Conservative and good enough for the context gate; an exact local tokenizer
/// can replace this later if the boundary proves too coarse.
pub fn estimate_prompt_tokens(req: &ChatRequest) -> usize {
    let mut chars = 0usize;
    for m in &req.messages {
        if let Some(t) = &m.text {
            chars += t.len();
        }
        for c in &m.tool_calls {
            chars += c.name.len() + c.arguments.len();
        }
        if let Some(tr) = &m.tool_result {
            chars += tr.content.len();
        }
    }
    for t in &req.tools {
        chars += t.name.len() + t.description.len() + t.parameters.to_string().len();
    }
    chars / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::{ChatMessage, ChatRequest, Role};

    fn pol() -> RoutingPolicy {
        Profile::SaveTokens.policy()
    }

    fn sig(prompt_tokens: usize, has_creds: bool, window: usize) -> Signals {
        Signals {
            prompt_tokens,
            local_ctx_window: window,
            n_tools: 0,
            n_messages: 1,
            has_cloud_creds: has_creds,
        }
    }

    #[test]
    fn under_window_stays_local() {
        assert_eq!(decide(&sig(100, true, 1000), &pol()), Decision::Local);
    }

    #[test]
    fn over_window_with_creds_goes_cloud() {
        // 0.95 * 1000 = 950; 980 > 950 → overflow.
        assert_eq!(
            decide(&sig(980, true, 1000), &pol()),
            Decision::Cloud(RouteReason::ContextOverflow)
        );
    }

    #[test]
    fn over_window_without_creds_stays_local_no_creds() {
        assert_eq!(decide(&sig(980, false, 1000), &pol()), Decision::LocalNoCreds);
    }

    #[test]
    fn local_only_never_routes_cloud_even_on_overflow() {
        let p = Profile::LocalOnly.policy();
        assert_eq!(decide(&sig(5000, true, 1000), &p), Decision::LocalNoCreds);
    }

    #[test]
    fn estimate_tokens_counts_message_text() {
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                text: Some("a".repeat(400)), // 400 chars / 4 = 100 tokens
                tool_calls: vec![],
                tool_result: None,
            }],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
            model: "m".into(),
        };
        assert_eq!(estimate_prompt_tokens(&req), 100);
    }
}
