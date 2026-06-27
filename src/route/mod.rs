//! Local-vs-cloud routing: a pure decision function plus its policy knobs.
//!
//! The decision logic (`decide`) performs no I/O so it is unit-testable in
//! isolation. The HTTP layer turns a `Decision` into either the local engine
//! path or a cloud reverse-proxy.

pub mod policy;

pub use policy::{Profile, RoutingPolicy};

use crate::api::common::ChatRequest;

/// Cheap, pre-generation signals used to decide where a request runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signals {
    /// Estimated prompt size in tokens.
    pub prompt_tokens: usize,
    /// Local model's usable context window (config `ctx_len`).
    pub local_ctx_window: usize,
    pub n_tools: usize,
    pub n_messages: usize,
    /// Whether the incoming request carries cloud credentials we can forward.
    pub has_cloud_creds: bool,
    /// Active local model's parameter size in billions (0.0 = unknown/neutral).
    /// Shifts the difficulty threshold: weaker local → more cloud.
    pub local_capability_b: f32,
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

/// Cheap pre-generation difficulty score in `[0.0, 1.0]`. Higher means the
/// request is harder/bigger and more likely to need the cloud model.
///
/// Transparent heuristic blend (weights sum to 1.0); context fill dominates:
/// - context fill: how full the local window is (prompt_tokens / window)
/// - tool load: tool count, saturating at 12 (agentic complexity)
/// - depth: message count, saturating at 20 (long multi-turn is harder)
///
/// A learned router could replace this later; the `decide` interface is unchanged.
pub fn difficulty_score(s: &Signals) -> f64 {
    let ctx_fill = if s.local_ctx_window == 0 {
        1.0
    } else {
        (s.prompt_tokens as f64 / s.local_ctx_window as f64).min(1.0)
    };
    let tool_load = (s.n_tools as f64 / 12.0).min(1.0);
    let depth = (s.n_messages as f64 / 20.0).min(1.0);
    0.6 * ctx_fill + 0.25 * tool_load + 0.15 * depth
}

/// Decide where a request runs from cheap signals and the active policy.
///
/// Order: (1) hard context gate, (2) cloud-impossible shortcut, (3) difficulty
/// score vs the profile threshold, (4) in-window low-score → cascade or local.
pub fn decide(s: &Signals, p: &RoutingPolicy) -> Decision {
    // 1. Hard context gate: prompt too big for the local window.
    let over_window =
        s.prompt_tokens as f64 > s.local_ctx_window as f64 * p.ctx_gate_frac;
    if over_window {
        if p.allow_cloud && s.has_cloud_creds {
            return Decision::Cloud(RouteReason::ContextOverflow);
        }
        return Decision::LocalNoCreds;
    }

    // 2. Cloud impossible (profile forbids it or no credential to forward) → local.
    if !p.allow_cloud || !s.has_cloud_creds {
        return Decision::Local;
    }

    // 3. Difficulty above the *capability-adjusted* threshold → cloud now.
    //    Weaker local model (cap < 7B) lowers the cutoff (more cloud); stronger
    //    (cap > 7B) raises it (more local); unknown (0.0) → no change.
    let adj = if s.local_capability_b > 0.0 {
        ((s.local_capability_b - 7.0) as f64 * 0.03).clamp(-0.3, 0.3)
    } else {
        0.0
    };
    let effective_threshold = (p.escalation_threshold + adj).clamp(0.0, 1.0);
    if difficulty_score(s) > effective_threshold {
        return Decision::Cloud(RouteReason::Difficulty);
    }

    // 4. In-window, below threshold: local, with cascade fallback when enabled.
    if p.cascade {
        Decision::LocalThenCascade
    } else {
        Decision::Local
    }
}

/// Whether a local result is "weak" enough to escalate to cloud. A length
/// truncation means the local model hit its token budget mid-answer — a strong
/// signal it under-served the request. (Logprob/judge confidence is future work.)
pub fn is_weak_result(r: &crate::api::common::ChatResult) -> bool {
    r.finish_reason == crate::api::common::FinishReason::Length
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
            local_capability_b: 0.0,
        }
    }

    #[test]
    fn under_window_with_creds_and_cascade_returns_local_then_cascade() {
        // SaveTokens has cascade=true and a high threshold; a tiny in-window
        // request scores well below it, so it serves local with cascade fallback.
        assert_eq!(
            decide(&sig(100, true, 1000), &pol()),
            Decision::LocalThenCascade
        );
    }

    #[test]
    fn under_window_without_creds_is_plain_local() {
        // No credential to forward → cannot cascade → plain Local.
        assert_eq!(decide(&sig(100, false, 1000), &pol()), Decision::Local);
    }

    #[test]
    fn high_difficulty_in_window_routes_cloud() {
        // Balanced threshold 0.6; a nearly-full window pushes the score over it.
        let p = Profile::Balanced.policy();
        // 850/1000 = 0.85 ctx fill (under the 0.9 ctx_gate, so NOT overflow),
        // score = 0.6*0.85 = 0.51 from ctx alone; add tools to clear 0.6.
        let s = Signals {
            prompt_tokens: 850,
            local_ctx_window: 1000,
            n_tools: 12,
            n_messages: 1,
            has_cloud_creds: true,
            local_capability_b: 0.0,
        };
        assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn max_quality_routes_nontrivial_to_cloud() {
        // MaxQuality threshold 0.2; a modest request clears it.
        let p = Profile::MaxQuality.policy();
        let s = Signals {
            prompt_tokens: 400,
            local_ctx_window: 1000,
            n_tools: 0,
            n_messages: 1,
            has_cloud_creds: true,
            local_capability_b: 0.0,
        };
        // ctx_fill 0.4 → 0.6*0.4 = 0.24 > 0.2 → cloud.
        assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn max_quality_keeps_trivial_local_no_cascade() {
        // Trivial request scores under 0.2; MaxQuality has cascade=false → plain Local.
        let p = Profile::MaxQuality.policy();
        assert_eq!(decide(&sig(50, true, 1000), &p), Decision::Local);
    }

    #[test]
    fn difficulty_score_is_low_for_trivial_and_high_for_full() {
        let trivial = Signals {
            prompt_tokens: 10, local_ctx_window: 1000, n_tools: 0,
            n_messages: 1, has_cloud_creds: true, local_capability_b: 0.0,
        };
        let full = Signals {
            prompt_tokens: 1000, local_ctx_window: 1000, n_tools: 12,
            n_messages: 20, has_cloud_creds: true, local_capability_b: 0.0,
        };
        assert!(difficulty_score(&trivial) < 0.1);
        assert!(difficulty_score(&full) > 0.95);
        // monotonic in tool count
        let more_tools = Signals { n_tools: 6, ..trivial };
        assert!(difficulty_score(&more_tools) > difficulty_score(&trivial));
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

    // A request scoring ~0.55: ctx 700/1000=0.7→0.42, 6 tools→0.125, 1 msg→~0.0075.
    fn mid_sig(cap: f32) -> Signals {
        Signals {
            prompt_tokens: 700, local_ctx_window: 1000, n_tools: 6,
            n_messages: 1, has_cloud_creds: true, local_capability_b: cap,
        }
    }

    #[test]
    fn weak_local_lowers_threshold_to_cloud() {
        let p = Profile::Balanced.policy(); // threshold 0.6
        // neutral (0.0) and 7B: score ~0.55 < 0.6 → stays local (cascade).
        assert_eq!(decide(&mid_sig(0.0), &p), Decision::LocalThenCascade);
        assert_eq!(decide(&mid_sig(7.0), &p), Decision::LocalThenCascade);
        // 3B: adj −0.12 → effective 0.48 → 0.55 > 0.48 → cloud.
        assert_eq!(decide(&mid_sig(3.0), &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn strong_local_raises_threshold_to_local() {
        let p = Profile::Balanced.policy();
        // 14B: adj +0.21 → effective 0.81 → 0.55 < 0.81 → stays local.
        assert_eq!(decide(&mid_sig(14.0), &p), Decision::LocalThenCascade);
    }

    #[test]
    fn capability_never_overrides_context_gate() {
        // Over-window even with a huge model → still ContextOverflow.
        let s = Signals { prompt_tokens: 5000, local_ctx_window: 1000, n_tools: 0,
            n_messages: 1, has_cloud_creds: true, local_capability_b: 32.0 };
        assert_eq!(decide(&s, &Profile::SaveTokens.policy()), Decision::Cloud(RouteReason::ContextOverflow));
    }

    #[test]
    fn weak_result_is_only_length_truncation() {
        use crate::api::common::{ChatResult, ContentPart, FinishReason};
        let mk = |fr| ChatResult {
            content: vec![ContentPart::Text("x".into())],
            finish_reason: fr,
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        assert!(is_weak_result(&mk(FinishReason::Length)));
        assert!(!is_weak_result(&mk(FinishReason::Stop)));
        assert!(!is_weak_result(&mk(FinishReason::ToolCalls)));
    }
}
