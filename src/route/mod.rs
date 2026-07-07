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
    /// Estimated tokens in just the latest turn (the last message) — the
    /// incremental ask, used as the difficulty signal instead of the whole
    /// (mostly static/cached) prompt.
    pub last_turn_tokens: usize,
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

/// Tokens in the latest turn that count as a "full-difficulty" ask.
const TURN_BUDGET_TOKENS: f64 = 2000.0;

/// Cheap pre-generation difficulty score in `[0.0, 1.0]`. Higher means the
/// request is harder/bigger and more likely to need the cloud model.
///
/// Scored from the LATEST turn's size + conversation depth (weights sum to 1.0):
/// - latest turn: the new ask, tokens / `TURN_BUDGET_TOKENS`, saturating at 1.0
/// - depth: message count, saturating at 20 (long multi-turn is harder)
///
/// It deliberately ignores the total prompt size and the available-tool count.
/// Those measure the *client's fixed overhead*, not task difficulty: Claude
/// Code sends a ~26k static prefix and dozens of tools on every request, which
/// made a trivial "ls" score as high as a big refactor and always route to
/// cloud. Near-overflow is still handled separately by the hard context gate in
/// `decide`; a local answer that comes back weak still escalates via cascade.
pub fn difficulty_score(s: &Signals) -> f64 {
    let turn = (s.last_turn_tokens as f64 / TURN_BUDGET_TOKENS).min(1.0);
    let depth = (s.n_messages as f64 / 20.0).min(1.0);
    0.8 * turn + 0.2 * depth
}

/// Capability adjustment to the escalation threshold: a stronger-than-7B local
/// model raises the cutoff (keeps more local), a weaker one lowers it (more
/// cloud). Unknown capability (`0.0`) is neutral. Clamped to ±0.3.
pub fn capability_adjustment(cap_b: f32) -> f64 {
    if cap_b > 0.0 {
        ((cap_b - CAPABILITY_PIVOT_B) as f64 * CAPABILITY_SLOPE).clamp(-0.3, 0.3)
    } else {
        0.0
    }
}

/// Neutral model size (billions) at which the capability adjustment is zero.
pub const CAPABILITY_PIVOT_B: f32 = 7.0;
/// Threshold shift per 1 B of capability above/below the pivot. Exported so
/// consumers that invert this mapping (e.g. the dashboard's effective-capability
/// estimate) reference the single source of truth instead of duplicating 0.03.
pub const CAPABILITY_SLOPE: f64 = 0.03;

/// The profile's escalation threshold after the capability adjustment, clamped
/// to `[0, 1]`. A request whose difficulty score exceeds this goes to cloud.
pub fn effective_threshold(p: &RoutingPolicy, cap_b: f32) -> f64 {
    (p.escalation_threshold + capability_adjustment(cap_b)).clamp(0.0, 1.0)
}

/// Decide where a request runs from cheap signals and the active policy.
///
/// Order: (1) hard context gate, (2) cloud-impossible shortcut, (3) difficulty
/// score vs the profile threshold, (4) in-window low-score → cascade or local.
pub fn decide(s: &Signals, p: &RoutingPolicy) -> Decision {
    // 1. Hard context gate: prompt too big for the local window.
    let over_window = s.prompt_tokens as f64 > s.local_ctx_window as f64 * p.ctx_gate_frac;
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
    if difficulty_score(s) > effective_threshold(p, s.local_capability_b) {
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
        chars += message_chars(m);
    }
    for t in &req.tools {
        chars += t.name.len() + t.description.len() + t.parameters.to_string().len();
    }
    chars / 4
}

/// Estimate the token count of a bare text string with the same cheap `chars/4`
/// heuristic used for prompts. Used to count streamed completion deltas.
pub fn estimate_text_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Character count of a single message's content (text + tool-call args +
/// tool-result content).
fn message_chars(m: &crate::api::common::ChatMessage) -> usize {
    let mut chars = 0usize;
    if let Some(t) = &m.text {
        chars += t.len();
    }
    for c in &m.tool_calls {
        chars += c.name.len() + c.arguments.len();
    }
    if let Some(tr) = &m.tool_result {
        chars += tr.content.len();
    }
    chars
}

/// Estimate the tokens in just the latest turn (the last message). This is the
/// difficulty signal — the incremental ask — NOT the whole prompt. Claude Code
/// sends a ~26k static prefix + dozens of tools on every request; scoring the
/// full prompt made a trivial "ls" look as hard as a large refactor. The last
/// turn reflects the actual new work.
pub fn estimate_last_turn_tokens(req: &ChatRequest) -> usize {
    req.messages
        .last()
        .map(|m| message_chars(m) / 4)
        .unwrap_or(0)
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
            // The whole prompt is one turn in these small fixtures.
            last_turn_tokens: prompt_tokens,
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
        // Balanced threshold 0.45; a large NEW turn pushes the score over it.
        let p = Profile::Balanced.policy();
        // last_turn 1600 → turn 0.8 → 0.8*0.8 = 0.64; +depth ~0.01 → 0.65 > 0.45.
        // The prompt still fits the window (no overflow gate).
        let s = Signals {
            prompt_tokens: 1600,
            local_ctx_window: 32768,
            n_tools: 12,
            n_messages: 1,
            last_turn_tokens: 1600,
            has_cloud_creds: true,
            local_capability_b: 0.0,
        };
        assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn max_quality_routes_nontrivial_to_cloud() {
        // MaxQuality threshold 0.2; a modest NEW turn clears it.
        let p = Profile::MaxQuality.policy();
        let s = Signals {
            prompt_tokens: 600,
            local_ctx_window: 32768,
            n_tools: 0,
            n_messages: 1,
            last_turn_tokens: 600,
            has_cloud_creds: true,
            local_capability_b: 0.0,
        };
        // turn 600/2000 = 0.3 → 0.8*0.3 = 0.24 > 0.2 → cloud.
        assert_eq!(decide(&s, &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn agentic_overhead_stays_local_when_new_turn_is_small() {
        // The regression this heuristic fixes: a Claude-Code-shaped request — a
        // huge static prefix (~26k tokens) filling most of the window and dozens
        // of available tools — but a trivial new ask ("ls"). Must stay LOCAL on
        // Balanced (the old ctx_fill+tool_count formula always sent this cloud).
        let p = Profile::Balanced.policy();
        let s = Signals {
            prompt_tokens: 26000,
            local_ctx_window: 32768, // fill 0.79 (under the 0.9 gate)
            n_tools: 35,
            n_messages: 7,
            last_turn_tokens: 4, // "ls"
            has_cloud_creds: true,
            local_capability_b: 8.0,
        };
        // score = 0.8*(4/2000) + 0.2*(7/20) = 0.0016 + 0.07 ≈ 0.07 << threshold.
        assert_eq!(decide(&s, &p), Decision::LocalThenCascade);
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
            prompt_tokens: 10,
            local_ctx_window: 1000,
            n_tools: 0,
            n_messages: 1,
            last_turn_tokens: 10,
            has_cloud_creds: true,
            local_capability_b: 0.0,
        };
        let full = Signals {
            prompt_tokens: 2000,
            local_ctx_window: 32768,
            n_tools: 12,
            n_messages: 20,
            last_turn_tokens: 2000,
            has_cloud_creds: true,
            local_capability_b: 0.0,
        };
        assert!(difficulty_score(&trivial) < 0.1);
        assert!(difficulty_score(&full) > 0.95);
        // monotonic in the size of the latest turn
        let bigger_turn = Signals {
            last_turn_tokens: 800,
            ..trivial
        };
        assert!(difficulty_score(&bigger_turn) > difficulty_score(&trivial));
        // NOT sensitive to available-tool count (that's client overhead)
        let more_tools = Signals {
            n_tools: 30,
            ..trivial
        };
        assert_eq!(difficulty_score(&more_tools), difficulty_score(&trivial));
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
        assert_eq!(
            decide(&sig(980, false, 1000), &pol()),
            Decision::LocalNoCreds
        );
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

    #[test]
    fn last_turn_tokens_counts_only_the_final_message() {
        let msg = |s: &str| ChatMessage {
            role: Role::User,
            text: Some(s.to_string()),
            tool_calls: vec![],
            tool_result: None,
        };
        let req = ChatRequest {
            messages: vec![msg(&"a".repeat(40000)), msg("run ls")], // huge prefix, tiny last turn
            tools: vec![],
            max_tokens: None,
            temperature: None,
            stream: false,
            model: "m".into(),
        };
        // Only the last message ("run ls", 6 chars) counts → 1 token, not the 40k prefix.
        assert_eq!(estimate_last_turn_tokens(&req), 6 / 4);
        assert!(estimate_prompt_tokens(&req) > 9000); // prompt is huge, but the turn is tiny
    }

    // A request scoring ~0.41: last_turn 1000/2000=0.5→0.8*0.5=0.40, 1 msg→~0.01.
    fn mid_sig(cap: f32) -> Signals {
        Signals {
            prompt_tokens: 1000,
            local_ctx_window: 32768,
            n_tools: 6,
            n_messages: 1,
            last_turn_tokens: 1000,
            has_cloud_creds: true,
            local_capability_b: cap,
        }
    }

    #[test]
    fn weak_local_lowers_threshold_to_cloud() {
        let p = Profile::Balanced.policy(); // threshold 0.45
                                            // neutral (0.0) and 7B: score ~0.41 < 0.45 → stays local (cascade).
        assert_eq!(decide(&mid_sig(0.0), &p), Decision::LocalThenCascade);
        assert_eq!(decide(&mid_sig(7.0), &p), Decision::LocalThenCascade);
        // 3B: adj −0.12 → effective 0.33 → 0.41 > 0.33 → cloud.
        assert_eq!(
            decide(&mid_sig(3.0), &p),
            Decision::Cloud(RouteReason::Difficulty)
        );
    }

    // A request scoring ~0.65: last_turn 1600/2000=0.8→0.8*0.8=0.64, 1 msg→~0.01.
    // This sits ABOVE the 7B/neutral threshold (0.45) but BELOW the 14B one (0.66),
    // so a strong model is what flips it Cloud→Local.
    fn high_sig(cap: f32) -> Signals {
        Signals {
            prompt_tokens: 1600,
            local_ctx_window: 32768,
            n_tools: 9,
            n_messages: 1,
            last_turn_tokens: 1600,
            has_cloud_creds: true,
            local_capability_b: cap,
        }
    }

    #[test]
    fn strong_local_raises_threshold_to_local() {
        let p = Profile::Balanced.policy(); // threshold 0.45
                                            // Score ~0.65: at 7B (effective 0.45) → Cloud; at 14B (effective 0.66) → local.
                                            // This proves the strong model RAISES the cutoff enough to flip the decision.
        assert_eq!(
            decide(&high_sig(7.0), &p),
            Decision::Cloud(RouteReason::Difficulty)
        );
        assert_eq!(decide(&high_sig(14.0), &p), Decision::LocalThenCascade);
    }

    #[test]
    fn capability_never_overrides_context_gate() {
        // Over-window even with a huge model → still ContextOverflow.
        let s = Signals {
            prompt_tokens: 5000,
            local_ctx_window: 1000,
            n_tools: 0,
            n_messages: 1,
            last_turn_tokens: 5000,
            has_cloud_creds: true,
            local_capability_b: 32.0,
        };
        assert_eq!(
            decide(&s, &Profile::SaveTokens.policy()),
            Decision::Cloud(RouteReason::ContextOverflow)
        );
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
