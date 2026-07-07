//! Smart history selection: choose WHICH past turns to keep (by relevance to
//! the latest turn) instead of plain recency, trimming to the `keep_turns`
//! budget. Pure and deterministic — no I/O, no model, no dependencies.
//!
//! Engine: BM25 lexical relevance + recency, then MMR for diversity (Option A
//! of docs/superpowers/specs/2026-07-02-smart-history-filter-research.md).
//!
//! Structure preserved: leading system/preamble always kept; the latest turn is
//! always kept; selected turns are re-sorted into chronological order so the
//! transcript still reads forward. OFF path stays `truncate_history`.

use crate::api::common::{ChatMessage, Role};

// BM25 params (standard defaults).
const K1: f64 = 1.5;
const B: f64 = 0.75;
// Relevance blend and MMR diversity.
const W_BM25: f64 = 0.7;
const W_RECENCY: f64 = 0.3;
const MMR_LAMBDA: f64 = 0.7;

/// Lowercase alphanumeric tokenization (splits on any non-alphanumeric char).
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// All searchable text of a message: its text plus tool-call names and any
/// tool-result content, so tool-heavy turns still score on their payload.
fn message_text(m: &ChatMessage) -> String {
    let mut s = m.text.clone().unwrap_or_default();
    for tc in &m.tool_calls {
        s.push(' ');
        s.push_str(&tc.name);
    }
    if let Some(tr) = &m.tool_result {
        s.push(' ');
        s.push_str(&tr.content);
    }
    s
}

/// Term-frequency map for a token list.
fn term_freq(tokens: &[String]) -> std::collections::HashMap<String, f64> {
    let mut tf = std::collections::HashMap::new();
    for t in tokens {
        *tf.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    tf
}

/// Cosine similarity between two term-frequency vectors.
fn cosine(a: &std::collections::HashMap<String, f64>, b: &std::collections::HashMap<String, f64>) -> f64 {
    let mut dot = 0.0;
    for (k, va) in a {
        if let Some(vb) = b.get(k) {
            dot += va * vb;
        }
    }
    let na: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// BM25 score of `doc_tf` (length `doc_len`) against `query_tokens`, given the
/// corpus document frequencies `df`, corpus size `n_docs`, and `avgdl`.
fn bm25(
    query_tokens: &[String],
    doc_tf: &std::collections::HashMap<String, f64>,
    doc_len: f64,
    df: &std::collections::HashMap<String, f64>,
    n_docs: f64,
    avgdl: f64,
) -> f64 {
    let mut score = 0.0;
    let mut seen = std::collections::HashSet::new();
    for q in query_tokens {
        if !seen.insert(q) {
            continue; // each query term contributes once
        }
        let tf = match doc_tf.get(q) {
            Some(v) => *v,
            None => continue,
        };
        let n_q = df.get(q).copied().unwrap_or(0.0);
        // idf with +1 so it stays positive even for common terms.
        let idf = ((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();
        let denom = tf + K1 * (1.0 - B + B * doc_len / avgdl.max(1.0));
        score += idf * (tf * (K1 + 1.0)) / denom.max(1e-9);
    }
    score
}

/// A candidate turn: original index (chronological) + its message span.
struct Turn {
    idx: usize,
    tokens: Vec<String>,
    tf: std::collections::HashMap<String, f64>,
}

/// Select which turns to keep, trimming to `keep_turns` total turns by
/// BM25 relevance to the latest turn + recency + MMR diversity. Leading
/// system/preamble and the latest turn are always kept. Deterministic.
///
/// Mirrors `truncate_history` boundaries: a turn starts at a `User` message;
/// when there is nothing to drop it returns the input unchanged.
pub fn select_history_smart(messages: Vec<ChatMessage>, keep_turns: u32) -> Vec<ChatMessage> {
    let n = keep_turns as usize;

    // Leading system messages are always kept.
    let lead_sys = messages.iter().take_while(|m| m.role == Role::System).count();
    if n == 0 {
        return messages.into_iter().take(lead_sys).collect();
    }

    let user_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect();

    // Nothing to drop → identical to recency truncation's no-op.
    if user_starts.len() <= n {
        return messages;
    }

    // Head = everything before the first user turn (system + any preamble). Kept.
    let head_end = user_starts[0];

    // Turn boundaries: turn t spans [user_starts[t], user_starts[t+1]).
    let turn_span = |t: usize| -> (usize, usize) {
        let start = user_starts[t];
        let end = user_starts.get(t + 1).copied().unwrap_or(messages.len());
        (start, end)
    };
    let last_t = user_starts.len() - 1;

    // The latest turn is the query and is always pinned.
    let (lq_s, lq_e) = turn_span(last_t);
    let query_tokens: Vec<String> = messages[lq_s..lq_e]
        .iter()
        .flat_map(|m| tokenize(&message_text(m)))
        .collect();

    // Build candidate turns (all but the last).
    let mut candidates: Vec<Turn> = Vec::with_capacity(last_t);
    for t in 0..last_t {
        let (s, e) = turn_span(t);
        let tokens: Vec<String> = messages[s..e]
            .iter()
            .flat_map(|m| tokenize(&message_text(m)))
            .collect();
        let tf = term_freq(&tokens);
        candidates.push(Turn { idx: t, tokens, tf });
    }

    // We keep the last turn plus (n-1) candidates.
    let want = n.saturating_sub(1);

    // Corpus stats over candidates for BM25.
    let n_docs = candidates.len() as f64;
    let avgdl = if candidates.is_empty() {
        1.0
    } else {
        candidates.iter().map(|c| c.tokens.len() as f64).sum::<f64>() / n_docs
    };
    let mut df: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for c in &candidates {
        for term in c.tf.keys() {
            *df.entry(term.clone()).or_insert(0.0) += 1.0;
        }
    }

    // Relevance = blend of normalized BM25 and recency (newer = higher).
    let raw_bm25: Vec<f64> = candidates
        .iter()
        .map(|c| bm25(&query_tokens, &c.tf, c.tokens.len() as f64, &df, n_docs, avgdl))
        .collect();
    let max_bm25 = raw_bm25.iter().cloned().fold(0.0_f64, f64::max);
    let rel: Vec<f64> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let bm = if max_bm25 > 0.0 { raw_bm25[i] / max_bm25 } else { 0.0 };
            let recency = if last_t <= 1 { 1.0 } else { c.idx as f64 / (last_t as f64 - 1.0) };
            W_BM25 * bm + W_RECENCY * recency
        })
        .collect();

    // MMR: greedily pick `want` candidates balancing relevance and novelty.
    let mut selected: Vec<usize> = Vec::new(); // indices into `candidates`
    let mut remaining: Vec<usize> = (0..candidates.len()).collect();
    while selected.len() < want && !remaining.is_empty() {
        let mut best = remaining[0];
        let mut best_score = f64::NEG_INFINITY;
        for &ci in &remaining {
            let novelty = selected
                .iter()
                .map(|&sj| cosine(&candidates[ci].tf, &candidates[sj].tf))
                .fold(0.0_f64, f64::max);
            let mmr = MMR_LAMBDA * rel[ci] - (1.0 - MMR_LAMBDA) * novelty;
            // Deterministic tie-break: prefer more recent (higher idx).
            if mmr > best_score
                || (mmr == best_score && candidates[ci].idx > candidates[best].idx)
            {
                best_score = mmr;
                best = ci;
            }
        }
        selected.push(best);
        remaining.retain(|&x| x != best);
    }

    // Re-sort selected turns chronologically, then flatten: head + selected + last.
    let mut keep_turn_idx: Vec<usize> = selected.iter().map(|&ci| candidates[ci].idx).collect();
    keep_turn_idx.sort_unstable();

    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    out.extend(messages[0..head_end].iter().cloned());
    for &t in &keep_turn_idx {
        let (s, e) = turn_span(t);
        out.extend(messages[s..e].iter().cloned());
    }
    out.extend(messages[lq_s..lq_e].iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::common::{ChatMessage, Role};

    fn msg(role: Role, text: &str) -> ChatMessage {
        ChatMessage { role, text: Some(text.into()), tool_calls: vec![], tool_result: None }
    }

    // Build a conversation: system + a series of (user, assistant) turns.
    fn convo(sys: &str, turns: &[(&str, &str)]) -> Vec<ChatMessage> {
        let mut v = vec![msg(Role::System, sys)];
        for (u, a) in turns {
            v.push(msg(Role::User, u));
            v.push(msg(Role::Assistant, a));
        }
        v
    }

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(tokenize("Read file.rs, now!"), vec!["read", "file", "rs", "now"]);
    }

    #[test]
    fn no_trim_when_within_budget() {
        let c = convo("sys", &[("a", "1"), ("b", "2")]);
        let out = select_history_smart(c.clone(), 3);
        assert_eq!(out.len(), c.len()); // unchanged
    }

    #[test]
    fn keeps_system_and_latest_turn_and_trims_to_n() {
        // 4 turns, keep 2 → system + 1 selected + last turn = 1 sys + 2 turns.
        let c = convo("sys", &[
            ("alpha widget", "x"),
            ("beta gadget", "y"),
            ("gamma sprocket", "z"),
            ("about the alpha widget again", "w"),
        ]);
        let out = select_history_smart(c, 2);
        // 1 system + 2 turns * 2 msgs = 5
        assert_eq!(out.iter().filter(|m| m.role == Role::System).count(), 1);
        let users: Vec<&str> = out.iter().filter(|m| m.role == Role::User)
            .map(|m| m.text.as_deref().unwrap()).collect();
        assert_eq!(users.len(), 2);
        // last turn always kept
        assert_eq!(users[1], "about the alpha widget again");
        // the relevant older turn ("alpha widget") beats unrelated recent ones
        assert_eq!(users[0], "alpha widget");
    }

    #[test]
    fn keeps_chronological_order() {
        let c = convo("s", &[
            ("shared token apple", "a"),
            ("shared token apple banana", "b"),
            ("unrelated zzz", "c"),
            ("apple please", "d"),
        ]);
        let out = select_history_smart(c, 3); // sys + 2 selected + last
        let idxs: Vec<&str> = out.iter().filter(|m| m.role == Role::User)
            .map(|m| m.text.as_deref().unwrap()).collect();
        // selected earlier turns must appear before the last, in original order
        assert_eq!(idxs.last().unwrap(), &"apple please");
        // first two are in chronological order (apple, apple banana)
        assert_eq!(idxs[0], "shared token apple");
    }

    #[test]
    fn deterministic() {
        let c = convo("s", &[("a b c", "1"), ("d e f", "2"), ("g h i", "3"), ("a d g", "4")]);
        let a = select_history_smart(c.clone(), 2);
        let b = select_history_smart(c, 2);
        let ta: Vec<_> = a.iter().map(|m| m.text.clone()).collect();
        let tb: Vec<_> = b.iter().map(|m| m.text.clone()).collect();
        assert_eq!(ta, tb);
    }

    #[test]
    fn n_one_keeps_only_last_turn() {
        let c = convo("s", &[("a", "1"), ("b", "2"), ("c", "3")]);
        let out = select_history_smart(c, 1);
        let users: Vec<&str> = out.iter().filter(|m| m.role == Role::User)
            .map(|m| m.text.as_deref().unwrap()).collect();
        assert_eq!(users, vec!["c"]);
    }
}
