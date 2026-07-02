# Smart History Filter — Research & Design (decision deferred)

Date: 2026-07-02
Status: **research captured; engine not yet chosen; not implemented.**

## Goal

Optionally select *which* past conversation turns to keep (instead of plain
recency truncation) when trimming history to the configured `history_turns`
budget. Exposed as a **global toggle on the Config page** ("Filtro inteligente
de histórico", default **OFF**). Must run **fast and locally**.

## Hard constraints (from the user)

- Global toggle in the Config screen, default OFF (OFF = current recency
  truncation, unchanged behavior).
- Must be **ordered/ranked**: if the user configures `history = 10` and the
  filter selects 11 turns, drop the lowest-scoring 1 → output is exactly N,
  already ranked then re-sorted chronologically.
- **Fast + local.** Prefer classic NLP/statistics; embeddings acceptable only
  if they run fast and locally (no cloud, no Python).

## Current behavior (baseline, OFF)

`api::common::truncate_history(messages, keep_turns)` keeps leading system
messages + the last N turns (a turn starts at a `User` message). Pure recency.
Plugged in via `server::apply_history_window` → `resolve_history_turns`.

## Research findings

Context selection is a well-studied problem: combine **relevance** (to the
current/last turn) + **recency** + **diversity**, then take top-N. Irrelevant
context measurably hurts accuracy (one study: +10% irrelevant content → −23%
accuracy), so trimming smartly (not just by recency) is worthwhile.

Signals / methods:

- **BM25 / TF-IDF** — lexical relevance by term overlap. Zero dependencies,
  instant, deterministic. Strong for coding agents (repeated identifiers, file
  paths, symbols). Weak on paraphrase/synonyms.
- **MMR (Maximal Marginal Relevance)** — re-ranks by a weighted mix of
  relevance and novelty, avoiding keeping N near-duplicate turns. Works on top
  of any relevance score (BM25 or embeddings).
- **Static embeddings (model2vec / potion)** — `model2vec-rs` is a pure-Rust
  crate (no Python); potion-base-8M is ~32 MB, 384-dim, ~54k sentences/s on one
  CPU core. Semantic match (paraphrase/synonyms), still fast and local. Cost: a
  new dependency + one-time ~32 MB model download.

Sources:
- Context engineering / relevance+recency+salience: https://mem0.ai/blog/context-engineering-ai-agents-guide
- Hybrid retrieval (BM25 + rerank): https://medium.com/@richardhightower/stop-the-hallucinations-hybrid-retrieval-with-bm25-pgvector-embedding-rerank-llm-rubric-rerank-895d8f7c7242
- MMR: https://inferensys.com/glossary/agentic-memory-and-context-management/semantic-indexing-and-chunking/maximal-marginal-relevance-mmr
- model2vec-rs (Rust static embeddings): https://github.com/MinishLab/model2vec-rs
- potion-base-8M model: https://huggingface.co/minishlab/potion-base-8M
- Response selection in retrieval dialogue (academic): https://arxiv.org/pdf/2509.22845

## Proposed design (engine-agnostic parts)

- **Unit = turns** (matches `history_turns`).
- **Pinned (always kept):** leading system messages, the latest turn, and any
  tool_call/tool_result pair kept together (never split a call from its result).
- **Score** each droppable turn: `score = α·relevance(turn, last_turn) +
  β·recency_decay(turn)`. Then apply **MMR** for diversity among the kept set.
- **Trim to N:** rank all candidate turns, keep the top `(N − pinned)`, then
  **re-sort the kept turns back into chronological order** before sending.
- Deterministic given the same input (important for prefix-cache reuse — a
  smarter but unstable selection would thrash the KV cache; weigh recency high
  enough that stable prefixes are preferred).
- Lives beside `truncate_history` as `select_history_smart(...)`, chosen in
  `apply_history_window` when the global toggle is ON.

### Prefix-cache caveat (important)

localllm relies on prefix KV-cache reuse. Reordering/dropping *middle* turns
changes the prompt prefix and can **invalidate the cache**, costing more than it
saves. Mitigations to evaluate: only re-select when the budget is actually
exceeded; keep a long stable recent tail and only smart-select the older
overflow; measure cache hit-rate impact before shipping ON by default.

## Engine alternatives (DECISION DEFERRED)

| Option | Pros | Cons |
|---|---|---|
| **A. BM25/TF-IDF + MMR** (recommended start) | Pure Rust, zero deps, ~0 ms, deterministic; great for code | Lexical only (no paraphrase) |
| **B. model2vec embeddings + MMR** | Semantic match; still fast+local | +dep `model2vec-rs`, ~32 MB model download |
| **C. Hybrid (BM25 prefilter → embedding rerank)** | Best quality | Most code + the model2vec dep/model |

Recommendation to revisit: ship **A** first (matches "prefer NLP/statistics";
zero risk, instant), keep **B** as an optional rerank stage behind the same
toggle if A proves insufficient.

## Open questions for later

- Chosen engine (A/B/C).
- α/β weights and recency decay shape; MMR λ.
- Whether ON-by-default is ever safe given the prefix-cache caveat.
- Turn-level vs message-level granularity for tool-heavy transcripts.
