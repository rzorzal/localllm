# Fase D4 — Semantic Routing (DEFERRED / NOT BUILT)

**Date:** 2026-07-07
**Status:** DEFERRED by user decision — not implemented. Documented for future reference.

## Decision

D4 was the last sub-project of the Fase D roadmap (original feature #2). After
shipping D1 (feedback loop) and D2 (per-model effective capability) and
deferring D3 (multi-local tiering), the user chose to **abort D4 before design**
("acho que isso pode causar um problema grande na real") and move to Fase E
(cross-platform tray) instead. This document records what D4 would have been and
the specific risks that stopped it, so it can be revisited without re-deriving
the analysis.

If revisited, the user wanted it built behind a **config feature flag (a toggle
in Config), off by default** — semantic routing must be strictly opt-in.

## What D4 was (original feature #2)

**Semantic routing:** replace or augment the current cheap `difficulty_score`
heuristic with a signal derived from an embedding of the prompt, so routing
reflects *what the request is about* (semantic difficulty) rather than only its
size and conversation depth.

Today's router (`src/route/mod.rs`) scores difficulty as
`0.8 * turn_size + 0.2 * depth` — purely lexical/structural. It cannot tell a
trivial 1500-token paste from a hard 1500-token algorithm question. A semantic
signal would embed the latest turn and estimate difficulty from meaning.

### Sketch of the design space (not decided)

- **Embedding backend** (the crux, unresolved — this is what stopped it):
  1. *Dedicated GGUF embedding model* (bge-small / nomic, ~100–400 MB) loaded in
     llama-cpp embedding mode. Fast and independent, but adds a second resident
     model (RAM) plus a download/lifecycle to manage.
  2. *Reuse the loaded chat model* via last-hidden-layer pooling. Zero extra RAM
     and no download, but embedding quality varies wildly by model and it likely
     needs a separate forward pass (latency on every request).
  3. *Cloud embedding API.* Zero local RAM, but per-request cost and network
     latency — and calling cloud to decide whether to call cloud contradicts the
     local-first premise.
- **Knowledge source:** a semantic score needs a notion of "hard vs easy."
  Either hand-curated anchor prompts shipped with the app (embed prompt, compare
  similarity to hard vs easy anchors) or labels mined from the D1 feedback log
  (cascade/truncated/reask → hard; cloud_trivial → easy). The latter is
  appealing (D1 already logs these) but couples routing to log volume/quality.
- **Combination:** blend the semantic score into `difficulty_score` (additive
  term or weighted mix) vs. let it override. Blending behind the flag is the
  safe form — flag off reproduces today's behavior exactly.

## Why deferred (the "big real problem")

- **Latency on the hot path:** every request would pay an embedding forward pass
  *before* the routing decision — the exact opposite of the current design,
  where routing is a pure, I/O-free function (`decide` does no inference). A slow
  or blocking embed step degrades every request, including the trivial ones the
  router is supposed to keep cheap and local.
- **RAM / model management:** a dedicated embedding model competes with the chat
  model for memory on a machine already tight enough that model residency is a
  first-class concern (see AirLLM per-model exec profile work).
- **Correctness risk vs. proven cheaper wins:** D1/D2 already improved routing
  quality with measurement and per-model capability — cheaply, with no hot-path
  cost. Semantic routing is a large systems change for an uncertain marginal
  gain, and a wrong semantic score silently misroutes.
- **YAGNI for now:** the feedback data from D1 should show whether semantic
  misrouting is even a real bottleneck before paying this cost.

## If revisited

Build it as an **opt-in config feature flag, default off** (explicit user
requirement). Answer these first, in order:
1. **Embedding backend** — pick the serving model (dedicated GGUF vs reuse chat
   model vs cloud) BEFORE any routing design. This is what blocked D4; it must be
   settled first, with a measured hot-path latency budget.
2. **Where the score lives on the hot path** — precomputed/cached vs synchronous
   per request. Never block a trivial local request on an embed pass.
3. **Knowledge source** — shipped anchor set vs D1-feedback-mined labels.
4. **Combination** — additive blend into `difficulty_score` behind the flag, so
   flag-off is byte-for-byte today's behavior.

Related: [[../../../.claude/projects/-Users-ricardo-Repos-localllm/memory/fases-roadmap-status]]
(roadmap status memory), and the D3 deferral doc
(`2026-07-07-fase-d3-multi-local-tiering-DEFERRED.md`). Fase E (cross-platform
tray) follows this deferral.
