# localllm → agentic proxy for constrained local hardware (design)

Date: 2026-06-25
Status: Proposed

## Goal

Make an agentic CLI client (Claude Code) usable against a **local** model on a
16 GB Apple Silicon machine, for small tasks, without the client timing out.

The blocker today: Claude Code sends a ~26k-token static prefix (system prompt +
~27 tool schemas) on **every** turn. Re-prefilling it each turn is slow, and the
machine swaps because model + KV cache + the user's apps exceed 16 GB. First-turn
latency exceeds Claude Code's request timeout → infinite retry.

This project attacks both bottlenecks, in research-backed phases:

- **Axis A — prefill latency:** stop recomputing the static prefix every turn.
- **Axis B — RAM pressure:** stop holding the whole model + KV in 16 GB at once.

## Research basis (one line each)

- Persistent KV cache on Apple Silicon gives 76–136× TTFT (47s→<1s) for static
  prefixes — *Agent Memory Below the Prompt* (arXiv 2603.04428).
- Tool-schema compression (Tscg) cuts tool tokens ~50% losslessly (arXiv 2605.26165).
- Tool retrieval / MCP-Zero injects only relevant tools (arXiv 2506.01056).
- llama.cpp `llama-server` already persists KV (`--slot-save-path`), reuses
  prefixes (`--cache-reuse`), and offloads MoE experts.
- MoE + expert offload (PowerInfer, LLM-in-a-Flash, HOBBIT, FlashMoE) runs
  big-quality models in limited RAM by keeping only hot experts resident.

## Architecture

`localllm` becomes a transparent **proxy** in front of `llama-server`:

```
Claude Code / Codex / curl
        │  Anthropic or OpenAI API
        ▼
  localllm proxy  (our existing Rust crate: dual-API translation,
        │          per-request logging, buffered streaming)
        │   + NEW: prefix freezing, Tscg, tool-gating, prewarm
        ▼  OpenAI API (localhost)
  llama-server  (llama.cpp: inference, KV persistence, MoE expert offload)
        ▼
  GGUF model (dense 3B now → MoE later)
```

Why a proxy (not embed): preserves all current translation/logging/streaming
work; gains llama.cpp's KV-save + MoE offload without reimplementing them; clean
process boundary. The current embedded-mistralrs engine path is retained behind a
flag for the no-llama-server case until the proxy path is proven.

## Phased milestones (each independently shippable)

### Phase 1 — Tool-schema compression (Tscg) in the proxy
Deterministic, lossless compression of the incoming JSON tool schemas before they
reach the model. Target ~45–50% token reduction on the tools block.
- Pure proxy logic; no engine change.
- Acceptance: a request with N tools has its serialized tool tokens cut ~half;
  tool-calling acceptance scripts still pass (compression is lossless).

### Phase 2 — llama-server backend + static-prefix prewarm
Run `llama-server` as the inference engine; the proxy forwards to it. On startup
(or first observed Claude Code prefix), the proxy **prewarms** the static prefix
KV in the background and relies on `--cache-reuse` / `--slot-save-path` so each
turn only prefills the small delta.
- Acceptance: after prewarm, a Claude-Code-shaped request returns first token in
  ≤ a few seconds (not minutes); measured TTFT before/after prewarm reported.

### Phase 3 — MoE model + expert offload
Switch the default model to an MoE (candidate: Qwen3-30B-A3B, ~3B active) loaded
by llama-server with expert offload so cold experts live on SSD and only hot
experts occupy RAM. Goal: larger-model quality within 16 GB without swap.
- Acceptance: model loads and serves within RAM budget (no sustained swap growth
  during a request); quality on a small task is visibly better than dense 3B;
  tokens/sec reported.

### Phase 4 — Lazy tool-gating + persistent Q4 KV
Inject only query-relevant tool schemas per turn (the rest kept "cold"); if the
model attempts an absent tool, fault and re-expand. Persist the prefix KV (Q4) to
disk so it survives restart/sleep.
- Acceptance: a small task injects a small subset of tools and still completes;
  after a server restart, the prefix is warm without re-prefill.

## Key decisions / open questions

- **Model for Phase 3:** Qwen3-30B-A3B is the candidate; if its Q4 footprint +
  offload doesn't fit 16 GB in practice, fall back to a smaller MoE. Decide with a
  real load test at Phase 3, not upfront.
- **Engine boundary:** proxy → `llama-server` over localhost HTTP. The proxy must
  detect and not corrupt llama-server's own prefix caching (send the prefix
  byte-stable).
- **Prewarm trigger:** prewarm on first seen prefix (cache its hash) vs a
  configured prompt. Start with first-seen-prefix capture.

## Non-goals

- Reimplementing PowerInfer/Deja Vu sparsity from scratch (use llama.cpp's
  existing MoE offload instead).
- Multi-device / distributed serving.
- Training or fine-tuning.

## Risks

- Phase 3 MoE may not fit 16 GB even with offload → fall back to smaller MoE or
  keep dense 3B; Phases 1–2 still deliver the prefill win regardless.
- llama-server flag/behaviour differences across versions → pin a version and
  verify flags at Phase 2 (like we locked the mistralrs API in the base project).
- Prewarm racing Claude Code's first connection → prewarm at startup before
  announcing readiness.
