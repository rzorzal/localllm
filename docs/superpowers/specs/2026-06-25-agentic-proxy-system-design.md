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

## Distribution constraint (drives every decision): plug-and-play

The end product ships as a **single Rust executable** for **non-technical users**.
They cannot install dependencies, toolchains, or CLIs. Therefore:

- **One self-contained binary.** No external `llama-server`, no Homebrew, no
  Python/`hf` CLI, no Xcode at run time.
- **Metal shaders compiled at build time** and baked into the binary — only the
  *developer* needs the Metal toolchain; users get a ready binary.
- **Model auto-downloads in-process** on first run over HTTPS (no `hf` CLI).
- **Zero-config start:** double-click / run, it serves on localhost.

This rules out proxying to an external `llama-server` process. The engine must be
**embedded in-process**.

## Architecture

`localllm` is a single Rust binary that embeds the inference engine and exposes
the dual API:

```
Claude Code / Codex / curl
        │  Anthropic or OpenAI API
        ▼
  localllm  (one Rust binary)
        │   • dual-API translation, per-request logging, buffered streaming  [done]
        │   • NEW: prefix freezing, Tscg, tool-gating, prewarm
        │   • embedded engine ↓
        ▼
  llama.cpp via the `llama-cpp-2` crate  (in-process)
        │   • inference + Metal (shaders baked at build)
        │   • KV state save/load to disk  (llama_state_seq_save_file)
        │   • MoE expert offload
        ▼
  GGUF model (dense 3B now → MoE later), auto-downloaded on first run
```

Why embed llama.cpp (`llama-cpp-2`) instead of mistralrs or an external server:
- Single binary (plug-and-play) — unlike an external `llama-server`.
- Exposes **KV state save/load to disk** (the persistence mistralrs lacks) — the
  core mechanism for the prefill cure.
- MoE expert offload and Metal, compiled into the binary.

The current mistralrs engine is kept behind a feature flag as a fallback until the
llama-cpp backend is proven, then retired.

## Phased milestones (each independently shippable)

> Reordered from first draft: the llama-server backend is the **foundation** (it
> provides KV reuse/prewarm — the timeout cure — and the prompt-serialization
> control that Tscg needs), so it goes first.

### Phase 1 — Embedded llama.cpp engine + static-prefix prewarm  (FOUNDATION)
Add a `llama-cpp-2`-backed engine in-process, behind the existing `Generator`
trait, so the dual-API/logging/streaming layer is unchanged. Auto-download the
GGUF on first run. On startup the engine **prewarms** the static prefix and uses
llama.cpp's KV state save/load + prefix reuse so each turn only prefills the small
delta.
- Acceptance: single binary runs with no external deps; model auto-downloads;
  after prewarm a Claude-Code-shaped request returns first token in ≤ a few
  seconds (not minutes); measured TTFT before/after prewarm reported.

### Phase 2 — Tool-schema compression (Tscg)
Deterministic, lossless compression of the incoming JSON tool schemas before they
reach llama-server (now that the proxy controls serialization). Target ~45–50%
token reduction on the tools block.
- Acceptance: a request with N tools has its serialized tool tokens cut ~half;
  tool-calling acceptance scripts still pass (compression is lossless).

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
