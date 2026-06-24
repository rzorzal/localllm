# localllm — Local LLM server in Rust (design)

Date: 2026-06-24
Status: Approved

## Goal

A single Rust binary that runs a quantized LLM locally on Apple Silicon (M1 Pro,
16 GB unified RAM) and exposes it over HTTP on `127.0.0.1`. The server speaks both
the Anthropic Messages API (so Claude Code can point at it) and the OpenAI Chat
Completions API (so Codex can point at it). The primary acceptance criterion is
correct **tool calling / tool chaining** through both API surfaces.

## Target hardware

- Apple M1 Pro, 10 CPU cores, 16 GB unified memory, macOS 26.5.
- Usable model budget ~10-11 GB (OS/apps take 4-6 GB). Default model sized well
  under that.

## Non-goals

- No multi-user / remote serving. Localhost only.
- No web UI. API + curl test scripts only.
- No fine-tuning, pruning, or distillation at runtime (those are offline,
  pre-deploy steps and out of scope for this binary).

## Inference backend

`mistralrs` crate (v0.8.x) embedded as a library (not the prebuilt server binary).
Rust-native, Metal acceleration for Apple Silicon, no C++ FFI.

Exact builder/method names (e.g. `GgufModelBuilder`, `with_isq` / `with_auto_isq`,
`with_paged_attn`, `IsqType::Q4K` vs `IsqBits::Four`) vary across 0.8.x point
releases and MUST be verified against `docs.rs/mistralrs` and the in-repo
`mistralrs/examples/` at implementation time. The design pins behavior, not exact
identifiers.

## Default model

Qwen2.5-7B-Instruct, GGUF Q4_K_M. Auto-downloaded from HuggingFace on first run
(cached under the HF cache dir). Chosen for strong open-source tool-calling in the
7-8B class. Model id and quant are configurable via CLI flags.

Approx footprint: weights ~4.5-5 GB + KV cache ~1-2 GB ≈ 6-7 GB total. Leaves
headroom on 16 GB.

## Optimizations (all applied)

1. **ISQ / GGUF Q4_K_M** — 4-bit weights.
2. **Metal** — `metal` cargo feature; GPU offload on M1.
3. **PagedAttention** — OS-style KV paging, reduces context memory fragmentation.
4. **FlashAttention** — via Metal backend where available.
5. **KV cache quantization** — enable if exposed in the pinned 0.8.x; if not
   available, document the gap rather than fake it.
6. **mmap** — default for GGUF loading; model mapped from disk, not copied.
7. **Prefix caching** — reuse shared prompt prefix (agents repeat the system prompt).
8. **Context window 16k** — enough for agent tool-chaining, bounds RAM.

Any optimization that the pinned mistralrs version does not actually expose is
documented as unavailable, not silently skipped or faked.

## Architecture

```
CLI (Claude Code / Codex / curl)
        │ HTTP localhost:8080
        ▼
  axum router
   /v1/messages          (Anthropic Messages API)
   /v1/chat/completions   (OpenAI Chat Completions API)
   /v1/models             (OpenAI model list)
   /health
        │  translate to internal request
        ▼
  engine (wraps mistralrs Model)
   GgufModelBuilder + ISQ Q4 + Metal + PagedAttn + KV-quant
        │
        ▼
  mistralrs inference
```

## Modules

| File | Responsibility |
|------|----------------|
| `src/main.rs` | CLI args (port, model id, quant, ctx len), startup, graceful shutdown |
| `src/config.rs` | Config struct + optimization flags, defaults |
| `src/engine.rs` | Load model via mistralrs, run generation, streaming, parse tool calls out |
| `src/api/common.rs` | Internal request/response representation + shared translation helpers |
| `src/api/openai.rs` | OpenAI request/response types + `/v1/chat/completions`, `/v1/models` handlers |
| `src/api/anthropic.rs` | Anthropic request/response types + `/v1/messages` handler |
| `src/tools.rs` | Tool schema passthrough, tool-call parsing, OpenAI↔Anthropic tool mapping |
| `scripts/` | curl test scripts exercising both endpoints with tools |
| `tests/` | Integration tests for translation layer and tool parsing |

Each module has one clear purpose and a narrow interface. `engine` does not know
about HTTP; `api/*` does not know about mistralrs internals. They meet at
`api/common.rs`'s internal types.

## Data flow — tool calling

1. Client sends request with `tools` (OpenAI: `tools[].function`; Anthropic:
   `tools[].input_schema`).
2. `api/*` translates to internal request; `tools.rs` normalizes tool schemas.
3. `engine` passes tool schemas to mistralrs (strict schema / grammar enforcement)
   and runs generation.
4. Model emits a tool call. `engine` parses it into the internal tool-call form.
5. `api/*` renders it in the client's format:
   - OpenAI: `choices[].message.tool_calls[]` with `finish_reason: "tool_calls"`.
   - Anthropic: `content[].{type: "tool_use", id, name, input}` with
     `stop_reason: "tool_use"`.
6. Client runs the tool, sends the result back (OpenAI: `role: "tool"` message;
   Anthropic: `tool_result` content block). Server feeds it in and the model
   produces the final answer. This is the chaining loop under test.

## Streaming

Server-Sent Events on both surfaces:
- OpenAI: `data: {chunk}` lines, terminated by `data: [DONE]`.
- Anthropic: `message_start`, `content_block_start`, `content_block_delta`,
  `content_block_stop`, `message_delta`, `message_stop` events.

## Error handling

- Model not yet downloaded → download with progress to stderr.
- Insufficient RAM / load failure → clear startup error, non-zero exit.
- Invalid tool schema in request → HTTP 400 with JSON error in the caller's format.
- Inference failure → HTTP 500 with JSON error.

## Acceptance test

1. `GET /health` → 200.
2. OpenAI `/v1/chat/completions` no tools → coherent completion.
3. OpenAI with tools → model emits `tool_calls`; we return a tool result; model
   produces the final answer (full chaining).
4. Anthropic `/v1/messages` → same chaining flow with `tool_use` / `tool_result`.
5. Point Codex CLI at `localhost:8080` (OpenAI base URL) → real conversation works.
6. Report measured peak RAM and tokens/sec for the default model.

Tools used in the test: `get_weather(location)` and `calculate(expression)`,
exercising a two-step chain.

## Open implementation questions (resolve at build time)

- Exact mistralrs 0.8.x API identifiers (verify against docs.rs + examples).
- Whether KV-cache quantization and prefix caching are exposed as builder options
  in the pinned version; document what is and isn't available.
