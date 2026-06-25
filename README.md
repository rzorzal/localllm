# localllm

A lightweight local LLM inference server. Loads a Qwen2.5 GGUF model via
[mistralrs](https://github.com/EricLBuehler/mistral.rs), runs it on the Apple
GPU (Metal), and serves two API surfaces so existing AI tools can point at it:

- **OpenAI** — `POST /v1/chat/completions`, `GET /v1/models`
- **Anthropic** — `POST /v1/messages`
- **Health** — `GET /health`

Both APIs support tool calling (function calling) with full two-step tool
chaining, and SSE streaming. The default model is the lightweight
**Qwen2.5-3B-Instruct** so it coexists with your other apps for small local
tasks; swap in a larger model when you have RAM to spare (see below).

---

## Build

```bash
cargo build --release
```

> **Metal toolchain note.** Building the Metal GPU shaders requires Apple's
> Metal Toolchain (`xcrun metal`). If you hit
> `cannot execute tool 'metal' due to missing Metal Toolchain`, install it:
> ```bash
> xcodebuild -runFirstLaunch          # repairs Xcode plugins if needed
> xcodebuild -downloadComponent MetalToolchain
> ```
> As a fallback (no GPU), you can build with `MISTRALRS_METAL_PRECOMPILE=0` and
> run with `--force-cpu true` — that skips shader precompilation and runs on CPU.

---

## Run

```bash
./target/release/localllm --port 31415
```

The model is downloaded to the HuggingFace cache on first run, then loaded
(~10s). You'll see:

```
INFO  localllm: loading model Qwen/Qwen2.5-3B-Instruct-GGUF (force_cpu=false)…
INFO  localllm: listening on http://127.0.0.1:31415
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `31415` | TCP port (binds `127.0.0.1` only) |
| `--model-id` | `Qwen/Qwen2.5-3B-Instruct-GGUF` | HuggingFace GGUF repo |
| `--gguf-file` | `qwen2.5-3b-instruct-q4_k_m.gguf` | GGUF filename(s); repeat for split models |
| `--ctx-len` | `8192` | Context window in tokens (sizes the GPU KV cache) |
| `--no-paged-attn` | `false` | Disable PagedAttention |
| `--force-cpu` | `false` | Force CPU instead of the Apple GPU |

---

## Choosing a model and context size

The two levers that matter on a 16 GB Mac:

- **Model size (parameters)** — bigger = smarter, slower, more RAM.
- **`--ctx-len` (context window)** — bigger = handles longer input, more KV-cache
  RAM (~55 KB/token: 8k ≈ 0.3 GB, 32k ≈ 1.8 GB).

### Light + fast (default) — small tasks alongside other apps
```bash
./target/release/localllm
# Qwen2.5-3B, 8k context. ~2.3 GB, ~39 tok/s on M1 Pro.
```

### Smarter — when you have RAM free
First download the model, then point the flags at it:
```bash
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf

./target/release/localllm \
  --model-id Qwen/Qwen2.5-7B-Instruct-GGUF \
  --gguf-file qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --gguf-file qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
# Qwen2.5-7B. Smarter but ~15 tok/s and heavier on RAM.
```

### Longer context (e.g. bigger documents)
```bash
./target/release/localllm --ctx-len 32768
# More KV-cache RAM; watch swap on a 16 GB machine.
```

The tokenizer repo is derived automatically by stripping `-GGUF` from
`--model-id`, so any `Qwen/...-GGUF` repo works out of the box.

---

## API examples

### OpenAI
```bash
curl http://localhost:31415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

### Anthropic
```bash
curl http://localhost:31415/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"local","max_tokens":128,"messages":[{"role":"user","content":"Hello!"}]}'
```

### Health
```bash
curl http://localhost:31415/health   # {"status":"ok"}
```

---

## Pointing AI tools at localllm

### Codex CLI
```bash
export OPENAI_BASE_URL=http://localhost:31415/v1
codex "explain this code"
```

### Claude Code
```bash
ANTHROPIC_BASE_URL=http://localhost:31415 ANTHROPIC_API_KEY=local claude
```

> **Heads-up on Claude Code.** Claude Code sends a large agentic prompt
> (~26k tokens of system instructions + ~27 tool schemas) on *every* turn. To
> use it you must raise `--ctx-len` to at least `32768`, and on a 16 GB machine
> that combination (model + 1.8 GB KV cache + your other apps) will swap and
> respond slowly. It works, but local agentic coding wants more RAM or a
> smaller model. Simple clients and small prompts run great.

---

## Request logging

Every request is logged with a short id so concurrent prompts are easy to
follow:

```
req-1a2b3c4d [anthropic] start: model=local msgs=3 tools=1 stream=true
req-1a2b3c4d [anthropic] done (buffered stream): finish=ToolCalls completion_tok=20 0.9s 22.1 tok/s
```

Set `RUST_LOG=localllm=info` (or `debug`) to control verbosity.

---

## Tool calling

Both APIs support function calling with two-step chaining. Streaming requests
that carry tools use a "buffered streaming" path: the response is generated in
full (reusing the non-streaming tool-call path) then replayed as correct SSE
(`tool_calls` for OpenAI, `tool_use` blocks for Anthropic). Tool-less streaming
requests stream token-by-token.

```bash
bash scripts/test_openai_tools.sh
bash scripts/test_anthropic_tools.sh
```

---

## Optimizations

| Optimization | Status | Notes |
|---|---|---|
| GGUF Q4_K_M weights | **ACTIVE** | Loaded via mmap |
| Metal GPU acceleration | **ACTIVE** | Runs on the Apple GPU (BF16 compute) |
| PagedAttention | **ACTIVE** | OS-style KV paging on the GPU; sized to `--ctx-len` |
| Prefix caching | **ACTIVE** | Reuses shared prompt prefixes across turns |
| mmap weight loading | **ACTIVE** | GGUF files are memory-mapped |
| Flash Attention | **UNAVAILABLE** | Requires CUDA (`flash-attn` feature); Metal uses its own kernels |
| KV-cache quantization | **UNAVAILABLE** | Not exposed by `GgufModelBuilder` |
| ISQ re-quantization | **UNAVAILABLE** | GGUF is already quantized; ISQ is for `TextModelBuilder` |

To fall back to CPU (no Metal), build with `MISTRALRS_METAL_PRECOMPILE=0` and run
with `--force-cpu true`. PagedAttention is GPU-only and is skipped on CPU.

---

## Performance (measured on this machine — Apple M1 Pro, 16 GB)

| Model | Context | Throughput | KV cache | Notes |
|---|---|---|---|---|
| **Qwen2.5-3B** (default) | 8k | **~39 tok/s** | 288 MB | Light; coexists with other apps |
| Qwen2.5-7B | 16k | ~15 tok/s | 896 MB | Smarter; heavier on 16 GB |

Prefill of very large prompts (e.g. Claude Code's ~26k tokens) is the slow part
on this hardware, especially when the machine is already low on RAM and swapping.

---

## Model

**Qwen2.5-3B-Instruct-GGUF** (Q4_K_M) by default.

- HuggingFace: [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
- Native context: 32,768 tokens (server default KV cache: 8,192)
- Architecture: Qwen2; strong open-source tool-calling in its size class
- Quantization: GGUF Q4_K_M (BF16 compute on the GPU)
