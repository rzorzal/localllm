# localllm

A local LLM inference server that loads Qwen2.5-7B-Instruct (GGUF Q4_K_M, split shards) via [mistralrs](https://github.com/EricLBuehler/mistral.rs) and serves two API surfaces:

- **OpenAI** — `POST /v1/chat/completions`, `GET /v1/models`
- **Anthropic** — `POST /v1/messages`
- **Health** — `GET /health`

Both APIs support tool calling (function calling) with full two-step tool chaining, and SSE streaming.

---

## Build

```bash
MISTRALRS_METAL_PRECOMPILE=0 cargo build --release
```

> **Why `MISTRALRS_METAL_PRECOMPILE=0`?**  The Metal shader toolchain is broken on this machine — the precompiled Metal shaders fail to compile with Apple's `metallib`. Setting this env var disables Metal shader precompilation so the binary can be built and run on CPU. If you fix the toolchain (`xcodebuild -downloadComponent MetalToolchain` and install full Xcode), you can remove this variable and build with: `cargo build --release --features metal` for GPU acceleration.

---

## Run

```bash
MISTRALRS_METAL_PRECOMPILE=0 ./target/release/localllm --port 8080
```

The server loads the model from HuggingFace cache on startup (~10s on CPU). You will see:

```
INFO  localllm: listening on http://127.0.0.1:8080
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | TCP port |
| `--model-id` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | HuggingFace repo |
| `--gguf-file` | (two shards) | GGUF shard filename(s) |
| `--ctx-len` | 16384 | KV-cache context length (informational) |
| `--force-cpu` | true | Force CPU inference |

---

## API Endpoints

### OpenAI (`/v1/chat/completions`)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### Anthropic (`/v1/messages`)

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Health check

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## Pointing AI tools at localllm

### Codex CLI

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
codex "explain this code"
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
claude "explain this code"
```

---

## Tool Calling

Both APIs support OpenAI-style function calling with two-step chaining. The acceptance tests in `scripts/` demonstrate and verify the full flow end-to-end.

Run the tests:

```bash
bash scripts/test_openai_tools.sh
bash scripts/test_anthropic_tools.sh
```

---

## Optimizations

| Optimization | Status | Notes |
|---|---|---|
| GGUF Q4_K_M weights | **ACTIVE** | ~4.5 GB on disk, loaded via mmap |
| Prefix caching | **ACTIVE** | Enabled by default in mistralrs (sequence-level) |
| mmap weight loading | **ACTIVE** | GGUF files are memory-mapped |
| 16k+ context window | **ACTIVE** | Model supports 131k; KV-cache default 16k |
| Metal GPU acceleration | **UNAVAILABLE** | Metal shader toolchain broken on this machine. Fix: `xcodebuild -downloadComponent MetalToolchain` then rebuild without `MISTRALRS_METAL_PRECOMPILE=0` |
| Paged attention | **UNAVAILABLE** | Requires working Metal runtime or CUDA; skipped on force_cpu |
| Flash Attention | **UNAVAILABLE** | Requires CUDA (`flash-attn` cargo feature) |
| KV-cache quantization | **UNAVAILABLE** | Not exposed by `GgufModelBuilder` API |
| ISQ re-quantization | **UNAVAILABLE** | Only works with `TextModelBuilder`, not GGUF |

**Current mode:** CPU-only inference. All model weights loaded from GGUF Q4_K_M shards via mmap with prefix caching enabled.

---

## Performance (measured on this machine)

Measured with `scripts/measure.sh` on CPU inference (Apple Silicon, force_cpu=true):

| Metric | Value |
|---|---|
| Peak RAM (RSS) | ~4518 MB (~4.5 GB) |
| Completion tokens | 72 |
| Elapsed time | 9s |
| Throughput | **8.0 tok/s** |

Prompt: *"In exactly two sentences, explain what a transformer neural network is."*

Model response: *"A transformer neural network is a type of deep learning model designed for tasks involving natural language processing, particularly excelling at understanding the context and meaning of text by analyzing the relationships between words in a sentence. It achieves this by using self-attention mechanisms to weigh the importance of different words relative to each other, without relying on fixed-length sequences of hidden layers."*

> **Note on RAM:** The auto device mapper uses `max_seq_len=512` for memory estimation (reduced from the default 4096 to fit within available RAM on this loaded machine). At runtime the model generates at its full context length. If you have more free RAM, increase or remove the device-map limit in `src/engine.rs`.

---

## Acceptance Tests

```bash
# Start server
MISTRALRS_METAL_PRECOMPILE=0 ./target/release/localllm --port 8080 &

# Wait for health
until curl -sf localhost:8080/health; do sleep 2; done

# Run tests
bash scripts/test_openai_tools.sh    # Tests OpenAI two-step tool chain
bash scripts/test_anthropic_tools.sh # Tests Anthropic two-step tool chain
bash scripts/measure.sh              # Measures RAM and tok/s
```

All tests pass on CPU inference. Each inference call on CPU takes ~5-15 seconds for short completions.

---

## Model

**Qwen2.5-7B-Instruct-GGUF** (Q4_K_M quantization, split into 2 shards)

- HuggingFace: [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- Context: 131,072 tokens (native), 16,384 KV-cache default
- Architecture: Qwen2, 28 layers, 3584 embedding dim
- Quantization: GGUF Q4_K_M (F16 dtype at runtime)
