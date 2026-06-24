# mistralrs 0.8.1 API — Verified Identifiers

> Source-verified from `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/mistralrs-0.8.1/`
> and `mistralrs-core-0.8.1/`, `mistralrs-quant-0.8.1/`. Every identifier below was read from actual source.

---

## 1. GGUF Model Builder

```rust
use mistralrs::{GgufModelBuilder, TextMessageRole, TextMessages};

let model = GgufModelBuilder::new(
    "Qwen/Qwen2.5-7B-Instruct-GGUF",          // HF repo ID
    vec!["qwen2.5-7b-instruct-q4_k_m.gguf"],  // file(s) in repo
)
.with_tok_model_id("Qwen/Qwen2.5-7B-Instruct") // optional: explicit tokenizer repo
.with_logging()
.build()
.await?;  // -> anyhow::Result<Model>
```

**CRITICAL CORRECTION vs. brief template:**
`GgufModelBuilder` does NOT have `.with_isq()` or `.with_auto_isq()`.
Those methods exist only on `TextModelBuilder`, `ModelBuilder`, and `MultimodalModelBuilder`.
GGUF files are pre-quantized; applying ISQ on top is not supported via this builder.

---

## 2. ISQ (In-situ Quantization) — for non-GGUF models only

```rust
use mistralrs::{IsqType, IsqBits, TextModelBuilder};

// Option A — specific type:
.with_isq(IsqType::Q4K)       // Method: TextModelBuilder::with_isq(isq: IsqType)

// Option B — auto (platform-optimal):
.with_auto_isq(IsqBits::Four) // Method: TextModelBuilder::with_auto_isq(bits: IsqBits)
```

### IsqType variants (mistralrs-quant-0.8.1 src/lib.rs:575)
```
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
HQQ8, HQQ4,
F8E4M3, AFQ8, AFQ6, AFQ4, AFQ3, AFQ2,
F8Q8, MXFP4
```

### IsqBits variants (mistralrs-quant-0.8.1 src/lib.rs:607)
```
Two, Three, Four, Five, Six, Eight
```

### Metal platform resolution (IsqBits -> IsqType)
| IsqBits | Metal (AFQ) | CUDA/CPU (GGUF-K) |
|---------|-------------|-------------------|
| Two     | AFQ2        | Q2K               |
| Three   | AFQ3        | Q3K               |
| Four    | AFQ4        | Q4K               |
| Five    | Q5K         | Q5K               |
| Six     | AFQ6        | Q6K               |
| Eight   | AFQ8        | Q8_0              |

---

## 3. Paged Attention

```rust
use mistralrs::{PagedAttentionMetaBuilder, MemoryGpuConfig, PagedCacheType};

// On GgufModelBuilder (verified in gguf.rs:186):
.with_paged_attn(PagedAttentionMetaBuilder::default().build()?)

// Builder defaults: block_size=None (auto), mem_gpu=MemoryGpuConfig::ContextSize(4096), cache_type=PagedCacheType::Auto
```

**Metal support:** `paged_attn_supported()` returns `true` when compiled with `metal` feature
(src: mistralrs-core-0.8.1/src/utils/mod.rs:248).
If `paged_attn_supported()` is false, `with_paged_attn()` is a no-op (silently ignored).

`PagedAttentionMetaBuilder` is re-exported from `text_model.rs` (NOT from GgufModelBuilder's own file).

---

## 4. Chat Request Types

### TextMessages (simple, deterministic sampling)
```rust
use mistralrs::{TextMessages, TextMessageRole};

let messages = TextMessages::new()
    .add_message(TextMessageRole::User, "Hello")
    .add_message(TextMessageRole::Assistant, "Hi")
    .add_message(TextMessageRole::System, "You are helpful");
```

### TextMessageRole variants (messages.rs:64)
```
User, Assistant, System, Tool, Custom(String)
```

### RequestBuilder (tools, sampling control, logprobs)
```rust
use mistralrs::{RequestBuilder, Tool, ToolChoice, TextMessageRole};

let req = RequestBuilder::new()
    .add_message(TextMessageRole::User, "What's the weather?")
    .set_tools(vec![tool])          // Vec<Tool>
    .set_tool_choice(ToolChoice::Auto);
```

---

## 5. Sending Requests

### Non-streaming
```rust
// Method: Model::send_chat_request<R: RequestLike>(request: R) -> Result<ChatCompletionResponse>
let resp: ChatCompletionResponse = model.send_chat_request(messages).await?;
```

### Streaming
```rust
use futures::StreamExt;
use mistralrs::Response;

// Method: Model::stream_chat_request<R: RequestLike>(request: R) -> Result<Stream<'_>>
let mut stream = model.stream_chat_request(messages).await?;
while let Some(chunk) = stream.next().await {
    if let Response::Chunk(c) = chunk {
        if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.as_ref()) {
            print!("{text}");
        }
    }
}
```

### Quick single-turn
```rust
// Method: Model::chat(message: impl ToString) -> Result<String>
let text = model.chat("Tell me a joke").await?;
```

---

## 6. Response Type Hierarchy

### ChatCompletionResponse (non-streaming, response.rs:144)
```rust
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,     // typically choices[0]
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
}
```

### Choice (response.rs:87)
```rust
pub struct Choice {
    pub finish_reason: String,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}
```

### ResponseMessage (response.rs:32)
```rust
pub struct ResponseMessage {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    pub reasoning_content: Option<String>,  // chain-of-thought, if enabled
}
```

### Text content field path
```rust
resp.choices[0].message.content   // Option<String>
// or use unwrap_or_default():
resp.choices[0].message.content.clone().unwrap_or_default()
```

### Tool calls field path
```rust
resp.choices[0].message.tool_calls  // Option<Vec<ToolCallResponse>>
```

---

## 7. Tool Call Types

### ToolCallResponse (tools/response.rs:23)
```rust
pub struct ToolCallResponse {
    pub index: usize,
    pub id: String,
    pub tp: ToolCallType,       // ToolCallType::Function
    pub function: CalledFunction,
}
```

### CalledFunction (mistralrs-mcp-0.8.1/src/tools.rs:54)
```rust
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,  // JSON string
}
```

### Tool definition (for set_tools)
```rust
use mistralrs::{Tool, ToolType, Function};

Tool {
    tp: ToolType::Function,
    function: Function {
        description: Some("description".to_string()),
        name: "function_name".to_string(),
        parameters: Some(serde_json::from_value(json!({ ... }))?),
    },
}
```

---

## 8. Streaming Chunk Type

### ChatCompletionChunkResponse (response.rs:160)
```rust
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Option<Usage>,
}
```

### ChunkChoice (response.rs:100)
```rust
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}
```

### Delta (response.rs:48)
```rust
pub struct Delta {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    pub reasoning_content: Option<String>,
}
```

### Streaming text extraction pattern
```rust
if let Response::Chunk(c) = chunk {
    if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.as_ref()) {
        print!("{text}");
    }
}
```

### Response enum variants (response.rs:240)
```rust
pub enum Response {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    ModelError(String, ChatCompletionResponse),
    Done(ChatCompletionResponse),          // non-streaming complete
    Chunk(ChatCompletionChunkResponse),    // streaming chunk — USE THIS
    CompletionModelError(String, CompletionResponse),
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    ImageGeneration(ImageGenerationResponse),
    Speech { pcm, rate, channels },
    Raw { logits_chunks, tokens },
    Embeddings { embeddings, prompt_tokens, total_tokens },
}
```

---

## 9. Feature Support Matrix for mistralrs 0.8.1 + metal

| Feature                        | Supported | Notes |
|-------------------------------|-----------|-------|
| Paged Attention (`with_paged_attn`) | YES | `paged_attn_supported()` returns `true` with `metal` feature |
| KV-cache quantization (ISQ on KV)   | NOT exposed via GgufModelBuilder | No `with_kv_cache_dtype()` on GgufModelBuilder |
| Prefix caching (`with_prefix_cache_n`) | YES | Default 16 sequences; use `.with_prefix_cache_n(None)` to disable |
| ISQ re-quantization on GGUF         | NO | GGUF files are pre-quantized; `GgufModelBuilder` has no `with_isq` |
| Auto ISQ (platform-optimal)         | N/A for GGUF | Only on TextModelBuilder/ModelBuilder |
| Flash Attention                      | NO | Requires `flash-attn` feature (CUDA only) |
| Streaming                            | YES | `model.stream_chat_request()` returns `Stream<'_>` implementing `futures::Stream` |
| Tool calling                         | YES | `RequestBuilder::set_tools(Vec<Tool>)` |
| Structured output (JSON schema)      | YES | `model.generate_structured::<T>()` |
| No-KV-cache mode                     | YES | `GgufModelBuilder::with_no_kv_cache()` |

---

## 10. GgufModelBuilder Complete Method List (gguf.rs)

```
new(model_id, files)                     -> Self  [required]
with_tok_model_id(id)                    -> Self  [recommended for Qwen GGUF]
with_logging()                           -> Self
with_paged_attn(PagedAttentionConfig)    -> Self  [no-op if not supported]
with_prefix_cache_n(Option<usize>)       -> Self  [default: Some(16)]
with_no_kv_cache()                       -> Self
with_max_num_seqs(usize)                 -> Self  [default: 32]
with_device_mapping(DeviceMapSetting)    -> Self
with_device(Device)                      -> Self
with_topology(Topology)                  -> Self
with_topology_from_path(path)            -> anyhow::Result<Self>
with_chat_template(str)                  -> Self
with_tokenizer_json(str)                 -> Self
with_force_cpu()                         -> Self
with_token_source(TokenSource)           -> Self
with_hf_revision(str)                    -> Self
with_jinja_explicit(String)              -> Self
with_search(SearchEmbeddingModel)        -> Self
with_search_callback(Arc<SearchCallback>) -> Self
with_tool_callback(name, Arc<ToolCallback>) -> Self
with_tool_callback_and_tool(name, cb, tool) -> Self
with_throughput_logging()                -> Self
build()                                  -> anyhow::Result<Model>  [async]
```

---

## 11. Import Map for localllm

```rust
use mistralrs::{
    // Builder
    GgufModelBuilder,
    // Request types
    TextMessages, TextMessageRole, RequestBuilder,
    // Response types
    ChatCompletionResponse, ChatCompletionChunkResponse,
    Choice, ChunkChoice, ResponseMessage, Delta,
    Response,
    // Tool types
    Tool, ToolType, Function, ToolCallResponse, ToolCallType, ToolChoice,
    // Paged attention (if used)
    PagedAttentionMetaBuilder, PagedAttentionConfig, MemoryGpuConfig, PagedCacheType,
    // ISQ (NOT for GGUF; only if switching to non-GGUF)
    // IsqType, IsqBits,
    // Utilities
    paged_attn_supported,
};
```
