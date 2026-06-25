# llama-cpp-2 API Notes (verified against v0.1.150 source)

## Crate details

| Item | Value |
|------|-------|
| Crate | `llama-cpp-2 v0.1.150` |
| Sys crate | `llama-cpp-sys-2 v0.1.150` |
| Metal feature name | `metal` (re-exports `llama-cpp-sys-2/metal`) |
| Builds llama.cpp at build time? | YES — `llama-cpp-sys-2` bundles the full llama.cpp C++ source and compiles it via `build.rs` using `cc`/`cmake`. The Metal shaders are compiled at build time with `xcrun metal`. First build takes several minutes. |
| Apple Silicon auto-Metal | YES — `Cargo.toml` unconditionally enables `features = ["metal"]` for `target_os = "macos"` + `aarch64`. Adding `features = ["metal"]` to your dependency is redundant but harmless. |

---

## 1. Backend initialization

```rust
use llama_cpp_2::llama_backend::LlamaBackend;

let backend = LlamaBackend::init()?;   // singleton — panics if called twice
// Only one LlamaBackend may exist per process.
```

- `LlamaBackend::init() -> Result<LlamaBackend, LlamaCppError>`
- `LlamaBackend::supports_gpu_offload(&self) -> bool`
- Drop calls `llama_backend_free` and resets the singleton so another `init()` is possible.

---

## 2. Model load

```rust
use llama_cpp_2::model::{LlamaModel, params::LlamaModelParams};

let model_params = LlamaModelParams::default()
    .with_n_gpu_layers(u32::MAX);   // offload all layers to Metal

let model = LlamaModel::load_from_file(
    &backend,
    PathBuf::from("/path/to/model.gguf"),
    &model_params,
)?;
```

Key `LlamaModelParams` builder methods:
- `.with_n_gpu_layers(n: u32) -> Self`  (u32::MAX = offload everything)
- `.with_use_mmap(bool) -> Self`
- `.with_use_mlock(bool) -> Self`
- `Default::default()` uses `n_gpu_layers = -1` (let llama.cpp decide)

Note: `n_gpu_layers` is stored as `i32` internally; `u32::MAX` clamps to `i32::MAX`.

---

## 3. Context creation

```rust
use llama_cpp_2::context::params::LlamaContextParams;
use std::num::NonZeroU32;

let ctx_params = LlamaContextParams::default()
    .with_n_ctx(NonZeroU32::new(2048));

let mut ctx = model.new_context(&backend, ctx_params)?;
```

- `LlamaContextParams::default()` → n_ctx = 512
- `.with_n_ctx(Option<NonZeroU32>) -> Self`
- `model.new_context(&backend, params: LlamaContextParams) -> Result<LlamaContext<'_>, LlamaContextLoadError>`
- `LlamaContext` lifetime is tied to `LlamaModel` (`'a`).

---

## 4. Tokenize

```rust
use llama_cpp_2::model::AddBos;

let tokens: Vec<LlamaToken> = model.str_to_token("hello world", AddBos::Always)?;
```

- `model.str_to_token(&str, AddBos) -> Result<Vec<LlamaToken>, StringToTokenError>`
- `AddBos::Always` | `AddBos::Never`

---

## 5. Batch creation and decode

```rust
use llama_cpp_2::llama_batch::LlamaBatch;

// allocate batch for n_tokens tokens, 1 sequence max
let mut batch = LlamaBatch::new(n_tokens, 1);

// add tokens: (token, position, seq_ids, emit_logits)
batch.add(token, pos as i32, &[0_i32], logits)?;

// alternatively: add a whole sequence (sets logit=true only on last)
batch.add_sequence(&tokens, seq_id, logits_all)?;

ctx.decode(&mut batch)?;     // fills logit cache
batch.clear();               // resets n_tokens=0 (no dealloc)
```

- `LlamaBatch::new(n_tokens: usize, n_seq_max: i32) -> Self`
- `batch.add(token: LlamaToken, pos: i32, seq_ids: &[i32], logits: bool) -> Result<(), BatchAddError>`
- `batch.add_sequence(tokens: &[LlamaToken], seq_id: i32, logits_all: bool) -> Result<(), BatchAddError>`
- `ctx.decode(&mut batch) -> Result<(), DecodeError>`
- `batch.clear()` — resets token count, keeps allocation

---

## 6. Sampling

```rust
use llama_cpp_2::sampling::LlamaSampler;

let mut sampler = LlamaSampler::chain_simple([
    LlamaSampler::temp(0.0),   // temperature=0 → greedy
    LlamaSampler::greedy(),
]);

// sample from logit slot idx (use last batch token index after prefill;
// use 0 after single-token decode batches)
let token: LlamaToken = sampler.sample(&ctx, idx);
sampler.accept(token);    // update sampler state (repetition penalties etc)
```

Key `LlamaSampler` constructors (all `-> Self`):
- `LlamaSampler::greedy()`
- `LlamaSampler::temp(t: f32)` — temperature scaling
- `LlamaSampler::dist(seed: u32)` — multinomial sampling
- `LlamaSampler::top_k(k: i32)`
- `LlamaSampler::top_p(p: f32, min_keep: usize)`
- `LlamaSampler::min_p(p: f32, min_keep: usize)`
- `LlamaSampler::penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present)`
- `LlamaSampler::chain(samplers, no_perf: bool) -> Self`
- `LlamaSampler::chain_simple(samplers) -> Self` — same with no_perf=false

- `sampler.sample(&ctx, idx: i32) -> LlamaToken`
- `sampler.accept(token: LlamaToken)`

---

## 7. Detokenize / streaming output

```rust
use encoding_rs;

let mut decoder = encoding_rs::UTF_8.new_decoder();  // stateful; reuse across tokens

let piece: String = model.token_to_piece(
    token,
    &mut decoder,
    special: bool,       // true = decode special tokens like <|im_start|>
    lstrip: None,        // Option<NonZeroU16>; None = no leading space strip
)?;
```

- `model.token_to_piece(token, decoder, special, lstrip) -> Result<String, TokenToStringError>`
- Uses `encoding_rs::UTF_8.new_decoder()` (stateful) for correct multi-byte streaming.
- `model.is_eog_token(token: LlamaToken) -> bool` — check for end-of-generation.
- `model.token_eos() -> LlamaToken`, `model.token_bos() -> LlamaToken`

---

## 8. KV State save/load to file — CONFIRMED WORKING

### Full-context state (all sequences)

```rust
// Save: must pass the tokens that have been processed
ctx.state_save_file(
    path: impl AsRef<Path>,
    tokens: &[LlamaToken],
) -> Result<(), SaveSessionError>

// Load into a context (fresh or existing):
let loaded_tokens: Vec<LlamaToken> = ctx2.state_load_file(
    path: impl AsRef<Path>,
    max_tokens: usize,   // must be >= number of tokens saved
) -> Result<Vec<LlamaToken>, LoadSessionError>
```

Underlying FFI: `llama_state_save_file` / `llama_state_load_file` (in `llama-cpp-sys-2`).

### Per-sequence state (multi-sequence scenarios)

```rust
// Save one sequence:
let bytes_written: usize = ctx.state_seq_save_file(
    filepath,
    seq_id: i32,
    tokens: &[LlamaToken],
) -> Result<usize, SaveSeqStateError>

// Load into a destination sequence ID:
let (loaded_tokens, bytes_read): (Vec<LlamaToken>, usize) = ctx2.state_seq_load_file(
    filepath,
    dest_seq_id: i32,
    max_tokens: usize,
) -> Result<(Vec<LlamaToken>, usize), LoadSeqStateError>
```

Underlying FFI: `llama_state_seq_save_file` / `llama_state_seq_load_file`.

### In-memory state (no file)

```rust
let size = ctx.get_state_size();         // max bytes needed
let n = unsafe { ctx.copy_state_data(dest: *mut u8) -> usize };
let n = unsafe { ctx.set_state_data(src: &[u8]) -> usize };
```

### Deprecated aliases (still functional)
- `ctx.save_session_file(path, tokens)` → delegates to `llama_save_session_file`
- `ctx.load_session_file(path, max_tokens)` → delegates to `llama_load_session_file`

---

## 9. Continuation after state load

After `state_load_file`, the KV cache is populated but no logits are available yet.
To continue generation you must re-decode the last token to prime the logit buffer:

```rust
let last_tok = *all_tokens.last().unwrap();
batch.clear();
batch.add(last_tok, last_pos, &[0], true)?;
ctx2.decode(&mut batch)?;
// now sample as normal
let next = sampler.sample(&ctx2, 0);
```

This is a one-token "warm-up" decode — it runs fast.

---

## 10. What is NOT available at the high-level API

| Feature | Status |
|---------|--------|
| KV state save/load | AVAILABLE (`state_save_file` / `state_load_file`) |
| Per-sequence state save/load | AVAILABLE (`state_seq_save_file` / `state_seq_load_file`) |
| Flash attention toggle | Not exposed at high level; set via `LlamaContextParams` raw field or `with_flash_attn` if present |
| Grammar-constrained sampling | Available but requires `common` feature |
| Embeddings | Available but requires `embeddings = true` context param |
| Multi-GPU tensor split | Available via `with_devices` / `with_split_mode` |

---

## 11. Build notes

- `llama-cpp-sys-2` downloads/vendors llama.cpp C++ sources at crate publish time (bundled in the crate, not downloaded at build time).
- The `build.rs` compiles llama.cpp via the `cc` crate. Metal shaders are compiled via `xcrun metal`.
- First compile: ~5–10 minutes on M1 Pro (compiles ~100+ C++ files + Metal shaders).
- Incremental rebuilds: seconds (only Rust code recompiles).
- The `metal` feature name is exactly `metal` in both `llama-cpp-2` and `llama-cpp-sys-2`.
- On Apple Silicon macOS, `metal` is automatically enabled by a `[target]` table in `llama-cpp-2`'s own `Cargo.toml`, so the explicit `features = ["metal"]` in your `Cargo.toml` is redundant but harmless.
