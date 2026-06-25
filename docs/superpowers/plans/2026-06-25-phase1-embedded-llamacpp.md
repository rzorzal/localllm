# Phase 1 — Embedded llama.cpp engine + static-prefix prewarm

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the inference backend with embedded llama.cpp (via `llama-cpp-2`)
so the single binary gains disk KV-state save/load, then prewarm the static
agentic prefix so Claude-Code-shaped requests return their first token in seconds,
not minutes.

**Architecture:** Keep the dual-API / translation / logging / buffered-streaming
layer untouched. Add a new `LlamaEngine` that implements the existing `Generator`
trait. Select the backend via config. Everything stays in one binary; the model
auto-downloads on first run.

**Tech Stack:** Rust, `llama-cpp-2` (+ `llama-cpp-sys-2`) with the `metal`
feature, existing axum/serde stack.

## Global Constraints

- **Single self-contained binary.** No external `llama-server`, no Homebrew, no
  Python/`hf` CLI at run time. Metal shaders compiled at build time.
- Model auto-downloads in-process over HTTPS on first run; cached under the app's
  data dir (or HF cache). Default model: Qwen2.5-3B-Instruct GGUF Q4_K_M.
- The new engine implements the existing `crate::server::Generator` trait
  (`generate`, `generate_stream`) — the HTTP layer must not change.
- Exact `llama-cpp-2` API identifiers (context/batch/sampler/tokenize and KV
  **state save/load**) are unknown until verified; Task 1 locks them in
  `NOTES-llamacpp-api.md`; later tasks use what Task 1 records. Do not invent APIs.
- `engine_llama` module must not import `axum`; api/* must not import llama types.
- Every claim of "works" requires a real run (build + load + generate).

---

### Task 1: Add llama-cpp-2, scaffold, and LOCK the API (spike)

**Files:**
- Modify: `Cargo.toml` (add `llama-cpp-2` with `metal` feature)
- Create: `src/bin/spike_llama.rs` (throwaway)
- Create: `NOTES-llamacpp-api.md`

**Interfaces:**
- Produces: `NOTES-llamacpp-api.md` documenting the exact, compiling API for:
  model load (with GGUF path + n_gpu_layers for Metal), context creation
  (n_ctx), tokenize, batch/decode, sampler, detokenize/streaming, and the
  **KV state save/load to file** functions (e.g. `save_session_file` /
  `state_seq_save_file` or the `llama-cpp-sys-2` FFI equivalents). Records which
  cargo feature enables Metal and whether the crate downloads/builds llama.cpp at
  build time.

- [ ] **Step 1: Add the dependency**
  Add to `Cargo.toml`: `llama-cpp-2 = { version = "*", features = ["metal"] }`
  (pin the version that resolves). Run `cargo fetch`.

- [ ] **Step 2: Read the crate's real API**
  Inspect the resolved crate source under `~/.cargo/registry/src/*/llama-cpp-2-*/`
  — read `src/lib.rs`, `model.rs`, `context.rs`, and any `examples/`. Note the
  exact builder/method names and the KV-state save/load functions (check
  `llama-cpp-sys-2` FFI for `llama_state_*save_file` if not surfaced高-level).

- [ ] **Step 3: Write a spike** in `src/bin/spike_llama.rs` that loads the
  Qwen2.5-3B GGUF from a local path, creates a context with Metal offload, encodes
  "Reply with the single word: ok", decodes a few tokens, and prints them.
  (Resolve the GGUF path from the HF cache for the spike.)

- [ ] **Step 4: Build and run the spike**
  Run: `cargo run --bin spike_llama --release`
  Expected: compiles (llama.cpp builds; Metal shaders compile), loads the model,
  prints `ok` (or close). Fix identifiers against Step 2 until it runs. The model
  MUST actually generate before moving on.

- [ ] **Step 5: Verify KV state save/load**
  Extend the spike: after decoding, save the sequence/session state to a temp file
  and reload it into a fresh context; confirm no error and that continuing
  generation works. This proves the persistence primitive Phase 1 depends on.

- [ ] **Step 6: Record `NOTES-llamacpp-api.md`** with every verified identifier
  (load, context, tokenize, batch/decode, sampler, detokenize, state save/load
  file functions, Metal feature). Note anything NOT available.

- [ ] **Step 7: Commit**
  `git add Cargo.toml Cargo.lock src/bin/spike_llama.rs NOTES-llamacpp-api.md`
  `git commit -m "chore(phase1): add llama-cpp-2 and lock the API (incl. KV state save/load)"`

---

### Task 2: In-process model download

**Files:**
- Create: `src/download.rs`
- Modify: `src/lib.rs` (`pub mod download;`)
- Test: inline unit test for URL/path construction

**Interfaces:**
- Produces: `async fn ensure_model(model_id: &str, files: &[String]) -> anyhow::Result<Vec<PathBuf>>`
  that downloads each GGUF from HuggingFace (`https://huggingface.co/{repo}/resolve/main/{file}`)
  to a local cache dir if missing, and returns local paths. Streams to disk with a
  simple progress log; skips files already present (size check).

- [ ] **Step 1: Failing test** for the resolve-URL + cache-path builder
  (`fn hf_url(repo, file) -> String`, `fn cache_path(repo, file) -> PathBuf`):
  assert the URL is `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf`
  and the cache path ends with the filename.
- [ ] **Step 2: Run test → fails** (`cargo test --lib download`).
- [ ] **Step 3: Implement** using `reqwest` (blocking-to-file async streaming) +
  `std::fs`. Add `reqwest` (rustls) to Cargo.toml. `ensure_model` skips existing
  files, downloads missing ones to the cache dir, logs progress.
- [ ] **Step 4: Run test → passes.**
- [ ] **Step 5: Manual check** — delete one cached shard, run a tiny `main` path
  that calls `ensure_model` for the 3B file, confirm it re-downloads. (Use the
  already-cached file to avoid a full re-download in CI.)
- [ ] **Step 6: Commit** `feat(phase1): in-process HuggingFace model download`.

---

### Task 3: LlamaEngine — load + non-streaming generate

**Files:**
- Create: `src/engine_llama.rs`
- Modify: `src/lib.rs`, `src/server.rs` (impl `Generator` for `LlamaEngine`)
- Test: manual run (inference)

**Interfaces:**
- Consumes: verified API from `NOTES-llamacpp-api.md`; `ensure_model` (Task 2);
  `ChatRequest`/`ChatResult` and the `Generator` trait.
- Produces: `LlamaEngine` with `async fn load(cfg) -> Result<Self>` and an impl of
  `Generator::generate` that builds the prompt from `ChatRequest` (apply the
  model's chat template incl. tools), tokenizes, decodes to completion, and maps
  to `ChatResult` (text + finish reason + token counts). Tool-call parsing: parse
  the model's tool-call output per the chat template (Qwen emits tool calls in a
  known format) into `ContentPart::Call`.

- [ ] **Step 1: Implement `load`** — `ensure_model`, then load model + context
  with Metal offload and `n_ctx = ctx_len`, using NOTES identifiers.
- [ ] **Step 2: Implement `generate`** — render `ChatRequest` to a prompt via the
  model's chat template (include tools), tokenize, decode with a sampler until EOS
  or max_tokens, detokenize, parse tool calls, fill token counts.
- [ ] **Step 3: Impl `Generator` for `LlamaEngine`** in server.rs (alongside the
  existing Engine impl).
- [ ] **Step 4: Temporary main smoke** — load LlamaEngine, send a `get_weather`
  tool request, print whether a tool call came back.
  Run: `cargo run --release` → prints a parsed `get_weather` call.
- [ ] **Step 5: Commit** `feat(phase1): llama.cpp engine with chat-template prompt + tool parsing`.

---

### Task 4: Streaming + backend selection + config

**Files:**
- Modify: `src/engine_llama.rs` (add `generate_stream`), `src/config.rs`
  (add `--backend llama|mistralrs`, default `llama`), `src/main.rs` (select backend)
- Test: inline + manual stream check

**Interfaces:**
- Produces: `Generator::generate_stream` for `LlamaEngine` (token-by-token via the
  decode loop, mapping to `StreamDelta`); `Config.backend` selecting which engine
  `main` constructs.

- [ ] **Step 1: Implement `generate_stream`** — same decode loop, emit a
  `StreamDelta{text}` per detokenized piece, terminal delta on EOS/length.
- [ ] **Step 2: Config flag** `--backend` (default `llama`) + test asserting default.
- [ ] **Step 3: Wire `main`** to build `LlamaEngine` or the mistralrs `Engine`
  based on `cfg.backend`, both coerced to `Arc<dyn Generator>`.
- [ ] **Step 4: Manual** — start server (llama backend), run
  `scripts/test_openai_tools.sh` and a streaming curl; both work.
- [ ] **Step 5: Commit** `feat(phase1): llama.cpp streaming + backend selection`.

---

### Task 5: Static-prefix prewarm + KV reuse + TTFT measurement

**Files:**
- Modify: `src/engine_llama.rs` (prefix prewarm + KV state reuse), `src/main.rs`
  (prewarm at startup before announcing readiness)
- Create: `scripts/measure_ttft.sh`

**Interfaces:**
- Produces: prewarm logic that, given the static prefix (captured from the first
  request or a configured prompt), prefills it once, saves the KV state to disk,
  and on each subsequent request restores it so only the delta is prefilled.
  `measure_ttft.sh` sends a Claude-Code-shaped request and reports time-to-first
  token, run twice (cold vs warm).

- [ ] **Step 1: Capture the static prefix** — on the first request, record the
  longest stable leading token span (system + tools); hash it; keep its KV state.
- [ ] **Step 2: Save/restore KV state** around the prefix using the NOTES
  save/load-file functions; on a request whose prefix hash matches, restore and
  prefill only the delta.
- [ ] **Step 3: Startup prewarm** — if a prefix is known/configured, prefill it in
  the background before logging "listening", so the first interactive request is
  already warm.
- [ ] **Step 4: `measure_ttft.sh`** — POST a ~5–10k-token system+tools request,
  measure TTFT; run cold then warm; print both.
- [ ] **Step 5: Run it** — report cold vs warm TTFT. Expected: warm is multiples
  faster (target sub-second prefill of the delta after the prefix is cached).
- [ ] **Step 6: Commit** `feat(phase1): static-prefix prewarm and KV reuse + TTFT measurement`.

---

## Phase boundary

After Task 5, measure against a real Claude-Code-shaped request and decide whether
TTFT is now within Claude Code's timeout. Then proceed to Phase 2 (Tscg), which
will be planned in its own document. Phases 3 (MoE + expert offload) and 4 (lazy
tool-gating + persistent Q4 KV) follow, each planned when reached.
