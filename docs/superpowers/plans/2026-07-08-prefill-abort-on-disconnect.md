# Prefill Abort on Disconnect Implementation Plan (sub-2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Abort the local model's prefill promptly when the client has disconnected, so an abandoned (timed-out) request stops pinning the single engine.

**Architecture:** Split the prompt-tail prefill (currently one uninterruptible `ctx.decode`) into fixed-size chunks and check a `cancelled()` predicate between them. Streaming and non-streaming jobs pass `|| reply.is_closed()` (mpsc and oneshot both expose it); on abort, keep the partial KV prefix warm and return early.

**Tech Stack:** Rust, llama-cpp-2 (`LlamaBatch`, `ctx.decode`), tokio channels.

## Global Constraints

- Prefill chunk size `PREFILL_CHUNK = 2048` (matches the context `n_ubatch`).
- `cancelled` is checked only at chunk boundaries; it must never panic (`is_closed()` is an infallible bool read).
- On abort: set `cached_tokens` to the prefix actually prefilled (`new_tokens[..effective_common + prefilled]`), log `aborted: client disconnected during prefill`, and return `Ok((prompt_len, 0, false))` — the send to the already-dropped receiver is a harmless no-op, so no separate "aborted" signal is threaded to callers.
- Logits are requested on exactly ONE token: the last token of the FINAL chunk. `last_idx` for the first sample = (final chunk length) − 1.
- Do not change the per-token decode abort (`on_piece` returning false) — it already handles disconnect during generation.
- Background `Job::Prefill` (warm-up) passes `|| false` (no cancellation); `Job::EstimateCold` does not call `run_decode_loop`.

---

### Task 1: `prefill_chunks` helper + `PREFILL_CHUNK` const

**Files:**
- Modify: `src/engine_llama.rs` (add the const + pure helper + test)

**Interfaces:**
- Produces: `const PREFILL_CHUNK: usize = 2048;`, `fn prefill_chunks(n_tail: usize, chunk: usize) -> Vec<(usize, usize, bool)>` returning `(start, len, is_final)` triples covering `[0, n_tail)`.

- [ ] **Step 1: Write the failing test**

Add to the `engine_llama.rs` tests module:

```rust
#[test]
fn prefill_chunks_covers_tail_and_marks_final() {
    // smaller than one chunk → single final chunk
    assert_eq!(prefill_chunks(100, 2048), vec![(0, 100, true)]);
    // exact multiple
    assert_eq!(prefill_chunks(4096, 2048), vec![(0, 2048, false), (2048, 2048, true)]);
    // 2.5 chunks → 3, only last is_final, lengths sum to n_tail
    let cs = prefill_chunks(5120, 2048);
    assert_eq!(cs, vec![(0, 2048, false), (2048, 2048, false), (4096, 1024, true)]);
    assert_eq!(cs.iter().map(|(_, l, _)| l).sum::<usize>(), 5120);
    // empty
    assert!(prefill_chunks(0, 2048).is_empty());
}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cargo test -p localllm prefill_chunks`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement the const + helper**

Add near the top of `src/engine_llama.rs` (module scope, by the other consts):

```rust
/// Prompt-prefill batch size (tokens per `ctx.decode`). Matches the context
/// n_ubatch so chunking adds no real overhead, and lets us check for client
/// disconnect between chunks instead of blocking on one giant decode.
const PREFILL_CHUNK: usize = 2048;

/// Split a prefill tail of `n_tail` tokens into `(start, len, is_final)` chunks
/// of at most `chunk` tokens. Only the last chunk is flagged final (its last
/// token is where logits are requested).
fn prefill_chunks(n_tail: usize, chunk: usize) -> Vec<(usize, usize, bool)> {
    let chunk = chunk.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < n_tail {
        let len = (n_tail - start).min(chunk);
        let is_final = start + len >= n_tail;
        out.push((start, len, is_final));
        start += len;
    }
    out
}
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cargo test -p localllm prefill_chunks`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine_llama.rs
git commit -m "feat(engine): prefill_chunks helper + PREFILL_CHUNK const"
```

---

### Task 2: Cancellable chunked prefill + wire the call sites

**Files:**
- Modify: `src/engine_llama.rs` (`run_decode_loop` signature + prefill body + `last_idx`; call sites at ~568 Generate, ~634 Stream, ~726 Prefill)

**Interfaces:**
- Consumes: `prefill_chunks`, `PREFILL_CHUNK` (Task 1)
- Produces: `run_decode_loop(..., on_piece, cancelled: impl Fn() -> bool)` — new trailing parameter.

- [ ] **Step 1: Add the `cancelled` parameter to `run_decode_loop`**

In `src/engine_llama.rs`, extend the signature (after `mut on_piece: impl FnMut(String) -> bool,`):

```rust
    mut on_piece: impl FnMut(String) -> bool,
    cancelled: impl Fn() -> bool,
) -> Result<(
```

Update the doc comment above the fn to add: `/// \`cancelled\` is polled between prefill chunks; return \`true\` to abort a prefill whose client has gone away.`

- [ ] **Step 2: Replace the single-batch prefill with a cancellable chunked loop**

Replace the current prefill block (the `let tail = &new_tokens[effective_common..];` through the `ctx.decode(&mut batch)?; batch.clear();` at ~lines 916–931) with:

```rust
    // --- Prefill new_tokens[effective_common..] in chunks, checking for a
    // disconnected client between chunks so a timed-out request stops pinning
    // the engine mid-prefill. ---
    let tail = &new_tokens[effective_common..];
    let n_tail = tail.len();
    let mut prefilled = 0usize;
    let mut final_len = 0usize;
    for (start, len, is_final) in prefill_chunks(n_tail, PREFILL_CHUNK) {
        if cancelled() {
            tracing::info!(
                target: "localllm::llama",
                "aborted: client disconnected during prefill ({prefilled}/{n_tail} tail tokens)"
            );
            // Keep the partial prefix warm for a retry, then bail cleanly.
            *cached_tokens = new_tokens[..effective_common + prefilled].to_vec();
            return Ok((prompt_len, 0, false));
        }
        let mut batch = LlamaBatch::new(len.max(1), 1);
        for j in 0..len {
            let idx = start + j;
            let pos = (effective_common + idx) as i32;
            // Logits only on the very last token of the final chunk.
            let is_last = is_final && j == len - 1;
            batch
                .add(tail[idx], pos, &[0_i32], is_last)
                .context("batch.add (prefill) failed")?;
        }
        ctx.decode(&mut batch).context("ctx.decode (prefill) failed")?;
        batch.clear();
        prefilled += len;
        final_len = len;
    }
```

(The KV-save block that follows — `if let Some(kv_dir) = kv_cache_dir { … }` — stays unchanged; on abort we returned before it.)

- [ ] **Step 3: Fix `last_idx` to index within the final chunk**

Find `let mut last_idx = (n_tail as i32) - 1;` (~line 973) and change it to the final chunk's last position:

```rust
    let mut last_idx = (final_len as i32) - 1;
```

(Only the final chunk's last token carries logits, and it was the last `ctx.decode`, so the sample index is relative to that batch.)

- [ ] **Step 4: Wire the Generate call site (~line 568)**

The Generate arm's `run_decode_loop(...)` closure ends with `|piece| { output.push_str(&piece); true }`. Add the `cancelled` argument after it. `reply` here is the `oneshot::Sender`:

```rust
                        |piece| {
                            output.push_str(&piece);
                            true
                        },
                        || reply.is_closed(),
                    )
```

- [ ] **Step 5: Wire the Stream call site (~line 634)**

After the Stream arm's `on_piece` closure (ends `reply.blocking_send(delta).is_ok() }`), add:

```rust
                            reply.blocking_send(delta).is_ok()
                        },
                        || reply.is_closed(),
                    )
```

`reply` is the `mpsc::Sender`; both closures borrow it immutably (`blocking_send`/`is_closed` take `&self`), which is allowed.

- [ ] **Step 6: Wire the Prefill call site (~line 726)**

The background warm-up `Job::Prefill` arm calls `run_decode_loop(..., |_piece| false)`. Add a non-cancelling predicate:

```rust
                        |_piece| false,
                        || false,
                    )
```

- [ ] **Step 7: Build**

Run: `cargo build -p localllm`
Expected: clean build. Fix any borrow error at the Stream site by confirming both closures take `&reply` (they do); if the compiler complains about moving `reply`, capture by reference explicitly (`let reply_ref = &reply;` is NOT needed — closures capture by ref for `&self` methods).

- [ ] **Step 8: Run the full suite**

Run: `cargo test -p localllm`
Expected: all pass (the `prefill_chunks` unit test plus the existing engine/http tests; the chunked prefill is exercised indirectly by any test that runs a real decode — if none run without a model, correctness rests on the `prefill_chunks` test + review).

- [ ] **Step 9: Commit**

```bash
git add src/engine_llama.rs
git commit -m "feat(engine): cancellable chunked prefill; abort on client disconnect"
```

---

### Task 3: Build + verify

**Files:** none (build + manual verify)

- [ ] **Step 1: Full build + tests**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean, all pass.

- [ ] **Step 2: Rebuild bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS`.

- [ ] **Step 3: Manual — abort on disconnect**

- Force a large cold prefill locally (Config → set the cold-prefill gate very high so a big-context request stays local, and use a fresh conversation so the KV is cold).
- Send a big-context request through a client, then kill the client (or Ctrl-C) mid-prefill.
- Expected: `/tmp/localllm.log` shows `aborted: client disconnected during prefill (…/… tail tokens)` within ~one chunk, and the next request proceeds promptly instead of waiting out the full prefill.

- [ ] **Step 4: Commit any tweaks**

```bash
git add -A && git commit -m "chore: prefill-abort verification tweaks"
```

---

## Self-Review

**Spec coverage:**
- `run_decode_loop` gains `cancelled` → Task 2 Step 1. ✓
- Chunked prefill with per-chunk cancellation, partial-KV warm, early return → Task 2 Step 2 + Task 1 helper. ✓
- Logits only on final chunk's last token; `last_idx` = final_len − 1 → Task 2 Steps 2–3. ✓
- Stream/Generate pass `|| reply.is_closed()`; Prefill `|| false`; EstimateCold untouched → Task 2 Steps 4–6. ✓
- Per-token decode abort unchanged → not touched (only prefill block replaced). ✓
- No aborted-flag threading (send to dead receiver is a no-op) → Global Constraints + Task 2 Step 2. ✓
- Chunk size 2048 → Task 1. ✓

**Placeholder scan:** all code steps carry concrete code; the manual verification is a real reproduction, not a placeholder. No TBD/handle-edge-cases.

**Type consistency:** `prefill_chunks(usize, usize) -> Vec<(usize, usize, bool)>` and `PREFILL_CHUNK` (Task 1) consumed in Task 2. `run_decode_loop`'s new `cancelled: impl Fn() -> bool` is passed at all three call sites with the correct predicate. `last_idx` derived from `final_len` set in the same loop.
