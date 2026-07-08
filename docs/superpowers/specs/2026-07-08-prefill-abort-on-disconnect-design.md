# Prefill abort on client disconnect (sub-2)

**Date:** 2026-07-08
**Status:** approved (design)
**Parent:** routing/timeout fix. sub-1 (cold-prefill routing) done, sub-3 (launcher) done, model-load resilience done. sub-2 = this (the last piece).

## Problem

When a client (Claude Code) times out on a slow cold prefill and retries, the abandoned request keeps the single local engine busy. The streaming decode loop already aborts on disconnect (the callback returns `reply.blocking_send(delta).is_ok()`, which is false once the SSE receiver is dropped — `engine_llama.rs:648`, checked per generated token at `:990`). But that check only runs during DECODE. The expensive part — the PREFILL of the accumulated context (observed ~19k tokens / 185 s) — is a single `ctx.decode(&mut batch)` over the whole prompt tail (`engine_llama.rs:920–930`), an uninterruptible blocking call. So a client that disconnects mid-prefill is not noticed until prefill finishes and the first token is sampled, and the engine stays pinned for the full prefill on a response nobody will read (compounding when the retry queues behind it).

## Goal

Abort the local prefill promptly when the client has disconnected, freeing the single engine for the next request.

## Non-goals

- Cancelling an in-flight request from the UI (only client-disconnect drives cancellation).
- Cancelling cloud-forwarded requests (the reverse-proxy already drops when the client goes away).
- Cancelling the background warm-up `Prefill` job or `EstimateCold` (fast / best-effort; no client to abandon).

## Mechanism

Split the prefill into chunks and check a cancellation predicate between them.

1. `run_decode_loop` (`engine_llama.rs`) gains a parameter `cancelled: impl Fn() -> bool`.
2. The tail prefill (currently one `LlamaBatch` of `n_tail` tokens + one `ctx.decode`) becomes a loop over fixed-size chunks (`PREFILL_CHUNK = 2048`, matching the context's `n_ubatch`). For each chunk: build a batch of that slice (logits requested only on the final token of the FINAL chunk), `ctx.decode`, clear. Between chunks, if `cancelled()` returns true → abort: set `cached_tokens` to the prefix actually prefilled so far (`new_tokens[..effective_common + tokens_prefilled]`) so the KV stays valid and partially warm for a retry, and return an "aborted" outcome that produces no client response.
3. Callers wire the predicate:
   - `Job::Stream` → `cancelled = || reply.is_closed()` (`tokio::sync::mpsc::Sender::is_closed`).
   - `Job::Generate` → `cancelled = || reply.is_closed()` (`tokio::sync::oneshot::Sender::is_closed`).
   - `Job::Prefill` (background warm-up) and `Job::EstimateCold` → `|| false` (no cancellation).
4. The existing per-token decode abort via `on_piece` (channel closed) is unchanged.

## Chunking correctness

Prefilling `[t0..tk]` as several sequential batches with correct absolute positions is numerically equivalent to one batch — llama.cpp already sub-batches internally at `n_ubatch`. Logits must be requested only on the very last token of the whole tail (as today), i.e. only the final chunk's last token sets `logits=true`; earlier chunks pass `false` for all. `last_idx` for the first sample is the index of that last token within the final chunk's batch.

## Abort outcome

On prefill abort, no `ChatResult`/`StreamDelta` is produced (the receiver is gone). The worker returns a distinct result the job arm recognizes to skip sending and to log `"{rid} aborted: client disconnected during prefill"`. The worker loop continues to the next job; the KV cache holds the partial prefix (valid, reusable). No `record_outcome` (there is no served response).

## Error handling

- Cancellation is best-effort and checked only at chunk boundaries — worst-case latency to notice is one chunk's decode (~2048 tokens), far better than the whole prompt.
- A `cancelled()` predicate must not panic (both `is_closed()` calls are infallible bool reads).
- Existing `catch_unwind`/`recover_context` around decode is unchanged.

## Testing

- Pure helper `prefill_chunks(n_tail, chunk) -> Vec<(start, len, is_final)>` (or equivalent index math) unit-tested: a tail smaller than one chunk → single chunk with `is_final`; a tail spanning 2.5 chunks → 3 chunks, only the last `is_final`; `n_tail == 0` → empty.
- Chunk-equivalence and real cancellation need a loaded model → integration/manual (mark `#[ignore]` if a CI model isn't available), consistent with the existing engine tests. Manual: start a large cold-prefill request through a client, kill the client mid-prefill, confirm the engine frees promptly (next request proceeds) and the log shows the abort line.

## Rollout

Backend only. Rebuild the `.app` bundle + restart. Manual verification as above.
