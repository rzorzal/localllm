# Sub-1: Cold-prefill-aware routing + warm/cold logging

**Date:** 2026-07-08
**Status:** approved (design)
**Parent:** routing/timeout fix, decomposed into sub-1 (this), sub-2 (never-error fallback + abort), sub-3 (tray launcher + terminal picker).

## Problem

Claude Code / Codex route through localllm. The first request of a new conversation carries a large accumulated context (observed: `prompt_tok≈24768`, `fill=0.76`, 32 tools). The routing difficulty score deliberately ignores prompt size and tool count (they are the client's fixed overhead, not task difficulty — see `route/mod.rs:69-74`), so the request scores low and stays local. The local 8B model must then prefill the context from a COLD KV cache:

```
req-910e9180  prefix reuse 16/19392 cached → decoding 19376 new  → 185.4s  (COLD)
req-f8a7eff7  prefix reuse 19135/19502 cached → decoding 367 new → fast     (WARM)
```

The cold prefill (~19k new tokens) takes minutes; Claude Code hits its request timeout and shows `API error · Retrying`. Subsequent turns reuse the KV cache (only a few hundred new tokens) and are fast. **The bottleneck is cold prefill of large context, not task difficulty.** The existing `ctx_gate_frac` gate only catches near-overflow (prompt doesn't fit), not prefill latency.

## Goals

- Detect, at decision time, when a request would require an expensive COLD prefill on the local model.
- When the estimated cold-prefill time exceeds a configurable threshold AND cloud is available: serve the response from cloud (fast) AND kick off a background local prefill so the next turn is warm and routes local.
- When warm (small cold_tokens) or when the estimated prefill is under the threshold: stay local (respond in its own time).
- Log warm/cold state + cold-token count + estimated prefill time so the dashboard shows why a request went where it did.

## Non-goals

- Client-side request-timeout raising and the tray launcher — sub-3.
- Aborting local generation on client disconnect / never-error fallback hardening — sub-2.
- Changing the existing difficulty score, capability adjustment, or `ctx_gate_frac` overflow gate — all unchanged; this adds a new, independent decision branch.

## Mechanism

1. **Engine exposes cached-prefix state.** `engine_llama.rs` already tracks `cached_tokens: Vec<LlamaToken>` (the KV-resident prefix). After each generation it publishes a small shared snapshot the router can read cheaply without touching the engine thread: `PrefixState { len: usize, hash: u64 }`, where `hash` is a rolling hash of the cached token ids (or of the cached prompt text prefix). Stored behind an `Arc<Mutex<PrefixState>>` (or `arc-swap`) owned by the manager, written by the engine loop, read by the router.

2. **Router estimates cold_tokens.** In `decide` (`route/mod.rs`), before the existing branches, compute how many of the incoming prompt's leading tokens are already cached:
   - Compare the incoming prompt against `PrefixState`: if the incoming prefix hash over the cached span matches, the request is WARM and `cold_tokens = prompt_tokens.saturating_sub(cached_len)`; otherwise COLD and `cold_tokens = prompt_tokens`.
   - `Signals` gains the inputs it needs: `cached_prefix_len: usize` and `cached_prefix_hash: u64` (populated by the server from `PrefixState` at request time), plus a computed `incoming_prefix_hash` over the same span.

3. **Estimate prefill time.** `prefill_secs = cold_tokens as f64 / prefill_tok_s`, where `prefill_tok_s` comes from the measured local latency (dashboard already records local `avg_tok_s` / TTFT); use a conservative constant fallback when no sample exists yet.

4. **New decision branch.** After the hard context gate and cloud-possible check, before the difficulty branch:
   - If `prefill_secs > policy.cold_prefill_gate_secs` AND cloud is allowed and available → `Decision::Cloud(RouteReason::ColdPrefill)`, and signal the caller to fire a background local prefill.
   - Else fall through to the existing difficulty/cascade/local branches.

5. **Background local prefill.** When the server routes a request as `ColdPrefill`, after forwarding to cloud it fires a fire-and-forget local **prefill-only** op: the engine decodes the prompt tokens into the KV cache WITHOUT sampling a completion, so `cached_tokens` becomes the warm prefix for the next turn. New engine op `prefill(prompt)`; best-effort (a failure never affects the served response). Serialized with real local requests — the local context is single-threaded, so a concurrent local request queues behind the background prefill (FIFO on the existing engine channel; no preemption).

## Configuration

New knob `cold_prefill_gate_secs` on the routing policy, default generous (patient-with-local; e.g. 360s — only truly monstrous cold prefills escalate). Exposed on the Config page like the existing Balanced threshold: `GET/POST /admin/cold-prefill-gate` returning `{ secs }`. Note in the UI copy that the client request timeout (sub-3) must exceed this so under-threshold local prefills don't error.

## Logging + dashboard

`RouteEntry` (route-log JSONL) gains, populated at the decision point:
- `cold_tokens: Option<u64>` — uncached tokens needing prefill.
- `prefill_secs_est: Option<f64>` — estimated cold-prefill seconds.
- `was_cold: Option<bool>` — cache-miss at decision time.
- `bg_prefill_fired: Option<bool>` — a background local prefill was kicked off.

All optional with `#[serde(default, skip_serializing_if = "Option::is_none")]` (legacy lines parse). The dashboard shows warm/cold + estimated prefill cost in the score popover (`scoreExplainNode` in `app.js`) and, when `reason == "ColdPrefill"`, explains the escalation and the background warm-up.

## Error handling

- `decide` stays a pure function of `Signals` + policy — fully unit-testable.
- Missing latency sample → constant fallback `prefill_tok_s`; never divide by zero.
- Background prefill is best-effort: any failure is logged and dropped, never touching the client's (cloud-served) response.
- `PrefixState` read is lock-guarded and cheap; a poisoned lock falls back to "assume cold" (safe: at worst escalates).

## Testing

- `route/mod.rs` unit tests: warm request (low cold_tokens) stays local; cold big request with cloud available → `Cloud(ColdPrefill)` + bg-prefill flag; cold big request with cloud unavailable → local; `prefill_secs` under gate → local; fallback `prefill_tok_s` when no sample; threshold boundary.
- `route_log.rs`: `RouteEntry` round-trips the four new fields; dashboard surfaces them.
- Engine: `prefill(prompt)` populates `cached_tokens` and leaves the model ready so a following generation reuses the full prefix (few new tokens decoded).
- Frontend: `node --check`; manual — a cold first request shows ColdPrefill + warm-up in the popover.

## Rollout

Backend + embedded frontend; rebuild the `.app` bundle (`scripts/build-app.sh --fast`) and restart to verify. Bundle rebuild is deferred to the combined build after sub-1 (per user: "depois geramos um build").
