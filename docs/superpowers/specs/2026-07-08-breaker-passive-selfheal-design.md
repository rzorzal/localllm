# Circuit-breaker passive self-heal

**Date:** 2026-07-08
**Status:** approved (design)
**Component:** cloud circuit breaker (`src/breaker.rs`, `src/server.rs`, `src/manager_ui/app.js`)

## Problem

The cloud circuit breaker advances `Open → HalfOpen` only inside `gate()`, and
`gate()` is consulted only on cloud-bound requests (`server.rs`
`breaker_gate_block`). `snapshot()` reports `state:"open",
next_probe_secs = until.saturating_sub(now)` once the cooldown has elapsed, but
nothing transitions the state without a cloud request. When recent traffic all
routes local (or the user is idle), the breaker sits in `Open` with
`next_probe_secs == 0` indefinitely, and the dashboard strip wedges at
`Cloud indisponível (Quota) · nova tentativa em 0s` for hours. There is no
background reaper (breaker.rs comment: "no reaper"); only a cloud request or the
manual "↻ Tentar cloud agora" button advances it.

Reported 2026-07-08 (banner stuck for hours, Quota reason).

## Goal

The breaker's elapsed-cooldown state must become visible/actionable without
requiring cloud traffic, while preserving the existing backoff escalation so a
genuinely-down cloud is not hammered.

## Non-goals

- No active/synthetic cloud probe on a timer — that would burn quota, which is
  exactly wrong when the trip reason is Quota.
- No new background tokio task — the dashboard already polls the breaker
  endpoint; that poll drives recovery.
- Not changing the trip/backoff/reset semantics (`on_failure`, `on_success`,
  `reset`, `gate`) — only adding the missing on-read transition.
- Not auto-clearing a genuinely-exhausted quota; it correctly shows degraded
  until a real request succeeds.

## Mechanism

Move the *elapsed-cooldown* `Open → HalfOpen` transition so it also happens on
read, not only inside `gate()`.

1. **New `CircuitBreaker::poll(&self, now: u64) -> BreakerSnapshot`.** If the
   state is `Open` and `now >= until`, transition to
   `HalfOpen { backoff, reason }` (carrying `backoff` and `reason` exactly as
   `gate()` does today), then return the snapshot of the resulting state. If
   `Open` and `now < until`, no transition — return the `open` snapshot with
   `next_probe_secs = until - now`. If `Closed` or `HalfOpen`, no-op — return
   the snapshot. Pure state machine; `now` injected for deterministic tests.

2. **`/admin/breaker` GET calls `poll(now)`** instead of `snapshot(now)`
   (`server.rs` `handle_breaker_get`). The dashboard's existing periodic poll of
   this endpoint therefore drives the recovery transition. No new task.

3. **`gate()`, `on_failure`, `on_success`, `reset` unchanged.** `gate()` already
   performs the same `Open → HalfOpen` transition for live cloud requests.
   Because `poll` carries `backoff` into `HalfOpen`, a subsequently-failed probe
   still doubles toward the 300s cap via `on_failure`'s `HalfOpen` arm — backoff
   escalation is fully preserved.

## State transitions (after this change)

| From | Condition | To | Notes |
|------|-----------|----|-------|
| Open | `now >= until` (via `gate` **or** `poll`) | HalfOpen | carries `backoff`, `reason` |
| Open | `now < until` | Open | `next_probe_secs = until - now` |
| HalfOpen | any read | HalfOpen | no-op |
| Closed | any read | Closed | no-op |
| HalfOpen | `on_failure` | Open | `backoff*2` capped 300s |
| HalfOpen/Open | `on_success` | Closed | signals recovery |

`snapshot()` remains a pure read (retained for the breaker's own unit tests and
any future non-mutating view); `poll()` is the mutating variant the dashboard
endpoint uses. The dashboard `/admin/breaker` GET is the only current
`snapshot()` caller; the tray only calls `reset()`, so no other call site
changes.

## Frontend (`app.js` `paintBreaker`)

The endpoint now returns `half-open` once an idle breaker's cooldown elapses,
instead of a stuck `open / 0s`. `paintBreaker` already handles `half-open`
(renders "Testando cloud (probe)…"). Refinements:

- Copy for `half-open`: **"Cloud pronto — testa na próxima chamada"** (reads as
  *recovering*, not *down*).
- Dot color: amber for `half-open` (distinct from red `open` and green
  `closed`). Use the existing `brk-<state>` class hook (`brk-half-open`) with a
  CSS amber rule.
- The "↻ Tentar cloud agora" button stays visible for any non-closed state
  (unchanged) — it force-closes via `/admin/breaker/reset`.

User-visible flow: cooldown counts `Ns → 0`; at 0 the strip flips amber "Cloud
pronto — testa na próxima chamada"; the next cloud-routed request probes;
success → green "Cloud disponível", failure → re-trips with doubled cooldown.

## Error handling

- `poll` is infallible (a `Mutex` lock recovered via `into_inner()` on poison,
  matching the existing methods). No panics.
- No network I/O introduced.

## Testing

Pure unit tests in `breaker.rs` (deterministic `now`):

- `poll` on `Open` with `now >= until` → snapshot `half-open`; a following
  `gate(now)` returns `Allow` (probe) — confirms real transition, not just a
  reported label.
- `poll` on `Open` with `now < until` → snapshot `open`, `next_probe_secs`
  correct; state still `Open` (a following `gate` within cooldown still
  `Block`s).
- `poll` on `Closed` and on `HalfOpen` → no-op snapshot, state unchanged.
- Backoff preserved: trip → advance past cooldown → `poll` (half-open) →
  `on_failure` → `next_probe_secs` doubled (not reset to base).

Existing breaker tests remain unchanged and must still pass.

## Rollout

Backend + one JS/CSS touch. Rebuild the `.app` bundle + restart. Manual check:
force a trip (or hit the real Quota), open the dashboard, watch the strip flip
from red "indisponível Ns" through "0s" to amber "Cloud pronto" at cooldown,
without sending any request.
