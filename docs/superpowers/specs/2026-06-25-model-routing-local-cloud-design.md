# localllm → local/cloud model router (load balancer) — design

Date: 2026-06-25
Status: Proposed

## Goal

Make localllm a **model load balancer**: serve a request on the **local** model
when the local model can handle it, and forward to a **cloud provider** model
(Anthropic / OpenAI) when the task is too big (context overflow) or too hard for
local quality. Primary objective: **burn fewer cloud tokens** without losing
answer quality on the tasks that need it.

The user picks the trade-off from a **menu-bar (tray) heuristic selector**:
spend fewer tokens, maximum reasoning power, a balance, or local-only.

## Research basis (one line each)

- **RouteLLM** (arXiv 2406.18665, ICLR'25): learn a router that predicts when a
  strong model is needed; >2× cost cut at equal quality. The escalation
  *threshold* is the user-facing knob.
- **Hybrid LLM** (arXiv 2404.14618, ICLR'24): quality-aware router with a
  **test-time tunable threshold** → 40% fewer big-model calls. Direct basis for
  the tray profiles.
- **FrugalGPT** (arXiv 2305.05176): LLM **cascade** — run cheap first, escalate
  only if the answer is not good enough; up to 98% cost cut.
- **Confident or Seek Stronger** (arXiv 2502.04428): uncertainty-based on-device
  (local↔cloud) routing — selective offload generalizes but needs a careful
  confidence threshold and a capable edge model.
- **Doing More with Less** (arXiv 2502.00409) & **Dynamic Model Routing and
  Cascading: A Survey** (arXiv 2603.04445): taxonomy of router (pre-generation)
  vs cascade (post-generation) strategies and their cost/quality trade-offs.

We adopt a **hybrid**: deterministic gate + pre-generation score + cascade
fallback. Local inference burns **zero cloud tokens**, so trying local first is
cheap by construction.

## Distribution constraint (unchanged)

Still a **single Rust binary**, plug-and-play, no external processes. This rules
out an external proxy sidecar; the router lives in-process.

## Credentials & privacy

The cloud target uses the **client's own credentials**, reused verbatim. When
Claude Code / Codex call localllm they already send `x-api-key` +
`anthropic-version` (Anthropic) or `Authorization: Bearer` (OpenAI). On a cloud
route, localllm forwards those headers unchanged to the real upstream. No
separate key configuration.

Provider is inferred from the endpoint that was hit:
`/v1/messages` → Anthropic, `/v1/chat/completions` → OpenAI.

Privacy note: the cloud path is exactly what the client would have done on its
own. Intercepting and serving some requests locally is therefore **strictly more
private** than the baseline, never less.

## Architecture

```
Claude Code / Codex  ──(Anthropic or OpenAI API + its own key)──▶  localllm handler
                                                                        │
                                          ┌──── route::decide(signals, policy) ────┐
                                          │   (pure fn: signals in, Decision out)   │
                                          ▼                                         ▼
                                   Decision::Local                          Decision::Cloud(reason)
                                          │                                         │
                                   existing Engine path              cloud::forward(raw body+headers)
                                          │                              reverse-proxy to upstream
                                          ▼                                         ▼
                                   local GGUF (Qwen-3B)              api.anthropic.com / api.openai.com
                                          │                                         │
                                   (cascade: weak? ─yes─▶ cloud::forward)           │
                                          └─────────────── SSE / JSON back to client ┘
```

Decision lives at the **HTTP layer** (chosen over a `Generator`-trait wrapper) so
the cloud path can be a **byte-faithful reverse-proxy**: forward the original raw
request and relay the raw response, preserving tool-calls and streaming exactly
rather than reconstructing them from internal types. The decision *logic* is
extracted into a pure, unit-testable function so it is not coupled to HTTP I/O.

### New modules

- `src/route/mod.rs` — `Decision`, `RouteReason`, `Signals`, pure `decide()`. No
  I/O. Primary test target.
- `src/route/policy.rs` — `Profile` enum and its `RoutingPolicy` knobs; shared as
  `Arc<RwLock<RoutingPolicy>>`; settings load/save.
- `src/cloud.rs` — reverse-proxy forwarder (uses existing `reqwest`).
- `src/usage.rs` — atomic session usage counters + alert thresholds.
- Extended: `src/server.rs` (handlers), `src/tray.rs` (selector), `src/config.rs`
  (`--profile`), `src/main.rs` (wire shared policy into `AppState` and tray).

`RoutingPolicy` is shared mutable state: tray writes it, handlers read it, behind
`Arc<RwLock<RoutingPolicy>>` cloned into `AppState`.

## Decision logic (`route::decide`)

Pure function `decide(signals: &Signals, policy: &RoutingPolicy) -> Decision`.

```
Signals {
    prompt_tokens: usize,      // estimated from the request
    local_ctx_window: usize,   // config ctx_len
    n_tools: usize,
    n_messages: usize,
    has_cloud_creds: bool,     // incoming key present?
}

Decision {
    Local,              // run local, no cascade
    LocalThenCascade,   // run local, escalate if weak (buffered path only)
    Cloud(RouteReason), // skip local, reverse-proxy now
    LocalNoCreds,       // wanted cloud but no key → local + warn
}

RouteReason { ContextOverflow, Difficulty, Profile }
```

Ordered gates:

1. **Hard context gate** — if `prompt_tokens > local_ctx_window *
   policy.ctx_gate_frac` → cloud. If `!has_cloud_creds` or profile is
   `LocalOnly` → `LocalNoCreds` (local then rejects an over-window prompt with
   the existing clean error, commit `1f8c134` — never silently truncate).
2. **Profile shortcut** — `LocalOnly` → always `Local`. `MaxQuality` →
   `Cloud(Profile)` for anything above a small trivial floor.
3. **Difficulty score** — cheap weighted, normalized 0..1 sum of signals
   (prompt_tokens, n_tools, n_messages). `score > policy.escalation_threshold`
   → `Cloud(Difficulty)`.
4. **Else** — `LocalThenCascade` if `policy.cascade` else `Local`.

The difficulty score is a **transparent documented heuristic** for v1. A learned
classifier (RouteLLM / DistilBERT-style) is explicit **future work**; the
`Signals`/`decide` interface leaves room but we do not build it now (YAGNI).

### Profiles → knobs

| Profile     | escalation_threshold | cascade | ctx_gate_frac | allow_cloud |
|-------------|----------------------|---------|---------------|-------------|
| SaveTokens  | 0.9 (high)           | yes     | 0.95          | yes         |
| Balanced    | 0.6                  | yes     | 0.9           | yes         |
| MaxQuality  | 0.2 (low)            | no      | 0.75          | yes         |
| LocalOnly   | —                    | no      | 1.0           | no          |

Default profile: **SaveTokens** (matches the primary goal).

### Cascade weakness signal

Post-local, **buffered/non-stream paths only**: escalate to cloud if
`finish_reason == Length` (truncated) **or** local generation errored.
Logprob/judge-based confidence is future work — start minimal and robust.
Incremental streaming cannot escalate mid-flight (tokens already sent), so there
`LocalThenCascade` degrades to `Local`.

## Request flow

Both endpoints (`/v1/messages`, `/v1/chat/completions`):

1. Extract `HeaderMap` + raw `Bytes` (replacing the `Json<T>` extractor), then
   `serde_json::from_slice` into the existing request struct. Keep raw bytes for
   forwarding. Local path is byte-for-byte unchanged in behavior.
2. Build `Signals` (estimate prompt tokens, count tools/messages, detect
   incoming credential header).
3. `decision = route::decide(&signals, &*policy.read())`.
4. Branch:
   - **Local / LocalNoCreds** → existing engine path. `LocalNoCreds` also logs
     and fires a tray warning.
   - **Cloud(reason)** → `cloud::forward(provider, raw_bytes, headers)`; relay
     upstream response to the client; record usage.
   - **LocalThenCascade** → buffered/non-stream: run local `generate()` fully,
     check weakness; weak → `cloud::forward`; else replay local as today.
     Incremental stream: treat as `Local`.

`cloud::forward` (`src/cloud.rs`):
- Provider → base URL (`https://api.anthropic.com/v1/messages`,
  `https://api.openai.com/v1/chat/completions`).
- Copy the auth headers the client sent (`x-api-key` + `anthropic-version`, or
  `Authorization`); send the raw body unchanged so upstream sees the original
  request verbatim.
- Relay the response body back without re-parsing (SSE or JSON bytes).
- Best-effort: parse `usage` from the final chunk/JSON to feed counters.

Logging: extend the `localllm::req` lines with `route=local|cloud reason=…
profile=…` so every decision is visible in `/tmp/localllm.log`.

## Tray selector, persistence, usage alerts

**Tray submenu** ("Routing") of 4 radio-style `CheckMenuItem`s — Save tokens
(default), Balanced, Max quality, Local only. Click writes the `Profile` into the
shared `Arc<RwLock<RoutingPolicy>>`; the next request uses it live (no restart).
Status line shows `Routing: <profile>`.

**Persistence.** Selected profile saved to `<config-dir>/localllm/settings.json`
(via `dirs`), loaded at startup. CLI `--profile
<save-tokens|balanced|max-quality|local-only>` overrides for that run. This is a
small dedicated settings file separate from the CLI `Config`.

**Usage tracking** (`src/usage.rs`): atomic per-session counters (cloud calls,
cloud prompt + completion tokens). Two alert behaviors, both one-shot to avoid
spam:
- **High usage** — session cloud tokens cross a configurable threshold → macOS
  notification suggesting a switch to Save tokens.
- **Degrade** — see error handling below.

## Error handling

`cloud::forward` has **three degrade triggers**, all → fall back to local when
the prompt fits the local window (else return a clean error to the client), each
with a one-shot notification:

1. **Auth / quota** — upstream `401 / 403 / 429`. Notify "Cloud quota/auth failed
   — now serving locally."
2. **Offline / network error** — connect / DNS / timeout. Notify "No internet —
   serving locally."
3. **Upstream 5xx.** Notify "Cloud error — serving locally."

Other cases:
- `LocalOnly` + context overflow → existing clean "prompt larger than context"
  error (`1f8c134`); never silently truncate.
- Cascade only on buffered/non-stream paths; incremental streaming never
  escalates mid-flight.
- Degrade notifications are one-shot and reset when a cloud call next succeeds.
- The router never hangs: any cloud failure resolves to local or a clean error.

## Phased milestones (each independently shippable)

### Phase A — Routing core + raw reverse-proxy (ctx-size gate only)
Pure `route::decide` with the hard context gate; `RoutingPolicy` plumbed through
`AppState`; handlers switched to raw-bytes extraction; `cloud::forward` reverse-
proxy reusing incoming creds. Profiles exist but only the ctx gate is active.
- Acceptance: an over-window request is reverse-proxied to upstream and the
  response matches the provider's native shape (tools + streaming intact); all
  other requests still served locally; existing tests pass.

### Phase B — Difficulty score + cascade fallback
Pre-generation heuristic score with documented weights; cascade on the
buffered/non-stream path (escalate on `Length`/error).
- Acceptance: a high-score request routes cloud upfront; a truncated local result
  on a cascade profile escalates and returns the cloud answer; `decide` unit
  tests cover every gate and profile boundary.

### Phase C — Tray profile selector + persistence
"Routing" submenu (4 profiles), live policy update, persisted settings,
`--profile` CLI override, status-line summary.
- Acceptance: switching the profile in the tray changes routing on the next
  request without restart; the choice survives a restart; `--profile` overrides.

### Phase D — Usage tracking + alerts + graceful degrade
Session counters; high-usage notification; the three degrade triggers
(auth/quota, offline, 5xx) → local fallback + one-shot alert.
- Acceptance: with no network, requests fall back to local and the user is
  notified; an upstream 429 degrades to local; crossing the usage threshold fires
  exactly one notification.

## Key decisions / open questions

- **Difficulty weights** — start with hand-set weights; tune against real
  Claude-Code/Codex traffic in `/tmp/localllm.log`. Learned router deferred.
- **Prompt-token estimation** — exact local tokenization vs a cheap `chars/4`
  heuristic. Start cheap; upgrade to exact local tokenizer if the ctx gate proves
  too coarse near the boundary.
- **High-usage threshold default** — pick a sane default (e.g. session cloud
  tokens); make it configurable. Confirm value at Phase D.

## Non-goals

- A learned/trained router model (heuristic only for now).
- Provider model *selection/upgrade* (e.g. forcing opus vs sonnet): forward the
  model the client requested; smart model picking is future work.
- Routing across more than the two existing API surfaces.
- Per-user billing/quota accounting beyond best-effort session counters.

## Risks

- Heuristic mis-routes (easy task → cloud, or hard task → weak local answer).
  Mitigated by cascade on buffered paths and the tunable per-profile threshold;
  logs expose every decision for tuning.
- Raw reverse-proxy must not corrupt streaming framing → relay bytes unmodified;
  integration-test against a mock upstream.
- Token estimation error near the ctx boundary → conservative `ctx_gate_frac`
  and the existing over-window error as a backstop.
