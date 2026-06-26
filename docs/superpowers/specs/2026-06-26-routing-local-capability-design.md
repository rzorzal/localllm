# Capability-adjusted routing threshold (design)

Date: 2026-06-26
Status: Proposed

## Goal

Make the local-vs-cloud router account for **how capable the active local model
is**, not just a fixed per-profile threshold. A weak 3B fails tasks (e.g.
agentic tool-calling — observed: a 3B emitted a tool call as plain text) that a
14B handles locally. So the routing threshold should shift with the active
model's parameter size: weaker local → route more to cloud; stronger local →
keep more local.

This is a small enhancement to Phase B routing, unblocked by the model-picker:
`ModelManager.status().current` now exposes the active model, and the catalog
(sub-project 2) carries each model's `params_b`.

## Decisions (from brainstorming)

- **Capability source:** catalog-first, name-parse fallback. Look up the active
  `ModelSpec` in `CATALOG` → its `params_b`; if absent (custom model), parse the
  param size from the file/repo name; unknown → neutral (no adjustment).
  (Rejected: name-parse only — less precise; explicit tier field — needs catalog
  edits and misses custom models.)
- **Adjustment:** an *effective threshold* = profile threshold shifted by
  capability, 7B as the neutral reference, ±0.03 per billion params, clamped
  ±0.3, final value clamped to [0,1].
- **Tool-heavy:** no separate rule — tools already raise `difficulty_score`, and
  a weak model's lowered threshold makes tool-heavy turns escalate more readily.

## Components

### Capability source (`src/catalog.rs`, pure)

- `pub fn params_b_from_name(repo: &str, file: &str) -> Option<f32>` — scan the
  file first, then the repo, for a number immediately followed by `b`/`B`
  (optional decimal): `qwen2.5-7b-instruct-q4_k_m.gguf` → 7.0;
  `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` → 8.0 (the `3.1` has no trailing `b`,
  `8B` matches); `Phi-3.5-mini-instruct-Q4_K_M.gguf` → None.
- `pub fn active_params_b(repo: &str, file: &str) -> f32` — exact `CATALOG`
  match (repo+file) → its `params_b`; else `params_b_from_name`; else `0.0`
  (unknown → neutral).

### Decision logic (`src/route/mod.rs`)

- Add `pub local_capability_b: f32` to `Signals` (0.0 = unknown/neutral).
- `decide` replaces the raw threshold comparison with an effective threshold:
  ```rust
  let adj = if s.local_capability_b > 0.0 {
      ((s.local_capability_b - 7.0) as f64 * 0.03).clamp(-0.3, 0.3)
  } else { 0.0 };
  let effective = (p.escalation_threshold + adj).clamp(0.0, 1.0);
  if difficulty_score(s) > effective {
      return Decision::Cloud(RouteReason::Difficulty);
  }
  ```
  Reference 7B → adj 0; 3B → −0.12; 14B → +0.21; 32B → +0.30 (clamped);
  unknown → 0. The ctx-gate, no-creds, and cascade/local branches are unchanged
  (capability only shifts the difficulty cutoff).

### Wiring (`src/server.rs`)

- `route_decision` reads `let active = state.manager.status().current;` and sets
  `local_capability_b: crate::catalog::active_params_b(&active.repo, &active.file)`
  in the `Signals` it builds.

## Error handling

- Unknown/unparseable model → `0.0` → neutral (exactly today's behavior). No
  failure path; pure arithmetic.
- Capability never overrides the hard context gate or the no-creds/LocalOnly
  rules — it only moves the difficulty cutoff within [0,1].

## Testing (headless)

- `params_b_from_name`: catalog filenames (7b→7.0, 8B→8.0, decimal like
  `3.8b`→3.8), no-match (Phi mini) → None, repo-only fallback.
- `active_params_b`: a CATALOG entry → its `params_b`; a non-catalog parseable
  name → parsed; unknown → 0.0.
- `decide`: with a mid difficulty score and Balanced profile — `local_capability_b`
  3.0 routes Cloud(Difficulty) where 7.0 stays local; 14.0 stays local where a
  cloud-bound (at 7.0) request would go — i.e. the effective threshold shifts as
  specified. Boundary at the clamps.
- Update existing `decide`/`Signals` tests to set `local_capability_b: 0.0`
  (neutral) — behavior unchanged for them.

## Non-goals

- Quality/benchmark-based capability (just param size for v1).
- Per-quantization adjustment (Q4 vs Q8) — param size only.
- Changing the difficulty_score weights.

## Risks

- Param size is a coarse capability proxy (a good 7B may beat a weak 14B) →
  acceptable heuristic; the per-profile threshold + cascade still apply. The
  neutral fallback means unknown models behave exactly as today.
