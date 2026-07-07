# Fase D2 — Per-Model Effective Capability (Design Spec)

**Date:** 2026-07-07
**Branch base:** `feat/fase-d1` (D2 consumes D1's feedback signals)
**Status:** Approved for planning

## Goal

Show, per local model, an *estimated effective capability* inferred from D1's
routing-quality signals — e.g. "your Qwen 3B behaves like ~1.6B under this
workload." Informational only: nothing is applied to routing, no manual knob.

## Context

Fase D decomposition (user-approved D1→D4). D1 landed four per-request quality
signals and a global threshold suggestion. D2 is scoped down to a **read-only
dashboard insight**: the user explicitly chose "suggest/informative only, no
knob." It reuses D1's `LogLine::Feedback` data.

Routing background: `effective_threshold = escalation_threshold +
capability_adjustment(cap_b)`, where `capability_adjustment(cap_b) =
((cap_b - 7.0) * 0.03).clamp(-0.3, 0.3)` (src/route/mod.rs). Bigger nominal
model size → higher threshold → more local. D2 inverts this slope to translate
an observed flag-rate into an effective-capability estimate.

## Non-Goals

- No routing change. `capability_adjustment` and `effective_threshold` are
  untouched; the estimate never feeds the router.
- No per-model capability override / settings knob.
- No auto-calibration. (The phase was originally "auto-calibrate capability_b";
  the user narrowed it to informational.)

## 1. Model attribution (data)

`RouteEntry` gains:

```rust
/// Compact key (model_ctx_key(repo,file)) of the LOCAL model active when this
/// decision was made. Set on every decision; None on legacy lines.
#[serde(default, skip_serializing_if = "Option::is_none")]
pub local_model: Option<String>,
```

Set in `route_decision` (server.rs) from the already-in-scope
`active = state.manager.status().current`:
`local_model: Some(crate::settings::model_ctx_key(&active.repo, &active.file))`.

This is distinct from the existing `model` field (the CLOUD model the client
asked for). Legacy lines without `local_model` group under "unknown" and are
excluded from D2 (below the sample floor).

## 2. Effective-capability estimate (the math)

Computed in `build_dashboard` per local model over the **30-day** retained log,
only for models with at least `CAP_MIN_SAMPLE = 30` local decisions:

- `nominal_b = params_b_for_key(model_key)` — nominal size in billions. The
  compact key is `format!("{repo}/{file}")` and `repo` itself contains `/`, so
  it is NOT reliably splittable back to (repo, file). A new
  `catalog::params_b_for_key(key: &str) -> f32` resolves it without splitting:
  (1) return `params_b` of the CATALOG entry whose `format!("{}/{}", repo, file)`
  equals `key`; (2) else scan the whole key string for the `NNb` size token
  (the same scan `params_b_from_name` uses); (3) else `0.0` (unknown → skip).
- `flagged_rate = flagged_locals / local_total` for that model, where a local
  is flagged if it has ≥1 of cascade/truncated/reask (reuse D1's `NEG_LOCAL`
  set and `.any()` counting — a decision counts once).
- Invert the `capability_adjustment` slope (0.03 threshold per 1 B):

```rust
const CAP_BASELINE_FLAG_RATE: f64 = 0.10; // "normal" flag rate
const CAP_SCALE: f64 = 0.5;               // how strongly excess flags cut capability
const CAP_SLOPE: f64 = 0.03;              // must match route::capability_adjustment slope

let excess = flagged_rate - CAP_BASELINE_FLAG_RATE;      // may be negative
let delta_b = -(excess * CAP_SCALE) / CAP_SLOPE;         // flags↑ ⇒ capability↓
let effective_b = (nominal_b + delta_b).clamp(0.5, nominal_b * 1.5);
```

High flag rate → `effective_b < nominal_b`. Low flag rate → slightly above.
Clamp `[0.5, nominal_b * 1.5]` bounds the estimate to sane values. If
`nominal_b == 0` (unknown model), skip the model (cannot estimate). All
tunables are named consts; they are documented starting guesses.

Purely a computed read. Does not touch routing.

## 3. Aggregation + UI

`build_dashboard` produces:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelCapability {
    pub model: String,        // compact key
    pub local_total: u64,
    pub flagged_rate: f64,
    pub nominal_b: f64,
    pub effective_b: f64,
}
```

`Dashboard` gains `model_capabilities: Vec<ModelCapability>` (models below the
sample floor or with unknown nominal size omitted; sorted by `local_total`
desc). Dashboard JSON serialises it.

UI (`app.js` dashboard, after the D1 accuracy card): a card/table "Capacidade
efetiva por modelo", one row per model:
`{model} · {local_total} locais · {flagged_rate*100:.0}% flag · nominal
{nominal_b}b → efetivo {effective_b:.1}b`, with an arrow/indicator for
below/above nominal. Empty vec → card hidden. Values rendered via
textContent (backend-controlled model keys — never innerHTML).

## Files Touched

- Modify: `src/route_log.rs` (RouteEntry.local_model; ModelCapability;
  aggregation + consts in build_dashboard; tests)
- Modify: `src/server.rs` (route_decision sets local_model; JSON passthrough is
  automatic via the Dashboard struct)
- Modify: `src/manager_ui/app.js` (capability card)
- Modify: `src/manager_ui/style.css` (card styling)

## Testing

- **route_log unit:** RouteEntry with/without `local_model` round-trips
  (legacy retrocompat); `ModelCapability` aggregation groups by local_model,
  computes flagged_rate, applies the effective_b formula with fixed inputs
  (assert an exact expected value), clamps at both bounds, omits
  below-sample-floor models and unknown-nominal models.
- **server wiring:** `route_decision` writes `local_model` equal to the active
  model's `model_ctx_key` (minimal-AppState test).
- **UI:** no harness; `cargo build` gate (assets string-embedded).

## Open Risks / Notes for later

- CAP_BASELINE_FLAG_RATE / CAP_SCALE are starting guesses; a later phase could
  calibrate them. The estimate is explicitly labeled an estimate in the UI.
- Budget/breaker forced-local decisions (Fase A/C) log `dest=local` with their
  own score and will be attributed to the active model; they inflate that
  model's local_total and, if flagged, its flag rate. Acceptable noise for an
  informational view.
