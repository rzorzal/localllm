# Capability-Adjusted Routing Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shift the local-vs-cloud routing threshold by the active local model's parameter size (weak local → more cloud, strong local → more local).

**Architecture:** A pure capability resolver in `catalog` (CATALOG lookup → name-parse → neutral) feeds a new `Signals.local_capability_b`; `route::decide` computes an effective threshold (7B neutral, ±0.03/B clamped ±0.3) instead of the raw profile threshold. The handler reads the active model from `ModelManager` and sets the signal.

**Tech Stack:** Rust (pure logic; no new deps).

## Global Constraints

- Builds on the model-picker branches (`feat/router-capability`, stacked on `feat/model-manager-window`): `catalog::CATALOG` (`CatalogEntry{ repo, file, params_b, … }`), `route::{Signals, decide, difficulty_score, RoutingPolicy, Decision, RouteReason}`, `ModelManager::status().current: ModelSpec{repo,file}`, handler `route_decision`.
- Capability source: exact `CATALOG` repo+file match → `params_b`; else parse param size from file then repo (`N`/`N.N` immediately followed by `b`/`B`); else `0.0` (unknown → neutral). No regex crate — manual scan.
- Effective threshold: `adj = if cap>0 { ((cap-7.0)*0.03).clamp(-0.3,0.3) } else { 0.0 }`; `effective = (escalation_threshold + adj).clamp(0.0,1.0)`; `difficulty_score > effective → Cloud(Difficulty)`. 7B→0, 3B→−0.12, 14B→+0.21, 32B→+0.30, unknown→0.
- Capability only moves the difficulty cutoff — the ctx gate, no-creds/LocalOnly, and cascade/local branches are unchanged.
- TDD; complete code each step; commit per task; pristine build.

---

### Task 1: `catalog` capability resolver (pure)

**Files:**
- Modify: `src/catalog.rs` (add `params_b_from_name`, `active_params_b`)
- Test: `src/catalog.rs` (`#[cfg(test)]`)

**Interfaces:**
- Produces: `pub fn params_b_from_name(repo: &str, file: &str) -> Option<f32>`; `pub fn active_params_b(repo: &str, file: &str) -> f32`.

- [ ] **Step 1: Write the failing tests**

Add to `src/catalog.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn params_b_from_name_parses_size() {
        assert_eq!(super::params_b_from_name("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"), Some(7.0));
        assert_eq!(super::params_b_from_name("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), Some(8.0));
        // decimal sizes
        assert_eq!(super::params_b_from_name("x/y", "model-3.8b-q4.gguf"), Some(3.8));
        // no "<N>b" token anywhere → None
        assert_eq!(super::params_b_from_name("x/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf"), None);
        // falls back to the repo when the file lacks it
        assert_eq!(super::params_b_from_name("org/thing-13b", "weights.gguf"), Some(13.0));
    }

    #[test]
    fn active_params_b_catalog_then_name_then_neutral() {
        // a real CATALOG entry → its params_b
        assert_eq!(super::active_params_b("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"), 3.0);
        // not in catalog but parseable name
        assert_eq!(super::active_params_b("foo/bar", "model-13b.gguf"), 13.0);
        // unknown → neutral 0.0
        assert_eq!(super::active_params_b("foo/bar", "model.gguf"), 0.0);
    }
```

> Confirm the CATALOG 3B entry's `params_b` is `3.0` (it is, per sub-project 2). If a curated `params_b` differs, match the test to the actual value.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib catalog::tests::params_b 2>&1 | head -20`
Expected: FAIL — `params_b_from_name`/`active_params_b` undefined.

- [ ] **Step 3: Implement the resolver**

In `src/catalog.rs`, add (module level, near `catalog_view`):

```rust
/// Parse a parameter size in billions from a model name: the first run of
/// digits (optionally with one decimal point) immediately followed by `b`/`B`.
/// Tries `file` first, then `repo`. `qwen2.5-7b…`→7.0, `…-8B-…`→8.0,
/// `…-3.8b-…`→3.8, names without an `<N>b` token → None.
pub fn params_b_from_name(repo: &str, file: &str) -> Option<f32> {
    fn scan(s: &str) -> Option<f32> {
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_digit() {
                let start = i;
                let mut seen_dot = false;
                while i < b.len()
                    && (b[i].is_ascii_digit()
                        || (b[i] == b'.' && !seen_dot && i + 1 < b.len() && b[i + 1].is_ascii_digit()))
                {
                    if b[i] == b'.' {
                        seen_dot = true;
                    }
                    i += 1;
                }
                if i < b.len() && (b[i] == b'b' || b[i] == b'B') {
                    if let Ok(v) = s[start..i].parse::<f32>() {
                        return Some(v);
                    }
                }
            } else {
                i += 1;
            }
        }
        None
    }
    scan(file).or_else(|| scan(repo))
}

/// Capability (parameter billions) of the active model: an exact CATALOG match
/// wins; else parse the name; else `0.0` (unknown → neutral routing).
pub fn active_params_b(repo: &str, file: &str) -> f32 {
    if let Some(e) = CATALOG.iter().find(|e| e.repo == repo && e.file == file) {
        return e.params_b;
    }
    params_b_from_name(repo, file).unwrap_or(0.0)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib catalog::tests::params_b 2>&1 | tail -10 && cargo test --lib catalog::tests::active_params 2>&1 | tail -8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/catalog.rs
git commit -m "feat(catalog): active_params_b capability resolver (catalog→name-parse→neutral)"
```

---

### Task 2: `Signals.local_capability_b` + effective threshold in `decide` + wiring

**Files:**
- Modify: `src/route/mod.rs` (`Signals` field; `decide` effective threshold; update existing tests; add capability tests)
- Modify: `src/server.rs` (`route_decision` sets the signal)
- Test: `src/route/mod.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `catalog::active_params_b`, `ModelManager::status().current`.
- Produces: `Signals { …, local_capability_b: f32 }`; `decide` using the effective threshold.

- [ ] **Step 1: Add the field + update existing Signals constructions**

In `src/route/mod.rs`, add to `Signals` (after `has_cloud_creds`):

```rust
    /// Active local model's parameter size in billions (0.0 = unknown/neutral).
    /// Shifts the difficulty threshold: weaker local → more cloud.
    pub local_capability_b: f32,
```

Update the test `sig(...)` helper (around line 146) to set it neutral:

```rust
    fn sig(prompt_tokens: usize, has_creds: bool, window: usize) -> Signals {
        Signals {
            prompt_tokens,
            local_ctx_window: window,
            n_tools: 0,
            n_messages: 1,
            has_cloud_creds: has_creds,
            local_capability_b: 0.0,
        }
    }
```

Add `local_capability_b: 0.0,` to every other explicit `Signals { … }` in the route tests (the `high_difficulty…`, `max_quality…`, and `difficulty_score…` tests — the ones around lines 178, 192, 212, 216; the `more_tools` spread `..trivial` needs no change). This keeps their behavior unchanged (neutral).

- [ ] **Step 2: Add the capability decide tests**

Add to `src/route/mod.rs` tests:

```rust
    // A request scoring ~0.55: ctx 700/1000=0.7→0.42, 6 tools→0.125, 1 msg→~0.0075.
    fn mid_sig(cap: f32) -> Signals {
        Signals {
            prompt_tokens: 700, local_ctx_window: 1000, n_tools: 6,
            n_messages: 1, has_cloud_creds: true, local_capability_b: cap,
        }
    }

    #[test]
    fn weak_local_lowers_threshold_to_cloud() {
        let p = Profile::Balanced.policy(); // threshold 0.6
        // neutral (0.0) and 7B: score ~0.55 < 0.6 → stays local (cascade).
        assert_eq!(decide(&mid_sig(0.0), &p), Decision::LocalThenCascade);
        assert_eq!(decide(&mid_sig(7.0), &p), Decision::LocalThenCascade);
        // 3B: adj −0.12 → effective 0.48 → 0.55 > 0.48 → cloud.
        assert_eq!(decide(&mid_sig(3.0), &p), Decision::Cloud(RouteReason::Difficulty));
    }

    #[test]
    fn strong_local_raises_threshold_to_local() {
        let p = Profile::Balanced.policy();
        // 14B: adj +0.21 → effective 0.81 → 0.55 < 0.81 → stays local.
        assert_eq!(decide(&mid_sig(14.0), &p), Decision::LocalThenCascade);
    }

    #[test]
    fn capability_never_overrides_context_gate() {
        // Over-window even with a huge model → still ContextOverflow.
        let s = Signals { prompt_tokens: 5000, local_ctx_window: 1000, n_tools: 0,
            n_messages: 1, has_cloud_creds: true, local_capability_b: 32.0 };
        assert_eq!(decide(&s, &Profile::SaveTokens.policy()), Decision::Cloud(RouteReason::ContextOverflow));
    }
```

- [ ] **Step 3: Run to verify they fail**

Run: `cargo test --lib route:: 2>&1 | head -25`
Expected: FAIL — `Signals` missing `local_capability_b` won't compile until Step 1 is applied; the capability tests fail until Step 4 changes `decide`.

- [ ] **Step 4: Apply the effective threshold in `decide`**

In `src/route/mod.rs`, replace the difficulty-threshold block (step 3 of `decide`):

```rust
    // 3. Difficulty above the *capability-adjusted* threshold → cloud now.
    //    Weaker local model (cap < 7B) lowers the cutoff (more cloud); stronger
    //    (cap > 7B) raises it (more local); unknown (0.0) → no change.
    let adj = if s.local_capability_b > 0.0 {
        ((s.local_capability_b - 7.0) as f64 * 0.03).clamp(-0.3, 0.3)
    } else {
        0.0
    };
    let effective_threshold = (p.escalation_threshold + adj).clamp(0.0, 1.0);
    if difficulty_score(s) > effective_threshold {
        return Decision::Cloud(RouteReason::Difficulty);
    }
```

- [ ] **Step 5: Run route tests to pass**

Run: `cargo test --lib route:: 2>&1 | tail -20`
Expected: PASS — the new capability tests + all pre-existing route tests (now neutral via `local_capability_b: 0.0`).

- [ ] **Step 6: Wire the signal in `route_decision`**

In `src/server.rs`, update `route_decision` to read the active model's capability and set the signal:

```rust
    let active = state.manager.status().current;
    let local_capability_b = crate::catalog::active_params_b(&active.repo, &active.file);
    let signals = crate::route::Signals {
        prompt_tokens,
        local_ctx_window: state.local_ctx_window,
        n_tools: internal.tools.len(),
        n_messages: internal.messages.len(),
        has_cloud_creds,
        local_capability_b,
    };
```

(Keep the rest of `route_decision` as-is: it reads the policy and returns `(decide(&signals, &policy), prompt_tokens)`.)

- [ ] **Step 7: Full suite + clippy + commit**

Run: `cargo test 2>&1 | tail -10`
Expected: full suite green (route + server + integration; integration tests use `router_for_test`'s ModelManager whose current spec `{repo:"test",file:"test"}` → `active_params_b` = 0.0 neutral, so their routing is unchanged).

Run: `cargo clippy --all-targets 2>&1 | grep -E "src/route/mod.rs|src/catalog.rs|src/server.rs" | grep -- "-->" | grep -v "to_string" || echo "no new clippy"`
Expected: no new clippy in changed code.

```bash
git add src/route/mod.rs src/server.rs
git commit -m "feat(route): capability-adjusted threshold (active model param size shifts routing)"
```

---

## Acceptance

- `active_params_b` resolves capability via CATALOG → name-parse → 0.0 neutral.
- `decide` routes a mid-difficulty in-window request to cloud on a 3B but keeps it local on 7B/14B (effective threshold shifts ±0.3 clamped, 7B neutral); the context gate / no-creds / cascade branches are unchanged; unknown capability behaves exactly as before.
- `route_decision` feeds the active model's capability; integration tests (neutral 0.0) unchanged. `cargo test`/`clippy` clean.

## Self-Review

- **Spec coverage:** capability source (T1 `params_b_from_name`/`active_params_b`), effective-threshold logic (T2 `decide`), `Signals` field + wiring (T2), neutral fallback + gate-precedence (T2 tests). All spec requirements mapped.
- **Placeholder scan:** none — complete code + commands; the `mid_sig` score arithmetic is shown so the thresholds are verifiable.
- **Type consistency:** `params_b_from_name(&str,&str)->Option<f32>`, `active_params_b(&str,&str)->f32`, `Signals.local_capability_b: f32`, the `decide` effective-threshold block, and `route_decision`'s `active_params_b` call are consistent. Existing route tests updated to the new `Signals` shape (neutral 0.0).
