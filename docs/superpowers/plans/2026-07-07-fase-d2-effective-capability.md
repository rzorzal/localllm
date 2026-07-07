# Fase D2 — Per-Model Effective Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show, per local model, an estimated effective capability inferred from D1's flag signals (e.g. "3B behaving like ~1.6B") as a read-only dashboard card — no routing change, no knob.

**Architecture:** A `params_b_for_key` catalog helper resolves nominal size from the compact model key. `RouteEntry` gains `local_model`, written at decision time. `build_dashboard` groups D1 feedback by `local_model`, inverts the `capability_adjustment` slope on each model's flag rate to estimate `effective_b`, and returns `Vec<ModelCapability>`. The dashboard renders a card.

**Tech Stack:** Rust (serde), vanilla-JS manager UI. No new crates.

## Global Constraints

- Informational only: does NOT touch `capability_adjustment`, `effective_threshold`, or any routing path.
- `local_model` = `crate::settings::model_ctx_key(repo, file)` = `format!("{repo}/{file}")`; `#[serde(default, skip_serializing_if="Option::is_none")]`; legacy lines without it group as unknown and are excluded.
- Estimate consts (named, in `route_log.rs`): `CAP_MIN_SAMPLE = 30`, `CAP_BASELINE_FLAG_RATE = 0.10`, `CAP_SCALE = 0.5`, `CAP_SLOPE = 0.03` (must equal `route::capability_adjustment`'s slope).
- Formula: `excess = flagged_rate - CAP_BASELINE_FLAG_RATE`; `delta_b = -(excess * CAP_SCALE) / CAP_SLOPE`; `effective_b = (nominal_b + delta_b).clamp(0.5, nominal_b * 1.5)`.
- A local is flagged if it has ≥1 of `["cascade","truncated","reask"]` (reuse D1's `NEG_LOCAL`, `.any()` — a decision counts once).
- Models below `CAP_MIN_SAMPLE` locals, or with `nominal_b == 0`, are omitted. Sorted by `local_total` desc.
- UI renders model keys via textContent (never innerHTML).
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
  ```
- Build machine slow (warm ~10-13s, cold ~6min). Run cargo in background or with long timeouts.
- Branch: `feat/fase-d2` off `feat/fase-d1`.

---

### Task 1: `params_b_for_key` catalog helper

**Files:**
- Modify: `src/catalog.rs` (new pub fn + test)

**Interfaces:**
- Consumes: existing `CATALOG` (slice of `CatalogEntry { repo, file, params_b, .. }`), existing `params_b_from_name(repo, file) -> Option<f32>`.
- Produces: `pub fn params_b_for_key(key: &str) -> f32`.

- [ ] **Step 1: Write the failing test**

In `src/catalog.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn params_b_for_key_catalog_then_scan_then_zero() {
        // A real CATALOG entry: key = "{repo}/{file}".
        let key = format!(
            "{}/{}",
            "Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"
        );
        assert_eq!(super::params_b_for_key(&key), 3.0);
        // Not in catalog but the key string carries a size token.
        assert_eq!(super::params_b_for_key("foo/bar-13b.gguf"), 13.0);
        // Unknown → 0.0.
        assert_eq!(super::params_b_for_key("foo/bar-model.gguf"), 0.0);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib params_b_for_key -- --nocapture`
Expected: FAIL to compile — `params_b_for_key` not found.

- [ ] **Step 3: Implement**

Add to `src/catalog.rs` (near `active_params_b`):

```rust
/// Nominal parameter size (billions) for a compact model key
/// (`model_ctx_key(repo,file)` = `"{repo}/{file}"`). The key is not reliably
/// splittable back to (repo,file) because `repo` itself contains `/`, so this
/// resolves without splitting: a CATALOG entry whose `"{repo}/{file}"` equals
/// the key wins; else scan the whole key for a `NNb` size token; else 0.0.
pub fn params_b_for_key(key: &str) -> f32 {
    if let Some(e) = CATALOG
        .iter()
        .find(|e| format!("{}/{}", e.repo, e.file) == key)
    {
        return e.params_b;
    }
    // Reuse the same digit+"b" scan the name parser uses by passing the whole
    // key as the "repo" arg and an empty file.
    params_b_from_name(key, "").unwrap_or(0.0)
}
```

Note: `params_b_from_name` is `scan(file).or_else(|| scan(repo))` — it scans
both args. `params_b_from_name(key, "")` therefore falls through to `scan(key)`
and finds the `NNb` token anywhere in the key. Confirmed correct for the test's
`"foo/bar-13b.gguf"` → 13.0 case.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --lib params_b_for_key -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/catalog.rs
git commit -F - <<'EOF'
feat(catalog): params_b_for_key resolves nominal size from compact key

Catalog lookup by "{repo}/{file}", else NNb token scan, else 0.0. Used by
the per-model capability estimate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 2: `RouteEntry.local_model` field + decision-time wiring

**Files:**
- Modify: `src/route_log.rs` (RouteEntry struct + a round-trip test)
- Modify: `src/server.rs` (`route_decision` sets `local_model`; a wiring test)

**Interfaces:**
- Consumes: existing `RouteEntry`, `crate::settings::model_ctx_key`, `state.manager.status().current` (a `ModelSpec { repo, file, .. }`).
- Produces: `RouteEntry.local_model: Option<String>`.

- [ ] **Step 1: Write the failing round-trip test (route_log)**

In `src/route_log.rs` `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn route_entry_local_model_round_trips_and_defaults() {
        let e = RouteEntry { ts: 1, rid: "r".into(), surface: "openai".into(),
            dest: "local".into(), local_model: Some("Owner/Repo/file.gguf".into()),
            ..Default::default() };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"local_model\":\"Owner/Repo/file.gguf\""), "got {s}");
        let back: RouteEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back.local_model.as_deref(), Some("Owner/Repo/file.gguf"));
        // Legacy line without the field → None.
        let legacy: RouteEntry = serde_json::from_str(
            r#"{"ts":1,"surface":"openai","dest":"local","score":0.1,"prompt_tok":5,"rid":"r"}"#
        ).unwrap();
        assert_eq!(legacy.local_model, None);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib route_entry_local_model -- --nocapture`
Expected: FAIL to compile — no `local_model` field.

- [ ] **Step 3: Implement the field**

In `src/route_log.rs`, add to `RouteEntry` (after the existing `model` field
near the end of the struct):

```rust
    /// Compact key (model_ctx_key(repo,file)) of the LOCAL model active when
    /// this decision was made. Set on every decision; None on legacy lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_model: Option<String>,
```

- [ ] **Step 4: Run the round-trip test**

Run: `cargo test --lib route_entry_local_model -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Write the failing wiring test (server)**

In `src/server.rs` `#[cfg(test)] mod tests`, add a test that drives
`route_decision` against a minimal AppState and asserts the appended decision
line carries the active model key. Reuse the `wiring_state` helper (already in
`mod tests`) — but note its ModelManager's current spec is
`{repo:"test", file:"test"}`, so `model_ctx_key` = `"test/test"`.

```rust
    #[tokio::test]
    async fn route_decision_records_active_local_model() {
        let _guard = crate::route_log::ROUTE_LOG_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("localllm-lm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let logp = dir.join("routing-log.jsonl");
        std::env::set_var("LOCALLLM_ROUTE_LOG", &logp);

        let breaker = std::sync::Arc::new(crate::breaker::CircuitBreaker::new());
        let state = wiring_state(breaker);
        let req = crate::api::common::ChatRequest {
            messages: vec![crate::api::common::ChatMessage {
                role: crate::api::common::Role::User,
                text: Some("oi".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let headers = axum::http::HeaderMap::new();
        let _ = super::route_decision(&state, &req, &headers, "rlm1", "openai");

        let lines = crate::route_log::read_all();
        let dec = lines.iter().find_map(|l| match l {
            crate::route_log::LogLine::Decision(d) if d.rid == "rlm1" => Some(d.clone()),
            _ => None,
        }).expect("decision line");
        assert_eq!(dec.local_model.as_deref(), Some("test/test"));

        std::env::remove_var("LOCALLLM_ROUTE_LOG");
        let _ = std::fs::remove_dir_all(&dir);
    }
```

Note: adapt the `ChatRequest`/`ChatMessage` literal to the real struct fields
(check `src/api/common.rs`; if `ChatMessage` has no `Default`, build it with the
fields the other server tests use). `route_decision` is `fn` (not async) but
lives in a module used by async handlers; calling it inside a `#[tokio::test]`
is fine. If `route_decision` is not reachable as `super::route_decision` from
the test module, confirm its visibility (it is a private `fn` in the same file,
so `super::route_decision` works).

- [ ] **Step 6: Run to verify failure**

Run: `cargo test --lib route_decision_records_active_local_model -- --nocapture`
Expected: FAIL — `local_model` is None (not yet wired).

- [ ] **Step 7: Wire it in `route_decision`**

In `src/server.rs` `route_decision`, the primary decision `RouteEntry` literal
uses `..Default::default()` and has `active` already in scope (computed near the
top as `let active = state.manager.status().current;`). Add the field to that
literal (alongside `capability_b: ...`, before `..Default::default()`):

```rust
        capability_b: Some(local_capability_b as f64),
        local_model: Some(crate::settings::model_ctx_key(&active.repo, &active.file)),
        prompt_snippet,
        ..Default::default()
```

(Only the ONE primary decision literal in `route_decision`. Do NOT add it to the
`log_degrade_fallback` literal — that path is a post-hoc fallback record, out of
scope; it keeps `local_model: None` via Default.)

- [ ] **Step 8: Run both tests + build**

Run: `cargo test --lib route_entry_local_model route_decision_records_active_local_model -- --nocapture`
Expected: PASS.

Run: `cargo build`
Expected: success.

- [ ] **Step 9: Commit**

```bash
git add src/route_log.rs src/server.rs
git commit -F - <<'EOF'
feat(route_log): record active local model per decision

RouteEntry.local_model carries the compact key of the serving local
model; route_decision sets it from the active model. Legacy lines
default to None.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 3: `ModelCapability` aggregation in `build_dashboard`

**Files:**
- Modify: `src/route_log.rs` (ModelCapability struct + consts + aggregation + Dashboard field + tests)

**Interfaces:**
- Consumes from Tasks 1-2: `crate::catalog::params_b_for_key`, `RouteEntry.local_model`, the existing `feedback: HashMap<&str, Vec<String>>` map and `decisions: Vec<&RouteEntry>` in `build_dashboard`, `NEG_LOCAL`.
- Produces: `pub struct ModelCapability { model, local_total, flagged_rate, nominal_b, effective_b }`; `Dashboard.model_capabilities: Vec<ModelCapability>`.

- [ ] **Step 1: Write the failing tests**

In `src/route_log.rs` `mod tests`:

```rust
    #[test]
    fn model_capability_estimates_and_filters() {
        let now = 1_000_000i64;
        // Model key that scans to 3.0 B nominal (params_b_for_key: "*-3b*").
        let m = "Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf";
        let mut lines = Vec::new();
        // 40 locals for model m; 12 flagged (30% flag rate).
        for i in 0..40 {
            let rid = format!("m{i}");
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: rid.clone(),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some(m.into()), ..Default::default() }));
            if i < 12 {
                lines.push(LogLine::Feedback(FeedbackEntry { rid, ts: now - 9, signal: "cascade".into() }));
            }
        }
        // A second model with too few locals → omitted.
        for i in 0..5 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: format!("s{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some("foo/tiny-1b.gguf".into()), ..Default::default() }));
        }
        let d = build_dashboard(&lines, now, 10, true);
        assert_eq!(d.model_capabilities.len(), 1, "only the 40-local model qualifies");
        let mc = &d.model_capabilities[0];
        assert_eq!(mc.model, m);
        assert_eq!(mc.local_total, 40);
        assert!((mc.flagged_rate - 0.30).abs() < 1e-9);
        assert!((mc.nominal_b - 3.0).abs() < 1e-9);
        // effective_b: excess=0.30-0.10=0.20; delta=-(0.20*0.5)/0.03 = -3.333..;
        // 3.0 + (-3.333) = -0.333 → clamp low 0.5.
        assert!((mc.effective_b - 0.5).abs() < 1e-9, "got {}", mc.effective_b);
    }

    #[test]
    fn model_capability_high_flag_clamps_low_and_healthy_above_nominal() {
        let now = 1_000_000i64;
        let m = "Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf";
        let mut lines = Vec::new();
        // 40 locals, ZERO flagged → excess = -0.10; delta = +(0.10*0.5)/0.03 = +1.667;
        // effective = 3.0 + 1.667 = 4.667 (< clamp high 3.0*1.5=4.5) → clamps to 4.5.
        for i in 0..40 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: format!("h{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some(m.into()), ..Default::default() }));
        }
        let d = build_dashboard(&lines, now, 10, true);
        let mc = &d.model_capabilities[0];
        assert!((mc.flagged_rate - 0.0).abs() < 1e-9);
        assert!((mc.effective_b - 4.5).abs() < 1e-9, "got {}", mc.effective_b);
    }

    #[test]
    fn model_capability_skips_unknown_nominal() {
        let now = 1_000_000i64;
        let mut lines = Vec::new();
        // 40 locals but the key has no size token and isn't in catalog → nominal 0 → skip.
        for i in 0..40 {
            lines.push(LogLine::Decision(RouteEntry { ts: now - 10, rid: format!("u{i}"),
                surface: "openai".into(), dest: "local".into(), score: 0.2,
                local_model: Some("foo/mystery-model.gguf".into()), ..Default::default() }));
        }
        let d = build_dashboard(&lines, now, 10, true);
        assert!(d.model_capabilities.is_empty());
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib model_capability -- --nocapture`
Expected: FAIL to compile — `ModelCapability` / `model_capabilities` missing.

- [ ] **Step 3: Implement the struct + consts**

In `src/route_log.rs`, near `FeedbackStats` / `ThresholdSuggestion`:

```rust
/// Per-local-model effective-capability estimate (informational; not routed).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelCapability {
    /// Compact model key (model_ctx_key).
    pub model: String,
    pub local_total: u64,
    pub flagged_rate: f64,
    /// Nominal size (billions) from params_b_for_key.
    pub nominal_b: f64,
    /// Estimated effective size (billions) after inverting the capability slope.
    pub effective_b: f64,
}

/// Effective-capability tunables — named so a later phase can calibrate them.
pub const CAP_MIN_SAMPLE: u64 = 30;
pub const CAP_BASELINE_FLAG_RATE: f64 = 0.10;
pub const CAP_SCALE: f64 = 0.5;
/// Must equal the slope in `route::capability_adjustment` (0.03 threshold / 1B).
pub const CAP_SLOPE: f64 = 0.03;
```

- [ ] **Step 4: Implement the aggregation**

In `build_dashboard`, after the existing feedback-stats/suggestion block and
before constructing `Dashboard`, add a per-model pass. It reuses `decisions`
(the `Vec<&RouteEntry>`) and `feedback` (the `HashMap<&str, Vec<String>>`), and
the existing `NEG_LOCAL` const (declared earlier in the fn — reuse it; do not
redeclare):

```rust
    // Per-model effective-capability estimate (informational).
    use std::collections::BTreeMap;
    let mut per_model: BTreeMap<&str, (u64, u64)> = BTreeMap::new(); // model → (total, flagged)
    for d in &decisions {
        if d.dest != "local" { continue; }
        let Some(m) = d.local_model.as_deref() else { continue; };
        let flagged = feedback
            .get(d.rid.as_str())
            .map(|s| s.iter().any(|x| NEG_LOCAL.contains(&x.as_str())))
            .unwrap_or(false);
        let e = per_model.entry(m).or_insert((0, 0));
        e.0 += 1;
        if flagged { e.1 += 1; }
    }
    let mut model_capabilities: Vec<ModelCapability> = per_model
        .into_iter()
        .filter(|(_, (total, _))| *total >= CAP_MIN_SAMPLE)
        .filter_map(|(model, (total, flagged))| {
            let nominal_b = crate::catalog::params_b_for_key(model) as f64;
            if nominal_b <= 0.0 { return None; }
            let flagged_rate = flagged as f64 / total as f64;
            let excess = flagged_rate - CAP_BASELINE_FLAG_RATE;
            let delta_b = -(excess * CAP_SCALE) / CAP_SLOPE;
            let effective_b = (nominal_b + delta_b).clamp(0.5, nominal_b * 1.5);
            Some(ModelCapability {
                model: model.to_string(),
                local_total: total,
                flagged_rate,
                nominal_b,
                effective_b,
            })
        })
        .collect();
    model_capabilities.sort_by(|a, b| b.local_total.cmp(&a.local_total));
```

- [ ] **Step 5: Add the Dashboard field**

Extend the `Dashboard` struct (after `suggestion`):

```rust
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<ThresholdSuggestion>,
    #[serde(default)]
    pub model_capabilities: Vec<ModelCapability>,
}
```

And include `model_capabilities,` in the returned `Dashboard { ... }` literal.

- [ ] **Step 6: Run tests + build**

Run: `cargo test --lib model_capability -- --nocapture`
Expected: PASS (all three).

Run: `cargo build`
Expected: success.

- [ ] **Step 7: Commit**

```bash
git add src/route_log.rs
git commit -F - <<'EOF'
feat(dashboard): per-model effective-capability estimate

build_dashboard groups D1 feedback by local_model and inverts the
capability slope on each model's flag rate to estimate effective_b
(clamped 0.5..nominal*1.5), skipping models below CAP_MIN_SAMPLE or with
unknown nominal size.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

### Task 4: Dashboard UI card

**Files:**
- Modify: `src/manager_ui/app.js` (`renderDashboard` — card after the D1 accuracy card)
- Modify: `src/manager_ui/style.css`

**Interfaces:**
- Consumes: dashboard JSON `d.model_capabilities: [{model, local_total, flagged_rate, nominal_b, effective_b}]`. Helpers `el(tag, cls, html)` (3rd arg is innerHTML — do not pass dynamic content through it).
- Produces: UI only.

- [ ] **Step 1: Add the card**

In `renderDashboard` (`src/manager_ui/app.js`), after the D1 accuracy card
append (search for the card whose head text is "ACERTO DO ROUTING" — append
right after it), add:

```javascript
  // Per-model effective capability (informational)
  const caps = d.model_capabilities || [];
  if (caps.length) {
    const capCard = el("div", "dash-card");
    capCard.append(el("div", "dash-card-head", "CAPACIDADE EFETIVA POR MODELO"));
    caps.forEach((c) => {
      const row = el("div", "cap-row");
      const name = el("span", "cap-model");
      name.textContent = c.model;
      row.append(name);
      const stat = el("span", "cap-stat");
      const arrow = c.effective_b < c.nominal_b ? "↓" : (c.effective_b > c.nominal_b ? "↑" : "=");
      stat.textContent =
        `${c.local_total} locais · ${Math.round(c.flagged_rate * 100)}% flag · `
        + `nominal ${c.nominal_b}b ${arrow} efetivo ${c.effective_b.toFixed(1)}b`;
      row.append(stat);
      capCard.append(row);
    });
    wrap.append(capCard);
  }
```

(Use the real variable name for the dashboard container — it is `wrap` in
`renderDashboard`, same as the D1 card. If the accuracy card variable is
`acc`, appending after `wrap.append(acc);` is the target spot.)

- [ ] **Step 2: Add CSS**

Append to `src/manager_ui/style.css`:

```css
/* Per-model effective capability card */
.cap-row { display: flex; justify-content: space-between; gap: 12px;
  padding: 4px 0; border-top: 1px solid var(--line); font-size: .9em; }
.cap-row:first-of-type { border-top: none; }
.cap-model { color: var(--muted); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; max-width: 45%; }
.cap-stat { text-align: right; }
```

- [ ] **Step 3: Build gate**

Run: `cargo build`
Expected: success (assets are string-embedded).

Manual visual check deferred to the user's end-to-end pass.

- [ ] **Step 4: Commit**

```bash
git add src/manager_ui/app.js src/manager_ui/style.css
git commit -F - <<'EOF'
feat(ui): per-model effective-capability dashboard card

Lists each qualifying local model with its flag rate and nominal→effective
capability estimate. Model keys rendered via textContent.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
```

---

## Final Verification (after all tasks)

- [ ] `cargo test --lib` (background) → all green (241 before D2 + new).
- [ ] `cargo test --test http -- --test-threads=1` → 47 green.
- [ ] `cargo build` clean.

## Self-Review Notes

- **Spec coverage:** helper (T1), attribution field + wiring (T2), aggregation + formula + filters (T3), UI (T4). All spec sections mapped.
- **Type consistency:** `params_b_for_key(&str)->f32`, `RouteEntry.local_model: Option<String>`, `ModelCapability{model,local_total,flagged_rate,nominal_b,effective_b}`, `Dashboard.model_capabilities`, consts `CAP_MIN_SAMPLE/CAP_BASELINE_FLAG_RATE/CAP_SCALE/CAP_SLOPE` — used identically across tasks.
- **Formula check:** worked the two clamp cases in the T3 tests by hand (30% flag on 3B → clamp 0.5; 0% flag on 3B → 4.667 clamp to 4.5). Consts match the spec.
- **Verification points for implementers:** `params_b_from_name` arg that carries the scan (T1); `ChatRequest`/`ChatMessage` real fields + `route_decision` visibility (T2); `NEG_LOCAL` reuse without redeclare (T3); accuracy-card append variable name (T4).
