# Circuit-Breaker Passive Self-Heal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the cloud circuit-breaker banner wedging at "Cloud indisponível · nova tentativa em 0s" — make the elapsed-cooldown `Open → HalfOpen` transition happen on a passive read (the dashboard poll), not only on a cloud request.

**Architecture:** Add a mutating `CircuitBreaker::poll(now)` that performs the same elapsed-cooldown `Open → HalfOpen` transition `gate()` already does (carrying `backoff` + `reason`), then returns the snapshot. Point the `/admin/breaker` GET endpoint at `poll` instead of `snapshot`, so the dashboard's existing periodic poll drives recovery. No new background task. Trip/backoff/reset semantics unchanged, so escalation toward the 300s cap is preserved.

**Tech Stack:** Rust (`std::sync::Mutex`, axum), vanilla JS dashboard.

## Global Constraints

- No active/synthetic cloud probe and no new tokio task — the dashboard poll is the only driver.
- `poll` carries `backoff` and `reason` into `HalfOpen` exactly as `gate()` does; `on_failure`/`on_success`/`reset`/`gate`/`snapshot` are NOT modified.
- `poll` is infallible: recover a poisoned lock via `.unwrap_or_else(|e| e.into_inner())`, matching the other methods. No panics, no network I/O.
- Half-open banner copy (JS): exactly `Cloud pronto — testa na próxima chamada`.
- No CSS change: `.brk-half-open { background: var(--amber); }` already exists and the dot class is `brk-${state}`.
- The dashboard `/admin/breaker` GET is the only `snapshot()` caller to change; the tray only calls `reset()` — do not touch it.

---

### Task 1: `CircuitBreaker::poll` + unit tests

**Files:**
- Modify: `src/breaker.rs` (add `poll` method next to `snapshot`; add tests to the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: existing `BreakerState`, `BreakerSnapshot`, `snapshot(now)`, `gate(now)`, `on_failure(now, reason)`.
- Produces: `pub fn poll(&self, now: u64) -> BreakerSnapshot` — mutating read that transitions an elapsed `Open` to `HalfOpen`, then returns the snapshot.

- [ ] **Step 1: Write the failing tests**

Add these three tests to the `mod tests` block in `src/breaker.rs`:

```rust
    #[test]
    fn poll_elapsed_open_transitions_to_half_open() {
        let b = CircuitBreaker::new();
        b.on_failure(100, DegradeReason::Quota); // Open until 130, backoff 30
        // Not elapsed: poll leaves it Open with the correct countdown.
        let s = b.poll(120);
        assert_eq!(s.state, "open");
        assert_eq!(s.next_probe_secs, Some(10));
        // Elapsed: poll transitions to half-open and reports the reason.
        let s = b.poll(130);
        assert_eq!(s.state, "half-open");
        assert_eq!(s.reason, Some("Quota"));
        // It is a REAL transition, not just a reported label: the next gate
        // returns Allow (the probe), which only happens from Closed/HalfOpen.
        assert_eq!(b.gate(130), Gate::Allow);
    }

    #[test]
    fn poll_is_noop_on_closed_and_half_open() {
        let b = CircuitBreaker::new();
        assert_eq!(b.poll(0).state, "closed"); // closed → no-op
        b.on_failure(0, DegradeReason::Offline); // Open until 30
        assert_eq!(b.gate(30), Gate::Allow); // → half-open via gate
        assert_eq!(b.poll(30).state, "half-open"); // no-op
        assert_eq!(b.poll(999).state, "half-open"); // still half-open, no-op
    }

    #[test]
    fn poll_preserves_backoff_escalation() {
        let b = CircuitBreaker::new();
        b.on_failure(0, DegradeReason::Quota); // backoff 30, Open until 30
        let s = b.poll(30); // half-open via poll (not gate)
        assert_eq!(s.state, "half-open");
        b.on_failure(30, DegradeReason::Quota); // failed probe → doubled to 60
        // Backoff escalated, NOT reset to base 30.
        assert_eq!(b.snapshot(30).next_probe_secs, Some(60));
    }
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `cargo test -p localllm poll_`
Expected: FAIL — `no method named poll found for struct CircuitBreaker`.

- [ ] **Step 3: Implement `poll`**

Add this method inside `impl CircuitBreaker`, immediately after the `snapshot` method in `src/breaker.rs`:

```rust
    /// Passive-read advance for the dashboard poll: if `Open` with an elapsed
    /// cooldown, transition to `HalfOpen` (carrying `backoff` + `reason`, as
    /// `gate` does) so recovery happens without needing a cloud request, then
    /// return the snapshot of the resulting state. Unlike `snapshot` this may
    /// MUTATE; unlike `gate` it grants no `Allow` to a caller. `on_failure`
    /// still owns backoff escalation, so a failed probe after this doubles the
    /// cooldown rather than resetting it.
    pub fn poll(&self, now: u64) -> BreakerSnapshot {
        {
            let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let BreakerState::Open {
                until,
                backoff,
                reason,
            } = *st
            {
                if now >= until {
                    *st = BreakerState::HalfOpen { backoff, reason };
                }
            }
        } // drop the lock before snapshot re-acquires it
        self.snapshot(now)
    }
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cargo test -p localllm poll_`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full breaker suite (no regressions)**

Run: `cargo test -p localllm breaker`
Expected: PASS — all existing breaker tests plus the 3 new ones.

- [ ] **Step 6: Commit**

```bash
git add src/breaker.rs
git commit -m "feat(breaker): poll() advances elapsed Open to HalfOpen on read"
```

---

### Task 2: Wire the dashboard endpoint to `poll` + half-open banner copy + build

**Files:**
- Modify: `src/server.rs` (`handle_breaker_get` — swap `snapshot` → `poll`)
- Modify: `src/manager_ui/app.js` (`paintBreaker` — half-open label copy)

**Interfaces:**
- Consumes: `CircuitBreaker::poll(now)` (Task 1).

- [ ] **Step 1: Point the GET endpoint at `poll`**

In `src/server.rs`, in `handle_breaker_get`, change the snapshot line:

```rust
    let s = state.breaker.poll(crate::route_log::now_secs() as u64);
```

(Only that one line changes — the `Json(json!({ ... }))` response body stays identical since `poll` returns the same `BreakerSnapshot`.)

- [ ] **Step 2: Update the half-open banner copy**

In `src/manager_ui/app.js`, in `paintBreaker`, change the `label` ternary's half-open branch. Replace:

```js
  const label = state === "closed" ? "Cloud disponível"
    : state === "half-open" ? "Testando cloud (probe)…"
    : `Cloud indisponível (${b.reason || "erro"})`;
```

with:

```js
  const label = state === "closed" ? "Cloud disponível"
    : state === "half-open" ? "Cloud pronto — testa na próxima chamada"
    : `Cloud indisponível (${b.reason || "erro"})`;
```

(The dot color needs no change: `el("span", "brk-dot brk-" + state)` already yields `brk-half-open`, and `.brk-half-open { background: var(--amber); }` already exists in `style.css`.)

- [ ] **Step 3: Build + full test suite**

Run: `cargo build -p localllm && cargo test -p localllm`
Expected: clean build; all tests pass (includes the Task 1 breaker tests and the existing HTTP suite).

- [ ] **Step 4: Commit**

```bash
git add src/server.rs src/manager_ui/app.js
git commit -m "feat(breaker): dashboard poll drives self-heal; half-open banner copy"
```

- [ ] **Step 5: Rebuild the app bundle**

Run: `bash scripts/build-app.sh --fast`
Expected: `==> SUCCESS`.

- [ ] **Step 6: Verify the copy is embedded in the bundle**

Run: `grep -c "testa na próxima chamada" src/manager_ui/app.js`
Expected: `1` (the source the bundle embeds via `include_str!`).

- [ ] **Step 7: Commit any tweaks**

```bash
git add -A && git commit -m "chore: breaker self-heal verification tweaks" --allow-empty
```

---

### Task 3: Manual verification (controller / user)

**Files:** none (manual).

- [ ] **Step 1: Restart the app bundle** (`target/localllm.app`).

- [ ] **Step 2: Force a trip** — either hit the real Quota, or use the tray/dashboard to trip cloud (send a cloud-routed request while cloud creds are bad), then STOP sending requests.

- [ ] **Step 3: Watch the dashboard breaker strip** with no further requests. Expected: red "Cloud indisponível (…) · nova tentativa em Ns" counts down to 0s, then at cooldown flips to amber "Cloud pronto — testa na próxima chamada" on the next dashboard poll — WITHOUT any request being sent. The next cloud-routed request then probes; success → green "Cloud disponível", failure → re-trips with a longer cooldown.

---

## Self-Review

**Spec coverage:**
- `poll(now)` mutating elapsed `Open → HalfOpen`, carries backoff+reason, returns snapshot → Task 1. ✓
- `/admin/breaker` GET uses `poll` → Task 2 Step 1. ✓
- `gate`/`on_failure`/`on_success`/`reset`/`snapshot` unchanged; backoff escalation preserved → Task 1 (poll only) + `poll_preserves_backoff_escalation` test. ✓
- No background task, no synthetic probe → nothing added; driver is the endpoint. ✓
- Half-open copy exactly `Cloud pronto — testa na próxima chamada`; amber dot already wired → Task 2 Steps 2 + note. ✓
- Poisoned-lock recovery, infallible → `poll` uses `unwrap_or_else(|e| e.into_inner())`. ✓
- Tests: elapsed→half-open (real, via gate Allow), not-elapsed stays open, no-op on closed/half-open, backoff preserved → Task 1 three tests. ✓
- Rollout: rebuild bundle + manual watch → Task 2 Steps 5–6 + Task 3. ✓

**Placeholder scan:** every code step has concrete code; the manual task is a real reproduction. No TBD/TODO.

**Type consistency:** `poll(&self, now: u64) -> BreakerSnapshot` defined in Task 1, consumed in Task 2 Step 1 with matching return type (the JSON body reads `s.state`/`s.reason`/`s.next_probe_secs`, all fields of `BreakerSnapshot`). `HalfOpen { backoff, reason }` matches the existing enum variant. Test helpers (`on_failure`, `gate`, `snapshot`, `Gate::Allow`, `DegradeReason::*`) all exist in `breaker.rs`.
