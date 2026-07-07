# Fase E1 — Backend Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feature-flag the Metal-hardwired inference dependencies into `metal`/`cuda`/`cpu` backends so the project compiles on macOS/Linux/Windows, with Metal auto-selected on macOS and zero macOS runtime behavior change.

**Architecture:** Pure build configuration. Turn off the three inference crates' default features, expose `metal`/`cuda`/`cpu` cargo features that forward to both engines (`mistralrs` + `llama-cpp-2`/`llama-cpp-sys-2`), auto-enable Metal on macOS via a target-specific dependency table, and add a `compile_error!` guard so a backend-less non-macOS build fails with an actionable message. The inference code is already backend-agnostic; no logic changes.

**Tech Stack:** Rust, Cargo features, `mistralrs 0.8`, `llama-cpp-2 0.1.150`, `llama-cpp-sys-2 0.1.150`.

## Global Constraints

- **No macOS runtime/behavior change.** A plain `cargo build` on macOS must stay Metal and produce an identical binary; the full test suite must stay green under Metal in default parallel mode.
- **Metal auto on macOS, zero new flags.** A local `cargo build`/`cargo test` needs no `--features`.
- **No `default` backend feature.** Linux/Windows builds require an explicit `--features cuda` or `--features cpu`; a backend-less non-macOS build must fail at compile time with a clear message.
- **Both engines build under every backend.** Each backend feature forwards to `mistralrs` AND `llama-cpp-2`/`llama-cpp-sys-2` (config default `Backend::Llama`, alternate `Backend::Mistralrs`).
- **Vulkan/ROCm excluded** (mistralrs supports neither).
- **`cpu` is portable/bare:** no `openmp` (avoids a libomp system dep), no `mkl`/`accelerate`. Slower CPU is a later optimization.
- **Local verification is wiring-only.** On macOS the target table forces Metal, so a non-metal *build* cannot be produced here. Non-metal build+runtime is CI's job (E2). Local checks: `cargo build` (metal) green, tests green, and `cargo tree -e features` resolves the new feature refs without a native build.
- **Commit trailers:** every commit message ends with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe`.
- **Branch:** work continues on `feat/fase-d1` (the stacked branch; user prefers same-branch).

---

### Task 1: Cargo backend feature wiring

**Files:**
- Modify: `Cargo.toml:7` (mistralrs dep), `Cargo.toml:21-22` (llama deps), plus a new `[features]` section and a new `[target.'cfg(target_os = "macos")'.dependencies]` section.

**Interfaces:**
- Consumes: nothing (first task).
- Produces: cargo features `metal`, `cuda`, `cpu`. `metal` forwards to `mistralrs/metal` + `llama-cpp-2/metal` + `llama-cpp-2/common` + `llama-cpp-sys-2/metal`. `cuda` forwards to `mistralrs/cuda` + `llama-cpp-2/cuda` + `llama-cpp-2/common` + `llama-cpp-sys-2/cuda`. `cpu` forwards to `llama-cpp-2/common` + `llama-cpp-sys-2/common` (mistralrs bare = candle CPU). macOS builds get `metal` + `common` + `openmp` upstream features unconditionally via the target table (preserving today's exact macOS feature set). Task 2 relies on the fact that a plain macOS build does NOT activate this crate's own `metal` feature flag (the target table activates the *upstream* crates directly).

- [ ] **Step 1: Record the current feature set (context, no change)**

The three dependency lines today (from `Cargo.toml`) are:
```toml
mistralrs = { version = "0.8", features = ["metal"] }
llama-cpp-2 = { version = "0.1.150", features = ["metal"] }
llama-cpp-sys-2 = { version = "0.1.150", features = ["metal"] }
```
Because `default-features` is not disabled, macOS currently builds with
`metal` PLUS each crate's defaults. For `llama-cpp-2`/`llama-cpp-sys-2` the
default set includes `openmp` and `common`. The macOS target table below
re-supplies `metal` + `common` + `openmp` so the macOS build is byte-for-byte
the same feature set as today.

- [ ] **Step 2: Rewrite the three dependency lines to disable defaults**

In `Cargo.toml`, replace the three lines from Step 1 with:
```toml
mistralrs = { version = "0.8", default-features = false }
llama-cpp-2 = { version = "0.1.150", default-features = false }
llama-cpp-sys-2 = { version = "0.1.150", default-features = false }
```

- [ ] **Step 3: Add the `[features]` backend table**

Add this section to `Cargo.toml` (place it after the `[dependencies]` block,
before `[dev-dependencies]`):
```toml
[features]
# Backend selection. Each forwards to BOTH engines so either the Llama
# (default) or Mistralrs engine builds under the chosen backend.
# `common` is llama.cpp's shared utilities (part of its old default set).
metal = ["mistralrs/metal", "llama-cpp-2/metal", "llama-cpp-2/common", "llama-cpp-sys-2/metal"]
cuda  = ["mistralrs/cuda",  "llama-cpp-2/cuda",  "llama-cpp-2/common", "llama-cpp-sys-2/cuda"]
# Portable CPU: no openmp (avoids a libomp system dep), no mkl/accelerate.
# mistralrs stays bare (candle CPU) — it needs no feature for CPU.
cpu   = ["llama-cpp-2/common", "llama-cpp-sys-2/common"]
```

- [ ] **Step 4: Add the macOS target dependency table (auto-Metal)**

Add this section to `Cargo.toml` (after the `[features]` section):
```toml
# Auto-enable Metal on macOS so a plain `cargo build` needs no flags and the
# existing dev/build loop is unchanged. Re-supplies the exact feature set the
# project built with before (metal + llama's default common + openmp). These
# activate the UPSTREAM crates' features directly — they do NOT set this
# crate's own `metal` feature flag (see Task 2's guard).
[target.'cfg(target_os = "macos")'.dependencies]
mistralrs = { version = "0.8", default-features = false, features = ["metal"] }
llama-cpp-2 = { version = "0.1.150", default-features = false, features = ["metal", "common", "openmp"] }
llama-cpp-sys-2 = { version = "0.1.150", default-features = false, features = ["metal", "common", "openmp"] }
```

- [ ] **Step 5: Validate feature wiring resolves (no native build)**

Run each and confirm cargo resolves the feature graph WITHOUT erroring on an
unknown `crate/feature` reference. This validates the manifest on macOS without
needing nvcc/MSVC (it does not compile the C libraries):
```bash
cargo tree -e features --features cuda >/dev/null && echo "cuda-refs-ok"
cargo tree -e features --features cpu  >/dev/null && echo "cpu-refs-ok"
cargo tree -e features --features metal >/dev/null && echo "metal-refs-ok"
```
Expected: three lines `cuda-refs-ok`, `cpu-refs-ok`, `metal-refs-ok` and no
`error: failed to select a version` / `does not have feature` messages.
If cargo reports a missing feature (e.g. a typo like `mistralrs/cudaa`), fix
the reference in the `[features]` table and re-run.

- [ ] **Step 6: Verify the macOS build is unchanged**

Run (this is the real macOS regression gate — the binary must still build with
Metal via the target table, zero flags):
```bash
cargo build 2>&1 | tail -5
```
Expected: `Finished` (a successful build). The build links Metal exactly as
before. On the slow build machine this can take several minutes — run with a
long timeout / in the background; do not shorten it.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
build(fase-e1): feature-flag inference backends (metal/cuda/cpu)

Disable default features on mistralrs/llama-cpp-2/llama-cpp-sys-2 and
expose metal/cuda/cpu cargo features forwarding to both engines. Auto-
enable Metal on macOS via a target dep table so a plain `cargo build`
stays Metal with zero flags and the same feature set as before. cpu is
portable/bare (no openmp/mkl). No default backend feature — Linux/Windows
must pass --features cuda|cpu.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

### Task 2: Compile-time backend guard

**Files:**
- Modify: `src/lib.rs:1` (add a guard block at the very top, before the `pub mod` lines).

**Interfaces:**
- Consumes: the `cuda` and `cpu` features defined in Task 1.
- Produces: a `compile_error!` that fires on a non-macOS build with no backend feature. macOS is exempt (its target table always supplies Metal, and a plain macOS build does not set this crate's `cuda`/`cpu`/`metal` feature flags).

- [ ] **Step 1: Add the guard at the top of `src/lib.rs`**

Insert at the very top of `src/lib.rs`, above the first `pub mod api;` line:
```rust
// Backend selection guard. On Linux/Windows a build must pick an inference
// backend explicitly; without one the native engines would fail to link with
// an inscrutable error. macOS is exempt: its target dependency table
// (Cargo.toml) always supplies Metal, so a plain `cargo build` needs no flag.
#[cfg(all(not(target_os = "macos"), not(feature = "cuda"), not(feature = "cpu")))]
compile_error!(
    "No inference backend selected. On Linux/Windows build with \
     `--features cuda` or `--features cpu`. macOS enables Metal automatically."
);

```

- [ ] **Step 2: Verify the guard does NOT fire on macOS**

The guard must be inert on macOS (exempt), so the crate still compiles. Run:
```bash
cargo build 2>&1 | tail -3
```
Expected: `Finished` — no `compile_error` output. (The `not(target_os = "macos")`
arm is false on macOS, so the whole `cfg(all(...))` is false and the
`compile_error!` is not emitted.) Long build — use a generous timeout.

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs
git commit -m "$(cat <<'EOF'
build(fase-e1): compile_error guard for backend-less non-macOS builds

A Linux/Windows build with neither --features cuda nor cpu now fails fast
with an actionable message instead of a deep native link error. macOS is
exempt (Metal auto-enabled via the target dep table).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

### Task 3: Explicit Metal feature in build-app.sh + macOS regression

**Files:**
- Modify: `scripts/build-app.sh:42` (the `cargo build` line).

**Interfaces:**
- Consumes: the `metal` feature from Task 1.
- Produces: nothing consumed by later tasks (terminal task of E1).

- [ ] **Step 1: Make the bundle build request Metal explicitly**

In `scripts/build-app.sh`, change line 42 from:
```bash
cargo build --profile "$PROFILE"
```
to:
```bash
cargo build --profile "$PROFILE" --features metal
```
Rationale: on macOS the target table already supplies Metal, so this is
belt-and-suspenders — it documents intent and keeps the bundle correct if the
macOS target default is ever removed. It is additive (no double-link issue).

- [ ] **Step 2: Audit platform `#[cfg]` gates (confirm no gaps)**

Confirm the existing gating already covers non-macOS (this is a read-only
verification, not a change). Run:
```bash
grep -rn "target_os = \"macos\"\|cfg(unix)\|cfg(not(unix))\|cfg(all(unix" src/ | grep -v "test"
```
Expected: each macOS-only item has a non-macOS counterpart or is genuinely
macOS-only. Confirm specifically:
- `src/main.rs` `--export-icon` is `#[cfg(target_os = "macos")]` (macOS-only, intended — the bundle icon export).
- `src/tray.rs` clipboard/open have `windows` and `all(unix, not(macos))` arms.
- `src/tray.rs` activation policy is `#[cfg(target_os = "macos")]` with no else needed.
- `src/tray.rs` `hard_exit` has `#[cfg(unix)]` + `#[cfg(not(unix))]`.
If any macOS-only item lacks a needed non-macOS branch, note it in the task
report (do not expand scope — E1 is build wiring; real cross-OS runtime gaps
surface in E2/CI).

- [ ] **Step 3: Full macOS test-suite regression**

The binary and its behavior must be unchanged on macOS. Run the full suite
under Metal in default parallel mode (the http flake was fixed in the polish
pass, so no `--test-threads=1`):
```bash
cargo test 2>&1 | tail -20
```
Expected: all suites pass — lib (~247), http (~47), plus the small bin unit
tests. No failures. Long run on the slow build machine — use a generous
timeout / background it; do not shorten.

- [ ] **Step 4: Commit**

```bash
git add scripts/build-app.sh
git commit -m "$(cat <<'EOF'
build(fase-e1): build-app.sh requests --features metal explicitly

Documents the macOS bundle's backend and keeps it correct if the macOS
target default is ever removed. Additive with the target dep table.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

## Verification Summary (whole-branch)

After all three tasks:
- `cargo build` (Metal, zero flags) → `Finished`. macOS binary unchanged.
- `cargo tree -e features --features cuda|cpu|metal` → all resolve, no bad refs.
- `cargo test` → full suite green in default parallel mode.
- `compile_error!` guard present for backend-less non-macOS builds.
- `scripts/build-app.sh` passes `--features metal`.

**Explicitly NOT verified in E1 (deferred to E2/CI + user):** real `cuda`/`cpu`
builds and runtime on Linux/Windows. E1's done bar is: wiring lands, macOS
stays green, feature refs resolve. The `plist`'s hardcoded `0.1.0` version is
left for E3 (release/versioning), not touched here.
