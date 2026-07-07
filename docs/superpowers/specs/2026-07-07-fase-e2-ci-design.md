# Fase E2 — Continuous Integration (Design)

**Date:** 2026-07-07
**Status:** Approved — ready for implementation plan.
**Part of:** Fase E (cross-platform release), sub-project 2 of 3. Follows E1
(backend portability, DONE), precedes E3 (release automation).

## Goal

A GitHub Actions workflow that builds and tests `localllm` across the five
E1 backend×OS configs on every push and PR, plus a lint gate (`fmt` + `clippy`).
It turns E1's untestable-on-macOS promise ("cuda/cpu compile on Linux/Windows")
into a real, automated check, and guards the codebase against regressions.

## Context / Prerequisite

E1 landed the `metal`/`cuda`/`cpu` cargo features (metal auto on macOS;
cuda/cpu opt-in via `--features`; a `compile_error!` guard for backend-less
non-macOS builds). E2 exercises that wiring on real runners. The test suite is
**pure**: tests use `localllm::router_for_test()` — no model download, no GPU,
no network — so the full suite runs anywhere the crate compiles.

The branch `feat/fase-d1` (which carries C/D1/D2/E1, stacked and unmerged) must
be **pushed to `github.com/rzorzal/localllm`** for the workflow to run. E2
authors and commits the workflow locally; the user pushes and iterates the runs
(the first real validation of the YAML happens on their repo — an expected,
flagged part of E2).

## Architecture

One workflow file: `.github/workflows/ci.yml`.

```
on:
  push:            # any branch
  pull_request:
    branches: [main]

concurrency:       # cancel superseded runs on the same ref (save minutes)
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:            # fmt + clippy, one cheap runner
  build-test:      # 5-config matrix
```

### Job: `lint`

- Runner: `ubuntu-latest`.
- Steps: checkout → install Linux system deps (see below) → `rustup` with
  `rustfmt` + `clippy` → `Swatinem/rust-cache` → `cargo fmt --all --check`
  (**blocking**) → `cargo clippy --workspace --all-targets --features cpu`
  (**non-blocking / informational — NO `-D warnings`**).
- Backend: `cpu` (clippy must compile the crate; cpu needs no GPU/toolkit).
- Rationale: `fmt --check` is instant and OS-agnostic; clippy needs a compile,
  so it rides the same cheap Linux/cpu runner rather than a costly macOS one.
- **fmt is blocking, clippy is not** (user decision): the tree is brought to a
  rustfmt baseline once (see below) so `fmt --check` stays green; clippy runs
  without `-D warnings`, surfacing its ~12 current warnings in the log without
  failing CI. This deliberately avoids touching `server.rs` lock scopes (the 7
  `await_holding_lock` warnings) or refactoring `too_many_arguments` in E2 — a
  real clippy cleanup is a separate future effort.

### fmt baseline (one-time, part of E2)

The codebase has never been rustfmt-formatted: `cargo fmt --all --check`
currently fails in ~35 files. E2 applies a single `cargo fmt --all` reformat in
a dedicated `style:` commit so the blocking fmt gate is green from the first
push. This is a large but purely mechanical diff with no behavior change; the
test suite is re-run after to confirm.

### Job: `build-test` (matrix, `fail-fast: false`)

| # | `os` | `backend` | `run_tests` | Notes |
|---|------|-----------|-------------|-------|
| 1 | `macos-14` | `metal` | yes | arm64; Metal auto (no `--features` needed, but pass `--features metal` for parity) |
| 2 | `ubuntu-latest` | `cpu` | yes | apt deps required |
| 3 | `windows-latest` | `cpu` | yes | MSVC + cmake preinstalled |
| 4 | `ubuntu-latest` | `cuda` | no (build-only) | CUDA toolkit via `Jimver/cuda-toolkit` |
| 5 | `windows-latest` | `cuda` | no (build-only) | CUDA toolkit via `Jimver/cuda-toolkit` |

- **Run vs build-only:** cpu and metal cells run `cargo test` (the pure suite).
  cuda cells run only `cargo build --features cuda` — hosted runners have no
  NVIDIA driver, and a cuda-linked test binary may fail to load `libcuda` at
  startup, so compilation is the meaningful check there.
- **`fail-fast: false`:** one config's failure must not cancel the others while
  the matrix is being stabilized.
- **Feature selection per cell:** macOS passes `--features metal` (additive with
  the target table); Linux/Windows pass `--features cpu` or `--features cuda`.
  macOS never passes cuda/cpu (no such hardware path).

### Matrix encoding

Use `matrix.include` with explicit `{os, backend, run_tests}` entries (not a
cross-product) so the two cuda cells are build-only and the three others test —
a plain `os × backend` product would generate invalid cells (macOS×cuda,
macOS×cpu) that must be excluded anyway.

## Supporting Pieces

### Linux system dependencies (apt)

Both the `lint` job and the ubuntu `build-test` cells install, before building:
```
libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev \
libxdo-dev cmake libclang-dev
```
- `webkit2gtk`/`gtk`/`appindicator`/`xdo`: `wry` + `tray-icon` + `tao` build
  deps (the tray/webview compile on Linux even though it is launched only with
  `--tray`).
- `cmake` + `libclang-dev`: `llama-cpp-sys-2` builds ggml via cmake and
  generates bindings via bindgen (needs libclang).
- Exact package names (e.g. `4.0` vs `4.1` webkit) are the most likely first-
  push failure and will be tuned during the user's CI iteration; the spec picks
  the current Ubuntu LTS names as the starting point.

### Caching

`Swatinem/rust-cache@v2` in every job, keyed by job + backend. It caches Rust
build artifacts and the cargo registry. The C++ ggml/cmake build is largely
outside this cache, so first runs (and cache-busting dependency changes) are
slow; this is accepted, not solved, in E2.

### CUDA toolkit

`Jimver/cuda-toolkit@v0.2.x` installs nvcc + CUDA libs on the two cuda cells so
`--features cuda` links. Pin a specific CUDA version compatible with the
`cudarc`/`mistralrs` stack (tuned in iteration; a recent 12.x to start).

## Local Deliverable (verifiable on the macOS dev box)

Before the workflow can be green, the code must satisfy the **blocking** part of
the lint gate (fmt). Runnable locally and part of E2:
- Apply `cargo fmt --all`, then confirm `cargo fmt --all --check` → clean.
- Re-run the full test suite → still green (fmt is mechanical; E1 verified
  298/0). This guards against fmt accidentally changing a doctest/macro layout.
- clippy is **not** cleaned: the CI clippy step is informational (no
  `-D warnings`), so the ~12 existing warnings are left in place and simply
  reported. No local clippy fixing is required in E2.

The YAML itself cannot be executed locally; its correctness is validated by the
user's first push (flagged expectation, not a gap E2 can close).

## Error Handling / Failure Modes

- **fmt failures:** the fmt gate is blocking; the one-time reformat baseline
  makes it green from the first push. clippy is non-blocking (no `-D`), so its
  warnings never turn CI red.
- **A matrix cell fails to build on first push:** `fail-fast: false` keeps the
  other cells reporting; the user iterates (most likely apt package names or the
  CUDA version). Expected, not a defect in E2's deliverable.
- **Long/slow builds:** mitigated by `rust-cache` + `concurrency` cancel; not
  eliminated.

## Testing Strategy

E2 is CI configuration; its "tests" are:
- Local: `cargo fmt --all --check` clean (after the baseline reformat), suite
  green. clippy is informational only — not required to be clean.
- Post-push (user): the five matrix cells go green. E2's authored deliverable is
  the workflow + a lint-clean tree; achieving all-green across runners is the
  iteration the user drives on their repo.

No new unit tests are added (there is no application logic in a CI workflow).

## Non-Goals

- **Release artifacts / versioning / changelog / publish** — E3.
- **Deploying or distributing binaries** — E3.
- **Running GPU inference in CI** — impossible on hosted runners; cuda cells are
  build-only by design.
- **Vulkan/ROCm cells** — excluded (mistralrs blocks them; consistent with E1).
- **Self-hosted GPU runners** — out of scope; would be the only way to run cuda
  tests, a separate future decision.

## References

- Roadmap: [[../../../.claude/projects/-Users-ricardo-Repos-localllm/memory/fases-roadmap-status]]
- Predecessor: `2026-07-07-fase-e1-backend-portability-design.md` (backend features).
- Successor: E3 (release automation) — will add a separate
  `release.yml` reusing this matrix to produce per-OS artifacts.
