# Fase E1 — Backend Portability (Design)

**Date:** 2026-07-07
**Status:** Approved — ready for implementation plan.
**Part of:** Fase E (cross-platform release), sub-project 1 of 3. Followed by
E2 (CI) and E3 (release automation).

## Goal

Make `localllm` compile — and run headless — on all five target configs
without changing any macOS runtime behavior:

| OS | Arch | Backend feature |
|----|------|-----------------|
| macOS | arm64 | `metal` |
| Linux | x86_64 | `cuda` |
| Linux | x86_64 | `cpu` |
| Windows | x86_64 | `cuda` |
| Windows | x86_64 | `cpu` |

Today the build is Metal-locked at the dependency level: `Cargo.toml`
hardwires `features = ["metal"]` on `mistralrs`, `llama-cpp-2`, and
`llama-cpp-sys-2`. Metal is Apple-only, so the project cannot currently
compile off macOS. E1 removes that lock via Cargo feature flags and a few
compile-time guards. The inference code itself is already backend-agnostic
(it calls generic APIs like `with_n_gpu_layers` and device mapping), so E1 is
overwhelmingly build configuration, not logic changes.

## Non-Goals (explicit scope fence)

- **Packaging / release artifacts** — E3. E1 delivers compilation + headless
  run per config, not `.app`/zip/installer output.
- **The CI workflow YAML** — E2. E1 lands the feature wiring that CI will
  exercise; the matrix jobs themselves are E2.
- **Vulkan / ROCm backends** — excluded. `mistralrs` supports neither
  (only `llama-cpp-2` does), and both engines must build under every selected
  backend. Revisit only if the mistralrs dependency is dropped from a build.
- **MKL / Accelerate CPU acceleration** — the `cpu` feature ships bare
  (portable, slower) candle-CPU + llama openmp. Faster CPU math is a later
  optimization, not E1.
- **Runtime verification of non-Metal backends** — see Verification below.
  E1 lands correct wiring; real cuda/cpu *runtime* on Linux/Windows is proven
  by CI (E2) and the user, not on the macOS dev machine.

## Backend Feature Wiring

### Dependency changes (`Cargo.toml`)

Turn off the hardwired defaults on the three inference crates and select
backends through this crate's own features:

```toml
[dependencies]
mistralrs        = { version = "0.8",       default-features = false }
llama-cpp-2      = { version = "0.1.150",   default-features = false }
llama-cpp-sys-2  = { version = "0.1.150",   default-features = false }

[features]
# Each backend feature forwards to BOTH engines so either engine
# (config `Backend::Llama` default, or `Backend::Mistralrs`) builds under it.
metal = ["mistralrs/metal", "llama-cpp-2/metal", "llama-cpp-sys-2/metal"]
cuda  = ["mistralrs/cuda",  "llama-cpp-2/cuda",  "llama-cpp-sys-2/cuda"]
# cpu: portable, no GPU. llama gets openmp+common; mistralrs stays bare
# candle-CPU (no mkl/accelerate) so it links on any Linux/Windows runner.
cpu   = ["llama-cpp-2/openmp", "llama-cpp-2/common",
         "llama-cpp-sys-2/openmp", "llama-cpp-sys-2/common"]

# Auto-enable Metal on macOS so a plain `cargo build` needs no flags and the
# existing dev/build loop is unchanged. This is additive: on macOS the metal
# upstream features are always linked.
[target.'cfg(target_os = "macos")'.dependencies]
mistralrs       = { version = "0.8",     default-features = false, features = ["metal"] }
llama-cpp-2     = { version = "0.1.150", default-features = false, features = ["metal"] }
llama-cpp-sys-2 = { version = "0.1.150", default-features = false, features = ["metal"] }
```

Notes:
- **No `default` backend feature.** A plain `cargo build` on macOS gets Metal
  from the target table; on Linux/Windows a plain build activates no backend
  and hits the compile guard below, forcing an explicit `--features`.
- The exact upstream feature names for the `cpu` set (`openmp`, `common`) are
  confirmed against the crates' `[features]` tables; if `common` proves
  unnecessary for a bare build, the plan drops it — it is included to match the
  crates' own `default` set which bundled `common`.

### Compile-time guard

Add one guard in a central module (e.g. top of `src/lib.rs`) so a
mis-configured non-macOS build fails immediately with an actionable message
instead of a deep linker error:

```rust
#[cfg(all(not(target_os = "macos"), not(feature = "cuda"), not(feature = "cpu")))]
compile_error!(
    "No inference backend selected. On Linux/Windows build with \
     `--features cuda` or `--features cpu`. macOS enables Metal automatically."
);
```

(macOS is exempt because the target table always supplies `metal`.)

## Code Gating

Most platform gating already exists and is preserved as-is:
- `--export-icon` — already `#[cfg(target_os = "macos")]` in `main.rs`.
- macOS activation policy (`ActivationPolicy::Accessory`) — already gated.
- `hard_exit` — already `#[cfg(unix)]` / `#[cfg(not(unix))]`.
- Clipboard / open-path — already cross-platform (`pbcopy`/`clip`/`wl-copy`/
  `xclip`, `open`/`start`/`xdg-open`).

E1's code work is therefore an **audit pass**, not a rewrite:
1. Grep every `#[cfg(target_os = "macos")]` / `#[cfg(unix)]` and confirm the
   `not` branch exists or the item is genuinely macOS-only (icon export).
2. Add the compile guard above.
3. Preserve the runtime Metal notes untouched (they are env/runtime, not
   build): `MISTRALRS_METAL_PRECOMPILE=0`, and "quantized KV requires flash
   attention on Metal". These do not gate compilation on other platforms.
4. Update `scripts/build-app.sh` to pass `--features metal` explicitly (it is
   macOS-only already; making the backend explicit documents intent and keeps
   it working if the macOS target default is ever removed).

## Data Flow / Behavior

No change to request handling, routing, or engine logic. The only observable
change on macOS is `Cargo.toml`; the compiled binary and its behavior are
identical. On Linux/Windows the same binary logic runs against a
cuda- or cpu-linked ggml/candle backend.

## Error Handling

- **Build-time:** the `compile_error!` guard turns "forgot a backend flag"
  into a one-line, self-explaining failure.
- **Runtime:** unchanged. Existing engine error paths (Metal OOM recovery,
  decode-failure context reset) are backend-agnostic in code; their comments
  reference Metal but the recovery logic (`context reset`, `fresh backend`)
  applies to any ggml backend.

## Verification

- **Local (macOS dev machine), part of E1:**
  - `cargo build` — Metal, unchanged; must succeed as today.
  - Full test suite under Metal — must stay green (lib + http + bin units),
    default parallel mode (the http flake was fixed in the polish pass).
  - `cargo check --features cpu` — proves the `cpu` feature set *resolves and
    type-checks* on top of the macOS metal target dependency. This is a
    wiring check, not a true CPU-only link.
- **Deferred to E2 (CI) / the user:** real `cuda` and `cpu` *builds and
  runtime* on Linux and Windows. E1 cannot verify these on an Apple-Silicon
  Mac (no nvcc, no MSVC, metal always linked via the target table). This is
  stated plainly so the plan's "done" bar for E1 is: wiring lands, macOS stays
  green, cpu features resolve — not "all five configs proven running."

## Testing Strategy

E1 is pure build configuration; it adds no runtime behavior to unit-test.
- Keep the existing suite as the regression net (must pass under metal).
- The compile guard is validated implicitly (a macOS build compiles because it
  is exempt; a hypothetical backend-less Linux build would fail to compile — 
  not expressible as a runtime test on macOS, so not asserted in a test).
- No new unit tests are added for feature wiring (there is no logic to assert);
  a plan step may add a one-line doc note near the guard instead.

## Open Risks

- **mistralrs cuda build** may pull heavy build-time deps (nvcc, cudnn) that
  only surface in CI — the highest-risk cell, deferred to E2.
- **Windows link** (MSVC, static C++ runtime) for `llama-cpp-sys-2` is the
  least-tested path; `static-stdcxx` may be needed and will be tuned in E2's
  CI, not guessed here.
- These risks are why E1 stops at "wiring + macOS green + cpu resolves"; CI is
  the instrument that closes them.

## References

- Roadmap: [[../../../.claude/projects/-Users-ricardo-Repos-localllm/memory/fases-roadmap-status]]
- Prior deferrals this phase: `2026-07-07-fase-d3-multi-local-tiering-DEFERRED.md`,
  `2026-07-07-fase-d4-semantic-routing-DEFERRED.md`.
- Follow-on sub-projects: E2 (CI matrix), E3 (release automation + changelog +
  publish).
