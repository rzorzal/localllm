# Install Helper (Design)

**Date:** 2026-07-08
**Status:** Approved — ready for implementation plan.

## Goal

Make it a one-liner for a user to install the right `localllm` build for their
machine. The backend is a compile-time choice (no runtime GPU auto-detect), and
the release ships five artifacts, so users must pick the correct one. This
delivers two things: auto-detecting install scripts (`install.sh` for
macOS/Linux, `install.ps1` for Windows) and a "Which download?" selector table
in the README for manual choice.

## Context / Prerequisite

The release workflow (`.github/workflows/release.yml`) publishes five assets per
tag: `localllm-vX.Y.Z-macos-arm64.zip` (a `.app` bundle), `…-linux-x64-cpu.tar.gz`,
`…-linux-x64-cuda.tar.gz`, `…-windows-x64-cpu.zip`, `…-windows-x64-cuda.zip`. The
cuda artifacts are built for **CUDA compute capability 8.0 (Ampere / RTX 30xx and
newer)**. The scripts download from the repo's **latest GitHub Release**, so they
only function after the first release is cut; until then they exit with a clear
"no release found" message. Repo: `rzorzal/localllm`.

## Architecture

Three deliverables, each independently usable:

1. `scripts/install.sh` — POSIX-ish bash, macOS + Linux.
2. `scripts/install.ps1` — PowerShell, Windows.
3. A README **Install** section: the selector table + the one-liner commands.

The two scripts share the same logical pipeline (detect → resolve → download →
install) but are separate files because their shells and install mechanics differ.

## Detection → Artifact Selection

Common decision, implemented in each script:

- **macOS** (`uname -s` = `Darwin`): artifact `macos-arm64`. If `uname -m` is not
  `arm64`, warn that only Apple Silicon (Metal) is published and exit.
- **Linux / Windows**: probe for an Nvidia GPU and its compute capability:
  - Run `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` (bash) or the
    PowerShell equivalent.
  - If `nvidia-smi` is present AND the highest compute cap is **≥ 8.0** → the
    `cuda` variant.
  - Otherwise (no `nvidia-smi`, or an older GPU like Turing/Pascal < 8.0) → the
    `cpu` variant. This is correct: the cuda binary targets sm_80 and would not
    run on older cards.
- Arch is assumed `x86_64` for Linux/Windows (the only published non-mac arch);
  a non-x86_64 Linux/Windows arch warns and exits.

## Release Resolution

- Query `https://api.github.com/repos/rzorzal/localllm/releases/latest`.
- Parse the asset whose name matches `localllm-v*-<variant>.<ext>` and take its
  `browser_download_url`.
- bash: prefer `jq` if present, else a `grep`/`sed` fallback so `jq` is not a
  hard dependency. PowerShell: `Invoke-RestMethod` returns parsed JSON directly.
- If the API returns no release, or no asset matches the variant, print a clear
  message (e.g. "No published release yet — see the Releases page") and exit
  non-zero.

## Install (user-local, no sudo)

- **macOS**: unzip the `.app` into `~/Applications/localllm.app` (create the dir
  if needed). Print that it is unsigned → first launch is right-click → **Open**.
- **Linux**: `tar xz` the binary to `~/.local/bin/localllm`, `chmod +x`. If
  `~/.local/bin` is not on `PATH`, print a line to add it.
- **Windows**: expand the zip into `%LOCALAPPDATA%\localllm`, add that dir to the
  **user** `PATH` (via `setx` / the registry user env), and note the SmartScreen
  "More info → Run anyway" step (unsigned).

## Flags / Modes

- `--print` (bash) / `-Print` (ps1): dry run — detect + resolve + print the
  chosen artifact and its download URL, then exit WITHOUT downloading or
  installing. This is the locally-testable path.
- Default (no flag): full detect → download → install.

## Error Handling

- No network / API failure → error message + non-zero exit.
- No matching release/asset → "no release yet" message.
- Unsupported OS (not Darwin/Linux for `.sh`) or arch → clear message, exit.
- No `nvidia-smi` → silently selects `cpu` (expected, not an error).
- Download/extract failure → error with the URL that failed.

## README Selector Table

An **Install** section containing:

| Your machine | Download |
|---|---|
| macOS (Apple Silicon) | `localllm-<ver>-macos-arm64.zip` (Metal) |
| Linux + Nvidia (Ampere / RTX 30xx+) | `localllm-<ver>-linux-x64-cuda.tar.gz` |
| Linux (other GPU / no GPU) | `localllm-<ver>-linux-x64-cpu.tar.gz` |
| Windows + Nvidia (Ampere / RTX 30xx+) | `localllm-<ver>-windows-x64-cuda.zip` |
| Windows (other GPU / no GPU) | `localllm-<ver>-windows-x64-cpu.zip` |

Plus the one-liners:
- macOS/Linux: `curl -fsSL https://raw.githubusercontent.com/rzorzal/localllm/main/scripts/install.sh | bash`
- Windows: `irm https://raw.githubusercontent.com/rzorzal/localllm/main/scripts/install.ps1 | iex`

(The `raw.githubusercontent.com/.../main/...` URLs resolve once `feat/fase-d1`
is merged to `main`; note this in the README.)

## Testing Strategy

Shell scripts are hard to unit-test; verification is:
- `bash -n scripts/install.sh` (syntax) and, if available, `shellcheck`.
- `scripts/install.sh --print` run locally on the macOS dev box → must resolve to
  `macos-arm64` (detection path exercised without any download).
- PowerShell: `-Print` mode is documented; real execution is on the user's
  Windows machine.
- The full download+install path is only exercisable after a release exists —
  that is the user's end-to-end test, matching the release itself.

No application unit tests (these are standalone shell/PowerShell utilities).

## Out of Scope

- Uninstall scripts, auto-update, version pinning (install always fetches the
  latest release). These can be added later if wanted.
- Package managers (Homebrew tap / winget / apt) — GitHub Releases only.
- Code signing / notarization (the artifacts are unsigned; the scripts document
  the Gatekeeper/SmartScreen workaround).

## References

- Release workflow producing the assets: `.github/workflows/release.yml`
  (Fase E3). Roadmap: [[../../../.claude/projects/-Users-ricardo-Repos-localllm/memory/fases-roadmap-status]].
