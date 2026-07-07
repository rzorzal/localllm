# Fase E3 — Release Automation (Design)

**Date:** 2026-07-07
**Status:** Approved — ready for implementation plan.
**Part of:** Fase E (cross-platform release), sub-project 3 of 3. Follows E1
(backend portability, DONE) and E2 (CI, DONE).

## Goal

A manually-triggered GitHub Actions release: the user picks a semver bump
(patch/minor/major) and a release title; the workflow bumps the version,
regenerates a changelog from the conventional-commit history, tags the release,
builds a downloadable artifact for each of the five E1 backend×OS configs, and
publishes a GitHub Release for others to download.

## Context / Prerequisites

- E1 gave the `metal`/`cuda`/`cpu` cargo features; E2 gave a build matrix that
  compiles all five configs on hosted runners (cuda build-only, no GPU).
- The version lives in `Cargo.toml` (`version = "0.1.0"`) and is hardcoded a
  second time in `scripts/build-app.sh`'s Info.plist (`CFBundleVersion` +
  `CFBundleShortVersionString`, both `0.1.0`). There are no git tags and no
  changelog yet.
- Commit history is strongly conventional (`feat`, `fix`, `docs`, `build`,
  `ci`, `style`, `refactor`), so `git-cliff` can group a clean changelog.
- `gh` CLI is available on GitHub runners (with `GITHUB_TOKEN`).
- Like E2, `release.yml` only executes once `feat/fase-d1` is pushed to
  `github.com/rzorzal/localllm`. E3 authors and commits it locally; the user
  pushes and validates with the first real dispatch.

## Single Source of Version Truth

**Refactor `scripts/build-app.sh` to derive the plist version from
`Cargo.toml`** rather than hardcoding it. The script computes
`VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"` and substitutes it
into both plist keys. After this, **`Cargo.toml` is the only place a human/CI
edits the version**; the macOS bundle version always follows automatically. This
removes the drift risk and means the release workflow only mutates `Cargo.toml`
(and `Cargo.lock`).

## Architecture

Two new files: `.github/workflows/release.yml` and `cliff.toml` (git-cliff
config). One modified file: `scripts/build-app.sh` (version derivation).

### Trigger

```yaml
on:
  workflow_dispatch:
    inputs:
      bump:
        description: Semver bump
        type: choice
        options: [patch, minor, major]
        default: patch
      title:
        description: Release title
        type: string
        required: true
```

### Job 1: `prepare`

Runs on `ubuntu-latest`. Produces the tag, the bumped version, and the release
body.

1. `actions/checkout` with `fetch-depth: 0` (full history + tags — git-cliff and
   the "since last tag" range need them).
2. Compute the next version: read the current `Cargo.toml` version, apply the
   `bump` input (patch/minor/major) with a small deterministic semver step, emit
   `X.Y.Z`.
3. Apply the bump to `Cargo.toml` + `Cargo.lock` (via `cargo set-version` from
   `cargo-edit`, or an equivalent edit + `cargo update -p localllm`).
4. `git-cliff` regenerates the repo `CHANGELOG.md` (full history, grouped by
   type) **and** writes the current release's section to a separate body file
   for the GitHub Release.
5. Commit `chore(release): vX.Y.Z [skip ci]` (the `[skip ci]` marker prevents
   the CI workflow from re-triggering on this push), create annotated tag
   `vX.Y.Z`, and push both the commit and the tag to the dispatched branch.
6. Job outputs: `version` (`X.Y.Z`), `tag` (`vX.Y.Z`).

### Job 2: `build` (matrix, `needs: prepare`)

Reuses the E2 five-config matrix, but builds **release** binaries and packages
one artifact per config. Each cell checks out the new tag (so the artifact
carries the bumped version).

| # | os | backend | package |
|---|----|---------| ------- |
| 1 | macos-14 | metal | run `scripts/build-app.sh`, zip the `.app` → `localllm-vX.Y.Z-macos-arm64.zip` |
| 2 | ubuntu-latest | cpu | `cargo build --release --features cpu`, tar the binary → `localllm-vX.Y.Z-linux-x64-cpu.tar.gz` |
| 3 | ubuntu-latest | cuda | `cargo build --release --features cuda` (+ CUDA toolkit), tar → `localllm-vX.Y.Z-linux-x64-cuda.tar.gz` |
| 4 | windows-latest | cpu | `cargo build --release --features cpu`, zip the `.exe` → `localllm-vX.Y.Z-windows-x64-cpu.zip` |
| 5 | windows-latest | cuda | `cargo build --release --features cuda` (+ CUDA toolkit), zip → `localllm-vX.Y.Z-windows-x64-cuda.zip` |

- Same Linux apt deps and `Jimver/cuda-toolkit` steps as E2.
- Each cell uploads its package via `actions/upload-artifact`.
- cuda cells build (no GPU needed to compile) and package the binary; they are
  never run, only shipped.

### Job 3: `release` (`needs: build`)

Runs on `ubuntu-latest`:
1. `actions/download-artifact` (all five).
2. `gh release create "$TAG"` (or `softprops/action-gh-release`) with:
   - the `title` input as the release title,
   - the git-cliff body file as the release notes,
   - all five artifact files attached.

## Changelog (`cliff.toml`)

A standard git-cliff conventional-commit config: group `feat` → Features,
`fix` → Bug Fixes, `docs` → Documentation, `build`/`ci` → Build & CI,
`refactor` → Refactor, `style`/`test` → skipped or a Misc group. `CHANGELOG.md`
is committed to the repo (persistent history) and its newest section is the
Release body. The macOS-Gatekeeper note (below) is appended to the release body
via the workflow, not the changelog.

## Error Handling / Failure Modes

- **CI re-trigger loop:** the release commit carries `[skip ci]` so E2's `push`
  trigger does not fire on it.
- **Racing releases:** a `concurrency` group (e.g. `release`) with
  `cancel-in-progress: false` serializes dispatches.
- **Permissions:** the workflow needs `permissions: contents: write` to push the
  commit/tag and create the Release.
- **Invalid bump / version parse failure:** the compute step fails fast (a
  non-`X.Y.Z` current version or an unknown bump aborts before any tag is made).
- **A build cell fails:** the `release` job `needs: build`, so a failed artifact
  build blocks publishing — no partial release. (The tag/commit from `prepare`
  will already exist; re-running after a fix is a documented manual step.)

## Out of Scope / Known Limitations

- **macOS code signing / notarization:** the `.app` is unsigned — downloaders
  get a Gatekeeper "unidentified developer" prompt. Signing needs a paid Apple
  Developer certificate; out of scope. The release-body template documents the
  right-click → Open workaround.
- **Windows code signing:** likewise unsigned (SmartScreen warning). Out of
  scope.
- **Homebrew/winget/apt package publishing:** out of scope — GitHub Releases
  only.
- **Runtime validation of the workflow:** cannot run locally; the first real
  dispatch on the user's repo is the validation. Expected first-dispatch
  iteration: the same apt/CUDA tuning as E2, plus artifact-path/upload details.
- **cuda artifacts are compile-only** — never executed in CI; a broken cuda
  runtime would only surface on a user's GPU machine.

## Testing Strategy

E3 is CI/release configuration; verifiable locally:
- The `build-app.sh` version-derivation change: after editing, run the grep
  extraction and confirm it yields the `Cargo.toml` version (and, if a full
  bundle build is run, that the plist shows it). The version-derivation line
  itself is checked without a full LTO build.
- The semver bump logic: a tiny, testable pure step (given `0.1.0` + `minor` →
  `0.2.0`, etc.).
- `cliff.toml`: `git-cliff --unreleased` (if git-cliff is available locally)
  produces a sensibly grouped section; otherwise the config is validated for
  TOML well-formedness and reviewed against the git-cliff schema.
- `release.yml`: YAML parses; three jobs with the correct `needs` wiring and the
  five-cell matrix.
- End-to-end (tag → 5 artifacts → published Release) is the user's first
  dispatch.

No new application unit tests (no application logic changes beyond the
build-app.sh version derivation, which is shell).

## References

- Roadmap: [[../../../.claude/projects/-Users-ricardo-Repos-localllm/memory/fases-roadmap-status]]
- Predecessors: `2026-07-07-fase-e1-backend-portability-design.md`,
  `2026-07-07-fase-e2-ci-design.md` (the matrix E3 reuses).
