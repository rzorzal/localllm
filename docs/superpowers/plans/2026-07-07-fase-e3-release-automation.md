# Fase E3 — Release Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A manually-dispatched GitHub Actions release that bumps the version (patch/minor/major), regenerates a git-cliff changelog, tags the release, builds a downloadable artifact for each of the five backend×OS configs, and publishes a GitHub Release.

**Architecture:** Three deliverables. (1) `scripts/build-app.sh` derives the plist version from `Cargo.toml` (single source of truth). (2) `cliff.toml` configures git-cliff over the repo's conventional commits. (3) `.github/workflows/release.yml` — `prepare` (bump + changelog + tag), `build` (5-config matrix → artifacts), `release` (publish).

**Tech Stack:** GitHub Actions, `cargo-edit` (`cargo set-version`), `orhun/git-cliff-action`, `gh` CLI, the E2 build matrix + `Jimver/cuda-toolkit` + `Swatinem/rust-cache`.

## Global Constraints

- **`Cargo.toml` is the sole human/CI-edited version source.** After Task 1, `scripts/build-app.sh` reads the version from `Cargo.toml`; nothing else hardcodes it.
- **Release commit carries `[skip ci]`** so E2's `push` trigger does not re-fire.
- **Artifacts: all five configs** — `macos-arm64` (zipped `.app`), `linux-x64-cpu`, `linux-x64-cuda` (`.tar.gz`), `windows-x64-cpu`, `windows-x64-cuda` (`.zip`). cuda artifacts are compile-only (built, never run).
- **`workflow_dispatch` inputs:** `bump` (choice patch/minor/major, default patch) and `title` (string, required).
- **`permissions: contents: write`** on the workflow (push tag/commit + create release). **`concurrency: release`** (no racing releases).
- **Unsigned artifacts** — no code signing/notarization (out of scope); the release body documents the macOS right-click→Open workaround.
- **The workflow only runs once `feat/fase-d1` is pushed** — end-to-end validation is the user's first dispatch. Local verification is limited to shell logic, TOML/YAML well-formedness, and structure.
- **Commit trailers:** every commit message ends with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe`.
- **Branch:** `feat/fase-d1` (stacked, unmerged).

---

### Task 1: `build-app.sh` derives version from `Cargo.toml`

**Files:**
- Modify: `scripts/build-app.sh` (add a `VERSION` var near the other `APP_*` vars; use it in the Info.plist heredoc at the two `0.1.0` lines).

**Interfaces:**
- Consumes: nothing.
- Produces: a `build-app.sh` whose bundle version always equals `Cargo.toml`'s. Task 3's release relies on this — it bumps only `Cargo.toml`, and the macOS `.app` picks the new version up automatically.

- [ ] **Step 1: Add the `VERSION` variable**

In `scripts/build-app.sh`, find the block that defines the app vars:
```bash
APP_NAME="localllm"
BUNDLE_ID="dev.localllm.app"
APP_OUT="$REPO_ROOT/target/${APP_NAME}.app"
```
Insert directly AFTER the `APP_OUT=...` line:
```bash
# Version is derived from Cargo.toml — the single source of truth. The release
# workflow (Fase E3) bumps only Cargo.toml; the bundle version follows.
VERSION="$(grep -m1 '^version' "$REPO_ROOT/Cargo.toml" | cut -d'"' -f2)"
if [[ -z "$VERSION" ]]; then
    echo "ERROR: could not read version from Cargo.toml" >&2
    exit 1
fi
```

- [ ] **Step 2: Use `${VERSION}` in the Info.plist heredoc**

In the same file, the plist heredoc currently has:
```
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
```
Replace BOTH `<string>0.1.0</string>` lines so it reads:
```
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
```
(The heredoc is unquoted — it already interpolates `${APP_NAME}` etc. — so `${VERSION}` expands.)

- [ ] **Step 3: Verify the derivation (no full build)**

Confirm the extraction yields the current Cargo.toml version without running the
slow LTO bundle build:
```bash
grep -m1 '^version' Cargo.toml | cut -d'"' -f2
```
Expected: `0.1.0` (the current version). Then confirm the script has no syntax
error:
```bash
bash -n scripts/build-app.sh && echo "syntax-ok"
```
Expected: `syntax-ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/build-app.sh
git commit -m "$(cat <<'EOF'
build(fase-e3): derive .app plist version from Cargo.toml

Single source of version truth: build-app.sh reads `version` from
Cargo.toml instead of a hardcoded 0.1.0, so the release workflow only
bumps Cargo.toml and the macOS bundle version follows automatically.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

### Task 2: `cliff.toml` (git-cliff changelog config)

**Files:**
- Create: `cliff.toml` (repo root).

**Interfaces:**
- Consumes: the repo's conventional-commit history.
- Produces: a git-cliff config used by Task 3's `prepare` job to write `CHANGELOG.md` and the release-notes body.

- [ ] **Step 1: Create `cliff.toml`**

Create `cliff.toml` at the repo root with EXACTLY this content:
```toml
# git-cliff configuration — groups conventional commits into a changelog.
# Used by the Fase E3 release workflow (.github/workflows/release.yml).

[changelog]
header = "# Changelog\n\nAll notable changes to localllm.\n"
body = """
{% if version %}\
## [{{ version | trim_start_matches(pat="v") }}] - {{ timestamp | date(format="%Y-%m-%d") }}
{% else %}\
## [Unreleased]
{% endif %}\
{% for group, commits in commits | group_by(attribute="group") %}
### {{ group | upper_first }}
{% for commit in commits %}
- {{ commit.message | upper_first }}\
{% endfor %}
{% endfor %}\n
"""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
split_commits = false
commit_parsers = [
  { message = "^feat", group = "Features" },
  { message = "^fix", group = "Bug Fixes" },
  { message = "^docs", group = "Documentation" },
  { message = "^perf", group = "Performance" },
  { message = "^refactor", group = "Refactor" },
  { message = "^tune", group = "Tuning" },
  { message = "^route", group = "Routing" },
  { message = "^build", group = "Build & CI" },
  { message = "^ci", group = "Build & CI" },
  { message = "^test", skip = true },
  { message = "^style", skip = true },
  { message = "^chore\\(release\\)", skip = true },
  { message = "^chore", skip = true },
]
protect_breaking_commits = true
filter_commits = false
tag_pattern = "v[0-9]*"
topo_order = false
sort_commits = "oldest"
```

- [ ] **Step 2: Validate the TOML parses**

```bash
python3 -c "import tomllib; tomllib.load(open('cliff.toml','rb')); print('cliff.toml OK')"
```
Expected: `cliff.toml OK`.

- [ ] **Step 3: (Best-effort) sanity-check git-cliff output if installed**

If `git-cliff` is available locally, confirm it renders without error; if not,
skip (the config is validated in CI on first dispatch):
```bash
if command -v git-cliff >/dev/null 2>&1; then
  git-cliff --config cliff.toml --unreleased 2>&1 | head -20
else
  echo "git-cliff not installed locally — config validated by TOML parse only"
fi
```
Expected: either a grouped changelog preview, or the skip message. Neither is a
failure.

- [ ] **Step 4: Commit**

```bash
git add cliff.toml
git commit -m "$(cat <<'EOF'
build(fase-e3): add git-cliff config for release changelog

Groups conventional commits (feat/fix/docs/build/ci/tune/route...) into
CHANGELOG.md sections. Consumed by the release workflow.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

### Task 3: `release.yml` (the release workflow)

**Files:**
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: `cliff.toml` (Task 2), the version-deriving `build-app.sh` (Task 1), the E1 cargo features, the E2 matrix pattern.
- Produces: nothing (terminal task).

- [ ] **Step 1: Create `.github/workflows/release.yml`**

Create the file with EXACTLY this content:
```yaml
name: Release

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

permissions:
  contents: write

concurrency:
  group: release
  cancel-in-progress: false

jobs:
  prepare:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.bump.outputs.version }}
      tag: ${{ steps.bump.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-edit
        run: cargo install cargo-edit --locked
      - name: Compute + apply version bump
        id: bump
        run: |
          cur=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
          IFS=. read -r MA MI PA <<< "$cur"
          case "${{ inputs.bump }}" in
            major) MA=$((MA+1)); MI=0; PA=0 ;;
            minor) MI=$((MI+1)); PA=0 ;;
            patch) PA=$((PA+1)) ;;
          esac
          NEW="$MA.$MI.$PA"
          echo "version=$NEW" >> "$GITHUB_OUTPUT"
          echo "tag=v$NEW" >> "$GITHUB_OUTPUT"
          cargo set-version "$NEW"
      - name: Generate CHANGELOG.md
        uses: orhun/git-cliff-action@v3
        with:
          config: cliff.toml
          args: --tag ${{ steps.bump.outputs.tag }} --output CHANGELOG.md
      - name: Generate release notes body
        uses: orhun/git-cliff-action@v3
        with:
          config: cliff.toml
          args: --tag ${{ steps.bump.outputs.tag }} --unreleased --strip all --output RELEASE_NOTES.md
      - name: Append unsigned-binary note to release body
        run: |
          cat >> RELEASE_NOTES.md <<'NOTE'

          ---
          **macOS:** the app is unsigned. On first launch, right-click
          `localllm.app` → **Open** to bypass Gatekeeper.
          **Windows:** the binary is unsigned; dismiss the SmartScreen prompt via
          **More info → Run anyway**.
          NOTE
      - name: Upload release notes
        uses: actions/upload-artifact@v4
        with:
          name: release-notes
          path: RELEASE_NOTES.md
      - name: Commit bump + tag + push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add Cargo.toml Cargo.lock CHANGELOG.md
          git commit -m "chore(release): ${{ steps.bump.outputs.tag }} [skip ci]"
          git tag -a "${{ steps.bump.outputs.tag }}" -m "${{ steps.bump.outputs.tag }}"
          git push origin HEAD
          git push origin "${{ steps.bump.outputs.tag }}"

  build:
    needs: prepare
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: macos-14
            backend: metal
            asset: macos-arm64
            kind: app
          - os: ubuntu-latest
            backend: cpu
            asset: linux-x64-cpu
            kind: tar
          - os: ubuntu-latest
            backend: cuda
            asset: linux-x64-cuda
            kind: tar
          - os: windows-latest
            backend: cpu
            asset: windows-x64-cpu
            kind: zip
          - os: windows-latest
            backend: cuda
            asset: windows-x64-cuda
            kind: zip
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ needs.prepare.outputs.tag }}
      - name: Install Linux system deps
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            libwebkit2gtk-4.1-dev libgtk-3-dev \
            libayatana-appindicator3-dev libxdo-dev cmake libclang-dev
      - name: Install CUDA toolkit
        if: matrix.backend == 'cuda'
        uses: Jimver/cuda-toolkit@v0.2.19
        with:
          cuda: '12.5.0'
          method: network
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          key: release-${{ matrix.os }}-${{ matrix.backend }}
      # macOS: build the .app bundle (build-app.sh runs cargo build --features metal).
      - name: Build macOS app
        if: matrix.kind == 'app'
        run: bash scripts/build-app.sh
      # Linux/Windows: build the release binary.
      - name: Build binary
        if: matrix.kind != 'app'
        run: cargo build --release --features ${{ matrix.backend }}
      # Package per OS.
      - name: Package macOS app (zip)
        if: matrix.kind == 'app'
        run: |
          cd target
          zip -r "localllm-v${{ needs.prepare.outputs.version }}-${{ matrix.asset }}.zip" localllm.app
      - name: Package Linux binary (tar.gz)
        if: matrix.kind == 'tar'
        run: |
          tar czf "localllm-v${{ needs.prepare.outputs.version }}-${{ matrix.asset }}.tar.gz" -C target/release localllm
      - name: Package Windows binary (zip)
        if: matrix.kind == 'zip'
        run: |
          Compress-Archive -Path target/release/localllm.exe -DestinationPath "localllm-v${{ needs.prepare.outputs.version }}-${{ matrix.asset }}.zip"
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.asset }}
          path: |
            target/localllm-v${{ needs.prepare.outputs.version }}-${{ matrix.asset }}.zip
            localllm-v${{ needs.prepare.outputs.version }}-${{ matrix.asset }}.tar.gz
            localllm-v${{ needs.prepare.outputs.version }}-${{ matrix.asset }}.zip
          if-no-files-found: ignore

  release:
    needs: [prepare, build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
      - name: Publish GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release create "${{ needs.prepare.outputs.tag }}" \
            --repo "${{ github.repository }}" \
            --title "${{ inputs.title }}" \
            --notes-file dist/release-notes/RELEASE_NOTES.md \
            $(find dist -type f ! -name RELEASE_NOTES.md)
```

- [ ] **Step 2: Validate YAML structure**

The workflow cannot run locally; validate its structure statically:
```bash
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/release.yml')); j=d['jobs']; assert set(j)=={'prepare','build','release'}, list(j); assert j['build']['needs']=='prepare'; assert j['release']['needs']==['prepare','build']; m=j['build']['strategy']['matrix']['include']; assert len(m)==5, len(m); assert {c['kind'] for c in m}=={'app','tar','zip'}; assert d['jobs']['prepare']; print('release.yml OK: 3 jobs, needs-wired, 5 build cells')"
```
Expected: `release.yml OK: 3 jobs, needs-wired, 5 build cells`.
(PyYAML parses the `on:` key as boolean `True`; the assertions avoid it.)

- [ ] **Step 3: Verify the bump arithmetic (pure logic)**

Confirm the semver step is correct in isolation (this is the one piece of real
logic in the workflow):
```bash
bash -c 'cur=0.1.0; IFS=. read -r MA MI PA <<< "$cur"; MI=$((MI+1)); PA=0; echo "$MA.$MI.$PA"'
```
Expected: `0.2.0` (a `minor` bump of `0.1.0`). Also spot-check patch: replace the
`MI`/`PA` lines with `PA=$((PA+1))` → expect `0.1.1`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "$(cat <<'EOF'
ci(fase-e3): release workflow (bump + changelog + 5-artifact publish)

workflow_dispatch(bump patch/minor/major + title): prepare job bumps
Cargo.toml via cargo set-version, regenerates CHANGELOG.md + release
notes via git-cliff, commits `chore(release): vX.Y.Z [skip ci]`, tags and
pushes. build matrix (5 configs) produces macos .app zip + linux/windows
cpu/cuda binaries. release job publishes a GitHub Release with the notes
and all artifacts. permissions: contents:write; concurrency: release.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

## Verification Summary (whole-branch)

- `build-app.sh` derives version from `Cargo.toml` (`grep` yields `0.1.0`; `bash -n` clean).
- `cliff.toml` parses (tomllib).
- `release.yml` parses; 3 jobs correctly `needs`-wired; 5-cell build matrix with `app`/`tar`/`zip` kinds.
- semver bump arithmetic verified (`0.1.0` +minor→`0.2.0`, +patch→`0.1.1`).

**Explicitly NOT verified in E3 (user's first dispatch):** the end-to-end run —
tag push, five artifact builds on real runners, and Release publication. Likely
first-dispatch iteration: artifact upload paths (the `upload-artifact` glob
lists all three extensions with `if-no-files-found: ignore` so each OS uploads
only its real file), the `cargo set-version`/git-cliff action pins, apt/CUDA
tuning inherited from E2, and confirming `gh release create`'s file-arg globbing.
These are tuned on the user's repo, not guessable locally. Unsigned-binary
Gatekeeper/SmartScreen behavior is documented in the release body, not solved.
