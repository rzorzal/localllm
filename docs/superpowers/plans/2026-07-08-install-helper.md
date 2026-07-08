# Install Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship auto-detecting install scripts (`install.sh`, `install.ps1`) plus a README selector table so a user gets the right `localllm` build (metal/cuda/cpu) for their machine in one command.

**Architecture:** Two standalone scripts sharing one pipeline (detect OS/arch + Nvidia compute cap → resolve latest GitHub Release asset → install user-local), plus a README section. No application code changes.

**Tech Stack:** bash + curl (+ optional jq), PowerShell, GitHub Releases API.

## Global Constraints

- Repo is `rzorzal/localllm`; latest-release API `https://api.github.com/repos/rzorzal/localllm/releases/latest`.
- Five asset variants: `macos-arm64` (zip), `linux-x64-cpu`/`linux-x64-cuda` (tar.gz), `windows-x64-cpu`/`windows-x64-cuda` (zip).
- GPU rule: `nvidia-smi` present AND max compute cap **≥ 8.0** → cuda; else cpu. macOS → metal (`macos-arm64`), Apple Silicon only.
- Install user-local, NO sudo: macOS `~/Applications/localllm.app`; Linux `~/.local/bin/localllm`; Windows `%LOCALAPPDATA%\localllm` + user PATH.
- `--print` (bash) / `-Print` (ps1): detect + resolve + print, no download.
- Unsigned artifacts: print the Gatekeeper (right-click Open) / SmartScreen (More info → Run anyway) note.
- Commit trailers: every commit ends with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe`.
- Branch: `feat/fase-d1`.

---

### Task 1: `scripts/install.sh` (macOS + Linux)

**Files:**
- Create: `scripts/install.sh`

**Interfaces:**
- Consumes: nothing.
- Produces: an installer with a `--print` dry-run. README (Task 3) references its raw URL.

- [ ] **Step 1: Create `scripts/install.sh`**

Create the file with EXACTLY this content:
```bash
#!/usr/bin/env bash
# scripts/install.sh — download + install the right localllm build for this machine.
# Usage:  ./install.sh          install
#         ./install.sh --print  detect + resolve the asset, print, don't download
set -euo pipefail

REPO="rzorzal/localllm"
API="https://api.github.com/repos/${REPO}/releases/latest"

PRINT_ONLY=0
[ "${1:-}" = "--print" ] && PRINT_ONLY=1

# True if an Nvidia GPU with compute capability >= 8.0 (Ampere) is present.
have_ampere_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | tr -d ' ' | sort -rn | head -1)" || return 1
    [ -n "$cap" ] || return 1
    awk -v c="$cap" 'BEGIN { exit !(c + 0 >= 8.0) }'
}

os="$(uname -s)"
arch="$(uname -m)"
variant=""
ext=""
case "$os" in
    Darwin)
        if [ "$arch" != "arm64" ]; then
            echo "Only Apple Silicon (arm64) builds are published; your arch is $arch." >&2
            exit 1
        fi
        variant="macos-arm64"; ext="zip"
        ;;
    Linux)
        if [ "$arch" != "x86_64" ]; then
            echo "Only x86_64 Linux builds are published; your arch is $arch." >&2
            exit 1
        fi
        if have_ampere_gpu; then variant="linux-x64-cuda"; else variant="linux-x64-cpu"; fi
        ext="tar.gz"
        ;;
    *)
        echo "Unsupported OS: $os (this handles macOS/Linux; use install.ps1 on Windows)." >&2
        exit 1
        ;;
esac

# Resolve the download URL for the chosen variant from the latest release.
json="$(curl -fsSL -H 'Accept: application/vnd.github+json' "$API")" \
    || { echo "Failed to reach the GitHub API." >&2; exit 1; }
suffix="-${variant}.${ext}"
if command -v jq >/dev/null 2>&1; then
    url="$(printf '%s' "$json" \
        | jq -r --arg s "$suffix" '.assets[]?.browser_download_url | select(endswith($s))' \
        | head -1)"
else
    url="$(printf '%s' "$json" \
        | grep -o '"browser_download_url":[[:space:]]*"[^"]*"' \
        | sed 's/.*"\(https[^"]*\)"/\1/' \
        | grep -- "${suffix}\$" | head -1)"
fi
if [ -z "${url:-}" ]; then
    echo "No published asset for '${variant}' yet (no release, or asset missing)." >&2
    echo "See https://github.com/${REPO}/releases" >&2
    exit 1
fi

echo "Detected: ${os}/${arch} -> variant '${variant}'"
echo "Asset:    ${url}"
[ "$PRINT_ONLY" = "1" ] && exit 0

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
pkg="${tmp}/pkg.${ext}"
echo "Downloading..."
curl -fsSL "$url" -o "$pkg"

case "$os" in
    Darwin)
        dest="${HOME}/Applications"
        mkdir -p "$dest"
        rm -rf "${dest}/localllm.app"
        unzip -q "$pkg" -d "$dest"
        echo "Installed -> ${dest}/localllm.app"
        echo "Unsigned: on first launch, right-click the app -> Open to bypass Gatekeeper."
        ;;
    Linux)
        dest="${HOME}/.local/bin"
        mkdir -p "$dest"
        tar xzf "$pkg" -C "$dest" localllm
        chmod +x "${dest}/localllm"
        echo "Installed -> ${dest}/localllm"
        case ":${PATH}:" in
            *":${dest}:"*) : ;;
            *) echo "Note: ${dest} is not on your PATH. Add: export PATH=\"${dest}:\$PATH\"" ;;
        esac
        ;;
esac
echo "Done. Run: localllm --help"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/install.sh
```

- [ ] **Step 3: Syntax + lint**

```bash
bash -n scripts/install.sh && echo "syntax-ok"
command -v shellcheck >/dev/null 2>&1 && shellcheck scripts/install.sh || echo "shellcheck not installed — skipped"
```
Expected: `syntax-ok`, and shellcheck clean (or the skip line). Fix any shellcheck error that is a real bug (quoting, undefined var); style-only SC hints may be left if benign.

- [ ] **Step 4: Dry-run detection on this macOS box**

```bash
scripts/install.sh --print
```
Expected: two lines — `Detected: Darwin/arm64 -> variant 'macos-arm64'` and an `Asset:` line. The Asset line resolves only if a release exists; if none yet, the script prints the "No published asset" message and exits 1 — that is ACCEPTABLE for this step (it proves detection + API path work; note which outcome occurred in the report).

- [ ] **Step 5: Commit**

```bash
git add scripts/install.sh
git commit -m "$(cat <<'EOF'
feat(install): install.sh — auto-detect + install localllm (macOS/Linux)

Detects OS/arch and Nvidia compute cap (>=8.0 Ampere -> cuda, else cpu;
macOS -> metal .app), resolves the latest GitHub Release asset, installs
user-local (~/Applications or ~/.local/bin), no sudo. `--print` dry-runs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

### Task 2: `scripts/install.ps1` (Windows)

**Files:**
- Create: `scripts/install.ps1`

**Interfaces:**
- Consumes: nothing.
- Produces: the Windows installer with a `-Print` dry-run. README (Task 3) references its raw URL.

- [ ] **Step 1: Create `scripts/install.ps1`**

Create the file with EXACTLY this content:
```powershell
# scripts/install.ps1 — download + install the right localllm build (Windows).
# Usage:  .\install.ps1          install
#         .\install.ps1 -Print   detect + resolve the asset, print, don't download
param([switch]$Print)
$ErrorActionPreference = 'Stop'

$Repo = 'rzorzal/localllm'
$Api  = "https://api.github.com/repos/$Repo/releases/latest"
$Headers = @{ 'User-Agent' = 'localllm-install'; 'Accept' = 'application/vnd.github+json' }

function Get-Variant {
    $cuda = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        foreach ($c in $caps) {
            $v = 0.0
            if ([double]::TryParse($c.Trim(), [ref]$v) -and $v -ge 8.0) { $cuda = $true }
        }
    }
    if ($cuda) { 'windows-x64-cuda' } else { 'windows-x64-cpu' }
}

$variant = Get-Variant
$rel = Invoke-RestMethod -Uri $Api -Headers $Headers
$asset = $rel.assets | Where-Object { $_.name -like "*-$variant.zip" } | Select-Object -First 1
if (-not $asset) {
    Write-Error "No published asset for '$variant' yet. See https://github.com/$Repo/releases"
    exit 1
}
Write-Host "Detected variant: $variant"
Write-Host "Asset: $($asset.browser_download_url)"
if ($Print) { exit 0 }

$dest = Join-Path $env:LOCALAPPDATA 'localllm'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$zip = Join-Path $env:TEMP "localllm-$variant.zip"
Write-Host "Downloading..."
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $Headers
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip -Force

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$dest", 'User')
    Write-Host "Added $dest to your user PATH (restart the terminal to pick it up)."
}
Write-Host "Installed -> $dest\localllm.exe"
Write-Host "Unsigned: if SmartScreen warns, choose 'More info -> Run anyway'."
```

- [ ] **Step 2: Structural check (cannot execute on macOS)**

PowerShell is not available on the macOS dev box, so verify by inspection, not
execution: confirm the file contains the `param([switch]$Print)` line, the
`Get-Variant` function with the `>= 8.0` cuda rule, the `$asset` name match
`"*-$variant.zip"`, the no-release guard, the `-Print` early exit, the
`%LOCALAPPDATA%\localllm` destination, and the user-PATH append. Note in the
report that runtime verification happens on the user's Windows machine.

- [ ] **Step 3: Commit**

```bash
git add scripts/install.ps1
git commit -m "$(cat <<'EOF'
feat(install): install.ps1 — auto-detect + install localllm (Windows)

Detects Nvidia compute cap (>=8.0 Ampere -> cuda, else cpu), resolves the
latest GitHub Release asset, installs to %LOCALAPPDATA%\localllm + user
PATH, no admin. `-Print` dry-runs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

### Task 3: README Install section

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `scripts/install.sh` and `scripts/install.ps1` (their raw URLs), the variant names.
- Produces: user-facing install docs (terminal task).

- [ ] **Step 1: Add an Install section near the top of the README**

Find a sensible spot (after the intro / before the detailed config table). Insert
this section (place it wherever it reads best; the exact neighbor line is the
implementer's judgment, but it must be a new `## Install` section):
```markdown
## Install

**One-liner (auto-detects your GPU and installs the right build):**

- macOS / Linux:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/rzorzal/localllm/main/scripts/install.sh | bash
  ```
- Windows (PowerShell):
  ```powershell
  irm https://raw.githubusercontent.com/rzorzal/localllm/main/scripts/install.ps1 | iex
  ```

The scripts install user-local (no admin): `~/Applications` (macOS),
`~/.local/bin` (Linux), `%LOCALAPPDATA%\localllm` + your user PATH (Windows).
The binaries are unsigned — on macOS right-click the app → **Open**; on Windows
choose **More info → Run anyway** if SmartScreen warns.

**Or pick the download manually** from the [Releases page](https://github.com/rzorzal/localllm/releases):

| Your machine | Download |
|---|---|
| macOS (Apple Silicon) | `localllm-<ver>-macos-arm64.zip` (Metal) |
| Linux + Nvidia (Ampere / RTX 30xx and newer) | `localllm-<ver>-linux-x64-cuda.tar.gz` |
| Linux (older/no Nvidia GPU) | `localllm-<ver>-linux-x64-cpu.tar.gz` |
| Windows + Nvidia (Ampere / RTX 30xx and newer) | `localllm-<ver>-windows-x64-cuda.zip` |
| Windows (older/no Nvidia GPU) | `localllm-<ver>-windows-x64-cpu.zip` |

The CUDA builds target compute capability 8.0 (Ampere). On older Nvidia cards
(Turing/Pascal) use the CPU build. The GPU backend is chosen at build time —
there is no runtime auto-switch — so download the row that matches your machine.

> The `curl … | bash` / `irm … | iex` URLs point at `main`; they work once this
> branch is merged to `main` and a release has been published.
```

- [ ] **Step 2: Verify the table + links render**

```bash
grep -n "## Install" README.md && grep -c "localllm-<ver>-" README.md
```
Expected: the `## Install` line is found, and the `localllm-<ver>-` count is 5
(one per variant row).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): add Install section (one-liners + download selector table)

Auto-detect curl|bash / irm|iex one-liners plus a manual "which download"
table (metal/cuda-Ampere/cpu). Notes the compile-time backend choice and
the unsigned-binary workarounds.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WnAnMWB1LLTXrzgR94APFe
EOF
)"
```

---

## Verification Summary (whole-branch)

- `scripts/install.sh`: `bash -n` clean, executable, `--print` on macOS resolves `macos-arm64`.
- `scripts/install.ps1`: structurally complete (runtime is the user's Windows box).
- README `## Install`: one-liners + 5-row selector table.
- **Not verified locally (user, after first release):** the real download+install
  E2E on each OS, and the PowerShell runtime path. The `curl|bash`/`irm|iex` URLs
  need `feat/fase-d1` merged to `main` + a published release.
