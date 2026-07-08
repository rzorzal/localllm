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
        | tr -d ' ' | awk 'BEGIN { m = 0 } { v = $1 + 0; if (v > m) m = v } END { print m }')" || return 1
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

# Print the detected variant first, so --print shows it even if the API call
# below fails (e.g. no release published yet).
echo "Detected: ${os}/${arch} -> variant '${variant}'"

# Fetch the latest release, capturing the HTTP status separately from the body
# so a 404 (no release yet) is distinguishable from a real network failure.
resp="$(curl -sSL -w '\n%{http_code}' -H 'Accept: application/vnd.github+json' "$API")" \
    || { echo "Failed to reach the GitHub API (network error)." >&2; exit 1; }
code="${resp##*$'\n'}"
json="${resp%$'\n'*}"
if [ "$code" = "404" ]; then
    echo "No published release yet for ${REPO}. See https://github.com/${REPO}/releases" >&2
    exit 1
fi
[ "$code" = "200" ] || { echo "GitHub API returned HTTP ${code}." >&2; exit 1; }

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
    echo "No asset matching *${suffix} in the latest release of ${REPO}." >&2
    echo "See https://github.com/${REPO}/releases" >&2
    exit 1
fi

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
