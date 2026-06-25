#!/usr/bin/env bash
# scripts/build-app.sh — Build localllm.app for macOS
#
# Assembles a double-clickable .app bundle that:
#   - Has LSUIElement=true (no Dock icon, no terminal window)
#   - Launches the binary with --tray so the server starts as a menu-bar app
#   - Uses a small shell launcher as CFBundleExecutable so we can pass --tray
#     without modifying the compiled binary's defaults
#
# Usage:
#   bash scripts/build-app.sh
#
# Output: target/localllm.app
#
# Requirements:
#   - Rust + cargo installed
#   - macOS (uses plutil to validate the plist)
#
# Optional:
#   - cargo-bundle (https://github.com/burtonageo/cargo-bundle): if installed,
#     the script uses it; otherwise it assembles the .app manually.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="localllm"
BUNDLE_ID="dev.localllm.app"
APP_OUT="$REPO_ROOT/target/${APP_NAME}.app"

echo "==> Building release binary…"
cargo build --release

BINARY="$REPO_ROOT/target/release/${APP_NAME}"
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: release binary not found at $BINARY" >&2
    exit 1
fi
echo "    binary: $BINARY ($(du -h "$BINARY" | cut -f1))"

# ---------------------------------------------------------------------------
# Assemble .app manually (reliable; no cargo-bundle dep required)
# ---------------------------------------------------------------------------
echo "==> Assembling ${APP_NAME}.app…"

CONTENTS="$APP_OUT/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"

rm -rf "$APP_OUT"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Copy the real binary
cp "$BINARY" "$MACOS_DIR/${APP_NAME}"

# --- Generate the app icon (.icns) from the binary's built-in renderer ---
echo "==> Generating app icon…"
ICON_PNG="$(mktemp -t localllm-icon).png"
"$BINARY" --export-icon "$ICON_PNG"
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
for sz in 16 32 64 128 256 512 1024; do
    sips -z "$sz" "$sz" "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
done
# Provide the @2x names iconutil expects (reuse the larger renders).
cp "$ICONSET/icon_32x32.png"   "$ICONSET/icon_16x16@2x.png"
cp "$ICONSET/icon_64x64.png"   "$ICONSET/icon_32x32@2x.png"
cp "$ICONSET/icon_256x256.png" "$ICONSET/icon_128x128@2x.png"
cp "$ICONSET/icon_512x512.png" "$ICONSET/icon_256x256@2x.png"
cp "$ICONSET/icon_1024x1024.png" "$ICONSET/icon_512x512@2x.png"
rm -f "$ICONSET/icon_64x64.png" "$ICONSET/icon_1024x1024.png"
iconutil -c icns "$ICONSET" -o "$RESOURCES_DIR/AppIcon.icns"
echo "    icon: $RESOURCES_DIR/AppIcon.icns"

# Create a tiny shell launcher (CFBundleExecutable) that execs the real binary
# with --tray prepended to any extra args.
# This is the simplest way to default the .app to tray mode without changing
# the binary's own default (so headless CLI usage is unaffected).
LAUNCHER="$MACOS_DIR/${APP_NAME}-launch"
cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/sh
# localllm-launch — launcher script for localllm.app
# Executes the real binary with --tray so the .app runs as a background agent.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/localllm" --tray "$@"
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# Write Info.plist
# LSUIElement=true → background agent: no Dock icon, no app menu, no terminal.
# CFBundleExecutable → the launcher script (not the binary directly).
cat > "$CONTENTS/Info.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}-launch</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>localllm</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST_EOF

echo "==> Validating Info.plist with plutil…"
plutil -lint "$CONTENTS/Info.plist"
echo "    plist: OK"

echo "==> Bundle contents:"
find "$APP_OUT" -not -path "*/\.*" | sort | sed "s|$REPO_ROOT/||"

echo ""
echo "==> SUCCESS: $APP_OUT"
echo "    Double-click to launch as menu-bar app (no Dock, no terminal)."
echo "    LSUIElement=true  CFBundleExecutable=${APP_NAME}-launch (passes --tray)"
