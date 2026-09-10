#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Vernacula-Desktop — macOS .app bundler
# Usage: ./package-macos.sh [--ep Cpu] [--out <dir>] [--self-contained]
#
# Produces Vernacula.app: the only form macOS gives a real Dock icon, a real
# application name in the menu bar, and a double-clickable launcher. A bare
# `dotnet run` gets none of those — the app sets its Dock icon at runtime
# (src/Vernacula.Avalonia/MacDockIcon.cs) precisely because it is usually not
# bundled during development.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ⚠ Cpu, not Cuda. There is no CUDA build of ONNX Runtime for arm64, and on macOS the
# plain package is also the one carrying the CoreML and WebGPU natives — "Cpu" names the
# package, not the providers you end up with. See docs/building.md.
EP="Cpu"
OUT_DIR="$SCRIPT_DIR/dist"
SELF_CONTAINED="false"
ICON_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ep)             EP="$2";      shift 2 ;;
        --out)            OUT_DIR="$2"; shift 2 ;;
        --self-contained) SELF_CONTAINED="true"; shift ;;
        --icon)           ICON_OVERRIDE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--ep Cpu] [--out <dir>] [--self-contained]"
            echo "  --ep              Execution provider (default: Cpu — the right one on Apple Silicon)"
            echo "  --out             Where to write Vernacula.app (default: ./dist)"
            echo "  --self-contained  Bundle the .NET runtime, so the app runs without dotnet installed"
            echo "  --icon            Square PNG to build the icon from (default: Assets/AppIcon.png)"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This packages a macOS .app and only runs on macOS." >&2
    exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    arm64)  RID="osx-arm64" ;;
    x86_64) RID="osx-x64"   ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

APP="$OUT_DIR/Vernacula.app"
CONTENTS="$APP/Contents"
ICON_SRC="${ICON_OVERRIDE:-$SCRIPT_DIR/src/Vernacula.Avalonia/Assets/AppIcon.png}"

echo "Building Vernacula.app (EP=$EP, $RID, self-contained=$SELF_CONTAINED)..."
rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

# -f is REQUIRED, not tidiness: Vernacula.Avalonia multi-targets net10.0;net10.0-windows
# and `dotnet publish` without -f fails with NETSDK1047. net10.0 is the portable flavour.
dotnet publish "$SCRIPT_DIR/src/Vernacula.Avalonia/Vernacula.Avalonia.csproj" \
    -c Release \
    -f net10.0 \
    -p:EP="$EP" \
    -r "$RID" \
    --self-contained "$SELF_CONTAINED" \
    -o "$CONTENTS/MacOS" \
    --nologo \
    -v quiet

# ── Icon ───────────────────────────────────────────────────────────────────
# macOS wants .icns, and iconutil wants a directory of specific sizes with specific names.
# The source is AppIcon.png: the same vern head, letterboxed onto a 1024x1024 transparent
# square. It is NOT vern.png directly -- that is 722x480, and a square icon slot stretches
# it (which is how a visibly squashed Dock icon first shipped). Regenerate it from vern.png
# with:
#   scale so the long edge is 1024, centre on a 1024x1024 transparent canvas.
echo "Generating icon..."

# ⚠ THE SOURCE MUST BE SQUARE. `sips -z h w` resizes to EXACT dimensions, so a non-square
# source is stretched, not letterboxed -- which is how vern.png (722x480) first shipped a
# visibly squashed Dock icon. Fail loudly instead of quietly distorting; pad the artwork to
# square yourself, or point --icon at something already square.
SRC_W="$(sips -g pixelWidth  "$ICON_SRC" | awk '/pixelWidth/{print $2}')"
SRC_H="$(sips -g pixelHeight "$ICON_SRC" | awk '/pixelHeight/{print $2}')"
if [[ "$SRC_W" != "$SRC_H" ]]; then
    echo "Icon source is ${SRC_W}x${SRC_H}, not square: macOS icons are square and this" >&2
    echo "would be stretched to fit. Pass --icon with a square PNG." >&2
    exit 1
fi

ICONSET="$(mktemp -d)/Vernacula.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z $size $size            "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png"      >/dev/null 2>&1
    sips -z $((size*2)) $((size*2)) "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$CONTENTS/Resources/Vernacula.icns"
rm -rf "$(dirname "$ICONSET")"

# ── Info.plist ─────────────────────────────────────────────────────────────
# NSMicrophoneUsageDescription is not optional theatre: macOS kills an app that
# touches audio input without it, and recording is a first-class feature here.
cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>              <string>Vernacula</string>
    <key>CFBundleDisplayName</key>       <string>Vernacula</string>
    <key>CFBundleIdentifier</key>        <string>com.christopherthompson81.vernacula</string>
    <key>CFBundleVersion</key>           <string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleExecutable</key>        <string>Vernacula.Avalonia</string>
    <key>CFBundleIconFile</key>          <string>Vernacula</string>
    <key>LSMinimumSystemVersion</key>    <string>12.0</string>
    <key>NSHighResolutionCapable</key>   <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Vernacula records audio so it can transcribe it.</string>
</dict>
</plist>
PLIST

chmod +x "$CONTENTS/MacOS/Vernacula.Avalonia"

# An unsigned bundle is quarantined when it arrives from anywhere but the local
# filesystem, and Gatekeeper's message ("damaged and can't be opened") sends people
# hunting for a corrupt download. Ad-hoc signing avoids that for a locally built app.
codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "Ad-hoc signed." \
    || echo "Note: ad-hoc signing failed; the app still runs locally."

echo
echo "Built $APP"
echo "  open \"$APP\"        # run it"
echo "  cp -r \"$APP\" /Applications/   # install it"
