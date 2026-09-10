#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Vernacula-Desktop — macOS .app bundler
# Usage: ./package-macos.sh [--ep Cpu] [--out <dir>] [--icon <square.png>]
#                           [--framework-dependent]
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
ICON_OVERRIDE=""

# ⚠ Self-contained by DEFAULT, which is not the usual preference. A framework-dependent
# bundle launched from Finder inherits no PATH and finds .NET only via DOTNET_ROOT or the
# official installer's location — so on a machine where dotnet came from Homebrew,
# double-clicking the app fails with "you must install .NET" even though `dotnet run`
# works fine. An .app that cannot be double-clicked has missed its point.
SELF_CONTAINED="true"

usage() {
    echo "Usage: $0 [--ep Cpu] [--out <dir>] [--icon <square.png>] [--framework-dependent]"
    echo "  --ep                   Execution provider (default: Cpu — the right one on Apple Silicon)"
    echo "  --out                  Where to write Vernacula.app (default: ./dist)"
    echo "  --icon                 SQUARE PNG to build the icon from (default: Assets/AppIcon.png)"
    echo "  --framework-dependent  Do not bundle the .NET runtime; needs .NET installed where"
    echo "                         Finder can find it (see the note in this script)"
}

# `set -u` turns a missing option argument into "$2: unbound variable" rather than
# anything a user can act on.
need_arg() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Option $1 needs a value." >&2
        usage >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ep)                  need_arg "$@"; EP="$2";            shift 2 ;;
        --out)                 need_arg "$@"; OUT_DIR="$2";       shift 2 ;;
        --icon)                need_arg "$@"; ICON_OVERRIDE="$2"; shift 2 ;;
        --framework-dependent) SELF_CONTAINED="false"; shift ;;
        --help|-h)             usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
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

# ── Validate before destroying anything ────────────────────────────────────
# ⚠ EVERY CHECK BELONGS HERE, above the rm -rf and the publish. Validating the icon after
# them meant a typo'd --icon deleted the working bundle and burned a full Release build
# before saying so.
if [[ ! -f "$ICON_SRC" ]]; then
    echo "Icon source not found: $ICON_SRC" >&2
    exit 1
fi

# ⚠ THE SOURCE MUST BE SQUARE. `sips -z h w` resizes to EXACT dimensions, so a non-square
# source is stretched, not letterboxed -- which is how vern.png (722x480) first shipped a
# visibly squashed Dock icon. Fail loudly instead of quietly distorting.
SRC_W="$(sips -g pixelWidth  "$ICON_SRC" | awk '/pixelWidth/{print $2}')"
SRC_H="$(sips -g pixelHeight "$ICON_SRC" | awk '/pixelHeight/{print $2}')"
if [[ -z "$SRC_W" || -z "$SRC_H" ]]; then
    echo "Could not read image dimensions from $ICON_SRC — is it a PNG?" >&2
    exit 1
fi
if [[ "$SRC_W" != "$SRC_H" ]]; then
    echo "Icon source is ${SRC_W}x${SRC_H}, not square: macOS icons are square and this" >&2
    echo "would be stretched to fit. Pass --icon with a square PNG." >&2
    exit 1
fi

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

# Debug symbols do not belong in a distributable bundle, and codesign counts a .pdb as a
# nested code object it cannot sign -- so leaving them in fails the bundle seal outright.
find "$CONTENTS/MacOS" -type f -name "*.pdb" -delete

# ── Icon ───────────────────────────────────────────────────────────────────
# macOS wants .icns, and iconutil wants a directory of specific sizes with specific names.
# The source is AppIcon.png: the same vern head, letterboxed onto a 1024x1024 transparent
# square. It is NOT vern.png directly -- that is 722x480, and a square icon slot stretches
# it. Regenerate AppIcon.png from vern.png by scaling so the long edge is 1024 and centring
# it on a 1024x1024 transparent canvas.
echo "Generating icon..."
ICONSET_DIR="$(mktemp -d)"
ICONSET="$ICONSET_DIR/Vernacula.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z $size $size             "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png"    >/dev/null 2>&1
    sips -z $((size*2)) $((size*2)) "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$CONTENTS/Resources/Vernacula.icns"
rm -rf "$ICONSET_DIR"

# ── Info.plist ─────────────────────────────────────────────────────────────
# NSMicrophoneUsageDescription is not optional theatre: macOS kills an app that touches
# audio input without it, and recording is a first-class feature here.
# NSPrincipalClass is what makes LaunchServices treat this as a Cocoa app at all -- without
# it the menu-bar name and activation behaviour this script exists to fix stay wrong.
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
    <key>NSPrincipalClass</key>          <string>NSApplication</string>
    <key>LSMinimumSystemVersion</key>    <string>12.0</string>
    <key>NSHighResolutionCapable</key>   <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Vernacula records audio so it can transcribe it.</string>
</dict>
</plist>
PLIST

chmod +x "$CONTENTS/MacOS/Vernacula.Avalonia"

# ── Signing ────────────────────────────────────────────────────────────────
# Ad-hoc signing so a locally built app opens without complaint. Be clear about what this
# does and does not buy: an UNSIGNED bundle that has picked up the quarantine attribute is
# reported as "damaged and can't be opened", which sends people hunting for a corrupt
# download; ad-hoc signing avoids that. It is NOT notarization -- `spctl -a -t exec` still
# says "rejected", so a copy downloaded from anywhere will need right-click → Open or a
# real Developer ID. Locally built and copied to /Applications, it just works.
#
# ⚠ --deep, DESPITE Apple deprecating it for signing. The usual advice is to sign nested
# code explicitly, inside out, and that advice cannot work on this layout: a .NET publish
# puts the whole payload in Contents/MacOS/, codesign treats EVERY file there as a nested
# code object, and it refuses to sign the .json config files the host needs at runtime --
# `code object is not signed at all ... Vernacula.Avalonia.runtimeconfig.json`. Signing all
# the .dylib and .dll files first does not help; the JSON is still there and cannot go.
# Measured both ways: explicit fails, --deep verifies clean ("satisfies its Designated
# Requirement"). The real fix is a layout with only executables under MacOS/ and everything
# else in Resources/, which is a bigger change than an icon PR should make.
#
# ⚠ AD-HOC SIGNING CHANGES THE CODE HASH ON EVERY REBUILD, and TCC keys permissions to
# bundle ID + signature. So a rebuilt app can read as a different app and quietly lose its
# microphone grant, needing a manual remove/re-add under System Settings → Privacy &
# Security → Microphone. Nothing to be done short of a real signing identity; know it
# before blaming the recording code.

# Report the real error rather than a shrug: an unsigned bundle is the thing Gatekeeper
# calls "damaged", which is exactly what signing was meant to avoid.
if SIGN_ERR="$(codesign --force --deep --sign - "$APP" 2>&1)"; then
    echo "Ad-hoc signed."
else
    echo "⚠ Ad-hoc signing failed; the app runs locally but Gatekeeper may call it damaged" >&2
    echo "$SIGN_ERR" | sed 's/^/    /' >&2
fi

echo
echo "Built $APP"
echo "  open \"$APP\"                    # run it"
echo "  cp -R \"$APP\" /Applications/     # install it (-R, not -r: -r mangles bundles)"
if [[ "$SELF_CONTAINED" != "true" ]]; then
    echo
    echo "  ⚠ Built framework-dependent: double-clicking needs .NET installed somewhere"
    echo "    Finder can find it (DOTNET_ROOT, or the official installer's location)."
fi
