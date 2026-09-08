#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------------------
# Vernacula-Desktop — Linux desktop installer
# Usage: ./install.sh [--ep Cuda|Cpu] [--prefix <dir>]
# ---------------------------------------------------------------------------

EP="Cuda"
PREFIX="$HOME/.local/share/vernacula-desktop"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ep)     EP="$2";     shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--ep Cuda|Cpu] [--prefix <dir>]"
            echo "  --ep      Execution provider: Cuda (default) or Cpu"
            echo "  --prefix  Install directory (default: ~/.local/share/vernacula-desktop)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICON_SRC="$SCRIPT_DIR/src/Vernacula.Avalonia/Assets/vern.png"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
DESKTOP_FILE="$HOME/.local/share/applications/vernacula-desktop.desktop"

echo "Building Vernacula-Desktop (EP=$EP)..."
# -f is REQUIRED, not tidiness: Vernacula.Avalonia multi-targets net10.0;net10.0-windows
# (NAudio 3 only hands WinMM to a Windows TFM — see the csproj), and `dotnet publish` on a
# multi-targeted project with no -f fails with NETSDK1047. This is the Linux installer, so
# net10.0 is the flavour it wants: the WaveOut path is compiled out and ffplay is the backend.
dotnet publish "$SCRIPT_DIR/src/Vernacula.Avalonia/Vernacula.Avalonia.csproj" \
    -c Release \
    -f net10.0 \
    -p:EP="$EP" \
    -p:Platform=x64 \
    -r linux-x64 \
    --self-contained true \
    -o "$PREFIX" \
    --nologo \
    -v quiet

echo "Installing icon..."
mkdir -p "$ICON_DIR"
cp "$ICON_SRC" "$ICON_DIR/vernacula-desktop.png"

echo "Creating .desktop entry..."
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Vernacula-Desktop
Comment=Local speech-to-text with speaker diarization
Exec=$PREFIX/Vernacula.Avalonia
Icon=vernacula-desktop
Categories=AudioVideo;Audio;
Terminal=false
EOF

echo "Refreshing desktop database..."
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo ""
echo "Done. Vernacula-Desktop installed to $PREFIX"
echo "Launch from your application menu or run: $PREFIX/Vernacula.Avalonia"
