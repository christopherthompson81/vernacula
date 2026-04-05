#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------------------
# Parakeet Transcription — Linux desktop installer
# Usage: ./install.sh [--ep Cuda|Cpu] [--prefix <dir>]
# ---------------------------------------------------------------------------

EP="Cuda"
PREFIX="$HOME/.local/share/parakeet"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ep)     EP="$2";     shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--ep Cuda|Cpu] [--prefix <dir>]"
            echo "  --ep      Execution provider: Cuda (default) or Cpu"
            echo "  --prefix  Install directory (default: ~/.local/share/parakeet)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICON_SRC="$SCRIPT_DIR/src/Vernacula.Avalonia/Assets/parakeet.png"
ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
DESKTOP_FILE="$HOME/.local/share/applications/parakeet.desktop"

echo "Building Parakeet Transcription (EP=$EP)..."
dotnet publish "$SCRIPT_DIR/src/Vernacula.Avalonia/Vernacula.Avalonia.csproj" \
    -c Release \
    -p:EP="$EP" \
    -p:Platform=x64 \
    -r linux-x64 \
    --self-contained true \
    -o "$PREFIX" \
    --nologo \
    -v quiet

echo "Installing icon..."
mkdir -p "$ICON_DIR"
cp "$ICON_SRC" "$ICON_DIR/parakeet.png"

echo "Creating .desktop entry..."
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Parakeet Transcription
Comment=Local speech-to-text with speaker diarization
Exec=$PREFIX/Vernacula.Avalonia
Icon=parakeet
Categories=AudioVideo;Audio;
Terminal=false
EOF

echo "Refreshing desktop database..."
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo ""
echo "Done. Parakeet Transcription installed to $PREFIX"
echo "Launch from your application menu or run: $PREFIX/Vernacula.Avalonia"
