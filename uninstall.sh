#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------------------
# Vernacula-Desktop — Linux desktop uninstaller
# Usage: ./uninstall.sh [--prefix <dir>] [--purge]
# ---------------------------------------------------------------------------

PREFIX="$HOME/.local/share/vernacula-desktop"
PURGE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --purge)  PURGE=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--prefix <dir>] [--purge]"
            echo "  --prefix  Install directory to remove (default: ~/.local/share/vernacula-desktop)"
            echo "  --purge   Also remove user data: settings, models, and job history"
            echo "            (~/.local/share/Vernacula/)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ICON="$HOME/.local/share/icons/hicolor/256x256/apps/vernacula-desktop.png"
DESKTOP_FILE="$HOME/.local/share/applications/vernacula-desktop.desktop"

removed=0

if [[ -d "$PREFIX" ]]; then
    echo "Removing $PREFIX ..."
    rm -rf "$PREFIX"
    removed=1
fi

if [[ -f "$DESKTOP_FILE" ]]; then
    echo "Removing $DESKTOP_FILE ..."
    rm -f "$DESKTOP_FILE"
    removed=1
fi

if [[ -f "$ICON" ]]; then
    echo "Removing $ICON ..."
    rm -f "$ICON"
    removed=1
fi

if [[ $PURGE -eq 1 ]]; then
    USERDATA="$HOME/.local/share/Vernacula"
    if [[ -d "$USERDATA" ]]; then
        echo "Purging user data at $USERDATA ..."
        rm -rf "$USERDATA"
        removed=1
    fi
fi

if [[ $removed -eq 1 ]]; then
    echo "Refreshing desktop database..."
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
    echo "Done. Vernacula-Desktop uninstalled."
else
    echo "Nothing to uninstall at $PREFIX"
fi
