# Installation

Vernacula runs on Linux, macOS, and Windows. The desktop app and CLI share the same runtime prerequisites — .NET 10, FFmpeg native libraries, and (optionally) a GPU stack.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`, `libswresample`, `libswscale`)
- **For GPU acceleration:** NVIDIA GPU with CUDA Toolkit installed (Linux/Windows), or DirectML support (Windows)

Install FFmpeg on common Linux distros:

```bash
# Arch / Manjaro
sudo pacman -S ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

On macOS, install FFmpeg via Homebrew (`brew install ffmpeg`). On Windows, install the FFmpeg shared libraries and ensure they are on `PATH`.

## Linux desktop installer

Run the installer from the repo root:

```bash
./install.sh
```

The script publishes a self-contained build, installs the icon, creates a `.desktop` entry, and refreshes the desktop database. The app will appear in your application launcher under Audio/Video.

The default build targets CUDA but falls back to CPU automatically if no NVIDIA GPU is present — no flags needed. Pass `--ep Cpu` only if you want a smaller install without the CUDA runtime libraries.

To install to a custom location:

```bash
./install.sh --prefix /opt/vernacula-desktop
```

### First-launch model download

The first launch opens a model download dialog. Approximate sizes:

- Core (Parakeet TDT + Sortformer + VAD, fp32): ~3 GB
- Core int8 quantized: ~820 MB
- DiariZen add-on: ~310 MB
- VoxLingua107 LID: ~100 MB
- Cohere Transcribe: ~7 GB
- VibeVoice-ASR: ~3 GB (CUDA-only)
- KenLM models: 17–67 MB each (optional)

All models are stored under `~/.local/share/Vernacula/models/`.

## Next steps

- [Building from source](building.md) — if you need a specific execution provider or a non-Linux build
- [Models](models.md) — manual download instructions for the CLI
- [Desktop app](desktop-app.md) — feature tour
- [CLI reference](cli-reference.md) — arguments and examples
