# Installation

Vernacula runs on Linux, macOS, and Windows. The desktop app and CLI share the same runtime prerequisites — .NET 10, FFmpeg native libraries, and (optionally) a GPU stack.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- FFmpeg, both parts of it:
  - the **libraries** (`libavformat`, `libavcodec`, `libavutil`, `libswresample`, `libswscale`), used in-process by the desktop app
  - the **`ffmpeg` executable on `PATH`**, used by the CLI tools to decode anything that is not PCM/IEEE-float WAV (MP3, FLAC, M4A, AAC, OGG, Opus, mu-law/A-law/ADPCM WAV)

  A distro package or a Homebrew install gives you both. A libraries-only install builds and runs the desktop app but leaves the CLIs unable to read compressed audio.
- **For GPU acceleration:** NVIDIA GPU with the **CUDA 13** runtime installed (Linux/Windows), or DirectML support (Windows)

> ⚠ **CUDA 13, not 12.** The bundled ONNX Runtime (1.29) links `libcudart.so.13` / `cudart64_13.dll`;
> the major version is part of the library name, so a CUDA 12 installation cannot load it. Earlier
> releases of this project used ONNX Runtime 1.24, which linked CUDA 12. If you are upgrading and
> still have only CUDA 12, install the CUDA 13 runtime — or build with `-p:EP=Cpu`.
>
> Startup detection knows about the major version: a CUDA 12 machine reports CUDA as unavailable
> and says why, in the settings window and in `cuda_debug.txt`, rather than appearing to have GPU
> support that never engages.

Install FFmpeg on common Linux distros:

```bash
# Arch / Manjaro
sudo pacman -S ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

On macOS, install FFmpeg via Homebrew (`brew install ffmpeg`), which provides the libraries and the executable together. On Windows, install an FFmpeg build that contains both the shared libraries and `ffmpeg.exe`, and ensure the directory holding them is on `PATH` — a libraries-only package leaves the CLI tools unable to decode compressed audio.

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
- VibeVoice-ASR Streaming: ~3.2 GB for the 1.5B, ~9 GB for the 7B (CUDA-only; pick one in Settings)
- KenLM models: 17–67 MB each (optional)

All models are stored under `~/.local/share/Vernacula/models/`.

## Next steps

- [Building from source](building.md) — if you need a specific execution provider or a non-Linux build
- [Models](models.md) — manual download instructions for the CLI
- [Desktop app](desktop-app.md) — feature tour
- [CLI reference](cli-reference.md) — arguments and examples
