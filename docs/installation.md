# Installation

Vernacula runs on Linux, macOS, and Windows. The desktop app and CLI share the same runtime prerequisites — .NET 10, optionally FFmpeg (for the formats Vernacula does not decode itself), and optionally a GPU stack.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- FFmpeg — **optional**, and only for some formats. See [Audio formats](#audio-formats) below.

  Install it with your package manager (`sudo pacman -S ffmpeg`, `sudo apt install ffmpeg`, `sudo dnf install ffmpeg`, `brew install ffmpeg`), which gives you the `ffmpeg` and `ffprobe` executables and the shared libraries together. On Windows the desktop app can download a copy for you the first time you open a file that needs one, so there is nothing to install by hand.

  Both executables matter: `ffmpeg` decodes, and `ffprobe` is what the desktop app uses to find the audio streams in a video file. A build carrying only `ffmpeg` leaves video files failing while audio works.
- **For GPU acceleration:** NVIDIA GPU with the **CUDA 13** runtime installed (Linux/Windows), or DirectML support (Windows)

> ⚠ **CUDA 13, not 12.** The bundled ONNX Runtime (1.29) links `libcudart.so.13` / `cudart64_13.dll`;
> the major version is part of the library name, so a CUDA 12 installation cannot load it. Earlier
> releases of this project used ONNX Runtime 1.24, which linked CUDA 12. If you are upgrading and
> still have only CUDA 12, install the CUDA 13 runtime — or build with `-p:EP=Cpu`.
>
> Startup detection knows about the major version: a CUDA 12 machine reports CUDA as unavailable
> and says why, in the settings window and in `cuda_debug.txt`, rather than appearing to have GPU
> support that never engages.

## Audio formats

Vernacula decodes the most common formats itself, with no FFmpeg installed at all:

| Decoded in-process, everywhere | Needs FFmpeg |
|---|---|
| WAV (PCM / IEEE float) | FLAC |
| MP3 | M4A, AAC |
| AIFF | WMA |
| Ogg Vorbis (`.ogg`, `.oga`) | Non-PCM WAV (mu-law, A-law, ADPCM) |
| Ogg Opus (`.opus`, and Opus inside `.ogg`) | Every video container (MP4, MOV, MKV, WEBM, AVI, …) |

So a machine with no FFmpeg still transcribes voice memos, podcast downloads and messaging-app voice notes. FFmpeg is what you need for lossless archives, iPhone/Zoom recordings, and pulling the audio out of video.

**On Windows the desktop app offers to fetch it.** The first time you add a file it cannot decode itself, it downloads a pinned FFmpeg build (about 110 MB, verified by SHA-256) into `%LOCALAPPDATA%\Vernacula\tools\ffmpeg` and uses it from there. Nothing is added to `PATH` and nothing else on the system changes; deleting that directory undoes it. An `ffmpeg` already on `PATH` always wins, so managing it yourself keeps working exactly as before.

The command-line tools use the same copy if the app has downloaded one, but do not download it themselves — on Linux and macOS, install FFmpeg from your package manager.

To point Vernacula at a specific build, set `VERNACULA_FFMPEG_DIR` to the directory holding `ffmpeg` and `ffprobe`. It takes precedence over everything else.

Install FFmpeg on common Linux distros:

```bash
# Arch / Manjaro
sudo pacman -S ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

On macOS, `brew install ffmpeg` provides the libraries and both executables together.

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
