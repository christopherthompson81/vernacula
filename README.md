# Vernacula

A .NET 10 speech pipeline library and toolset for local, offline inference using ONNX models. Supports automatic speech recognition (ASR), speaker diarization, and voice activity detection — no cloud, no telemetry.

Built around the [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) ASR model with pluggable backends for each pipeline stage.

## Components

| Project | Description | License |
|---|---|---|
| `Vernacula.Base` | Core inference library — ASR, diarization, VAD, audio utilities | MIT |
| `Vernacula.CLI` | Command-line transcription tool | MIT |
| `Vernacula.Avalonia` | Desktop GUI app for Linux (Parakeet Transcription) | PolyForm Shield 1.0.0 |

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`, `libswresample`, `libswscale`)
- **For GPU acceleration:** NVIDIA GPU with CUDA Toolkit installed

Install FFmpeg on common distros:

```bash
# Arch / Manjaro
sudo pacman -S ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

## Models

Models are hosted on HuggingFace and downloaded automatically by the Avalonia app. For CLI use, download them manually:

- **Core models** (ASR + Sortformer diarization + VAD): [christopherthompson81/sortformer_parakeet_onnx](https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx)
- **DiariZen models** (optional, advanced diarization): [christopherthompson81/diarizen_onnx](https://huggingface.co/christopherthompson81/diarizen_onnx)

Download with `huggingface-cli` or `git lfs`:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

huggingface-cli download christopherthompson81/sortformer_parakeet_onnx --local-dir ~/models/parakeet
```

---

## Building

All projects are built with `dotnet build`. The `EP` property selects the ONNX Runtime execution provider:

| `-p:EP=` | Hardware | Notes |
|---|---|---|
| `Cuda` | NVIDIA GPU | Default. Requires CUDA Toolkit. |
| `Cpu` | Any CPU | No GPU required. Slower. |
| `DirectML` | Windows only | Not supported on Linux. |

### Vernacula.CLI

```bash
cd src/Vernacula.CLI

# GPU (CUDA)
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# CPU only
dotnet build -c Release -p:EP=Cpu -p:Platform=x64
```

### Vernacula.Avalonia (Linux Desktop)

```bash
cd src/Vernacula.Avalonia

# Build
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# Or publish as self-contained (recommended for desktop install)
dotnet publish -c Release -p:EP=Cuda -p:Platform=x64 \
  -r linux-x64 --self-contained true \
  -o ~/apps/parakeet
```

---

## CLI Usage

```
Usage: vernacula-cli --audio <file> --model <dir> [options]

Required:
  --audio <path>           Audio file to transcribe
  --model <dir>            Directory containing ONNX model files

Output:
  --output <path>          Output file path (auto-named if omitted)
  --export-format <fmt>    Output format: md (default), txt, json, srt

Diarization:
  --diarization <backend>  Speaker diarization backend:
                             sortformer  NVIDIA Sortformer (default, fast)
                             diarizen    DiariZen clustering (more accurate, slower)
                             vad         Silero VAD only (no speaker identity)
  --segments <path>        Load pre-computed segments JSON, skip diarization
                           Format: [{start, end, speaker}, ...]
  --ahc-threshold <float>  DiariZen AHC clustering threshold (default: 0.6)

Model:
  --precision <fp32|int8>  Model precision (default: fp32)
  --skip-asr               Export diarization segments only, skip transcription

Diagnostics:
  --benchmark              Print timing and real-time factor (RTF) after run
```

### Examples

```bash
# Basic transcription with Sortformer diarization
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio meeting.wav --model ~/models/parakeet

# SRT output using DiariZen
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio interview.flac --model ~/models/parakeet \
  --diarization diarizen --export-format srt --output interview.srt

# CPU-only, int8 quantized models
dotnet run --project src/Vernacula.CLI -p:EP=Cpu -- \
  --audio recording.wav --model ~/models/parakeet --precision int8
```

---

## Linux Desktop Installation (Avalonia)

### 1. Publish a self-contained build

```bash
cd src/Vernacula.Avalonia

dotnet publish -c Release -p:EP=Cuda -p:Platform=x64 \
  -r linux-x64 --self-contained true \
  -o ~/.local/share/parakeet
```

For CPU-only systems, replace `-p:EP=Cuda` with `-p:EP=Cpu`.

### 2. Copy the icon

```bash
mkdir -p ~/.local/share/icons/hicolor/256x256/apps
cp Assets/parakeet.png ~/.local/share/icons/hicolor/256x256/apps/parakeet.png
```

### 3. Create a .desktop entry

Create `~/.local/share/applications/parakeet.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Parakeet Transcription
Comment=Local speech-to-text with speaker diarization
Exec=/home/YOUR_USERNAME/.local/share/parakeet/Vernacula.Avalonia
Icon=parakeet
Categories=AudioVideo;Audio;
Terminal=false
```

Replace `YOUR_USERNAME` with your actual username, or use `$HOME`:

```bash
cat > ~/.local/share/applications/parakeet.desktop << EOF
[Desktop Entry]
Type=Application
Name=Parakeet Transcription
Comment=Local speech-to-text with speaker diarization
Exec=$HOME/.local/share/parakeet/Vernacula.Avalonia
Icon=parakeet
Categories=AudioVideo;Audio;
Terminal=false
EOF
```

### 4. Refresh the desktop database

```bash
update-desktop-database ~/.local/share/applications
gtk-update-icon-cache ~/.local/share/icons/hicolor
```

The app should now appear in your DE's application launcher under Audio/Video.

> **Note:** The first launch opens a model download dialog. Models total ~1.5 GB (core) or ~2.5 GB (with DiariZen). They are stored in `~/.config/Parakeet/models/`.

---

## DiariZen Environment Variables

DiariZen's segmentation and embedding pipeline can be tuned via environment variables for your hardware:

| Variable | Description |
|---|---|
| `PARAKEET_DIARIZEN_SEG_THREADS` | Segmentation intra-op thread count |
| `PARAKEET_DIARIZEN_SEG_MAX_WORKERS` | Max parallel segmentation workers |
| `PARAKEET_DIARIZEN_SEG_BATCH_SIZE` | Segmentation batch size |
| `PARAKEET_DIARIZEN_EMBED_THREADS` | Embedding intra-op thread count |
| `PARAKEET_DIARIZEN_EMBED_MAX_WORKERS` | Max parallel embedding workers |
| `PARAKEET_DIARIZEN_EMBED_GPU_SAFETY_MB` | GPU memory safety margin (MB) |
| `PARAKEET_DIARIZEN_EMBED_GPU_MAX_BATCH_SIZE` | Max embedding batch size |
| `PARAKEET_DIARIZEN_EMBED_GPU_MAX_BATCH_FRAMES` | Max frames per embedding batch |

---

## Pipeline Backends

### ASR
- **NVIDIA Parakeet TDT 0.6B** — CTC/Transducer hybrid, English, streaming-friendly

### Diarization
| Backend | Speed | Accuracy | Overlap detection |
|---|---|---|---|
| Sortformer | Fast | Good | Yes (4-speaker max per chunk) |
| DiariZen | Slower | Better | Yes (powerset, 4-speaker max) |
| Silero VAD | Fastest | None (no identity) | No |

### Execution Providers
| EP | Platform | Notes |
|---|---|---|
| CUDA | Linux, Windows | Best performance on NVIDIA GPUs |
| CPU | All | Works everywhere, no GPU needed |
| DirectML | Windows only | AMD/Intel/NVIDIA via DirectX 12 |

---

## License

- `Vernacula.Base` and `Vernacula.CLI` — [MIT](src/Vernacula.Base/LICENSE)
- `Vernacula.Avalonia` — [PolyForm Shield 1.0.0](src/Vernacula.Avalonia/LICENSE) (free to use and build; may not be used to create a competing commercial product)
- Model weights — see respective HuggingFace repository licenses
