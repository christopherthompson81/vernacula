# Vernacula

<p align="center">
  A .NET 10 speech pipeline library and toolset for local, offline inference using ONNX models.<br/>
  No cloud. No telemetry. Runs entirely on your hardware.
</p>

<p>
  Construct audio pipelines including:
  <ul>
    <li>Denoising</li>
    <li>Segmetation (VAD, Diarization)</li>
    <li>Speaker Identification (Diarization)</li>
    <li>Speech Recognition (ASR)</li>
  </ul>
</p>

---

## Vernacula Desktop App

<p align="center">
  <img src="src/Vernacula.Avalonia/Assets/vern_waveform.png" width="400" alt="Vernacula-Desktop" />
</p>

Vernacula-Desktop converts audio files into accurate, multi-speaker transcripts — entirely on your own computer. No cloud uploads, no subscriptions, no privacy concerns.

Powered by NVIDIA's [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and [Sortformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v2.1) models, it delivers a **Word Error Rate of 4.85** on Google's FLEURS benchmark — among the best available anywhere. Most modern computers will transcribe one hour of audio in about five minutes. GPU-accelerated systems will be even faster.

<!-- Screenshots -->
<!-- Add screenshots here once available -->

- Local, private transcription — audio never leaves your computer
- Multi-speaker detection — identifies and labels up to four speakers
- No audio length limits — streaming and segmentation handle indefinite file lengths
- Queue multiple files; pause and resume long transcription jobs
- Automatic punctuation and capitalization
- Transcript editor with confidence colouring, audio playback, and segment editing
- Wide format support — common audio formats plus MP4, MOV, MKV, AVI, WMV, FLV, MTS, and more
- Export to XLSX, CSV, JSON, SRT, Markdown, DOCX, and SQLite3
- Full analysis data in SQLite3 with word-level timestamps and confidence scoring
- GPU acceleration via CUDA, with automatic CPU fallback
- Supports 25 European languages: English, French, German, Spanish, Portuguese, Italian, Dutch, Polish, Russian, Ukrainian, Czech, Slovak, Romanian, Hungarian, Bulgarian, Croatian, Slovenian, Greek, Swedish, Danish, Finnish, Estonian, Latvian, Lithuanian, and Maltese

Built with [Avalonia UI](https://avaloniaui.net/) — runs on any Linux desktop environment.

---

## Library Components

| Project | Description | License |
|---|---|---|
| `Vernacula.Base` | Core inference library — ASR, diarization, VAD, audio utilities | MIT |
| `Vernacula.CLI` | Command-line transcription tool | MIT |
| `Vernacula.Avalonia` | Desktop GUI app for Linux (Vernacula-Desktop) | PolyForm Shield 1.0.0 |

Built around the [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) ASR model with pluggable backends for each pipeline stage.

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

huggingface-cli download christopherthompson81/sortformer_parakeet_onnx --local-dir ~/models/vernacula
```

---

## Running Vernacula-Desktop
```bash
cd src/Vernacula.Avalonia

dotnet run 
```



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
  -o ~/apps/vernacula-desktop
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
  --audio meeting.wav --model ~/models/vernacula

# SRT output using DiariZen
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio interview.flac --model ~/models/vernacula \
  --diarization diarizen --export-format srt --output interview.srt

# CPU-only, int8 quantized models
dotnet run --project src/Vernacula.CLI -p:EP=Cpu -- \
  --audio recording.wav --model ~/models/vernacula --precision int8
```

---

## Linux Desktop Installation (Avalonia)

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

> **Note:** The first launch opens a model download dialog. Model sizes:
> - Core fp32: ~3 GB (encoder data file is 2.44 GB)
> - Core int8: ~820 MB
> - DiariZen add-on: ~310 MB
>
> Models are stored in `~/.local/share/Vernacula/models/`.

---

## DiariZen Environment Variables

DiariZen's segmentation and embedding pipeline can be tuned via environment variables for your hardware:

| Variable | Description |
|---|---|
| `VERNACULA_DIARIZEN_SEG_THREADS` | Segmentation intra-op thread count |
| `VERNACULA_DIARIZEN_SEG_MAX_WORKERS` | Max parallel segmentation workers |
| `VERNACULA_DIARIZEN_SEG_BATCH_SIZE` | Segmentation batch size |
| `VERNACULA_DIARIZEN_EMBED_THREADS` | Embedding intra-op thread count |
| `VERNACULA_DIARIZEN_EMBED_MAX_WORKERS` | Max parallel embedding workers |
| `VERNACULA_DIARIZEN_EMBED_GPU_SAFETY_MB` | GPU memory safety margin (MB) |
| `VERNACULA_DIARIZEN_EMBED_GPU_MAX_BATCH_SIZE` | Max embedding batch size |
| `VERNACULA_DIARIZEN_EMBED_GPU_MAX_BATCH_FRAMES` | Max frames per embedding batch |

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

## Benchmarks

### Throughput

10-minute English audio file, fp32 models. RTF < 1.0 = faster than real-time.

| Backend | Hardware | Diarization | ASR | Total | RTF |
|---|---|---|---|---|---|
| Silero VAD | AMD Ryzen 7 7840U | 2.1s | 50.5s | 52.7s | **0.088** |
| Sortformer | AMD Ryzen 7 7840U | 33.2s | 49.2s | 82.4s | **0.137** |
| DiariZen | AMD Ryzen 7 7840U | 502.0s | 55.8s | 557.8s | 0.930 |
| Silero VAD | NVIDIA RTX 3090 | 2.1s | 5.4s | 7.4s | **0.012** |
| Sortformer | NVIDIA RTX 3090 | 16.0s | 5.5s | 21.4s | **0.036** |
| DiariZen | NVIDIA RTX 3090 | 16.8s | 5.4s | 22.2s | **0.037** |

> DiariZen's segmentation and embedding pipeline is heavily GPU-accelerated — CUDA reduces diarization time from 502s to 16.8s (~30×) and brings total runtime in line with Sortformer.

### Accuracy (DER)

Diarization Error Rate from published benchmarks. Lower is better.

| Backend | AMI-SDM | VoxConverse | DIHARD III | Source |
|---|---|---|---|---|
| Sortformer v2-stream | 20.6% | 13.9% | 20.2% | [HuggingFace](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) |
| DiariZen-Large | 13.9% | 9.1% | 14.5% | [BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen) |

> Benchmarks use different evaluation conditions (collar, overlap handling) — direct cross-model comparison should be treated as indicative only. The independent survey [Benchmarking Diarization Models (2509.26177)](https://arxiv.org/abs/2509.26177) found Sortformer v2-stream and DiariZen among the top open-source performers overall.

---

## License

- `Vernacula.Base` and `Vernacula.CLI` — [MIT](src/Vernacula.Base/LICENSE)
- `Vernacula.Avalonia` — [PolyForm Shield 1.0.0](src/Vernacula.Avalonia/LICENSE) (free to use and build; may not be used to create a competing commercial product)
- Model weights — see respective HuggingFace repository licenses
