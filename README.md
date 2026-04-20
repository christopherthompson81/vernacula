# Vernacula

<p align="center">
  A .NET 10 speech pipeline library and toolset for local, offline inference using ONNX models.<br/>
  No cloud. No telemetry. Runs entirely on your hardware.
</p>

<p>
  Construct audio pipelines including:
  <ul>
    <li>Denoising</li>
    <li>Segmentation (VAD, Diarization)</li>
    <li>Speaker Identification (Diarization)</li>
    <li>Speech Recognition (ASR) with multiple backends</li>
    <li>Language identification (VoxLingua107)</li>
    <li>Shallow KenLM fusion for domain-specific transcription</li>
  </ul>
</p>

---

## Vernacula Desktop App

<p align="center">
  <img src="src/Vernacula.Avalonia/Assets/vern_waveform.png" width="400" alt="Vernacula-Desktop" />
</p>

Vernacula-Desktop converts audio files into accurate, multi-speaker transcripts — entirely on your own computer. No cloud uploads, no subscriptions, no privacy concerns. Works on Linux, Mac, and Windows (Android, iOS, and WebAssembly are untested).

Powered by NVIDIA's [Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Sortformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v2.1) by default, with optional pluggable backends (Cohere Transcribe, Qwen3-ASR, VibeVoice-ASR). Parakeet v3 posts a **Word Error Rate of 4.85** on Google's FLEURS benchmark — among the best available anywhere. Most modern computers will transcribe one hour of audio in about five minutes. GPU-accelerated systems are significantly faster.

### Demo

https://github.com/user-attachments/assets/42015635-03b9-4c6b-868c-248e8c29c352

### Screenshots

**Jobs view** — queue and manage transcription jobs

![Jobs view](screenshots/jobs_view.png)

**Results view** — review the completed transcript with speaker labels and timestamps

![Results view](screenshots/results_view.png)

**Transcript editor** — correct text, adjust timing, and verify segments with audio playback

![Transcript editor view](screenshots/transcript_editor_view.png)

### Highlights

- **Local, private transcription** — audio never leaves your computer
- **Multi-speaker detection** — identifies and labels up to four concurrent speakers
- **No audio length limits** — streaming and segmentation handle indefinite file lengths
- **Job queue** — pause and resume long transcription jobs
- **Automatic punctuation and capitalization** from the acoustic model
- **Transcript editor** with confidence colouring, audio playback, word-level highlighting, and segment editing
- **Word-level timestamps** (real TDT duration-head values for Parakeet, synthesized for others)
- **Beam search + shallow KenLM fusion** for Parakeet — opt-in domain biasing via the Settings → Language model dropdown
- **Language identification** (VoxLingua107, optional) with file-level or per-segment modes
- **Wide format support** — common audio formats plus MP4, MOV, MKV, AVI, WMV, FLV, MTS, and more
- **Export** to XLSX, CSV, JSON, SRT, Markdown, DOCX, and SQLite
- **Full analysis data** in SQLite with per-token durations, logprobs, and speaker labels
- **GPU acceleration** via CUDA (DirectML on Windows), with automatic CPU fallback
- **Parakeet v3 covers 25 languages**: English, French, German, Spanish, Portuguese, Italian, Dutch, Polish, Russian, Ukrainian, Czech, Slovak, Romanian, Hungarian, Bulgarian, Croatian, Slovenian, Greek, Swedish, Danish, Finnish, Estonian, Latvian, Lithuanian, and Maltese
- **Qwen3-ASR and Cohere Transcribe backends** add ~30 and ~15 additional language options respectively

Built with [Avalonia UI](https://avaloniaui.net/) — runs on any desktop environment.

---

## Library Components

| Project | Description | License |
|---|---|---|
| `Vernacula.Base` | Core inference library — ASR, diarization, VAD, audio utilities, KenLM scorer | MIT |
| `Vernacula.CLI` | Command-line transcription tool | MIT |
| `Vernacula.Avalonia` | Desktop GUI app (Vernacula-Desktop) | PolyForm Shield 1.0.0 |

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- FFmpeg libraries (`libavformat`, `libavcodec`, `libavutil`, `libswresample`, `libswscale`)
- **For GPU acceleration:** NVIDIA GPU with CUDA Toolkit installed (Linux/Windows), or DirectML support (Windows)

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

Models are hosted on HuggingFace under [christopherthompson81](https://huggingface.co/christopherthompson81) and downloaded automatically by the Avalonia app. For CLI use, download them manually with `huggingface-cli` or `git lfs`.

| Repository | Contents | Required? |
|---|---|---|
| [sortformer_parakeet_onnx](https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx) | Parakeet TDT v3 ASR + Sortformer diarization + Silero VAD | Core |
| [diarizen_onnx](https://huggingface.co/christopherthompson81/diarizen_onnx) | DiariZen higher-accuracy diarization | Optional |
| [cohere-transcribe-03-2026-onnx](https://huggingface.co/christopherthompson81/cohere-transcribe-03-2026-onnx) | Cohere Transcribe ASR backend | Optional |
| [vibevoice-asr-onnx](https://huggingface.co/christopherthompson81/vibevoice-asr-onnx) | VibeVoice-ASR (all-in-one transcription + diarization) | Optional, CUDA-only |
| [voxlingua107-lid-onnx](https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx) | VoxLingua107 language ID model | Optional |
| [kenlm-parakeet](https://huggingface.co/christopherthompson81/kenlm-parakeet) | Shallow-fusion KenLM n-gram models for Parakeet — `en-general`, `en-medical` | Optional |

```bash
pip install huggingface_hub
huggingface-cli download christopherthompson81/sortformer_parakeet_onnx \
  --local-dir ~/models/vernacula
```

The Avalonia app surfaces each repo through Settings → Models and handles downloads, resuming, and hash verification.

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
| `DirectML` | Windows only | Uses DirectX 12; works on AMD/Intel/NVIDIA. |

### Vernacula.CLI

```bash
cd src/Vernacula.CLI

# GPU (CUDA)
dotnet build -c Release -p:EP=Cuda -p:Platform=x64

# CPU only
dotnet build -c Release -p:EP=Cpu -p:Platform=x64
```

### Vernacula.Avalonia

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
  --audio <path>                      Audio file to transcribe
  --model <dir>                       Directory containing ONNX model files

Output:
  --output <path>                     Output file path (auto-named if omitted)
  --export-format <md|txt|json|srt>   Output format (default: md)

ASR backend:
  --asr <parakeet|cohere|qwen3asr|vibevoice>   ASR backend (default: parakeet)
  --language <code>                   Force language for Cohere ASR (ISO 639-1: en, fr, de, ...)
  --cohere-model <dir>                Override Cohere model dir (default: <model>/cohere_transcribe)
  --qwen3asr-model <dir>              Override Qwen3-ASR model dir
  --vibevoice-model <dir>             Override VibeVoice-ASR model dir

Parakeet decoding:
  --precision <fp32|int8>             Model precision (default: fp32)
  --parakeet-beam <N>                 Beam width (default: 1 = greedy; 4–8 = beam search)
  --lm <path>                         Shallow LM fusion — ARPA(.gz) subword n-gram; auto-bumps beam to 4
  --lm-weight <w>                     Fusion weight (default: 0.3; typical 0.1–0.5)
  --lm-length-penalty <p>             Per-token length reward (default: 0.6; offsets LM shortening bias)

Segmentation:
  --diarization <backend>             sortformer (default), diarizen, vad, vibevoice-asr-builtin
  --segments <path>                   Load pre-computed segments JSON, skip diarization
  --ahc-threshold <float>             DiariZen AHC clustering threshold (default: 0.6)

Pre-processing:
  --denoiser <none|dfn3>              Optional DeepFilterNet3 denoiser (default: none)
  --denoiser-models <dir>             Override denoiser model dir

Other:
  --skip-asr                          Export diarization segments only
  --lid                               Run VAD + VoxLingua107 LID on --audio and print language
  --download-voxlingua                Download VoxLingua107 LID model and exit
  --benchmark                         Print timing and real-time factor (RTF)
  -h, --help                          Show full help
```

### Examples

```bash
# Basic Parakeet transcription with Sortformer diarization
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio meeting.wav --model ~/models/vernacula

# Parakeet + shallow KenLM fusion for medical dictation
dotnet run --project src/Vernacula.CLI -- \
  --audio clinic-note.wav --model ~/models/vernacula \
  --lm ~/models/kenlm-parakeet/en-medical.arpa.gz \
  --lm-weight 0.15

# Cohere Transcribe backend with forced French
dotnet run --project src/Vernacula.CLI -- \
  --audio interview.flac --model ~/models/vernacula \
  --asr cohere --language fr \
  --export-format srt --output interview.srt

# Language identification only
dotnet run --project src/Vernacula.CLI -- \
  --audio unknown.mp3 --model ~/models/vernacula --lid
```

---

## Pipeline Backends

### ASR backends

| Backend | Latency | Languages | Notes |
|---|---|---|---|
| **Parakeet TDT v3 (default)** | Fast | 25 European | Streaming-friendly TDT. Supports beam search + KenLM fusion. |
| **Cohere Transcribe (03-2026)** | Medium | 15 (auto-detect or forced) | Transformer encoder-decoder. |
| **Qwen3-ASR 1.7B** | Medium | ~30 | Multilingual. Can force language via prompt. |
| **VibeVoice-ASR** | Slow (CUDA only) | English, broad domain | All-in-one transcription with built-in diarization. |

### Diarization

| Backend | Speed | Accuracy | Overlap detection |
|---|---|---|---|
| Silero VAD | Fastest | No speaker identity | No |
| Sortformer v2-stream | Fast | Good | Yes (4-speaker max per chunk) |
| DiariZen | Slower | Better | Yes (powerset, 4-speaker max) |
| VibeVoice built-in | — | Bundled with VibeVoice-ASR | Yes |

### Execution providers

| EP | Platform | Notes |
|---|---|---|
| CUDA | Linux, Windows | Best performance on NVIDIA GPUs |
| CPU | All | Works everywhere; slower |
| DirectML | Windows only | AMD/Intel/NVIDIA via DirectX 12 |

---

## Language Model Fusion (Parakeet)

Vernacula ships a pure-C# shallow KenLM fusion path for the Parakeet TDT decoder. Selecting a language model in Settings auto-enables beam search and biases decoding toward the chosen domain — fixing the occasional multilingual-drift artifact Parakeet shows on conversational English and preserving specialty vocabulary.

Catalog currently published at [christopherthompson81/kenlm-parakeet](https://huggingface.co/christopherthompson81/kenlm-parakeet):

| LM | Size | Target |
|---|---|---|
| `en-general` | ~67 MB | Conversational English (GigaSpeech + People's Speech) |
| `en-medical` | ~17 MB | Medical English — clinical dictation + patient↔doctor dialogue + specialty drug names |

Each domain LM is built from speech-register sources only (spoken transcripts + synthetic dialogue + class-aware template-generated specialty vocabulary). Building your own is fully scripted under [`scripts/kenlm_build/`](scripts/kenlm_build/) — extract corpora from HuggingFace, tokenise with Parakeet's tokenizer, drive KenLM's `lmplz`, validate with the included scispaCy-based harness. See the [README there](scripts/kenlm_build/README.md) for the design notes on why each LM is layered the way it is.

---

## Linux Desktop Installation

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

> **Note:** The first launch opens a model download dialog. Approximate sizes:
> - Core (Parakeet TDT + Sortformer + VAD, fp32): ~3 GB
> - Core int8 quantized: ~820 MB
> - DiariZen add-on: ~310 MB
> - VoxLingua107 LID: ~100 MB
> - Cohere Transcribe: ~7 GB
> - VibeVoice-ASR: ~3 GB (CUDA-only)
> - KenLM models: 17–67 MB each (optional)
>
> All models are stored under `~/.local/share/Vernacula/models/`.

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

Parakeet TDT beam search (`--parakeet-beam 4`) adds roughly 3–5× to ASR latency. With KenLM fusion at typical weights the additional lookup cost is a few hundred milliseconds per clip (one-time LM load plus microsecond-per-beam-expansion scoring).

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
