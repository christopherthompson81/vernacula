# Vernacula

<p align="center">
  A .NET 10 speech pipeline library and toolset for local, offline inference using ONNX models.<br/>
  No cloud. No telemetry. Runs entirely on your hardware.
</p>

<p align="center">
  <img src="src/Vernacula.Avalonia/Assets/vern_waveform.png" width="400" alt="Vernacula-Desktop" />
</p>

Vernacula converts audio into accurate, multi-speaker transcripts on your own computer. It ships as a reusable library (`Vernacula.Base`), a command-line tool (`Vernacula.CLI`), and a cross-platform desktop app (`Vernacula-Desktop`, built on Avalonia UI).

Powered by NVIDIA's [Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Sortformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v2.1) by default, with optional pluggable backends (Cohere Transcribe, Qwen3-ASR, VibeVoice-ASR). Parakeet v3 posts a **Word Error Rate of 4.85** on Google's FLEURS benchmark. Most modern computers will transcribe one hour of audio in about five minutes; GPU-accelerated systems are significantly faster.

## Demo

https://github.com/user-attachments/assets/42015635-03b9-4c6b-868c-248e8c29c352

![Results view](screenshots/results_view.png)

More screenshots and a feature tour live in [docs/desktop-app.md](docs/desktop-app.md).

## Highlights

- **Local, private transcription** — audio never leaves your computer
- **Multi-speaker detection** — identifies and labels up to four concurrent speakers
- **No audio length limits** — streaming and segmentation handle indefinite file lengths
- **Transcript editor** with confidence colouring, audio playback, and word-level timestamps
- **Pluggable ASR backends** — Parakeet TDT v3, Cohere Transcribe, Qwen3-ASR, VibeVoice-ASR
- **Shallow KenLM fusion** for domain-specific English (general, medical)
- **Export** to XLSX, CSV, JSON, SRT, Markdown, DOCX, and SQLite
- **GPU acceleration** via CUDA (DirectML on Windows), with automatic CPU fallback
- **52 languages** covered across the four backends — see the [support matrix](docs/reference/backends.md#language-support-matrix)

## Quick start

**Install prerequisites** — [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) plus FFmpeg. Full setup (including GPU) is in [docs/installation.md](docs/installation.md).

**Run the desktop app:**

```bash
cd src/Vernacula.Avalonia
dotnet run
```

On Linux, `./install.sh` from the repo root builds a self-contained package and registers a `.desktop` entry.

**Run the CLI:**

```bash
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio meeting.wav --model ~/models/vernacula
```

Full argument reference and more examples in [docs/cli-reference.md](docs/cli-reference.md). Build configurations (CUDA / CPU / DirectML) in [docs/building.md](docs/building.md).

## Documentation

Full documentation lives in [`docs/`](docs/).

**Getting started**
- [Installation](docs/installation.md) — .NET 10, FFmpeg, GPU prerequisites, Linux installer
- [Desktop app](docs/desktop-app.md) — features, screenshots, walkthrough
- [CLI reference](docs/cli-reference.md) — invocation, arguments, examples
- [Models](docs/models.md) — required and optional model downloads
- [Building from source](docs/building.md) — build configurations and publish guidance

**Reference**
- [Pipeline backends and language support](docs/reference/backends.md)
- [Language model fusion (KenLM)](docs/reference/language-model-fusion.md)
- [Benchmarks](docs/reference/benchmarks.md)

**Project**
- [Licensing](docs/licensing.md)
- [Developer notes](docs/dev/)

## License

- `Vernacula.Base` and `Vernacula.CLI` — [MIT](src/Vernacula.Base/LICENSE)
- `Vernacula.Avalonia` — [PolyForm Shield 1.0.0](src/Vernacula.Avalonia/LICENSE) (free to use and build; may not be used to create a competing commercial product)
- Model weights — see respective HuggingFace repository licenses

See [docs/licensing.md](docs/licensing.md) for the full breakdown.
