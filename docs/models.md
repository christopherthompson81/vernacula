# Models

Models are hosted on HuggingFace under [christopherthompson81](https://huggingface.co/christopherthompson81) and downloaded automatically by the Avalonia app. For CLI use, download them manually with `huggingface-cli` or `git lfs`.

## Catalogue

| Repository | Contents | Required? |
|---|---|---|
| [sortformer_parakeet_onnx](https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx) | Parakeet TDT v3 ASR + Sortformer diarization + Silero VAD | Core |
| [diarizen_onnx](https://huggingface.co/christopherthompson81/diarizen_onnx) | DiariZen higher-accuracy diarization | Optional |
| [cohere-transcribe-03-2026-onnx](https://huggingface.co/christopherthompson81/cohere-transcribe-03-2026-onnx) | Cohere Transcribe ASR backend | Optional |
| [vibevoice-asr-onnx](https://huggingface.co/christopherthompson81/vibevoice-asr-onnx) | VibeVoice-ASR (all-in-one transcription + diarization) | Optional, CUDA-only |
| [voxlingua107-lid-onnx](https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx) | VoxLingua107 language ID model | Optional |
| [kenlm-parakeet](https://huggingface.co/christopherthompson81/kenlm-parakeet) | Shallow-fusion KenLM n-gram models for Parakeet — `en-general`, `en-medical` | Optional |

## Manual download (CLI)

```bash
pip install huggingface_hub
huggingface-cli download christopherthompson81/sortformer_parakeet_onnx \
  --local-dir ~/models/vernacula
```

The `--model` argument to `vernacula-cli` points at the directory that holds these ONNX files.

## Which model feeds which backend

- **Parakeet TDT v3 / Sortformer / Silero VAD** — all inside `sortformer_parakeet_onnx`. This is the default pipeline.
- **DiariZen** — `diarizen_onnx`. Selected via `--diarization diarizen` (see [backends reference](reference/backends.md)).
- **Cohere Transcribe** — `cohere-transcribe-03-2026-onnx`. Selected via `--asr cohere`.
- **VibeVoice-ASR** — `vibevoice-asr-onnx`. Selected via `--asr vibevoice`. CUDA only.
- **VoxLingua107 LID** — `voxlingua107-lid-onnx`. Used by `--lid` and by per-segment language identification.
- **KenLM (`en-general`, `en-medical`)** — `kenlm-parakeet`. Supplied to Parakeet via `--lm`. See [language model fusion](reference/language-model-fusion.md).

The Avalonia app surfaces each repo through Settings → Models and handles downloads, resuming, and hash verification.

## First-launch sizes

See [Installation › First-launch model download](installation.md#first-launch-model-download) for approximate disk footprint per model.
