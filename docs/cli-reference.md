# CLI Reference

`vernacula-cli` is the command-line transcription tool. It runs the same pipeline as the desktop app (diarization → ASR → optional LID/LM fusion) and writes transcripts in Markdown, plain text, JSON, or SRT.

## Running

From a built binary:

```bash
vernacula-cli --audio <file> --model <dir> [options]
```

From source:

```bash
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio meeting.wav --model ~/models/vernacula
```

Build configurations (CUDA / CPU / DirectML) are covered in [Building from source](building.md).

## Arguments

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

## Examples

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

## See also

- [Pipeline backends and language support](reference/backends.md) — pick the right `--asr` backend for your language
- [Language model fusion (KenLM)](reference/language-model-fusion.md) — when to use `--lm`
- [Models](models.md) — download the model directories referenced by `--model`
