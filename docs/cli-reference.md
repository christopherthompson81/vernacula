# CLI Reference

`vernacula-cli` is the command-line transcription tool. It runs the same pipeline as the desktop app (diarization → ASR → optional LID/LM fusion) and writes transcripts in Markdown, plain text, JSON, or SRT.

## Running

From a built binary:

```bash
vernacula-cli --audio <file> [--models-dir <dir>] [options]
```

With no `--models-dir`, the models root defaults to the desktop app's own location
(`~/.local/share/Vernacula/models` on Linux, `%LOCALAPPDATA%\Vernacula\models` on Windows), so
models downloaded in the app are found without any flag.

From source:

```bash
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- --audio meeting.wav
```

Build configurations (CUDA / CPU / DirectML) are covered in [Building from source](building.md).

## Arguments

```
Usage: vernacula-cli --audio <file> [--models-dir <dir>] [options]

Required:
  --audio <path>                      Audio file to transcribe

Models:
  --models-dir <dir>                  Models root, one subdirectory per backend (parakeet/,
                                      silero/, sortformer/, granite_speech_4_1_2b/, ...) — the
                                      same layout the desktop app uses.
                                      Default: <LocalApplicationData>/Vernacula/models
  --model <dir>                       Older spelling of --models-dir. Also accepts a flat
                                      bundle directory with no per-backend subdirectories,
                                      which is how it used to be passed for Parakeet.

Output:
  --output <path>                     Output file path (auto-named if omitted)
  --export-format <md|txt|json|srt>   Output format (default: md)

ASR backend:
  --asr <parakeet|cohere|qwen3asr|vibevoice|vibevoice-streaming|whisper|granite>
                                      ASR backend (default: parakeet)
  --language <code>                   Force language for Cohere ASR (ISO 639-1: en, fr, de, ...)
  --cohere-model <dir>                Override Cohere model dir (default: <models-dir>/cohere_transcribe)
  --qwen3asr-model <dir>              Override Qwen3-ASR model dir
  --vibevoice-model <dir>             Override VibeVoice-ASR model dir
  --vibevoice-streaming-model <dir>   Override VibeVoice-ASR Streaming model dir
  --hotwords <a,b,c>                  Bias VibeVoice-ASR Streaming toward these names or terms

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

Other:
  --skip-asr                          Export diarization segments only
  --lid                               Run VAD + VoxLingua107 LID on --audio and print language
  --download-voxlingua                Download VoxLingua107 LID model and exit
  --benchmark                         Print timing and real-time factor (RTF)
  -h, --help                          Show full help
```

## Examples

```bash
# Basic Parakeet transcription with Sortformer diarization, using the default models root
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- \
  --audio meeting.wav

# Parakeet + shallow KenLM fusion for medical dictation
dotnet run --project src/Vernacula.CLI -- \
  --audio clinic-note.wav \
  --lm ~/models/kenlm-parakeet/en-medical.arpa.gz \
  --lm-weight 0.15

# Cohere Transcribe backend with forced French
dotnet run --project src/Vernacula.CLI -- \
  --audio interview.flac \
  --asr cohere --language fr \
  --export-format srt --output interview.srt

# Language identification only
dotnet run --project src/Vernacula.CLI -- \
  --audio unknown.mp3 --lid
```

## See also

- [Pipeline backends and language support](reference/backends.md) — pick the right `--asr` backend for your language
- [Language model fusion (KenLM)](reference/language-model-fusion.md) — when to use `--lm`
- [Models](models.md) — download the model directories referenced by `--model`
