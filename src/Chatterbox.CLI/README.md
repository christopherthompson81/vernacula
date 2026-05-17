# Chatterbox.CLI

One-shot command-line front end over
[`src/Chatterbox.Base/`](../Chatterbox.Base/). Synthesizes a single
utterance and writes a 24 kHz mono float32 WAV.

## Usage

```bash
# Inline text:
dotnet run --project src/Chatterbox.CLI -- \
    --onnx-dir /path/to/cb_dyn5 \
    --voice ~/Downloads/voice_prompt.wav \
    --text "Hello world. This is a fresh test sentence." \
    --out /tmp/out.wav

# Text from a file:
dotnet run --project src/Chatterbox.CLI -- \
    --onnx-dir /path/to/cb_dyn5 \
    --voice ~/Downloads/voice_prompt.wav \
    --text-file passage.txt \
    --out /tmp/out.wav
```

Once `dotnet publish -c Release` is run, the binary is named `chatterbox`
(see the `AssemblyName` in `Chatterbox.CLI.csproj`).

## Required flags

| Flag | Purpose |
|---|---|
| `--onnx-dir <dir>` | Directory containing the Chatterbox ONNX bundle produced by `scripts/chatterbox_export/export_chatterbox_to_onnx.py`. Must include `speech_encoder`, `embed_tokens`, `language_model`, and one of the cond-decoder layouts (merged Loop / split 3-graph / monolithic — auto-detected). |
| `--voice <wav>` | Reference voice clip. Any sample rate / channel count; resampled to 24 kHz mono internally. |
| `--text "..."` *or* `--text-file <path>` | Text to synthesize. Exactly one required. |

## Optional flags

| Flag | Default | Purpose |
|---|---|---|
| `--out <wav>` | `chatterbox_out.wav` | Output WAV path. |
| `--ep <name>` | `auto` | Execution provider — one of `auto` (CUDA → DirectML fallback), `cuda` (strict), `directml` (strict), or `cpu`. The csproj `-p:EP=...` build flag must include the runtime you ask for at runtime. |
| `--tokenizer-json <path>` | auto-locate | Path to chatterbox `tokenizer.json`. Auto-located from `~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/`. |
| `--io-binding` / `--no-io-binding` | auto-detect from effective EP | Force or disable GPU-resident KV-cache chaining for the LM. |
| `--exaggeration <float>` | `0.5` | Conditioning scalar passed to `embed_tokens`. Typical range 0.0–1.0; out-of-range values are accepted but produce increasingly unusual audio. |
| `--max-steps <int>` | `256` | Cap on LM rollout length. |
| `--verbose` / `-v` | off | Per-stage timing, cache state, effective-EP info per session. |

## Exit codes

- `0` — success
- `1` — runtime error (e.g. missing input file, no tokenizer.json found)
- `2` — bad arguments / usage error

## Sample output

Quiet (default):

```
Synthesized 6.92s of audio → /tmp/out.wav (7.0s total, 2.9s synth)
```

`--verbose`:

```
Loading ONNX bundle from /tmp/cb_dyn5 (ep=auto) ...
  speech_encoder.onnx: 1167 ms  cache=HIT  ep=cuda  src=1 MB
  embed_tokens.onnx: 32 ms  cache=HIT  ep=cuda  src=0 MB
  language_model.onnx: 860 ms  cache=HIT  ep=cuda  src=2048 MB
  conditional_decoder_loop.onnx: 1791 ms  cache=HIT  ep=cuda  src=168 MB
  vocoder mode: Merged
Loaded sessions in 3850 ms total  (requested-ep=auto)
Tokenized "Hello world. This is a fresh test sentence." → 14 tokens
speech_encoder: cond_emb=(1,33,1024)  audio_tokens=(1,250)
LM: 174 steps, generated 174 tokens
Synthesized 6.92s of audio → /tmp/out.wav (7.0s total, 2.9s synth)
```

## What this does not do (yet)

- **Markdown parsing.** `--text-file` reads the file as plain text; any
  markdown punctuation (`#`, `**`, list markers) will appear in the
  spoken output. Stripping markdown is Stage 1 step 4 in
  `chatterbox.scratch.md`.
- **Chunked synthesis / streaming.** The whole utterance is synthesized
  before writing the WAV. Long passages will hit the `--max-steps` cap
  and clip. Chunked synthesis and gapless concatenation are
  Stage 1 step 5+.
- **Forced alignment / word-level timestamps.** Not needed for the
  one-shot CLI; will be added when the GUI needs the highlight track.
