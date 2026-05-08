# Granite Speech 4.1 — C# CLI parity smoke

Console app that drives the exported ONNX bundle from C# end-to-end on a
real WAV file and asserts the resulting transcript matches the golden
output produced by `model.generate(...)` in Python.

This is the C# side of the **"Python parity → C# CLI + parity"** stage
of the Granite Speech workflow tracked in
[#28](https://github.com/christopherthompson81/vernacula/issues/28). It
mirrors `scripts/granite_export/transcribe_smoke.py` but runs through
the Microsoft.ML.OnnxRuntime C# bindings rather than `onnxruntime`
Python.

## What it validates

- `mel.onnx` runs against a NAudio-decoded WAV (no host-side mel port).
- The four-graph pipeline composes correctly across language
  boundaries: numpy-equivalent arrays → ORT tensors → next stage's ORT
  tensors.
- KV-cache handoff between `decoder_init.onnx` and `decoder_step.onnx`
  works under the C# ORT runtime (40 layers × split K/V × GQA 4-head).
- GPT-2 ByteLevel BPE decode produces the same UTF-8 text as
  `tokenizer.decode(...)` in Python.

## What it does NOT validate (yet)

- **Encoding arbitrary prompts.** The smoke loads pre-tokenised
  `input_ids` from `Fixtures/input_ids.bin` instead of running a full
  GPT-2 BPE encoder in C#. The fixture was produced by Python's
  `GraniteSpeechProcessor` for the bundled VCTK clip + the default
  ASR-with-punctuation prompt. A full BPE encoder in C# is a follow-up
  alongside the real `GraniteSpeech.cs` backend.
- Batching, IOBinding, GPU execution, dtype paths.

## Running

```bash
# 1. One-time: regenerate fixtures if you change audio/prompt.
source .venv-granite-export/bin/activate
python scripts/granite_export/dump_inputs_for_csharp_smoke.py \
    --audio ~/Downloads/VCTK_p307.wav \
    --output-dir tests/GraniteSpeechSmoke/Fixtures \
    --max-new-tokens 32

# 2. Run the smoke against an exported bundle.
dotnet run --project tests/GraniteSpeechSmoke/GraniteSpeechSmoke.csproj /p:EP=Cpu -- \
    --onnx-dir ~/models/granite_speech_4_1_2b \
    --audio ~/Downloads/VCTK_p307.wav \
    --fixtures tests/GraniteSpeechSmoke/Fixtures \
    --max-new-tokens 32
```

Expected output:

```
audio: 102704 samples @ 16000 Hz (6.42s)
prompt: 84 tokens (from fixtures)
ONNX sessions loaded.
mel: input_features=(1, 321, 160)
encoder: encoder_hidden=(1, 321, 1024)
projector: audio_embeds=(1, 66, 2048)
decoder_init: logits=(1, 84, 100353)
decoder_step: 22 tokens

  ORT  transcript: "Hello, I'm from Ontario. I hope that you will select my voice for your project. Thank you."
  Ref  transcript: "Hello, I'm from Ontario. I hope that you will select my voice for your project. Thank you."
  exact match: True
```

Exit code 0 = transcripts match, 1 = divergence.

## Fixtures layout

| File | Size | Purpose |
|---|---|---|
| `input_ids.bin` | 672 B | Pre-tokenised prompt for the bundled VCTK clip; `int64[84]` |
| `expected_tokens.bin` | 184 B | Golden continuation token IDs from `model.generate(...)`; `int64[23]` |
| `expected_text.txt` | 90 B | Decoded transcript (UTF-8) |
| `shape.json` | 466 B | Audio metadata + shape + prompt + revision (for cross-checking) |

Re-run `dump_inputs_for_csharp_smoke.py --include-features` to also
emit `input_features.bin` (~200 KB) and `attention_mask.bin` for
reference-only mel parity debugging — those aren't needed by the
default smoke path because `mel.onnx` produces the features at runtime.
