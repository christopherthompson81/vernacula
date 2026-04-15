# VibeVoice-ASR Export

Exports `microsoft/VibeVoice-ASR-HF` into the ONNX package consumed by Vernacula.

This public folder keeps only the core export flow plus the static-vs-dynamic parity check. Investigation, benchmarking, and smoke-test scripts stay in the root `scripts/vibevoice_export` area.

## Files

- `export_vibevoice_asr_to_onnx.py` - exports `audio_encoder.onnx`, decoder graph(s), config files, and `export-report.json`
- `test_static_kv_parity.py` - compares `decoder_single_static.onnx` token output against `decoder_single.onnx`
- `_common.py` - shared helpers used by the export and parity scripts
- `requirements.txt` - Python dependencies for this workflow

## Environment

Use Python `3.11` or `3.12`.

VibeVoice-ASR currently depends on a newer Transformers build than the stable release used elsewhere in this repo:

```bash
python3 -m venv .venv-vibe-export
source .venv-vibe-export/bin/activate
pip install -r public/scripts/vibevoice_export/requirements.txt
```

If the model or processor is gated for your account, log in first:

```bash
huggingface-cli login
```

## Main Export

Export the default recommended package with the unified decoder graph:

```bash
python public/scripts/vibevoice_export/export_vibevoice_asr_to_onnx.py \
  --output-dir ./models/vibevoice_asr \
  --opset 18 \
  --dtype float16 \
  --deterministic-audio
```

Useful options:

- `--device cuda` to export on GPU
- `--revision <commit-or-tag>` to pin the exact model snapshot
- `--decoder-exporter auto|legacy|torch-export` to choose the decoder exporter path
- `--decoder-graph-mode split|single|both|static-single` to choose which decoder graphs to emit
- `--f32-kv-cache` to export the decoder with float32 KV cache
- `--f32-lm-head` to export the decoder with a float32 LM head projection
- `--skip-audio-encoder`, `--skip-prefill`, or `--skip-step` to rerun just part of the export
- `--overwrite` to replace an existing export

Default outputs:

- `audio_encoder.onnx`
- `decoder_single.onnx`
- `config.json`
- `generation_config.json`
- `processor_config.json`
- `export-report.json`

If you want both the unified decoder and the split decoder pair:

```bash
python public/scripts/vibevoice_export/export_vibevoice_asr_to_onnx.py \
  --output-dir ./models/vibevoice_asr_both \
  --opset 18 \
  --dtype float16 \
  --decoder-graph-mode both \
  --deterministic-audio
```

That additionally emits:

- `decoder_prefill.onnx`
- `decoder_step.onnx`

## Static KV Parity Check

If you export both the dynamic and static single-decoder variants, compare them with:

```bash
python public/scripts/vibevoice_export/test_static_kv_parity.py \
  --static-dir ./models/vibevoice_asr_static \
  --dynamic-dir ./models/vibevoice_asr_dynamic \
  --audio ./sample.wav \
  --max-tokens 256
```

The script runs the audio encoder, then compares generated token IDs from:

- `decoder_single_static.onnx`
- `decoder_single.onnx`

## Notes

- The regular single-decoder export is the default and the path Vernacula.Avalonia currently expects.
- Static-KV export remains supported for parity and experimentation, but it is optional.
