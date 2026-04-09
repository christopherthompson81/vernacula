# Cohere Transcribe Export

Exports `CohereLabs/cohere-transcribe-03-2026` into the ONNX package consumed by Vernacula.

This folder contains the export scripts only. Investigation and smoke-test scripts remain in the root `scripts/cohere_export` area for now.

## Files

- `export_cohere_transcribe_to_onnx.py` - exports the main Cohere package: `encoder.onnx`, `mel.onnx`, `config.json`, tokenizer assets, and by default the KV-cache decoder pair
- `requirements.txt` - Python dependencies for the export environment

## Environment

Use Python `3.11` or `3.12`.

The model is gated on Hugging Face, so log in before export:

```bash
huggingface-cli login
```

Install dependencies:

```bash
pip install -r public/scripts/cohere_export/requirements.txt
```

`onnxruntime-gpu` is listed by default. If you want a CPU-only environment, install `onnxruntime` instead.

## Main Export

Export the standard Cohere ONNX package. This now exports the KV-cache decoder path by default:

```bash
python public/scripts/cohere_export/export_cohere_transcribe_to_onnx.py \
  --output-dir ./models/cohere_transcribe \
  --opset 17
```

Useful options:

- `--device cuda` to export on GPU
- `--dtype float16` to reduce exported weight size
- `--revision <commit-or-tag>` to pin Hugging Face remote code and weights
- `--overwrite` to replace an existing export
- `--skip-encoder`, `--skip-decoder`, or `--skip-mel` to rerun only part of the export
- `--conventional-decoder` to export the older `decoder.onnx` path instead of `decoder_init.onnx` + `decoder_step.onnx`

Outputs:

- `encoder.onnx`
- `mel.onnx`
- `decoder_init.onnx`
- `decoder_step.onnx`
- `config.json`
- `vocab.json`
- `tokenizer.json`
- `tokenizer.model`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `export-report.json`

## Conventional Decoder Export

If you want the older full-sequence decoder graph instead of the default KV-cache export:

```bash
python public/scripts/cohere_export/export_cohere_transcribe_to_onnx.py \
  --output-dir ./models/cohere_transcribe \
  --opset 17 \
  --conventional-decoder
```

This produces `decoder.onnx` instead of `decoder_init.onnx` and `decoder_step.onnx`.

## Notes

- The main exporter is the primary supported path today.
- KV-cache export is now the default decoder mode.
- If Hugging Face warns that remote-code files changed, rerun with `--revision <commit-hash>` to pin the exact model snapshot you want to trust.
