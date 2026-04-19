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
  --opset 18
```

Useful options:

- `--device cuda` to export on GPU
- `--dtype float16` to reduce exported weight size
- `--revision <commit-or-tag>` to pin Hugging Face remote code and weights
- `--overwrite` to replace an existing export
- `--skip-encoder`, `--skip-decoder`, or `--skip-mel` to rerun only part of the export
- `--conventional-decoder` to export the older `decoder.onnx` path instead of `decoder_init.onnx` + `decoder_step.onnx`
- `--legacy-exporter` to fall back to the deprecated TorchScript ONNX path (`dynamo=False`); kept as an escape hatch for parity comparison

## ONNX Exporter Choice

The script defaults to the modern `torch.export`-based ONNX exporter (PyTorch's
`torch.onnx.export(..., dynamo=True)`). Notes on the migration from the legacy
TorchScript exporter:

- `dynamic_axes` is replaced with `dynamic_shapes` (`torch.export.Dim`) per call site.
- `onnxscript` is now a runtime requirement of the exporter; install via the updated `requirements.txt`.
- Opset is auto-clamped to ≥ 18 on the modern path because the dynamo registry only ships v18+ implementations for several ops (e.g. `Pad`); the legacy path still accepts the historical opset 17.
- External-data layout: the dynamo exporter writes a single consolidated `<name>.onnx.data` per model directly, so the consolidation pass becomes a no-op resave (it still sweeps stale `Constant_*` files left by previous runs).
- The encoder monkey patches in `make_encoder_wrapper` (RelPositionMultiHeadAttention forward, `_needs_conv_split`, `_materialize_pe`) remain necessary: the FX exporter is *stricter* about Python-bool branches on dynamic shapes and rejects them with `GuardOnDataDependentSymNode`. The patches eliminate those branches up front and apply equally to both export paths.
- KV-cache wrappers use B=2 / past_len=2 dummy shapes so `torch.export` does not specialise the symbolic dim to a static `1`. The exported graphs still accept seq_len=1 at runtime.

Parity (CPU, fp32, 5 s dummy audio) on torch 2.11 / onnx 1.21 / ORT 1.24:

| Submodel       | Legacy max-diff | Modern max-diff |
| -------------- | --------------- | --------------- |
| `mel`          | ~3.6e-5         | ~3.5e-5         |
| `encoder`      | ~5.83e-4        | ~3e-6           |
| `decoder_init` | ~3.78e-3        | ~2.3e-5         |
| `decoder_step` | ~3.18e-2        | ~4.8e-5         |

The modern path is consistently tighter, mostly because there is no opset 17
downgrade pass and FX preserves shared buffers (positional encoding) as
initialisers instead of duplicating them per layer.

### Risks to revisit if PyTorch / onnxscript change

- The dynamo exporter currently traces `strict=False`; future versions may flip the default and start rejecting some of the in-wrapper Python conditionals. Re-test the encoder monkey patches when bumping torch.
- Opset auto-clamping assumes `Pad` (and similar ops) lack a v17 adapter. If a future onnxscript ships full v17 coverage, the clamp can be relaxed.
- The conventional `decoder.onnx` path uses a `BaseModelOutput` wrapper internally; if `transformers` changes how the decoder consumes `encoder_outputs`, the wrapper may need updating before re-exporting.

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
  --opset 18 \
  --conventional-decoder
```

This produces `decoder.onnx` instead of `decoder_init.onnx` and `decoder_step.onnx`.

## Notes

- The main exporter is the primary supported path today.
- KV-cache export is now the default decoder mode.
- If Hugging Face warns that remote-code files changed, rerun with `--revision <commit-hash>` to pin the exact model snapshot you want to trust.
