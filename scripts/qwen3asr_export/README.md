# Qwen3-ASR Export

Exports `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` into the ONNX package we can use as a baseline for Vernacula.

This folder is intentionally the "start here" subset of [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx): just the core export path and helper modules we need inside `public/`. Benchmarking, quantization sweeps, WER evaluation, and packaging scripts can stay separate while we get our own export landed and then iterate on performance.

## Files

- `export_qwen3_asr_to_onnx.py` - exports `encoder.onnx`, `decoder_init.onnx`, `decoder_step.onnx`, `embed_tokens.bin`, tokenizer assets, and config files
- `optimize_qwen3_asr_graphs.py` - applies ORT transformer fusions to an exported package in place
- `profile_qwen3_asr_pipeline.py` - profiles the exported ONNX package and reports which stages dominate runtime
- `sweep_qwen3_asr_batching.py` - sweeps `encoder_batched.onnx` and `decoder_init_batched.onnx` on CUDA to map safe / unsafe batching regions and fit a first-pass VRAM heuristic
- `src/` - helper modules vendored from the upstream exporter and kept local to this workflow
- `requirements.txt` - Python dependencies for the export environment

## Environment

Use Python `3.11` or `3.12`.

Install dependencies:

```bash
python3 -m venv .venv-qwen3asr-export
source .venv-qwen3asr-export/bin/activate
pip install -r public/scripts/qwen3asr_export/requirements.txt
```

If the model or remote code is gated for your account, log in first:

```bash
huggingface-cli login
```

If Hugging Face downloads appear stuck at `0%`, disable the Xet transfer path for this workflow:

```bash
export HF_HUB_DISABLE_XET=1
```

## Main Export

Export the 1.7B baseline package:

```bash
python public/scripts/qwen3asr_export/export_qwen3_asr_to_onnx.py \
  --model Qwen/Qwen3-ASR-1.7B \
  --output ./models/qwen3-asr-1.7b \
  --opset 18
```

Useful options:

- `--device cuda` to trace the export on GPU
- `--dtype fp16` to shrink the exported ONNX weights after FP32 export
- `--export-batching-artifacts` to also emit experimental `encoder_batched.onnx` and `decoder_init_batched.onnx`
- `--skip-graph-optimization` to keep the raw exported graphs instead of applying ORT offline fusions
- `--skip-encoder` or `--skip-decoder` to rerun only one half of the package
- `--no-share-weights` to keep separate decoder external-data files instead of renaming the init weights blob

Default outputs:

- `encoder.onnx`
- `decoder_init.onnx`
- `decoder_init.onnx.data`
- `decoder_step.onnx` or `decoder_step.onnx.data` when too large to inline
- `embed_tokens.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `preprocessor_config.json`

Experimental batching artifacts:

- `encoder_batched.onnx`
  Inputs: `mel [batch, 128, time]`, `input_lengths [batch]`
  Outputs: `audio_features [batch, audio_len_max, 2048]`, `audio_feature_lengths [batch]`
- `decoder_init_batched.onnx`
  Inputs: `input_ids [batch, seq_len]`, `position_ids [batch, seq_len]`, `audio_features [batch, audio_len, 2048]`, `audio_lengths [batch]`, `audio_offset [1]`
  Outputs: `logits [batch, seq_len, vocab]`, `present_keys`, `present_values`

The batched prefill graph keeps `<|audio_pad|>` embeddings in positions past each item's `audio_lengths`, so a batch can share one padded prompt length.

## Profiling

Profile an exported package on a sample clip:

```bash
python public/scripts/qwen3asr_export/profile_qwen3_asr_pipeline.py \
  --onnx-dir ./models/qwen3-asr-1.7b \
  --audio ./sample.wav \
  --max-tokens 128
```

Useful options:

- `--execution-provider cpu|cuda` to force the ORT execution provider
- `--enable-ort-profiling` to emit ORT JSON traces and print the hottest operators in each graph

## Batching Sweep

Sweep the experimental batched encoder and batched decoder prefill on CUDA:

```bash
python public/scripts/qwen3asr_export/sweep_qwen3_asr_batching.py \
  --onnx-dir ./models/qwen3-asr-1.7b \
  --audio ./sample.wav \
  --durations-seconds 16,20,24,28,32 \
  --batch-sizes 6,8,10,12
```

This runs each sweep point in a fresh child process so CUDA allocator state does not pollute the VRAM readings. The output includes:

- per-point success / OOM status
- median encoder and decoder-prefill latency
- rough encoder / decoder VRAM deltas from `nvidia-smi`
- an observed safe / unsafe frontier by segment duration

Use this to derive a conservative runtime heuristic for choosing Qwen batch counts from free VRAM and planned batch duration.

## Graph Optimization

Apply ORT transformer fusions to an exported package:

```bash
python public/scripts/qwen3asr_export/optimize_qwen3_asr_graphs.py \
  --input ./models/qwen3-asr-1.7b
```

This applies the same style of offline fusions used upstream:

- decoder RMSNorm to `SimplifiedLayerNormalization`
- encoder `SkipLayerNormalization` and `BiasGelu` where matched

## Notes

- This is the upstream split-decoder export adapted into Vernacula's `public/scripts` layout.
- The current goal is a clean baseline export for Qwen3-ASR 1.7B; optimization and performance work comes next.
- The helper modules stay local under `src/` so we can patch the export flow without depending on an external checkout.
- In practice the current PyTorch exporter emits opset 18 kernels for this model family, so `--opset 18` is the recommended setting.
- The main export now applies the proven offline ORT graph fusions by default after writing the ONNX files.
- The experimental batching artifacts currently have exact ORT parity with the single-item graphs for duplicated-input batch tests.
