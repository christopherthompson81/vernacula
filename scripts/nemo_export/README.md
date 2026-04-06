# NeMo To ONNX Export

This folder is the starting point for generating your own ONNX package from a local NeMo checkpoint.

It now covers both models in your pipeline:

- Parakeet RNNT/TDT ASR export
- Streaming Sortformer diarization export

## Files

- `export_parakeet_nemo_to_onnx.py`: exports Parakeet `.nemo` to the split ONNX package used by Vernacula.
- `export_sortformer_nemo_to_onnx.py`: exports streaming Sortformer `.nemo` to the same six-input / three-output ONNX contract used by Vernacula's inference code.
- `export_silero_vad_to_onnx.py`: exports Silero VAD to ONNX.
- `tune_nemo128_export.py`: runs multiple preprocessor export candidates and scores them against a legacy reference — use this if the default export mode needs tuning.
- `setup_nemo_export_env.py`: creates the Python export venv.
- `requirements.txt`: export dependencies.

## Environment

Use Python `3.11` or `3.12`.

NeMo requires a specific Python version range — `3.14` is too new for a dependable export environment.

```bash
python3 scripts/nemo_export/setup_nemo_export_env.py
source .venv-nemo-export/bin/activate
```

Install with a specific CUDA build or CPU-only:

```bash
# CUDA 12.1
python3 scripts/nemo_export/setup_nemo_export_env.py --cuda-version cu121
# CPU only
python3 scripts/nemo_export/setup_nemo_export_env.py --cuda-version ""
```

## Parakeet Export

```bash
python scripts/nemo_export/export_parakeet_nemo_to_onnx.py \
  --nemo ~/models/parakeet-tdt-0.6b-v3.nemo \
  --output-dir ~/models/parakeet_onnx \
  --opset 17
```

Outputs:

- `encoder-model.onnx`
- `decoder_joint-model.onnx`
- `nemo128.onnx` when preprocessor export succeeds
- `vocab.txt`
- `config.json`
- `export-report.json`

Preprocessor modes:

- Default `wrapper` mode exports NeMo's live preprocessor module directly (consistently fails on this toolchain — NeMo's STFT is not traceable).
- `custom` mode rebuilds the preprocessor using `torch.stft` — exports, but ONNX Runtime diverges (cosine ~0.23).
- **`dft` mode** rebuilds the preprocessor using a conv1d DFT basis matrix — no `STFT` op, numerically equivalent to PyTorch `torch.stft`, ONNX-safe.

```bash
python scripts/nemo_export/export_parakeet_nemo_to_onnx.py \
  --nemo ~/models/parakeet-tdt-0.6b-v3.nemo \
  --output-dir ~/models/parakeet_onnx_dft \
  --opset 17 \
  --preprocessor-mode dft \
  --preprocessor-dynamo false \
  --preprocessor-optimize false
```

### Why `dft` mode works where `custom` does not

`torch.stft` exports to ONNX opset 17 as the `STFT` op. ONNX Runtime's implementation diverges from PyTorch's on this toolchain, producing features with cosine similarity ~0.23 vs NeMo.

`dft` mode avoids the `STFT` op entirely. It precomputes a windowed DFT basis matrix (cos/sin rows, window center-padded from `win_length` to `n_fft`) in `__init__`, stores it as a buffer, and computes the mel spectrogram via:

```
frames = F.conv1d(padded_waveform, dft_basis, stride=hop_length)
magnitude = sqrt(real_frames² + imag_frames²)
mel = log(fb @ magnitude^mag_power + guard)
```

The only ops in the ONNX graph are `Conv`, `Mul`, `Pow`, `Add`, `Sqrt`, `Log`, `MatMul` — all first-class ONNX ops with no normalization or convention ambiguities.

`get_seq_len` also matches NeMo's `FilterbankFeatures` formula exactly (the `custom` wrapper was missing the final `+1`, producing length 2000 where NeMo returns 2001).

Batching notes:

- `encoder-model.onnx` exports with dynamic batch and sequence dimensions.
- `decoder_joint-model.onnx` exports with dynamic batch-compatible input axes.
- `nemo128.onnx` currently exports successfully on this toolchain, but in practice still behaves
  like a batch-1 preprocessor export. That means post-diarization encoder batching is available
  today, while full waveform-to-text batching still needs more export work.

## Sortformer Export

```bash
python scripts/nemo_export/export_sortformer_nemo_to_onnx.py \
  --nemo ~/models/diar_streaming_sortformer_4spk-v2.1.nemo \
  --output ~/models/diar_streaming_sortformer_4spk-v2.1.onnx \
  --opset 17 \
  --overwrite
```

Outputs:

- `diar_streaming_sortformer_4spk-v2.1.onnx`
- `diar_streaming_sortformer_4spk-v2.1.onnx.report.json`

The exported model uses this streaming inference contract:

Inputs:

- `chunk` with shape `[batch, time_chunk, 128]`
- `chunk_lengths` with shape `[batch]`
- `spkcache` with shape `[batch, time_cache, 512]`
- `spkcache_lengths` with shape `[batch]`
- `fifo` with shape `[batch, time_fifo, 512]`
- `fifo_lengths` with shape `[batch]`

Outputs:

- `spkcache_fifo_chunk_preds` with shape `[batch, time_out, 4]`
- `chunk_pre_encode_embs` with shape `[batch, time_pre_encode, 512]`
- `chunk_pre_encode_lengths` with shape `[batch]`

## Sortformer Notes

NeMo's built-in `streaming_export()` is currently broken for this checkpoint for two separate reasons:

1. `streaming_input_examples()` hardcodes `chunk` as `(batch, 120, 80)`, but this model expects 128 mel features.
2. `SortformerModules.concat_and_pad()` uses dynamic per-batch slicing that fails during ONNX export.

The custom Sortformer exporter in this folder works around both issues by:

- supplying a correct export example with 128 mel features
- replacing the problematic concat logic with an ONNX-friendly implementation
- using the legacy `torch.onnx.export(..., dynamo=False)` path, which succeeded here where the newer exporter did not

## References

- [Parakeet export guidance discussion](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/9)
- [NeMo transducer ONNX example](https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/export/transducer/infer_transducer_onnx.py)
- [NeMo Sortformer issue](https://github.com/NVIDIA/NeMo/issues/15077)

