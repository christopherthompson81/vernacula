---
license: other
license_name: mixed-see-license-section
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - onnxruntime
  - automatic-speech-recognition
  - speaker-diarization
  - parakeet
  - sortformer
  - silero-vad
  - vernacula
base_model:
  - nvidia/parakeet-tdt-0.6b-v3
  - nvidia/diar_streaming_sortformer_4spk-v2.1
language:
  - en
---

# Parakeet TDT v3 + Streaming Sortformer — ONNX bundle for Vernacula

Combined ONNX shipping bundle used by [Vernacula](https://github.com/christopherthompson81/vernacula)
as its default ASR + diarization + VAD stack. Three upstream models are
co-located here so a single download brings up the full pipeline.

**Conversion scripts:** [`scripts/nemo_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/nemo_export) ·
**Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula) ·
**Upstream models:** [Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), [Silero VAD](https://github.com/snakers4/silero-vad)

## Highlights

- **DFT-basis mel frontend replaces `torch.stft`.** ORT's STFT op diverged from NeMo (cosine ≈ 0.23 on first inspection); the replacement uses precomputed cos/sin basis matrices as Conv1D weights, with center-padded windows and standard ops only. Restored bit-for-bit parity to the NeMo reference.
- **Streaming Sortformer 6→3 ONNX contract.** NeMo's `concat_and_pad()` isn't ONNX-traceable; the custom exporter replaces dynamic per-batch slicing with fixed-shape ops at 992 chunk frames (124 subsampled at 8× downsampling). Inputs: `chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths`. Outputs: `spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths`.
- **Dynamic-batch encoder + dynamic-batch joint decoder** for Parakeet TDT (preprocessor still batch-1 post-export). INT8 variants of encoder, decoder-joint, and Sortformer ship for CPU-only inference.
- **Chunk-by-chunk parity diagnostic** ([`compare_sortformer_chunk_outputs.py`](https://github.com/christopherthompson81/vernacula/blob/main/scripts/nemo_export/compare_sortformer_chunk_outputs.py)) compares NeMo vs ONNX state evolution across streaming chunks to localise drift to either model output or carry-state.
- **Preprocessor export sweep** ([`tune_nemo128_export.py`](https://github.com/christopherthompson81/vernacula/blob/main/scripts/nemo_export/tune_nemo128_export.py)) scores wrapper / custom / DFT modes against a legacy reference with feature-level and encoder-output deltas — the tooling that picked the DFT path in the first bullet.

## Contents

| File | Source | Purpose |
|---|---|---|
| `encoder-model.onnx` (+ `.data`) | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Parakeet TDT FastConformer encoder |
| `encoder-model.int8.onnx` | (quantized from above) | INT8-quantized encoder for CPU |
| `decoder_joint-model.onnx` | Parakeet TDT v3 | Joint decoder + prediction network |
| `decoder_joint-model.int8.onnx` | (quantized) | INT8-quantized decoder for CPU |
| `vocab.txt` | Parakeet TDT v3 | Subword vocabulary |
| `nemo128.onnx` | NeMo preprocessor | 80-mel log-FBANK frontend (128-dim hop config) |
| `diar_streaming_sortformer_4spk-v2.1.onnx` | [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | Streaming 4-speaker diarization |
| `diar_streaming_sortformer_4spk-v2.1_int8.onnx` | (quantized) | INT8-quantized diarization |
| `sortformer/diar_streaming_sortformer_4spk-v2.1.onnx` | Sortformer | Same graph in subdir layout for legacy clients |
| `silero_vad.onnx` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | Voice activity detection |
| `config.json`, `manifest.json` | Vernacula | Runtime config + per-file MD5 hashes |

## Export provenance

Exported via [`scripts/nemo_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/nemo_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo, which contains:

- `export_parakeet_nemo_to_onnx.py` — Parakeet `.nemo` → split ONNX with TDT decoder state wired explicitly
- `export_sortformer_nemo_to_onnx.py` — Streaming Sortformer `.nemo` → six-input / three-output ONNX contract
- `export_silero_vad_to_onnx.py` — Silero VAD → ONNX

The Parakeet export traces the RNNT/TDT decoder loop into a separate joint
graph so each step is a fixed-shape ORT call. Sortformer is exported as a
streaming graph that takes incoming frames + carry state and returns
diarization logits chunk-by-chunk.

## License

This bundle aggregates three upstream models under three different
licenses. Each component retains its upstream license; redistribution
here does not change those terms.

| Component | Upstream license |
|---|---|
| Parakeet TDT v3 weights | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |
| Streaming Sortformer weights | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| Silero VAD weights | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) |
| NeMo mel-frontend code (`nemo128.onnx`) | [Apache-2.0](https://github.com/NVIDIA/NeMo/blob/main/LICENSE) |

If you redistribute this bundle, propagate all four licenses with it.

## Using these files

The cleanest path is via Vernacula, which downloads, caches, and validates
this package against `manifest.json` automatically. Outside Vernacula, pull
the package with `huggingface_hub` and load each `.onnx` with `onnxruntime`
directly — input / output tensor contracts are documented in
[`scripts/nemo_export/README.md`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/nemo_export).

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/sortformer_parakeet_onnx")
```

## Limitations

These graphs preserve the numerical behavior of the upstream PyTorch
checkpoints distributed via [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).
Accuracy, language coverage, and known failure modes inherit from the
upstream model cards
([Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
[Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)) —
see those for the authoritative discussion. INT8 variants trade a small
amount of WER for ~2× CPU throughput; use the float32 variants where
accuracy is the priority.

## Citation

For the underlying models, please cite the upstream authors. See:
- [Parakeet TDT v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Streaming Sortformer model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [Silero VAD repository](https://github.com/snakers4/silero-vad)

## Acknowledgments

- Original Parakeet TDT v3 and Streaming Sortformer: NVIDIA NeMo team
- Original Silero VAD: Silero Team ([snakers4](https://github.com/snakers4))
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying models: see the upstream model cards.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion scripts (`scripts/nemo_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/nemo_export) — the export pipelines that produced these files
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — upstream Parakeet model card
- [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) — upstream Sortformer model card
- [Silero VAD on GitHub](https://github.com/snakers4/silero-vad) — upstream VAD source
- [NVIDIA NeMo on GitHub](https://github.com/NVIDIA/NeMo) — toolkit used to train and export the NVIDIA models
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
