---
license: openmdw-1.1
library_name: onnxruntime
pipeline_tag: voice-activity-detection
tags:
  - onnx
  - onnxruntime
  - speaker-diarization
  - streaming-sortformer
  - nemotron
  - vernacula
base_model:
  - nvidia/Nemotron-3-Diarization
---

# Nemotron-3-Diarization — ONNX for Vernacula

A single-file ONNX export of [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(streaming, up to 8 speakers, 10 ms output), in the streaming contract that
[Vernacula](https://github.com/christopherthompson81/vernacula) runs.

- **Conversion script:** [`scripts/nemo_export/export_nemotron3_diarization_to_onnx.py`](https://github.com/christopherthompson81/vernacula/blob/main/scripts/nemo_export/export_nemotron3_diarization_to_onnx.py)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

## Contents

| File | Purpose |
|---|---|
| `nemotron-3-diarization.onnx` | The network for one streaming step, fp32, opset 17, ~400 MB. Its metadata carries the streaming schedule and the learned silence embedding. |
| `nemotron-3-diarization.onnx.report.json` | Export report: the metadata written into the graph, and the ORT-vs-NeMo parity measured at export time |
| `manifest.json` | Per-file MD5 hashes, for Vernacula's update check |

## Contract

One call is one streaming step over `[speaker cache | FIFO | chunk]`. Batch 1; every time
axis is dynamic.

| | name | shape | |
|---|---|---|---|
| in | `chunk` | `[1, T_mel, 128]` | log-mel frames, **zero-padded to a multiple of 8 and no further** |
| in | `chunk_lengths` | `[1]` int64 | real mel frames in `chunk` |
| in | `spkcache` | `[1, T_cache, 512]` | speaker-cache embeddings |
| in | `spkcache_lengths` | `[1]` int64 | |
| in | `fifo` | `[1, T_fifo, 512]` | FIFO embeddings |
| in | `fifo_lengths` | `[1]` int64 | |
| out | `spkcache_fifo_chunk_preds` | `[1, T, 8]` | 80 ms speaker probabilities — feed the cache/FIFO update |
| out | `chunk_pre_encode_embs` | `[1, T_ch, 512]` | the chunk's embeddings, to append to the FIFO |
| out | `chunk_pre_encode_lengths` | `[1]` | |
| out | `spkcache_fifo_chunk_preds_hr` | `[1, 8·T, 8]` | 10 ms speaker probabilities — what to report |

The cache/FIFO bookkeeping around the graph (compression, FIFO pops) is NeMo's
`SortformerModules.streaming_update`; Vernacula's C# port is `SortformerStreamer`. Two points
differ from Streaming Sortformer v2.x:

- disabled speaker-cache slots are filled with a **learned** silence embedding rather than the
  running mean of silent frames;
- the chunk should be sent with its right context (the schedule below), and must not be padded
  past the next multiple of 8 — the 10 ms head is a k=3 convolution, so the last real frame
  would read the padded frame's hidden state.

### Metadata (`metadata_props`, prefix `vernacula.diar.`)

`contract` (`sortformer-hr-1`), `num_speakers` 8, `emb_dim` 512, `n_mels` 128,
`subsampling` 8, `upsample_factor` 8, the card's offline schedule — `spkcache_len` 264,
`fifo_len` 40, `chunk_len` 340, `chunk_right_context` 40, `spkcache_update_period` 300 (all in
80 ms frames) — the compression constants, and `learnable_sil_emb_f32le_b64` (512 float32,
little-endian, base64).

## Export provenance

NeMo's encoder uses PyTorch FlexAttention, which does not trace to ONNX. The export replaces it
with scaled-dot-product attention under a key-padding mask — equivalent for this checkpoint's
`attn_mode: full` with no causal tail — and refuses any checkpoint whose encoder configuration
differs. Needs NeMo from `main` (the 3.0.0 release cannot build this checkpoint); the recipe is in
[`docs/investigations/nemotron3_diarization_onnx_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/investigations/nemotron3_diarization_onnx_investigation.md).

Verification, both recorded in that document:

- **Graph:** against NeMo's own modules on first, steady-state and short tail chunks, max
  |ORT − NeMo| ≤ 5e-8 on the 80 ms and ≤ 1.4e-7 on the 10 ms probabilities.
- **Streaming loop:** Vernacula's C# loop against NeMo's `forward_streaming` over public
  multi-speaker recordings (VoxConverse dev, AMI SDM test excerpts): identical to float noise on
  four of six files and ≤ 0.01 % fidelity DER on the other two, given the same features.

## License

The weights are NVIDIA's, released under the [OpenMDW License 1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization)
— see the upstream model card for its terms. This repository only repackages them.

## Using these files

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort

path = hf_hub_download("christopherthompson81/nemotron3_diarization_onnx", "nemotron-3-diarization.onnx")
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print({p.key: p.value for p in __import__("onnx").load(path, load_external_data=False).metadata_props
       if not p.key.endswith("_b64")})
```

The graph is one streaming step, not a whole-file diarizer: a caller has to run the streaming
loop around it (see Contract).

## Limitations

See the [upstream model card](https://huggingface.co/nvidia/Nemotron-3-Diarization) for the
model's own limitations. Specific to this repackaging:

- Only the card's offline schedule (30.4 s input buffer) is recorded in the metadata and has been
  verified end to end. The graph itself accepts the low-latency schedules too.
- Batch 1 only. fp32 only.
- Vernacula thresholds the 10 ms probabilities at 0.5 like NeMo does, but also merges a speaker's
  segments separated by less than 0.5 s, to avoid sending sub-second fragments to ASR.

## Citation

See the [upstream model card](https://huggingface.co/nvidia/Nemotron-3-Diarization#references)
for the Sortformer and Streaming Sortformer papers.

## Acknowledgments

Nemotron-3-Diarization is by NVIDIA. Repackaged for ONNX Runtime by Chris Thompson for Vernacula.

## See also

- [Vernacula](https://github.com/christopherthompson81/vernacula)
- [sortformer_parakeet_onnx](https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx) — the default Streaming Sortformer v2.1 diarizer
- [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
