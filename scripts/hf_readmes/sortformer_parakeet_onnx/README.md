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

- **Conversion scripts:** [`scripts/nemo_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/nemo_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream models:** [Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), [Silero VAD](https://github.com/snakers4/silero-vad)

## Highlights

- **DFT-basis mel frontend replaces `torch.stft`.** ORT's STFT op diverged from NeMo (cosine ≈ 0.23 on first inspection); the replacement uses precomputed cos/sin basis matrices as Conv1D weights, with center-padded windows and standard ops only. Restored bit-for-bit parity to the NeMo reference.
- **Streaming Sortformer 6→3 ONNX contract.** NeMo's `concat_and_pad()` isn't ONNX-traceable; the custom exporter replaces dynamic per-batch slicing with fixed-shape ops at 992 chunk frames (124 subsampled at 8× downsampling). Inputs: `chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths`. Outputs: `spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths`.
- **CoreML-compilable Sortformer variant for Apple Silicon.** The stock diarization graph cannot be compiled by ONNX Runtime's CoreML EP at all (`Failed to create MLModel … error code: -14`), so the Neural Engine was unreachable on macOS. `diar_streaming_sortformer_4spk-v2.1.coreml.onnx` is a fully static re-export plus four value-preserving graph rewrites that compile as a **single** CoreML partition — 51.5 ms/chunk vs 171.8 ms on CPU (M5, ORT 1.29.0). It is a **steady-state-only** graph with a different input contract; read [Sortformer CoreML variant](#sortformer-coreml-variant) before using it.
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
| `diar_streaming_sortformer_4spk-v2.1.coreml.onnx` | Sortformer | Static steady-state diarization graph for the CoreML EP — **different contract**, see below |
| `sortformer/diar_streaming_sortformer_4spk-v2.1.onnx` | Sortformer | Same graph in subdir layout for legacy clients |
| `silero_vad.onnx` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | Voice activity detection |
| `config.json`, `manifest.json` | Vernacula | Runtime config + per-file MD5 hashes |

## Sortformer CoreML variant

`diar_streaming_sortformer_4spk-v2.1.coreml.onnx` exists because ONNX Runtime's CoreML
execution provider is the only route to the Apple Neural Engine, and the stock graph
cannot be compiled by it at all — it slices by tensor *values*, which makes every
downstream shape data-dependent, and CoreML's MIL runtime rejects unbounded dimensions
outright. The variant fixes the shapes at export time and applies four rewrites (delete
51 all-False `Where` masks, rewrite 34 constant zero `Pad`s as `Concat`s, pre-transpose
180 `Gemm` weights) to reach a single CoreML partition instead of 71.

Measured on an M5 (Mac17,3, macOS 26.6.2), **ORT 1.29.0** — the version Vernacula builds
against — chunk=992 / spkcache=188 / fifo=124:

| model | EP | inference | cold load | warm load |
|---|---|---|---|---|
| stock dynamic | CPU | 171.8 ms | 0.50 s | — |
| stock dynamic | CoreML | *fails to compile* | — | — |
| CoreML variant | CPU | 154.0 ms | — | — |
| **CoreML variant** | **CoreML** | **51.5 ms** | **2.2 s** | **0.16 s** |

**3.34× vs the stock graph on CPU.** The CoreML EP takes the whole graph as a single
partition (1751 of 1751 nodes), measured at `ORT_ENABLE_BASIC` (see note 3). Outputs match the stock graph to `4.470E-07` (`preds`,
rms 7.616E-08) running CoreML against stock-on-CPU, and to `3.576E-07` CPU-to-CPU. The
static shapes alone also make it ~10% faster on plain CPU, so it is not purely a macOS
artifact.

The stock graph's CoreML failure reproduces on 1.29.0 exactly as described above —
`Input: _ConstantOfShape_1_output_0 has unbounded dimension which is not supported`.

An earlier round on **ORT 1.24.4** measured 52.3 ms for the variant on CoreML, alongside
196.3 ms (CPU) and 113.0 ms (WebGPU) for the stock graph and 101.7 ms for the variant on
WebGPU. Those CPU/WebGPU baselines disagree with other figures recorded for the same
machine and ORT (163.5 ms / 94.0 ms) and have not been reconciled; treat the 1.29.0 table
above as authoritative and the 1.24.4 baselines as indicative only. WebGPU was not
re-measured on 1.29.0 — the provider is absent from the `onnxruntime` PyPI wheel and ships
only in the packaged app.

**This is not a drop-in replacement.** Four things differ:

| | stock | CoreML variant |
|---|---|---|
| inputs | 6 (`chunk`, `spkcache`, `fifo` + 3 `*_lengths`) | **3** (`chunk`, `spkcache`, `fifo`) |
| shapes | dynamic | fixed `[1,992,128]` / `[1,188,512]` / `[1,124,512]` |
| `spkcache_fifo_chunk_preds` | `[batch, time_out, 4]` | `[1, 436, 4]` |
| graph optimization level | any | **`ORT_ENABLE_BASIC` or lower** |

1. **All three lengths are baked in, so this graph is steady-state only.** Pass full-size,
   zero-padded buffers. Zero-filled `spkcache`/`fifo` during warm-up are fine — the stock
   runtime already passes those two lengths as the full buffer size on every call.
2. **Never feed it the final, short chunk of a recording.** With `chunk_lengths` baked to
   992 the graph attends to that chunk's zero-padded tail as real audio, which moves its
   speaker probabilities by up to **0.54** (rms 0.24) on a 0..1 scale — enough to flip
   speaker assignments. Route that one chunk per recording to the stock dynamic graph,
   which is in this same repo.
3. **Create the session at `ORT_ENABLE_BASIC` or lower.** At `ORT_ENABLE_EXTENDED` and
   above, ORT fails to load the file outright — `AddInitializedOrtValue Attempt to replace
   the existing tensor`, from `MatMulAddFusion` re-running over an already-optimized graph.
   Confirmed on both 1.26.0 and 1.29.0.

   ⚠ **This is masked when the CoreML EP is registered.** CoreML claims the whole graph
   before the CPU fusions run, so on macOS all four levels appear to load — but the moment
   the graph falls back to the CPU EP (CoreML absent from the build, a non-Apple machine,
   or `MLComputeUnits` declining it) the same session options throw. Pin BASIC regardless
   of platform; it costs nothing, since the file is already an optimized graph.

   | level | CPU EP | CoreML EP registered |
   |---|---|---|
   | `ORT_DISABLE_ALL` | loads | loads |
   | `ORT_ENABLE_BASIC` | loads | loads |
   | `ORT_ENABLE_EXTENDED` | **fails** | loads |
   | `ORT_ENABLE_ALL` | **fails** | loads |

   A separate 1.24.4 measurement found EXTENDED also shattering CoreML partitioning
   (69 → 191) on the earlier four-input graph.
4. **CoreML partitioning is ORT-version dependent — re-validate on any ORT upgrade.**
   Validated on **1.24.4** and **1.29.0**, which both reach a single partition. An earlier
   note here claimed 1.29.0 split the graph into 194 partitions and diverged at ~1e-2;
   that was measured against a different build of the graph and **does not reproduce** —
   1.29.0 gives 1 partition and `4.470E-07`. Set the CoreML EP's `ModelCacheDirectory` or
   every load pays the 2.2 s compile instead of 0.16 s.

`fp16` was measured faster still (22.3 ms) but is **not** shipped: `chunk_pre_encode_embs`
diverges to 9.8e-02 there and those embeddings feed back into the speaker cache and FIFO,
so it needs an end-to-end DER check rather than a single-chunk comparison.

Reproduce it with:

```bash
python scripts/nemo_export/export_sortformer_nemo_to_onnx.py \
  --nemo <path>/diar_streaming_sortformer_4spk-v2.1.nemo \
  --output sortformer_coreml.onnx --opset 17 \
  --coreml-static-batch1 --coreml-const-lengths --coreml-const-chunk-length \
  --chunk-frames 992 --fixed-spkcache-frames 188 --fixed-fifo-frames 124 --overwrite

python scripts/nemo_export/coreml_optimize_sortformer.py \
  --input sortformer_coreml.onnx \
  --output diar_streaming_sortformer_4spk-v2.1.coreml.onnx \
  --verify --reference diar_streaming_sortformer_4spk-v2.1.onnx
```

The techniques generalize; see
[`docs/coreml_onnx_playbook.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/coreml_onnx_playbook.md).

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
