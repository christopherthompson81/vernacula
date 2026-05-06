---
license: cc-by-nc-4.0
license_link: https://github.com/BUTSpeechFIT/DiariZen/blob/main/MODEL_LICENSE
library_name: onnxruntime
pipeline_tag: voice-activity-detection
tags:
  - onnx
  - onnxruntime
  - speaker-diarization
  - diarizen
  - wespeaker
  - vernacula
  - non-commercial
base_model:
  - BUT-FIT/diarizen-wavlm-large-s80-md
---

# DiariZen — ONNX export for Vernacula

ONNX export of the [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
diarization pipeline (segmentation + WeSpeaker embedding + LDA/PLDA),
packaged for use with [Vernacula](https://github.com/christopherthompson81/vernacula)
as its high-quality diarization backend.

- **Conversion script:** [`scripts/diarizen_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/diarizen_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`BUT-FIT/diarizen-wavlm-large-s80-md`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)

> **Non-commercial use only.** The upstream DiariZen segmentation model
> is licensed [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).
> This ONNX repackaging is a derivative work and inherits the same
> non-commercial restriction. See the License section below.

## Highlights

- **RTF 1.75 → 0.66 (2.65× speedup)** vs the upstream Python pipeline, achieved by fusing mel-filter postprocess with a streaming chunk queue and eliminating intermediate `filtered` / `softScores` arrays in segmentation decode.
- **Adaptive CPU threading** (`min(12, 0.75 × cpu_count)`, 8 threads × 2 workers): segmentation path 157 s → 59 s on RTX 3090. The 8×2 layout outperformed many-tiny-workers in the threading sweep.
- **LDA/PLDA shipped as six raw float32 `.bin` files** (`mean1`, `lda`, `mean2`, `plda_mu`, `plda_tr`, `plda_psi`) with `sqrt(256)` / `sqrt(128)` normalisation factors pre-baked in. C# runs the speaker pipeline without a matrix library.
- **Un-normalised embedding output.** L2 normalisation is deferred to post-LDA in the runtime, removing redundant ops from the ONNX graph.
- **WeSpeaker Kaldi-Fbank contract spelled out exactly** (80-dim, 25 ms window, 10 ms shift, Hamming, 32768 waveform scale) so the C# Fbank implementation matches without numerical surprises.
- **Segmentation export batching evaluated and rejected.** Batch-safe export technically succeeded (`batch_size ∈ {1, 2, 4, 8, 16}` validated) but the original batch-1 export was still faster at runtime — the gains were ≤ 3.2% and net negative in practice.

## Contents

| File | Purpose |
|---|---|
| `diarizen_segmentation.onnx` (+ `.data`) | DiariZen segmentation model (WavLM-Large + segmentation head) |
| `wespeaker_pyannote_weighted.onnx` | WeSpeaker embedding model (pyannote-weighted variant) |
| `plda/lda.bin`, `mean1.bin`, `mean2.bin`, `plda_mu.bin`, `plda_psi.bin`, `plda_tr.bin` | LDA + PLDA transform parameters as flat binary tensors |
| `metadata.json` | Per-file integrity hashes and runtime config |

The PLDA `.bin` files are raw float32 tensor dumps in the shapes Vernacula's
C# inference code expects — see the export script for layout.

## Export provenance

Exported via [`scripts/diarizen_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/diarizen_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo. The pipeline is split into three on-disk artifacts
(segmentation ONNX, embedding ONNX, PLDA bins) so each can be loaded into
ORT independently and the LDA/PLDA transforms can be applied as plain
linear algebra in C#.

## License

This bundle is governed by **CC-BY-NC-4.0** because the segmentation model
inherits that license from upstream. Specifically:

| Component | Upstream license |
|---|---|
| `diarizen_segmentation.onnx` | [CC-BY-NC-4.0](https://github.com/BUTSpeechFIT/DiariZen/blob/main/MODEL_LICENSE) (from [BUT-FIT/diarizen-wavlm-large-s80-md](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)) |
| `wespeaker_pyannote_weighted.onnx` | Apache-2.0 (WeSpeaker), with pyannote-weighted variant inheriting upstream terms |
| LDA/PLDA `.bin` parameters | Derived from DiariZen training; inherit CC-BY-NC-4.0 |

The most restrictive license (CC-BY-NC-4.0) governs the bundle. **No
commercial use.** If you need a commercially-usable diarization backend
in Vernacula, use the default Sortformer pipeline instead.

## Using these files

In Vernacula, this package is downloaded automatically once you accept the
gated-model notice in **Settings → Manage Gated Models**. Outside Vernacula,
pull with `huggingface_hub` and load with `onnxruntime`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/diarizen_onnx")
```

The PLDA `.bin` files are read as raw float32 tensors — see
[`scripts/diarizen_export/README.md`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/diarizen_export)
for the expected shapes and the post-processing pipeline.

## Limitations

Numerical behavior matches the upstream
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen) pipeline. Speaker-count
accuracy, language and acoustic-domain coverage, and known failure modes
inherit from the upstream model cards
([segmentation](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md),
[WeSpeaker](https://github.com/wenet-e2e/wespeaker)). The segmentation
model was trained on data with non-commercial restrictions, which is why
this bundle cannot be redistributed for commercial use.

## Citation

For the underlying models, please cite the upstream authors. See:
- [DiariZen segmentation model card](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)
- [DiariZen repository](https://github.com/BUTSpeechFIT/DiariZen)
- [WeSpeaker repository](https://github.com/wenet-e2e/wespeaker)

## Acknowledgments

- Original DiariZen: [BUT Speech@FIT](https://www.fit.vut.cz/research/group/speech) (Brno University of Technology)
- Original WeSpeaker: WeNet community
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying models: see the upstream repositories.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/diarizen_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/diarizen_export) — the export pipeline that produced these files
- [DiariZen on GitHub](https://github.com/BUTSpeechFIT/DiariZen) — upstream pipeline source
- [`BUT-FIT/diarizen-wavlm-large-s80-md`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) — upstream segmentation model card
- [WeSpeaker on GitHub](https://github.com/wenet-e2e/wespeaker) — upstream embedding model source
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
