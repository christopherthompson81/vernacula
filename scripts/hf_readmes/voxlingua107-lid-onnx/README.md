---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: audio-classification
tags:
  - onnx
  - onnxruntime
  - audio-classification
  - language-identification
  - voxlingua107
  - ecapa-tdnn
  - speechbrain
  - vernacula
base_model: speechbrain/lang-id-voxlingua107-ecapa
---

# VoxLingua107 ECAPA-TDNN — ONNX export for Vernacula

Re-packaged ONNX export of
[`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
for use as the language-identification backend in
[Vernacula](https://github.com/christopherthompson81/vernacula).

- **Conversion script:** [`scripts/voxlingua107_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/voxlingua107_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)

## Highlights

- **`STFT` op replaced with two Conv1D passes** (cos + sin DFT basis, windowed): preprocessing wall-time share drops from 85.6% to 14% — roughly 27× faster on CUDA than the stock SpeechBrain export, which forced host fallback for the STFT op.
- **Single 83 MB FP32 ONNX file**, weights inlined, no `.data` sidecar — minimal distribution friction for clients.
- **Parity validated at Δprob 3e-11 to 6e-5** and cosine similarity ≥ 0.9999 across a 5-clip set (en, de, fr, ru, hu; 90–602 s). Top-1 language matches PyTorch on every clip.
- **Duration-accuracy sweep** ([`sweep_duration_accuracy.py`](https://github.com/christopherthompson81/vernacula/blob/main/scripts/voxlingua107_export/sweep_duration_accuracy.py)) shows confidence plateau beyond ~30 s; clips under 5 s are the noisy regime.
- **IOBinding profiling harness** ([`bench_iobinding.py`](https://github.com/christopherthompson81/vernacula/blob/main/scripts/voxlingua107_export/bench_iobinding.py)) isolates H2D / D2H allocation and copy overhead by comparing numpy ↔ `session.run` vs GPU `OrtValue` buffers + `run_with_iobinding` for both serial and batched (b=16) workloads.

## Contents

| File | Purpose |
|---|---|
| `voxlingua107.onnx` | End-to-end graph: raw 16 kHz audio → 107-class logits + 256-dim embedding |
| `lang_map.json` | Class index → `{ iso, name }` lookup |
| `manifest.json` | Per-file MD5 hashes for integrity checks |

Preprocessing (FBANK via Conv1D, per-utterance mean-variance norm) is
folded into the graph, so consumers just send raw PCM.

## Export provenance

Exported via [`scripts/voxlingua107_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/voxlingua107_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo. The `STFT` op is replaced with two `Conv1D` passes
(cos + sin basis, windowed) so the preprocessing path has CUDA kernels
end-to-end — roughly a 27× speedup on CUDA vs the stock SpeechBrain export.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited from
the [SpeechBrain source model](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa).

## Using these files

In Vernacula, language-ID runs automatically when the active ASR backend
needs to choose a language. Outside Vernacula, pull with `huggingface_hub`
and run with `onnxruntime`:

```python
from huggingface_hub import snapshot_download
import onnxruntime as ort
import json

path = snapshot_download(repo_id="christopherthompson81/voxlingua107-lid-onnx")
sess = ort.InferenceSession(f"{path}/voxlingua107.onnx")
lang_map = json.load(open(f"{path}/lang_map.json"))
# Feed raw 16 kHz mono PCM as float32 [batch, samples]
# Outputs: logits [batch, 107] and embedding [batch, 256]
```

## Limitations

Covers 107 languages (see the
[upstream VoxLingua107 paper](https://arxiv.org/abs/2011.12998) for the
full list). Accuracy varies by language and acoustic domain; the model
was trained on YouTube audio and performs best on similar conversational
speech. Short clips (<3 s) are noticeably less reliable than longer ones.

## Citation

For the underlying model:

```bibtex
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
```

See the [upstream model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
for additional citations.

## Acknowledgments

- Original model: SpeechBrain (Jörgen Valk, Tanel Alumäe — Tartu University)
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: see the upstream model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/voxlingua107_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/voxlingua107_export) — the export pipeline that produced these files
- [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) — upstream SpeechBrain model card
- [SpeechBrain on GitHub](https://github.com/speechbrain/speechbrain) — upstream toolkit
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
