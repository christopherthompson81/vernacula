---
license: mit
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - onnxruntime
  - automatic-speech-recognition
  - indic
  - indicconformer
  - ai4bharat
  - vernacula
base_model: ai4bharat/indic-conformer-600m-multilingual
language:
  - as
  - bn
  - brx
  - doi
  - gu
  - hi
  - kn
  - kok
  - ks
  - mai
  - ml
  - mni
  - mr
  - ne
  - or
  - pa
  - sa
  - sat
  - sd
  - ta
  - te
  - ur
---

# IndicConformer 600M — ONNX export for Vernacula

Re-packaged ONNX export of AI4Bharat's 22-language
[`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual),
in the on-disk shape that [Vernacula](https://github.com/christopherthompson81/vernacula)'s
desktop ASR app expects. The CTC head only — the RNNT components from the
source repo are not shipped here.

- **Conversion script:** [`scripts/indicconformer_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/indicconformer_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)

All numerical behavior is identical to the upstream encoder + CTC graph;
only the on-disk packaging differs.

## Highlights

- **22 languages, one shared CTC head.** Encoder dim → 5633 logits with the shared blank at id 5632; per-language vocab spans live in `language_spans.json` as `{start, length}` pairs (22 × 256 tokens). Language selection is a C# post-argmax mask, not an ONNX input — one model serves every language.
- **Phase 1 parity (CPU-CPU FP32): max-abs logits delta 6.87e-5** at 1e-3 tolerance. CUDA-vs-CPU cross-device drift hit 1e-2 scale — typical for a 17-layer Conformer; CPU is the numerically exact path.
- **Real-audio parity on a 9.4 s hi-IN Fleurs clip:** ~2 word-edit WER on an 11-word reference confirmed vocab, SentencePiece detokenisation, and language-span masking end-to-end.
- **Repackaged 2.43 GB of AI4Bharat external-data from 366 per-tensor blobs** into a single `.data` sidecar. The repackaging walks initialisers + node attributes recursively so nothing is left behind.
- **Reused the Parakeet DFT preprocessor** (no `STFT` op), with a `getattr()` shim for NeMo 1.23.0rc0 compatibility (`exact_pad`, `stft_pad_amount` are 2.x-only field names).

## Contents

| File | Purpose |
|---|---|
| `encoder-model.onnx` (+ `.data`) | Conformer encoder, `[features, features_lens] -> [encoded, encoded_lens]` |
| `ctc_decoder-model.onnx` | Single `Conv1d` → 5633-dim logits (22 × 256 language tokens + 1 shared CTC blank at id 5632) |
| `nemo128.onnx` | DFT-conv1d 80-mel preprocessor, `[waveforms, waveforms_lens] -> [features, features_lens]` |
| `vocab.txt` | Flat 5632-line vocab, id = line index; shared CTC blank is implicit at id 5632 |
| `language_spans.json` | 22 × `{start, length}` — which slice of `vocab.txt` each language's 256 tokens occupy |
| `config.json` | Preprocessor frontend params + CTC blank id |
| `manifest.json` | Per-file MD5 hashes (used by Vernacula's download verifier) |

## Export provenance

Exported via [`scripts/indicconformer_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/indicconformer_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo. The export uses [AI4Bharat's NeMo fork](https://github.com/AI4Bharat/NeMo)
(kept in an isolated venv from the main [NeMo](https://github.com/NVIDIA/NeMo) export tooling, since the fork pins
different NeMo internals).

## License

[MIT](https://opensource.org/licenses/MIT), inherited from the upstream
[`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
model.

## Using these files

In Vernacula, select IndicConformer as the ASR backend in Settings and the
package will be downloaded and verified automatically. Outside Vernacula,
pull with `huggingface_hub` and load with `onnxruntime`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/indicconformer-600m-onnx")
```

CTC decoding is performed against `vocab.txt` with the blank id at 5632.
The `language_spans.json` file lets you mask the logits to a specific
language's 256-token span before greedy / beam decoding. See [`scripts/indicconformer_export/README.md`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/indicconformer_export)
for details.

## Limitations

Covers 22 official Indian languages (listed in frontmatter). Accuracy and
known failure modes inherit from the
[upstream AI4Bharat model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).
The RNNT head from the source model is not included — only the CTC path —
which trades a small amount of accuracy for substantially simpler decoding.

## Citation

For the underlying model, see the
[upstream model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
for the canonical citation.

## Acknowledgments

- Original model: [AI4Bharat](https://ai4bharat.iitm.ac.in/) (IIT Madras)
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: see the upstream model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/indicconformer_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/indicconformer_export) — the export pipeline that produced these files
- [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) — upstream model card
- [AI4Bharat](https://ai4bharat.iitm.ac.in/) — upstream research group at IIT Madras
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
