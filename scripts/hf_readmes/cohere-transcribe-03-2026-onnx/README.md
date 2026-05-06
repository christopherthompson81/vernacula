---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - onnxruntime
  - automatic-speech-recognition
  - cohere
  - vernacula
base_model: CohereLabs/cohere-transcribe-03-2026
language:
  - ar
  - de
  - el
  - en
  - es
  - fr
  - it
  - ja
  - ko
  - nl
  - pl
  - pt
  - vi
  - zh
---

# Cohere Transcribe (03-2026) — ONNX export for Vernacula

ONNX export of [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026),
a 2B-parameter Conformer encoder + Transformer decoder ASR model covering
14 languages, packaged for use with
[Vernacula](https://github.com/christopherthompson81/vernacula) and
ONNX Runtime.

**Conversion script:** [`scripts/cohere_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/cohere_export) ·
**Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula) ·
**Upstream model:** [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)

## Highlights

- **38× realtime on RTX 3090 (RTF 0.026)** via duration-sorted batched segments and `cudaMemGetInfo`-driven batch sizing (B ≤ 32, 2 GB safety margin). Decoder weights read once per step, amortised across the batch.
- **48-layer positional-embedding deduplication.** The encoder was patched to share PE across layers and constant folding was disabled to prevent per-layer duplication: the export size went from ~7 GB to ~120 MB without changing semantics.
- **`input_lengths` masking on a re-exported encoder** kills padding contamination. At a 2× length ratio in a batch, the original global self-attention catastrophically looped on shorter segments; the patched attention masks padded positions pre-softmax.
- **Float32 encoder + float16 KV-cache hybrid.** A float16 encoder hallucinated on low-energy segments; bfloat16 was blocked by the ONNX Conv opset gap. The KV cache cast to f16 halves footprint, with logits cast back to f32 at the graph boundary.
- **IOBinding output ordering matters.** Decoder outputs must be bound in *grouped* passes (all `self_key`, then all `self_val`, then all `cross_key`, then all `cross_val`) — interleaving causes MatMul dim mismatches on step 1.
- **`torch.export` dynamo path with KV-cache split as default.** Opset auto-clamped to ≥ 18 (onnxscript lacks v17 adapters for some ops). Encoder parity tightened from 5.83e-4 to 3e-6, decoder-step from 3.18e-2 to 4.8e-5 vs the legacy TorchScript path on fp32.

## Contents

| File | Purpose |
|---|---|
| `mel.onnx` | Log-Mel spectrogram frontend (16 kHz waveform → mel features) |
| `encoder.onnx` (+ `.data`) | Conformer acoustic encoder |
| `decoder_init.onnx` (+ `.data`) | Initial decoder step — emits logits + KV tensors |
| `decoder_step.onnx` (+ `.data`) | Subsequent decoder steps — consumes cached KV for fast autoregressive decode |
| `vocab.json` | Tokenizer vocabulary |
| `config.json` | Decoder config + special tokens |
| `manifest.json` | Per-file MD5 hashes (used by Vernacula's download verifier) |

## Export provenance

Exported via [`scripts/cohere_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/cohere_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo. The decoder is split into a prefill graph
(`decoder_init.onnx`) and a per-step graph (`decoder_step.onnx`) so
autoregressive generation reuses cached attention KV tensors instead of
recomputing them on every token — typical KV-cache decoder layout for ORT.

## Performance

Measured on a local RTX 3090 with Vernacula's batched ONNX pipeline:

- 600 s audio / 157 segments
- VAD: 2.0 s
- ASR: 13.5 s
- Total pipeline: 15.5 s
- RTF: 0.026 (≈ 38× realtime)

Uses batched encoder inference plus the KV-cache decoder split.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited from
the upstream [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
model. The upstream HF repo is gated (one-time acceptance flow before
download), but the license itself is permissive — commercial use is
allowed with attribution.

## Using these files

In Vernacula, select Cohere Transcribe as the ASR backend in Settings and
the package will be downloaded and verified against `manifest.json`
automatically. Outside Vernacula, pull with `huggingface_hub` and load with
`onnxruntime`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/cohere-transcribe-03-2026-onnx")
```

The decoder split (`decoder_init.onnx` + `decoder_step.onnx`) follows the
standard ORT KV-cache contract — see [`scripts/cohere_export/README.md`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/cohere_export)
for the input / output tensor names and shapes.

## Limitations

Numerical behavior matches the upstream
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
model. Language coverage (14 languages, listed above), accuracy, and
known failure modes inherit from the
[upstream model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

## Citation

```bibtex
@misc{julian_mack_2026,
    author       = { Julian Mack and Ekagra Ranjan and Walter Beller-Morales and Bharat Venkitesh and Pierre Richemond },
    title        = { cohere-transcribe-03-2026 (Revision d96e814) },
    year         = 2026,
    url          = { https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 },
    doi          = { 10.57967/hf/8653 },
    publisher    = { Hugging Face }
}
```

## Acknowledgments

- Original model: [Cohere Labs](https://cohere.com/research)
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: contact
[labs@cohere.com](mailto:labs@cohere.com) or open an issue on the upstream
model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/cohere_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/cohere_export) — the export pipeline that produced these files
- [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — upstream model card
- [Cohere Transcribe demo Space](https://huggingface.co/spaces/CohereLabs/cohere-transcribe-03-2026) — official upstream demo
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
