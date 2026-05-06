---
license: mit
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - onnxruntime
  - automatic-speech-recognition
  - speaker-diarization
  - vibevoice
  - vernacula
base_model: microsoft/VibeVoice-ASR-HF
---

# VibeVoice-ASR — ONNX export for Vernacula

ONNX export of [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF),
packaged for use with [Vernacula](https://github.com/christopherthompson81/vernacula)
and ONNX Runtime.

- **Conversion script:** [`scripts/vibevoice_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/vibevoice_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF)

## Highlights

- **GPU-resident KV cache via `OrtIoBinding`: RTF 2.78× → 0.50× (5.5× speedup).** The naive path copied 56 KV tensors of `[1, 4, cache_len, 128]` between CPU and GPU on every decode step (~4 TB of PCIe traffic across a 4000-token generate). `BindOutputToDevice` + per-step re-registration eliminates the round-trip entirely.
- **Float32 KV + attention defers a BF16 divergence** from token 11 (whole-seconds digit) to token 14 (sub-decimals only). The cost is ~23% Cast-node overhead and the win is killing spurious 7-second segment-boundary shifts that BF16 introduced.
- **`ORT_ENABLE_EXTENDED` graph optimisation** (not `_ALL`) for the decoder: avoids a BF16 attention-fusion regression that shifts divergence earlier, while still applying GeLU / LayerNorm fusions. The audio encoder uses `_ALL` since it benefits from memory-layout transforms.
- **Audio-encoder session disposed after encoding** in built-in diarization mode (frees 1–3 GB of CUDA arena before the 600 s autoregressive decode). Segmented mode persists the session to avoid 70+ session-load round-trips at ~20 s overhead each.
- **Profiled framework overhead at 56% of wall time** — 1950 CUDA kernel launches × ~7 µs each = ~14 ms/step of structural overhead from `SequentialExecutor` dispatch + launch latency. Reaching below this requires CUDA graphs or TensorRT.
- **Static-vs-dynamic KV parity test** ships alongside the export ([`test_static_kv_parity.py`](https://github.com/christopherthompson81/vernacula/blob/main/scripts/vibevoice_export/test_static_kv_parity.py)) — token-by-token comparison with bf16 ↔ f32 `OrtValue` conversion.

## Contents

| File | Purpose |
|---|---|
| `audio_encoder.onnx` (+ `.data`) | Audio encoder — waveform features → encoded representations |
| `decoder_single.onnx` (+ `.data`) | Unified decoder graph (default and recommended path) |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer assets |
| `chat_template.jinja` | Prompt / chat template |
| `config.json`, `processor_config.json` | Model + processor config |
| `export-report.json` | Export-time metadata (op coverage, parity check results) |
| `manifest.json` | Per-file MD5 hashes (used by Vernacula's download verifier) |

## Export provenance

Exported via [`scripts/vibevoice_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/vibevoice_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo. The export folder also contains a static-vs-dynamic
KV-cache parity check (`test_static_kv_parity.py`) that validates the
decoder graph against the reference PyTorch implementation.

## Runtime notes

- Vernacula uses the unified decoder graph (`decoder_single.onnx`) as the
  default and recommended path.
- The current Vernacula runtime treats this package as **CUDA-only**.
- `manifest.json` stores MD5 hashes used by Vernacula.Avalonia to verify
  and re-download missing or outdated files.

## License

[MIT](https://opensource.org/licenses/MIT), inherited from the upstream
[`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
model. Commercial and modification rights both granted.

## Using these files

In Vernacula, select VibeVoice-ASR as the ASR backend in Settings and the
package will be downloaded and verified automatically. Outside Vernacula,
pull with `huggingface_hub` and load with `onnxruntime`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/vibevoice-asr-onnx")
```

The decoder graph contract is documented in [`scripts/vibevoice_export/README.md`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/vibevoice_export).

## Limitations

Numerical behavior matches the upstream
[`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
model. Language coverage, accuracy, and known failure modes inherit from
the [upstream model card](https://huggingface.co/microsoft/VibeVoice-ASR-HF).

## Citation

For the underlying model, see the
[upstream model card](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
for the canonical citation.

## Acknowledgments

- Original model: Microsoft Research
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: see the upstream model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/vibevoice_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/vibevoice_export) — the export pipeline that produced these files
- [`microsoft/VibeVoice-ASR-HF`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) — upstream model card
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
