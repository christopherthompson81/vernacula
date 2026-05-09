---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - onnxruntime
  - automatic-speech-recognition
  - granite
  - granite-speech
  - vernacula
base_model: ibm-granite/granite-speech-4.1-2b
language:
  - en
  - fr
  - de
  - es
  - pt
  - ja
---

# IBM Granite Speech 4.1 (2B) — ONNX export for Vernacula

ONNX export of [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b),
a speech-to-text model combining a 16-layer Conformer acoustic encoder, a
BLIP-2 Q-Former projector, and a 1.84 B-parameter Granite-4.0 LLM
decoder. Packaged for use with
[Vernacula](https://github.com/christopherthompson81/vernacula) and
ONNX Runtime.

- **Conversion script:** [`scripts/granite_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- **BF16 sibling bundle (Ampere+ GPUs):** [`christopherthompson81/granite-speech-4-1-2b-onnx-bf16`](https://huggingface.co/christopherthompson81/granite-speech-4-1-2b-onnx-bf16) — same content with the LM decoder in BF16, ~39% smaller and ~27% faster end-to-end on hardware with BF16 tensor cores. CPU and pre-Ampere GPU users should stay on this FP32 bundle.

## Highlights

- **23× realtime on RTX 3090 (RTF 0.043)** on a 10-min English clip with VAD-segmented input. Achieved via VRAM-budgeted batching (B ≤ 16, `cudaMemGetInfo`-driven), the chained Run-with-OrtValue decode pattern that keeps KV GPU-resident across steps without `OrtIoBinding`, and a per-row periodic-loop detector that forces EOS on runaway rows so stragglers do not drag finished batch-mates to the `max_new_tokens` cap.
- **Unified prefill+step decoder graph.** Prefill is signalled by passing zero-length past-KV inputs and `cache_position=[0..S-1]`; step is signalled by populated past-KV and `cache_position=[past_len]`. One graph, one set of weights — eliminates the duplicate 7 GB LM copy that an init/step split would require, while preserving exact fp32 parity with the upstream forward.
- **Encoder rewritten to full attention.** The upstream block-attention encoder cannot be made symbolic for ONNX export. The exporter replaces it with a mathematically equivalent full-attention pass guarded by a regression test (`test_encoder_math_equivalence.py`) — max-abs-diff vs the original block path is ~1e-5 (fp32 noise).
- **Three trace-time patches** for `torch.onnx.export(dynamo=True)`: 5-D SDPA replaced with a manual softmax-attention math, `attn_implementation="eager"` forced through the LM, and the audio-token `masked_scatter` replaced with a `cumsum + gather + where` pattern that drops the need for an explicit audio mask and keeps the merge symbolic.
- **fp32 throughout.** A fp16 LM lost ~1 token of greedy parity on long contexts. Decoder unification recovered that loss while halving resident weight memory, so fp32 is the shipping precision.

## Contents

| File | Purpose |
|---|---|
| `mel.onnx` | Log-Mel spectrogram frontend (16 kHz waveform → mel features) |
| `encoder.onnx` (+ `.data`) | Conformer acoustic encoder, full-attention rewrite |
| `projector.onnx` | BLIP-2 Q-Former audio-to-text projector (encoder hidden → 2048-D audio embeds) |
| `decoder.onnx` (+ `.data`) | Unified prefill + step Granite-4.0 LM decoder with cumsum-gather audio merge |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer_config.json` | GPT-2-family ByteLevel BPE tokenizer assets |
| `chat_template.jinja` | Chat template (used to construct the ASR prompt) |
| `preprocessor_config.json`, `processor_config.json` | Mel/preprocessor parameters from upstream |
| `export-report.json` | Per-stage export timings + dtype/opset metadata |
| `manifest.json` | Per-file MD5 hashes (used by Vernacula's download verifier) |

## Export provenance

Exported via [`scripts/granite_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export)
in the [Vernacula](https://github.com/christopherthompson81/vernacula) repo using
the dynamo-based `torch.onnx.export` path at opset 18, dtype fp32. The
decoder is a single graph that handles both prefill and step via empty
vs populated past-KV inputs; runtime sees the same weights for the whole
autoregressive trajectory.

The export and parity work, including the three trace-time patches and
the encoder full-attention rewrite, is logged step-by-step in
[`docs/dev/granite_speech_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/dev/granite_speech_investigation.md);
the perf iteration loop (chained Run-with-OrtValue, batching, runaway
detection) is in
[`docs/dev/granite_speech_perf_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/dev/granite_speech_perf_investigation.md).

## Performance

Measured on a local RTX 3090 with Vernacula's batched ONNX pipeline:

| Workload | Mode | Wall ASR | RTF |
|---|---|---:|---:|
| 6.4 s VCTK clip | B=1 | 3.25 s | 0.508 |
| 25 s, 4 VCTK speakers | B=4 | 3.66 s | 0.147 |
| 90 s VAD-segmented | mixed B (≤16) | 7.80 s | 0.091 |
| 600 s (10 min) VAD-segmented, 159 segs | mixed B (≤16) | 23.9 s | **0.043** |

For comparison, on the same 600 s clip Vernacula's serial Qwen3-ASR-1.7B
backend lands RTF ≈ 0.075; Granite Speech 4.1 batched is ~1.7× faster
end-to-end.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited
from the upstream
[`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
model. Permissive — commercial use is allowed with attribution.

## Using these files

In Vernacula, select Granite Speech as the ASR backend
(`--asr granite` on the CLI) and the package will be downloaded and
verified against `manifest.json` automatically. Outside Vernacula, pull
with `huggingface_hub` and load the four graphs with `onnxruntime`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/granite-speech-4-1-2b-onnx")
```

The unified decoder follows a single contract for both prefill and
step — see
[`scripts/granite_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export)
for the input/output tensor names, the audio-token-count formula
(`mel_length / 2 → projector blocks → 3 audio tokens / block`), and the
ASR prompt template.

## Limitations

Numerical behavior matches the upstream
[`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
model. Language coverage (English, French, German, Spanish,
Portuguese, Japanese), accuracy, and known failure modes inherit from
the [upstream model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b).

Specific to this ONNX repackaging:

- **Greedy-decode loops on near-silence segments.** On very short or
  near-silent VAD chunks the model can fall into a fixed
  short-period token loop (`"uh, uh, uh, …"`). Vernacula mitigates
  this with a runtime detector that forces EOS on the affected row
  after 3 cycles of any period in [1..4]. Without that mitigation, a
  single runaway row drags the whole batch to `max_new_tokens`. If you
  call the ONNX decoder directly, replicate the detector or pre-filter
  out chunks under ~0.3 s of speech.
- **No speaker attribution or word-level timestamps.** Those features
  are in the upstream `granite-speech-4.1-2b-plus` variant, which
  requires a different output graph and is not exported here yet.
- **Segments capped at ~6.7 minutes of audio per call** by the
  decoder's 4096-token context (with `max_new_tokens=256`). Pre-segment
  longer audio with VAD or diarization.

## Citation

```bibtex
@misc{granite-speech-4-1,
    author       = { IBM Granite Team },
    title        = { Granite Speech 4.1 (2B) },
    year         = 2025,
    url          = { https://huggingface.co/ibm-granite/granite-speech-4.1-2b },
    publisher    = { Hugging Face }
}
```

## Acknowledgments

- Original model: [IBM Granite Team](https://www.ibm.com/granite)
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: see the upstream
[`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/granite_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export) — the export pipeline that produced these files
- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — upstream model card
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
