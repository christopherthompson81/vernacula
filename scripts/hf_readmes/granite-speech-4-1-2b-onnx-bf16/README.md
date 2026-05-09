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
  - bfloat16
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

# IBM Granite Speech 4.1 (2B) — BF16 mixed-precision ONNX export for Vernacula

ONNX export of [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
with the 1.84 B-parameter Granite-4.0 LLM decoder in **bfloat16**, packaged
for use with [Vernacula](https://github.com/christopherthompson81/vernacula)
and ONNX Runtime on hardware with native BF16 acceleration.

- **Conversion script:** [`scripts/granite_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
- **FP32 sibling bundle (default):** [`christopherthompson81/granite-speech-4-1-2b-onnx`](https://huggingface.co/christopherthompson81/granite-speech-4-1-2b-onnx)

## When to use this bundle

This is the **GPU bundle** for hardware with NVIDIA Ampere or newer
(compute capability ≥ 8.0). On those parts the LM decoder runs through
BF16 tensor cores and the ASR loop is meaningfully faster than the FP32
sibling. On older GPUs and on CPU-only systems use the FP32 bundle —
ORT's CPU EP currently lacks a `Where(BF16)` kernel that this graph
needs at the audio merge, so the BF16 bundle does not load on CPU
regardless of whether the host CPU has `avx512_bf16` or `amx_bf16` in
hardware. See the FP32 sibling bundle linked above for that path.

Vernacula auto-detects this with `HardwareInfo.SupportsBf16Acceleration`
and pulls whichever bundle is appropriate.

## Highlights

- **31× realtime on RTX 3090 (RTF 0.032)**, up from 23× (RTF 0.043) on the FP32 sibling on the same 10-min English clip — **1.37× faster end-to-end** despite encoder + projector staying FP32.
- **LM decoder ~1.7× faster.** Step-loop drops 40% (13.0 s → 7.8 s) and prefill drops 26% (2.3 s → 1.7 s) — the BF16-affected portion of the pipeline (prefill + step-loop) runs 38% faster overall. Encoder + projector are unchanged at ~5.2 s by design (see Mixed-precision design below).
- **KV cache memory halved.** Per-token KV at B=16 drops from 167 MB (FP32) to 84 MB (BF16). On smaller GPUs this unlocks higher batch caps; on a 3090 we already saturate the 16-row hard cap before the VRAM budget bites.
- **Bundle 39% smaller (5.3 GB vs 8.7 GB).** Decoder weights: 6.9 GB → 3.5 GB. Encoder/projector unchanged from the FP32 sibling.
- **Greedy-argmax fidelity preserved.** Word count on the 10 min en-US benchmark: 1521 (BF16) vs 1514 (FP32) — within 0.5%. Spot-checks find only minor word-level variations consistent with BF16 mantissa truncation; no semantic drift, no runaway-loop reappearance.

## Contents

| File | Purpose | Precision |
|---|---|---|
| `mel.onnx` | Log-Mel spectrogram frontend (16 kHz waveform → mel features) | FP32 (DSP, dtype-invariant) |
| `encoder.onnx` (+ `.data`) | Conformer acoustic encoder, full-attention rewrite | FP32 |
| `projector.onnx` | BLIP-2 Q-Former audio-to-text projector | FP32 |
| `decoder.onnx` (+ `.data`) | Unified prefill + step Granite-4.0 LM decoder with cumsum-gather audio merge; **BF16 weights and KV** | BF16 (FP32 audio_embeds in, FP32 logits out) |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer_config.json` | GPT-2-family ByteLevel BPE tokenizer assets | text |
| `chat_template.jinja` | Chat template (used to construct the ASR prompt) | text |
| `preprocessor_config.json`, `processor_config.json` | Mel/preprocessor parameters from upstream | text |
| `export-report.json` | Per-stage export timings + dtype/opset metadata | text |
| `manifest.json` | Per-file MD5 hashes (used by Vernacula's download verifier) | text |

## Mixed-precision design

The ONNX export traces the model loaded at BF16 then casts the encoder
and projector back to FP32 before tracing those graphs:

```python
model = GraniteSpeechForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16)
model.encoder = model.encoder.to(torch.float32)
model.projector = model.projector.to(torch.float32)
```

Two ORT BF16 gaps drove the encoder/projector → FP32 carve-out:

1. **Conv at opset 18** has no BF16 type binding; opset 22 added it but
   fails type-binding on the Conformer's depthwise convs at trace time.
2. **CPU EP `Where`** has no BF16 kernel registered — the audio merge
   uses `torch.where` and would fail to load even on hardware that has
   AVX-512 BF16 support, because the kernel isn't built into ORT's CPU EP
   regardless of host capability.

The decoder graph has FP32 boundary casts at exactly two places:

- **Input:** `audio_embeds` arrives as FP32 (projector output) and is cast to BF16 inside the decoder graph before the audio merge.
- **Output:** `logits` is cast back to FP32 so the C# argmax stays cheap and dtype-uniform with the FP32 bundle.

Past-KV stays in BF16 across the chained `Run` loop — GPU-resident, never read on the host. The C# side detects the dtype from the decoder's `InputMetadata["past_key_0"].ElementDataType` at session construction and creates the empty-prefill OrtValues via a switch on `Float / BFloat16 / Float16`.

## Performance

Measured on an RTX 3090 with Vernacula's batched ONNX pipeline:

| Workload | Mode | FP32 wall | **BF16 wall** | FP32 RTF | **BF16 RTF** |
|---|---|---:|---:|---:|---:|
| 6.4 s VCTK clip | B=1 | 3.25 s | 2.50 s | 0.508 | 0.392 |
| 600 s (10 min) VAD-segmented | mixed B (≤16) | 23.9 s | **17.4 s** | 0.043 | **0.032** |

Per-stage breakdown on the 10-min run (BF16):

| Stage | ms | % |
|---|---:|---:|
| mel | 534 | 3.4% |
| encoder | 5,191 | 33.4% |
| projector | 42 | 0.3% |
| prefill | 1,723 | 11.1% |
| step-loop | **7,755** | **49.9%** |
| overhead | 300 | 1.9% |

Encoder is unchanged from FP32 (intentional — see Mixed-precision design above). The win is concentrated in prefill + step-loop, which together account for ~60% of total ASR time at this precision.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited from the upstream [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) model. Permissive — commercial use is allowed with attribution.

## Using these files

In Vernacula, the auto-detect picks this bundle when both bundles are present on disk and the host has CUDA EP available with an Ampere+ GPU. The CLI flag is `--asr granite`; force a specific bundle with `--granite-model <path>` if you need to override the heuristic.

Outside Vernacula, pull with `huggingface_hub` and load the four graphs with `onnxruntime`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="christopherthompson81/granite-speech-4-1-2b-onnx-bf16")
```

The contract on the decoder graph differs from the FP32 sibling in two places that callers need to handle:

- **`audio_embeds` is still declared as FP32** at the graph boundary (cast to BF16 internally). No caller change needed there.
- **`past_key_<L>` and `past_value_<L>` are declared as BF16.** The empty-prefill tensors at the start of decode must be BF16, not FP32. ORT C# exposes this via `Microsoft.ML.OnnxRuntime.BFloat16`; ORT Python's BF16-numpy round-trip is unhelpful (numpy lacks a native BF16 dtype) and ML callers should keep KV as `OrtValue` objects without ever materialising on the host.

See [`scripts/granite_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export) for the full input/output tensor names and shapes, and [`docs/dev/granite_speech_bf16_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/dev/granite_speech_bf16_investigation.md) for the run-by-run journey including the ORT compatibility gaps that shaped the mixed-precision policy.

## Limitations

- **Hardware required: NVIDIA Ampere or newer (compute capability ≥ 8.0).** Older GPUs may load the bundle but fall back to slower BF16 emulation; CPU-only systems do not load at all due to the `Where(BF16)` kernel gap. For those use the FP32 sibling bundle.
- **Audio segments capped at ~6.7 minutes per call** by the decoder's 4096-token context (with `max_new_tokens=256`). Pre-segment longer audio with VAD or diarization.
- **Greedy decode loops on near-silence segments** — same model property as the FP32 bundle. Vernacula mitigates with a runtime period-1..4 loop detector that forces EOS on stuck rows. Non-Vernacula consumers should replicate that pattern or pre-filter silent VAD chunks.
- **No speaker attribution or word-level timestamps** — those are in the upstream `granite-speech-4.1-2b-plus` variant, which has not been exported.

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
- [`granite-speech-4-1-2b-onnx`](https://huggingface.co/christopherthompson81/granite-speech-4-1-2b-onnx) — FP32 sibling bundle (default for non-Ampere GPUs and CPU)
- [Conversion script (`scripts/granite_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/granite_export) — the export pipeline that produced these files
- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — upstream model card
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
