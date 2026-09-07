---
license: mit
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - onnxruntime
  - automatic-speech-recognition
  - speaker-diarization
  - streaming
  - vibevoice
  - vernacula
base_model: microsoft/VibeVoice-ASR-Streaming-1.5B
language:
  - en
  - zh
  - es
  - pt
  - de
  - ja
  - ko
  - fr
  - ru
  - it
---

# VibeVoice-ASR-Streaming 1.5B — ONNX export for Vernacula

An ONNX export of [`microsoft/VibeVoice-ASR-Streaming-1.5B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B),
Microsoft's streaming speaker-attributed ASR model (Qwen2.5-1.5B decoder over a pair of causal
audio tokenizers), for use as an ASR backend in
[Vernacula](https://github.com/christopherthompson81/vernacula).

The model transcribes **who said what** as audio arrives, with no separate diarizer: it emits
one text chunk per 2.93 s of audio and marks speaker turns inline.

- **Conversion scripts:** [`scripts/vibevoice_streaming_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/vibevoice_streaming_export)
- **Investigation log:** [`docs/dev/vibevoice_asr_streaming_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/dev/vibevoice_asr_streaming_investigation.md)

## Files

| file | what it is |
|---|---|
| `audio_encoder.onnx` (+ `.data`) | acoustic + semantic tokenizer encoders and both connectors, float16. One fixed 83,200-sample window (3.47 s) in, 26 frames out. |
| `decoder_gqa.onnx` (+ `.data`) | Qwen2.5-1.5B decoder, INT8 weight-only, attention as `com.microsoft::GroupQueryAttention` over a shared float16 KV cache. |
| `export-report.json` | shapes, the streaming constants, and the prompt/special token ids the runtime needs. |
| `tokenizer.json`, `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | metadata carried through from the source checkpoint. |

## How it differs from the source checkpoint

- **Deterministic audio encoding.** Upstream samples the acoustic latents with Gaussian
  noise; this export uses the latent mean, which is bit-repeatable and sits inside the
  seed-to-seed variation of the original.
- **INT8 decoder weights.** Measured indistinguishable from float16 on transcript accuracy,
  while halving the weights.
- **Shared, pre-allocated KV cache.** VRAM is flat in recording length rather than growing,
  and throughput does not decay over a long file.

## Requirements and limits

- ONNX Runtime with the **CUDA** execution provider. `GroupQueryAttention` and the float16
  graphs are not supported on the CPU provider here.
- Peak VRAM about **7.4 GB**; real-time factor about **0.060** on an RTX 3090.
- The KV cache ceiling in `export-report.json` bounds recording length and is set to the
  checkpoint's trained context (65,536 positions, roughly 68 minutes of audio). The runtime refuses a
  longer recording rather than failing partway. Attention cost grows with the filled cache,
  so decoding slows as a long recording proceeds; VRAM does not.
- Speaker attribution at this size is **less reliable — on the test clip it labelled every turn Speaker 0**.

## Licence

MIT, matching the source checkpoint.
