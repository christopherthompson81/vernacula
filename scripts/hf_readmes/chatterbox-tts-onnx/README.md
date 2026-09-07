---
license: mit
library_name: onnxruntime
pipeline_tag: text-to-speech
tags:
  - onnx
  - onnxruntime
  - text-to-speech
  - voice-cloning
  - chatterbox
  - vernacula
base_model: ResembleAI/chatterbox
language:
  - en
---

# Chatterbox — ONNX export for Vernacula

A complete ONNX export of [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)
(Resemble AI's open zero-shot TTS: an 0.5B Llama-style speech-token LM over a CosyVoice-style
flow-matching vocoder), for use as the Chatterbox text-to-speech engine in
[Vernacula](https://github.com/christopherthompson81/vernacula).

- **Conversion script:** [`scripts/chatterbox_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/chatterbox_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)

## Highlights

- **All four stages in one reproducible recipe**, pinned to an upstream revision. The
  other ONNX sources either publish artifacts without the export (Resemble), a bundle
  assembled from several places (`onnx-community`), or three of the four graphs without
  the language model.
- **Language model with KV cache**, exported from scratch, so a rollout is one graph
  call per step.
- **Vocoder in both layouts:** a single merged graph that runs the whole CFM solve as an
  ONNX `Loop` (what the C# runtime loads), and the three split graphs it was built from
  (flow encoder, CFM estimator, mel-to-wave) for orchestrating the Euler loop outside the
  graph. Per-graph parity against upstream eager passes on all five checks.
- **Word timings for free:** the LM's cross-attention over the text tokens gives per-word
  alignment during synthesis, no forced-alignment pass — see
  [`docs/tts_alignment_desync_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/tts_alignment_desync_investigation.md).

## Contents

Every graph keeps its weights in an external-data sidecar named `<graph>.onnx_data`
(that spelling, with an underscore, is what the graphs reference — keep the pair
together).

| File | Purpose |
|---|---|
| `speech_encoder.onnx` (+ `.onnx_data`, 2.1 GB) | Reference clip → speaker embedding, speaker features, prompt speech tokens |
| `embed_tokens.onnx` (+ `.onnx_data`, 118 MB) | Text + speech token embedding |
| `language_model.onnx` (+ `.onnx_data`, 5.9 GB) | The speech-token LM, one step per call with KV cache in/out |
| `conditional_decoder_loop.onnx` (+ `.onnx_data`, 549 MB) | Merged vocoder: speech tokens + speaker conditioning → 24 kHz waveform, CFM solve as an ONNX `Loop` |
| `flow_encoder.onnx`, `cfm_estimator.onnx`, `mel2wav.onnx` (+ `.onnx_data`) | The same vocoder split in three, for driving the Euler loop yourself |
| `tokenizer.json` | Chatterbox's English text tokenizer (from the upstream repo, unchanged) |
| `export-report.json` | What was exported from where: upstream revision, opsets, per-graph hashes, environment |
| `manifest.json` | Per-file MD5 hashes for integrity checks |

The bundle is about 9.3 GB in fp32.

## Export provenance

Exported via [`scripts/chatterbox_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/chatterbox_export)
in the Vernacula repo (`export_chatterbox_to_onnx.py --device cuda --dtype float32`). The
export applies scoped, self-restoring patches to the upstream modules for tracing —
real-valued STFT/iSTFT in place of the complex ones, a dynamic-shape `solve_euler`,
deterministic noise — and records the pinned `ResembleAI/chatterbox` revision it read in
`export-report.json`. The design log is
[`docs/chatterbox_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/chatterbox_investigation.md);
the merged-`Loop` vocoder is Run 10 of
[`docs/chatterbox_perf_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/chatterbox_perf_investigation.md).

## License

[MIT](https://opensource.org/licenses/MIT), inherited from
[`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox). `tokenizer.json`
is upstream's file, redistributed unchanged.

**Not included:** upstream's Perth audio watermarker. The Python package applies it to
every output; this export is the synthesis path only. Please respect the upstream
project's guidance on responsible use of voice cloning.

## Using these files

In Vernacula, point **Settings → Text-to-Speech → Chatterbox** at a folder holding these
files (or use its Download button) and pick a reference clip per job. Outside Vernacula
the graphs need orchestration — speaker encoding, the LM rollout with its KV cache, then
the vocoder — which the C# `ChatterboxPipeline` in
[`src/Vernacula.Tts.Base`](https://github.com/christopherthompson81/vernacula/tree/main/src/Vernacula.Tts.Base)
implements end to end; it is the reference for the I/O names and shapes.

```python
from huggingface_hub import snapshot_download
import onnxruntime as ort

path = snapshot_download(repo_id="christopherthompson81/chatterbox-tts-onnx")
lm = ort.InferenceSession(f"{path}/language_model.onnx")
print([i.name for i in lm.get_inputs()])   # token embeddings + KV cache in; logits + KV cache out
```

## Limitations

English only. A rollout is capped at 1024 LM steps per chunk; Vernacula splits documents
into paragraphs first and flags a chunk that hits the cap. Everything else is upstream's —
see the [Chatterbox model card](https://huggingface.co/ResembleAI/chatterbox).

## Citation

See the [upstream model card](https://huggingface.co/ResembleAI/chatterbox) and
[github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox).

## Acknowledgments

- Original model: [Resemble AI](https://www.resemble.ai/) (Chatterbox)
- ONNX export: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: see the upstream model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/chatterbox_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/chatterbox_export)
- [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) — upstream model card
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
