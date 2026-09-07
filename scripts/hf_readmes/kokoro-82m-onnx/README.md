---
license: apache-2.0
library_name: onnxruntime
pipeline_tag: text-to-speech
tags:
  - onnx
  - onnxruntime
  - text-to-speech
  - kokoro
  - styletts2
  - vernacula
base_model: hexgrad/Kokoro-82M
language:
  - en
---

# Kokoro-82M — ONNX export for Vernacula

Re-packaged ONNX export of [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)
(v1.0, StyleTTS2 / iSTFTNet, 24 kHz mono) plus its English voice packs in a flat binary
layout, for use as the Kokoro text-to-speech engine in
[Vernacula](https://github.com/christopherthompson81/vernacula).

- **Conversion script:** [`scripts/kokoro_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/kokoro_export)
- **Vernacula:** [github.com/christopherthompson81/vernacula](https://github.com/christopherthompson81/vernacula)
- **Upstream model:** [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)

## Highlights

- **Our own export pipeline**, not the `onnx-community` artifacts: we control the opset,
  the I/O contract and the validation metric. There is no upstream export script to
  reproduce theirs from.
- **Exported with `disable_complex=True`.** The model's default complex-valued STFT cannot
  be exported at all (the TorchScript exporter dies on `Unknown number type: complex`);
  the real-valued STFT that replaces it is not bit-identical in the waveform domain
  (vocoder phase is not uniquely determined) but sits at ~0.37 log-spectral L1 against
  PyTorch — below a single frame of jitter (~0.77) and inaudible in A/B listening. The
  full argument, with numbers, is in
  [`docs/kokoro_onnx_investigation.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/kokoro_onnx_investigation.md).
- **G2P stays outside the graph.** The model takes token ids; in Vernacula the phonemes
  come from [vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer)
  rendered into Kokoro's vocabulary, so the same frontend serves every engine.
- **Voice packs as flat float32**, indexed by phoneme-string length, readable without a
  tensor library.

## Contents

| File | Purpose |
|---|---|
| `kokoro.onnx` | The whole model: token ids + style vector + speed → 24 kHz waveform (fp32, ~310 MB, weights inlined) |
| `voices/<name>.bin` | One voice pack per voice: `510 × 256` float32, little-endian — row *n* is the style vector for a phoneme string of length *n + 1* |
| `manifest.json` | Per-file MD5 hashes for integrity checks |

### ONNX contract

| Name | Shape | dtype | Description |
|---|---|---|---|
| `input_ids` (in) | `[1, tokens]` | int64 | Padded token ids: `[0, *ids, 0]` |
| `style` (in) | `[1, 256]` | float32 | Style/voice vector (`ref_s`) |
| `speed` (in) | `[1]` | float32 | Speech-rate multiplier (1.0 = natural) |
| `audio` (out) | `[samples]` | float32 | 24 kHz waveform |

`tokens` and `samples` are dynamic. The context window is 510 tokens; split longer text
on sentence boundaries first.

### Voices

The 28 English voices of Kokoro v1.0 — American (`af_*` / `am_*`) and British
(`bf_*` / `bm_*`), the prefix selecting the accent's phonemization:

`af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky`
`am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa`
`bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis`

Upstream also ships voices for other languages (`export_voices.py --all`); they are not
included here because Vernacula's Kokoro frontend is English-only.

## Export provenance

Exported via [`scripts/kokoro_export/`](https://github.com/christopherthompson81/vernacula/tree/main/scripts/kokoro_export)
in the Vernacula repo: `export_kokoro.py` exports `KModel.forward_with_tokens` at opset 17
with `disable_complex=True` and validates it against the PyTorch reference on a real
`(input_ids, ref_s, speed)` capture using **log-spectral L1** (waveform SNR and random
token ids both give meaningless verdicts here — see the investigation doc);
`export_voices.py` flattens the `voices/*.pt` packs.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), inherited from
[`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M). The voice packs are
upstream's, redistributed unchanged in layout only.

## Using these files

In Vernacula, point **Settings → Text-to-Speech → Kokoro-82M** at a folder holding these
files (or use its Download button). Outside Vernacula:

```python
from huggingface_hub import snapshot_download
import numpy as np, onnxruntime as ort

path = snapshot_download(repo_id="christopherthompson81/kokoro-82m-onnx")
sess = ort.InferenceSession(f"{path}/kokoro.onnx")

# ids: Kokoro vocabulary ids for the phoneme string (see upstream / misaki)
ids = np.array([[0, *phoneme_ids, 0]], dtype=np.int64)
pack = np.fromfile(f"{path}/voices/af_heart.bin", dtype="<f4").reshape(510, 256)
style = pack[len(phoneme_ids) - 1][None, :]
audio, = sess.run(None, {"input_ids": ids, "style": style, "speed": np.array([1.0], np.float32)})
# audio: float32 mono at 24 kHz
```

## Limitations

English only, as packaged here. Inherits Kokoro-82M's own limits (see the
[upstream model card](https://huggingface.co/hexgrad/Kokoro-82M)); the ONNX export adds
the phase difference described above and nothing else. Word timing in Vernacula comes
from the model's predicted durations, which is exact to the frame.

## Citation

See the [upstream model card](https://huggingface.co/hexgrad/Kokoro-82M) and the
[StyleTTS 2 paper](https://arxiv.org/abs/2306.07691).

## Acknowledgments

- Original model: [hexgrad](https://huggingface.co/hexgrad) (Kokoro-82M), building on StyleTTS 2 and iSTFTNet
- ONNX repackaging: [Chris Thompson](https://github.com/christopherthompson81) for [Vernacula](https://github.com/christopherthompson81/vernacula)

Issues with the ONNX export specifically: open an issue on
[the Vernacula repo](https://github.com/christopherthompson81/vernacula/issues).
Issues with the underlying model: see the upstream model card.

## See also

- [Vernacula on GitHub](https://github.com/christopherthompson81/vernacula) — the speech pipeline app this package is built for
- [Conversion script (`scripts/kokoro_export/`)](https://github.com/christopherthompson81/vernacula/tree/main/scripts/kokoro_export)
- [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) — upstream model card
- [Other Vernacula model packages](https://huggingface.co/christopherthompson81)
