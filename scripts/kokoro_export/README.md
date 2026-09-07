# Kokoro-82M → ONNX export

Exports the [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) StyleTTS2-based
TTS model to ONNX for use in Vernacula.

There is no official conversion script published alongside the
[onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
release — the HF/Xenova conversion shipped the artifacts but not the export pipeline. This
folder builds our own so we control the opset, I/O contract, and quantization.

## The STFT situation

The custom **STFT layer** is why this export is non-trivial. The model's default
complex-valued STFT **cannot be exported at all**: the legacy TorchScript exporter dies with
`Unknown number type: complex`, and the torch.export (dynamo) exporter dies earlier on a
data-dependent guard in the BERT attention mask. We therefore load with
`KModel(disable_complex=True)`, which swaps in a real-valued STFT that exports cleanly.

`disable_complex` is **not** bit-identical to the complex path — in the waveform domain it
looks like a large change (~5.5 dB "SNR"), but that is almost entirely vocoder phase
(iSTFTNet is a GAN vocoder; its output phase isn't uniquely determined). In the
phase-invariant **log-spectral** domain the difference is small (~0.37, vs ~0.77 for a single
frame of jitter), and A/B/C renderings are perceptually indistinguishable by ear.

**Validate with the right metric.** Do NOT validate with random token ids or waveform SNR —
random inputs push the duration predictor off-distribution and waveform SNR flags inaudible
phase differences as huge errors. `export_kokoro.py` captures the real `(input_ids, ref_s,
speed)` from an actual pipeline run and checks **log-spectral L1**. Full reasoning, with
numbers, is in [`docs/kokoro_onnx_investigation.md`](../../docs/kokoro_onnx_investigation.md)
(see also Adrian Lyjak's writeup: https://www.adrianlyjak.com/p/onnx/).

## Files

- `export_kokoro.py` — exports `kokoro.onnx` and validates it against the PyTorch reference
- `requirements.txt` — export dependencies

## Environment

```bash
python3 -m venv scripts/kokoro_export/.venv
scripts/kokoro_export/.venv/bin/pip install -r scripts/kokoro_export/requirements.txt
```

## Export

```bash
scripts/kokoro_export/.venv/bin/python scripts/kokoro_export/export_kokoro.py \
  --out external/kokoro_onnx --opset 17
```

Add `--skip-validate` to export without the parity check, or `--atol` to tune the
validation tolerance.

## ONNX Contract

Export target is `KModel.forward_with_tokens` — the G2P frontend (misaki) stays *outside*
the graph; the model takes already-tokenized ids.

| Name | Shape | dtype | Description |
|---|---|---|---|
| `input_ids` (in) | `[1, tokens]` | int64 | Padded token ids: `[0, *ids, 0]` |
| `ref_s` (in) | `[1, 256]` | float32 | Style/voice vector |
| `speed` (in) | `[1]` | float32 | Speech-rate multiplier |
| `audio` (out) | `[samples]` | float32 | 24 kHz waveform |

`tokens` and `samples` are dynamic axes.

## Publishing to HuggingFace

The bundle ships as
[`christopherthompson81/kokoro-82m-onnx`](https://huggingface.co/christopherthompson81/kokoro-82m-onnx)
(`kokoro.onnx` + `voices/*.bin` + `manifest.json`); the model card's source is
[`scripts/hf_readmes/kokoro-82m-onnx/README.md`](../hf_readmes/kokoro-82m-onnx/README.md), and
`Vernacula.Avalonia`'s Settings → Text-to-Speech tab downloads from this repo
(`ModelManagerService.KokoroRepoBase`).

```bash
# 1. Export the graph and the English voice packs into one folder
scripts/kokoro_export/.venv/bin/python scripts/kokoro_export/export_kokoro.py --out ~/models/kokoro --opset 17
scripts/kokoro_export/.venv/bin/python scripts/kokoro_export/export_voices.py --out ~/models/kokoro

# 2. Hash + upload. ⚠ --exclude the ONNX Runtime optimisation caches: a folder the app has
#    run from holds kokoro.opt.<ep>.<hash>.ort files (300 MB each) beside the graph, and
#    --all would ship them.
python scripts/make_manifest.py --model-dir ~/models/kokoro --all --exclude '*.ort' '*.use-ort'
python scripts/upload_to_hf.py \
    --model-dir ~/models/kokoro \
    --repo-id christopherthompson81/kokoro-82m-onnx \
    --exclude '*.ort' '*.use-ort' \
    --sync-readme --create-repo
```

The app's download list names the 28 English voices explicitly (`ModelManagerService.KokoroVoices`);
if `export_voices.py --all` ever ships more, add them there too or they will not be fetched.
