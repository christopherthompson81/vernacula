# VoxLingua107 Language-ID Export

Exports [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
into a single end-to-end ONNX graph for use as the language-identification
backend in Vernacula.

The model is an ECAPA-TDNN classifier over 80-dim mel filterbank features,
trained on the VoxLingua107 corpus. It takes raw 16 kHz mono audio and emits
both a 107-class logit vector (for language classification) and a 192-dim
pooled embedding (useful for language-similarity clustering). License is
Apache 2.0, so it can be bundled or auto-downloaded without a gated-accept
flow.

## Files

- `export_voxlingua_to_onnx.py` — exports `voxlingua107.onnx` + `lang_map.json`
- `verify_voxlingua_parity.py` — PyTorch ↔ ONNX parity check on real audio clips
- `src/ecapa_wrapper.py` — thin `nn.Module` composing SpeechBrain's submodules for a clean export graph
- `src/lang_map.py` — builds the `{index: {iso, name}}` lookup that ships alongside the ONNX
- `requirements.txt` — Python dependencies for the export environment

## Output graph

Single file, no external data.

| Tensor       | Shape              | Dtype   | Notes                                  |
|--------------|--------------------|---------|----------------------------------------|
| `audio`      | `[batch, samples]` | float32 | 16 kHz mono, variable length           |
| `logits`     | `[batch, 107]`     | float32 | Language classification logits         |
| `embedding`  | `[batch, 256]`     | float32 | Pooled ECAPA embedding                 |

Preprocessing (FBANK, per-utterance mean-variance normalisation) is folded
into the graph, so the C# runtime just sends raw PCM.

## Environment

Use Python `3.11` or `3.12`.

```bash
python3 -m venv .venv-voxlingua-export
source .venv-voxlingua-export/bin/activate
pip install -r public/scripts/voxlingua107_export/requirements.txt
```

## Export

```bash
python public/scripts/voxlingua107_export/export_voxlingua_to_onnx.py \
    --out-dir ./voxlingua107
```

This produces:

```
voxlingua107/
├── voxlingua107.onnx         # ~700 KB ONNX structure
├── voxlingua107.onnx.data    # ~82 MB external weights (FP32)
└── lang_map.json             # 107 entries
```

PyTorch's dynamo-based exporter (`torch>=2.8`) splits weights into a
companion `.onnx.data` file by default. Both files must ship together; ORT
resolves the reference automatically as long as they sit side-by-side.

The script downloads the SpeechBrain checkpoint into `./.voxlingua107-cache/`
on first run; subsequent runs reuse the cached weights.

## Parity verification

Prepare a handful of 16 kHz mono WAV clips across several languages (ten clips
spanning `en, zh, hi, ar, es, de, fr, ru, ja, sw` is a good baseline), then:

```bash
python public/scripts/voxlingua107_export/verify_voxlingua_parity.py \
    --model-dir ./voxlingua107 \
    --clips clip_en.wav clip_zh.wav clip_hi.wav ...
```

Pass criteria:

- Top-1 language matches between PyTorch and ONNX on every clip
- Softmax probability max-abs-diff ≤ `1e-3` (scale-invariant; raw-logit diff
  grows with clip length because of FP32 accumulation, so we compare probs
  rather than logits)
- Embedding cosine similarity ≥ `0.9999`

Observed values on a 5-clip test set (en, de, fr, ru, hu, durations 90 s–
602 s) are Δprob `3e-11` to `6e-5` and cosine `1.000000` across the board,
so the gates have comfortable headroom.

Failures print the reason(s) and return a non-zero exit code so the check
can gate CI if needed.
