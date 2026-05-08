# Granite Speech 4.1 → ONNX export

Exports `ibm-granite/granite-speech-4.1-2b` (Apache-2.0) into the ONNX
package consumed by Vernacula.

> **Status:** Python export landed and parity-green. The four ONNX
> graphs (encoder, projector, decoder_init, decoder_step) match the
> reference `transformers` forward to within fp32 numerical noise.
> See [`docs/dev/granite_speech_investigation.md`](../../docs/dev/granite_speech_investigation.md)
> Runs 1–3 for the architectural probes, the patches needed at trace
> time, and the masked_scatter→cumsum-gather workaround for the audio
> merge.

Tracking issue: [#28](https://github.com/christopherthompson81/vernacula/issues/28).
Standard cadence — Python export → Python parity → C# CLI + parity →
performance → C# GUI — with each step iterated independently as
needed.

## Files

- `export_granite_speech_to_onnx.py` — exports the four ONNX graphs +
  tokenizer/processor assets + `export-report.json`
- `test_parity.py` — loads the exported package and the reference
  model side-by-side, runs each piece on the same dummy input, and
  reports max-abs-diff per stage
- `inspect_granite_speech.py` — read-only architecture probe used in
  Run 2 of the investigation; safe to keep around for re-checks
  against future model revisions
- `requirements.txt` — Python dependencies for the export environment

Variants `granite-speech-4.1-2b-plus` (speaker-attributed + word-level
timestamps) and `granite-speech-4.1-2b-nar` (non-autoregressive) are
deferred until the base export's C# CLI integration lands. The `-plus`
variant needs a second output graph or post-processing path; the `-nar`
variant changes the decoder topology entirely (no AR KV cache) and is a
parallel pipeline rather than a flag on this exporter.

## Environment

Use Python `3.11` or `3.12`.

The base 4.1 model is Apache-2.0, so no `huggingface-cli login` is
required. Install dependencies:

```bash
python3 -m venv .venv-granite-export
source .venv-granite-export/bin/activate
pip install -r public/scripts/granite_export/requirements.txt
```

## Main export

```bash
python public/scripts/granite_export/export_granite_speech_to_onnx.py \
  --output-dir ./models/granite_speech_4_1_2b \
  --opset 18
```

Useful options (mirrors cohere_export):

- `--device cuda` — trace on GPU
- `--dtype float16` — shrink exported weights
- `--revision <commit>` — pin the HF snapshot
- `--overwrite` — replace an existing export
- `--skip-encoder`, `--skip-projector`, `--skip-decoder` — iterate on one
  piece at a time (useful when debugging)
- `--legacy-exporter` — fall back to TorchScript ONNX path

## ONNX graph contract

| ONNX | Inputs | Outputs |
|---|---|---|
| `encoder.onnx` | `input_features [B, T, 160]` float32 | `encoder_hidden [B, T, 1024]` |
| `projector.onnx` | `encoder_hidden [B, T, 1024]` | `audio_embeds [B, A, 2048]` |
| `decoder_init.onnx` | `input_ids [B, S]` int64, `audio_embeds [B, A, 2048]` float32, `attention_mask [B, S]` int64 | `logits [B, S, 100353]` + `present_key_<L>`/`present_value_<L>` for L in 0..39, each `[B, 4, S, 128]` |
| `decoder_step.onnx` | `input_id [B, 1]`, `attention_mask [B, T]`, `cache_position [1]`, `past_key_<L>`/`past_value_<L>` each `[B, 4, T-1, 128]` | `logits [B, 1, 100353]` + `present_key_<L>`/`present_value_<L>` each `[B, 4, T, 128]` |

Notes on the decoder_init contract:

- **No `audio_mask` input.** The audio merge uses `cumsum(input_ids ==
  audio_token)` to derive the per-position audio index, gather the
  audio embedding, and `torch.where` it into the LLM's text embeddings.
  This avoids the `masked_scatter`→`ScatterND` translation bug in
  torch 2.11 (see investigation Run 3) and removes the need for the
  caller to separately track audio padding.
- **Audio token id is hardcoded to 100352.** That's the position of
  `<|audio|>` in the Granite tokenizer; baked into the graph at trace
  time.
- **40 layers × 2 (K/V) = 80 KV outputs** with names `present_key_0`
  through `present_value_39`. Same names are used as inputs on the
  step graph (`past_key_0` / `past_value_0` etc.).

## Parity check

After exporting, validate each piece against the reference forward:

```bash
python public/scripts/granite_export/test_parity.py \
  --onnx-dir ./models/granite_speech_4_1_2b
```

Expected per-stage max-abs-diff (CPU fp32, 2 s dummy audio):

| Stage | Threshold |
|---|---|
| encoder         | ~3e-4 |
| projector       | ~1e-6 |
| decoder_init    | ~5e-5 (logits and KV) |
| decoder_step    | ~3e-5 (logits and KV) |

The encoder is two orders looser than the rest because the export
substitutes a manual `softmax((Q @ K^T) * scale + bias) @ V` for the
upstream's `F.scaled_dot_product_attention`; the converter doesn't
support 5-D SDPA. Mathematically identical, but accumulation order
differs. Acceptable for fp32 LM consumption.

## Architecture summary (driving the layout)

| Block | Shape |
|---|---|
| Mel frontend (host-side) | sr 16 kHz, n_fft 512, hop 160, win 400, 80 mels, frame-stacked to 160 dim |
| Encoder | 16-layer Conformer, hidden 1024, 8 heads × 128, conv kernel 15, output_dim 348 (training-only graphemic CTC head) |
| Projector | BLIP-2 Q-Former, 2 layers, 16 heads, 5× temporal downsample (3 trainable queries per 15-frame window) |
| Decoder | Granite-4.0-1b (40 layers, hidden 2048, 16 heads, GQA 4 KV heads, head_dim 128, vocab 100,353) |
| Audio token | id `100352` (`<|audio|>`) |
| Context window | **4096** tokens (not 128k — the speech checkpoint reset position embeddings) |

## Notes

- Targets `transformers >= 4.57` (matches the upstream config's
  `transformers_version: 4.57.6`).
- LoRA is already merged into the released checkpoint
  (`has_lora_adapter: false`) — no separate adapter loading.
- `tie_word_embeddings: false` — embed and LM-head are distinct
  matrices. The decoder pair currently ships two full copies of the
  ~7 GB LM weights; sharing them via Cohere's external-data rename
  trick is queued for a follow-up.
- Three patches at trace time, all justified in the investigation Run 3:
  encoder 5-D SDPA → manual math, LM `attn_implementation="eager"` to
  dodge a data-dependent guard in `sdpa_attention_forward`, and the
  audio-merge cumsum-gather-where workaround for the masked_scatter
  conversion bug.
- The Granite decoder applies four scalar multipliers
  (`attention_multiplier`, `embedding_multiplier`, `logits_scaling`,
  `residual_multiplier`) that are part of the base architecture. They
  trace through `transformers` automatically and parity is tight, so
  no special handling is needed.
- The mel frontend (`GraniteSpeechFeatureExtractor`) is intentionally
  NOT exported as ONNX. The C# runtime should reproduce
  torchaudio's mel + frame-stacking on the host.
