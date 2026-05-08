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
- `test_parity.py` — per-stage smoke parity: runs each ONNX graph on
  dummy input alongside the reference forward and reports max-abs-diff
- `transcribe_smoke.py` — end-to-end transcription parity: runs the
  full ORT pipeline (encoder → projector → decoder_init → step loop)
  on a real audio clip and compares against `model.generate()`. Catches
  integration bugs across the full pipeline that per-stage parity
  cannot.
- `test_encoder_math_equivalence.py` — regression check that the Run 4
  full-attention encoder rewrite is mathematically identical to the
  upstream block-attention encoder. Loads the model twice (unpatched +
  patched) and compares `model.encoder(features)` outputs directly.
  Expected max-abs-diff ~1e-5 (pure fp32 noise). The other parity
  scripts compare ORT vs PyTorch *both running the patched math*, so
  they cannot detect a math bug introduced by the patch itself.
- `dump_inputs_for_csharp_smoke.py` — generates fixtures
  (`input_ids.bin`, `expected_text.txt`) for the C# CLI parity smoke
  in `tests/GraniteSpeechSmoke/`.
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
| `mel.onnx` | `audio [B, samples]` float32 | `input_features [B, T, 160]` |
| `encoder.onnx` | `input_features [B, T, 160]` float32 | `encoder_hidden [B, T, 1024]` |
| `projector.onnx` | `encoder_hidden [B, T, 1024]` | `audio_embeds [B, A, 2048]` |
| `decoder_init.onnx` (split mode) | `input_ids [B, S]` int64, `audio_embeds [B, A, 2048]` float32, `attention_mask [B, S]` int64 | `logits [B, S, 100353]` + `present_key_<L>`/`present_value_<L>` for L in 0..39, each `[B, 4, S, 128]` |
| `decoder_step.onnx` (split mode) | `input_id [B, 1]`, `attention_mask [B, T]`, `cache_position [1]`, `past_key_<L>`/`past_value_<L>` each `[B, 4, T-1, 128]` | `logits [B, 1, 100353]` + `present_key_<L>`/`present_value_<L>` each `[B, 4, T, 128]` |
| `decoder.onnx` (unified mode, **recommended**) | `input_ids [B, S]`, `audio_embeds [B, A>=1, 2048]`, `attention_mask [B, T]`, `cache_position [S]`, `past_key_<L>`/`past_value_<L>` each `[B, 4, past_len, 128]` (past_len=0 at prefill) | `logits [B, S, 100353]` + `present_key_<L>`/`present_value_<L>` each `[B, 4, T, 128]` |

**Pass `--unified-decoder` to export the single `decoder.onnx`** instead
of the split init/step pair. The unified graph handles both prefill
(zero-length past_kv) and step (populated past_kv) modes through the
same set of inputs. It's:

- **Smaller on disk and resident weight footprint:** 7 GB vs 14 GB
  (split has two copies of the 1.84 B-param LM).
- **Faster at long audio:** 5.4× realtime on a 3090 at 90 s, vs 2.0×
  for the fp16 split bundle. fp16 Cast overhead at the boundaries
  cancels its kernel speedup; unified fp32 wins on cache locality.
- **No parity loss.** fp16 dropped a single filler token at position
  173 of the 90 s clip; unified fp32 preserves it.

See [`docs/dev/granite_speech_perf_investigation.md`](../../docs/dev/granite_speech_perf_investigation.md)
Run 3 for the full perf and parity numbers.

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

Two stages: per-graph numerical parity (`test_parity.py`) and
end-to-end transcription parity (`transcribe_smoke.py`). Run both.

### Per-stage numerical parity

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
upstream's `F.scaled_dot_product_attention`. Mathematically identical,
but accumulation order differs. Acceptable for fp32 LM consumption.

### End-to-end transcription parity

```bash
python public/scripts/granite_export/transcribe_smoke.py \
  --onnx-dir ./models/granite_speech_4_1_2b \
  --audio /path/to/clip.wav \
  --max-new-tokens 64
```

Runs the full ORT pipeline on the audio and compares the decoded
transcript against `model.generate(..., do_sample=False, num_beams=1)`.
Validated on a 6.4 s VCTK clip (multi-block) with exact text match;
also on a 3.5 s clip (single-block). Use `--skip-reference` to skip the
reference run when iterating on the ORT path only.

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
- Four patches at trace time, all justified in the investigation:
  - **Encoder full attention** (Run 4): replaces upstream block
    attention with full attention + block-diagonal mask, because
    `num_blocks` cannot be made symbolic through the dynamo exporter.
    Mathematically identical, ~7-15× more attention work at long T.
  - **Encoder 5-D SDPA → manual math** (Run 3): no longer needed after
    Run 4's full-attention rewrite (which is naturally 4-D).
  - **LM `attn_implementation="eager"`** (Run 3): dodges a
    data-dependent guard in `sdpa_attention_forward`.
  - **Audio merge: cumsum-gather-where** (Run 3): works around the
    `masked_scatter` → `ScatterND` conversion bug in torch 2.11.
- The Granite decoder applies four scalar multipliers
  (`attention_multiplier`, `embedding_multiplier`, `logits_scaling`,
  `residual_multiplier`) that are part of the base architecture. They
  trace through `transformers` automatically and parity is tight, so
  no special handling is needed.
- `mel.onnx` exports the upstream `GraniteSpeechFeatureExtractor`
  pipeline (torchaudio MelSpectrogram + log10/-8 dB clamp + frame
  stack). Mel parity is ~5e-5 vs the reference processor; including
  it in the bundle removes the need for a host-side torchaudio port
  and matches the precedent in cohere_export / nemo_export.
