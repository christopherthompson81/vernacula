# Granite Speech 4.1 → ONNX export

Exports `ibm-granite/granite-speech-4.1-2b` (Apache-2.0) into the ONNX
package consumed by Vernacula.

> **Status: skeleton.** The export script is currently a stub that
> validates the architecture and prints the planned graph layout. It
> does not yet emit ONNX. See [`docs/dev/granite_speech_investigation.md`](../../docs/dev/granite_speech_investigation.md)
> Run 1 for the architecture findings driving this layout, and the
> sequenced plan for the issue split (export → parity → C# CLI/parity →
> performance → C# GUI).

Tracking issue: [#28](https://github.com/christopherthompson81/vernacula/issues/28).

## Files

- `export_granite_speech_to_onnx.py` — entry point. Currently a stub
  that loads the model and prints the planned ONNX graph layout. Will
  emit `mel.onnx`, `encoder.onnx`, `projector.onnx`, `decoder_init.onnx`,
  `decoder_step.onnx`, tokenizer assets, and `export-report.json`.
- `requirements.txt` — Python dependencies for the export environment.

Variants `granite-speech-4.1-2b-plus` (speaker-attributed + word-level
timestamps) and `granite-speech-4.1-2b-nar` (non-autoregressive) are
deferred until the base export is parity-validated. The `-plus` variant
needs a second output graph or post-processing path; the `-nar` variant
changes the decoder topology entirely (no AR KV cache) and is a parallel
pipeline rather than a flag on this exporter.

## Environment

Use Python `3.11` or `3.12`.

The base 4.1 model is Apache-2.0, so no `huggingface-cli login` is
required. Install dependencies:

```bash
python3 -m venv .venv-granite-export
source .venv-granite-export/bin/activate
pip install -r public/scripts/granite_export/requirements.txt
```

## Planned graph layout

| ONNX | Inputs | Outputs |
|---|---|---|
| `mel.onnx` | `audio [batch, samples]` | `mel [batch, 80, T]` |
| `encoder.onnx` | `mel [batch, 80, T]` | `acoustic [batch, T/2, 1024]` (frame-stack baked in) |
| `projector.onnx` | `acoustic [batch, T/2, 1024]` | `audio_embeds [batch, T/10, 2048]` (BLIP-2 Q-Former, 5× downsample, 3 queries × 15-frame windows) |
| `decoder_init.onnx` | `input_ids`, `audio_embeds`, `audio_mask` | `logits`, `present_kv` (40 layers × split K/V × GQA 4 heads × seq × 128 dim) |
| `decoder_step.onnx` | `input_id`, `past_kv` | `logits`, `present_kv` |

Whether `projector.onnx` stays separate or fuses into the encoder is a
post-spike call; see investigation doc Run 1 open questions.

## Architecture summary (driving the layout)

| Block | Shape |
|---|---|
| Mel frontend | sr 16 kHz, n_fft 512, hop 160, win 400, 80 mels |
| Encoder | 16-layer Conformer, hidden 1024, 8 heads × 128, conv kernel 15, output_dim 348 (graphemic CTC head) |
| Frame stacking | encoder `input_dim 160 = 80 × 2`; adjacent-frame stack before encoder |
| Projector | BLIP-2 Q-Former, 2 layers, 16 heads, hidden 1024, 5× temporal downsample, 3 trainable queries per 15-frame window |
| Decoder | Granite-4.0 (40 layers, hidden 2048, 16 heads, 4 KV heads / GQA, head_dim 128, vocab 100,353) |
| Audio token | id `100352` (`<|audio|>`) |
| Context window | **4096** tokens (not 128k — speech checkpoint reset position embeddings) |

The "dual-head CTC encoder" wording in [#28](https://github.com/christopherthompson81/vernacula/issues/28)
appears to describe a *training-time* loss shape, not two parallel
encoder outputs at inference time. The 348-dim graphemic head is the
encoder's only ONNX output; the BPE head with vocab 100,353 is the
decoder LM head. To be confirmed in Run 2 by tracing the public model's
forward.

## Usage (intended; stub today)

```bash
python public/scripts/granite_export/export_granite_speech_to_onnx.py \
  --output-dir ./models/granite_speech_4_1_2b \
  --opset 18
```

Planned options (mirror cohere_export):

- `--device cuda` — export on GPU
- `--dtype float16` — shrink exported weights
- `--revision <commit>` — pin the HF snapshot
- `--overwrite` — replace an existing export
- `--skip-encoder`, `--skip-decoder`, `--skip-mel`, `--skip-projector`
- `--legacy-exporter` — fall back to TorchScript ONNX path

## Notes

- Targets `transformers >= 4.57` (matches the upstream config's
  `transformers_version: 4.57.6`).
- LoRA is already merged into the released checkpoint
  (`has_lora_adapter: false`) — no separate adapter loading.
- `tie_word_embeddings: false`, so embedding and LM-head weights are
  distinct. The Cohere-style trick of sharing `decoder_init`'s
  external-data file with `decoder_step` still applies between the two
  decoder graphs.
- The Granite decoder applies four scalar multipliers
  (`attention_multiplier`, `embedding_multiplier`, `logits_scaling`,
  `residual_multiplier`) that are part of the base architecture. They
  trace through `transformers` automatically; flagged here as a parity
  watch item.
