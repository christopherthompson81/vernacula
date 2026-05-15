# Chatterbox → ONNX export

Exports ResembleAI's Chatterbox TTS into the four-graph ONNX package
consumed by `Chatterbox.Base` (forthcoming) and eventually folded into
`Vernacula.Avalonia`. See [chatterbox.scratch.md](../../chatterbox.scratch.md)
for the project context and [docs/chatterbox_investigation.md](../../docs/chatterbox_investigation.md)
for the running design log.

**Status: scaffold (Stage 0 step E1).** The CLI surface and dependency
layout are in place; no graphs are wired up yet. `python
export_chatterbox_to_onnx.py --output-dir ...` prints a placeholder run
plan and emits a stub `export-report.json`.

## Why we own this

The upstream provenance for Chatterbox's ONNX bundle is split between
ResembleAI (publishes artifacts, not export scripts), HF
`onnx-community` (publishes a Frankenstein bundle), and
[VladOS95-cyber's script](https://github.com/VladOS95-cyber/onnx_conversion_scripts/tree/main/chatterbox)
(covers three of the four graphs, omits the language model). We need a
single reproducible recipe anchored at a pinned `ResembleAI/chatterbox`
revision. Details: Run 1 in the investigation log.

## Files

- `export_chatterbox_to_onnx.py` — main export entry point. Emits
  `embed_tokens.onnx`, `speech_encoder.onnx`, `language_model.onnx`,
  `conditional_decoder.onnx`, and `export-report.json`.
- `_common.py` — shared helpers (device/dtype resolution, audio I/O at
  24 kHz, ORT provider selection, `export-report.json` schema,
  KV-cache name conventions).
- `test_chatterbox_parity.py` — three-layer parity check (LM token
  sequence, decoder waveform spectral distance, end-to-end audio).
- `requirements.txt` — pinned dependencies. The `chatterbox-tts`
  package and `transformers==4.46.3` are load-bearing; newer
  transformers versions may break the Chatterbox internals
  re-imported by the wrapper modules.

## Environment

Use Python `3.11` or `3.12`.

```bash
python3 -m venv .venv-chatterbox-export
source .venv-chatterbox-export/bin/activate
pip install -r scripts/chatterbox_export/requirements.txt
```

If the upstream `ResembleAI/chatterbox` HF repo becomes gated:

```bash
huggingface-cli login
```

## Usage (current scaffold)

```bash
python scripts/chatterbox_export/export_chatterbox_to_onnx.py \
  --output-dir ./models/chatterbox_export \
  --device cuda \
  --dtype float32 \
  --opset-language-model 18
```

Useful flags (all wired up; behavior arrives stepwise):

- `--repo-id <id>` — defaults to `ResembleAI/chatterbox`. Use
  `ResembleAI/chatterbox-turbo` once we add turbo support.
- `--revision <commit-or-tag>` — pin exact model snapshot. Recorded in
  `export-report.json`.
- `--dtype {float32,float16,bfloat16}` — float32 for parity testing,
  float16 for the 3090 ship target.
- `--lm-graph-mode {unified,prefill+step}` — unified is the
  reference-script style; prefill+step matches Vernacula's vibevoice
  export pattern and is what we'll likely ship.
- `--skip-{embed-tokens,speech-encoder,language-model,conditional-decoder}`
  — re-run a single graph.
- `--no-onnxslim` — skip the post-export slim + external-data pass.
- `--overwrite` — replace existing files in `--output-dir`.

## Roadmap

See [docs/chatterbox_investigation.md](../../docs/chatterbox_investigation.md)
for the running log; the staged plan:

| Step | Status | Description |
|---|---|---|
| E1 | done | Scaffold CLI surface, `_common.py`, parity skeleton |
| E2 | todo | Adapt Vlad's three graphs (embed_tokens, speech_encoder, conditional_decoder); validate `SafeDenseLayer` substitution; resolve `.onnx_data` sidecar question |
| E3 | todo | Write the Llama LM export from scratch with KV-cache; emit `language_model.onnx` (and optionally `language_model_prefill.onnx` + `language_model_step.onnx`) |
| E4 | todo | End-to-end parity test (token sequence, decoder spectral distance, full pipeline) |
| E5 | todo | `export-report.json` finalization: source SHA, file hashes, opset table, tool versions |

## Notes

- **Opset choices** mirror Vlad's reference (20 / 20 / 17) on the three
  graphs he covers, plus 18 for the LM to match Vernacula's existing
  Llama exports. Try bumping the conditional decoder to 18+ during E2 —
  Vlad's choice of 17 was driven by `ISTFT` op support and may no
  longer be required.
- **The `SafeDenseLayer` monkeypatch** (BatchNorm1d → LayerNorm on
  `s3gen.speaker_encoder.xvector.dense`) is required for the
  speech_encoder graph to export. The substitution is asserted-safe at
  inference time by upstream; the E2 parity step validates this
  numerically rather than trusting the assertion.
- **The Llama backbone** for E3 is loaded via the `chatterbox-tts`
  package's internals, not from `vladislavbro/llama_backbone_0.5`
  (Vlad's personal HF mirror). Keeps our provenance chain anchored at
  ResembleAI.
