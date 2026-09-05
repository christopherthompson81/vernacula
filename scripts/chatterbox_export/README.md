# Chatterbox → ONNX export

Exports ResembleAI's Chatterbox TTS into the four-graph ONNX package
consumed by `Vernacula.Tts.Base` (forthcoming) and eventually folded into
`Vernacula.Avalonia`. See [chatterbox.scratch.md](../../chatterbox.scratch.md)
for the project context and [docs/chatterbox_investigation.md](../../docs/chatterbox_investigation.md)
for the running design log.

**Status: Stage 0 complete.** All four graphs export, the cond decoder
is fully dynamic-shape (accepts arbitrary `speech_tokens` length), and
the per-graph parity suite passes 5/5 (lm, embed, enc, dec, solve_euler).
End-to-end audio is intelligible at native LM-emitted lengths with no
padding. Runs 1–11 of the investigation doc cover the methodology.

## Why we own this

The upstream provenance for Chatterbox's ONNX bundle is split between
ResembleAI (publishes artifacts, not export scripts), HF
`onnx-community` (publishes a Frankenstein bundle), and
[VladOS95-cyber's script](https://github.com/VladOS95-cyber/onnx_conversion_scripts/tree/main/chatterbox)
(covers three of the four graphs, omits the language model). We need a
single reproducible recipe anchored at a pinned `ResembleAI/chatterbox`
revision. Details: Run 1 in the investigation log.

## Files

- `export_chatterbox_to_onnx.py` — main export entry point. Default
  emits the flagship runtime bundle: `embed_tokens.onnx`,
  `speech_encoder.onnx`, `language_model.onnx`, `flow_encoder.onnx`
  + `cfm_estimator.onnx` + `mel2wav.onnx` (split for C# orchestration
  fallback), `conditional_decoder_loop.onnx` (merged-Loop, Path B /
  Run 10 of `docs/chatterbox_perf_investigation.md` — the C# Vocoder
  picks this at load time), plus `export-report.json`. Pass
  `--no-split-cond-decoder --no-merge-cond-decoder` to swap the split +
  merged graphs for a single `conditional_decoder.onnx` (smaller
  pre-perf-work bundle; useful on disk-constrained dev).
- `_chatterbox_internals.py` — thin export wrappers around upstream
  `s3gen` / `t3` modules (`PrepareConditionalsModel`, `InputsEmbeds`,
  `ConditionalDecoder`, `ISTFT`). Most of what used to be vendored
  here has been de-vendored to scoped patches; see Runs 6–7.
- `_export_patches.py` — context-managed monkey-patches for the
  trace path. Covers complex-tensor STFT/iSTFT replacements, the
  dynamic-shape `solve_euler` rewrite, deterministic `rand_noise` for
  reproducibility, and the inference-mode strip. Every patch
  restores its original on context exit.
- `_common.py` — shared helpers (device/dtype resolution, audio I/O
  at 24 kHz, ORT provider selection, `export-report.json` schema,
  KV-cache name conventions).
- `test_chatterbox_parity.py` — 5-test parity suite (lm, embed, enc,
  dec, solve_euler) against upstream eager. All pass.
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

## Usage

```bash
python scripts/chatterbox_export/export_chatterbox_to_onnx.py \
  --output-dir ./models/chatterbox_export \
  --device cuda \
  --dtype float32
```

Useful flags:

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
- `--audio-prompt <wav>` — use a real reference clip instead of the
  default 13s `torch.randn` dummy. Required for any parity work that
  depends on speaker identity matching upstream.
- `--no-onnxslim` — skip the post-export slim + external-data pass.
- `--overwrite` — replace existing files in `--output-dir`.

After export, verify:

```bash
python scripts/chatterbox_export/test_chatterbox_parity.py \
  --onnx-dir ./models/chatterbox_export --tests all
```

To audit an export for trace-time shape bakes:

```bash
python scripts/_export_utils/scan_bakes.py \
  ./models/chatterbox_export/conditional_decoder.onnx --suspect 505 1010
```

## Roadmap

See [docs/chatterbox_investigation.md](../../docs/chatterbox_investigation.md)
for the running log; the staged plan:

| Step | Status | Description |
|---|---|---|
| E1 | done | Scaffold CLI surface, `_common.py`, parity skeleton |
| E2 | done | Three graphs (embed_tokens, speech_encoder, conditional_decoder); `SafeDenseLayer` replaced with verified `_DenseLayerExportShim` (Run 5) |
| E3 | done | Llama LM export from scratch with KV-cache; emits `language_model.onnx` |
| E4 | done | Per-graph parity (5 tests, all PASS); dynamic-shape cond decoder (Run 10); rand_noise reproducibility (Run 11) |
| E5 | todo | `export-report.json` artifact hashing + opset/version manifest; CLI sweep tests; voice-cloning variance tests |

## Notes

- **Opset choices**: 20 (embed_tokens, speech_encoder), 18 (LM, cond
  decoder). Cond decoder needs ≥18 for Col2Im (used in the F.fold
  window_sumsquare iSTFT replacement).
- **`SafeDenseLayer` was dropped** in favor of `_DenseLayerExportShim`
  in `_export_patches.py` (verified by `parity_enc[onnx-vs-upstream]`).
  See Run 5 — the upstream "asserted-safe" substitution turned out to
  drift by 93% cosine when the model wasn't fully zero-input.
- **The Llama backbone** is composed from `chatterbox.t3.tfmr` +
  `chatterbox.t3.speech_head` at export time, not loaded from
  `vladislavbro/llama_backbone_0.5` (Vlad's personal HF mirror).
  Keeps our provenance chain anchored at ResembleAI.
- **Reproducibility:** every export bakes a canonical seeded
  `rand_noise` (the CFM ODE's initial latent). Upstream sets it as a
  plain attribute via `torch.randn(...)` at `__init__`, so without
  pinning, two consecutive exports produce different `.onnx` files
  (and parity tests are apples-to-oranges). See Run 11.
