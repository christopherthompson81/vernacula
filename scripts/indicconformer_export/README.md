# IndicConformer ONNX Export

This folder will hold the ONNX export pipeline for AI4Bharat's IndicConformer
(Hybrid CTC-RNNT Conformer, 22 official Indian languages). It parallels
`scripts/nemo_export/` but uses AI4Bharat's NeMo fork rather than upstream
NeMo, so the two envs must stay isolated.

See [../../README.md](../../README.md) for the full project context and
[IndicConformer_plan.md](../../IndicConformer_plan.md) once promoted, or the
6-phase plan kept in `~/Programming/parakeet_csharp/data/IndicConformer_plan.md`.

## Status

**Phase 1 — Feasibility spike (in progress).** Nothing C#-side is worth
touching until CTC export + parity are proven.

## Environment

Use Python 3.10, 3.11, or 3.12. The AI4Bharat fork requires Python >= 3.10.

```bash
python3 scripts/indicconformer_export/setup_indicconformer_env.py
source .venv-indicconformer-export/bin/activate
```

This env is separate from `.venv-nemo-export/` on purpose: the fork pins
different NeMo/Transformer internals, and mixing them would break Parakeet
export.

Install with a specific CUDA build or CPU-only:

```bash
# CUDA 12.1
python3 scripts/indicconformer_export/setup_indicconformer_env.py --cuda-version cu121
# CPU only
python3 scripts/indicconformer_export/setup_indicconformer_env.py --cuda-version ""
```

## Phase 1 — Discovery

Before writing any exporter, load a checkpoint and dump its structure:

```bash
python scripts/indicconformer_export/inspect_indicconformer.py \
  --hf-repo ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large \
  --out /tmp/indicconformer_hi.json
```

What we're looking for in the report:

- **`has_ctc_head`** — must be true (CTC-head export is the whole plan)
- **`ctc_decoder_param_names`** — are there per-language weight tensors?
  That would confirm AI4Bharat's "multi-softmax" design
- **`tokenizer_langs` / `tokenizer_class`** — `AggregateTokenizer` with 22
  langs is the shape we expect for the unified model
- **`cfg_preprocessor`** — confirms we can reuse the DFT preprocessor from
  `nemo_export/` (same 80-mel Conformer frontend, presumably)

Once that report lands, Phase 1's open question — "how is language selected
at inference?" — is answered, and we can commit Phase 2's export shape
(single ONNX file with a language-id input vs. one ONNX per language).

## References

- [AI4Bharat/NeMo fork (nemo-v2 branch)](https://github.com/AI4Bharat/NeMo/tree/nemo-v2)
- [ai4bharat/IndicConformer (unified, 22 langs)](https://huggingface.co/ai4bharat/IndicConformer)
- [ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large (Hindi-only)](https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large)
- [AI4Bharat/indic-asr-api-backend](https://github.com/AI4Bharat/indic-asr-api-backend) — reference inference code
