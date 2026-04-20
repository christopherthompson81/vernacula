#!/usr/bin/env python3
"""
inspect_indicconformer.py

Phase 1 discovery script for IndicConformer. Loads a .nemo checkpoint via
the AI4Bharat NeMo fork and dumps the information we need to decide:

  1. Is there a usable CTC head (so we can skip RNNT export)?
  2. Is this a single-language softmax, a multi-softmax, or a shared-vocab
     single-softmax?
  3. What do the vocab(s) look like per language?
  4. What is the preprocessor configuration (so we can reuse the DFT
     preprocessor from nemo_export/)?

This script does NOT export anything. It only reads the checkpoint.

Usage:
    # Hindi-only checkpoint (simpler — use this for the initial spike)
    python scripts/indicconformer_export/inspect_indicconformer.py \
      --hf-repo ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large

    # Unified 22-language checkpoint (what we actually want to ship)
    python scripts/indicconformer_export/inspect_indicconformer.py \
      --hf-repo ai4bharat/IndicConformer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nemo", help="Path to a local .nemo file.")
    group.add_argument("--hf-repo", help="HF repo id to pull from (e.g. ai4bharat/IndicConformer).")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the full inspection report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Imports deferred so --help works without NeMo installed.
    import torch
    import nemo.collections.asr as nemo_asr

    if args.nemo:
        print(f"Restoring from local checkpoint: {args.nemo}")
        model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    else:
        print(f"Downloading from HF: {args.hf_repo}")
        model = nemo_asr.models.ASRModel.from_pretrained(args.hf_repo, map_location="cpu")

    model.eval()

    report: dict = {
        "model_class": type(model).__name__,
        "mro": [cls.__name__ for cls in type(model).__mro__],
    }

    # --- Decoders ------------------------------------------------------------
    has_ctc = hasattr(model, "ctc_decoder") or hasattr(model, "decoder") and "CTC" in type(getattr(model, "decoder", None)).__name__
    has_rnnt = hasattr(model, "decoding") and hasattr(model, "joint")
    report["has_ctc_head"] = has_ctc
    report["has_rnnt_head"] = has_rnnt
    report["cfg_decoder"] = _safe_cfg(model, "cfg.decoder")
    report["cfg_joint"] = _safe_cfg(model, "cfg.joint")
    report["cfg_ctc_decoder"] = _safe_cfg(model, "cfg.ctc_decoder")
    report["cfg_aux_ctc"] = _safe_cfg(model, "cfg.aux_ctc")

    # --- Language heads / multi-softmax --------------------------------------
    # AI4Bharat's fork advertises "multi-softmax" — look for per-language
    # decoder state_dict keys or a `language_masking_prob` style config.
    ctc_decoder = getattr(model, "ctc_decoder", None) or getattr(model, "decoder", None)
    if ctc_decoder is not None:
        report["ctc_decoder_class"] = type(ctc_decoder).__name__
        # Flatten parameter names so we can see per-language keys if they exist.
        param_names = [n for n, _ in ctc_decoder.named_parameters()]
        report["ctc_decoder_param_names"] = param_names[:40]  # cap for sanity
        report["ctc_decoder_param_count"] = len(param_names)

    # --- Vocab(s) ------------------------------------------------------------
    # Single-language CTC: model.decoder.vocabulary is a flat list
    # Multi-softmax: vocabularies live per-language, possibly on a tokenizer
    # with language_id indexing.
    vocab = None
    try:
        vocab = ctc_decoder.vocabulary
    except AttributeError:
        pass
    if vocab is not None:
        report["ctc_vocab_size"] = len(vocab)
        report["ctc_vocab_preview"] = list(vocab)[:20]

    if hasattr(model, "tokenizer"):
        tok = model.tokenizer
        report["tokenizer_class"] = type(tok).__name__
        report["tokenizer_attrs"] = [a for a in dir(tok) if not a.startswith("_")][:40]
        # AggregateTokenizer (multi-lang) has .langs / .tokenizers_dict
        if hasattr(tok, "langs"):
            report["tokenizer_langs"] = list(tok.langs)

    # --- Preprocessor --------------------------------------------------------
    report["cfg_preprocessor"] = _safe_cfg(model, "cfg.preprocessor")

    # --- CTC inference smoke (no audio) --------------------------------------
    # Just confirm we can switch decoder modes without crashing.
    try:
        model.cur_decoder = "ctc"
        report["can_set_cur_decoder_ctc"] = True
    except Exception as exc:
        report["can_set_cur_decoder_ctc"] = False
        report["set_cur_decoder_ctc_error"] = repr(exc)

    # --- Print ---------------------------------------------------------------
    pretty = json.dumps(report, indent=2, default=str)
    print("\n===== IndicConformer inspection report =====")
    print(pretty)

    if args.out:
        Path(args.out).write_text(pretty)
        print(f"\nWrote {args.out}")


def _safe_cfg(model, dotted_path: str):
    cur = model
    for part in dotted_path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    # OmegaConf DictConfig -> plain dict
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cur, resolve=True)
    except Exception:
        return str(cur)


if __name__ == "__main__":
    main()
