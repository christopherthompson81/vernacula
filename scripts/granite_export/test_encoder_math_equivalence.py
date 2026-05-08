#!/usr/bin/env python3
"""
Regression check: my full-attention encoder vs the upstream block attention.

The Run 4 patch in `_patch_encoder_attention_for_export` replaces Granite's
block-windowed attention with full attention plus a block-diagonal additive
mask so the dynamo->ONNX exporter doesn't bake `num_blocks` as a static
trace constant. Mathematically the two should be identical (within fp32
accumulation noise); this script verifies that claim by loading the model
twice — once unpatched, once patched — and comparing the encoder output on
the same input.

The other parity scripts (`test_parity.py`, `transcribe_smoke.py`) compare
ORT vs PyTorch *both running the patched math*, so they cannot detect a
math bug introduced by the patch itself. This one can.

Run on a multi-block input (>=ctx*2 = 400 stacked frames) so the cross-block
mask actually exercises non-trivial code paths. The default 6.4 s VCTK clip
yields 321 stacked frames → padded to 400 → 2 blocks.

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/test_encoder_math_equivalence.py \\
        --audio /path/to/clip.wav

Expected: max-abs-diff ~1e-5 (pure fp32 accumulation noise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model-repo", default="ibm-granite/granite-speech-4.1-2b")
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Fail if max-abs-diff exceeds this. fp32 noise is ~1e-5; 1e-3 is a generous floor.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import soundfile as sf
        import torch
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    # Local import: pulls _patch_encoder_attention_for_export from the export script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_granite_speech_to_onnx import _patch_encoder_attention_for_export

    processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision)
    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    proc_in = processor("dummy", audio, return_tensors="pt")
    features = proc_in["input_features"]
    print(f"input_features: {tuple(features.shape)} ({features.shape[1]} stacked frames)")

    # Two independent loads so the patch in one doesn't bleed into the other.
    print("Loading unpatched model (upstream block attention) ...")
    m_block = GraniteSpeechForConditionalGeneration.from_pretrained(
        args.model_repo,
        revision=args.revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        out_block = m_block.encoder(features)

    print("Loading patched model (full attention) ...")
    m_full = GraniteSpeechForConditionalGeneration.from_pretrained(
        args.model_repo,
        revision=args.revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    _patch_encoder_attention_for_export(torch, m_full.encoder)
    with torch.no_grad():
        out_full = m_full.encoder(features)

    diff = (out_full - out_block).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\n  block vs full  max-abs-diff: {max_diff:.3e}")
    print(f"  block vs full mean-abs-diff: {mean_diff:.3e}")

    ok = max_diff <= args.threshold
    print(f"\n  pass (max <= {args.threshold:.0e}): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
