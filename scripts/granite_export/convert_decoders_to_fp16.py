#!/usr/bin/env python3
"""
Convert the Granite Speech decoder ONNX graphs to fp16 in place.

Only touches `decoder_init.onnx` and `decoder_step.onnx`. The encoder
and projector are left at fp32 — Granite's encoder is a 16-layer
Conformer with Shaw relpos attention, which has historically been
sensitive to fp16 dynamic-range issues (the additive relpos bias can
exceed fp16 max). We don't lose much by keeping it fp32: encoder is
~21 ms on a 3090 at 6.4 s of audio anyway.

I/O types stay fp32 (`keep_io_types=True`) so the surrounding C# /
Python runtime doesn't need to know about the precision change. The
graph adds Cast ops at the boundaries.

Memory savings on a 25 GB GPU:
  fp32 bundle: encoder 1.7 GB + decoder_init 7.0 GB + decoder_step 7.0 GB = 15.7 GB
  decoder-fp16: encoder 1.7 GB + decoder_init 3.5 GB + decoder_step 3.5 GB = 8.7 GB

This unblocks 90 s GPU inference (which OOMs at fp32 on a 3090).

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/convert_decoders_to_fp16.py \\
        --onnx-dir ./models/granite_speech_4_1_2b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument(
        "--files",
        nargs="+",
        default=["decoder_init.onnx", "decoder_step.onnx"],
        help="ONNX files to convert in place (default: the two decoders).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from onnxruntime.transformers.optimizer import optimize_model
    except ImportError as e:
        print(f"Missing dependency: {e}. pip install onnxruntime-tools or onnxruntime>=1.17", file=sys.stderr)
        return 2

    for filename in args.files:
        path = args.onnx_dir / filename
        if not path.exists():
            print(f"Skipping {filename} (not found)")
            continue

        data_path = path.with_suffix(path.suffix + ".data")
        size_before = path.stat().st_size + (data_path.stat().st_size if data_path.exists() else 0)

        print(f"Converting {filename} to fp16 (keep_io_types=True) ...")
        # opt_level=0: no fusion, just load/convert. We don't trust the
        # built-in GPT-2 fusion patterns to match Granite's exact graph
        # (different RoPE convention, different residual scalars). Pure
        # weight conversion is the safest first step.
        model = optimize_model(str(path), model_type="gpt2", opt_level=0)
        model.convert_float_to_float16(keep_io_types=True)

        # Save back. The fp16 LM weights are still ~3.7 GB which exceeds
        # the protobuf 2 GB single-file limit, so we MUST write with external
        # data. `all_tensors_to_one_file=True` consolidates into a single
        # `<name>.onnx.data` sidecar instead of the dynamo exporter's
        # legacy scatter-of-Constant-files layout.
        if data_path.exists():
            data_path.unlink()
        model.save_model_to_file(
            str(path),
            use_external_data_format=True,
            all_tensors_to_one_file=True,
        )

        size_after = path.stat().st_size + (data_path.stat().st_size if data_path.exists() else 0)
        print(f"  {size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB "
              f"({(size_after / size_before) * 100:.1f}% of fp32)")

    print("\nDone. Run test_parity.py and transcribe_smoke.py against the converted bundle "
          "before relying on the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
