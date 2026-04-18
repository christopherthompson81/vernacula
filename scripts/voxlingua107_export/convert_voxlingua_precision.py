"""Produce FP16 and INT8 variants of the FP32 VoxLingua107 ONNX.

Outputs:
    <out-dir>/voxlingua107_fp16.onnx   (CUDA target)
    <out-dir>/voxlingua107_int8.onnx   (CPU target, dynamic quantization)

Each is re-saved inline (no external data sidecar) to match the
single-file shipping pattern the FP32 export uses.

Usage:
    python convert_voxlingua_precision.py \\
        --fp32 ./voxlingua107/voxlingua107.onnx \\
        --out-dir ./voxlingua107
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic


def save_inline(model: onnx.ModelProto, out_path: Path) -> None:
    """Save as a single file (no external-data sidecar)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # save_as_external_data=False forces inline weights.
    onnx.save(model, str(out_path), save_as_external_data=False)


def convert_fp16(fp32_path: Path, out_path: Path) -> None:
    print(f"[fp16] loading {fp32_path}", file=sys.stderr)
    model = onnx.load(str(fp32_path))
    # keep_io_types=True: graph still accepts fp32 audio in, fp32 logits out.
    # Internal compute runs in fp16; input/output casts are auto-inserted.
    #
    # op_block_list: leave these in fp32. These ops either have implicit dtype
    # contracts the converter mis-handles (scalar helpers, shape ops, range
    # builders used in the dynamo-exported preprocessing graph) or produce
    # outputs consumed by shape-sensitive consumers.
    op_block_list = [
        "ConstantOfShape", "Range", "ArgMax", "Equal", "Where",
        "Expand", "Shape", "Cast",
    ]
    # Also block specific Cast nodes by name — the dynamo-exported graph has a
    # scalar_tensor helper whose fp16 output breaks a downstream fp32 consumer.
    node_block_list = [
        n.name for n in model.graph.node
        if n.op_type == "Cast" and "scalar_tensor" in n.name.lower()
    ]
    fp16_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
    )
    save_inline(fp16_model, out_path)
    print(f"[fp16] wrote {out_path} ({out_path.stat().st_size // 1024 // 1024} MB)",
          file=sys.stderr)


def convert_int8(fp32_path: Path, out_path: Path) -> None:
    print(f"[int8] quantizing {fp32_path} → {out_path}", file=sys.stderr)
    # Dynamic quantization: weights quantized ahead of time, activations quantized
    # per-call. Fine for conv/matmul-heavy models on CPU; no calibration needed.
    # QUInt8 weights are the default; specifying explicitly for clarity.
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"[int8] wrote {out_path} ({out_path.stat().st_size // 1024 // 1024} MB)",
          file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp32", type=Path, required=True,
                   help="Input FP32 ONNX graph (voxlingua107.onnx).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write the variants.")
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    args = p.parse_args()

    if not args.fp32.exists():
        raise SystemExit(f"{args.fp32} not found")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fp16:
        convert_fp16(args.fp32, args.out_dir / "voxlingua107_fp16.onnx")
    if not args.skip_int8:
        convert_int8(args.fp32, args.out_dir / "voxlingua107_int8.onnx")


if __name__ == "__main__":
    main()
