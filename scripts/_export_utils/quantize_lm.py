"""Quantize an exported LM ONNX graph for the Run 8 perf sweep.

Operates on a single .onnx file (with sibling external-data file if
present) and writes a new .onnx + .onnx_data sibling pair. Sized for
the Chatterbox language_model.onnx (~2 GB external-data fp32), but the
modes are generic — any KV-cache-style LM export with MatMul-heavy
attention blocks should respond similarly.

Three modes:

  fp16  — onnxconverter_common.float16.convert_float_to_float16
          (keep_io_types=True so the C# binding still feeds fp32).
          Halves weight bytes; gates open the GPU's fp16 MMA pipes.
          Lowest accuracy risk for transformer LMs.

  int8  — onnxruntime.quantization.quantize_dynamic (W8A8 dynamic).
          Quarter weight bytes vs fp32; needs no calibration data.
          Higher risk than fp16, especially for residual streams.

  int4  — onnxruntime.quantization.matmul_nbits_quantizer
          .MatMulNBitsQuantizer with RTN (round-to-nearest) config.
          ~1/8 weight bytes vs fp32; LLM-standard for inference.
          accuracy_level=4 = int8 activations during MatMul.

Mode-specific limitations are documented inline at each handler.

Invocation (from repo root, with the chatterbox export venv active):

    .venv-chatterbox-export/bin/python scripts/_export_utils/quantize_lm.py \\
        --input /tmp/cb_dyn5/language_model.onnx \\
        --output /tmp/cb_dyn5/language_model.fp16.onnx \\
        --mode fp16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def quantize_fp16(input_path: Path, output_path: Path, keep_io_types: bool = True) -> None:
    # keep_io_types=True leaves graph inputs/outputs as fp32; ORT inserts
    # Cast nodes at the boundary so the C# binding doesn't need to change.
    # disable_shape_infer=True bypasses a known crash on graphs with
    # large external-data initializers — shape inference walks the whole
    # weight tensor and runs out of stack on the 30-layer KV graph.
    # ORT's transformers fp16 converter (vs onnxconverter_common's) handles
    # transformer LMs robustly: it does proper type unification at Add/Mul
    # boundaries (LayerNorm internals were tripping the onnxconverter_common
    # 1.16.0 path with a "Type Error: T bound to different types" failure
    # at /layers.0/input_layernorm/Add on this graph).
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    print(f"[fp16] loading {input_path} (with external data) ...")
    model = onnx.load(str(input_path), load_external_data=True)
    print(f"[fp16] converting fp32 → fp16 (keep_io_types={keep_io_types}) ...")
    t0 = time.time()
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=True,
    )
    print(f"[fp16] convert done in {time.time() - t0:.1f}s; saving ...")
    # Save with external data — the fp16 weights are still ~1 GB, well
    # above the 2 GB protobuf cap, and saving inline would silently
    # truncate. Use the sibling .onnx_data convention the export script
    # already established.
    data_name = output_path.name + "_data"
    onnx.save_model(
        model_fp16,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"[fp16] wrote {output_path} (+ {data_name})")


def quantize_int8_dynamic(input_path: Path, output_path: Path) -> None:
    # Dynamic quantization picks per-tensor activation scales at runtime
    # from observed min/max — no calibration set needed. Quantizes weights
    # to int8 statically; quantizes activations to int8 dynamically each
    # call. Cheapest path to "is int8 viable for this LM at all?".
    #
    # ORT's quantize_dynamic mutates external-data references in place
    # by default; safer to copy the source into a temp location first.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("[int8] WARNING: Run 8b in docs/chatterbox_perf_investigation.md "
          "found int8 dynamic falls back to CPU on CUDA EP for this LM — "
          "3.4× slower and quality collapse at step 0. Continuing anyway.")
    print(f"[int8] dynamic-quantizing {input_path} -> {output_path} ...")
    t0 = time.time()
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        # MatMul + Gather (the embedding lookup) cover the heavy weights.
        # Attention.RotaryEmbedding and norm ops stay fp32, which is fine
        # — they're cheap.
        op_types_to_quantize=["MatMul", "Gather"],
        per_channel=True,
        reduce_range=False,
        use_external_data_format=True,
    )
    print(f"[int8] done in {time.time() - t0:.1f}s")


def quantize_int4(input_path: Path, output_path: Path, block_size: int = 32) -> None:
    # RTN (round-to-nearest) weight-only 4-bit MatMul quantization.
    # accuracy_level=4 = "int8 activations during the int4 MatMul"
    # (vs 0=fp32, 1=fp16, 2=bf16, 3=int4 activations). int8 acts give
    # ~all the speedup with ~no accuracy regression on attention LLMs.
    # block_size=32 is the LLM default; smaller = more granular scales
    # = better quality at slight compression cost.
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        RTNWeightOnlyQuantConfig,
    )

    print(f"[int4] loading {input_path} ...")
    model = onnx.load(str(input_path), load_external_data=True)
    print(f"[int4] running MatMulNBitsQuantizer (RTN, block_size={block_size}) ...")
    t0 = time.time()
    cfg = RTNWeightOnlyQuantConfig()
    q = MatMulNBitsQuantizer(
        model=model,
        block_size=block_size,
        is_symmetric=True,  # symmetric = simpler kernels, slightly worse quality
        accuracy_level=4,
        algo_config=cfg,
    )
    q.process()
    print(f"[int4] done in {time.time() - t0:.1f}s; saving ...")
    data_name = output_path.name + "_data"
    onnx.save_model(
        q.model.model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"[int4] wrote {output_path} (+ {data_name})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Source language_model.onnx")
    ap.add_argument("--output", required=True, type=Path, help="Destination quantized .onnx")
    ap.add_argument("--mode", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--block-size", type=int, default=32, help="int4 RTN block size (default 32)")
    ap.add_argument(
        "--no-keep-io-types",
        action="store_true",
        help="[fp16 only] Convert I/O tensors to fp16 too (true fp16, no boundary Casts). "
             "Caller must feed fp16 inputs and decode fp16 outputs.",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any stale sidecar from a previous run; otherwise onnx.save_model
    # may concatenate to it.
    sidecar = args.output.with_name(args.output.name + "_data")
    if sidecar.exists():
        sidecar.unlink()

    if args.mode == "fp16":
        quantize_fp16(args.input, args.output, keep_io_types=not args.no_keep_io_types)
    elif args.mode == "int8":
        quantize_int8_dynamic(args.input, args.output)
    elif args.mode == "int4":
        quantize_int4(args.input, args.output, block_size=args.block_size)
    else:
        print(f"ERROR: unknown mode {args.mode}", file=sys.stderr)
        return 1

    # Report size.
    out_sz = args.output.stat().st_size
    side_sz = sidecar.stat().st_size if sidecar.exists() else 0
    print(f"[size] {args.output.name}: {out_sz / 1e6:.1f} MB"
          + (f"  (+ {sidecar.name}: {side_sz / 1e6:.1f} MB)" if side_sz else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
