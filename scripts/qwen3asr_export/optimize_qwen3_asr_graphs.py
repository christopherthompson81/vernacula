#!/usr/bin/env python3
"""
Apply ORT transformer optimizer fusions to Qwen3-ASR ONNX graphs.

This mirrors the upstream graph optimization flow:
  - encoder: SkipLayerNormalization / BiasGelu fusions
  - decoders: SimplifiedLayerNormalization fusion for decomposed RMSNorm
"""

from __future__ import annotations

import argparse
import json
import os

import onnx
from onnxruntime.transformers.optimizer import optimize_model


PROTOBUF_LIMIT = 2 * 1024**3


def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def optimize_graph(onnx_path: str, num_heads: int, hidden_size: int, *, use_external_data: bool) -> None:
    basename = os.path.basename(onnx_path)
    print(f"\nOptimizing {basename}...")

    model_proto = onnx.load(onnx_path, load_external_data=False)
    nodes_before = len(model_proto.graph.node)
    del model_proto

    model = optimize_model(
        onnx_path,
        model_type="gpt2",
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=1,
    )

    fusions = model.get_fused_operator_statistics() or {}
    applied = {op: count for op, count in fusions.items() if count > 0}
    if not applied:
        print("  No fusions applied, skipping save")
        return

    print("  Fusions applied:")
    for op_type, count in sorted(applied.items()):
        print(f"    {op_type}: {count}")

    if not use_external_data:
        estimated_size = model.model.ByteSize()
        if estimated_size > PROTOBUF_LIMIT * 0.85:
            print(
                f"  WARNING: estimated model size {estimated_size / 1024**3:.2f} GB "
                "is close to the protobuf size limit"
            )

    model.save_model_to_file(onnx_path, use_external_data_format=use_external_data)

    nodes_after = len(model.model.graph.node)
    reduction = nodes_before - nodes_after
    pct = (reduction / nodes_before * 100.0) if nodes_before else 0.0
    print(f"  Nodes: {nodes_before} -> {nodes_after} ({reduction} removed, {pct:.1f}% reduction)")


def optimize_exported_package(model_dir: str, *, skip_encoder: bool = False, skip_decoders: bool = False) -> None:
    config = load_config(model_dir)

    if not skip_encoder:
        encoder_cfg = config.get("encoder")
        if encoder_cfg is not None:
            optimize_graph(
                os.path.join(model_dir, "encoder.onnx"),
                encoder_cfg["num_heads"],
                encoder_cfg["hidden_size"],
                use_external_data=False,
            )

    if not skip_decoders:
        decoder_cfg = config.get("decoder")
        if decoder_cfg is not None:
            for decoder_name in ("decoder_init", "decoder_step"):
                path = os.path.join(model_dir, f"{decoder_name}.onnx")
                if os.path.exists(path):
                    optimize_graph(
                        path,
                        decoder_cfg["num_attention_heads"],
                        decoder_cfg["hidden_size"],
                        use_external_data=True,
                    )


def main():
    parser = argparse.ArgumentParser(description="Apply ORT transformer optimizer to Qwen3-ASR ONNX graphs")
    parser.add_argument(
        "--input",
        required=True,
        help="Model directory containing encoder.onnx, decoder_init.onnx, decoder_step.onnx",
    )
    parser.add_argument("--skip-encoder", action="store_true", help="Skip encoder optimization")
    parser.add_argument("--skip-decoders", action="store_true", help="Skip decoder optimization")
    args = parser.parse_args()

    optimize_exported_package(
        args.input,
        skip_encoder=args.skip_encoder,
        skip_decoders=args.skip_decoders,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
