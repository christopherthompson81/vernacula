#!/usr/bin/env python3
"""
Export ibm-granite/granite-speech-4.1-2b to ONNX format.

STATUS: skeleton. This script currently loads the model, prints the
planned ONNX graph layout against the loaded config, and exits without
emitting any ONNX. The full export will follow in a subsequent issue
(see docs/dev/granite_speech_investigation.md sequenced plan).

Planned package
---------------
    mel.onnx
        audio [batch, samples] -> mel [batch, 80, T]

    encoder.onnx
        mel [batch, 80, T] -> acoustic [batch, T/2, 1024]
        (adjacent-frame stack baked in: encoder input_dim 160 = 80 * 2)

    projector.onnx
        acoustic [batch, T/2, 1024] -> audio_embeds [batch, T/10, 2048]
        (BLIP-2 Q-Former, 5x temporal downsample, 3 trainable queries
         per 15-frame window)

    decoder_init.onnx / decoder_step.onnx
        Split prefill/step graphs with KV-cache carry, mirroring the
        cohere_export and qwen3asr_export pattern.
        KV layout: 40 layers x split K/V x GQA (num_kv_heads=4) x seq x
        head_dim=128.

    config.json, tokenizer assets, export-report.json.

The audio placeholder token id (100352) is spliced in the prompt before
projector embeddings are concatenated into decoder_init's input_embeds
stream. Exact prefill input shape (audio_offset/audio_lengths vs implicit
host splice) is a Run 2 decision.

Usage (intended; stub today)
----------------------------
    python public/scripts/granite_export/export_granite_speech_to_onnx.py \\
        --output-dir ./models/granite_speech_4_1_2b \\
        --opset 18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL_REPO = "ibm-granite/granite-speech-4.1-2b"
DEFAULT_OPSET = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export ibm-granite/granite-speech-4.1-2b to ONNX. "
            "Currently a skeleton stub; see docs/dev/granite_speech_investigation.md."
        )
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_MODEL_REPO}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Pin a specific commit/tag for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where the ONNX package will be written.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET}).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to trace the export on.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Export dtype.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing export at --output-dir.",
    )
    return parser.parse_args()


def describe_planned_layout(config) -> dict:
    """Return the planned ONNX package layout derived from the loaded config.

    Keeping this as a returnable dict (rather than just printing) so the
    parity script in the next issue can import and assert against the
    same source of truth.
    """
    enc = config.encoder_config
    proj = config.projector_config
    text = config.text_config

    return {
        "model_repo": getattr(config, "_name_or_path", None),
        "audio_token_index": config.audio_token_index,
        "downsample_rate_total": 2 * config.downsample_rate,  # encoder 2x * projector 5x
        "encoder": {
            "num_layers": enc.num_layers,
            "hidden_dim": enc.hidden_dim,
            "num_heads": enc.num_heads,
            "dim_head": enc.dim_head,
            "input_dim": enc.input_dim,
            "output_dim": enc.output_dim,
            "conv_kernel_size": enc.conv_kernel_size,
        },
        "projector": {
            "model_type": proj.model_type,
            "num_hidden_layers": proj.num_hidden_layers,
            "num_attention_heads": proj.num_attention_heads,
            "hidden_size": proj.hidden_size,
            "encoder_hidden_size": proj.encoder_hidden_size,
            "window_size": config.window_size,
            "downsample_rate": config.downsample_rate,
        },
        "decoder": {
            "model_type": text.model_type,
            "num_hidden_layers": text.num_hidden_layers,
            "hidden_size": text.hidden_size,
            "num_attention_heads": text.num_attention_heads,
            "num_key_value_heads": text.num_key_value_heads,
            "head_dim": text.hidden_size // text.num_attention_heads,
            "vocab_size": text.vocab_size,
            "max_position_embeddings": text.max_position_embeddings,
            "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
        },
    }


def main() -> int:
    args = parse_args()

    # Defer imports so `--help` works without the heavy stack installed.
    try:
        from transformers import AutoConfig
    except ImportError:
        print(
            "transformers is not installed. Install requirements first:\n"
            "  pip install -r public/scripts/granite_export/requirements.txt",
            file=sys.stderr,
        )
        return 2

    if args.output_dir.exists() and not args.overwrite:
        print(
            f"Output directory {args.output_dir} already exists. "
            "Pass --overwrite to replace it.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading config from {args.model_repo} (revision={args.revision})")
    config = AutoConfig.from_pretrained(
        args.model_repo,
        revision=args.revision,
        trust_remote_code=False,
    )

    layout = describe_planned_layout(config)
    print("Planned ONNX package layout:")
    print(json.dumps(layout, indent=2))

    print(
        "\nSkeleton run complete — no ONNX written.\n"
        "The full export is tracked in a follow-up issue under #28; see\n"
        "docs/dev/granite_speech_investigation.md sequenced plan.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
