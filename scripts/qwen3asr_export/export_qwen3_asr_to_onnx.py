#!/usr/bin/env python3
"""
Export Qwen3-ASR to ONNX format.

Produces:
    encoder.onnx         - Audio encoder (mel -> features), weights embedded in proto
    decoder_init.onnx    - Decoder prefill (input_ids + audio -> logits + KV cache)
    decoder_step.onnx    - Decoder step (token embed + KV cache -> logits + KV cache)
    embed_tokens.bin     - Token embedding matrix for host-side decoder_step lookup
    tokenizer.json       - HuggingFace tokenizer assets
    config.json          - Architecture config + special tokens + mel params
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import onnx
import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.decoder_wrapper import (
    export_decoder_init,
    export_decoder_init_batched,
    export_decoder_step,
    export_decoder_step_static,
)
from src.encoder_wrapper import export_encoder, export_encoder_batched
from optimize_qwen3_asr_graphs import optimize_exported_package
from src.prompt import (
    ASR_TEXT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    ENDOFTEXT_TOKEN_ID,
    EOS_TOKEN_IDS,
    IM_END_TOKEN_ID,
    IM_START_TOKEN_ID,
)


def load_model(model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """
    Load Qwen3-ASR.

    We prefer `trust_remote_code=True` so the export works against the public
    Hugging Face repo without requiring an extra local package install.
    """
    print(f"Loading model {model_id}...")

    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"AutoModel.from_pretrained failed: {exc}")
        print("Trying explicit qwen_asr import...")
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        )

    model.eval()
    print(f"Model loaded. Device: {device}, dtype: {dtype}")
    return model


def copy_tokenizer(model_id: str, output_dir: str):
    """Save the tokenizer assets we need beside the ONNX package."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    keep_files = {"tokenizer.json", "tokenizer_config.json"}
    remove_extensions = {".py", ".txt"}
    for filename in os.listdir(output_dir):
        if filename in keep_files:
            continue
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(filename)
        if ext in remove_extensions or filename == "special_tokens_map.json":
            os.remove(path)

    print(f"Tokenizer saved to {output_dir}")


def verify_special_tokens(model_id: str):
    """Make sure our hardcoded token IDs still match the upstream tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    checks = {
        "<|audio_start|>": AUDIO_START_TOKEN_ID,
        "<|audio_end|>": AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": AUDIO_PAD_TOKEN_ID,
        "<|im_start|>": IM_START_TOKEN_ID,
        "<|im_end|>": IM_END_TOKEN_ID,
        "<|endoftext|>": ENDOFTEXT_TOKEN_ID,
    }

    all_ok = True
    for token_str, expected_id in checks.items():
        actual_id = tokenizer.convert_tokens_to_ids(token_str)
        if actual_id != expected_id:
            print(f"  MISMATCH: {token_str} expected={expected_id} actual={actual_id}")
            all_ok = False
        else:
            print(f"  OK: {token_str} = {actual_id}")

    if not all_ok:
        raise ValueError("Special token IDs do not match. Update src/prompt.py with the current tokenizer IDs.")
    print("All special token IDs verified.")


def write_preprocessor_config(output_dir: str):
    """Write Whisper-style mel frontend metadata for downstream consumers."""
    chunk_length = 30
    sample_rate = 16000
    hop_length = 160
    n_fft = 400
    n_mels = 128

    config = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": n_mels,
        "sampling_rate": sample_rate,
        "hop_length": hop_length,
        "n_fft": n_fft,
        "chunk_length": chunk_length,
        "n_samples": chunk_length * sample_rate,
        "nb_max_frames": chunk_length * sample_rate // hop_length,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": False,
    }

    output_path = os.path.join(output_dir, "preprocessor_config.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Preprocessor config saved to {output_path}")


def write_config(model, output_dir: str):
    """Write the model metadata Vernacula will need to consume the export."""
    thinker_config = model.config.thinker_config
    audio_config = thinker_config.audio_config
    text_config = thinker_config.text_config

    config = {
        "model_type": "qwen3_asr",
        "encoder": {
            "num_layers": audio_config.encoder_layers,
            "hidden_size": audio_config.d_model,
            "num_heads": audio_config.encoder_attention_heads,
            "ffn_dim": audio_config.encoder_ffn_dim,
            "conv_channels": audio_config.downsample_hidden_size,
            "output_dim": audio_config.output_dim,
            "downsample_factor": 8,
            "num_mel_bins": audio_config.num_mel_bins,
        },
        "decoder": {
            "num_layers": text_config.num_hidden_layers,
            "hidden_size": text_config.hidden_size,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": text_config.num_key_value_heads,
            "head_dim": text_config.head_dim,
            "intermediate_size": text_config.intermediate_size,
            "vocab_size": text_config.vocab_size,
            "rope_theta": text_config.rope_theta,
            "rms_norm_eps": text_config.rms_norm_eps,
            "tie_word_embeddings": text_config.tie_word_embeddings,
            "rope_scaling": {
                "mrope_section": text_config.rope_scaling.get("mrope_section", [24, 20, 20]),
                "interleaved": text_config.rope_scaling.get("mrope_interleaved", True),
            },
        },
        "embed_tokens_shape": list(model.thinker.model.embed_tokens.weight.shape),
        "embed_tokens_dtype": "float32",
        "mel": {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 128,
            "fmin": 0,
            "fmax": 8000,
        },
        "special_tokens": {
            "eos_token_ids": EOS_TOKEN_IDS,
            "pad_token_id": ENDOFTEXT_TOKEN_ID,
            "im_start_token_id": IM_START_TOKEN_ID,
            "im_end_token_id": IM_END_TOKEN_ID,
            "audio_start_token_id": AUDIO_START_TOKEN_ID,
            "audio_end_token_id": AUDIO_END_TOKEN_ID,
            "audio_pad_token_id": AUDIO_PAD_TOKEN_ID,
            "asr_text_token_id": ASR_TEXT_TOKEN_ID,
        },
    }

    output_path = os.path.join(output_dir, "config.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Config saved to {output_path}")


def _convert_to_fp16(output_dir: str, filenames: list[str]):
    """Convert exported ONNX files to FP16 while preserving FP32 I/O types."""
    from onnxruntime.transformers.optimizer import optimize_model

    for filename in filenames:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue

        size_before = os.path.getsize(path)
        data_path = path + ".data"
        if os.path.exists(data_path):
            size_before += os.path.getsize(data_path)

        print(f"  Converting {filename} to FP16...")
        model = optimize_model(path, model_type="gpt2", opt_level=0)
        model.convert_float_to_float16(keep_io_types=True)

        if os.path.exists(data_path):
            os.remove(data_path)
        model.save_model_to_file(path)

        size_after = os.path.getsize(path)
        if os.path.exists(data_path):
            size_after += os.path.getsize(data_path)
        print(f"    {size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB")


def _default_output_dir(model_id: str) -> str:
    name = model_id.rstrip("/").rsplit("/", 1)[-1]
    return os.path.join("output", name.lower())


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR to ONNX")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Hugging Face model ID or local path")
    parser.add_argument("--output", default=None, help="Output directory for ONNX files")
    parser.add_argument("--device", default="cpu", help="Device for export tracing (cpu or cuda)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--skip-encoder", action="store_true", help="Skip encoder export")
    parser.add_argument("--skip-decoder", action="store_true", help="Skip decoder export")
    parser.add_argument(
        "--export-batching-artifacts",
        action="store_true",
        help="Also export experimental encoder_batched.onnx and decoder_init_batched.onnx",
    )
    parser.add_argument(
        "--skip-graph-optimization",
        action="store_true",
        help="Skip the offline ORT graph fusions normally applied after export",
    )
    parser.add_argument(
        "--export-static-step",
        action="store_true",
        help="Also export decoder_step_static.onnx with pre-allocated KV buffers",
    )
    parser.add_argument(
        "--static-kv-max-tokens",
        type=int,
        default=4096,
        help="Pre-allocated token capacity for decoder_step_static.onnx",
    )
    parser.add_argument(
        "--no-share-weights",
        action="store_true",
        help="Keep the default decoder_init external-data filename instead of renaming it",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Weight storage dtype (fp16 halves model size after export)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = _default_output_dir(args.model)

    os.makedirs(args.output, exist_ok=True)

    model = load_model(args.model, device=args.device, dtype=torch.float32)

    print("\nVerifying special token IDs...")
    verify_special_tokens(args.model)

    export_any_encoder = (not args.skip_encoder) or args.export_batching_artifacts
    if export_any_encoder:
        if args.skip_encoder:
            print("\n=== Skipping standard encoder export; emitting requested experimental encoder artifacts ===")
        else:
            print("\n=== Exporting encoder ===")
            encoder_path = os.path.join(args.output, "encoder.onnx")
            export_encoder(model, encoder_path, opset_version=args.opset, device=args.device)

            encoder_data = encoder_path + ".data"
            if os.path.exists(encoder_data):
                print("  Embedding encoder weights into .onnx proto...")
                encoder_model = onnx.load(encoder_path, load_external_data=True)
                onnx.save(encoder_model, encoder_path)
                os.remove(encoder_data)

            if args.dtype == "fp16":
                print("\n=== Converting encoder to FP16 ===")
                _convert_to_fp16(args.output, ["encoder.onnx"])

        if args.export_batching_artifacts:
            print("\n=== Exporting batched encoder (experimental) ===")
            export_encoder_batched(
                model,
                os.path.join(args.output, "encoder_batched.onnx"),
                opset_version=args.opset,
                device=args.device,
            )

    export_any_decoder = (not args.skip_decoder) or args.export_static_step or args.export_batching_artifacts

    if export_any_decoder:
        if args.skip_decoder:
            print("\n=== Skipping standard decoder export; emitting requested experimental decoder artifacts ===")
        else:
            print("\n=== Exporting decoder (init) ===")
            export_decoder_init(
                model,
                os.path.join(args.output, "decoder_init.onnx"),
                opset_version=args.opset,
                device=args.device,
            )

        if args.export_batching_artifacts:
            print("\n=== Exporting decoder (init, batched experimental) ===")
            export_decoder_init_batched(
                model,
                os.path.join(args.output, "decoder_init_batched.onnx"),
                opset_version=args.opset,
                device=args.device,
            )

        if not args.skip_decoder:
            print("\n=== Exporting decoder (step) ===")
            export_decoder_step(
                model,
                os.path.join(args.output, "decoder_step.onnx"),
                opset_version=args.opset,
                device=args.device,
            )

            if args.dtype == "fp16":
                print("\n=== Converting decoders to FP16 ===")
                _convert_to_fp16(args.output, ["decoder_init.onnx", "decoder_step.onnx"])

            step_path = os.path.join(args.output, "decoder_step.onnx")
            step_data = step_path + ".data"
            if os.path.exists(step_data):
                step_model = onnx.load(step_path, load_external_data=True)
                proto_size = sum(len(tensor.raw_data) for tensor in step_model.graph.initializer)
                if proto_size < 1_800_000_000:
                    print("  Inlining decoder_step weights into .onnx proto...")
                    onnx.save(step_model, step_path)
                    os.remove(step_data)
                    print(f"  decoder_step.onnx: {os.path.getsize(step_path) / 1e6:.1f} MB (self-contained)")
                else:
                    print(f"  decoder_step too large to inline ({proto_size / 1e6:.0f} MB), keeping external data")

            if not args.no_share_weights:
                init_path = os.path.join(args.output, "decoder_init.onnx")
                init_data = init_path + ".data"
                if os.path.exists(init_data):
                    final_data = os.path.join(args.output, "decoder_init.onnx.data")
                    if init_data != final_data:
                        os.rename(init_data, final_data)

    if args.export_static_step:
        print("\n=== Exporting decoder (static step) ===")
        export_decoder_step_static(
            model,
            os.path.join(args.output, "decoder_step_static.onnx"),
            static_kv_max_tokens=args.static_kv_max_tokens,
            opset_version=args.opset,
            device=args.device,
        )

    if export_any_decoder:
        print("\n=== Saving embedding cache ===")
        embed_weight = model.thinker.model.embed_tokens.weight.data
        embed_np = embed_weight.cpu().float().numpy()
        embed_path = os.path.join(args.output, "embed_tokens.bin")
        embed_np.tofile(embed_path)
        print(f"  {embed_np.shape} ({embed_np.nbytes / 1e6:.1f} MB)")

    print("\n=== Copying tokenizer ===")
    copy_tokenizer(args.model, args.output)

    print("\n=== Writing config ===")
    write_config(model, args.output)
    write_preprocessor_config(args.output)

    if not args.skip_graph_optimization:
        print("\n=== Optimizing exported graphs ===")
        optimize_exported_package(
            args.output,
            skip_encoder=args.skip_encoder,
            skip_decoders=args.skip_decoder,
        )

    print(f"\nExport complete. Output directory: {args.output}")
    print("Files:")
    for filename in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, filename)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if size > 1e6:
            print(f"  {filename}: {size / 1e6:.1f} MB")
        else:
            print(f"  {filename}: {size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
