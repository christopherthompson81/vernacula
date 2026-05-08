#!/usr/bin/env python3
"""
Parity smoke test for the Granite Speech 4.1 ONNX export.

Loads the exported ONNX package and the reference transformers model side
by side, runs each piece on the same dummy input, and reports max abs diff.

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/test_parity.py \\
        --onnx-dir ./models/granite_speech_4_1_2b \\
        --model-repo ibm-granite/granite-speech-4.1-2b \\
        --audio-seconds 2.0

Stages
------
    1. encoder.onnx       vs model.encoder(input_features)
    2. projector.onnx     vs model.projector(encoder_out)
    3. decoder_init.onnx  vs language_model(inputs_embeds=merged) prefill
    4. decoder_step.onnx  vs language_model(input_ids=next, past_kv) step

Tolerances are reported absolutely; current expected order of magnitude
(based on cohere_export's modern dynamo path) is < 1e-3 max-diff per stage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--model-repo", default="ibm-granite/granite-speech-4.1-2b")
    p.add_argument("--revision", default=None)
    p.add_argument("--audio-seconds", type=float, default=2.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip the slow decoder parity stages (40-layer LLM is ~15 s/run on CPU).",
    )
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


def max_abs_diff(a, b) -> float:
    import numpy as np
    a = a if isinstance(a, np.ndarray) else a.detach().cpu().numpy()
    b = b if isinstance(b, np.ndarray) else b.detach().cpu().numpy()
    return float(abs(a.astype("float64") - b.astype("float64")).max())


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    section("Loading reference model + processor")
    processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        args.model_repo,
        revision=args.revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    print(f"  model on {args.device}, {sum(p.numel() for p in model.parameters()):,} params")

    section("Building dummy inputs (B=1)")
    sr = processor.audio_processor.sampling_rate
    n_samples = int(args.audio_seconds * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|audio|>transcribe the speech with proper punctuation and capitalization."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(prompt, audio, return_tensors="pt")
    inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}, dtype={v.dtype}")

    input_features = inputs["input_features"]
    input_features_mask = inputs["input_features_mask"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    # --- 1. Encoder parity --------------------------------------------------
    section("Encoder parity")
    enc_path = args.onnx_dir / "encoder.onnx"
    sess = ort.InferenceSession(str(enc_path), so, providers=providers)
    t0 = time.time()
    enc_onnx = sess.run(None, {"input_features": input_features.cpu().numpy()})[0]
    print(f"  ORT encoder: {enc_onnx.shape}, {time.time() - t0:.2f}s")
    with torch.no_grad():
        enc_torch = model.encoder(input_features)
    diff = max_abs_diff(enc_onnx, enc_torch)
    print(f"  max-abs-diff(encoder): {diff:.3e}")
    enc_ref = enc_torch  # reuse for downstream

    # --- 2. Projector parity ----------------------------------------------
    section("Projector parity")
    proj_path = args.onnx_dir / "projector.onnx"
    sess = ort.InferenceSession(str(proj_path), so, providers=providers)
    t0 = time.time()
    proj_onnx = sess.run(None, {"encoder_hidden": enc_ref.cpu().numpy()})[0]
    print(f"  ORT projector: {proj_onnx.shape}, {time.time() - t0:.2f}s")
    with torch.no_grad():
        proj_torch = model.projector(enc_ref)
    diff = max_abs_diff(proj_onnx, proj_torch)
    print(f"  max-abs-diff(projector): {diff:.3e}")
    proj_ref = proj_torch

    if args.skip_decoder:
        section("Skipped decoder stages (--skip-decoder)")
        return 0

    # --- 3. Decoder init parity -------------------------------------------
    section("Decoder init parity (prefill, audio merge)")
    init_path = args.onnx_dir / "decoder_init.onnx"
    sess = ort.InferenceSession(str(init_path), so, providers=providers)
    t0 = time.time()
    init_outs = sess.run(
        None,
        {
            "input_ids": input_ids.cpu().numpy(),
            "audio_embeds": proj_ref.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        },
    )
    print(f"  ORT decoder_init: logits {init_outs[0].shape}, {time.time() - t0:.2f}s")

    # Reference: run the full top-level forward.
    with torch.no_grad():
        ref = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    logits_diff = max_abs_diff(init_outs[0], ref.logits)
    print(f"  max-abs-diff(logits): {logits_diff:.3e}")

    n = 40
    onnx_keys = init_outs[1:1 + n]
    onnx_values = init_outs[1 + n:1 + 2 * n]
    ref_keys = [layer.keys for layer in ref.past_key_values.layers]
    ref_values = [layer.values for layer in ref.past_key_values.layers]
    kv_max = max(
        max(max_abs_diff(k_o, k_r) for k_o, k_r in zip(onnx_keys, ref_keys)),
        max(max_abs_diff(v_o, v_r) for v_o, v_r in zip(onnx_values, ref_values)),
    )
    print(f"  max-abs-diff(KV): {kv_max:.3e}")

    # --- 4. Decoder step parity --------------------------------------------
    section("Decoder step parity (single token continuation)")
    step_path = args.onnx_dir / "decoder_step.onnx"
    sess = ort.InferenceSession(str(step_path), so, providers=providers)

    # Take the argmax of the last position as the next token (typical decode).
    next_id = ref.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    past_len = input_ids.shape[1]
    cache_position = torch.tensor([past_len], dtype=torch.long, device=args.device)
    step_attention_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=args.device)

    feed: dict[str, Any] = {
        "input_id": next_id.cpu().numpy(),
        "attention_mask": step_attention_mask.cpu().numpy(),
        "cache_position": cache_position.cpu().numpy(),
    }
    for i in range(n):
        feed[f"past_key_{i}"] = ref_keys[i].cpu().numpy()
        feed[f"past_value_{i}"] = ref_values[i].cpu().numpy()
    t0 = time.time()
    step_outs = sess.run(None, feed)
    print(f"  ORT decoder_step: logits {step_outs[0].shape}, {time.time() - t0:.2f}s")

    # Reference: rebuild a DynamicCache from the prefill KV and run one step.
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache.from_legacy_cache(tuple(zip(ref_keys, ref_values)))
    with torch.no_grad():
        ref_step = model.language_model(
            input_ids=next_id,
            attention_mask=step_attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
    step_logits_diff = max_abs_diff(step_outs[0], ref_step.logits)
    print(f"  max-abs-diff(step logits): {step_logits_diff:.3e}")

    step_keys = step_outs[1:1 + n]
    step_values = step_outs[1 + n:1 + 2 * n]
    new_keys = [layer.keys for layer in ref_step.past_key_values.layers]
    new_values = [layer.values for layer in ref_step.past_key_values.layers]
    step_kv_max = max(
        max(max_abs_diff(k_o, k_r) for k_o, k_r in zip(step_keys, new_keys)),
        max(max_abs_diff(v_o, v_r) for v_o, v_r in zip(step_values, new_values)),
    )
    print(f"  max-abs-diff(step KV): {step_kv_max:.3e}")

    section("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
