#!/usr/bin/env python3
"""Parity check for the unified decoder.onnx graph.

The unified graph handles both prefill (zero-length past_kv) and step
(populated past_kv) modes through a single set of inputs. This script
exercises both modes and compares against the PyTorch reference.

  Mode A — prefill: past_kv = zero-length, cache_position = [0..S-1].
            Compare against `model(input_features, ...)` reference.
  Mode B — step:    past_kv = populated, cache_position = [past_len].
            Compare against `model.language_model(...)` step reference.

The encoder + projector ONNX paths are reused as-is.

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/test_parity_unified.py \\
        --onnx-dir ./models/granite_speech_unified
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


NUM_DECODER_LAYERS = 40
NUM_KV_HEADS = 4
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--model-repo", default="ibm-granite/granite-speech-4.1-2b")
    p.add_argument("--revision", default=None)
    p.add_argument("--audio-seconds", type=float, default=2.0)
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


def max_abs_diff(a: Any, b: Any) -> float:
    import numpy as np
    a_np = a if isinstance(a, np.ndarray) else a.detach().cpu().numpy()
    b_np = b if isinstance(b, np.ndarray) else b.detach().cpu().numpy()
    return float(abs(a_np.astype("float64") - b_np.astype("float64")).max())


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
        from transformers.cache_utils import DynamicCache
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
    ).eval()
    print(f"  loaded; {sum(p.numel() for p in model.parameters()):,} params")

    section("Building dummy inputs (B=1)")
    sr = processor.audio_processor.sampling_rate
    audio = np.zeros(int(args.audio_seconds * sr), dtype=np.float32)
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|audio|>transcribe the speech with proper punctuation and capitalization."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(prompt, audio, return_tensors="pt")
    input_features = inputs["input_features"]
    input_features_mask = inputs["input_features_mask"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    enc_sess = ort.InferenceSession(str(args.onnx_dir / "encoder.onnx"), so, providers=providers)
    proj_sess = ort.InferenceSession(str(args.onnx_dir / "projector.onnx"), so, providers=providers)
    dec_sess = ort.InferenceSession(str(args.onnx_dir / "decoder.onnx"), so, providers=providers)

    enc_out = enc_sess.run(None, {"input_features": input_features.cpu().numpy()})[0]
    proj_out = proj_sess.run(None, {"encoder_hidden": enc_out})[0]
    print(f"  encoder out: {enc_out.shape}")
    print(f"  projector out: {proj_out.shape}")

    # ── Mode A: prefill (past_kv zero-length) ────────────────────────────
    section("Mode A: prefill via unified decoder (past_kv zero-length)")
    B, S = input_ids.shape
    cache_position = np.arange(S, dtype=np.int64)
    empty_keys = [
        np.zeros((B, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        for _ in range(NUM_DECODER_LAYERS)
    ]
    empty_values = [
        np.zeros((B, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
        for _ in range(NUM_DECODER_LAYERS)
    ]

    feed = {
        "input_ids": input_ids.cpu().numpy().astype(np.int64),
        "audio_embeds": proj_out,
        "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
        "cache_position": cache_position,
    }
    for L in range(NUM_DECODER_LAYERS):
        feed[f"past_key_{L}"] = empty_keys[L]
        feed[f"past_value_{L}"] = empty_values[L]
    t0 = time.time()
    out = dec_sess.run(None, feed)
    print(f"  unified prefill: logits={out[0].shape}, kv0={out[1].shape}  ({time.time()-t0:.2f}s)")

    # Reference: top-level model.forward (the "real" prefill path).
    with torch.no_grad():
        ref = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    ref_keys = [layer.keys for layer in ref.past_key_values.layers]
    ref_values = [layer.values for layer in ref.past_key_values.layers]

    diff_logits = max_abs_diff(out[0], ref.logits)
    onnx_keys = out[1 : 1 + NUM_DECODER_LAYERS]
    onnx_values = out[1 + NUM_DECODER_LAYERS : 1 + 2 * NUM_DECODER_LAYERS]
    diff_kv = max(
        max(max_abs_diff(k_o, k_r) for k_o, k_r in zip(onnx_keys, ref_keys)),
        max(max_abs_diff(v_o, v_r) for v_o, v_r in zip(onnx_values, ref_values)),
    )
    print(f"  max-abs-diff(prefill logits): {diff_logits:.3e}")
    print(f"  max-abs-diff(prefill KV):     {diff_kv:.3e}")

    # ── Mode B: step (populated past_kv) ─────────────────────────────────
    section("Mode B: step via unified decoder (populated past_kv)")
    next_id = ref.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_len = S
    cache_position = np.array([past_len], dtype=np.int64)
    step_attn = np.ones((1, past_len + 1), dtype=np.int64)
    # Caller supplies a 1-row dummy audio_embeds at step time (the cumsum
    # merge collapses to a no-op because no audio token is in next_id).
    audio_dummy = np.zeros((1, 1, proj_out.shape[-1]), dtype=np.float32)

    feed = {
        "input_ids": next_id.cpu().numpy().astype(np.int64),
        "audio_embeds": audio_dummy,
        "attention_mask": step_attn,
        "cache_position": cache_position,
    }
    for L in range(NUM_DECODER_LAYERS):
        feed[f"past_key_{L}"] = ref_keys[L].cpu().numpy().astype(np.float32)
        feed[f"past_value_{L}"] = ref_values[L].cpu().numpy().astype(np.float32)
    t0 = time.time()
    step_out = dec_sess.run(None, feed)
    print(f"  unified step: logits={step_out[0].shape}, kv0={step_out[1].shape}  ({time.time()-t0:.2f}s)")

    # Reference: language_model step (no audio merge needed because input is just the next token).
    cache = DynamicCache.from_legacy_cache(tuple(zip(ref_keys, ref_values)))
    with torch.no_grad():
        ref_step = model.language_model(
            input_ids=next_id,
            attention_mask=torch.from_numpy(step_attn),
            past_key_values=cache,
            cache_position=torch.tensor([past_len], dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
    diff_step_logits = max_abs_diff(step_out[0], ref_step.logits)
    new_keys = [layer.keys for layer in ref_step.past_key_values.layers]
    new_values = [layer.values for layer in ref_step.past_key_values.layers]
    diff_step_kv = max(
        max(max_abs_diff(step_out[1 + L], new_keys[L]) for L in range(NUM_DECODER_LAYERS)),
        max(max_abs_diff(step_out[1 + NUM_DECODER_LAYERS + L], new_values[L]) for L in range(NUM_DECODER_LAYERS)),
    )
    print(f"  max-abs-diff(step logits): {diff_step_logits:.3e}")
    print(f"  max-abs-diff(step KV):     {diff_step_kv:.3e}")

    section("Done")
    ok = diff_logits < 1e-3 and diff_step_logits < 1e-3
    print(f"  verdict (both diffs < 1e-3): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
