#!/usr/bin/env python3
"""
End-to-end transcription smoke test for the Granite Speech 4.1 ONNX export.

Runs the full ORT pipeline (encoder -> projector -> decoder_init ->
decoder_step loop) on a real audio clip, then compares against
`model.generate(...)` from the reference transformers model. This is the
"Python parity" stage of the workflow — it exercises the KV cache
plumbing across decoder_init/decoder_step that the per-stage parity
test (test_parity.py) cannot reach on its own.

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/transcribe_smoke.py \\
        --onnx-dir ./models/granite_speech_4_1_2b \\
        --audio /path/to/clip.wav \\
        --max-new-tokens 64
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
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--prompt",
        default="transcribe the speech with proper punctuation and capitalization.",
        help="The user-side instruction after the <|audio|> placeholder.",
    )
    p.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the transformers reference run (useful when only the ORT path matters).",
    )
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


def load_audio(path: Path) -> tuple[Any, int]:
    """Load audio as float32 mono at the file's native sample rate."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


NUM_DECODER_LAYERS = 40


def run_ort_pipeline(
    onnx_dir: Path,
    processor: Any,
    audio_array: Any,
    sr: int,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, list[int], dict[str, float]]:
    """Run the ORT pipeline end-to-end and return (text, token_ids, timings)."""
    import numpy as np
    import onnxruntime as ort

    if sr != processor.audio_processor.sampling_rate:
        raise SystemExit(
            f"Audio sample rate {sr} != model expected "
            f"{processor.audio_processor.sampling_rate}. Resample first."
        )

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    timings: dict[str, float] = {}

    full_prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|audio|>{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # GraniteSpeechProcessor ignores return_tensors="np" and always returns
    # torch tensors. Convert to numpy at the boundary.
    proc_in = processor(full_prompt, audio_array, return_tensors="pt")
    input_ids = proc_in["input_ids"].cpu().numpy().astype(np.int64)
    attention_mask = proc_in["attention_mask"].cpu().numpy().astype(np.int64)

    # input_features: prefer mel.onnx if it ships with the bundle (the C#
    # runtime does this too — mel.onnx replaces a host-side torchaudio
    # port). Falls back to processor output for backwards compat.
    mel_path = onnx_dir / "mel.onnx"
    if mel_path.exists():
        mel_sess = ort.InferenceSession(str(mel_path), so, providers=providers)
        audio_np = audio_array.astype(np.float32)[np.newaxis, :]
        t0 = time.time()
        input_features = mel_sess.run(None, {"audio": audio_np})[0]
        timings["mel_s"] = time.time() - t0
        print(f"  mel.onnx: input_features={input_features.shape} ({timings['mel_s']:.2f}s)")
    else:
        input_features = proc_in["input_features"].cpu().numpy().astype(np.float32)
        print(f"  processor input_features: {input_features.shape} (mel.onnx not found)")
    print(f"  prompt tokens: input_ids={input_ids.shape}")

    enc_sess = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), so, providers=providers)
    t0 = time.time()
    enc_out = enc_sess.run(None, {"input_features": input_features})[0]
    timings["encoder_s"] = time.time() - t0
    print(f"  encoder: {enc_out.shape}  ({timings['encoder_s']:.2f}s)")

    proj_sess = ort.InferenceSession(str(onnx_dir / "projector.onnx"), so, providers=providers)
    t0 = time.time()
    proj_out = proj_sess.run(None, {"encoder_hidden": enc_out})[0]
    timings["projector_s"] = time.time() - t0
    print(f"  projector: {proj_out.shape}  ({timings['projector_s']:.2f}s)")

    # Prefer the unified decoder.onnx (single graph for both prefill and step)
    # if present; fall back to the split decoder_init/decoder_step pair.
    unified_path = onnx_dir / "decoder.onnx"
    is_unified = unified_path.exists()
    if is_unified:
        init_sess = ort.InferenceSession(str(unified_path), so, providers=providers)
        step_sess = init_sess
        print(f"  using unified decoder.onnx")
    else:
        init_sess = ort.InferenceSession(str(onnx_dir / "decoder_init.onnx"), so, providers=providers)
        step_sess = ort.InferenceSession(str(onnx_dir / "decoder_step.onnx"), so, providers=providers)
        print(f"  using split decoder_init/decoder_step")

    t0 = time.time()
    if is_unified:
        S = input_ids.shape[1]
        feed: dict[str, Any] = {
            "input_ids": input_ids,
            "audio_embeds": proj_out,
            "attention_mask": attention_mask,
            "cache_position": np.arange(S, dtype=np.int64),
        }
        for L in range(NUM_DECODER_LAYERS):
            feed[f"past_key_{L}"] = np.zeros((1, 4, 0, 128), dtype=np.float32)
            feed[f"past_value_{L}"] = np.zeros((1, 4, 0, 128), dtype=np.float32)
        init_outs = init_sess.run(None, feed)
    else:
        init_outs = init_sess.run(
            None,
            {
                "input_ids": input_ids,
                "audio_embeds": proj_out,
                "attention_mask": attention_mask,
            },
        )
    timings["decoder_init_s"] = time.time() - t0
    logits = init_outs[0]
    keys = list(init_outs[1 : 1 + NUM_DECODER_LAYERS])
    values = list(init_outs[1 + NUM_DECODER_LAYERS : 1 + 2 * NUM_DECODER_LAYERS])
    print(
        f"  decoder_init: logits={logits.shape}, kv0={keys[0].shape}  "
        f"({timings['decoder_init_s']:.2f}s)"
    )

    next_token = int(logits[0, -1, :].argmax(axis=-1))

    eos_id = processor.tokenizer.eos_token_id or processor.tokenizer.encode("<|end_of_text|>")[-1]
    generated: list[int] = []
    past_len = input_ids.shape[1]

    # Step-time dummy audio_embeds for the unified graph (cumsum-merge no-op).
    audio_dummy = np.zeros((1, 1, proj_out.shape[-1]), dtype=np.float32)

    t0 = time.time()
    for _ in range(max_new_tokens):
        if next_token == eos_id:
            break
        generated.append(next_token)
        cache_position = np.array([past_len], dtype=np.int64)
        attn = np.ones((1, past_len + 1), dtype=np.int64)
        if is_unified:
            feed = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "audio_embeds": audio_dummy,
                "attention_mask": attn,
                "cache_position": cache_position,
            }
        else:
            feed = {
                "input_id": np.array([[next_token]], dtype=np.int64),
                "attention_mask": attn,
                "cache_position": cache_position,
            }
        for i in range(NUM_DECODER_LAYERS):
            feed[f"past_key_{i}"] = keys[i]
            feed[f"past_value_{i}"] = values[i]
        step_outs = step_sess.run(None, feed)
        step_logits = step_outs[0]
        keys = list(step_outs[1 : 1 + NUM_DECODER_LAYERS])
        values = list(step_outs[1 + NUM_DECODER_LAYERS : 1 + 2 * NUM_DECODER_LAYERS])
        next_token = int(step_logits[0, -1, :].argmax(axis=-1))
        past_len += 1
    timings["decoder_step_total_s"] = time.time() - t0
    timings["decoder_step_per_token_s"] = (
        timings["decoder_step_total_s"] / max(len(generated), 1)
    )

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    print(
        f"  decoder_step: {len(generated)} tokens, "
        f"{timings['decoder_step_per_token_s']*1000:.1f}ms/token"
    )
    return text, generated, timings


def run_reference(
    model_repo: str,
    revision: str | None,
    device: str,
    audio_array: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    import torch
    from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_repo, revision=revision)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_repo,
        revision=revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device).eval()

    full_prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|audio|>{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    proc_in = processor(full_prompt, audio_array, return_tensors="pt")
    proc_in = {k: v.to(device) for k, v in proc_in.items() if hasattr(v, "to")}

    with torch.no_grad():
        out = model.generate(
            **proc_in,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    n_in = proc_in["input_ids"].shape[1]
    new_ids = out[0, n_in:].tolist()
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, new_ids


def main() -> int:
    args = parse_args()

    try:
        from transformers import AutoProcessor
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    section("Loading processor + audio")
    processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision)
    audio, sr = load_audio(args.audio)
    print(f"  audio: {len(audio)} samples @ {sr} Hz ({len(audio) / sr:.2f}s)")

    section("ORT pipeline")
    ort_text, ort_ids, timings = run_ort_pipeline(
        args.onnx_dir, processor, audio, sr, args.prompt, args.max_new_tokens
    )
    print(f"  total: encoder + projector + init + {len(ort_ids)} steps")
    print(f"\n  ORT transcript: {ort_text!r}")
    print(f"  ORT token ids:  {ort_ids[:25]}{'...' if len(ort_ids) > 25 else ''}")

    if args.skip_reference:
        section("Skipped reference (--skip-reference)")
        return 0

    section("Reference pipeline (model.generate)")
    t0 = time.time()
    ref_text, ref_ids = run_reference(
        args.model_repo, args.revision, args.device, audio, args.prompt, args.max_new_tokens
    )
    print(f"  ran in {time.time() - t0:.1f}s")
    print(f"\n  Ref transcript: {ref_text!r}")
    print(f"  Ref token ids:  {ref_ids[:25]}{'...' if len(ref_ids) > 25 else ''}")

    section("Comparison")
    text_match = ort_text.strip() == ref_text.strip()
    ids_match = ort_ids == ref_ids
    print(f"  exact text match: {text_match}")
    print(f"  exact id match:   {ids_match}  (ORT {len(ort_ids)} ids, ref {len(ref_ids)} ids)")
    if not ids_match:
        # Find first divergence.
        n = min(len(ort_ids), len(ref_ids))
        first_div = next((i for i in range(n) if ort_ids[i] != ref_ids[i]), None)
        if first_div is not None:
            print(
                f"  first divergence at token {first_div}: "
                f"ORT={ort_ids[first_div]}, ref={ref_ids[first_div]}"
            )
        elif len(ort_ids) != len(ref_ids):
            print(f"  prefix matches; one stream is longer")

    return 0 if (text_match and ids_match) else 1


if __name__ == "__main__":
    raise SystemExit(main())
