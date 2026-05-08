#!/usr/bin/env python3
"""
Profile the Granite Speech 4.1 ONNX bundle and identify runtime hotspots.

Times each stage of the pipeline independently:
  - audio load
  - mel.onnx
  - encoder.onnx
  - projector.onnx
  - decoder_init.onnx (prefill)
  - decoder_step.onnx loop (per-token + total)

Reports wall-clock seconds per stage and tokens/sec for the AR loop.
With --enable-ort-profiling, aggregates the hottest ONNX ops per stage.

The first run on a fresh process pays the ORT session-init + initial
kernel JIT cost. By default we run twice and report the second run; use
`--warmup 0` to print the first run as well.

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/profile_granite_speech.py \\
        --onnx-dir ./models/granite_speech_4_1_2b \\
        --audio /path/to/clip.wav \\
        --max-tokens 256 \\
        --execution-provider cpu

    # CUDA + ORT op profiling
    python public/scripts/granite_export/profile_granite_speech.py \\
        --onnx-dir ./models/granite_speech_4_1_2b \\
        --audio /path/to/clip.wav \\
        --execution-provider cuda \\
        --enable-ort-profiling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


NUM_DECODER_LAYERS = 40
EOS_TOKEN_ID = 100257


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model-repo", default="ibm-granite/granite-speech-4.1-2b")
    p.add_argument("--revision", default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--execution-provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs before the timed run (default 1).",
    )
    p.add_argument(
        "--enable-ort-profiling",
        action="store_true",
        help="Enable ORT session profiling and print the hottest ops per stage.",
    )
    p.add_argument(
        "--prompt",
        default="transcribe the speech with proper punctuation and capitalization.",
    )
    return p.parse_args()


def get_providers(choice: str, ort: Any) -> list[str]:
    avail = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise SystemExit("CUDAExecutionProvider is not available")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail
        else ["CPUExecutionProvider"]
    )


def make_session(path: Path, providers: list[str], enable_profiling: bool, ort: Any) -> Any:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if enable_profiling:
        so.enable_profiling = True
    # Use kSameAsRequested for the CUDA arena to avoid over-pre-allocation that
    # causes OOM on long-audio decoder_init when both decoder graphs already
    # take ~14 GB of weights on a 25 GB GPU.
    if "CUDAExecutionProvider" in providers:
        provider_opts = [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
        return ort.InferenceSession(str(path), sess_options=so, providers=provider_opts)
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def aggregate_profile(profile_path: Path, top_n: int = 10) -> list[tuple[str, int, float]]:
    """Read an ORT profile JSON and return the top-N kernel ops by total ms."""
    with profile_path.open() as f:
        events = json.load(f)
    op_totals: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        args = ev.get("args", {})
        op = args.get("op_name") or ev.get("name", "?")
        # ORT logs "_kernel_time" events for each op; "dur" is in microseconds.
        if "_kernel_time" in ev.get("name", ""):
            op_totals[op].append(ev["dur"] / 1000.0)  # us -> ms
    rows = sorted(
        ((op, len(times), sum(times)) for op, times in op_totals.items()),
        key=lambda x: x[2],
        reverse=True,
    )
    return rows[:top_n]


def run_pipeline(
    sessions: dict[str, Any],
    audio_array: Any,
    processor: Any,
    prompt: str,
    max_tokens: int,
    label: str,
    np_mod: Any,
) -> dict[str, float]:
    """Run the whole pipeline once and return per-stage timings (seconds).

    Handles both the split decoder pair (`decoder_init` + `decoder_step`)
    and the unified `decoder` graph automatically based on which sessions
    are present.
    """
    timings: dict[str, float] = {}

    is_unified = "decoder" in sessions

    full_prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|audio|>{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    proc_in = processor(full_prompt, audio_array, return_tensors="pt")
    input_ids = proc_in["input_ids"].cpu().numpy().astype(np_mod.int64)
    attention_mask = proc_in["attention_mask"].cpu().numpy().astype(np_mod.int64)

    audio_np = audio_array.astype(np_mod.float32)[np_mod.newaxis, :]

    t = time.perf_counter()
    input_features = sessions["mel"].run(None, {"audio": audio_np})[0]
    timings["mel"] = time.perf_counter() - t

    t = time.perf_counter()
    enc_out = sessions["encoder"].run(None, {"input_features": input_features})[0]
    timings["encoder"] = time.perf_counter() - t

    t = time.perf_counter()
    proj_out = sessions["projector"].run(None, {"encoder_hidden": enc_out})[0]
    timings["projector"] = time.perf_counter() - t

    t = time.perf_counter()
    if is_unified:
        S = input_ids.shape[1]
        feed: dict[str, Any] = {
            "input_ids": input_ids,
            "audio_embeds": proj_out,
            "attention_mask": attention_mask,
            "cache_position": np_mod.arange(S, dtype=np_mod.int64),
        }
        for i in range(NUM_DECODER_LAYERS):
            feed[f"past_key_{i}"] = np_mod.zeros((1, 4, 0, 128), dtype=np_mod.float32)
            feed[f"past_value_{i}"] = np_mod.zeros((1, 4, 0, 128), dtype=np_mod.float32)
        init_outs = sessions["decoder"].run(None, feed)
    else:
        init_outs = sessions["decoder_init"].run(
            None,
            {
                "input_ids": input_ids,
                "audio_embeds": proj_out,
                "attention_mask": attention_mask,
            },
        )
    timings["decoder_init"] = time.perf_counter() - t

    logits = init_outs[0]
    keys = list(init_outs[1 : 1 + NUM_DECODER_LAYERS])
    values = list(init_outs[1 + NUM_DECODER_LAYERS : 1 + 2 * NUM_DECODER_LAYERS])
    next_token = int(logits[0, -1, :].argmax(axis=-1))

    past_len = input_ids.shape[1]
    audio_dummy = np_mod.zeros((1, 1, proj_out.shape[-1]), dtype=np_mod.float32)
    generated: list[int] = []
    step_start = time.perf_counter()
    per_token: list[float] = []
    step_sess = sessions["decoder"] if is_unified else sessions["decoder_step"]
    for _ in range(max_tokens):
        if next_token == EOS_TOKEN_ID:
            break
        generated.append(next_token)
        cache_position = np_mod.array([past_len], dtype=np_mod.int64)
        attn = np_mod.ones((1, past_len + 1), dtype=np_mod.int64)
        if is_unified:
            feed = {
                "input_ids": np_mod.array([[next_token]], dtype=np_mod.int64),
                "audio_embeds": audio_dummy,
                "attention_mask": attn,
                "cache_position": cache_position,
            }
        else:
            feed = {
                "input_id": np_mod.array([[next_token]], dtype=np_mod.int64),
                "attention_mask": attn,
                "cache_position": cache_position,
            }
        for i in range(NUM_DECODER_LAYERS):
            feed[f"past_key_{i}"] = keys[i]
            feed[f"past_value_{i}"] = values[i]
        t = time.perf_counter()
        step_outs = step_sess.run(None, feed)
        per_token.append(time.perf_counter() - t)
        keys = list(step_outs[1 : 1 + NUM_DECODER_LAYERS])
        values = list(step_outs[1 + NUM_DECODER_LAYERS : 1 + 2 * NUM_DECODER_LAYERS])
        next_token = int(step_outs[0][0, -1, :].argmax(axis=-1))
        past_len += 1
    timings["decoder_step_total"] = time.perf_counter() - step_start
    timings["_n_steps"] = float(len(generated))
    timings["_avg_step_ms"] = (
        1000.0 * sum(per_token) / max(len(per_token), 1)
    )

    print(f"  [{label}] T_audio={input_features.shape[1]}, prompt_len={input_ids.shape[1]}, "
          f"steps={len(generated)}")
    return timings


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
        from transformers import AutoProcessor
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    # Audio load
    t = time.perf_counter()
    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_load_s = time.perf_counter() - t
    print(f"audio: {len(audio)} samples @ {sr} Hz "
          f"({len(audio) / sr:.2f}s) [load {audio_load_s*1000:.1f} ms]")

    if sr != 16000:
        raise SystemExit("Audio must be 16 kHz")

    processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision)
    providers = get_providers(args.execution_provider, ort)
    print(f"providers: {providers}")

    # Load sessions. Auto-detect unified (`decoder.onnx`) vs split
    # (`decoder_init.onnx` + `decoder_step.onnx`) bundles.
    unified = (args.onnx_dir / "decoder.onnx").exists()
    decoder_files = ["decoder"] if unified else ["decoder_init", "decoder_step"]
    print(f"Loading ONNX sessions ({'unified' if unified else 'split'} decoder) ...")
    t = time.perf_counter()
    sessions = {
        name: make_session(
            args.onnx_dir / f"{name}.onnx", providers, args.enable_ort_profiling, ort
        )
        for name in ["mel", "encoder", "projector", *decoder_files]
    }
    print(f"  loaded in {time.perf_counter() - t:.2f}s")

    # Warm-up runs
    for w in range(args.warmup):
        print(f"\nWarm-up run {w + 1}/{args.warmup} ...")
        run_pipeline(sessions, audio, processor, args.prompt, args.max_tokens, f"warm{w + 1}", np)

    # Timed run
    print("\nTimed run ...")
    t = run_pipeline(sessions, audio, processor, args.prompt, args.max_tokens, "timed", np)

    # Report
    print("\n=== Stage timings (timed run) ===")
    print(f"  mel              {t['mel']*1000:8.1f} ms")
    print(f"  encoder          {t['encoder']*1000:8.1f} ms")
    print(f"  projector        {t['projector']*1000:8.1f} ms")
    print(f"  decoder_init     {t['decoder_init']*1000:8.1f} ms")
    print(f"  decoder_step x{int(t['_n_steps']):<3} {t['decoder_step_total']*1000:8.1f} ms total"
          f"   ({t['_avg_step_ms']:.1f} ms/token, {1000.0 / t['_avg_step_ms']:.1f} tok/s)")
    total = (
        t["mel"] + t["encoder"] + t["projector"] + t["decoder_init"]
        + t["decoder_step_total"]
    )
    print(f"  --------")
    print(f"  TOTAL            {total*1000:8.1f} ms")
    print(f"  RTF (audio_s/wall_s) = {(len(audio) / 16000) / total:.2f}x realtime")

    # ORT op breakdown if profiling was on
    if args.enable_ort_profiling:
        print("\n=== Top ONNX kernels per stage ===")
        for name, sess in sessions.items():
            prof_path = Path(sess.end_profiling())
            print(f"\n[{name}.onnx] {prof_path.name}")
            try:
                rows = aggregate_profile(prof_path, top_n=10)
                print(f"  {'op':<32s} {'count':>6s} {'total_ms':>10s}")
                for op, count, ms in rows:
                    print(f"  {op:<32s} {count:>6d} {ms:>10.2f}")
            except Exception as e:
                print(f"  could not aggregate: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
