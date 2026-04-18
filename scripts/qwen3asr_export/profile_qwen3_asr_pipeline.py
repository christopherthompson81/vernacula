#!/usr/bin/env python3
"""
Profile a Qwen3-ASR ONNX package and identify the main runtime hotspots.

The profiler reports coarse stage timings first:
  - audio load / resample
  - log-mel frontend
  - encoder
  - prompt build
  - decoder prefill
  - autoregressive decode loop

With --enable-ort-profiling it also aggregates the hottest ONNX Runtime ops
for encoder / decoder_init / decoder_step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.mel import log_mel_spectrogram
from src.prompt import EOS_TOKEN_IDS, build_prompt_ids, get_audio_pad_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Qwen3-ASR ONNX pipeline timing.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing encoder/decoder ONNX files")
    parser.add_argument("--audio", required=True, help="Audio file to profile")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum decode tokens")
    parser.add_argument(
        "--decoder-mode",
        choices=("auto", "dynamic", "static-step"),
        default="auto",
        help="Which decoder path to profile",
    )
    parser.add_argument(
        "--static-kv-max-tokens",
        type=int,
        default=4096,
        help="Token capacity for decoder_step_static.onnx",
    )
    parser.add_argument(
        "--execution-provider",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Which ORT execution provider to use",
    )
    parser.add_argument(
        "--enable-ort-profiling",
        action="store_true",
        help="Enable ORT session profiling and print the hottest operators",
    )
    return parser.parse_args()


def load_audio(path: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return audio, time.perf_counter() - start


def get_providers(choice: str) -> list[str]:
    available = ort.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this environment")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(path: str, providers: list[str], enable_profiling: bool) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    if enable_profiling:
        session_options.enable_profiling = True
    return ort.InferenceSession(path, sess_options=session_options, providers=providers)


def load_sessions(
    onnx_dir: str,
    providers: list[str],
    enable_profiling: bool,
    decoder_mode: str,
) -> dict[str, ort.InferenceSession]:
    sessions = {}
    required_names = ["encoder", "decoder_init"]
    if decoder_mode == "dynamic":
        required_names.append("decoder_step")
    elif decoder_mode == "static-step":
        required_names.append("decoder_step_static")
    else:
        static_step_path = os.path.join(onnx_dir, "decoder_step_static.onnx")
        required_names.append("decoder_step_static" if os.path.exists(static_step_path) else "decoder_step")

    for name in required_names:
        path = os.path.join(onnx_dir, f"{name}.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        sessions[name] = make_session(path, providers, enable_profiling)
    return sessions


def load_embed_tokens(onnx_dir: str) -> np.ndarray:
    config_path = os.path.join(onnx_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    shape = tuple(config["embed_tokens_shape"])
    dtype_name = config.get("embed_tokens_dtype", "float32")
    dtype = np.float32 if dtype_name == "float32" else np.float16
    return np.fromfile(os.path.join(onnx_dir, "embed_tokens.bin"), dtype=dtype).reshape(shape)


def summarize_stage_times(stage_times: list[tuple[str, float]], total: float):
    print("\nStage breakdown")
    for label, seconds in stage_times:
        share = (seconds / total * 100.0) if total > 0 else 0.0
        print(f"  {label:22s} {seconds:8.3f}s  {share:6.1f}%")
    print(f"  {'total_pipeline':22s} {total:8.3f}s")


def summarize_decode_steps(step_times: list[float]):
    print("\nDecode-step stats")
    if not step_times:
        print("  No autoregressive steps were needed; EOS was emitted from prefill.")
        return
    step_ms = np.array(step_times, dtype=np.float64) * 1000.0
    print(f"  steps:               {len(step_times)}")
    print(f"  mean step time:      {step_ms.mean():8.2f} ms")
    print(f"  median step time:    {np.median(step_ms):8.2f} ms")
    print(f"  p95 step time:       {np.percentile(step_ms, 95):8.2f} ms")
    print(f"  max step time:       {step_ms.max():8.2f} ms")


def summarize_ort_profile(name: str, session: ort.InferenceSession, limit: int = 10):
    profile_path = session.end_profiling()
    if not profile_path:
        return

    with open(profile_path, "r", encoding="utf-8") as handle:
        events = json.load(handle)

    op_totals_us: dict[str, float] = defaultdict(float)
    op_counts: dict[str, int] = defaultdict(int)

    for event in events:
        args = event.get("args", {})
        op_name = args.get("op_name")
        dur_us = event.get("dur")
        if not op_name or dur_us is None:
            continue
        op_totals_us[op_name] += float(dur_us)
        op_counts[op_name] += 1

    top_ops = sorted(op_totals_us.items(), key=lambda item: item[1], reverse=True)[:limit]
    print(f"\n{name} hottest ORT ops")
    for op_name, total_us in top_ops:
        total_ms = total_us / 1000.0
        count = op_counts[op_name]
        mean_ms = total_ms / count
        print(f"  {op_name:28s} total={total_ms:9.2f}ms  calls={count:5d}  mean={mean_ms:8.3f}ms")


def main():
    args = parse_args()
    providers = get_providers(args.execution_provider)
    sessions = load_sessions(args.onnx_dir, providers, args.enable_ort_profiling, args.decoder_mode)
    embed_tokens = load_embed_tokens(args.onnx_dir)

    audio, audio_load_s = load_audio(args.audio)

    pipeline_start = time.perf_counter()

    start = time.perf_counter()
    mel = log_mel_spectrogram(audio).cpu().numpy()
    mel_s = time.perf_counter() - start

    start = time.perf_counter()
    audio_features = sessions["encoder"].run(["audio_features"], {"mel": mel})[0]
    encoder_s = time.perf_counter() - start

    start = time.perf_counter()
    prompt_ids = build_prompt_ids(audio_features.shape[1])
    position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]
    audio_start, _ = get_audio_pad_range(prompt_ids)
    input_ids = np.array(prompt_ids, dtype=np.int64)[np.newaxis, :]
    audio_offset = np.array([audio_start], dtype=np.int64)
    prompt_build_s = time.perf_counter() - start

    start = time.perf_counter()
    logits, present_keys, present_values = sessions["decoder_init"].run(
        ["logits", "present_keys", "present_values"],
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "audio_features": audio_features,
            "audio_offset": audio_offset,
        },
    )
    prefill_s = time.perf_counter() - start

    next_token = int(np.argmax(logits[0, -1, :]))
    tokens = [next_token]
    decode_step_times: list[float] = []
    use_static_step = args.decoder_mode == "static-step" or (
        args.decoder_mode == "auto" and "decoder_step_static" in sessions
    )

    if next_token not in EOS_TOKEN_IDS:
        pos = len(prompt_ids)
        static_keys = None
        static_values = None
        if use_static_step:
            num_layers, batch, kv_heads, prompt_len, head_dim = present_keys.shape
            static_keys = np.zeros(
                (num_layers, batch, kv_heads, args.static_kv_max_tokens, head_dim),
                dtype=present_keys.dtype,
            )
            static_values = np.zeros(
                (num_layers, batch, kv_heads, args.static_kv_max_tokens, head_dim),
                dtype=present_values.dtype,
            )
            static_keys[:, :, :, :prompt_len, :] = present_keys
            static_values[:, :, :, :prompt_len, :] = present_values

        for _ in range(args.max_tokens - 1):
            token_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]
            step_pos = np.array([[pos]], dtype=np.int64)

            start = time.perf_counter()
            if use_static_step:
                logits, static_keys, static_values = sessions["decoder_step_static"].run(
                    ["logits", "present_keys", "present_values"],
                    {
                        "input_embeds": token_embed,
                        "position_ids": step_pos,
                        "kv_pos": np.array(pos, dtype=np.int64),
                        "past_keys": static_keys,
                        "past_values": static_values,
                    },
                )
            else:
                logits, present_keys, present_values = sessions["decoder_step"].run(
                    ["logits", "present_keys", "present_values"],
                    {
                        "input_embeds": token_embed,
                        "position_ids": step_pos,
                        "past_keys": present_keys,
                        "past_values": present_values,
                    },
                )
            decode_step_times.append(time.perf_counter() - start)

            next_token = int(np.argmax(logits[0, -1, :]))
            tokens.append(next_token)
            pos += 1

            if next_token in EOS_TOKEN_IDS:
                break

    total_pipeline_s = time.perf_counter() - pipeline_start
    decode_total_s = float(sum(decode_step_times))

    stage_times = [
        ("audio_load", audio_load_s),
        ("mel", mel_s),
        ("encoder", encoder_s),
        ("prompt_build", prompt_build_s),
        ("decoder_prefill", prefill_s),
        ("decoder_steps_total", decode_total_s),
    ]

    print(f"Providers: {providers}")
    print(f"Audio samples: {audio.shape[0]}")
    print(f"Mel shape: {mel.shape}")
    print(f"Audio token count: {audio_features.shape[1]}")
    print(f"Generated tokens: {len(tokens)}")
    print(f"Decoder mode: {'static-step' if use_static_step else 'dynamic'}")

    summarize_stage_times(stage_times, audio_load_s + total_pipeline_s)
    summarize_decode_steps(decode_step_times)

    sorted_stages = sorted(stage_times, key=lambda item: item[1], reverse=True)
    print("\nTop bottlenecks")
    for label, seconds in sorted_stages[:3]:
        print(f"  {label:22s} {seconds:8.3f}s")

    if args.enable_ort_profiling:
        summarize_ort_profile("encoder", sessions["encoder"])
        summarize_ort_profile("decoder_init", sessions["decoder_init"])
        if use_static_step:
            summarize_ort_profile("decoder_step_static", sessions["decoder_step_static"])
        else:
            summarize_ort_profile("decoder_step", sessions["decoder_step"])


if __name__ == "__main__":
    main()
