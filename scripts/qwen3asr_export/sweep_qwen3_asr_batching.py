#!/usr/bin/env python3
"""
Sweep experimental Qwen3-ASR batching artifacts on CUDA and fit a first-pass
VRAM heuristic for batch sizing.

The current export only batches the encoder and decoder prefill. This tool:
  1. builds synthetic segment batches by trimming/repeating one reference clip
  2. runs encoder_batched.onnx and decoder.onnx (unified) or decoder_init_batched.onnx (legacy)
  3. records success / failure, latency, and rough VRAM deltas
  4. fits a simple linear model over successful runs

The fitted model is intended as a sizing heuristic, not a guarantee.

Decoder selection (auto-detected from --onnx-dir):
  decoder.onnx             unified decoder — takes input_embeds + attention_mask + past KV
  decoder_init_batched.onnx  legacy split decoder — takes input_ids + audio_features
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from profile_qwen3_asr_pipeline import load_audio, make_session  # noqa: E402
from src.mel import log_mel_spectrogram  # noqa: E402
from src.prompt import build_prompt_ids, get_audio_pad_range  # noqa: E402
from src.unified_batch import DecoderConfig, build_prefill_inputs  # noqa: E402


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def get_cuda_providers() -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider not available. Providers: {available}")
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def query_gpu_memory_mb() -> tuple[int, int] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_line = completed.stdout.strip().splitlines()[0]
    total_mb, free_mb = [int(part.strip()) for part in first_line.split(",")]
    return total_mb, free_mb


def sample_free_mb() -> int | None:
    stats = query_gpu_memory_mb()
    return None if stats is None else stats[1]


def force_gpu_settle(delay_s: float) -> int | None:
    gc.collect()
    time.sleep(delay_s)
    return sample_free_mb()


def loop_audio_to_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= audio.shape[0]:
        return audio[:target_samples].copy()

    reps = math.ceil(target_samples / audio.shape[0])
    tiled = np.tile(audio, reps)
    return tiled[:target_samples].copy()


@dataclass
class BatchInputs:
    mel: np.ndarray
    input_lengths: np.ndarray
    input_ids: np.ndarray
    position_ids: np.ndarray
    audio_offset: np.ndarray
    audio_lengths: np.ndarray
    max_audio_tokens: int
    total_audio_seconds: float
    max_audio_seconds: float
    batch_size: int


def build_batch_inputs(audio: np.ndarray, durations_s: list[float]) -> BatchInputs:
    mel_items: list[np.ndarray] = []
    mel_lengths: list[int] = []

    for seconds in durations_s:
        target_samples = max(1, int(round(seconds * 16000.0)))
        segment_audio = loop_audio_to_duration(audio, target_samples)
        mel = log_mel_spectrogram(segment_audio).cpu().numpy().astype(np.float32)
        mel_items.append(mel[0])
        mel_lengths.append(int(mel.shape[2]))

        # Match the batched encoder contract by deriving output lengths from real mel lengths later.
        # The decoder prompt is sized from the batched encoder output length.

    max_mel_frames = max(mel_lengths)
    batch_size = len(mel_items)
    mel_batch = np.zeros((batch_size, 128, max_mel_frames), dtype=np.float32)
    for index, mel in enumerate(mel_items):
        mel_batch[index, :, : mel.shape[1]] = mel

    return BatchInputs(
        mel=mel_batch,
        input_lengths=np.array(mel_lengths, dtype=np.int64),
        input_ids=np.empty((0, 0), dtype=np.int64),
        position_ids=np.empty((0, 0), dtype=np.int64),
        audio_offset=np.empty((0,), dtype=np.int64),
        audio_lengths=np.empty((0,), dtype=np.int64),
        max_audio_tokens=0,
        total_audio_seconds=float(sum(durations_s)),
        max_audio_seconds=float(max(durations_s)),
        batch_size=batch_size,
    )


def attach_decoder_inputs(batch: BatchInputs, audio_feature_lengths: np.ndarray) -> BatchInputs:
    max_audio_tokens = int(audio_feature_lengths.max())
    prompt_ids = build_prompt_ids(max_audio_tokens)
    audio_start, _ = get_audio_pad_range(prompt_ids)
    seq_len = len(prompt_ids)
    batch_size = batch.batch_size
    input_ids = np.repeat(np.array(prompt_ids, dtype=np.int64)[np.newaxis, :], batch_size, axis=0)
    position_ids = np.repeat(np.arange(seq_len, dtype=np.int64)[np.newaxis, :], batch_size, axis=0)

    return BatchInputs(
        mel=batch.mel,
        input_lengths=batch.input_lengths,
        input_ids=input_ids,
        position_ids=position_ids,
        audio_offset=np.array([audio_start], dtype=np.int64),
        audio_lengths=audio_feature_lengths.astype(np.int64, copy=False),
        max_audio_tokens=max_audio_tokens,
        total_audio_seconds=batch.total_audio_seconds,
        max_audio_seconds=batch.max_audio_seconds,
        batch_size=batch.batch_size,
    )


def load_decoder_config(onnx_dir: str) -> DecoderConfig:
    config_path = Path(onnx_dir) / "config.json"
    config = json.loads(config_path.read_text())
    dec = config["decoder"]
    return DecoderConfig(
        hidden_size=dec["hidden_size"],
        num_layers=dec["num_layers"],
        num_kv_heads=dec["num_key_value_heads"],
        head_dim=dec["head_dim"],
    )


def load_embed_tokens(onnx_dir: str) -> np.ndarray:
    config_path = Path(onnx_dir) / "config.json"
    config = json.loads(config_path.read_text())
    vocab_size, hidden_size = config["embed_tokens_shape"]
    embed_path = Path(onnx_dir) / "embed_tokens.bin"
    return np.fromfile(embed_path, dtype=np.float32).reshape(vocab_size, hidden_size)


def build_unified_decoder_inputs_with_kv(
    base_batch: BatchInputs,
    audio_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    embed_table: np.ndarray,
    cfg: DecoderConfig,
) -> tuple[dict[str, np.ndarray], int]:
    """Build inputs for decoder.onnx (unified prefill call with zero-size past KV)."""
    decoder_inputs, _ = build_prefill_inputs(audio_features, audio_feature_lengths, embed_table, cfg)
    return decoder_inputs, int(audio_feature_lengths.max())


def timed_run(session: ort.InferenceSession, outputs: list[str], inputs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    start = time.perf_counter()
    result = session.run(outputs, inputs)
    return result, time.perf_counter() - start


def is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "failed to allocate memory" in message
        or "cuda failure 2" in message
        or "cuda error 2" in message
        or "cudnn status alloc failed" in message
    )


def fit_linear_model(rows: list[dict[str, object]]) -> dict[str, float] | None:
    success_rows = [row for row in rows if row["status"] == "ok" and row["peak_vram_delta_mb"] is not None]
    if len(success_rows) < 4:
        return None

    x = np.array(
        [
            [
                1.0,
                float(row["batch_size"]),
                float(row["total_audio_seconds"]),
                float(row["max_audio_seconds"]),
                float(row["max_audio_tokens"]),
            ]
            for row in success_rows
        ],
        dtype=np.float64,
    )
    y = np.array([float(row["peak_vram_delta_mb"]) for row in success_rows], dtype=np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    predictions = x @ coeffs
    residual_rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

    return {
        "intercept_mb": float(coeffs[0]),
        "per_batch_item_mb": float(coeffs[1]),
        "per_total_audio_second_mb": float(coeffs[2]),
        "per_max_audio_second_mb": float(coeffs[3]),
        "per_max_audio_token_mb": float(coeffs[4]),
        "rmse_mb": residual_rmse,
    }


def summarize_frontier(rows: list[dict[str, object]], reference_free_mb: float) -> dict[str, object]:
    by_duration: dict[float, dict[str, list[int]]] = {}
    for row in rows:
        duration = float(row["duration_seconds"])
        stats = by_duration.setdefault(duration, {"ok": [], "oom": []})
        stats["ok" if row["status"] == "ok" else "oom" if row["status"] == "oom" else "other"] = stats.get(
            "ok" if row["status"] == "ok" else "oom" if row["status"] == "oom" else "other",
            [],
        )
        if row["status"] == "ok":
            stats["ok"].append(int(row["batch_size"]))
        elif row["status"] == "oom":
            stats["oom"].append(int(row["batch_size"]))

    duration_rows = []
    frontier_totals: list[float] = []
    for duration in sorted(by_duration):
        ok_batches = sorted(by_duration[duration]["ok"])
        oom_batches = sorted(by_duration[duration]["oom"])
        max_safe = max(ok_batches) if ok_batches else None
        min_fail = min(oom_batches) if oom_batches else None
        safe_total_seconds = (max_safe * duration) if max_safe is not None else None
        if max_safe is not None and min_fail is not None and min_fail > max_safe:
            frontier_totals.append(safe_total_seconds)
        duration_rows.append(
            {
                "duration_seconds": duration,
                "max_safe_batch": max_safe,
                "min_oom_batch": min_fail,
                "safe_total_seconds": safe_total_seconds,
            }
        )

    conservative_total_seconds = min(frontier_totals) if frontier_totals else None
    return {
        "reference_free_mb": reference_free_mb,
        "duration_rows": duration_rows,
        "conservative_total_seconds": conservative_total_seconds,
    }


def estimate_capacity_seconds(
    free_mb: float,
    batch_size: int,
    average_seconds: float,
    fit: dict[str, float],
    safety_mb: float,
) -> float | None:
    per_total_sec = fit["per_total_audio_second_mb"]
    if per_total_sec <= 0:
        return None

    constant_mb = (
        fit["intercept_mb"]
        + fit["per_batch_item_mb"] * batch_size
        + fit["per_max_audio_second_mb"] * average_seconds
    )
    available_mb = free_mb - safety_mb - constant_mb
    if available_mb <= 0:
        return 0.0

    return available_mb / per_total_sec


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_single_point(
    onnx_dir: str,
    audio_path: str,
    duration_s: float,
    batch_size: int,
    warmup_runs: int,
    repeat_runs: int,
    settle_s: float,
) -> dict[str, object]:
    providers = get_cuda_providers()

    unified_path = Path(onnx_dir) / "decoder.onnx"
    split_path = Path(onnx_dir) / "decoder_init_batched.onnx"
    use_unified = unified_path.exists()
    decoder_file = unified_path if use_unified else split_path

    encoder = make_session(str(Path(onnx_dir) / "encoder_batched.onnx"), providers, enable_profiling=False)
    decoder = make_session(str(decoder_file), providers, enable_profiling=False)

    cfg = load_decoder_config(onnx_dir) if use_unified else None
    embed_table = load_embed_tokens(onnx_dir) if use_unified else None

    audio, _ = load_audio(audio_path)
    durations_s = [duration_s] * batch_size
    base_batch = build_batch_inputs(audio, durations_s)

    status = "ok"
    error_message = None
    encoder_ms_values: list[float] = []
    decoder_ms_values: list[float] = []
    free_before_values: list[int] = []
    free_after_encoder_values: list[int] = []
    free_after_decoder_values: list[int] = []
    max_audio_tokens = None

    try:
        for run_index in range(warmup_runs + repeat_runs):
            free_before = force_gpu_settle(settle_s)
            free_before_values.append(-1 if free_before is None else free_before)

            (audio_features, audio_feature_lengths), encoder_s = timed_run(
                encoder,
                ["audio_features", "audio_feature_lengths"],
                {
                    "mel": base_batch.mel,
                    "input_lengths": base_batch.input_lengths,
                },
            )

            free_after_encoder = force_gpu_settle(settle_s)
            free_after_encoder_values.append(-1 if free_after_encoder is None else free_after_encoder)

            if use_unified:
                decoder_inputs, max_audio_tokens = build_unified_decoder_inputs_with_kv(
                    base_batch, audio_features, audio_feature_lengths, embed_table, cfg
                )
            else:
                batch = attach_decoder_inputs(base_batch, audio_feature_lengths)
                max_audio_tokens = batch.max_audio_tokens
                padded_audio_features = audio_features[:, : batch.max_audio_tokens, :]
                decoder_inputs = {
                    "input_ids": batch.input_ids,
                    "position_ids": batch.position_ids,
                    "audio_features": padded_audio_features,
                    "audio_lengths": batch.audio_lengths,
                    "audio_offset": batch.audio_offset,
                }

            _, decoder_s = timed_run(
                decoder,
                ["logits", "present_keys", "present_values"],
                decoder_inputs,
            )

            free_after_decoder = force_gpu_settle(settle_s)
            free_after_decoder_values.append(-1 if free_after_decoder is None else free_after_decoder)

            if run_index >= warmup_runs:
                encoder_ms_values.append(encoder_s * 1000.0)
                decoder_ms_values.append(decoder_s * 1000.0)
    except Exception as exc:  # noqa: BLE001
        status = "oom" if is_oom(exc) else "error"
        error_message = str(exc)

    valid_before = [value for value in free_before_values if value >= 0]
    valid_after_encoder = [value for value in free_after_encoder_values if value >= 0]
    valid_after_decoder = [value for value in free_after_decoder_values if value >= 0]

    peak_vram_delta_mb = None
    encoder_vram_delta_mb = None
    decoder_vram_delta_mb = None
    if valid_before and valid_after_encoder and valid_after_decoder:
        baseline_free = max(valid_before)
        encoder_low = min(valid_after_encoder)
        decoder_low = min(valid_after_decoder)
        encoder_vram_delta_mb = float(baseline_free - encoder_low)
        decoder_vram_delta_mb = float(baseline_free - decoder_low)
        peak_vram_delta_mb = max(encoder_vram_delta_mb, decoder_vram_delta_mb)

    return {
        "duration_seconds": duration_s,
        "batch_size": batch_size,
        "total_audio_seconds": duration_s * batch_size,
        "max_audio_seconds": duration_s,
        "status": status,
        "decoder_mode": "unified" if use_unified else "split",
        "max_audio_tokens": max_audio_tokens,
        "encoder_ms_median": statistics.median(encoder_ms_values) if encoder_ms_values else None,
        "decoder_init_ms_median": statistics.median(decoder_ms_values) if decoder_ms_values else None,
        "encoder_vram_delta_mb": encoder_vram_delta_mb,
        "decoder_init_vram_delta_mb": decoder_vram_delta_mb,
        "peak_vram_delta_mb": peak_vram_delta_mb,
        "error": error_message,
    }


def run_single_point_subprocess(args: argparse.Namespace, duration_s: float, batch_size: int) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--onnx-dir",
        args.onnx_dir,
        "--audio",
        args.audio,
        "--warmup-runs",
        str(args.warmup_runs),
        "--repeat-runs",
        str(args.repeat_runs),
        "--gpu-settle-ms",
        str(args.gpu_settle_ms),
        "--child-duration-seconds",
        str(duration_s),
        "--child-batch-size",
        str(batch_size),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        return json.loads(completed.stdout.strip().splitlines()[-1])

    error = completed.stderr.strip() or completed.stdout.strip() or f"Child sweep failed with exit code {completed.returncode}"
    status = "oom" if "out of memory" in error.lower() else "error"
    return {
        "duration_seconds": duration_s,
        "batch_size": batch_size,
        "total_audio_seconds": duration_s * batch_size,
        "max_audio_seconds": duration_s,
        "status": status,
        "max_audio_tokens": None,
        "encoder_ms_median": None,
        "decoder_init_ms_median": None,
        "encoder_vram_delta_mb": None,
        "decoder_init_vram_delta_mb": None,
        "peak_vram_delta_mb": None,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Qwen3-ASR batching artifacts and fit a VRAM heuristic.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing encoder_batched.onnx and decoder.onnx (unified) or decoder_init_batched.onnx (legacy)")
    parser.add_argument("--audio", required=True, help="Reference audio file used to synthesize sweep batches")
    parser.add_argument("--durations-seconds", default="2,4,8,12,16,24,32", help="Comma-separated segment durations to test")
    parser.add_argument("--batch-sizes", default="1,2,4,6,8,10,12", help="Comma-separated batch sizes to test")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing")
    parser.add_argument("--repeat-runs", type=int, default=2, help="Measured runs per sweep point")
    parser.add_argument("--gpu-settle-ms", type=int, default=750, help="Delay between VRAM samples")
    parser.add_argument("--output-json", default=None, help="Optional path to write structured JSON results")
    parser.add_argument("--output-csv", default=None, help="Optional path to write CSV results")
    parser.add_argument("--safety-mb", type=float, default=1024.0, help="Extra safety margin to subtract when turning fit into a heuristic")
    parser.add_argument("--child-duration-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_duration_seconds is not None or args.child_batch_size is not None:
        if args.child_duration_seconds is None or args.child_batch_size is None:
            raise ValueError("Both child sweep arguments are required together")
        row = run_single_point(
            onnx_dir=args.onnx_dir,
            audio_path=args.audio,
            duration_s=args.child_duration_seconds,
            batch_size=args.child_batch_size,
            warmup_runs=args.warmup_runs,
            repeat_runs=args.repeat_runs,
            settle_s=args.gpu_settle_ms / 1000.0,
        )
        print(json.dumps(row))
        return

    onnx_dir = Path(args.onnx_dir)
    encoder_path = onnx_dir / "encoder_batched.onnx"
    unified_decoder_path = onnx_dir / "decoder.onnx"
    split_decoder_path = onnx_dir / "decoder_init_batched.onnx"
    if not encoder_path.exists():
        raise FileNotFoundError(encoder_path)
    if not unified_decoder_path.exists() and not split_decoder_path.exists():
        raise FileNotFoundError(f"Neither {unified_decoder_path} nor {split_decoder_path} found")
    decoder_mode = "unified" if unified_decoder_path.exists() else "split"
    durations = parse_csv_floats(args.durations_seconds)
    batch_sizes = parse_csv_ints(args.batch_sizes)

    initial_gpu = query_gpu_memory_mb()
    if initial_gpu is None:
        raise RuntimeError("Could not query GPU memory via nvidia-smi")

    total_mb, initial_free_mb = initial_gpu
    print(f"GPU memory at start: total={total_mb} MB free={initial_free_mb} MB")
    print(f"Decoder mode: {decoder_mode}")
    print(f"Testing durations={durations} seconds batch_sizes={batch_sizes}")

    rows: list[dict[str, object]] = []

    for duration_s in durations:
        for batch_size in batch_sizes:
            print(f"\n=== Sweep point: duration={duration_s:.1f}s batch={batch_size} total={duration_s * batch_size:.1f}s ===")
            row = run_single_point_subprocess(args, duration_s, batch_size)
            rows.append(row)

            print(json.dumps(row, indent=2))

    fit = fit_linear_model(rows)
    summary = {
        "gpu_total_mb": total_mb,
        "gpu_free_mb_at_start": initial_free_mb,
        "fit": fit,
        "frontier": summarize_frontier(rows, initial_free_mb),
    }

    print("\nObserved frontier")
    for frontier_row in summary["frontier"]["duration_rows"]:
        print(
            "  "
            f"duration={frontier_row['duration_seconds']:>4.1f}s "
            f"max_safe_batch={frontier_row['max_safe_batch']} "
            f"min_oom_batch={frontier_row['min_oom_batch']} "
            f"safe_total_seconds={frontier_row['safe_total_seconds']}"
        )
    if summary["frontier"]["conservative_total_seconds"] is not None:
        print(
            "  "
            f"conservative_total_seconds_at_{initial_free_mb:.0f}MB_free="
            f"{summary['frontier']['conservative_total_seconds']:.1f}s"
        )

    if fit is not None:
        print("\nFitted VRAM model (MB)")
        print(json.dumps(fit, indent=2))
        print("\nExample capacity estimates")
        for average_seconds in (2.0, 4.0, 8.0, 16.0):
            for batch_size in (1, 2, 4, 8):
                estimate = estimate_capacity_seconds(initial_free_mb, batch_size, average_seconds, fit, args.safety_mb)
                if estimate is None:
                    continue
                print(
                    f"  avg_seg={average_seconds:4.1f}s batch={batch_size:2d} "
                    f"=> estimated total_seconds_ceiling={estimate:6.1f}s"
                )
    else:
        print("\nNot enough successful points with VRAM samples to fit a heuristic.")

    payload = {
        "summary": summary,
        "rows": rows,
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.output_json}")

    if args.output_csv:
        write_csv(args.output_csv, rows)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
