"""Phase 4: measure CUDA IO binding vs plain session.run.

Two scenarios are interesting, both at the 15 s target:
  A) 50 serial b=1 calls, one per segment — the per-segment LID workload.
  B) Batched b=16 calls — same work, batched.

For each, we compare:
  - Plain path: np → session.run(feed) → np back out every call.
  - IO-bound:   OrtValue on GPU for audio and outputs, refill input buffer
                each call, run via session.run_with_iobinding().

This isolates whether Python↔ORT allocation overhead and H2D/D2H copy
setup is hiding in the per-segment budget.

Usage:
    python bench_iobinding.py \\
        --model-dir ./voxlingua107 \\
        --audio ./en-US_sample_01_first90.wav \\
        --n-segments 50 --duration 15 --runs 10 --warmup 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


def load_audio(path: Path, sr: int = 16_000) -> np.ndarray:
    audio, r = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if r != sr:
        raise SystemExit(f"{path} is {r} Hz; expected {sr}")
    return audio.astype(np.float32, copy=False)


def prepare_segments(audio: np.ndarray, n: int, duration_s: float,
                     sr: int = 16_000) -> list[np.ndarray]:
    """Make n distinct same-length clips by tiling if needed."""
    clip_len = int(duration_s * sr)
    if len(audio) < clip_len:
        audio = np.tile(audio, (clip_len // len(audio)) + 1)[:clip_len * 2]
    # Return n views with slightly different start offsets so there's real
    # per-call variation (prevents kernel caching illusions).
    segments = []
    for i in range(n):
        start = (i * 1000) % max(1, len(audio) - clip_len)
        segments.append(audio[start:start + clip_len].astype(np.float32, copy=False))
    return segments


def bench_plain_serial(session: ort.InferenceSession,
                       segments: list[np.ndarray],
                       runs: int, warmup: int) -> dict[str, float]:
    """50-segment serial workload via plain session.run."""
    feed = segments[0][None, :]
    for _ in range(warmup):
        session.run(["logits", "embedding"], {"audio": feed})

    totals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for seg in segments:
            session.run(["logits", "embedding"], {"audio": seg[None, :]})
        totals.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(totals, 50)),
        "p95_ms": float(np.percentile(totals, 95)),
        "mean_ms": statistics.fmean(totals),
    }


def bench_iobinding_serial(session: ort.InferenceSession,
                           segments: list[np.ndarray],
                           runs: int, warmup: int,
                           device_id: int = 0) -> dict[str, float]:
    """50-segment serial workload via IO binding — inputs stay device-resident."""
    clip_len = segments[0].shape[0]
    # Pre-allocate device buffers: input [1, T], logits [1, 107], embedding [1, 256].
    in_buf = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, clip_len], np.float32, "cuda", device_id)
    logits_buf = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, 107], np.float32, "cuda", device_id)
    emb_buf = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, 256], np.float32, "cuda", device_id)

    binding = session.io_binding()
    binding.bind_ortvalue_input("audio", in_buf)
    binding.bind_ortvalue_output("logits", logits_buf)
    binding.bind_ortvalue_output("embedding", emb_buf)

    # Warm up.
    for seg in segments[:warmup]:
        in_buf.update_inplace(seg[None, :])
        session.run_with_iobinding(binding)

    totals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for seg in segments:
            in_buf.update_inplace(seg[None, :])  # H2D copy into pre-allocated buffer
            session.run_with_iobinding(binding)
        totals.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(totals, 50)),
        "p95_ms": float(np.percentile(totals, 95)),
        "mean_ms": statistics.fmean(totals),
    }


def bench_plain_batched(session: ort.InferenceSession,
                        segments: list[np.ndarray],
                        batch_size: int,
                        runs: int, warmup: int) -> dict[str, float]:
    """Same n segments, batched b=batch_size via plain session.run."""
    n_batches = (len(segments) + batch_size - 1) // batch_size
    batches: list[np.ndarray] = []
    for i in range(n_batches):
        chunk = segments[i * batch_size:(i + 1) * batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + [chunk[-1]] * (batch_size - len(chunk))
        batches.append(np.stack(chunk, axis=0).astype(np.float32))

    for _ in range(warmup):
        session.run(["logits", "embedding"], {"audio": batches[0]})

    totals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for b in batches:
            session.run(["logits", "embedding"], {"audio": b})
        totals.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(totals, 50)),
        "p95_ms": float(np.percentile(totals, 95)),
        "mean_ms": statistics.fmean(totals),
        "n_batches": n_batches,
    }


def bench_iobinding_batched(session: ort.InferenceSession,
                            segments: list[np.ndarray],
                            batch_size: int,
                            runs: int, warmup: int,
                            device_id: int = 0) -> dict[str, float]:
    """Same n segments, batched b=batch_size via IO binding."""
    n_batches = (len(segments) + batch_size - 1) // batch_size
    clip_len = segments[0].shape[0]

    batches: list[np.ndarray] = []
    for i in range(n_batches):
        chunk = segments[i * batch_size:(i + 1) * batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + [chunk[-1]] * (batch_size - len(chunk))
        batches.append(np.stack(chunk, axis=0).astype(np.float32))

    in_buf = ort.OrtValue.ortvalue_from_shape_and_type(
        [batch_size, clip_len], np.float32, "cuda", device_id)
    logits_buf = ort.OrtValue.ortvalue_from_shape_and_type(
        [batch_size, 107], np.float32, "cuda", device_id)
    emb_buf = ort.OrtValue.ortvalue_from_shape_and_type(
        [batch_size, 256], np.float32, "cuda", device_id)

    binding = session.io_binding()
    binding.bind_ortvalue_input("audio", in_buf)
    binding.bind_ortvalue_output("logits", logits_buf)
    binding.bind_ortvalue_output("embedding", emb_buf)

    for b in batches[:warmup]:
        in_buf.update_inplace(b)
        session.run_with_iobinding(binding)

    totals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for b in batches:
            in_buf.update_inplace(b)
            session.run_with_iobinding(binding)
        totals.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(totals, 50)),
        "p95_ms": float(np.percentile(totals, 95)),
        "mean_ms": statistics.fmean(totals),
        "n_batches": n_batches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--n-segments", type=int, default=50)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device-id", type=int, default=0)
    args = p.parse_args()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(
        str(args.model_dir / "voxlingua107.onnx"),
        sess_options=opts,
        providers=["CUDAExecutionProvider"],
    )

    audio = load_audio(args.audio)
    segments = prepare_segments(audio, args.n_segments, args.duration)
    print(f"[bench] {args.n_segments} segments × {args.duration} s @ 15 s clip",
          file=sys.stderr)

    # Serial scenario: 50 × b=1. (Batched path requires variable-batch export
    # which Phase 5 showed wasn't a useful lever, so we don't re-run it here.)
    plain_serial = bench_plain_serial(session, segments, args.runs, args.warmup)
    iob_serial = bench_iobinding_serial(session, segments, args.runs,
                                        args.warmup, args.device_id)

    def fmt(tag: str, s: dict) -> str:
        return (f"  {tag:<30s} p50={s['p50_ms']:>7.2f}ms  "
                f"p95={s['p95_ms']:>7.2f}ms  mean={s['mean_ms']:>7.2f}ms")

    print("\nSerial (50 × b=1):")
    print(fmt("plain", plain_serial))
    print(fmt("iobinding", iob_serial))
    speedup_serial = plain_serial["p50_ms"] / iob_serial["p50_ms"]
    print(f"  speedup: {speedup_serial:.2f}×")


if __name__ == "__main__":
    main()
