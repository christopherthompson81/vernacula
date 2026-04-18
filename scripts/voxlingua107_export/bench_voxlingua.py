"""Benchmark VoxLingua107 ONNX inference across providers / precisions / durations.

The same harness is reused across every phase of the performance investigation
so results are comparable. Outputs a CSV (stdout by default) with one row per
configuration.

Usage:
    python bench_voxlingua.py \\
        --model-dir ./voxlingua107 \\
        --audio ./en-US_sample_01_first90.wav \\
        --providers cpu cuda \\
        --precisions fp32 \\
        --batch-sizes 1 \\
        --durations 10 30 60 90 180 \\
        --runs 50 --warmup 5 \\
        --csv ./phase0_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def load_audio(path: Path, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        raise SystemExit(f"audio at {path} is {sr} Hz; this harness expects {target_sr} Hz")
    return audio


def slice_to_duration(audio: np.ndarray, duration_s: float, sr: int = 16_000) -> np.ndarray:
    n = int(duration_s * sr)
    if len(audio) >= n:
        return audio[:n].astype(np.float32, copy=False)
    # Pad with reflection so short files still produce a realistic signal.
    reps = (n // len(audio)) + 1
    tiled = np.tile(audio, reps)[:n]
    return tiled.astype(np.float32, copy=False)


def build_session(
    model_path: Path,
    provider: str,
    graph_opt_level: str = "ORT_ENABLE_BASIC",
    intra_op: int | None = None,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, graph_opt_level, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    )
    if intra_op is not None:
        opts.intra_op_num_threads = intra_op
    ep = PROVIDER_MAP[provider]
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=[ep])


def pXX(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, pct))


def bench_one_config(
    session: ort.InferenceSession,
    audio: np.ndarray,
    duration_s: float,
    batch_size: int,
    runs: int,
    warmup: int,
) -> dict[str, float]:
    clip = slice_to_duration(audio, duration_s)
    batch = np.repeat(clip[None, :], batch_size, axis=0).astype(np.float32)

    # Warm up.
    for _ in range(warmup):
        session.run(["logits", "embedding"], {"audio": batch})

    latencies_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(["logits", "embedding"], {"audio": batch})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "n_runs": runs,
        "p50_ms": pXX(latencies_ms, 50),
        "p95_ms": pXX(latencies_ms, 95),
        "p99_ms": pXX(latencies_ms, 99),
        "mean_ms": statistics.fmean(latencies_ms),
        "stdev_ms": statistics.pstdev(latencies_ms),
    }


def run_matrix(args: argparse.Namespace) -> list[dict]:
    model_path = args.model_dir / "voxlingua107.onnx"
    audio = load_audio(args.audio)

    rows: list[dict] = []
    for precision in args.precisions:
        if precision != "fp32":
            # Phase 3 will introduce fp16 / int8 variants; skip for now.
            print(f"[bench] skipping precision={precision} (not built yet)", file=sys.stderr)
            continue

        for provider in args.providers:
            available = ort.get_available_providers()
            if PROVIDER_MAP[provider] not in available:
                print(f"[bench] provider={provider} not available; skipping", file=sys.stderr)
                continue

            t_load0 = time.perf_counter()
            session = build_session(model_path, provider,
                                    graph_opt_level=args.graph_opt_level,
                                    intra_op=args.intra_op)
            load_ms = (time.perf_counter() - t_load0) * 1000.0
            print(f"[bench] loaded provider={provider} in {load_ms:.1f} ms", file=sys.stderr)

            for batch_size in args.batch_sizes:
                for duration_s in args.durations:
                    stats = bench_one_config(session, audio, duration_s, batch_size,
                                             args.runs, args.warmup)
                    row = {
                        "provider": provider,
                        "precision": precision,
                        "batch_size": batch_size,
                        "duration_s": duration_s,
                        "load_ms": round(load_ms, 2),
                        **{k: round(v, 3) for k, v in stats.items()},
                    }
                    rows.append(row)
                    print(
                        f"[bench] {provider:4s} fp32 b={batch_size} "
                        f"d={duration_s:3.0f}s  "
                        f"p50={stats['p50_ms']:7.2f}ms  "
                        f"p95={stats['p95_ms']:7.2f}ms  "
                        f"p99={stats['p99_ms']:7.2f}ms",
                        file=sys.stderr,
                    )
    return rows


def write_csv(rows: list[dict], path: Path | None) -> None:
    if not rows:
        print("[bench] no rows to write", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[bench] wrote {len(rows)} rows → {path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory containing voxlingua107.onnx.")
    p.add_argument("--audio", type=Path, required=True,
                   help="16 kHz mono WAV file used as the source clip for all durations.")
    p.add_argument("--providers", nargs="+", default=["cpu"],
                   choices=sorted(PROVIDER_MAP.keys()))
    p.add_argument("--precisions", nargs="+", default=["fp32"],
                   choices=["fp32", "fp16", "int8"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--durations", type=float, nargs="+",
                   default=[10, 30, 60, 90, 180])
    p.add_argument("--runs", type=int, default=50,
                   help="Measured runs per (provider, precision, batch, duration).")
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup runs discarded before measurement.")
    p.add_argument("--graph-opt-level", default="ORT_ENABLE_BASIC",
                   choices=["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC",
                            "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"])
    p.add_argument("--intra-op", type=int, default=None,
                   help="Override intra_op_num_threads (ORT default if unset).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Write CSV to this path. Stdout if omitted.")
    args = p.parse_args()

    rows = run_matrix(args)
    write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
