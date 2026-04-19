"""Phase 2: sweep session-level knobs on CPU at the 15 s target duration.

Varies:
  - graph_optimization_level: DISABLE_ALL, BASIC, EXTENDED, ALL
  - intra_op_num_threads: 1, 2, 4, 6, 8, 16
  - execution_mode: SEQUENTIAL, PARALLEL (with inter_op=2 when parallel)

Reports p50 / p95 latency per configuration, sorted best-first, so we
can pick a winner and update the default.

Usage:
    python phase2_session_sweep.py \\
        --model-dir ./voxlingua107 \\
        --audio ./en-US_sample_01_first90.wav \\
        --duration 15 \\
        --runs 50 --warmup 5 \\
        --csv ./phase2_sweep.csv
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


GRAPH_OPT_LEVELS = [
    "ORT_DISABLE_ALL",
    "ORT_ENABLE_BASIC",
    "ORT_ENABLE_EXTENDED",
    "ORT_ENABLE_ALL",
]


def load_clip(path: Path, duration_s: float, sr: int = 16_000) -> np.ndarray:
    audio, r = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if r != sr:
        raise SystemExit(f"{path} is {r} Hz; expected {sr}")
    n = int(duration_s * sr)
    if len(audio) < n:
        audio = np.tile(audio, (n // len(audio)) + 1)[:n]
    return audio[:n].astype(np.float32, copy=False)[None, :]


def bench_config(
    model_path: Path,
    audio: np.ndarray,
    graph_opt: str,
    intra_op: int,
    execution_mode: str,
    inter_op: int,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, graph_opt)
    opts.intra_op_num_threads = intra_op
    opts.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL if execution_mode == "PARALLEL"
        else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    if execution_mode == "PARALLEL":
        opts.inter_op_num_threads = inter_op

    t_load0 = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"],
    )
    load_ms = (time.perf_counter() - t_load0) * 1000.0

    for _ in range(warmup):
        session.run(["logits", "embedding"], {"audio": audio})

    lats: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(["logits", "embedding"], {"audio": audio})
        lats.append((time.perf_counter() - t0) * 1000.0)

    return {
        "load_ms": round(load_ms, 2),
        "p50_ms": round(float(np.percentile(lats, 50)), 2),
        "p95_ms": round(float(np.percentile(lats, 95)), 2),
        "p99_ms": round(float(np.percentile(lats, 99)), 2),
        "mean_ms": round(statistics.fmean(lats), 2),
        "stdev_ms": round(statistics.pstdev(lats), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    audio = load_clip(args.audio, args.duration)
    model_path = args.model_dir / "voxlingua107.onnx"

    # Threading sweep (sequential) — full matrix.
    thread_counts = [1, 2, 4, 6, 8, 16]
    configs: list[tuple[str, int, str, int]] = []
    for graph_opt in GRAPH_OPT_LEVELS:
        for n in thread_counts:
            configs.append((graph_opt, n, "SEQUENTIAL", 1))

    # A few PARALLEL configs just to confirm the conventional wisdom that
    # ECAPA-TDNN gains nothing from inter-op parallelism on CPU.
    for n in [4, 8]:
        for inter in [2, 4]:
            configs.append(("ORT_ENABLE_EXTENDED", n, "PARALLEL", inter))

    rows: list[dict] = []
    print(f"{'graph_opt':<20s}  {'intra':>5s}  {'mode':>11s}  {'inter':>5s}  "
          f"{'p50_ms':>8s}  {'p95_ms':>8s}  {'load_ms':>8s}", file=sys.stderr)
    for graph_opt, intra, mode, inter in configs:
        stats = bench_config(model_path, audio, graph_opt, intra, mode, inter,
                             args.warmup, args.runs)
        row = {
            "graph_opt": graph_opt,
            "intra_op": intra,
            "execution_mode": mode,
            "inter_op": inter if mode == "PARALLEL" else 0,
            **stats,
        }
        rows.append(row)
        inter_display = inter if mode == "PARALLEL" else "-"
        print(f"{graph_opt:<20s}  {intra:>5d}  {mode:>11s}  {inter_display:>5}  "
              f"{stats['p50_ms']:>8.2f}  {stats['p95_ms']:>8.2f}  {stats['load_ms']:>8.2f}",
              file=sys.stderr)

    # Summary: top 5 by p50
    rows_sorted = sorted(rows, key=lambda r: r["p50_ms"])
    print("\nTop 5 by p50:", file=sys.stderr)
    for r in rows_sorted[:5]:
        print(f"  {r['graph_opt']:<20s} intra={r['intra_op']:>2} {r['execution_mode']:>11s} "
              f"inter={r['inter_op'] or '-':>1}  p50={r['p50_ms']:>7.2f}ms  "
              f"p95={r['p95_ms']:>7.2f}ms",
              file=sys.stderr)

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\nwrote {len(rows)} rows → {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
