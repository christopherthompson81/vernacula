"""Profile VoxLingua107 ONNX inference and summarize per-op time.

Runs the model under ORT's built-in profiler, loads the resulting JSON,
and prints two summaries:
  1. Per-op-type aggregated timing (sorted by total μs)
  2. Provider assignments: which nodes ran on which EP (flags memcpy)

Usage:
    python profile_voxlingua.py \\
        --model-dir ./voxlingua107 \\
        --audio ./en-US_sample_01_first90.wav \\
        --provider cuda \\
        --duration 30 \\
        --out-dir ./profile_cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def load_clip(path: Path, duration_s: float, sr: int = 16_000) -> np.ndarray:
    audio, r = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if r != sr:
        raise SystemExit(f"audio is {r} Hz, expected {sr}")
    n = int(duration_s * sr)
    if len(audio) < n:
        audio = np.tile(audio, (n // len(audio)) + 1)[:n]
    return audio[:n].astype(np.float32, copy=False)[None, :]


def run_profiled(model_path: Path, provider: str, audio: np.ndarray,
                 out_dir: Path, runs: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = str(out_dir / f"voxlingua_{provider}")
    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=[PROVIDER_MAP[provider]],
    )
    # Warm up + measured runs (profile captures everything, we'll filter later).
    for _ in range(3):
        session.run(["logits", "embedding"], {"audio": audio})
    for _ in range(runs):
        session.run(["logits", "embedding"], {"audio": audio})

    profile_path = Path(session.end_profiling())
    print(f"[profile] wrote {profile_path}", file=sys.stderr)
    return profile_path


def summarize_ops(profile_path: Path, top_n: int) -> None:
    entries = json.loads(profile_path.read_text())
    # Filter to node-level inference events only (skip session/graph events).
    node_events = [e for e in entries
                   if e.get("cat") == "Node" and "op_name" in e.get("args", {})]
    if not node_events:
        print("[profile] no node-level events — is profiling enabled?", file=sys.stderr)
        return

    per_op: dict[str, list[int]] = defaultdict(list)
    per_provider: dict[str, list[int]] = defaultdict(list)
    node_assignment: dict[str, tuple[str, str]] = {}  # name → (op, ep)

    for e in node_events:
        args = e["args"]
        op = args["op_name"]
        ep = args.get("provider", "Unknown")
        dur_us = int(e.get("dur", 0))
        per_op[op].append(dur_us)
        per_provider[ep].append(dur_us)
        node_assignment[args.get("node_name", e["name"])] = (op, ep)

    total_us = sum(dur for durs in per_op.values() for dur in durs)

    print("\n=== Per-op-type time (all runs combined) ===")
    print(f"{'op':<30s}  {'calls':>6s}  {'total_ms':>10s}  {'mean_us':>10s}  {'% total':>8s}")
    rows = sorted(per_op.items(), key=lambda kv: -sum(kv[1]))
    for op, durs in rows[:top_n]:
        total_ms = sum(durs) / 1000.0
        mean_us = statistics.fmean(durs)
        pct = 100.0 * sum(durs) / total_us if total_us else 0.0
        print(f"{op:<30s}  {len(durs):>6d}  {total_ms:>10.2f}  {mean_us:>10.1f}  {pct:>7.1f}%")
    remaining = sum(sum(d) for _, d in rows[top_n:])
    if remaining:
        pct = 100.0 * remaining / total_us
        print(f"{'... rest':<30s}  {'':>6s}  {remaining / 1000.0:>10.2f}  "
              f"{'':>10s}  {pct:>7.1f}%")

    print("\n=== Time per provider ===")
    for ep, durs in sorted(per_provider.items(), key=lambda kv: -sum(kv[1])):
        total_ms = sum(durs) / 1000.0
        pct = 100.0 * sum(durs) / total_us if total_us else 0.0
        print(f"{ep:<35s}  {len(durs):>6d} calls  {total_ms:>10.2f} ms  {pct:>6.1f}%")

    print("\n=== Nodes assigned to non-preferred providers ===")
    cpu_on_cuda_session = [
        (name, op) for name, (op, ep) in node_assignment.items()
        if ep == "CPUExecutionProvider" and op not in {"MemcpyToHost", "MemcpyFromHost"}
    ]
    memcpy_nodes = [
        (name, op) for name, (op, ep) in node_assignment.items()
        if op in {"MemcpyToHost", "MemcpyFromHost"}
    ]
    if memcpy_nodes:
        print(f"Memcpy nodes: {len(memcpy_nodes)}")
        for name, op in memcpy_nodes:
            print(f"  {op:<18s}  {name}")
    if cpu_on_cuda_session:
        print(f"\nNon-Memcpy CPU-assigned nodes ({len(cpu_on_cuda_session)}):")
        for name, op in cpu_on_cuda_session[:20]:
            print(f"  {op:<18s}  {name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--provider", choices=sorted(PROVIDER_MAP), default="cuda")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--runs", type=int, default=10,
                   help="Measured runs recorded in the profile.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory to write the raw profile JSON.")
    p.add_argument("--top-n", type=int, default=15,
                   help="Rows in the per-op-type summary table.")
    args = p.parse_args()

    audio = load_clip(args.audio, args.duration)
    print(f"[profile] clip: {audio.shape} at 16 kHz ({args.duration} s)", file=sys.stderr)
    profile_path = run_profiled(args.model_dir / "voxlingua107.onnx",
                                args.provider, audio, args.out_dir, args.runs)
    summarize_ops(profile_path, args.top_n)


if __name__ == "__main__":
    main()
