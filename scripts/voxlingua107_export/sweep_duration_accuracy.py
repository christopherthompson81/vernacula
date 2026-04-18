"""Measure VoxLingua107 top-1 accuracy vs clip duration.

For each (clip, duration) pair, slices the clip into N non-overlapping
chunks of that duration and runs ONNX inference on each. Reports
per-duration accuracy (top-1 correctness against the clip's known
language) and mean top-1 probability (confidence).

Usage:
    python sweep_duration_accuracy.py \\
        --model-dir ./voxlingua107 \\
        --provider cpu \\
        --durations 5 10 15 30 60 \\
        --clip en=./en-US_sample_01_first90.wav \\
        --clip de=./de-DE_sample_01.wav \\
        --clip fr=./fr-FR_sample_01.wav \\
        --clip ru=./ru-RU_sample_01.wav \\
        --clip hu=./hu-HU_sample_01.wav \\
        --csv ./phase6_accuracy.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=-1, keepdims=True)


def load_clip(path: Path, sr: int = 16_000) -> np.ndarray:
    audio, r = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if r != sr:
        raise SystemExit(f"{path} is {r} Hz; expected {sr}")
    return audio.astype(np.float32, copy=False)


def chunks(audio: np.ndarray, duration_s: float, max_chunks: int,
           sr: int = 16_000) -> list[np.ndarray]:
    n = int(duration_s * sr)
    if len(audio) < n:
        return []  # can't sample at this duration
    available = len(audio) // n
    count = min(available, max_chunks)
    return [audio[i * n:(i + 1) * n] for i in range(count)]


def iso_of_top1(logits: np.ndarray, lang_map: dict) -> tuple[str, float]:
    probs = softmax(logits[0])
    top1 = int(np.argmax(probs))
    return lang_map[str(top1)]["iso"], float(probs[top1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--clip", action="append", required=True, metavar="ISO=PATH",
                   help="Labelled clip; repeatable. Example: en=./foo.wav")
    p.add_argument("--durations", type=float, nargs="+",
                   default=[5, 10, 15, 30, 60])
    p.add_argument("--provider", choices=sorted(PROVIDER_MAP), default="cpu")
    p.add_argument("--max-chunks", type=int, default=10,
                   help="Max non-overlapping chunks to sample per (clip, duration).")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    clips: list[tuple[str, Path]] = []
    for entry in args.clip:
        if "=" not in entry:
            raise SystemExit(f"--clip expects ISO=PATH, got: {entry}")
        iso, path = entry.split("=", 1)
        clips.append((iso.strip(), Path(path.strip())))

    lang_map = json.loads((args.model_dir / "lang_map.json").read_text())
    session = ort.InferenceSession(
        str(args.model_dir / "voxlingua107.onnx"),
        providers=[PROVIDER_MAP[args.provider]],
    )

    rows: list[dict] = []
    header = f"{'duration':>8s}  {'clip':>10s}  {'n':>3s}  {'correct':>7s}  " \
             f"{'acc':>6s}  {'mean_p':>7s}  {'predictions'}"
    print(header, file=sys.stderr)

    for duration in args.durations:
        for expected_iso, path in clips:
            audio = load_clip(path)
            chs = chunks(audio, duration, args.max_chunks)
            if not chs:
                print(f"{duration:>8.0f}s  {expected_iso:>10s}  "
                      f"(clip too short: {len(audio) / 16000:.1f}s)",
                      file=sys.stderr)
                continue

            predictions: list[str] = []
            probs: list[float] = []
            for chunk in chs:
                logits, _ = session.run(["logits", "embedding"],
                                        {"audio": chunk[None, :]})
                iso, prob = iso_of_top1(logits, lang_map)
                predictions.append(iso)
                probs.append(prob)

            correct = sum(1 for p in predictions if p == expected_iso)
            acc = correct / len(predictions)
            mean_prob = statistics.fmean(probs)
            # Compact histogram of predictions.
            counts: dict[str, int] = {}
            for p in predictions:
                counts[p] = counts.get(p, 0) + 1
            hist = ", ".join(f"{k}×{v}" for k, v in
                             sorted(counts.items(), key=lambda kv: -kv[1]))

            print(f"{duration:>8.0f}s  {expected_iso:>10s}  {len(predictions):>3d}  "
                  f"{correct:>7d}  {acc:>6.2%}  {mean_prob:>7.3f}  {hist}",
                  file=sys.stderr)
            rows.append({
                "duration_s": duration,
                "expected_iso": expected_iso,
                "n_chunks": len(predictions),
                "correct": correct,
                "accuracy": round(acc, 4),
                "mean_top1_prob": round(mean_prob, 4),
                "predictions": hist,
            })

    # Aggregate per-duration accuracy across clips.
    print("\nPer-duration aggregate:", file=sys.stderr)
    print(f"{'duration':>8s}  {'n_total':>7s}  {'correct':>7s}  {'acc':>6s}  "
          f"{'mean_p':>7s}", file=sys.stderr)
    for duration in args.durations:
        matching = [r for r in rows if r["duration_s"] == duration]
        if not matching:
            continue
        total_n = sum(r["n_chunks"] for r in matching)
        total_correct = sum(r["correct"] for r in matching)
        acc = total_correct / total_n if total_n else 0.0
        mean_p = statistics.fmean(r["mean_top1_prob"] for r in matching)
        print(f"{duration:>8.0f}s  {total_n:>7d}  {total_correct:>7d}  "
              f"{acc:>6.2%}  {mean_p:>7.3f}", file=sys.stderr)

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
