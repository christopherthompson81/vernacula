#!/usr/bin/env python3
"""Score the C# Nemotron-3-Diarization port against NeMo, and both models against labels.

Inputs come from two tools that must run first:

  * scripts/nemo_export/nemotron3_diarization_reference.py  -> <stem>.nemo_preds.npy
    (NeMo's own forward_streaming -- the REFERENCE, never a Python mirror of the port);
  * tests/Nemotron3DiarizationParity (C#)                    -> <stem>.<model>.preds.f32 / .rttm

Reports, per file:

  1. Port parity on the raw 10 ms probabilities: max / mean |C# - NeMo| and the fraction of
     (frame, speaker) cells that binarize differently at 0.5.
  2. Port fidelity DER: the C# segments against NeMo's probabilities put through the SAME
     binarization (threshold 0.5, 10 ms frames) -- so only the streaming loop is under test.
  3. Accuracy DER against reference RTTMs, for Nemotron-3 and (if present) v2.1, as the
     application emits them. This is the only number here that says anything about quality.

Usage:

    python scripts/nemo_export/nemotron3_diarization_fidelity.py \\
        --audio-dir <dir with <stem>.rttm labels> --nemo-dir <ref> --csharp-dir <out> \\
        --stems vox_dev_a ami_sdm_a ... [--max-fidelity-der 0.01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FRAME = 0.01
NUM_SPK = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--nemo-dir", type=Path, required=True)
    p.add_argument("--csharp-dir", type=Path, required=True)
    p.add_argument("--stems", nargs="+", required=True)
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--max-fidelity-der", type=float, default=None,
                   help="Exit 1 if any file's fidelity DER exceeds this.")
    return p.parse_args()


def binarize(preds: np.ndarray, threshold: float = 0.5):
    """Mirror of SortformerStreamer.BinarizePredToSegments at Nemotron-3's profile
    (onset = offset = 0.5, no pads, no duration filtering, 10 ms frames)."""
    segs = []
    T, S = preds.shape
    for s in range(S):
        on = preds[:, s] >= threshold
        t = 0
        while t < T:
            if on[t]:
                u = t
                while u < T and on[u]:
                    u += 1
                segs.append((t * FRAME, u * FRAME, f"speaker_{s}"))
                t = u
            else:
                t += 1
    return segs


def read_rttm(path: Path):
    segs = []
    for line in path.read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            st, du = float(f[3]), float(f[4])
            segs.append((st, st + du, f[7]))
    return segs


def annotation(segs, uri):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for st, en, spk in segs:
        if en > st:
            ann[Segment(st, en)] = spk
    return ann


def der(ref, hyp, uri, collar, skip_overlap):
    from pyannote.metrics.diarization import DiarizationErrorRate
    metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip_overlap)
    return metric(annotation(ref, uri), annotation(hyp, uri))


def main() -> int:
    args = parse_args()
    worst_fid = 0.0
    rows = []
    for stem in args.stems:
        nemo = np.load(args.nemo_dir / f"{stem}.nemo_preds.npy")
        cs = np.fromfile(args.csharp_dir / f"{stem}.nemotron3.preds.f32", dtype="<f4").reshape(-1, NUM_SPK)
        n = min(len(nemo), len(cs))
        d = np.abs(cs[:n].astype(np.float64) - nemo[:n])
        flips = float(((cs[:n] >= 0.5) != (nemo[:n] >= 0.5)).mean())

        fid = der(binarize(nemo[:n]), read_rttm(args.csharp_dir / f"{stem}.nemotron3.rttm"),
                  stem, collar=0.0, skip_overlap=False)
        worst_fid = max(worst_fid, fid)

        row = {"stem": stem, "frames": f"{len(cs)}/{len(nemo)}", "max|d|": d.max(),
               "mean|d|": d.mean(), "flip%": 100 * flips, "fidDER%": 100 * fid}
        gt_path = args.audio_dir / f"{stem}.rttm"
        if gt_path.exists():
            gt = read_rttm(gt_path)
            for tag in ("nemotron3", "v21"):
                hyp_path = args.csharp_dir / f"{stem}.{tag}.rttm"
                if hyp_path.exists():
                    hyp = read_rttm(hyp_path)
                    row[f"{tag} DER%"] = 100 * der(gt, hyp, stem, args.collar, skip_overlap=False)
                    row[f"{tag} DER0%"] = 100 * der(gt, hyp, stem, 0.0, skip_overlap=False)
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    print("  ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print("  ".join(f"{r.get(k, ''):>14.4g}" if isinstance(r.get(k), float) else f"{str(r.get(k, '')):>14}"
                        for k in keys))
    print(f"\nDER% = collar {args.collar}s, overlap scored; DER0% = collar 0, overlap scored. "
          "fidDER% = C# segments vs NeMo probabilities through the same binarization, collar 0.")
    if args.max_fidelity_der is not None and worst_fid > args.max_fidelity_der:
        print(f"FAIL: worst fidelity DER {worst_fid:.4f} > {args.max_fidelity_der}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
