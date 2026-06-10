#!/usr/bin/env python3
"""Parity sweep for the Kokoro ONNX export across voices and input lengths.

Validates that ONNX matches the PyTorch (disable_complex) reference not just on one
utterance but across a spread of style vectors (voices) and token lengths — including
the long-input edge near the model's context_length ceiling. Metric is phase-invariant
log-spectral L1 (see export_kokoro.log_spectral_l1 for why, and the investigation doc).

Only English voices are swept: misaki's English G2P is installed; other languages need
extra backends. Language is irrelevant to the ONNX graph anyway — what varies that the
graph cares about is ref_s and token count.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_kokoro import KokoroONNX, capture_real_inputs, log_spectral_l1

# Spread across accent (a=American, b=British) and gender (f/m). Each voice is a
# distinct ref_s style vector — a voice-pack indexing bug hides on a single voice.
VOICES = [
    "af_heart", "af_bella", "af_nicole",
    "am_michael", "am_fenrir",
    "bf_emma", "bm_george", "bm_lewis",
]

# Length sweep. The long text stays under the 512-token context_length ceiling but
# exercises the long end where pred_dur / STFT framing accumulates. Keep each as a
# SINGLE segment so the pipeline doesn't chunk it (capture grabs the first segment).
TEXTS = {
    "short": "Hello there.",
    "medium": "The quick brown fox jumps over the lazy dog.",
    "long": (
        "In the quiet hours before dawn, the old lighthouse keeper climbed the "
        "spiral stair, lit the great lamp, and watched its slow beam sweep across "
        "the restless grey water far below the cliffs."
    ),
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Kokoro ONNX parity sweep")
    p.add_argument("--onnx", type=Path, default=Path("external/kokoro_onnx/kokoro.onnx"))
    p.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="max log-spectral L1 before a cell is flagged FAIL")
    args = p.parse_args(argv)

    from kokoro import KModel

    print(f"[sweep] loading {args.repo_id} (disable_complex=True)…")
    kmodel = KModel(repo_id=args.repo_id, disable_complex=True).eval()
    wrapper = KokoroONNX(kmodel).eval()
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    rows = []
    worst = 0.0
    n_fail = 0
    for voice in VOICES:
        for length, text in TEXTS.items():
            try:
                ids, ref_s, speed = capture_real_inputs(kmodel, args.repo_id, text=text, voice=voice)
            except Exception as e:
                print(f"[sweep] {voice:11s} {length:6s} capture FAILED: {type(e).__name__}: {e}")
                rows.append((voice, length, None, None, "ERR"))
                n_fail += 1
                continue

            with torch.no_grad():
                ref = wrapper(ids, ref_s, speed).squeeze().cpu().numpy()
            onx = sess.run(["audio"], {
                "input_ids": ids.numpy(), "ref_s": ref_s.numpy(), "speed": speed.numpy(),
            })[0].squeeze()

            n = min(len(ref), len(onx))
            dist = log_spectral_l1(ref[:n], onx[:n])
            ok = dist < args.threshold and ref.shape == onx.shape
            worst = max(worst, dist)
            n_fail += (not ok)
            tok = int(ids.shape[1])
            rows.append((voice, length, tok, dist, "PASS" if ok else "FAIL"))
            print(f"[sweep] {voice:11s} {length:6s} tok={tok:3d} "
                  f"len_torch={len(ref):6d} len_onnx={len(onx):6d} logL1={dist:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")

    print("\n=== summary ===")
    print(f"cells={len(rows)} fail={n_fail} worst_logL1={worst:.4f} threshold={args.threshold:.2f}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
