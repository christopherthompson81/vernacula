#!/usr/bin/env python3
"""Length sweep to close the codec shape-branch question (follow-up #3).

The legacy exporter froze a few data-dependent branches in the Higgs codec into constants
(notably the encoder's semantic÷320 vs acoustic÷960 stream-alignment check). The export was
traced at one length and validated at two; this sweeps many hop-multiple lengths to confirm
the frozen branches stay correct.

For each length we compare ONNX-CPU against torch-CPU on the SAME device (so any mismatch is
a graph bug, not RVQ device sensitivity):
  - encoder: exact integer code-match  (input wav [1,1,L], L a multiple of hop=960)
  - decoder: round-trip waveform max-abs on the encoder's codes

Real inputs are constrained to hop multiples by create_voice_clone_prompt, so we sweep that
grid (plus a couple of long clips). Exits non-zero if any length diverges.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice

HOP = 960  # higgs hop_length (acoustic downsample 8*5*4*2*3)


def _build_wav(base: np.ndarray, L: int) -> np.ndarray:
    if L <= len(base):
        w = base[:L]
    else:
        w = np.tile(base, int(np.ceil(L / len(base))))[:L]
    return w.astype(np.float32)[None, None, :]  # [1,1,L]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument("--onnx-dir", default=str(Path(__file__).resolve().parent / "onnx"))
    p.add_argument("--ref", default=str(Path(__file__).resolve().parent / "capture/ref_voice.wav"))
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)
    base, sr = sf.read(args.ref)
    base = np.asarray(base, dtype=np.float32)
    if base.ndim > 1:
        base = base.mean(axis=1)

    # hop-multiple k values: dense at the low end (where semantic/acoustic alignment is most
    # likely to flip), plus a spread up to ~25s. 58 and 82 were the previously-tested points.
    ks = sorted(set(list(range(5, 21)) + [25, 30, 40, 58, 64, 82, 100, 128, 200,
                                          256, 312, 400, 500, 625]))
    lengths = [k * HOP for k in ks]

    print(f"Loading {args.model} (cpu, fp32) ...")
    model = OmniVoice.from_pretrained(args.model, device_map="cpu", dtype=torch.float32).eval()
    tok = model.audio_tokenizer

    enc = ort.InferenceSession(str(onnx_dir / "higgs_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(onnx_dir / "higgs_decoder.onnx"), providers=["CPUExecutionProvider"])

    print(f"{'k':>4} {'len':>8} {'dur_s':>6} | {'codes_T':>7} {'code_match':>10} {'dec_maxabs':>10}  result")
    fails = []
    for k, L in zip(ks, lengths):
        wav = _build_wav(base, L)
        with torch.inference_mode():
            codes_t = tok.encode(torch.from_numpy(wav)).audio_codes.cpu().numpy()
        codes_o = enc.run(["audio_codes"], {"input_values": wav})[0]

        shape_ok = codes_o.shape == codes_t.shape
        n = min(codes_o.shape[-1], codes_t.shape[-1])
        match = float(np.mean(codes_o[..., :n] == codes_t[..., :n])) if n else 0.0

        # decoder round-trip on torch codes (same input both sides -> should be ~exact)
        with torch.inference_mode():
            wav_t = tok.decode(torch.from_numpy(codes_t)).audio_values.cpu().numpy()
        wav_o = dec.run(["audio_values"], {"audio_codes": codes_t})[0]
        m = min(wav_t.shape[-1], wav_o.shape[-1])
        dec_maxabs = float(np.max(np.abs(wav_t[..., :m] - wav_o[..., :m]))) if m else float("nan")
        dec_shape_ok = wav_t.shape == wav_o.shape

        ok = shape_ok and match == 1.0 and dec_shape_ok and dec_maxabs < 1e-3
        if not ok:
            fails.append((k, shape_ok, match, dec_shape_ok, dec_maxabs))
        print(f"{k:>4} {L:>8} {L/sr:>6.2f} | {codes_t.shape[-1]:>7} {match:>10.5f} "
              f"{dec_maxabs:>10.2e}  {'OK' if ok else 'FAIL'}"
              + ("" if shape_ok and dec_shape_ok else f"  shape enc{codes_o.shape}->ref{codes_t.shape} dec{wav_o.shape}->ref{wav_t.shape}"))

    print()
    if fails:
        print(f"SWEEP FAIL — {len(fails)} length(s) diverged: {[f[0] for f in fails]}")
        raise SystemExit(1)
    print(f"SWEEP PASS — all {len(ks)} hop-multiple lengths "
          f"({min(ks)*HOP/sr:.2f}s..{max(ks)*HOP/sr:.2f}s) match torch exactly.")


if __name__ == "__main__":
    main()
