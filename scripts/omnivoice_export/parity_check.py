#!/usr/bin/env python3
"""Validate the OmniVoice ONNX graphs against captured PyTorch reference tensors.

Metrics chosen per component (see docs/kokoro_onnx_investigation.md for why waveform
SNR is the wrong tool for a GAN-trained codec):

  - transformer : argmax-token agreement rate + logit max-abs / MSE. What matters is
                  that the diffusion loop picks the same tokens, so argmax agreement is
                  the headline number.
  - encoder     : exact integer code-match rate (codes are discrete indices).
  - decoder     : log-spectral L1 (phase-invariant). DAC is GAN-trained -> output phase
                  is not unique, so raw waveform SNR/correlation flags inaudible phase
                  differences as huge errors. Waveform corr is printed FYI only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _providers(provider: str):
    """Provider list for a parity/perf trade-off.

    Phase-1 parity is validated in FULL fp32 (TF32 disabled). ORT's CUDA EP defaults to
    TF32 matmul, whose ~1e-2 logit error compounds through the 32-step diffusion loop into
    audibly different (though still valid) speech. cpu / cuda-fp32 are mathematically
    faithful; cuda-tf32 is the fast path to revisit in the performance phase.
    """
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda-tf32":
        return [("CUDAExecutionProvider", {"use_tf32": 1}), "CPUExecutionProvider"]
    # cuda-fp32 (faithful): TF32 off
    return [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]


def _session(path: Path, provider: str = "cpu") -> ort.InferenceSession:
    try:
        return ort.InferenceSession(str(path), providers=_providers(provider))
    except Exception:  # noqa: BLE001
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _log_spectral_l1(a: np.ndarray, b: np.ndarray, n_fft: int = 1024,
                     hop: int = 256) -> float:
    """Mean L1 distance between log-magnitude STFTs. Phase-invariant."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win = np.hanning(n_fft).astype(np.float64)

    def logmag(x):
        frames = []
        for i in range(0, len(x) - n_fft + 1, hop):
            frames.append(np.abs(np.fft.rfft(x[i:i + n_fft] * win)))
        S = np.stack(frames, axis=0) if frames else np.zeros((1, n_fft // 2 + 1))
        return np.log(S + 1e-7)

    return float(np.mean(np.abs(logmag(a) - logmag(b))))


def check_transformer(onnx_dir: Path, cap, provider: str) -> bool:
    path = onnx_dir / "omnivoice_transformer.onnx"
    if not path.exists():
        print("[transformer] skipped (no onnx)")
        return True
    sess = _session(path, provider)
    feeds = {
        "input_ids": cap["tf_input_ids"].astype(np.int64),
        "audio_mask": cap["tf_audio_mask"].astype(np.bool_),
        "attention_mask": cap["tf_attention_mask"].astype(np.bool_),
    }
    onnx_logits = sess.run(["logits"], feeds)[0]
    ref = cap["tf_logits"]
    agree = float(np.mean(onnx_logits.argmax(-1) == ref.argmax(-1)))
    max_abs = float(np.max(np.abs(onnx_logits - ref)))
    mse = float(np.mean((onnx_logits - ref) ** 2))
    ok = agree > 0.9999 and max_abs < 5e-3
    print(f"[transformer] argmax-agreement={agree:.5f} max_abs={max_abs:.3e} "
          f"mse={mse:.3e} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_encoder(onnx_dir: Path, cap, provider: str) -> bool:
    path = onnx_dir / "higgs_encoder.onnx"
    if not path.exists():
        print("[encoder] skipped (no onnx)")
        return True
    sess = _session(path, provider)
    onnx_codes = sess.run(["audio_codes"], {"input_values": cap["enc_input_values"]})[0]
    ref = cap["enc_audio_codes"]
    n = min(onnx_codes.shape[-1], ref.shape[-1])
    match = float(np.mean(onnx_codes[..., :n] == ref[..., :n]))
    ok = match > 0.99 and onnx_codes.shape[-1] == ref.shape[-1]
    print(f"[encoder] code-match={match:.5f} shape_onnx={onnx_codes.shape} "
          f"shape_ref={ref.shape} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_decoder(onnx_dir: Path, cap, provider: str) -> bool:
    path = onnx_dir / "higgs_decoder.onnx"
    if not path.exists():
        print("[decoder] skipped (no onnx)")
        return True
    sess = _session(path, provider)
    onnx_wav = sess.run(["audio_values"], {"audio_codes": cap["dec_audio_codes"]})[0]
    ref = cap["dec_audio_values"]
    lsl1 = _log_spectral_l1(onnx_wav, ref)
    a, b = onnx_wav.reshape(-1), ref.reshape(-1)
    n = min(len(a), len(b))
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 1 else float("nan")
    ok = lsl1 < 0.2
    print(f"[decoder] log-spectral-L1={lsl1:.4f} (waveform-corr={corr:.4f} FYI) "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-dir", default=str(Path(__file__).resolve().parent / "onnx"))
    p.add_argument(
        "--capture", default=str(Path(__file__).resolve().parent / "capture/reference.npz")
    )
    p.add_argument(
        "--provider", default="cpu", choices=["cpu", "cuda-fp32", "cuda-tf32"],
        help="cpu / cuda-fp32 are faithful (full fp32); cuda-tf32 is the fast path "
        "(TF32 matmul, ~1e-2 logit error that compounds in the diffusion loop).",
    )
    args = p.parse_args()

    cap = np.load(args.capture)
    onnx_dir = Path(args.onnx_dir)
    print(f"provider={args.provider}")

    results = [
        check_transformer(onnx_dir, cap, args.provider),
        check_encoder(onnx_dir, cap, args.provider),
        check_decoder(onnx_dir, cap, args.provider),
    ]
    print("\nOVERALL:", "PASS" if all(results) else "FAIL")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
