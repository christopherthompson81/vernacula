#!/usr/bin/env python3
"""Export hexgrad/Kokoro-82M (StyleTTS2-based TTS) to ONNX.

There is no official conversion script shipped with the onnx-community v1.0 ONNX
release, so we build our own to control the opset, I/O contract, and quantization.

Export target is `KModel.forward_with_tokens(input_ids, ref_s, speed)` — the
token-level entry point, so the G2P frontend (misaki) stays outside the graph and
lives in C# / Python at inference time.

The STFT layer is the known landmine: the default complex-valued STFT does not
trace to ONNX cleanly and yields subtly wrong audio. `KModel(disable_complex=True)`
swaps in a real-valued STFT that exports correctly. We always validate the exported
graph against the PyTorch reference before declaring success.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class KokoroONNX(torch.nn.Module):
    """Thin wrapper exposing a single, ONNX-friendly forward.

    Inputs:
        input_ids : LongTensor [1, T]  — padded token ids ([0, *ids, 0])
        ref_s     : FloatTensor [1, S] — style/voice vector (S = 256)
        speed     : FloatTensor [1]    — speech-rate multiplier

    Output:
        audio     : FloatTensor [num_samples] — 24 kHz waveform
        pred_dur  : LongTensor  [tokens]      — per-input-token predicted duration
                    (frames). Used downstream for word-level alignment: word time =
                    cumulative pred_dur × (len(audio) / sum(pred_dur)) / sample_rate.
    """

    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids, ref_s, speed):
        audio, pred_dur = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return audio, pred_dur


# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------
# WARNING: do NOT validate with random token ids / random ref_s. Random inputs drive
# the duration predictor + prosody LSTM off-distribution, where tiny numerical diffs
# compound into large *waveform* divergence that has nothing to do with export quality
# (random-input parity measured corr 0.76; real-input parity measured corr 0.997 on the
# same graph). Always capture the real (input_ids, ref_s, speed) a normal pipeline run
# feeds the model. See docs/kokoro_onnx_investigation.md, Runs 1 & 3.

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog, and then it pauses to think."
DEFAULT_VOICE = "af_heart"


def capture_real_inputs(kmodel, repo_id, text=DEFAULT_TEXT, voice=DEFAULT_VOICE):
    """Run a real KPipeline and capture the exact tensors it feeds the model.

    Returns (input_ids[1,T] int64, ref_s[1,256] float32, speed[1] float32). These are
    in-distribution — the only meaningful inputs for tracing and parity validation.
    """
    from kokoro import KPipeline

    cap = {}
    orig = kmodel.forward_with_tokens

    def spy(input_ids, ref_s, speed=1):
        if "input_ids" not in cap:
            cap["input_ids"] = input_ids.detach().clone()
            cap["ref_s"] = ref_s.detach().clone()
            cap["speed"] = torch.tensor([float(speed)], dtype=torch.float32)
        return orig(input_ids, ref_s, speed)

    kmodel.forward_with_tokens = spy
    try:
        pipe = KPipeline(lang_code=voice[0], repo_id=repo_id, model=kmodel)
        for _ in pipe(text, voice=voice):
            break
    finally:
        kmodel.forward_with_tokens = orig  # restore

    if "input_ids" not in cap:
        raise RuntimeError("pipeline produced no segments; could not capture inputs")
    return cap["input_ids"], cap["ref_s"], cap["speed"]


def log_spectral_l1(a: np.ndarray, b: np.ndarray, n_fft: int = 1024, hop: int = 256) -> float:
    """Phase-invariant log-magnitude spectral L1 distance.

    The right acceptance metric for a GAN vocoder: iSTFTNet's output phase is not
    uniquely determined, so waveform SNR / correlation flags inaudible phase
    differences as huge errors. Log-magnitude spectra ignore phase. For reference, one
    STFT frame of time jitter ≈ 0.77 on this scale. See investigation Runs 4 & 5.
    """
    win = torch.hann_window(n_fft)
    def logmag(x):
        t = torch.tensor(np.asarray(x), dtype=torch.float32)
        return torch.log(torch.stft(t, n_fft, hop, window=win, return_complex=True).abs() + 1e-5)
    la, lb = logmag(a), logmag(b)
    n = min(la.shape[-1], lb.shape[-1])
    return float((la[..., :n] - lb[..., :n]).abs().mean())


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(out_dir: Path, opset: int, repo_id: str, dynamo: bool) -> Path:
    from kokoro import KModel

    print(f"[export] loading {repo_id} (disable_complex=True)…")
    kmodel = KModel(repo_id=repo_id, disable_complex=True).eval()
    wrapper = KokoroONNX(kmodel).eval()

    # Trace with real, in-distribution inputs (not random ids) so the duration
    # predictor / prosody LSTM trace in a sane regime.
    print("[export] capturing real pipeline inputs for tracing…")
    input_ids, ref_s, speed = capture_real_inputs(kmodel, repo_id)
    print(f"[export] traced shapes: input_ids={tuple(input_ids.shape)} ref_s={tuple(ref_s.shape)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "kokoro.onnx"

    print(f"[export] tracing → {onnx_path} (opset {opset}, dynamo={dynamo})…")
    if dynamo:
        # torch.export-based exporter — handles packed-LSTM / control flow far better
        # than the legacy TorchScript path. dynamic_shapes uses named Dim objects.
        tokens = torch.export.Dim("tokens", min=2)
        torch.onnx.export(
            wrapper,
            (input_ids, ref_s, speed),
            str(onnx_path),
            input_names=["input_ids", "ref_s", "speed"],
            output_names=["audio", "pred_dur"],
            dynamic_shapes={
                "input_ids": {1: tokens},
                "ref_s": None,
                "speed": None,
            },
            opset_version=opset,
            dynamo=True,
        )
    else:
        torch.onnx.export(
            wrapper,
            (input_ids, ref_s, speed),
            str(onnx_path),
            input_names=["input_ids", "ref_s", "speed"],
            output_names=["audio", "pred_dur"],
            dynamic_axes={
                "input_ids": {1: "tokens"},
                "audio": {0: "samples"},
                "pred_dur": {0: "tokens"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # legacy TorchScript exporter
        )
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(onnx_path: Path, repo_id: str, max_logspec_l1: float = 0.25) -> bool:
    """Compare ONNX vs the PyTorch (disable_complex) reference on REAL inputs.

    Metric is phase-invariant log-spectral L1, not waveform SNR — see log_spectral_l1
    and investigation Runs 3–5 for why waveform-domain metrics are misleading for this
    vocoder. Default threshold 0.25 sits well below one frame of jitter (~0.77) and
    above the observed export residual (~0.20).
    """
    import onnxruntime as ort
    from kokoro import KModel

    print("[validate] loading PyTorch reference…")
    kmodel = KModel(repo_id=repo_id, disable_complex=True).eval()
    wrapper = KokoroONNX(kmodel).eval()
    input_ids, ref_s, speed = capture_real_inputs(kmodel, repo_id)

    with torch.no_grad():
        ref_audio = wrapper(input_ids, ref_s, speed)[0].squeeze().cpu().numpy()

    print("[validate] running onnxruntime…")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_audio = sess.run(
        ["audio"],
        {
            "input_ids": input_ids.numpy(),
            "ref_s": ref_s.numpy(),
            "speed": speed.numpy(),
        },
    )[0].squeeze()

    if ref_audio.shape != onnx_audio.shape:
        print(f"[validate] FAIL: shape mismatch torch={ref_audio.shape} onnx={onnx_audio.shape}")
        return False

    dist = log_spectral_l1(ref_audio, onnx_audio)
    print(f"[validate] log-spec L1={dist:.4f}  (threshold {max_logspec_l1:.2f})  samples={onnx_audio.shape}")

    ok = dist < max_logspec_l1
    print(f"[validate] {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[validate] hint: real STFT divergence in the export (not a phase artifact).")
    return ok


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Export Kokoro-82M to ONNX")
    p.add_argument("--out", type=Path, default=Path("external/kokoro_onnx"),
                   help="output directory for kokoro.onnx")
    p.add_argument("--repo-id", default="hexgrad/Kokoro-82M",
                   help="HuggingFace repo id for weights/config")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-logspec-l1", type=float, default=0.25,
                   help="max log-spectral L1 distance allowed in validation "
                        "(phase-invariant; ~0.77 = one frame of jitter)")
    p.add_argument("--dynamo", action="store_true",
                   help="use the torch.export-based exporter (currently FAILS for this "
                        "model: data-dependent guard in the BERT attention mask — see "
                        "investigation Run 2). Default is the legacy TorchScript exporter.")
    p.add_argument("--skip-validate", action="store_true")
    args = p.parse_args(argv)

    onnx_path = export(args.out, args.opset, args.repo_id, args.dynamo)

    if args.skip_validate:
        print("[main] validation skipped")
        return 0

    return 0 if validate(onnx_path, args.repo_id, args.max_logspec_l1) else 1


if __name__ == "__main__":
    sys.exit(main())
