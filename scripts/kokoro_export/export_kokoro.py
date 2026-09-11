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

from batched_kokoro import FRAME


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
def export_batched(out_dir: Path, opset: int, repo_id: str) -> Path:
    """Export the variable-length BATCHED graph (see batched_kokoro.py).

    Replaces, rather than accompanies, the batch=1 graph: at B=1 it measures 1.08x faster than
    kokoro.onnx, and at B=8 it is 2.27x faster, so there is no reason to ship both.
    Investigation Runs 26-36.
    """
    from kokoro import KModel
    from batched_kokoro import BatchedKokoroONNX, install_masking

    print(f"[export] loading {repo_id} (disable_complex=True)…")
    kmodel = KModel(repo_id=repo_id, disable_complex=True).eval()
    norms, hooks = install_masking(kmodel)
    print(f"[export] fidelity masking: {norms} masked norms, {hooks} re-zero hooks")
    wrapper = BatchedKokoroONNX(kmodel).eval()

    # Trace with TWO real items of DIFFERENT length, so the padding path is actually traced.
    print("[export] capturing two real, unequal-length inputs for tracing…")
    ids_a, ref_a, _ = capture_real_inputs(kmodel, repo_id, text=DEFAULT_TEXT)
    ids_b, ref_b, _ = capture_real_inputs(
        kmodel, repo_id,
        text="The quarterly figures were not included in the summary, and nobody said why.")
    lens = [ids_a.shape[1], ids_b.shape[1]]
    max_t = max(lens)
    input_ids = torch.zeros((2, max_t), dtype=torch.long)
    input_ids[0, :lens[0]] = ids_a[0]
    input_ids[1, :lens[1]] = ids_b[0]
    ref_s = torch.cat([ref_a, ref_b], dim=0)
    input_lengths = torch.tensor(lens, dtype=torch.long)
    speed = torch.tensor([1.0])
    print(f"[export] traced shapes: input_ids={tuple(input_ids.shape)} lengths={lens}")

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "kokoro_batched.onnx"
    print(f"[export] tracing → {onnx_path} (opset {opset})…")
    torch.onnx.export(
        wrapper,
        (input_ids, ref_s, speed, input_lengths),
        str(onnx_path),
        input_names=["input_ids", "ref_s", "speed", "input_lengths"],
        output_names=["audio", "pred_dur"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "tokens"},
            "ref_s": {0: "batch"},
            "input_lengths": {0: "batch"},
            "audio": {0: "batch", 1: "samples"},
            "pred_dur": {0: "batch", 1: "tokens"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def validate_batched(onnx_path: Path, repo_id: str, max_logspec_l1: float = 0.25) -> bool:
    """Batched-specific acceptance, which is stricter than the batch=1 one in the way that
    matters: durations must be IDENTICAL to the solo render regardless of batch composition
    (they drive word alignment), and audio must land at the model's own run-to-run floor.

    ⚠ Kokoro is nondeterministic — SourceModuleHnNSF draws noise every call, so two solo
    renders of the same input already differ by ~0.13 log-spec. The floor is measured here
    rather than assumed, and the batched result is judged against it.
    """
    import onnxruntime as ort
    from kokoro import KModel

    kmodel = KModel(repo_id=repo_id, disable_complex=True).eval()
    texts = [
        DEFAULT_TEXT,
        "The deadline moved again.",
        "In the quiet hours before dawn the old keeper climbed the spiral stair "
        "and lit the great lamp, and the light swung out across the water.",
    ]
    caps = [capture_real_inputs(kmodel, repo_id, text=t) for t in texts]
    with torch.no_grad():
        solo = [kmodel.forward_with_tokens(i, r, 1.0) for i, r, _ in caps]
        solo2 = [kmodel.forward_with_tokens(i, r, 1.0)[0].squeeze().numpy() for i, r, _ in caps]
    ref_audio = [a.squeeze().numpy() for a, _ in solo]
    ref_dur = [d.numpy() for _, d in solo]
    floor = [log_spectral_l1(ref_audio[i], solo2[i]) for i in range(len(caps))]
    print(f"[validate] model's own run-to-run floor: "
          + " ".join(f"{f:.3f}" for f in floor))

    lens = [c[0].shape[1] for c in caps]
    max_t = max(lens)
    input_ids = np.zeros((len(caps), max_t), dtype=np.int64)
    for j, c in enumerate(caps):
        input_ids[j, :lens[j]] = c[0][0].numpy()
    ref_s = np.concatenate([c[1].numpy() for c in caps], 0).astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    audio, pred_dur = sess.run(None, {
        "input_ids": input_ids,
        "ref_s": ref_s,
        "speed": np.array([1.0], dtype=np.float32),
        "input_lengths": np.array(lens, dtype=np.int64),
    })

    ok = True
    for j in range(len(caps)):
        dur_ok = np.array_equal(pred_dur[j, :lens[j]], ref_dur[j])
        n = int(pred_dur[j, :lens[j]].sum())
        dist = log_spectral_l1(ref_audio[j], audio[j, :n * FRAME])
        pad = 100 * (1 - n / max(int(pred_dur[k, :lens[k]].sum()) for k in range(len(caps))))
        good = dur_ok and dist <= max(max_logspec_l1, floor[j] * 1.3)
        ok &= good
        print(f"[validate] item{j} pad={pad:3.0f}%  durations={'match' if dur_ok else 'DIFFER'}  "
              f"logspec={dist:.4f} (floor {floor[j]:.3f})  {'PASS' if good else 'FAIL'}")
    if not ok:
        print("[validate] hint: durations differing with batch composition means a "
              "bidirectional LSTM is reading padding — check the pack_padded_sequence calls.")
    print(f"[validate] {'PASS' if ok else 'FAIL'}")
    return ok


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
    p.add_argument("--batched", action="store_true",
                   help="export the variable-length batched graph (kokoro_batched.onnx) "
                        "instead of the batch=1 one. Faster at every batch size including "
                        "B=1; see batched_kokoro.py and investigation Runs 26-36.")
    args = p.parse_args(argv)

    if args.batched:
        onnx_path = export_batched(args.out, args.opset, args.repo_id)
        if args.skip_validate:
            print("[main] validation skipped")
            return 0
        return 0 if validate_batched(onnx_path, args.repo_id, args.max_logspec_l1) else 1

    onnx_path = export(args.out, args.opset, args.repo_id, args.dynamo)

    if args.skip_validate:
        print("[main] validation skipped")
        return 0

    return 0 if validate(onnx_path, args.repo_id, args.max_logspec_l1) else 1


if __name__ == "__main__":
    sys.exit(main())
