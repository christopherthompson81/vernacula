#!/usr/bin/env python3
"""End-to-end OmniVoice inference with the ONNX graphs as drop-in replacements.

This is the rigorous proof that the graph split is correct: we load the real PyTorch
pipeline (only for host orchestration — tokenizer, RuleDurationEstimator, the diffusion
masking schedule, CFG + top-k/gumbel scoring, post-processing) and replace ONLY the three
neural callables with onnxruntime sessions:

    model.forward            -> omnivoice_transformer.onnx
    audio_tokenizer.encode   -> higgs_encoder.onnx
    audio_tokenizer.decode   -> higgs_decoder.onnx

If the resulting WAV matches the captured PyTorch reference (log-spectral L1, and by ear),
the exports are faithful. The host pieces wrapped here are exactly what the future C#
runtime must reimplement; this file documents that boundary.

Run capture_reference.py with the SAME args first so the comparison is apples-to-apples
(deterministic: position/class temperature = 0, fixed seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig


def _providers(provider: str):
    """Faithful full-fp32 by default (cpu / cuda-fp32). ORT's CUDA TF32 matmul (cuda-tf32)
    introduces ~1e-2 logit error that compounds through the 32-step diffusion loop into
    audibly different speech — a performance-phase concern, not a parity path."""
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda-tf32":
        return [("CUDAExecutionProvider", {"use_tf32": 1}), "CPUExecutionProvider"]
    return [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]


def _session(path: Path, provider: str = "cpu") -> ort.InferenceSession:
    try:
        return ort.InferenceSession(str(path), providers=_providers(provider))
    except Exception:  # noqa: BLE001
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _log_spectral_l1(a, b, n_fft=1024, hop=256):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win = np.hanning(n_fft)

    def lm(x):
        fr = [np.abs(np.fft.rfft(x[i:i + n_fft] * win))
              for i in range(0, len(x) - n_fft + 1, hop)]
        S = np.stack(fr) if fr else np.zeros((1, n_fft // 2 + 1))
        return np.log(S + 1e-7)

    return float(np.mean(np.abs(lm(a) - lm(b))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--provider", default="cpu", choices=["cpu", "cuda-fp32", "cuda-tf32"],
        help="ONNX execution provider. Default cpu (faithful). cuda-tf32 diverges.",
    )
    p.add_argument("--onnx-dir", default=str(Path(__file__).resolve().parent / "onnx"))
    p.add_argument(
        "--ref-audio",
        default=str(Path(__file__).resolve().parent / "capture/ref_voice.wav"),
    )
    p.add_argument(
        "--ref-text", default=None,
        help="Transcript of --ref-audio; if omitted, read from the sidecar .txt / "
        "capture/ref_voice.txt. MUST match the audio.",
    )
    p.add_argument(
        "--text",
        default="Hello, this is a test of the OmniVoice O N N X export pipeline.",
    )
    p.add_argument("--language", default="English")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "capture/onnx_e2e.wav"))
    p.add_argument(
        "--compare",
        default=str(Path(__file__).resolve().parent / "capture/reference.npz"),
        help="Captured PyTorch reference .npz to compare against (log-spectral L1).",
    )
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)

    ref_text = args.ref_text
    if ref_text is None:
        sidecar = Path(args.ref_audio).with_suffix(".txt")
        fallback = Path(__file__).resolve().parent / "capture/ref_voice.txt"
        src = sidecar if sidecar.exists() else fallback
        if not src.exists():
            raise SystemExit(
                f"No --ref-text and no transcript at {sidecar} or {fallback}. "
                "Run make_reference.py first."
            )
        ref_text = src.read_text().strip()
    args.ref_text = ref_text

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    print(f"Loading {args.model} (host orchestration only) ...")
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=torch.float32)
    model.eval()

    tf_sess = _session(onnx_dir / "omnivoice_transformer.onnx", args.provider)
    enc_sess = _session(onnx_dir / "higgs_encoder.onnx", args.provider)
    dec_sess = _session(onnx_dir / "higgs_decoder.onnx", args.provider)
    print(f"ONNX provider={args.provider}")

    tok = model.audio_tokenizer
    tok_dev = tok.device

    # ---- replace transformer forward -----------------------------------------
    def onnx_forward(input_ids, audio_mask, attention_mask, **_):
        logits = tf_sess.run(
            ["logits"],
            {
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
                "audio_mask": audio_mask.detach().cpu().numpy().astype(np.bool_),
                "attention_mask": attention_mask.detach().cpu().numpy().astype(np.bool_),
            },
        )[0]
        return SimpleNamespace(logits=torch.from_numpy(logits).to(dev))

    # ---- replace codec encode / decode ---------------------------------------
    def onnx_encode(input_values, *a, **k):
        codes = enc_sess.run(
            ["audio_codes"], {"input_values": input_values.detach().cpu().numpy().astype(np.float32)}
        )[0]
        return SimpleNamespace(audio_codes=torch.from_numpy(codes).to(tok_dev))

    def onnx_decode(audio_codes, *a, **k):
        wav = dec_sess.run(
            ["audio_values"], {"audio_codes": audio_codes.detach().cpu().numpy().astype(np.int64)}
        )[0]
        return SimpleNamespace(audio_values=torch.from_numpy(wav).to(tok_dev))

    model.forward = onnx_forward  # type: ignore[method-assign]
    tok.encode = onnx_encode  # type: ignore[method-assign]
    tok.decode = onnx_decode  # type: ignore[method-assign]

    gen_config = OmniVoiceGenerationConfig(
        num_step=args.num_step, guidance_scale=args.guidance_scale,
        position_temperature=0.0, class_temperature=0.0,
    )

    print("Running ONNX-backed generate() ...")
    audios = model.generate(
        text=args.text, language=args.language,
        ref_audio=args.ref_audio, ref_text=args.ref_text,
        generation_config=gen_config,
    )
    audio = audios[0]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, model.sampling_rate)
    print(f"Saved ONNX e2e WAV -> {args.out}")

    if args.compare and Path(args.compare).exists():
        ref = np.load(args.compare)["final_audio"]
        lsl1 = _log_spectral_l1(audio, ref)
        print(f"\nEnd-to-end vs PyTorch reference: log-spectral-L1={lsl1:.4f} "
              f"({'OK' if lsl1 < 0.2 else 'CHECK'}; listen to confirm)")


if __name__ == "__main__":
    main()
