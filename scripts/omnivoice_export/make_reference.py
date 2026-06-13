#!/usr/bin/env python3
"""Build a *matched* voice-clone reference for the parity harness.

Voice cloning conditions on (ref_audio_tokens, ref_text) and expects them to correspond;
a mismatched transcript derails the diffusion (quiet / unintelligible output). Rather than
hunt for a real clip with a known transcript, we synthesise one with the model itself in
voice-design mode for a fixed sentence — so ref_text provably matches the audio — and save
it for capture_reference.py / infer_onnx.py to clone from.

Output: capture/ref_voice.wav + capture/ref_voice.txt (the transcript).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--text",
        default="This is a reference voice sample for testing the export pipeline.",
    )
    p.add_argument("--instruct", default="female, american accent")
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "capture"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"Loading {args.model} ({args.device}) ...")
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=torch.float32).eval()

    print("Synthesising reference voice (design mode, model defaults) ...")
    audio = model.generate(text=args.text, language=args.language, instruct=args.instruct)[0]

    wav_path = out_dir / "ref_voice.wav"
    txt_path = out_dir / "ref_voice.txt"
    sf.write(str(wav_path), audio, model.sampling_rate)
    txt_path.write_text(args.text.strip() + "\n")
    print(f"Saved reference -> {wav_path} ({len(audio)/model.sampling_rate:.2f}s)")
    print(f"Saved transcript -> {txt_path}: {args.text!r}")


if __name__ == "__main__":
    main()
