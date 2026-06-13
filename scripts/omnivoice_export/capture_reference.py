#!/usr/bin/env python3
"""Capture in-distribution reference tensors from a real OmniVoice PyTorch run.

We never validate ONNX exports with random ids / random codes — random inputs push
the diffusion transformer and codec off-distribution and turn tiny numerical diffs
into large, misleading divergences (the hard-won lesson from the Kokoro export; see
docs/kokoro_onnx_investigation.md). Instead we wrap the three neural modules, run a
normal ``model.generate()``, and dump the *actual* tensors that flowed through:

  - transformer : (input_ids, audio_mask, attention_mask) -> logits   [step 0]
  - higgs encoder: input_values (ref wav)                  -> audio_codes
  - higgs decoder: audio_codes (final)                     -> audio_values

Generation is made deterministic (greedy: position/class temperature = 0, fixed seed)
so the end-to-end ONNX harness can be compared frame-for-frame against this run.

Output: an .npz next to this script (default capture/reference.npz) plus the reference
WAV (capture/py_reference.wav).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--ref-audio",
        default=str(Path(__file__).resolve().parent / "capture/ref_voice.wav"),
        help="Reference WAV for voice cloning. Default is the matched reference built by "
        "make_reference.py (run that first). ref_text MUST match the audio or the "
        "diffusion derails into quiet/unintelligible output.",
    )
    p.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of --ref-audio. If omitted, read from the sidecar "
        "<ref-audio>.txt / capture/ref_voice.txt written by make_reference.py.",
    )
    p.add_argument(
        "--text",
        default="Hello, this is a test of the OmniVoice O N N X export pipeline.",
    )
    p.add_argument("--language", default="English")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "capture")
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve ref_text: explicit arg, else sidecar <ref-audio>.txt, else capture/ref_voice.txt.
    ref_text = args.ref_text
    if ref_text is None:
        sidecar = Path(args.ref_audio).with_suffix(".txt")
        fallback = out_dir / "ref_voice.txt"
        src = sidecar if sidecar.exists() else fallback
        if not src.exists():
            raise SystemExit(
                f"No --ref-text and no transcript at {sidecar} or {fallback}. "
                "Run make_reference.py first (it builds a matched reference)."
            )
        ref_text = src.read_text().strip()
        print(f"Using ref_text from {src}: {ref_text!r}")
    args.ref_text = ref_text

    torch.manual_seed(args.seed)

    print(f"Loading {args.model} (fp32) on {args.device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=args.device, dtype=torch.float32
    )
    model.eval()

    cap: dict[str, np.ndarray] = {}

    # ---- wrap transformer forward (capture step 0 only) ----------------------
    orig_forward = model.forward

    def forward_hook(*fa, **fkw):
        out = orig_forward(*fa, **fkw)
        if "tf_input_ids" not in cap:
            cap["tf_input_ids"] = _to_np(fkw["input_ids"]).astype(np.int64)
            cap["tf_audio_mask"] = _to_np(fkw["audio_mask"]).astype(np.bool_)
            cap["tf_attention_mask"] = _to_np(fkw["attention_mask"]).astype(np.bool_)
            cap["tf_logits"] = _to_np(out.logits).astype(np.float32)
        return out

    model.forward = forward_hook  # type: ignore[method-assign]

    # ---- wrap codec encode / decode (capture first call each) ----------------
    tok = model.audio_tokenizer
    orig_encode, orig_decode = tok.encode, tok.decode

    def encode_hook(input_values, *ea, **ekw):
        out = orig_encode(input_values, *ea, **ekw)
        if "enc_input_values" not in cap:
            cap["enc_input_values"] = _to_np(input_values).astype(np.float32)
            cap["enc_audio_codes"] = _to_np(out.audio_codes).astype(np.int64)
        return out

    def decode_hook(audio_codes, *da, **dkw):
        out = orig_decode(audio_codes, *da, **dkw)
        if "dec_audio_codes" not in cap:
            cap["dec_audio_codes"] = _to_np(audio_codes).astype(np.int64)
            cap["dec_audio_values"] = _to_np(out.audio_values).astype(np.float32)
        return out

    tok.encode = encode_hook  # type: ignore[method-assign]
    tok.decode = decode_hook  # type: ignore[method-assign]

    gen_config = OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        position_temperature=0.0,  # determinism: no gumbel on position selection
        class_temperature=0.0,  # determinism: greedy token choice
    )

    print("Running deterministic generate() ...")
    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        generation_config=gen_config,
    )
    audio = audios[0]

    ref_wav_path = out_dir / "py_reference.wav"
    sf.write(str(ref_wav_path), audio, model.sampling_rate)
    cap["final_audio"] = audio.astype(np.float32)
    cap["sampling_rate"] = np.int64(model.sampling_rate)

    # metadata for reproducibility
    meta = dict(
        text=args.text,
        ref_text=args.ref_text,
        ref_audio=args.ref_audio,
        language=args.language,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    npz_path = out_dir / "reference.npz"
    np.savez_compressed(npz_path, **cap)
    (out_dir / "reference_meta.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in meta.items()) + "\n"
    )

    print(f"\nSaved capture -> {npz_path}")
    for k, v in cap.items():
        shp = getattr(v, "shape", None)
        print(f"  {k:20s} shape={shp} dtype={getattr(v,'dtype',type(v))}")
    print(f"Saved reference WAV -> {ref_wav_path}")


if __name__ == "__main__":
    main()
