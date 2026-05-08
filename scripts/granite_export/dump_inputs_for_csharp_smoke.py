#!/usr/bin/env python3
"""
Dump pre-computed input tensors for the C# Granite Speech smoke test.

The C# CLI parity smoke (under tests/GraniteSpeechSmoke/) wants to drive
the exported ONNX bundle end-to-end without first re-implementing
torchaudio's MelSpectrogram + Granite's frame-stacking + the BPE prompt
encoder in C#. This helper runs the upstream `GraniteSpeechProcessor`
once on a fixed audio clip and writes the resulting tensors as raw
binary files that the C# side just memcopies in.

The mel + tokenizer ports become a follow-up (tracked in the
investigation doc Run 4 deferred items). For now the C# smoke proves
the ORT-side wiring (encoder + projector + decoder_init + decoder_step
loop + ByteLevel BPE decode) matches Python end-to-end.

Outputs (in --output-dir):
    input_features.bin      float32 [T_stacked, 160]
    input_ids.bin           int64   [prompt_len]
    expected_tokens.bin     int64   [N_generated]  (greedy continuation)
    expected_text.txt       UTF-8 transcript from `model.generate(...)`
    shape.json              {"T_stacked": ..., "prompt_len": ..., ...}

Usage
-----
    source .venv-granite-export/bin/activate
    python public/scripts/granite_export/dump_inputs_for_csharp_smoke.py \\
        --audio /path/to/clip.wav \\
        --output-dir tests/GraniteSpeechSmoke/Fixtures \\
        --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-repo", default="ibm-granite/granite-speech-4.1-2b")
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--prompt",
        default="transcribe the speech with proper punctuation and capitalization.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--include-features",
        action="store_true",
        help=(
            "Also write input_features.bin and attention_mask.bin. The smoke uses "
            "mel.onnx instead, so these are reference-only and skipped by default."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_repo, revision=args.revision)
    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    full_prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|audio|>{args.prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    proc_in = processor(full_prompt, audio, return_tensors="pt")
    input_features = proc_in["input_features"].cpu().numpy().astype(np.float32)
    input_ids = proc_in["input_ids"].cpu().numpy().astype(np.int64)
    attention_mask = proc_in["attention_mask"].cpu().numpy().astype(np.int64)

    # Drop the batch dim for the C# fixtures (smoke runs B=1).
    if input_features.shape[0] != 1:
        raise SystemExit(f"Expected B=1, got {input_features.shape[0]}")
    input_features_b1 = input_features[0]  # [T_stacked, 160]
    input_ids_b1 = input_ids[0]              # [prompt_len]
    attention_mask_b1 = attention_mask[0]    # [prompt_len]

    # Run the reference model so the C# side has a golden token stream.
    print("Loading model for golden continuation ...")
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        args.model_repo,
        revision=args.revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    proc_in_pt = {k: v for k, v in proc_in.items() if hasattr(v, "to")}
    with torch.no_grad():
        out = model.generate(
            **proc_in_pt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    generated = out[0, input_ids.shape[1]:].cpu().numpy().astype(np.int64)
    expected_text = processor.tokenizer.decode(generated.tolist(), skip_special_tokens=True)

    # Write outputs. The C# smoke needs only input_ids.bin (the prompt
    # tokenisation) and expected_text.txt (the golden continuation).
    # input_features.bin / attention_mask.bin are reference-only and gated
    # behind --include-features.
    (args.output_dir / "input_ids.bin").write_bytes(input_ids_b1.tobytes())
    (args.output_dir / "expected_tokens.bin").write_bytes(generated.tobytes())
    (args.output_dir / "expected_text.txt").write_text(expected_text, encoding="utf-8")
    if args.include_features:
        (args.output_dir / "input_features.bin").write_bytes(input_features_b1.tobytes())
        (args.output_dir / "attention_mask.bin").write_bytes(attention_mask_b1.tobytes())

    shape = {
        "audio_path": str(args.audio),
        "audio_seconds": float(len(audio) / sr),
        "T_stacked": int(input_features_b1.shape[0]),
        "feature_dim": int(input_features_b1.shape[1]),
        "prompt_len": int(input_ids_b1.shape[0]),
        "n_generated_with_eos": int(len(generated)),
        "expected_text": expected_text,
        "max_new_tokens": args.max_new_tokens,
        "model_repo": args.model_repo,
        "revision": args.revision,
        "prompt": args.prompt,
    }
    (args.output_dir / "shape.json").write_text(json.dumps(shape, indent=2))

    print(f"Wrote fixtures to {args.output_dir}")
    print(f"  input_features.bin: {input_features_b1.shape} float32")
    print(f"  input_ids.bin:      {input_ids_b1.shape} int64")
    print(f"  expected_tokens.bin {generated.shape} int64")
    print(f"  expected_text:      {expected_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
