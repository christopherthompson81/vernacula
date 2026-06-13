#!/usr/bin/env python3
"""Stage-C fixtures: duration-estimator outputs and _prepare_inference_inputs token rows.

- duration_cases: RuleDurationEstimator.estimate_duration for (target, ref, ref_tokens),
  plus the int() target-token count, across scripts — validates the C# port's formula.
- prepare_cases: cond_input_ids row 0 + audio_mask from _prepare_inference_inputs for
  no-reference configs (design/auto mode), with an explicit num_target — validates the C#
  style/text token construction and audio-mask layout. (The ref-audio path is validated
  end-to-end in Stage D.)

Output: tests/fixtures/omnivoice_textprep_fixtures.json (committed; small).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.duration import RuleDurationEstimator

DURATION_CASES = [
    ("Hello, this is a test of the speech system.", "Nice to meet you.", 25),
    ("Short.", "Nice to meet you.", 25),
    ("你好世界，这是一个测试。", "你好。", 10),
    ("Mixed 中文 and English 123.", "Reference text here.", 40),
    ("A much longer sentence that should take quite a while to say aloud properly.",
     "Brief reference.", 30),
    ("12345 67890", "Numbers here.", 20),
]

# (text, lang, instruct, denoise, num_target)  -- ref_text/ref_audio = None (no-ref mode)
PREPARE_CASES = [
    ("Hello world.", "en", None, True, 50),
    ("Hello world.", None, None, True, 12),
    ("Testing voice design.", "en", "female, american accent", True, 64),
    ("你好世界。", "zh", None, False, 20),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data/models/omnivoice/k2-fsa-OmniVoice")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[2]
                   / "tests/fixtures/omnivoice_textprep_fixtures.json"))
    args = p.parse_args()

    est = RuleDurationEstimator()
    duration_cases = []
    for target, ref, ref_tokens in DURATION_CASES:
        d = est.estimate_duration(target, ref, ref_tokens)
        duration_cases.append({
            "target": target, "ref": ref, "ref_tokens": ref_tokens,
            "estimate": d, "target_tokens": max(1, int(d)),
        })

    model = OmniVoice.from_pretrained(args.model, device_map="cpu", dtype=torch.float32).eval()
    prepare_cases = []
    for text, lang, instruct, denoise, num_target in PREPARE_CASES:
        inp = model._prepare_inference_inputs(
            text=text, num_target_tokens=num_target, ref_text=None,
            ref_audio_tokens=None, lang=lang, instruct=instruct, denoise=denoise,
        )
        row0 = inp["input_ids"][0, 0, :].tolist()
        amask = inp["audio_mask"][0].to(torch.int32).tolist()
        prepare_cases.append({
            "text": text, "lang": lang, "instruct": instruct, "denoise": denoise,
            "num_target": num_target, "row0_ids": row0, "audio_mask": amask,
        })

    out = {"duration_cases": duration_cases, "prepare_cases": prepare_cases}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(duration_cases)} duration + {len(prepare_cases)} prepare cases -> {out_path}")


if __name__ == "__main__":
    main()
