#!/usr/bin/env python3
"""Rare-primitive validation: does the IPA adapter render sounds English LACKS —
the phones the 24-lang corpus was balanced for? Base vs fine-tuned, natural duration,
on held-out dev utterances that contain each distinctive primitive.

Not full-sentence intelligibility (unjudgeable for these langs) but presence/absence of
the characteristic sound (click / ejective / prenasal / tone), audible + spectrally checkable.
"""
import json, os
import numpy as np
import soundfile as sf
import torch
from omnivoice.models.omnivoice import OmniVoice
from scripts.omnivoice_ipa.gen_accept_test import decode_codes, gen, BASE, ROOT, SR

OUT = f"{ROOT}/train/rare_test_v2"
ADAPTER = f"{ROOT}/train/checkpoints_v2/checkpoint-4000"

# (lang, ref_id, target_id, label) — v2 focus: Sindhi implosives (incl newly-covered ɠ),
# Hausa retest, Amharic ejectives (new lang), Zulu clicks (no-regression check).
TESTS = [
    ("sd_in", "10374408950024345416", "10051174211934897563", "implosives_g"),
    ("ha_ng", "10314810220142301751", "10102412684314580007", "implosive_ejective"),
    ("am_et", "10117353659503784748", "10372921613758145905", "ejectives"),
    ("zu_za", "10386821834488770056", "10120271725404688148", "clicks_regression"),
]


def load_case(lang, ref_id, target_id):
    codes = np.load(f"{ROOT}/corpus/tokens/codes_{lang}.npz")
    ipa = {json.loads(l)["id"]: json.loads(l)["text"]
           for l in open(f"{ROOT}/train/shards/{lang}/dev.jsonl")}
    ref_wav = decode_codes(codes[ref_id])
    return ref_wav, ipa[ref_id], ipa[target_id], decode_codes(codes[target_id])


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = []
    for lang, ref_id, target_id, label in TESTS:
        ref_wav, ref_ipa, tgt_ipa, tgt_gt = load_case(lang, ref_id, target_id)
        sf.write(f"{OUT}/{lang}_{label}_ref.wav", ref_wav, SR)
        sf.write(f"{OUT}/{lang}_{label}_groundtruth.wav", tgt_gt, SR)
        cases.append((lang, label, ref_wav, ref_ipa, tgt_ipa))
        print(f"{lang} [{label}]: tgt={tgt_ipa[:70]}")

    print("\nloading base ...")
    model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()
    for lang, label, ref_wav, ref_ipa, tgt_ipa in cases:
        print(f"  base gen {lang}/{label} ...")
        w = gen(model, tgt_ipa, ref_wav, ref_ipa, duration=None)
        sf.write(f"{OUT}/{lang}_{label}_base.wav", w, SR)

    print("merging adapter ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
    for lang, label, ref_wav, ref_ipa, tgt_ipa in cases:
        print(f"  finetuned gen {lang}/{label} ...")
        w = gen(model, tgt_ipa, ref_wav, ref_ipa, duration=None)
        sf.write(f"{OUT}/{lang}_{label}_finetuned.wav", w, SR)

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
