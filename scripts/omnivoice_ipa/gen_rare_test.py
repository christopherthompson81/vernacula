#!/usr/bin/env python3
"""Rare-primitive regression test: does the adapter still render the sounds English LACKS?

The corpus is balanced so specific languages OWN specific primitives — Zulu the clicks (ǀ ǁ ǃ) and
breathy voice, Hausa and Amharic the ejectives, Sindhi the implosives. A fine-tune that improves
English can quietly cost those, and nothing in the loss curve would say so. This renders held-out dev
utterances that contain each primitive, base vs fine-tuned, for a listening and spectral check.

Not full-sentence intelligibility (unjudgeable for these languages) but presence or absence of the
characteristic sound, which is audible.

⚠ UTTERANCES ARE SELECTED BY PRIMITIVE CONTENT, NOT BY HARDCODED ID. The previous version pinned
eight dev ids; grouping the train/dev split by `sentence_id` (so the splits are text-disjoint)
reshuffled every dev set and FIVE OF THE EIGHT ids no longer existed. A test that names its fixtures
by id silently stops testing what it claims the moment the split changes -- and this one had also
gone stale against `gen()`, which lost its `duration` parameter when duration forcing was dropped,
so it raised TypeError before it got as far as the missing ids.

  python3 -m scripts.omnivoice_ipa.gen_rare_test --adapter .../checkpoints_v7/checkpoint-4000
"""
import argparse, json, os

import numpy as np
import soundfile as sf
import torch
from omnivoice.models.omnivoice import OmniVoice

from scripts.omnivoice_ipa.gen_accept_test import BASE, ROOT, SR, decode_codes, gen

# (lang, label, the primitives that language is the greedy-cover OWNER of)
TESTS = [
    ("sd_in", "implosives",         ["ɓ", "ɗ", "ʄ", "ɠ"]),
    ("ha_ng", "implosive_ejective", ["ɓ", "ɗ", "kʼ", "sʼ"]),
    ("am_et", "ejectives",          ["pʼ", "tʼ", "kʼ", "sʼ", "t͡ʃʼ"]),
    ("zu_za", "clicks",             ["ǀ", "ǁ", "ǃ", "ɮ", "̤"]),
]


# The model warns above 20 s and recommends 3-10 s. Both ends are real: a 2 s clip carries too little
# speaker evidence, and a 30 s one degrades cloning -- and because the penalty applies to BOTH
# checkpoints it flattens the very comparison the reference is there to support.
REF_MIN_S, REF_MAX_S = 4.0, 15.0


def pick(lang: str, prims: list[str], dur: dict[str, float]) -> tuple[str, str, str, str]:
    """(ref_id, ref_ipa, tgt_id, tgt_ipa) — richest dev utterance for the target, best-sized for the ref.

    ⚠ NOT simply the longest. Choosing the longest dev utterance to avoid a thin reference produced
    29.6 s clips and the model's own ">20s degrades voice cloning" warning. Longest WITHIN the usable
    band, falling back to whatever is closest to it if nothing lands inside.
    """
    rows = [json.loads(l) for l in open(f"{ROOT}/train/shards/{lang}/dev.jsonl") if l.strip()]
    score = lambda r: sum(r["text"].count(p) for p in prims)
    tgt = max(rows, key=score)
    cands = [r for r in rows if r["id"] != tgt["id"] and r["id"] in dur]
    ok = [r for r in cands if REF_MIN_S <= dur[r["id"]] <= REF_MAX_S]
    ref = (max(ok, key=lambda r: dur[r["id"]]) if ok
           else min(cands, key=lambda r: abs(dur[r["id"]] - REF_MAX_S)))
    return ref["id"], ref["text"], tgt["id"], tgt["text"]


def durations(lang: str) -> dict[str, float]:
    return {r["id"]: r["dur_s"] for r in
            (json.loads(l) for l in open(f"{ROOT}/corpus/tokens/manifest_{lang}.jsonl", encoding="utf8")
             if l.strip())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    # ⚠ THE BASELINE FOR A REGRESSION IS THE PREVIOUS FINE-TUNE, NOT THE BASE MODEL. Base has never
    # been IPA-tuned, so base-vs-new measures "fine-tuning works" and would hide a loss against the
    # run actually being replaced. Same mistake cost a wrong answer on the en-GB probes (Run 57).
    ap.add_argument("--baseline", default=None,
                    help="adapter to compare against (default: unadapted base weights)")
    ap.add_argument("--out", default=f"{ROOT}/train/rare_test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cases = []
    for lang, label, prims in TESTS:
        ref_id, ref_ipa, tgt_id, tgt_ipa = pick(lang, prims, durations(lang))
        codes = np.load(f"{ROOT}/corpus/tokens/codes_{lang}.npz")
        ref_wav = decode_codes(codes[ref_id])
        sf.write(f"{a.out}/{lang}_{label}_ref.wav", ref_wav, SR)
        sf.write(f"{a.out}/{lang}_{label}_groundtruth.wav", decode_codes(codes[tgt_id]), SR)
        hits = {p: tgt_ipa.count(p) for p in prims if p in tgt_ipa}
        cases.append((lang, label, ref_wav, ref_ipa, tgt_ipa))
        print(f"{lang} [{label}] ref {len(ref_wav)/SR:.1f}s, target {tgt_id}: {hits}")
        if not hits:
            print(f"  ⚠ no target primitive present in any {lang} dev utterance — this case proves nothing")

    # Base FIRST: merge_and_unload() is destructive, so one process cannot return to base weights.
    base_tag = "v6" if a.baseline else "base"
    print(f"\nloading {base_tag} ...")
    model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()
    if a.baseline:
        from peft import PeftModel as _P
        model = _P.from_pretrained(model, a.baseline).merge_and_unload().eval()
    for lang, label, ref_wav, ref_ipa, tgt_ipa in cases:
        sf.write(f"{a.out}/{lang}_{label}_{base_tag}.wav", gen(model, tgt_ipa, ref_wav, ref_ipa), SR)
        print(f"  {base_tag} {lang}/{label}")
    # merge_and_unload() is destructive, so the second model needs clean base weights.
    del model
    torch.cuda.empty_cache()
    model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()

    print("merging adapter ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    for lang, label, ref_wav, ref_ipa, tgt_ipa in cases:
        sf.write(f"{a.out}/{lang}_{label}_finetuned.wav", gen(model, tgt_ipa, ref_wav, ref_ipa), SR)
        print(f"  finetuned {lang}/{label}")

    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
