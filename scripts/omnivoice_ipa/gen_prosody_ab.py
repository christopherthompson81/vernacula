#!/usr/bin/env python3
"""v5 vs v6 on PAUSE PLACEMENT — the one axis every other instrument is blind to.

⚠ WHY THIS EXISTS. The v6 corpus differs from v5's almost entirely in prosody: restoring the FLEURS raw
column put punctuation back into the text the IPA is derived from, and 88.2% of rows changed by pause
marks alone (7.7% changed segmentally). Three instruments have now failed to see it, exactly as
predicted in vernacula-phonemizer#873 BEFORE the change was applied:

    recognizer distance   67 closer / 56 further, mean delta -0.00014
    training loss         v5 and v6 both flat around 3.9
    eval loss             final 3.9658 vs 3.9777, inside v5's own +/-0.04 wobble

None of them can be: `notate(units(...))` strips pause marks, and next-token loss over 8 audio
codebooks weighted [8,8,6,6,4,4,2,2] barely moves for a shifted silence. So the verdict has to come
from generated audio, measured against the human reading.

⚠ EACH MODEL IS FED ITS OWN TRAINING DISTRIBUTION, which is the only fair comparison. v5 never saw a
pause token, so handing it v6's comma-bearing IPA tests an out-of-distribution input, not the change we
made. v5 gets the pre-restoration IPA from `corpus/tokens_v5_backup/manifest_<lang>.jsonl`, v6 gets the
current one. What is being compared is the whole pipeline change, not a token ablation.

    v5 ... vˈɛɹi lˈoᶷ pɹˈɛʃɚ ˈɛɹ   wˌɪt͡ʃ sˈʌks ...
    v6 ... vˈɛɹi lˈoᶷ pɹˈɛʃɚ ˈɛɹ , wˌɪt͡ʃ sˈʌks ...      "...pressure air, which sucks..."

⚠ GROUND TRUTH IS THE HUMAN, decoded from that utterance's own codes. The question is not "did v6
pause more" — a model that pauses everywhere would win that. It is "did v6 pause WHERE THE READER DID".

⚠ MATCHED STEPS. Both runs kept checkpoints 2000/3000/4000, so the controlled pair is v5@4000 vs
v6@4000 — same step count, same 28-language coverage set, same weights, same hyperparameters, corpus
the only variable. v6@2000 is generated too, as a check that any difference is not an artifact of one
particular step.

  python3 gen_prosody_ab.py --lang en_us --n 6
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

ROOT = "/mnt/data/omnivoice_ipa"
ONNX = "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx"
BASE = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice"
OUT = f"{ROOT}/train/prosody_ab"
SR = 24000


def decode_codes(dec, codes):
    rec = dec.run(["audio_values"], {"audio_codes": codes[None].astype(np.int64)})[0]
    return np.asarray(rec).reshape(-1).astype(np.float32)


def pauses(wav, sr=SR, thresh_db=-38.0, min_ms=90):
    """Interior silence midpoints as a fraction of total duration.

    ⚠ INTERIOR ONLY — leading and trailing silence is recording margin, not phrasing, and including it
    would let a model score well by simply starting late.
    """
    win = int(sr * 0.02)
    if len(wav) < win * 3:
        return []
    n = len(wav) // win
    frames = wav[:n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms / (np.abs(wav).max() + 1e-9))
    quiet = db < thresh_db
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 20 >= min_ms and i > 0 and j < n:      # interior only
                out.append(((i + j) / 2) * win / len(wav))
            i = j
        else:
            i += 1
    return out


def pause_score(gen_p, ref_p, tol=0.06):
    """Matched pauses / union — 1.0 means the same phrasing, 0.0 means none in common."""
    if not gen_p and not ref_p:
        return 1.0
    used, hit = set(), 0
    for g in gen_p:
        for k, r in enumerate(ref_p):
            if k not in used and abs(g - r) <= tol:
                used.add(k); hit += 1; break
    union = len(gen_p) + len(ref_p) - hit
    return hit / union if union else 1.0


def gen(model, text, ref_wav, ref_text):
    out = model.generate(text=text, language=None,
                         ref_audio=(torch.from_numpy(ref_wav).float(), SR),
                         ref_text=ref_text, num_step=32, guidance_scale=2.0, denoise=False)
    return np.asarray(out[0]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en_us")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--v5", default=f"{ROOT}/train/checkpoints_v5/checkpoint-4000")
    ap.add_argument("--v6", default=f"{ROOT}/train/checkpoints_v6/checkpoint-4000")
    ap.add_argument("--v6_mid", default=f"{ROOT}/train/checkpoints_v6/checkpoint-2000")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    v5_ipa = {json.loads(l)["id"]: json.loads(l)["ipa"]
              for l in open(f"{ROOT}/corpus/tokens_v5_backup/manifest_{a.lang}.jsonl", encoding="utf8")}
    v6_ipa = {json.loads(l)["id"]: json.loads(l)["ipa"]
              for l in open(f"{ROOT}/corpus/tokens/manifest_{a.lang}.jsonl", encoding="utf8")}
    both = [i for i in v6_ipa if i in v5_ipa]
    # ⚠ PUNCTUATION-DENSE ONLY. If the sentence has no interior pause mark the two inputs are identical
    #   and the comparison is vacuous — it would dilute the result with rows that cannot differ.
    cand = [i for i in both if v6_ipa[i].count(",") >= 1 and v5_ipa[i].count(",") == 0]
    cand.sort(key=lambda i: -v6_ipa[i].count(","))
    picks = cand[:a.n]
    ref_id = next(i for i in both if i not in picks)
    print(f"{len(cand)} punctuation-dense candidates in {a.lang}; using {len(picks)}")

    codes = np.load(f"{ROOT}/corpus/tokens/codes_{a.lang}.npz")
    dec = ort.InferenceSession(f"{ONNX}/higgs_decoder.onnx", providers=["CPUExecutionProvider"])
    ref_wav = decode_codes(dec, codes[ref_id])
    truth = {i: decode_codes(dec, codes[i]) for i in picks}
    for i in picks:
        sf.write(f"{OUT}/{i}_human.wav", truth[i], SR)

    from peft import PeftModel
    from omnivoice.models.omnivoice import OmniVoice
    results = {}
    for tag, adapter, ipa_src in (("v5@4000", a.v5, v5_ipa),
                                  ("v6@4000", a.v6, v6_ipa),
                                  ("v6@2000", a.v6_mid, v6_ipa)):
        print(f"loading {tag} ...", flush=True)
        model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        scores = []
        for i in picks:
            w = gen(model, ipa_src[i], ref_wav, ipa_src[ref_id])
            sf.write(f"{OUT}/{i}_{tag.replace('@','_')}.wav", w, SR)
            s = pause_score(pauses(w), pauses(truth[i]))
            scores.append(s)
            print(f"   {i}: pause-match {s:.2f}  ({len(pauses(w))} gen vs {len(pauses(truth[i]))} human)",
                  flush=True)
        results[tag] = scores
        del model
        torch.cuda.empty_cache()

    print(f"\n{'model':10}{'mean pause-match':>18}{'per-utterance':>34}")
    for tag, s in results.items():
        print(f"{tag:10}{sum(s)/len(s):18.3f}   {' '.join(f'{x:.2f}' for x in s)}")
    print(f"\nwavs in {OUT} — listen; the score is a proxy, not the verdict")


if __name__ == "__main__":
    main()
