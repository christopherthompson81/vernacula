#!/usr/bin/env python3
"""Generation acceptance test for the IPA LoRA fine-tune — the REAL verdict that eval
loss can't give (loss only weakly reflects text conditioning; see diag_ipa_conditioning).

Voice-clone TTS from a held-out reference clip, driven by held-out target IPA:
  - ref_audio  = dev utterance A's audio (decoded from its codes)
  - ref_text   = dev utterance A's IPA  (IPA, matching the IPA-only training distribution)
  - text       = dev utterance B's IPA  (the target to render)
Generates with BASE (no adapter) and FINE-TUNED (checkpoint merged) for A/B comparison,
plus B's ground-truth audio (decoded from its real codes) as the target reference.

Outputs wavs to train/gen_test/ for listening.
"""
import argparse, json, os
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from omnivoice.models.omnivoice import OmniVoice

ROOT = "/mnt/data/omnivoice_ipa"
ONNX = "/mnt/data/Programming/vernacula/scripts/omnivoice_export/onnx"
BASE = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice"
OUT = f"{ROOT}/train/gen_test"
SR = 24000


def decode_codes(codes):
    dec = ort.InferenceSession(f"{ONNX}/higgs_decoder.onnx", providers=["CPUExecutionProvider"])
    rec = dec.run(["audio_values"], {"audio_codes": codes[None].astype(np.int64)})[0]
    return np.asarray(rec).reshape(-1).astype(np.float32)


# ⚠ A REFERENCE IS BOUNDED AT BOTH ENDS, and picking "the longest" satisfies neither. A 2 s clip
# carries too little speaker evidence; past 20 s the model itself warns that cloning degrades — and
# because that penalty lands on EVERY checkpoint equally, it flattens the very comparison the
# reference exists to support. Choosing the longest dev utterance produced a 29.6 s clip in
# gen_rare_test; --ref_lang zu_za would have produced 23.2 s in the offglide test.
REF_MIN_S, REF_MAX_S = 4.0, 15.0


def durations(lang: str) -> dict[str, float]:
    """{id: dur_s} from the ingest manifest — dev.jsonl carries only id and text."""
    return {r["id"]: r["dur_s"] for r in
            (json.loads(l) for l in open(f"{ROOT}/corpus/tokens/manifest_{lang}.jsonl", encoding="utf8")
             if l.strip())}


def pick_reference(rows: list[dict], dur: dict[str, float], exclude_id: str | None = None) -> dict:
    """Longest dev utterance WITHIN the usable band; nearest to it if nothing lands inside."""
    cands = [r for r in rows if r["id"] != exclude_id and r["id"] in dur]
    inside = [r for r in cands if REF_MIN_S <= dur[r["id"]] <= REF_MAX_S]
    return (max(inside, key=lambda r: dur[r["id"]]) if inside
            else min(cands, key=lambda r: abs(dur[r["id"]] - REF_MAX_S)))


def gen(model, text, ref_wav, ref_text):
    # NO duration forcing (user decision, Run 30): forcing ground-truth clip length handed the
    # model the human's pause time + lead/trail silence as speech budget, which it filled with
    # drag and misplaced pauses. The model's own estimate is imperfect on IPA but the pacing it
    # produces is its honest pacing — that's what the acceptance test should judge.
    out = model.generate(
        text=text, language=None,
        ref_audio=(torch.from_numpy(ref_wav).float(), SR),
        ref_text=ref_text,
        num_step=32, guidance_scale=2.0, denoise=False,
    )
    return np.asarray(out[0]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en_us")
    ap.add_argument("--ref_id", default="903")
    ap.add_argument("--target_id", default="279")
    # ⚠ NOT `checkpoints/checkpoint-3000`. That default pointed at the July run — older than v5, v6
    # and v7 — so an acceptance test invoked without --adapter silently judged a model nobody was
    # shipping. Same stale-pin bug already fixed in extract_diff.py, apply_diff.py and publish_hf.py;
    # a default that is never bumped is how it comes back. This tracks the CURRENT fine-tune.
    ap.add_argument("--adapter", default=f"{ROOT}/train/checkpoints_v7/checkpoint-6000")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    codes = np.load(f"{ROOT}/corpus/tokens/codes_{a.lang}.npz")
    ipa = {json.loads(l)["id"]: json.loads(l)["text"]
           for l in open(f"{ROOT}/train/shards/{a.lang}/dev.jsonl")}
    ref_ipa, tgt_ipa = ipa[a.ref_id], ipa[a.target_id]
    print(f"REF  ({a.ref_id}): {ref_ipa[:80]}")
    print(f"TGT  ({a.target_id}): {tgt_ipa[:80]}")

    ref_wav = decode_codes(codes[a.ref_id])
    sf.write(f"{OUT}/ref_{a.ref_id}.wav", ref_wav, SR)
    tgt_gt = decode_codes(codes[a.target_id])
    sf.write(f"{OUT}/target_groundtruth_{a.target_id}.wav", tgt_gt, SR)
    # base first (clean weights), then merge the adapter in and regenerate
    print("loading base ...")
    model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()
    print("generating (base) ...")
    sf.write(f"{OUT}/gen_base_{a.target_id}.wav", gen(model, tgt_ipa, ref_wav, ref_ipa), SR)

    print("applying LoRA adapter + merge ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload().eval()
    print("generating (fine-tuned) ...")
    sf.write(f"{OUT}/gen_finetuned_{a.target_id}.wav", gen(model, tgt_ipa, ref_wav, ref_ipa), SR)

    print(f"\n-> wavs in {OUT}")


if __name__ == "__main__":
    main()
