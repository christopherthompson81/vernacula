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
ONNX = "/home/chris/Programming/vernacula/scripts/omnivoice_export/onnx"
BASE = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice"
OUT = f"{ROOT}/train/gen_test"
SR = 24000


def decode_codes(codes):
    dec = ort.InferenceSession(f"{ONNX}/higgs_decoder.onnx", providers=["CPUExecutionProvider"])
    rec = dec.run(["audio_values"], {"audio_codes": codes[None].astype(np.int64)})[0]
    return np.asarray(rec).reshape(-1).astype(np.float32)


def gen(model, text, ref_wav, ref_text, duration=None):
    # duration passed explicitly: RuleDurationEstimator is orthographic-calibrated and
    # badly under-estimates IPA (stress/tie/length marks), truncating output. Passing the
    # known target duration isolates IPA-rendering quality from the duration-estimil issue.
    out = model.generate(
        text=text, language=None,
        ref_audio=(torch.from_numpy(ref_wav).float(), SR),
        ref_text=ref_text, duration=duration,
        num_step=32, guidance_scale=2.0, denoise=False,
    )
    return np.asarray(out[0]).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en_us")
    ap.add_argument("--ref_id", default="903")
    ap.add_argument("--target_id", default="279")
    ap.add_argument("--adapter", default=f"{ROOT}/train/checkpoints/checkpoint-3000")
    ap.add_argument("--natural", action="store_true",
                    help="no duration correction (let generate() estimate); files suffixed _natural")
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
    # duration=None (natural) lets generate() estimate; else pass the ground-truth length.
    # NOTE that GT length can include leading/trailing silence, so it's not a clean target.
    dur = None if a.natural else len(tgt_gt) / SR
    sfx = "_natural" if a.natural else ""
    print(f"duration mode: {'estimate (natural)' if a.natural else f'{dur:.2f}s from GT'}")

    # base first (clean weights), then merge the adapter in and regenerate
    print("loading base ...")
    model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()
    print("generating (base) ...")
    sf.write(f"{OUT}/gen_base_{a.target_id}{sfx}.wav", gen(model, tgt_ipa, ref_wav, ref_ipa, duration=dur), SR)

    print("applying LoRA adapter + merge ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload().eval()
    print("generating (fine-tuned) ...")
    sf.write(f"{OUT}/gen_finetuned_{a.target_id}{sfx}.wav", gen(model, tgt_ipa, ref_wav, ref_ipa, duration=dur), SR)

    print(f"\n-> wavs in {OUT}")


if __name__ == "__main__":
    main()
