#!/usr/bin/env python3
"""Control: does BASE OmniVoice render Zulu clicks from conventional ORTHOGRAPHY?

OmniVoice natively supports 646 langs incl. 'zu'. If base+orthography produces clicks
(which base+IPA does NOT), it proves the click capability is already in the pretrained
model and our IPA fine-tune re-routed that existing capability to IPA input — the
"input adapter, not new sounds" thesis. Same sentence/voice as the rare-primitive test.
"""
import numpy as np, soundfile as sf, torch
from omnivoice.models.omnivoice import OmniVoice
from scripts.omnivoice_ipa.gen_accept_test import decode_codes, gen, BASE, ROOT, SR

OUT = f"{ROOT}/train/rare_test"
# sentence id 600 (target) and 1138 (ref) — conventional Zulu orthography
TGT_ORTHO = "ayebukeka njengamakamelo wayengumuntu wokuqala wokuhlola amangqamuzana afile"
REF_ORTHO = "ngaphambili inhlangano yezindaba yaseshayina i-xinhua yabika ukuthi kunendiza ethunjiwe"

codes = np.load(f"{ROOT}/corpus/tokens/codes_zu_za.npz")
ref_wav = decode_codes(codes["1138"])

model = OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=torch.float16).eval()
print("base + Zulu orthography, language='zu' ...")
out = model.generate(text=TGT_ORTHO, language="zu",
                     ref_audio=(torch.from_numpy(ref_wav).float(), SR),
                     ref_text=REF_ORTHO, duration=None,
                     num_step=32, guidance_scale=2.0, denoise=False)
sf.write(f"{OUT}/zu_za_clicks_base_ORTHO.wav", np.asarray(out[0]).reshape(-1), SR)
print(f"-> {OUT}/zu_za_clicks_base_ORTHO.wav")
