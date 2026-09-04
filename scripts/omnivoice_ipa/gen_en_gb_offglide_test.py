#!/usr/bin/env python3
"""Did v7 learn the en-GB offglide pairings? The targeted acceptance test.

`gen_accept_test.py` renders held-out dev utterances chosen by id, which is the right general
instrument but will not reliably exercise the vowels the en_gb data was added for. This renders
PROBE SENTENCES built around those vowels, including the three failures that started the work:
"smoke" heard as "smik", "show" heard as "shoe", and "televisi'epots".

⚠ THE CONTROL IS THE POINT. Each probe is rendered from BOTH its en-GB and its en-US phonemization
(`en_gb_probes.json`, written by the companion .mts). The pair differs exactly where the accents do:

    smoke   GB smˈəᶷk      US smˈoᶷk
    share   GB ʃˈɛə        US ʃˈɛɹ
    fire    GB fˈaᶦə       US fˈaᶦɚ

The base model has seen `oᶷ` tens of thousands of times and `əᶷ` almost never, so it should render
the US line well and the GB line badly. If the fine-tune has worked, the GB line comes up to meet it
WITHOUT the US line regressing. A GB improvement bought by a US regression is not a fix, and only
rendering the GB side would hide that.

  python3 gen_en_gb_offglide_test.py --adapter .../checkpoints_v7/checkpoint-4000
  python3 gen_en_gb_offglide_test.py --adapter ... --ref_lang en_us   # cross-accent reference
"""
import argparse, json, os

import numpy as np
import soundfile as sf
import torch
from omnivoice.models.omnivoice import OmniVoice

from scripts.omnivoice_ipa.gen_accept_test import BASE, ROOT, SR, decode_codes, gen

PROBES = f"{ROOT}/train/en_gb_probes.json"


def pick_ref(lang: str, ref_id: str | None) -> tuple[str, str]:
    """(id, ipa) of the dev utterance used as the voice-clone reference.

    Default is the LONGEST dev utterance: a reference clip is the model's only evidence of the
    speaker, and the shortest ones in a dev split are a couple of seconds, which clones badly and
    would show up as a difference between checkpoints that is really a difference in reference.
    """
    rows = [json.loads(l) for l in open(f"{ROOT}/train/shards/{lang}/dev.jsonl") if l.strip()]
    by_id = {r["id"]: r["text"] for r in rows}
    if ref_id:
        return ref_id, by_id[ref_id]
    best = max(rows, key=lambda r: len(r["text"]))
    return best["id"], best["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint dir to merge")
    ap.add_argument("--ref_lang", default="en_gb")
    ap.add_argument("--ref_id", default=None)
    ap.add_argument("--out", default=f"{ROOT}/train/en_gb_offglide_test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    probes = json.load(open(PROBES, encoding="utf8"))
    ref_id, ref_ipa = pick_ref(a.ref_lang, a.ref_id)
    codes = np.load(f"{ROOT}/corpus/tokens/codes_{a.ref_lang}.npz")
    ref_wav = decode_codes(codes[ref_id])
    sf.write(f"{a.out}/_reference_{a.ref_lang}_{ref_id}.wav", ref_wav, SR)
    print(f"reference {a.ref_lang}/{ref_id} ({len(ref_wav)/SR:.1f}s): {ref_ipa[:70]}")

    # Base FIRST on clean weights, then merge -- merge_and_unload() is destructive, so a single
    # process cannot go back to base afterwards.
    for tag, build in (("base", lambda: OmniVoice.from_pretrained(BASE, device_map="cuda",
                                                                  dtype=torch.float16).eval()),
                       ("v7", None)):
        if tag == "base":
            model = build()
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
        print(f"--- {tag}")
        for name, p in probes.items():
            for accent in ("gb", "us"):
                wav = gen(model, p[f"ipa_{accent}"], ref_wav, ref_ipa)
                sf.write(f"{a.out}/{name}__{accent}__{tag}.wav", wav, SR)
            print(f"  {name}: {p['text']}   [{p['targets']}]")

    print(f"\n-> {a.out}\n   compare <probe>__gb__base vs <probe>__gb__v7 for the fix,\n"
          f"   and <probe>__us__base vs <probe>__us__v7 for the regression it must not cause.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
