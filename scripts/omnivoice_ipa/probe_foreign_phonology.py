#!/usr/bin/env python3
"""
Which phonology does a reader actually use for a Latin proper noun embedded in non-English speech?

The corpus now reads those names, and reads them as AMERICAN ENGLISH (core/foreign.ts routes an
unclaimed Latin run to `en`). That is a routing decision the text cannot settle — a Korean reader
saying "atlanta" is plausibly saying it with Korean phonology, in which case the target we train on
carries an accent the audio does not. Run 31's lesson applies: the audio can answer it.

Parakeet is European-only, so this uses `facebook/wav2vec2-xlsr-53-espeak-cv-ft` — a multilingual
IPA phone recognizer, which is language-agnostic by construction and therefore does not presuppose
either answer. It is NOT a transcription: it is a phone string, which is exactly what we want to
compare against two competing phone hypotheses.

Only utterances whose transcript BEGINS with the Latin token are used, so the region of interest is
the head of the phone string and no alignment is needed. That is the whole reason for the constraint.

Usage:
  python3 probe_foreign_phonology.py                       # all candidates
  python3 probe_foreign_phonology.py --candidates X.tsv    # lang \t id \t wav \t lead \t text
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = "/mnt/data/omnivoice_ipa"
PROBE = f"{ROOT}/work/asr_probe"
MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


def load_candidates(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4:
                out.append(dict(lang=c[0], id=c[1], wav=c[2], lead=c[3], text=c[4] if len(c) > 4 else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=f"{PROBE}/candidates.tsv")
    ap.add_argument("--wavdir", default=f"{PROBE}/wav")
    ap.add_argument("--out", default=f"{PROBE}/phones.tsv")
    a = ap.parse_args()

    import torch  # imported here so --help works without the venv
    import soundfile as sf
    from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer

    cands = load_candidates(a.candidates)
    print(f"# {len(cands)} candidates, model {MODEL}", file=sys.stderr)

    # ⚠ `do_phonemize=False`, and the two halves loaded separately rather than through AutoProcessor.
    # Wav2Vec2PhonemeCTCTokenizer initialises an espeak backend in its constructor and hard-requires the
    # `phonemizer` package — but only for the ENCODE direction (text → phones), which this probe never uses.
    # We decode CTC ids to a phone string. Building the tokenizer with phonemization off skips that backend,
    # so the probe needs no espeak and, more to the point, no second phonemizer in the loop it is refereeing.
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL, do_phonemize=False)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForCTC.from_pretrained(MODEL)
    model.eval()

    rows = []
    for c in cands:
        p = os.path.join(a.wavdir, c["wav"])
        if not os.path.exists(p):
            print(f"  MISSING {p}", file=sys.stderr)
            continue
        audio, sr = sf.read(p, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # The recognizer is 16 kHz; FLEURS ships 16 kHz, so assert rather than silently resample.
        assert sr == 16000, f"{p}: expected 16 kHz, got {sr}"
        inputs = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        phones = tokenizer.batch_decode(torch.argmax(logits, dim=-1))[0]
        rows.append({**c, "phones": phones})
        # Only the head matters — the lead token is the first thing said.
        head = "".join(phones.split())[:36]
        print(f"{c['lang']:<14} {c['lead']:<22} head={head}", file=sys.stderr)

    with open(a.out, "w", encoding="utf8") as f:
        f.write("lang\tid\twav\tlead\tphones\ttext\n")
        for r in rows:
            f.write(f"{r['lang']}\t{r['id']}\t{r['wav']}\t{r['lead']}\t{r['phones']}\t{r['text']}\n")
    print(f"\nwrote {a.out} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
