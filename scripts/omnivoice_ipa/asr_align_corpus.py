#!/usr/bin/env python3
"""
Recognize phones for EVERY FLEURS training utterance and log them beside our IPA, for QC.

Every quality gate so far has compared TEXT to TEXT — our IPA against espeak's, against an earlier
generation of our own, against a model's opinion. Run 31 named the limit: the FLEURS transcript is the
SCRIPT the reader was given, not a record of what they said, so a phonemizer can be perfectly correct and
the PAIR still teach a wrong alignment. This is the first gate that listens to the audio.

  input text  ->  our IPA (phonemized_vernacula)      what we will train on
  audio       ->  recognized phones (wav2vec2)        what was actually said

`facebook/wav2vec2-xlsr-53-espeak-cv-ft` is a multilingual IPA phone recognizer, so it is
language-agnostic by construction and does not presuppose either side. It is NOT ground truth: it has its
own espeak-flavoured inventory and its own error rate, which is exactly why the report stage scores each
utterance against ITS OWN LANGUAGE's distribution rather than an absolute threshold. What we are hunting
is OUTLIERS — and even those split three ways: reader divergence, our bugs, and recognizer artefacts.

Audio is streamed straight out of the per-language `train.tar.gz` (104 GB across 28 languages); nothing
is extracted to disk. Rows land in SQLite as they are produced, so the run is resumable and inspectable
while it is still going.

Usage:
  python3 asr_align_corpus.py                      # all 28 corpus languages
  python3 asr_align_corpus.py --langs xh_za zu_za
  python3 asr_align_corpus.py --limit 50 --langs en_us     # throughput probe
"""
from __future__ import annotations

import argparse
import io
import os
import sqlite3
import sys
import tarfile
import time

ROOT = "/mnt/data/omnivoice_ipa"
TSV = f"{ROOT}/corpus/fleurs_transcripts/data"
AUDIO = f"{ROOT}/corpus/audio_cache/data"
BYID = f"{ROOT}/work/phonemized_vernacula/byid"
DB = f"{ROOT}/work/asr_align/align.sqlite"
MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

SCHEMA = """
CREATE TABLE IF NOT EXISTS utt (
    lang        TEXT NOT NULL,
    sentence_id TEXT NOT NULL,
    wav         TEXT NOT NULL,
    text        TEXT,          -- FLEURS normalized transcript (col 3), the phonemizer's input
    ipa         TEXT,          -- our current output for that sentence_id
    phones      TEXT,          -- what wav2vec2 heard
    n_samples   INTEGER,
    PRIMARY KEY (lang, wav)
);
CREATE INDEX IF NOT EXISTS utt_lang ON utt(lang);
"""


def langs_available() -> list[str]:
    return sorted(
        f[:-4] for f in os.listdir(BYID)
        if f.endswith(".tsv") and not f.endswith(".errors.tsv")
    )


def load_rows(lang: str) -> dict[str, tuple[str, str]]:
    """wav basename -> (sentence_id, text). The wav is the key: FLEURS repeats a sentence_id across
    speakers, and each RECORDING is a separate observation of what a reader said."""
    out: dict[str, tuple[str, str]] = {}
    p = f"{TSV}/{lang}/train.tsv"
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[3].strip():
                out[c[1]] = (c[0], c[3].strip())
    return out


def load_ipa(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    p = f"{BYID}/{lang}.tsv"
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            if v:
                out.setdefault(k, v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="utterances per language (probe)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer

    os.makedirs(os.path.dirname(a.db), exist_ok=True)
    db = sqlite3.connect(a.db)
    db.executescript(SCHEMA)
    # WAL so the report stage can read the table while this is still writing.
    db.execute("PRAGMA journal_mode=WAL")
    db.commit()

    # ⚠ `do_phonemize=False`: the tokenizer builds an espeak backend in its constructor and hard-requires
    # the `phonemizer` package, but only for the ENCODE direction. We decode CTC ids to a phone string.
    tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL, do_phonemize=False)
    ext = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForCTC.from_pretrained(MODEL)
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.half()
    print(f"# model on {dev}, batch {a.batch}", file=sys.stderr)

    todo = a.langs or langs_available()
    for lang in todo:
        rows, ipa = load_rows(lang), load_ipa(lang)
        tar_path = f"{AUDIO}/{lang}/audio/train.tar.gz"
        if not os.path.exists(tar_path):
            print(f"{lang}: no train.tar.gz, skipped", file=sys.stderr)
            continue
        have = {r[0] for r in db.execute("SELECT wav FROM utt WHERE lang=?", (lang,))}
        t0, n, batch = time.time(), 0, []

        def flush(batch: list[tuple[str, str, str, np.ndarray]]) -> None:
            if not batch:
                return
            feats = ext([b[3] for b in batch], sampling_rate=16000, return_tensors="pt", padding=True)
            iv = feats.input_values.to(dev)
            if dev == "cuda":
                iv = iv.half()
            with torch.no_grad():
                logits = model(iv, attention_mask=feats.get("attention_mask", None)).logits
            for (wav, sid, txt, aud), ids in zip(batch, torch.argmax(logits, dim=-1)):
                db.execute(
                    "INSERT OR REPLACE INTO utt(lang,sentence_id,wav,text,ipa,phones,n_samples) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (lang, sid, wav, txt, ipa.get(sid), tok.decode(ids), len(aud)),
                )
            db.commit()

        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".wav"):
                    continue
                wav = os.path.basename(m.name)
                if wav not in rows or wav in have:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                try:
                    aud, sr = sf.read(io.BytesIO(fh.read()), dtype="float32")
                except Exception as e:  # a truncated member must not kill the language
                    print(f"  {lang}/{wav}: unreadable ({e})", file=sys.stderr)
                    continue
                if aud.ndim > 1:
                    aud = aud.mean(axis=1)
                if sr != 16000:
                    print(f"  {lang}/{wav}: {sr} Hz, skipped", file=sys.stderr)
                    continue
                sid, txt = rows[wav]
                batch.append((wav, sid, txt, aud))
                n += 1
                if len(batch) >= a.batch:
                    flush(batch)
                    batch = []
                if a.limit and n >= a.limit:
                    break
        flush(batch)
        dt = time.time() - t0
        print(f"{lang}: {n} utterances in {dt:.0f}s ({n / max(dt, 1e-9):.1f}/s)", file=sys.stderr)

    total = db.execute("SELECT COUNT(*) FROM utt").fetchone()[0]
    print(f"\n{a.db}: {total} rows", file=sys.stderr)
    db.close()


if __name__ == "__main__":
    main()
