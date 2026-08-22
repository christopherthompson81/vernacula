#!/usr/bin/env python3
"""
Rebuild `manifest_<lang>.jsonl` from the codes + the align DB. NO GPU, NO AUDIO, seconds per language.

⚠ THE npz IS THE DURABLE ARTIFACT AND THIS IS NOT. `codes_<lang>.npz` is `{utterance_id: (8, T) int16}`
— a pure function of the waveform and the encoder, so it is write-once and stable. The manifest is
metadata *about* those vectors: which IPA they pair with, and whether we currently think the pair is
usable. Those are revisable, so they must not share a lifecycle with the encode.

⚠ THE FAILURE THIS EXISTS TO PREVENT. Exclusion used to be applied at ENCODE time — a row the align DB
marked `defective_audio` was never encoded. That baked a judgement into a GPU artifact, so revising it
meant a re-encode, and worse, an old judgement stayed frozen in whichever languages were encoded before
it changed. cy_gb and es_419 were ingested 2026-07-01 and carried 970 `defective_audio` rows in their
manifests for two months while the other 96 languages were pruned. Nothing in any log said so; it
surfaced only because `patch_manifest_ipa.py --db` reported them as "not in source".

So: encode once, label freely. Change your mind about a status and re-run this, not the GPU.

⚠ WHAT IS DERIVED AND WHAT IS LOOKED UP. `n_frames` is the codes' own shape and `dur_s` is exactly
`n_frames * 960 / SR_OUT` — verified against 400 rows, zero mismatch beyond the one frame of encoder
padding. So neither needs the audio. `gender`/`sentence_id` come from the FLEURS TSV, `ipa`/`status`/
`ipa_src` from the align DB.

⚠ AN EMPTY `status` MEANS THE DB HAS NO VERDICT, NOT THAT THE ROW IS CLEAN. The align pass does not
necessarily cover a whole language — as_in had 1,120 DB rows against 2,812 in the corpus — and reading
absence as a verdict is how 60% of Assamese was once dropped from training and logged as deliberate.
`corpus_filter` excludes only on known-bad statuses and never infers from a blank.

  python3 build_manifest.py                 # every language that has codes
  python3 build_manifest.py cy_gb es_419
  python3 build_manifest.py --check         # report, write nothing
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys

import numpy as np

ROOT = os.environ.get("OMNIVOICE_ROOT", "/mnt/data/omnivoice_ipa")
TOKENS = f"{ROOT}/corpus/tokens"
TSV = f"{ROOT}/corpus/fleurs_transcripts/data"
ALIGN_DB = f"{ROOT}/work/asr_align/align.sqlite"
SR_OUT, HOP = 24000, 960


def tsv_meta(lang: str) -> dict[str, tuple[str, str]]:
    """{wav_stem: (sentence_id, gender)} from the FLEURS transcript."""
    out: dict[str, tuple[str, str]] = {}
    path = f"{TSV}/{lang}/train.tsv"
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 7 and c[1].endswith(".wav"):
                out.setdefault(c[1][:-4], (c[0], c[6]))
    return out


def db_rows(lang: str) -> dict[str, tuple[str, str, str]]:
    """{wav_stem: (ipa, read_text_src, status)} — no status filter; the label is data, not a decision."""
    if not os.path.exists(ALIGN_DB):
        return {}
    db = sqlite3.connect(f"file:{ALIGN_DB}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT wav, ipa, COALESCE(read_text_src,''), COALESCE(status,'') FROM utt "
        "WHERE lang=? AND ipa IS NOT NULL AND TRIM(ipa) <> ''", (lang,)).fetchall()
    db.close()
    return {(w[:-4] if w.endswith(".wav") else w): (i, s, st) for w, i, s, st in rows}


def build(lang: str, check: bool) -> tuple[int, int, int]:
    npz = f"{TOKENS}/codes_{lang}.npz"
    if not os.path.exists(npz):
        return 0, 0, 0
    z = np.load(npz)
    meta, dbm = tsv_meta(lang), db_rows(lang)
    # ⚠ Keep the OLD manifest's ipa for rows the DB cannot supply, rather than dropping the row: the
    #   codes exist and an id with no metadata is worse than an id with stale metadata that says so.
    old = {}
    mf = f"{TOKENS}/manifest_{lang}.jsonl"
    if os.path.exists(mf):
        old = {d["id"]: d for d in (json.loads(l) for l in open(mf, encoding="utf8"))}
    rows, no_ipa, flagged = [], 0, 0
    for uid in z.files:
        hit = dbm.get(uid)
        prev = old.get(uid, {})
        if hit:
            ipa, src, status = hit
        else:
            ipa, src, status = prev.get("ipa"), prev.get("ipa_src", "none"), prev.get("status", "")
        if not ipa:
            no_ipa += 1
            continue
        sid, gender = meta.get(uid, (prev.get("sentence_id"), prev.get("gender")))
        n = int(z[uid].shape[-1])
        rows.append(dict(id=uid, sentence_id=sid, lang=lang, ipa=ipa, gender=gender,
                         dur_s=round(n * HOP / SR_OUT, 2), n_frames=n,
                         ipa_src=src, status=status))
        if status:
            flagged += 1
    # ⚠ THE JOIN IS BY `id`, NEVER BY POSITION, and this asserts it rather than trusting it. The npz is
    #   a KEYED archive (`np.savez(**{utterance_id: codes})`, that way since the first commit), so a
    #   consumer must look up `codes[row["id"]]`. ⚠ THE TWO FILES HAPPEN TO BE IN THE SAME ORDER TODAY —
    #   this loop walks `z.files` — and that coincidence invites a zip()-based loader that works right
    #   up until it does not. `topup_codes.py` appends at the end of the dict, so the orders WILL
    #   diverge. Order is not a contract; the id set is.
    # ⚠ THIS CANNOT FIRE AS THE CODE STANDS — `rows` is built by walking `z.files`, so every id is a key
    #   by construction, and a deliberate test with a bogus id did not trip it. It is here as a guard on
    #   that construction, not as a live check. ⚠ THE ACTUAL RISK IS ON THE CONSUMER SIDE: a loader that
    #   zips the two files instead of looking up `codes[row["id"]]`. That is a documentation problem and
    #   the dataset card states it.
    ids = {r["id"] for r in rows}
    orphan = ids - set(z.files)
    if orphan:
        raise AssertionError(f"{lang}: {len(orphan)} manifest ids have no codes, e.g. {sorted(orphan)[:3]}")
    if len(ids) != len(rows):
        raise AssertionError(f"{lang}: duplicate id in manifest ({len(rows)} rows, {len(ids)} ids)")
    # ⚠ The reverse is NOT an error: a code with no row is a row this build dropped for having no IPA,
    #   which `no_ipa` already counts and prints. Only a row pointing at absent codes is unrecoverable.
    if not check:
        with open(mf, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), no_ipa, flagged


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    langs = args or sorted(os.path.basename(f)[len("codes_"):-len(".npz")]
                           for f in glob.glob(f"{TOKENS}/codes_*.npz"))
    tot = tot_no = tot_flag = 0
    for lang in langs:
        n, no_ipa, flagged = build(lang, check)
        tot += n; tot_no += no_ipa; tot_flag += flagged
        if no_ipa:
            print(f"  {lang:14}{n:6} rows  ⚠ {no_ipa} codes with no IPA in the DB or the old manifest")
    print(f"\n{'would write' if check else 'wrote'} {tot} rows across {len(langs)} languages "
          f"({tot_flag} carry a status; corpus_filter decides which of those are excluded"
          + (f"; {tot_no} codes had no IPA and were left out" if tot_no else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
