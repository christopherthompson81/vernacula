"""Refresh the `ipa` field of the token manifests from the current phonemizer output,
WITHOUT re-encoding audio.

The IPA transcript is baked into corpus/tokens/manifest_<lang>.jsonl at ingest time; when the
phonemizer changes (e.g. espeak-ng-portable offglide/flap relabels) only the IPA text changes —
the paired audio codes (codes_<lang>.npz) are byte-identical. So re-ingest (which re-runs the
GPU encoder) is wasted work: this reads the refreshed work/phonemized/byid/<lang>.tsv (keyed by
FLEURS sentence_id) and rewrites each manifest row's `ipa` in place, matched by `sentence_id`.

Usage: patch_manifest_ipa.py [--byid <dir>] [--db] [lang ...]   (default: all manifest_*.jsonl)

⚠ `--db` IS THE RIGHT SOURCE NOW, and `--byid` cannot express what it carries. The alignment DB keys
IPA by WAV; `byid` keys it by SENTENCE. `read_text` records what ONE reader actually said on ONE
recording — an English numeral inside a Hausa sentence, a Portuguese one inside Umbundu, the Bengali
year form — and its IPA is re-derived from that. Under a per-sentence key every take of a sentence gets
the same string, so those corrections cannot reach the manifest at all. `--db` also drops rows the
alignment marked unusable, which `--byid` has no way to know about.

--byid selects the phonemizer output tree (default work/phonemized/byid, the espeak corpus;
pass work/phonemized_vernacula/byid to retarget the manifests at the vernacula engine). A row
whose sentence_id is absent from the byid file keeps its OLD IPA — that would silently mix
engines, so misses are counted and reported per language.
"""
import glob
import json
import os
import sys

ROOT = "/mnt/data/omnivoice_ipa"
TOKENS = f"{ROOT}/corpus/tokens"
BYID = f"{ROOT}/work/phonemized_vernacula/byid"
ALIGN_DB = f"{ROOT}/work/asr_align/align.sqlite"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filter import EXCLUDE_STATUSES, EXCLUDE_UNLESS_HAND_READ_TEXT  # noqa: E402


def load_byid(lang):
    m = {}
    with open(f"{BYID}/{lang}.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 1)
            if len(p) == 2:
                m[p[0]] = p[1]
    return m


def load_db(lang):
    """{wav_basename: (ipa, read_text_src)} from the alignment DB, unusable rows already dropped."""
    import sqlite3
    q = ",".join("?" * len(EXCLUDE_STATUSES))
    q2 = ",".join("?" * len(EXCLUDE_UNLESS_HAND_READ_TEXT))
    db = sqlite3.connect(f"file:{ALIGN_DB}?mode=ro", uri=True)
    rows = db.execute(
        f"SELECT wav, ipa, COALESCE(read_text_src,'') FROM utt "
        f"WHERE lang=? AND ipa IS NOT NULL AND TRIM(ipa) <> '' "
        f"  AND COALESCE(status,'') NOT IN ({q}) "
        f"  AND NOT (COALESCE(status,'') IN ({q2}) AND COALESCE(read_text_src,'') <> 'hand')",
        (lang, *EXCLUDE_STATUSES, *EXCLUDE_UNLESS_HAND_READ_TEXT),
    ).fetchall()
    db.close()
    # manifest ids are the wav basename without ".wav"
    return {w[:-4] if w.endswith(".wav") else w: (i, src) for w, i, src in rows}


def patch(lang, use_db=False):
    mf = f"{TOKENS}/manifest_{lang}.jsonl"
    rows = [json.loads(l) for l in open(mf, encoding="utf-8")]
    changed = missed = hand = 0
    src = load_db(lang) if use_db else load_byid(lang)
    for r in rows:
        if use_db:
            hit = src.get(r["id"])
            new, rt = (hit if hit else (None, None))
            if rt == "hand":
                hand += 1
        else:
            new, rt = src.get(r["sentence_id"]), None
        if new is None:
            # ⚠ Under --db a miss is usually an EXCLUSION, not an absence: the row is in the corpus but
            # the alignment marked it unusable. Either way the old IPA is kept and the count printed —
            # a silent keep is how two engines end up mixed in one manifest.
            missed += 1
            continue
        if use_db and rt:
            r["ipa_src"] = rt
        if new != r["ipa"]:
            r["ipa"] = new
            changed += 1
    if changed:
        with open(mf, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), changed, missed, hand


def main():
    args = sys.argv[1:]
    global BYID
    use_db = False
    if args and args[0] == "--db":
        use_db = True
        args = args[1:]
    if args and args[0] == "--byid":
        BYID = args[1]
        args = args[2:]
    langs = args or sorted(
        os.path.basename(f)[len("manifest_"):-len(".jsonl")]
        for f in glob.glob(f"{TOKENS}/manifest_*.jsonl"))
    total = total_missed = 0
    total_hand = 0
    for lang in langs:
        n, changed, missed, hand = patch(lang, use_db)
        total += changed
        total_missed += missed
        total_hand += hand
        if changed or missed:
            miss = f"  ({missed} not in source — kept old IPA)" if missed else ""
            h = f"  [{hand} reader-corrected]" if hand else ""
            print(f"{lang}: {changed}/{n} rows IPA-refreshed{h}{miss}")
    print(f"\ntotal rows refreshed: {total} across {len(langs)} langs"
          + (f"; {total_hand} carry a hand read_text" if total_hand else "")
          + (f"; {total_missed} not found in the source" if total_missed else ""))


if __name__ == "__main__":
    main()
