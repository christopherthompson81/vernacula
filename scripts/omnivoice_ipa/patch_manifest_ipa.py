"""Refresh the `ipa` field of the token manifests from the current phonemizer output,
WITHOUT re-encoding audio.

The IPA transcript is baked into corpus/tokens/manifest_<lang>.jsonl at ingest time; when the
phonemizer changes (e.g. espeak-ng-portable offglide/flap relabels) only the IPA text changes —
the paired audio codes (codes_<lang>.npz) are byte-identical. So re-ingest (which re-runs the
GPU encoder) is wasted work: this reads the refreshed work/phonemized/byid/<lang>.tsv (keyed by
FLEURS sentence_id) and rewrites each manifest row's `ipa` in place, matched by `sentence_id`.

Usage: patch_manifest_ipa.py [--byid <dir>] [lang ...]   (default: all manifest_*.jsonl)

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
BYID = f"{ROOT}/work/phonemized/byid"


def load_byid(lang):
    m = {}
    with open(f"{BYID}/{lang}.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 1)
            if len(p) == 2:
                m[p[0]] = p[1]
    return m


def patch(lang):
    mf = f"{TOKENS}/manifest_{lang}.jsonl"
    byid = load_byid(lang)
    rows = [json.loads(l) for l in open(mf, encoding="utf-8")]
    changed = missed = 0
    for r in rows:
        new = byid.get(r["sentence_id"])
        if new is None:
            missed += 1
        elif new != r["ipa"]:
            r["ipa"] = new
            changed += 1
    if changed:
        with open(mf, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), changed, missed


def main():
    args = sys.argv[1:]
    global BYID
    if args and args[0] == "--byid":
        BYID = args[1]
        args = args[2:]
    langs = args or sorted(
        os.path.basename(f)[len("manifest_"):-len(".jsonl")]
        for f in glob.glob(f"{TOKENS}/manifest_*.jsonl"))
    total = total_missed = 0
    for lang in langs:
        n, changed, missed = patch(lang)
        total += changed
        total_missed += missed
        if changed or missed:
            miss = f"  (MISSED {missed} — kept old IPA!)" if missed else ""
            print(f"{lang}: {changed}/{n} rows IPA-refreshed{miss}")
    print(f"\ntotal rows refreshed: {total} across {len(langs)} langs"
          + (f"; {total_missed} rows had no byid entry" if total_missed else ""))


if __name__ == "__main__":
    main()
