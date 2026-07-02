"""Refresh the `ipa` field of the token manifests from the current phonemizer output,
WITHOUT re-encoding audio.

The IPA transcript is baked into corpus/tokens/manifest_<lang>.jsonl at ingest time; when the
phonemizer changes (e.g. espeak-ng-portable offglide/flap relabels) only the IPA text changes —
the paired audio codes (codes_<lang>.npz) are byte-identical. So re-ingest (which re-runs the
GPU encoder) is wasted work: this reads the refreshed work/phonemized/byid/<lang>.tsv (keyed by
FLEURS sentence_id) and rewrites each manifest row's `ipa` in place, matched by `sentence_id`.

Usage: patch_manifest_ipa.py [lang ...]   (default: all manifest_*.jsonl)
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
    changed = 0
    for r in rows:
        new = byid.get(r["sentence_id"])
        if new is not None and new != r["ipa"]:
            r["ipa"] = new
            changed += 1
    if changed:
        with open(mf, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), changed


def main():
    langs = sys.argv[1:] or sorted(
        os.path.basename(f)[len("manifest_"):-len(".jsonl")]
        for f in glob.glob(f"{TOKENS}/manifest_*.jsonl"))
    total = 0
    for lang in langs:
        n, changed = patch(lang)
        total += changed
        if changed:
            print(f"{lang}: {changed}/{n} rows IPA-refreshed")
    print(f"\ntotal rows refreshed: {total} across {len(langs)} langs")


if __name__ == "__main__":
    main()
