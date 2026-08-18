#!/usr/bin/env python3
"""
Fetch every FLEURS language we do not already have — audio tarballs and transcripts.

FLEURS ships 102 languages with a train split. The corpus was built from 24, the audio cache grew to
66 opportunistically, and this closes the gap to all 102 so the QC and any future expansion are not
limited by what happens to be on disk.

⚠ RESUMABLE BY CONSTRUCTION, and deliberately so. This pulls ~70 GB over a slow link;
`hf_hub_download` is content-addressed and skips anything already complete, so an interrupted run
costs the file in flight and nothing else. The silence sweep in this same directory had to be
rewritten for exactly this reason after a kill lost an hour of work — a long job that cannot resume
is a job that has to be run twice.

Audio and transcripts are fetched independently: a language whose audio fails still gets its text, and
the per-language result is printed as it lands rather than buffered to the end.

  python3 fetch_fleurs_all.py            # everything missing
  python3 fetch_fleurs_all.py --dry-run  # list what would be fetched
  python3 fetch_fleurs_all.py --text-only
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = "/mnt/data/omnivoice_ipa"
AUDIO_CACHE = f"{ROOT}/corpus/audio_cache"
TEXT_CACHE = f"{ROOT}/corpus/fleurs_transcripts"
REPO = "google/fleurs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--audio-only", action="store_true")
    a = ap.parse_args()

    files = list_repo_files(REPO, repo_type="dataset")
    langs = sorted({m.group(1) for f in files
                    if (m := re.match(r"data/([^/]+)/audio/train\.tar\.gz$", f))})
    have_audio = set(os.listdir(f"{AUDIO_CACHE}/data")) if os.path.isdir(f"{AUDIO_CACHE}/data") else set()
    have_text = set(os.listdir(f"{TEXT_CACHE}/data")) if os.path.isdir(f"{TEXT_CACHE}/data") else set()

    need_audio = [] if a.text_only else [x for x in langs if x not in have_audio]
    need_text = [] if a.audio_only else [x for x in langs if x not in have_text]

    print(f"# FLEURS languages with a train split: {len(langs)}")
    print(f"#   audio cached {len(have_audio & set(langs))}, missing {len(need_audio)}")
    print(f"#   text  cached {len(have_text & set(langs))}, missing {len(need_text)}")
    if a.dry_run:
        print("\naudio:", " ".join(need_audio) or "(none)")
        print("text :", " ".join(need_text) or "(none)")
        return 0

    # transcripts first — they are tiny, and having text without audio is still useful for a
    # phonemizer-side pass, whereas audio without text is useless.
    for i, lang in enumerate(need_text, 1):
        for split in ("train.tsv", "dev.tsv", "test.tsv"):
            try:
                hf_hub_download(REPO, f"data/{lang}/{split}", repo_type="dataset",
                                local_dir=TEXT_CACHE)
            except Exception as e:  # a language may not ship every split
                print(f"  [{i}/{len(need_text)}] {lang} {split}: {type(e).__name__}", flush=True)
        print(f"[text {i}/{len(need_text)}] {lang}", flush=True)

    total = 0.0
    for i, lang in enumerate(need_audio, 1):
        try:
            p = hf_hub_download(REPO, f"data/{lang}/audio/train.tar.gz", repo_type="dataset",
                                local_dir=AUDIO_CACHE)
            gb = os.path.getsize(p) / 1e9
            total += gb
            print(f"[audio {i}/{len(need_audio)}] {lang}: {gb:.2f} GB  (cumulative {total:.1f} GB)",
                  flush=True)
        except Exception as e:
            print(f"[audio {i}/{len(need_audio)}] {lang}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"\ndone — {total:.1f} GB of audio fetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
