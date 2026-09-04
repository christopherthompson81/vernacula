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
import multiprocessing as mp
import os
import queue
import re
import sys
import time

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = "/mnt/data/omnivoice_ipa"
AUDIO_CACHE = f"{ROOT}/corpus/audio_cache"
TEXT_CACHE = f"{ROOT}/corpus/fleurs_transcripts"
REPO = "google/fleurs"


# ⚠ A STALL WATCHDOG, ADDED AFTER ONE. A run sat on nso_za for ELEVEN AND A HALF HOURS: the process was alive
# and asleep, the .incomplete file frozen at 335 MB, and nothing in the log since. hf_hub_download has a 10s
# read timeout (hub 1.8.0) and it did not help — a dead-but-open socket, or the library's own retry/backoff,
# leaves the call blocked with no way for the caller to notice.
#
# So the caller watches the BYTES instead of trusting the call. The download runs in a child process while the
# parent polls the cache size; if it has not grown in STALL_S the child is terminated. Watching progress rather
# than imposing a fixed deadline is the point — a genuinely slow link keeps its time, and only a frozen one is
# cut. The partial file stays on disk either way, because hf_hub_download is content-addressed and resumes.
STALL_S = 300      # no new bytes for this long → assume the socket is dead
POLL_S = 30
RETRIES = 3


def _cache_bytes() -> int:
    total = 0
    for root, _dirs, files in os.walk(AUDIO_CACHE):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:  # a file can vanish mid-walk as the downloader renames it into place
                pass
    return total


def _child(lang: str, q) -> None:
    try:
        q.put(("ok", hf_hub_download(REPO, f"data/{lang}/audio/train.tar.gz", repo_type="dataset",
                                     local_dir=AUDIO_CACHE)))
    except Exception as e:  # noqa: BLE001 — reported verbatim to the parent
        q.put(("err", f"{type(e).__name__}: {e}"))


def fetch_one(lang: str) -> str:
    """Download one language's tarball, aborting if the byte count stops moving."""
    q = mp.Queue()
    proc = mp.Process(target=_child, args=(lang, q), daemon=True)
    proc.start()
    last_size, last_change = _cache_bytes(), time.monotonic()
    try:
        while proc.is_alive():
            proc.join(POLL_S)
            if not q.empty():
                break
            size = _cache_bytes()
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            elif time.monotonic() - last_change > STALL_S:
                raise TimeoutError(f"no new bytes for {STALL_S}s")
        # ⚠ NOT get_nowait(). mp.Queue hands off through a feeder thread, so a child that has already
        # exited may not have flushed yet and an immediate get raises Empty — which the caller would
        # report as a stall and retry a download that actually succeeded.
        kind, payload = q.get(timeout=30)
    except queue.Empty:
        raise TimeoutError("child exited without reporting") from None
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
    if kind == "err":
        raise RuntimeError(payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--audio-only", action="store_true")
    a = ap.parse_args()

    files = list_repo_files(REPO, repo_type="dataset")
    langs = sorted({m.group(1) for f in files
                    if (m := re.match(r"data/([^/]+)/audio/train\.tar\.gz$", f))})
    # ⚠ TEST FOR THE TARBALL, NOT THE DIRECTORY. This used to be a listdir of data/, which counts a language as
    # cached the moment hf_hub_download creates its folder — so an interrupted download left a language looking
    # complete and it was skipped on every later run, silently and permanently. ast_es and nso_za were both in
    # that state (empty audio/ dirs) after two stalls, and the dry-run reported 9 missing when 11 were.
    have_audio = {d for d in (os.listdir(f"{AUDIO_CACHE}/data") if os.path.isdir(f"{AUDIO_CACHE}/data") else [])
                  if os.path.isfile(f"{AUDIO_CACHE}/data/{d}/audio/train.tar.gz")}
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
        ok, tries = False, 0
        for attempt in range(1, RETRIES + 1):
            tries = attempt
            try:
                p = fetch_one(lang)
            except TimeoutError as e:
                print(f"[audio {i}/{len(need_audio)}] {lang}: STALLED ({e}), attempt {attempt}/{RETRIES}",
                      flush=True)
                continue
            except Exception as e:
                print(f"[audio {i}/{len(need_audio)}] {lang}: FAILED {type(e).__name__}: {e}", flush=True)
                break
            gb = os.path.getsize(p) / 1e9
            total += gb
            print(f"[audio {i}/{len(need_audio)}] {lang}: {gb:.2f} GB  (cumulative {total:.1f} GB)", flush=True)
            ok = True
            break
        if not ok:
            # ⚠ Report the attempts actually MADE. A hard error breaks out on the first one, and saying
            # "after 3 attempts" would send the next reader looking for two failures that never happened.
            print(f"[audio {i}/{len(need_audio)}] {lang}: GIVING UP after {tries} "
                  f"attempt{'s' if tries != 1 else ''} — the partial file stays on disk, so a later run "
                  f"resumes it", flush=True)
    print(f"\ndone — {total:.1f} GB of audio fetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
