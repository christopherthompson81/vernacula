"""Materialize the training exclusion list from the alignment DB.

Reads the `status` column that `asr_align_label.py` wrote and emits `work/exclusions.tsv`
(lang, id, reason) for every utterance that cannot be used as a training pair. Run this FIRST — see
the ordering note in `corpus_filter.py`; the weights in `sampling_budget.py` are computed over
whatever survives, so excluding afterwards targets the wrong number silently.

  python3 exclude_defective.py            # write work/exclusions.tsv
  python3 exclude_defective.py --check    # report only, write nothing

⚠ THE COVERAGE CHECK IS NOT OPTIONAL, and it is why this prints what it prints. cy_gb was selected
into the corpus as the OWNER of a census primitive — U+0325, the voiceless ring — and is its SOLE
source across all 28 languages. Dropping 17.1% of Welsh therefore risks a primitive, not just rows.
It survives (checked at Run 36: 1,610 of 1,937 occurrences remain, in 1,148 utterances, and no phone
vanishes entirely), but it survives *because it was checked*. A defect concentrated on the primitive
a language was chosen for would have been a different answer, so this script re-checks it every time
rather than trusting the note.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filter import EXCLUDE_STATUSES, EXCLUSIONS, ROOT, TOKENS  # noqa: E402

DB = f"{ROOT}/work/asr_align/align.sqlite"
SILENT = f"{ROOT}/work/silent_audio.tsv"


def silent_rows(path: str) -> dict[str, set[str]]:
    """{lang: {utterance id}} from scan_silent_audio.py, or empty if it has not been run.

    ⚠ TWO SOURCES, UNIONED, BECAUSE THEY FIND DIFFERENT THINGS. The DB `status` column carries what
    the recognizer-based sweep concluded — including the Welsh TRUNCATION, which is audible and would
    never trip a silence test. `silent_audio.tsv` carries what measuring the waveform found — the
    Spanish files that are FULL LENGTH AND EMPTY, which no duration or transcript check can see.
    Neither is a superset of the other, so the exclusion takes both.
    """
    out: dict[str, set[str]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            wav = parts[1]
            out.setdefault(parts[0], set()).add(wav[:-4] if wav.endswith(".wav") else wav)
    return out


def phone_counts(rows: list[dict]) -> Counter:
    """Occurrences of every character in the IPA of `rows` (a cheap proxy for phone inventory)."""
    c: Counter = Counter()
    for r in rows:
        c.update(ch for ch in r["ipa"] if not ch.isspace())
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=EXCLUSIONS)
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    a = ap.parse_args()

    db = sqlite3.connect(a.db)
    q = ",".join("?" * len(EXCLUDE_STATUSES))
    rows = db.execute(
        f"SELECT lang, wav, status FROM utt WHERE status IN ({q}) ORDER BY lang, wav",
        EXCLUDE_STATUSES,
    ).fetchall()

    by_lang: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[str]] = {}
    for lang, wav, status in rows:
        uid = wav[:-4] if wav.endswith(".wav") else wav
        by_lang.setdefault(lang, []).append((uid, status))
        seen.setdefault(lang, set()).add(uid)
    # union in anything the waveform scan found that the DB has not been told about yet
    n_from_scan = 0
    for lang, ids in silent_rows(SILENT).items():
        for uid in sorted(ids - seen.get(lang, set())):
            by_lang.setdefault(lang, []).append((uid, "silent_audio"))
            seen.setdefault(lang, set()).add(uid)
            n_from_scan += 1
    if n_from_scan:
        print(f"# +{n_from_scan} from {SILENT} not yet labelled in the DB")

    print(f"# exclusion statuses: {', '.join(EXCLUDE_STATUSES)} (+ silent_audio.tsv)")
    print(f"# {len(rows)} utterances across {len(by_lang)} languages\n")

    total_dropped = 0
    for lang in sorted(by_lang):
        mf = f"{TOKENS}/manifest_{lang}.jsonl"
        if not os.path.exists(mf):
            print(f"{lang:<12} {len(by_lang[lang]):>4} flagged   (no manifest — not in the corpus)")
            continue
        man = [json.loads(l) for l in open(mf, encoding="utf-8")]
        ex = {uid for uid, _ in by_lang[lang]}
        keep = [r for r in man if r["id"] not in ex]
        drop = [r for r in man if r["id"] in ex]
        total_dropped += len(drop)

        # ⚠ per-language coverage delta — see the module note. Report any phone that would VANISH.
        before, after = phone_counts(man), phone_counts(keep)
        lost = sorted(p for p in before if after.get(p, 0) == 0)
        secs = sum(r.get("dur_s", 0.0) for r in drop)
        pct = 100 * len(drop) / max(len(man), 1)
        flag = "  ⚠ PHONES LOST: " + " ".join(lost) if lost else ""
        print(f"{lang:<12} drop {len(drop):>4}/{len(man):<5} ({pct:4.1f}%)  "
              f"{secs/60:5.1f} min  -> {len(keep)} kept{flag}")

    print(f"\ntotal dropped from the corpus: {total_dropped}")

    if a.check:
        print("\n--check: nothing written")
        return 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# lang\tid\treason — generated by exclude_defective.py; see corpus_filter.py\n")
        for lang in sorted(by_lang):
            for uid, status in sorted(by_lang[lang]):
                f.write(f"{lang}\t{uid}\t{status}\n")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
