#!/usr/bin/env python3
"""
Aggregate the PHONE-LEVEL disagreements between our IPA and the recognizer, per language.

⚠ WHY THIS EXISTS. Reading the investigate queue row by row found every large defect class in this
corpus — but it is a per-row instrument, and a defect that is spread thinly across many rows looks
like noise in each one. A single wrong mapping that costs one phone per utterance never makes the
worst-first queue at all, because one phone in a hundred does not move the distance.

So: align our IPA against the recognizer's phones for the rows already flagged, and count the
SUBSTITUTIONS. A systematic error shows up as one pair dominating the tally; recognizer noise shows
up as a long flat tail of unrelated pairs.

⚠ AND MOST OF WHAT IT REPORTS IS *EXPECTED*. The recognizer has its own inventory and its own
conventions: rhotics collapse, aspiration is unmarked, our tie bars and length marks have no
counterpart. The `fold()` shared with asr_align_report.py removes the worst of that, but plenty
remains, so the output is a RANKING to read, never a defect list. The question to ask of the top
pair is not "is this a mismatch" — it is "is this mismatch one the recognizer would make anyway?"

  python3 confusion_pairs.py --lang en_us
  python3 confusion_pairs.py --all --top 6
"""
from __future__ import annotations

import argparse
import difflib
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, "/mnt/data/Programming/vernacula/scripts/omnivoice_ipa")
from asr_align_report import fold  # noqa: E402

DB = "/mnt/data/omnivoice_ipa/work/asr_align/align.sqlite"


def confusions(db: sqlite3.Connection, lang: str, status: str) -> tuple[Counter, Counter, Counter, int]:
    """(substitutions, ours-only, theirs-only, rows) over one language's flagged rows."""
    sub: Counter = Counter()
    ours: Counter = Counter()
    theirs: Counter = Counter()
    n = 0
    q = "SELECT ipa, phones FROM utt WHERE lang=? AND phones IS NOT NULL AND phones!=''"
    if status != "all":
        q += " AND status=?"
    rows = db.execute(q, (lang, status) if status != "all" else (lang,))
    for ipa, ph in rows:
        a, b = list(fold(ipa or "")), list(fold(ph or ""))
        if not a or not b:
            continue
        n += 1
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag == "replace":
                # only count 1:1 replacements — longer blocks are alignment slippage, not a mapping
                if i2 - i1 == 1 and j2 - j1 == 1:
                    sub[(a[i1], b[j1])] += 1
            elif tag == "delete":
                for c in a[i1:i2]:
                    ours[c] += 1
            elif tag == "insert":
                for c in b[j1:j2]:
                    theirs[c] += 1
    return sub, ours, theirs, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--lang")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--status", default="investigate", help="'investigate' (default) or 'all'")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    db = sqlite3.connect(a.db)
    langs = ([a.lang] if a.lang else
             [r[0] for r in db.execute("SELECT DISTINCT lang FROM utt ORDER BY lang")])

    for lang in langs:
        sub, ours, theirs, n = confusions(db, lang, a.status)
        if not n:
            continue
        total = sum(sub.values()) or 1
        print(f"\n### {lang}  ({n} rows, status={a.status})")
        print("  substitutions ours->theirs        deletions (ours only)     insertions (theirs only)")
        s = sub.most_common(a.top)
        o = ours.most_common(a.top)
        t = theirs.most_common(a.top)
        for i in range(max(len(s), len(o), len(t))):
            c1 = f"{s[i][0][0]} -> {s[i][0][1]}  {s[i][1]:>4} {100*s[i][1]/total:>4.1f}%" if i < len(s) else ""
            c2 = f"{o[i][0]} {o[i][1]:>5}" if i < len(o) else ""
            c3 = f"{t[i][0]} {t[i][1]:>5}" if i < len(t) else ""
            print(f"  {c1:<34}{c2:<26}{c3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
