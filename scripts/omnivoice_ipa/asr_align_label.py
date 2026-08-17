#!/usr/bin/env python3
"""
Add and maintain the review columns on the alignment table: `status` and `comment`.

The scoring pass produces a ranking; this produces a RECORD. The difference matters because the ranking
is recomputed every time the scorer changes — and it changed three times already, when modifier letters,
tone digits and recognizer failures each turned out to be scoring artefacts rather than defects. A verdict
written into the row survives that; a position in a sorted file does not.

Status values, and the split they encode:

  verified          the three views agree: input text, our IPA, and the recognized phones. Inside the
                    language's own distribution, so there is nothing here to look at. Set in bulk.
  investigate       in the tail for its language — a real disagreement worth a human read. Set in bulk.
  recognizer_short  the recognizer returned far too little to compare (a whole Welsh sentence came back
                    as the single phone `k`). Says nothing about our IPA. Set in bulk.
  ---- below here are set by hand, and only ever by hand ----
  defect            our phonemization is wrong. The thing we are hunting.
  reader_divergence the reader did not say what the transcript says (Run 31's finding). Not ours to fix,
                    but it makes the PAIR bad training data, which is its own decision.
  convention        we and the recognizer disagree about notation, not about the reading (ʈ vs t, b vs v).
  artefact          the recognizer is simply wrong here.

⚠ A BULK PASS NEVER OVERWRITES A HAND VERDICT. Re-running the scorer must not silently erase review work,
so the automatic statuses are only written where `status` is NULL or is itself automatic.

Usage:
  python3 asr_align_label.py --apply                  # (re)apply the automatic labels
  python3 asr_align_label.py --set defect --lang en_us --id 28 --comment "adjacent numbers merged"
  python3 asr_align_label.py --stats
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys

ROOT = "/mnt/data/omnivoice_ipa"
DB = f"{ROOT}/work/asr_align/align.sqlite"
AUTOMATIC = ("verified", "investigate", "recognizer_short")


def ensure_columns(db: sqlite3.Connection) -> None:
    cols = {r[1] for r in db.execute("PRAGMA table_info(utt)")}
    if "status" not in cols:
        db.execute("ALTER TABLE utt ADD COLUMN status TEXT")
    if "comment" not in cols:
        db.execute("ALTER TABLE utt ADD COLUMN comment TEXT")
    if "dist" not in cols:
        # Cached so a hand review can sort and filter without recomputing the fold every time.
        db.execute("ALTER TABLE utt ADD COLUMN dist REAL")
    db.execute("CREATE INDEX IF NOT EXISTS utt_status ON utt(status)")
    db.commit()


def apply_auto(db: sqlite3.Connection) -> None:
    sys.path.insert(0, "/mnt/data/Programming/vernacula/scripts/omnivoice_ipa")
    from asr_align_report import dist, fold

    langs = [r[0] for r in db.execute("SELECT DISTINCT lang FROM utt ORDER BY lang")]
    for lang in langs:
        rows = list(db.execute(
            "SELECT lang,wav,ipa,phones FROM utt WHERE lang=? AND ipa IS NOT NULL AND phones IS NOT NULL",
            (lang,)))
        scored, short = [], []
        for lg, wav, ipa, ph in rows:
            fi, fp = fold(ipa or ""), fold(ph or "")
            if not fp or (len(fi) >= 12 and len(fp) < 0.35 * len(fi)):
                short.append((lg, wav))
                continue
            scored.append((lg, wav, dist(fi, fp)))
        if len(scored) < 20:
            continue
        ds = [s[2] for s in scored]
        med = statistics.median(ds)
        mad = statistics.median([abs(d - med) for d in ds]) or 1e-9
        for lg, wav, d in scored:
            z = 0.6745 * (d - med) / mad
            st = "investigate" if z > 3.0 else "verified"
            # ⚠ Never clobber a hand verdict.
            db.execute(
                "UPDATE utt SET status=?, dist=? WHERE lang=? AND wav=? "
                "AND (status IS NULL OR status IN ('verified','investigate','recognizer_short'))",
                (st, d, lg, wav))
        for lg, wav in short:
            db.execute(
                "UPDATE utt SET status='recognizer_short' WHERE lang=? AND wav=? "
                "AND (status IS NULL OR status IN ('verified','investigate','recognizer_short'))",
                (lg, wav))
        db.commit()
        print(f"  {lang}: {len(scored)} scored, {len(short)} short", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--apply", action="store_true", help="(re)apply the automatic labels")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--set")
    ap.add_argument("--lang")
    ap.add_argument("--id", help="sentence_id")
    ap.add_argument("--wav")
    ap.add_argument("--comment", default="")
    a = ap.parse_args()

    db = sqlite3.connect(a.db)
    ensure_columns(db)

    if a.apply:
        apply_auto(db)
    if a.set:
        if not a.lang or not (a.id or a.wav):
            sys.exit("--set needs --lang and (--id or --wav)")
        where, args = ("sentence_id=?", [a.id]) if a.id else ("wav=?", [a.wav])
        n = db.execute(f"UPDATE utt SET status=?, comment=? WHERE lang=? AND {where}",
                       [a.set, a.comment, a.lang, *args]).rowcount
        db.commit()
        print(f"{n} row(s) set to {a.set}", file=sys.stderr)
    if a.stats or a.apply:
        print("\nstatus                 rows", file=sys.stderr)
        for st, n in db.execute(
                "SELECT COALESCE(status,'(unlabelled)'), COUNT(*) FROM utt GROUP BY 1 ORDER BY 2 DESC"):
            print(f"  {st:<20} {n}", file=sys.stderr)
    db.close()


if __name__ == "__main__":
    main()
