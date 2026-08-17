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
  defective_audio   the AUDIO is broken, not the IPA: far too short for its text. 585 Welsh files (17.1%
                    of that corpus) hold ~1.5 s for a sentence needing ~96 phones. Verified against the
                    source tar — the members really are that small there, so it is upstream FLEURS data
                    and not our download. NOT OURS TO FIX; the action is to exclude the pair from training
                    and report it upstream. Set in bulk.
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
import os
import sqlite3
import statistics
import sys

ROOT = "/mnt/data/omnivoice_ipa"
DB = f"{ROOT}/work/asr_align/align.sqlite"
AUTOMATIC = ("verified", "investigate", "recognizer_short", "defective_audio")
SILENT_TSV = f"{ROOT}/work/silent_audio.tsv"


def load_silent(path: str = SILENT_TSV) -> dict[str, set[str]]:
    """{lang: {wav basename}} from scan_silent_audio.py — empty if it has not been run."""
    out: dict[str, set[str]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out.setdefault(p[0], set()).add(p[1])
    return out


SILENT = load_silent()


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
                "AND (status IS NULL OR status IN " + str(AUTOMATIC) + ")",
                (lg, wav))

        # ⚠ DEFECTIVE AUDIO LAST, so it wins over the scoring labels — a pair whose audio is broken should
        # not sit in the investigate queue as though its IPA were the problem. Judged in seconds-per-phone
        # against THIS language's own median, so a fast-speaking language is not penalised, and at a third
        # of it, which is far outside any speaking-rate variation.
        # ⚠ FOLDED PHONE COUNT, not LENGTH(ipa). SQL's LENGTH counts characters — diacritics, spaces and
        # stress marks included — which is a different scale per language and made this disagree with the
        # analysis that found the defect (608 rows vs 611). The rate has to be seconds per PHONE.
        rate = [(w, (ns / 16000) / len(fold(ipa))) for w, ipa, ns in db.execute(
            "SELECT wav,ipa,n_samples FROM utt WHERE lang=? AND ipa IS NOT NULL AND n_samples>0",
            (lang,)) if ipa and len(fold(ipa)) >= 12]
        if rate:
            med_r = statistics.median(r for _, r in rate)
            for w, r in rate:
                if r < med_r / 3:
                    db.execute(
                        "UPDATE utt SET status='defective_audio', comment=COALESCE(NULLIF(comment,''),?) "
                        "WHERE lang=? AND wav=? AND status IN " + str(AUTOMATIC),
                        ("audio far too short for its text; upstream FLEURS data, not our download",
                         lang, w))

        # ⚠ SILENT AUDIO IS A SECOND, INDEPENDENT DEFECT AND THE RATE TEST ABOVE CANNOT SEE IT. The Welsh
        # files are TRUNCATED, so they fail on seconds-per-phone. The 490 Spanish ones are FULL LENGTH AND
        # EMPTY — a perfectly normal duration containing nothing — so their rate is unremarkable and they
        # score `recognizer_short` instead. Measured from the waveform by scan_silent_audio.py.
        #
        # ⚠ AND IT HAS TO LIVE HERE, not in a one-off UPDATE. `defective_audio` is in AUTOMATIC, so the
        # label was applied by hand once and then silently REVERTED the next time this pass ran — the
        # comment survived, the status did not. A verdict inside an automatic category is only durable if
        # the automatic pass can reproduce it.
        for w in SILENT.get(lang, ()):  # noqa: B007 — set of wav basenames
            db.execute(
                "UPDATE utt SET status='defective_audio', comment=COALESCE(NULLIF(comment,''),?) "
                "WHERE lang=? AND wav=? AND status IN " + str(AUTOMATIC),
                ("silent audio (rms < 1e-4 at full duration); upstream FLEURS data", lang, w))
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
