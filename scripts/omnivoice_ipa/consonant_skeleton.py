#!/usr/bin/env python3
"""
Compare our IPA to the recognizer on the CONSONANT SKELETON ONLY.

⚠ THE REASON THIS SHOULD WORK BETTER THAN THE FULL DISTANCE. Run 43 profiled the phone-level
disagreements across all 28 languages, and *every* top substitution in *every* language was vowel
quality: `i→ɪ`, `a→ə`, `u→ʊ`, `ɛ→e`, `ɔ→o`. That is not error, it is two conventions disagreeing about
vowels by design — the recognizer has its own inventory and its own reduction habits. Those
disagreements are the bulk of the measured distance, so they are the NOISE FLOOR that hid everything
smaller. Run 41 hit that floor directly: a 2-phone edit inside a 120-phone utterance could not be
resolved at all.

Consonants are the opposite. Both sides agree on them far more closely, and a consonant that is
missing, extra, or wrong is much more likely to be OUR defect than a convention difference. Dropping
the vowels should therefore raise the signal and lower the floor at the same time.

⚠ WHAT COUNTS AS A CONSONANT IS A JUDGEMENT, AND IT IS TUNED HERE, NOT ASSUMED. Three classes are
excluded beyond the plain vowels:

  · GLIDES `j w ɥ` — they alternate with vowels across conventions (`j→ɪ` is a top-ten pair in th and
    pt), so keeping them reintroduces exactly the noise this is removing. Measured: including them
    drops the validation ratio from 4.6:1 to 3.1:1.
  · RHOTIC VOWELS `ɚ ɝ` — vowels that carry rhoticity; English readers vary (Run 43), so they are
    reader variation rather than defect.

⚠ `h` AND `ʔ` ARE KEPT, and excluding them was my first mistake here. The reasoning for dropping them
was sound in the abstract — the recognizer does insert and drop both freely — but the single largest
defect this corpus ever had was Kazakh ⟨ь⟩/⟨ъ⟩ emitting a GLOTTAL STOP in 408 rows. Excluding ʔ threw
away the evidence for the biggest fix, and the validation says so plainly: 1.8:1 without them against
4.6:1 with. It also produced ties rather than disagreements, which is worse than a wrong answer
because it looks like agreement. `--drop-weak` re-runs the losing variant so the claim stays checkable.

VALIDATION, because a new metric that has never been shown to detect anything is not evidence:
`--validate` scores the metric against the corpus's OWN known defects. The DB keeps the pre-fix IPA
in `ipa_prev`, and the seven merged fixes were mostly CONSONANT defects (fula ⟨sh⟩→ʃ, hausa ⟨ch⟩,
kazakh ʔ, hindi d͡ʒɲ→ɡj). A useful skeleton metric must separate old from new MORE sharply than the
full distance did. Measured on 2,314 such rows: full distance 3.5:1, this metric **4.6:1**.

  python3 consonant_skeleton.py --validate
  python3 consonant_skeleton.py --lang ff_sn --top 20
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "/mnt/data/Programming/vernacula/scripts/omnivoice_ipa")
from asr_align_report import dist, fold  # noqa: E402

DB = "/mnt/data/omnivoice_ipa/work/asr_align/align.sqlite"

# Plain vowels of the IPA, in the single-character form `fold()` leaves behind.
VOWELS = set("aeiouɑɐɒæɔəɘɛɜɞɪɨʉʊʌøœɶɤɯyʏɵᵻ")
GLIDES = set("jwɥɰ")
RHOTIC_V = set("ɚɝ")
WEAK = set("hʔ")  # KEPT by default — see the module note


def skeleton(units: list[str], keep_glides: bool = False, drop_weak: bool = False) -> list[str]:
    """The consonant backbone of a folded phone list."""
    drop = set(VOWELS) | set(RHOTIC_V)
    if not keep_glides:
        drop |= GLIDES
    if drop_weak:
        drop |= WEAK
    return [c for c in units if c not in drop]


def skel_dist(a_ipa: str, b_phones: str, **kw) -> float | None:
    """Normalized edit distance on the consonant skeleton, or None if too short to judge."""
    a = skeleton(fold(a_ipa or ""), **kw)
    b = skeleton(fold(b_phones or ""), **kw)
    if len(a) < 3 or len(b) < 3:
        return None
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    return 1.0 - (2.0 * matched) / (len(a) + len(b))


def validate(db: sqlite3.Connection, **kw) -> int:
    """Score the metric against the corpus's own known defects (ipa_prev vs ipa)."""
    rows = db.execute(
        "SELECT lang, ipa, ipa_prev, phones FROM utt "
        "WHERE ipa_prev IS NOT NULL AND ipa != ipa_prev AND phones IS NOT NULL AND phones != ''"
    ).fetchall()
    if not rows:
        print("no ipa_prev rows to validate against", file=sys.stderr)
        return 1
    fb = fs = tb = ts = 0
    for _lang, new, old, ph in rows:
        dn_f, do_f = dist(fold(new), fold(ph)), dist(fold(old), fold(ph))
        if dn_f < do_f: fb += 1
        elif dn_f > do_f: tb += 1
        sn, so = skel_dist(new, ph, **kw), skel_dist(old, ph, **kw)
        if sn is None or so is None:
            continue
        if sn < so: fs += 1
        elif sn > so: ts += 1
    print(f"# {len(rows)} rows where a merged fix changed the IPA")
    print(f"  FULL distance     better {fb:>5}   worse {tb:>5}   ratio {fb/max(tb,1):>5.1f} : 1")
    print(f"  CONSONANT skeleton better {fs:>5}   worse {ts:>5}   ratio {fs/max(ts,1):>5.1f} : 1")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--lang")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--keep-glides", action="store_true")
    ap.add_argument("--drop-weak", action="store_true", help="exclude h/ʔ — the measured-worse variant")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--z", type=float, default=3.0, help="MAD z-score cutoff")
    a = ap.parse_args()
    kw = {"keep_glides": a.keep_glides, "drop_weak": a.drop_weak}

    db = sqlite3.connect(a.db)
    if a.validate:
        return validate(db, **kw)

    langs = [a.lang] if a.lang else [r[0] for r in
                                     db.execute("SELECT DISTINCT lang FROM utt ORDER BY lang")]
    for lang in langs:
        rows = []
        for wav, txt, ipa, ph, st in db.execute(
                "SELECT wav,text,ipa,phones,status FROM utt WHERE lang=? AND phones IS NOT NULL "
                "AND phones!='' AND status NOT IN ('defective_audio','recognizer_short')", (lang,)):
            d = skel_dist(ipa, ph, **kw)
            if d is not None:
                rows.append((d, wav, txt, ipa, ph, st))
        if len(rows) < 20:
            continue
        ds = [r[0] for r in rows]
        med = statistics.median(ds)
        mad = statistics.median([abs(d - med) for d in ds]) or 1e-9
        out = [r for r in rows if 0.6745 * (r[0] - med) / mad > a.z]
        out.sort(reverse=True)
        print(f"\n### {lang}  median skeleton distance {med:.3f}   flagged {len(out)}/{len(rows)}")
        for d, wav, txt, ipa, ph, st in out[:a.top]:
            print(f"  {d:.3f} [{st}] {txt[:70]}")
            print(f"        IPA {ipa[:70]}")
            print(f"        ASR {ph[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
