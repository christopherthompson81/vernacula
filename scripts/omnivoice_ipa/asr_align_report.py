#!/usr/bin/env python3
"""
Score our IPA against what the reader actually said, and split the corpus into "leave it alone" and
"investigate".

Reads the table `asr_align_corpus.py` wrote (text · our IPA · recognized phones) and labels every
utterance. The output is a worklist, not a verdict: a mismatch is one of at least three things, and the
report deliberately does not pretend to tell them apart —

  1. READER DIVERGENCE — the transcript is the script the reader was given, not what they said (Run 31).
  2. OUR BUG — the phonemization is wrong, which is what we are hunting.
  3. RECOGNIZER ARTEFACT — wav2vec2 has its own inventory and its own error rate.

⚠ CATEGORY 3 HAS A DEGENERATE MODE THAT MUST BE SPLIT OFF FIRST, or it owns the whole worklist. On some
utterances the recognizer returns almost nothing — a full Welsh sentence came back as the single phone
`k`. Those score a distance near 1.0 and would fill the investigate queue with cases that say nothing
about our IPA. They are classified `recognizer_short` by a length ratio and reported separately, since a
cluster of them in one language is itself a finding (bad audio, or a language the model cannot handle).

⚠ SCORING IS RELATIVE TO EACH LANGUAGE, NEVER ABSOLUTE. The recognizer is an espeak-flavoured multilingual
model: it is systematically closer to some languages than others, and its inventory does not match ours
(we write t̬, ᶦ-offglides, tone letters; it writes ɾ, eɪ, none). An absolute distance threshold would
therefore rank LANGUAGES by how well the recognizer knows them, not utterances by how wrong they are. So
every utterance is scored against its OWN language's median, and what gets flagged is the tail.

Comparison is on a FOLDED phone string — stress, tone, length and the diacritics neither side agrees on
are stripped — because those are exactly where two IPA conventions disagree without either being wrong.

Usage:
  python3 asr_align_report.py                    # all languages present in the db
  python3 asr_align_report.py --langs en_us
  python3 asr_align_report.py --top 40           # investigate-queue size per language
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import statistics
import sys
import unicodedata
from difflib import SequenceMatcher

ROOT = "/mnt/data/omnivoice_ipa"
DB = f"{ROOT}/work/asr_align/align.sqlite"
OUT = f"{ROOT}/work/asr_align"

# Marks the two sides do not agree on, and which do not decide whether a word was READ correctly.
STRIP = set("ˈˌːˑ˥˦˧˨˩꜀꜁꜂꜃꜄꜅꜆꜇ꜛꜜ|‖")
# Punctuation is prosody in our stream (deliberate) but the recognizer never emits it, so it must not
# count as a difference.
PUNCT = set(",.!?;:()[]{}\"'«»„“”‘’—–")


def fold(ipa: str) -> list[str]:
    """IPA string → comparable phone units: the segmental backbone both sides can be judged on.

    ⚠ MODIFIER LETTERS MUST GO TOO, and missing that silently broke two whole languages. `ˠ ʲ ʰ ʷ ᶦ` are
    Unicode category **Lm** with combining class 0, so a `unicodedata.combining()` test keeps them and each
    counts as its own phone. Irish marks velarisation/palatalisation on nearly every consonant (n̪ˠ, sˠ,
    ɾʲ), so its IPA carried about twice the recognizer's phone count and every Irish utterance scored ~0.4
    before correctness entered into it — ga_ie's minimum over 2,845 utterances was 0.371 and its
    "investigate" list came out EMPTY, because when everything is uniformly bad nothing looks like an
    outlier. English `ᶦ`-offglides had the same problem in miniature.

    The recognizer emits none of these marks, so they cannot inform the comparison in either direction."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", ipa):
        if ch in STRIP or ch in PUNCT or ch.isspace():
            continue
        if unicodedata.combining(ch) or unicodedata.category(ch) in ("Lm", "Sk"):
            continue
        # ⚠ AND THE RECOGNIZER'S TONE DIGITS. It writes tone as a trailing number (`siɛ5`, `ŋo5`, `konɡ5`)
        # where we write tone letters (˥˦˧˨˩), which STRIP already removes. Keeping the digits made the
        # comparison asymmetric — we dropped our tone, it kept its own — and every tonal utterance carried
        # a fixed penalty for it. vi_vn's median was 0.611 with an EMPTY investigate list, the same
        # everything-is-uniformly-bad degeneracy the modifier letters caused for Irish.
        if ch.isdigit():
            continue
        out.append(ch)
    return out


def dist(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - SequenceMatcher(None, a, b, autojunk=False).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--top", type=int, default=30, help="investigate rows kept per language")
    a = ap.parse_args()

    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    langs = a.langs or [r[0] for r in db.execute("SELECT DISTINCT lang FROM utt ORDER BY lang")]
    os.makedirs(a.out, exist_ok=True)

    summary = []
    queue_path = f"{a.out}/investigate.tsv"
    short_path = f"{a.out}/recognizer_short.tsv"
    with open(queue_path, "w", encoding="utf8") as q, open(short_path, "w", encoding="utf8") as sh:
        q.write("lang\tsentence_id\twav\tz\tdist\tmedian\ttext\tipa\tphones\n")
        sh.write("lang\tsentence_id\twav\tn_ipa\tn_heard\ttext\tphones\n")
        for lang in langs:
            rows = list(db.execute(
                "SELECT sentence_id,wav,text,ipa,phones FROM utt "
                "WHERE lang=? AND ipa IS NOT NULL AND phones IS NOT NULL AND phones!=''", (lang,)))
            if len(rows) < 20:
                print(f"{lang}: {len(rows)} usable rows, skipped", file=sys.stderr)
                continue
            scored, short = [], []
            for sid, wav, txt, ipa, ph in rows:
                fi, fp = fold(ipa), fold(ph)
                # The recognizer produced far too little to be compared with. 0.35 is well below any
                # convention difference: two IPA transcriptions of the same utterance do not differ
                # threefold in phone count.
                if len(fi) >= 12 and len(fp) < 0.35 * len(fi):
                    short.append((sid, wav, txt, ipa, ph, len(fi), len(fp)))
                    continue
                scored.append((dist(fi, fp), sid, wav, txt, ipa, ph))
            if len(scored) < 20:
                print(f"{lang}: only {len(scored)} comparable rows, skipped", file=sys.stderr)
                continue
            ds = [s[0] for s in scored]
            med = statistics.median(ds)
            # MAD, not stdev: the tail we are hunting is exactly what would inflate stdev and hide itself.
            mad = statistics.median([abs(d - med) for d in ds]) or 1e-9
            scored.sort(key=lambda s: -s[0])
            worst = scored[: a.top]
            for d, sid, wav, txt, ipa, ph in worst:
                z = 0.6745 * (d - med) / mad
                q.write(f"{lang}\t{sid}\t{wav}\t{z:.2f}\t{d:.3f}\t{med:.3f}\t"
                        f"{(txt or '')[:160]}\t{(ipa or '')[:160]}\t{(ph or '')[:160]}\n")
            # "Good" = within the bulk of this language's own distribution.
            good = sum(1 for d in ds if 0.6745 * (d - med) / mad <= 3.0)
            summary.append((lang, len(rows), len(short), med, statistics.mean(ds), good,
                            len(scored) - good))
            for sid, wav, txt, ipa, ph, ni, np in short:
                sh.write(f"{lang}\t{sid}\t{wav}\t{ni}\t{np}\t{(txt or '')[:120]}\t{(ph or '')[:80]}\n")
            print(f"{lang:<14} n={len(rows):<5} short={len(short):<4} median={med:.3f} "
                  f"within-3MAD={good} ({100*good/len(scored):.1f}%)  investigate={len(scored)-good}",
                  file=sys.stderr)

    with open(f"{a.out}/summary.tsv", "w", encoding="utf8") as f:
        f.write("lang\tn\trecognizer_short\tmedian_dist\tmean_dist\twithin_3mad\tinvestigate\n")
        for lang, n, nsh, med, mean, good, tail in sorted(summary, key=lambda r: r[3]):
            f.write(f"{lang}\t{n}\t{nsh}\t{med:.4f}\t{mean:.4f}\t{good}\t{tail}\n")

    tot = sum(r[1] for r in summary)
    good = sum(r[5] for r in summary)
    print(f"\n{tot} utterances scored; {good} ({100*good/max(tot,1):.1f}%) inside their language's bulk, "
          f"{tot-good} in the tail", file=sys.stderr)
    print(f"wrote {a.out}/summary.tsv, {queue_path}, {short_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
