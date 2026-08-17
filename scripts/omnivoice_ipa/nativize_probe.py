#!/usr/bin/env python3
"""
Does "route to English, then fold the result into the host's inventory" actually work?

The proposal under test: a delegated foreign word is read by English, and the resulting IPA is then
NATIVISED by replacing every phone the host language does not have with its nearest neighbour in
articulatory-feature space. That is a phone-level layer the repo does not have — `core/latinPhones.ts`
folds LETTERS the g2p cannot read, and Japanese carries its adaptation in the katakana orthography, but
neither maps an IPA string onto another language's inventory.

Two things make it testable here rather than a matter of taste:

  · THE INVENTORY IS FREE. A language's phone inventory is exactly the set of phones our own phonemizer
    emits for it across the corpus. No new data to author, and it is by construction the inventory the
    TTS model is being trained on — which is the one that matters, not a textbook's.
  · THE ANSWER IS IN THE AUDIO. The wav2vec2 pass gives recognized phones for utterances that contain a
    delegated English word, so a candidate nativisation can be scored against what the reader said.

What this deliberately does NOT do is phonotactic repair — epenthesis, coda restriction, the vowel-final
requirement Nguni has. Substitution alone cannot turn `microsoft` into Japanese *maikurosofuto*. Measuring
substitution on its own is the point: it says how much of the gap the cheap half closes.

Usage:
  python3 nativize_probe.py --lang am_et
  python3 nativize_probe.py --lang am_et --limit 40
"""
from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import sys
import unicodedata

ROOT = "/mnt/data/omnivoice_ipa"
BYID = f"{ROOT}/work/phonemized_vernacula/byid"
DB = f"{ROOT}/work/asr_align/align.sqlite"

STRIP = set("ˈˌ|‖")
PUNCT = set(",.!?;:()[]{}\"'«»„“”‘’—–")
# English-only phones: the marker that a delegated run went through untouched. Kept deliberately small
# and uncontroversial — each is absent from most of the hosts we delegate from.
ENGLISH_ONLY = ("ɹ", "ɚ", "ɝ", "θ", "ð", "æ", "ʌ", "ɫ", "oᶷ", "eᶦ", "aᶦ", "aᶷ", "t̬", "d̬")


def seg(ipa: str) -> list[str]:
    """IPA → phone units: a base plus its combining marks / modifiers, tie bars held together."""
    out: list[str] = []
    i, n = 0, len(ipa)
    while i < n:
        ch = ipa[i]
        if ch.isspace() or ch in STRIP or ch in PUNCT:
            i += 1
            continue
        unit = ch
        i += 1
        while i < n:
            c = ipa[i]
            if c in ("͡", "͜"):
                unit += c
                i += 1
                if i < n:
                    unit += ipa[i]
                    i += 1
            elif unicodedata.category(c) in ("Mn", "Me", "Sk", "Lm") or c in "ːˑ":
                unit += c
                i += 1
            else:
                break
        out.append(unit)
    return out


def inventory(lang: str, min_count: int = 5) -> collections.Counter:
    """The phones this language actually emits — counted ONLY over utterances whose source text carries no
    Latin run.

    ⚠ THIS EXCLUSION IS THE WHOLE EXPERIMENT. Counting the full corpus makes the inventory contain the very
    English phones we are trying to fold away, because the delegated words are IN the corpus: the first
    version of this probe mapped ɹ->ɹ at distance 0.00 (55 occurrences, all of them delegated) and every
    other English-only phone to itself, so the "nativisation" was the identity and the measurement said
    nothing. A frequency floor does not save it — 55 is well clear of any floor worth setting. The
    inventory has to come from material the host actually produced."""
    text: dict[str, str] = {}
    with open(f"{ROOT}/corpus/fleurs_transcripts/data/{lang}/train.tsv", encoding="utf8") as f:
        for line in f:
            c0 = line.rstrip("\n").split("\t")
            if len(c0) >= 4 and c0[3].strip():
                text.setdefault(c0[0], c0[3])
    latin = re.compile(r"[A-Za-z]{2,}")
    c: collections.Counter = collections.Counter()
    with open(f"{BYID}/{lang}.tsv", encoding="utf8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            src = text.get(k)
            if src is None or latin.search(src):
                continue
            c.update(seg(v))
    return collections.Counter({k: n for k, n in c.items() if n >= min_count})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="am_et")
    ap.add_argument("--limit", type=int, default=25)
    a = ap.parse_args()

    import panphon.distance

    dist = panphon.distance.Distance()
    inv = inventory(a.lang)
    print(f"{a.lang}: inventory of {len(inv)} phones (>=5 occurrences)", file=sys.stderr)

    # ⚠ FEATURE DISTANCE ALONE PICKS THE WRONG NEIGHBOUR FOR RHOTICS. panphon scores English /ɹ/ closer to
    # /j/ than to /ɾ/ — both are approximants while a tap is not — which is articulatorily defensible and
    # perceptually wrong: loanword adaptation sends /ɹ/ to the host's rhotic almost universally. So the
    # search is constrained to the source phone's major class first, and only falls back to the open field
    # when the host has nothing in that class. This is the general shape of the finding: a raw feature
    # metric needs class guidance to model borrowing.
    CLASS = {
        **{p: "rhotic" for p in "ɹɾrɽʀʁɻ"},
        **{p: "lateral" for p in "lɫʎɭ"},
        **{p: "vowel" for p in "aeiouəɛɔæʌɪʊɑɒøyɯɨʉɜɐ"},
    }

    def klass(p: str) -> str:
        return CLASS.get(p[0], "other")

    targets = [p for p in inv if p.strip()]
    mapping: dict[str, tuple[str, float]] = {}
    for src in ENGLISH_ONLY:
        same = [t for t in targets if klass(t) == klass(src)] if klass(src) != "other" else []
        pool = same or targets
        scored_t = []
        for t in pool:
            try:
                scored_t.append((dist.weighted_feature_edit_distance(src, t), t))
            except Exception:
                continue
        if not scored_t:
            continue
        scored_t.sort()
        # ⚠ A FREQUENCY PRIOR, not just a tiebreak. Constrained to the rhotics, the metric preferred
        # Amharic's LABIALIZED tap ɾʷ (66 occurrences) over plain ɾ (tens of thousands), because it scored
        # marginally closer. A borrowing lands on the language's ordinary phone, not on a rare conditioned
        # variant, so among everything within a band of the best distance take the most frequent. The band has to be TIGHT: at 0.5 it let ð->s (0.75, 11292x) beat ð->z
        # (0.50, 3444x), i.e. frequency overrode a strictly better match.
        band = scored_t[0][0] + 0.15
        best = max((t for d, t in scored_t if d <= band), key=lambda t: inv[t])
        mapping[src] = (best, dict((t, d) for d, t in scored_t)[best])

    print(f"\n{a.lang} nearest-neighbour map for English-only phones:", file=sys.stderr)
    for src, (tgt, d) in sorted(mapping.items(), key=lambda kv: kv[1][1]):
        print(f"    {src:<3} -> {tgt:<4} (d={d:.2f}, target occurs {inv[tgt]}x)", file=sys.stderr)

    def nativise(ipa: str) -> str:
        return " ".join(mapping.get(p, (p, 0))[0] for p in seg(ipa))

    # Score against the audio: utterances whose IPA still carries English-only phones.
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = [r for r in db.execute(
        "SELECT sentence_id,text,ipa,phones FROM utt WHERE lang=? AND ipa IS NOT NULL AND phones!=''",
        (a.lang,)) if any(p in r[2] for p in ENGLISH_ONLY)]
    print(f"\n{len(rows)} utterances carry a delegated English phone; scoring {min(len(rows), a.limit)}",
          file=sys.stderr)

    from difflib import SequenceMatcher

    def d2(x: str, y: str) -> float:
        A, B = seg(x), seg(y)
        return 1.0 - SequenceMatcher(None, A, B, autojunk=False).ratio() if A and B else 1.0

    before = after = 0.0
    shown = 0
    for sid, txt, ipa, ph in rows[: a.limit]:
        b, n = d2(ipa, ph), d2(nativise(ipa), ph)
        before += b
        after += n
        if shown < 3:
            shown += 1
            print(f"\n  #{sid}  dist {b:.3f} -> {n:.3f}", file=sys.stderr)
            print(f"    ours    : {ipa[:96]}", file=sys.stderr)
            print(f"    nativised: {nativise(ipa)[:96]}", file=sys.stderr)
            print(f"    heard   : {ph[:96]}", file=sys.stderr)
    k = min(len(rows), a.limit)
    if k:
        print(f"\nmean distance to the audio: {before/k:.4f} -> {after/k:.4f} "
              f"({'BETTER' if after < before else 'WORSE'})", file=sys.stderr)


if __name__ == "__main__":
    main()
