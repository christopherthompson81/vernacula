#!/usr/bin/env python3
"""
Digraph-coverage audit for the rule-table g2p engines.

⚠ THE DEFECT THIS EXISTS TO CATCH. A longest-match rule scan reads the longest key that matches and
then advances. If the orthography has a digraph the table does not list, the scan silently reads it
as its two PARTS -- and because both parts usually have rules, nothing is dropped, nothing errors,
and the output is a plausible-looking IPA string containing an impossible sequence. Fulfulde <ch>
read as t͡ʃ + h in 514 of 530 corpus rows this way, and <sh> came out as the LITERAL two letters
"sh" sitting in an IPA stream, in all 280 rows containing it. Neither was visible to any
distance-based check; both were found by reading the IPA column by eye.

So: enumerate the letter sequences the CORPUS actually contains, subtract the ones the table already
has a rule for, and rank what is left by frequency. The output is a review queue, not a verdict --
most surviving bigrams are ordinary letter adjacency (`ta`, `an`), and the reader is looking for the
ones that are a DIGRAPH of the language: typically C+h, C+y, C+w, or a doubled consonant.

Only the five jsonc "rules" engines are auditable this way (fula hausa hungarian xhosa zulu); the
other ~177 engines carry their mapping in code and need a different gate.

  python3 audit_digraph_coverage.py --lang ff_sn --engine fula
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PHONEMIZER = Path("/home/chris/Programming/vernacula-phonemizer/src/languages")
DB = "/mnt/data/omnivoice_ipa/work/asr_align/align.sqlite"

# corpus lang code -> engine directory
ENGINES = {"ff_sn": "fula", "ha_ng": "hausa", "xh_za": "xhosa", "zu_za": "zulu"}

# ⚠ A sequence is INTERESTING only if it looks like a DIGRAPH rather than incidental letter adjacency.
# The first cut used "second letter in h y w j g n" and drowned the signal: `an` ×14139, `in` ×8831,
# `iy`, `ay`, `aw` are just vowels next to sonorants. A digraph in these orthographies is
# CONSONANT + a modifier letter, or a doubled consonant -- so require the first letter to be a
# consonant too. That is what separates `sh`/`ch`/`kh` from `ah`/`oy`.
VOWELS = set("aeiou")
DIGRAPH_TAIL = set("hywj'")


def strip_jsonc(src: str) -> str:
    """Remove // comments and trailing commas so json can parse it."""
    out, i, n = [], 0, len(src)
    in_str = False
    while i < n:
        c = src[i]
        if in_str:
            out.append(c)
            if c == "\\":
                if i + 1 < n:
                    out.append(src[i + 1])
                    i += 2
                    continue
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    txt = "".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", txt)


def rule_keys(engine: str) -> tuple[set[str], set[str]]:
    """(all rule keys, keys of length >= 2) for an engine's jsonc table."""
    files = list((PHONEMIZER / engine).glob("*.jsonc"))
    if not files:
        sys.exit(f"no jsonc for engine {engine}")
    data = json.loads(strip_jsonc(files[0].read_text(encoding="utf-8")))
    if "rules" not in data:
        sys.exit(f"{files[0]} has no 'rules' block")
    keys = {r[0].lower() for r in data["rules"]}
    return keys, {k for k in keys if len(k) >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--max-n", type=int, default=3, help="longest sequence to consider")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="do not filter to digraph-shaped sequences")
    a = ap.parse_args()

    engine = a.engine or ENGINES.get(a.lang)
    if engine is None:
        sys.exit(f"no engine known for {a.lang}; pass --engine")
    keys, multi = rule_keys(engine)

    db = sqlite3.connect(a.db)
    rows = [r[0] for r in db.execute("SELECT text FROM utt WHERE lang=?", (a.lang,))]
    if not rows:
        sys.exit(f"no rows for {a.lang}")

    # count letter sequences inside words only (digits/punctuation are the number tier's business)
    word_re = re.compile(r"[^\W\d_]+", re.UNICODE)
    seqs: dict[int, Counter[str]] = {n: Counter() for n in range(2, a.max_n + 1)}
    words: Counter[str] = Counter()
    for text in rows:
        for w in word_re.findall(text.lower()):
            words[w] += 1
            for n in range(2, a.max_n + 1):
                for i in range(len(w) - n + 1):
                    seqs[n][w[i : i + n]] += 1

    print(f"# {a.lang} via engine '{engine}': {len(rows)} rows, {sum(words.values())} word tokens")
    print(f"# rule table has {len(keys)} keys, {len(multi)} of them multi-character")
    print(f"# listing sequences with NO rule, count >= {a.min_count}"
          + ("" if a.all else ", filtered to digraph-shaped (2nd/3rd char in h y w j g n ')"))
    print()

    for n in range(2, a.max_n + 1):
        uncovered = []
        for seq, c in seqs[n].most_common():
            if c < a.min_count or seq in keys:
                continue
            # ⚠ A multigraph spells ONE sound, so it contains no vowel letter anywhere -- the first cut
            # allowed any consonant-initial sequence ending in a modifier and the 3-grams filled with
            # `way`, `ley`, `daw`, `kuy`: a consonant, a vowel and a glide, which is three sounds and
            # not a trigraph. Requiring the whole run to be consonants is what leaves `tyh`, `ntsh`,
            # `cch` standing.
            if not a.all:
                if any(ch in VOWELS for ch in seq):
                    continue
                modifier = seq[-1] in DIGRAPH_TAIL
                doubled = seq[0] == seq[1]
                if not (modifier or doubled):
                    continue
            # a sequence whose parts are all single-letter rules is the dangerous kind: it scans
            # cleanly into pieces and produces no error anywhere.
            parts_known = all(ch in keys for ch in seq)
            uncovered.append((seq, c, parts_known))
        if not uncovered:
            continue
        print(f"## {n}-grams with no rule ({len(uncovered)})")
        for seq, c, pk in uncovered[:40]:
            ex = [w for w in words if seq in w][:4]
            flag = "SCANS-CLEAN" if pk else "has-gap    "
            print(f"  {seq:<5} {c:>6}  {flag}  {' '.join(ex)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
