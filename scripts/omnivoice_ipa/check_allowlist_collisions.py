#!/usr/bin/env python3
"""
⚠ THE ALLOWLIST IS GLOBAL. THE EVIDENCE FOR IT WAS NOT.

`restoreInitialismCasing()` takes no language argument — every token in `INITIALISM_UPPERCASE` is
uppercased in ALL 28 corpora. But each token was added on evidence from ONE language, usually
English. A token that is an initialism in English can be an ordinary, frequent WORD somewhere else,
and uppercasing it there turns a word into a spelled-out letter run.

This is not hypothetical. `un` was added from English evidence ("waste from the UN camp", 6 rows) and
is the indefinite article in French (1,050), Spanish (847) and Catalan (778), "one" in Welsh (297),
and "flour" in Turkish (44) — 3,064 standalone occurrences across 15 languages against the 6 it was
added for.

It turned out FINE, and the reason is worth knowing: those languages read `UN` as the word regardless
of case, so the repair is inert there. But that is LUCK, not design — it depends on which engines
happen to have a capital-keyed initialism rule. The next token added this way may not be so lucky.

So: for every allowlist token, report where it occurs as a standalone word in a corpus OTHER than the
one it was justified by, AND whether that language's phonemization actually changes under
uppercasing. The second half is what separates a real collision from an inert one.

  python3 check_allowlist_collisions.py            # counts only (fast, no phonemizer)
  python3 check_allowlist_collisions.py --resolve  # emit the token list for the tsx probe

⚠ COUNTS ALONE ARE NOT A VERDICT. A high count in another language is a QUESTION — is that token the
foreign initialism there too? For `un` it was: xh `be-un`, zu `le-un`, ko `un 회원국`, sd/th, all
United Nations, all correctly spelled out, all confirmed against the audio (the recognizer heard
`j u e n`). Read the contexts before removing anything.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

DB = "/mnt/data/omnivoice_ipa/work/asr_align/align.sqlite"
ALLOWLIST = Path(__file__).with_name("initialism_casing.mts")


def allowlist_tokens() -> list[str]:
    """The INITIALISM_UPPERCASE entries, read straight out of the .mts."""
    src = ALLOWLIST.read_text(encoding="utf-8")
    m = re.search(r"INITIALISM_UPPERCASE[^=]*=\s*\[(.*?)\]", src, re.S)
    if not m:
        sys.exit("could not find INITIALISM_UPPERCASE")
    body = re.sub(r"//[^\n]*", "", m.group(1))  # strip the review comments
    return re.findall(r'"([^"]+)"', body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--min", type=int, default=3, help="report a language at or above this count")
    ap.add_argument("--resolve", action="store_true", help="print colliding tokens, one per line")
    a = ap.parse_args()

    tokens = allowlist_tokens()
    db = sqlite3.connect(a.db)
    langs = [r[0] for r in db.execute("SELECT DISTINCT lang FROM utt ORDER BY lang")]
    texts = {L: [t or "" for t, in db.execute("SELECT text FROM utt WHERE lang=?", (L,))]
             for L in langs}

    print(f"# {len(tokens)} allowlist tokens vs {len(langs)} corpora "
          f"(standalone occurrences, >= {a.min})\n")
    collided = []
    for tok in sorted(tokens):
        pat = re.compile(rf"(?<![\w]){re.escape(tok)}(?![\w])", re.I)
        hits = []
        for L in langs:
            n = sum(len(pat.findall(t)) for t in texts[L])
            if n >= a.min:
                hits.append((L, n))
        if len(hits) > 1:  # present in more than one corpus = worth a look
            collided.append(tok)
            hits.sort(key=lambda x: -x[1])
            total = sum(n for _, n in hits)
            shown = "  ".join(f"{L}={n}" for L, n in hits[:8])
            print(f"{tok:<8} {total:>5} across {len(hits):>2} langs   {shown}")

    print(f"\n{len(collided)} of {len(tokens)} tokens appear in more than one corpus.")
    print("⚠ A hit is a QUESTION, not a verdict — read the contexts, and check whether that")
    print("  language's output actually CHANGES under uppercasing before acting.")
    if a.resolve:
        print("\n# colliding tokens:")
        for t in collided:
            print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
