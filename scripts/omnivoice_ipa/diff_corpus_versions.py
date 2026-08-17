#!/usr/bin/env python3
"""
Diff two *generations* of the same-engine IPA corpus (not two engines).

`compare_ipa_engines.py` answers "where do espeak and vernacula disagree" — it
samples ~1200 utterances per language and ranks them worst-first, because there
the whole corpus differs and the question is which disagreements are systematic.

A version bump is the opposite shape: almost every row is byte-identical and the
interesting set is the handful that moved. So this tool sweeps *every* row, and
reports the changed ones exhaustively — that changed set is what gets hand-read
before a retrain, and what tells us whether the token inventory grew.

  OLD  work/phonemized_v5/byid/<code>.tsv           the corpus the live model trained on
  NEW  work/phonemized_vernacula/byid/<code>.tsv    freshly regenerated

Outputs, under work/ipa_version_diff/ (override with --out):
  summary.tsv          one row per language: changed counts, token retention, symbol delta
  report.md            human-readable roll-up, incl. the corpus-wide new/lost symbol table
  <code>.changed.txt   EVERY changed utterance: source text, old IPA, new IPA
  <code>.subs.tsv      phoneme substitutions, computed over the changed rows only
  <code>.symbols.tsv   per-language symbol inventory delta with counts

`token_retention` is new-corpus IPA whitespace tokens / source orthography tokens,
the Run 28 silent-content-loss detector: a language that drops words scores <1.0
even when every phoneme it *does* emit is correct.

Usage:
  python3 diff_corpus_versions.py                      # all languages
  python3 diff_corpus_versions.py en_us de_de
  python3 diff_corpus_versions.py --old .../byid --new .../byid --out .../dir
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from difflib import SequenceMatcher

from compare_ipa_engines import ROOT, load_ipa, load_text, norm_dist, seg, strip_marks

DEF_OLD = f"{ROOT}/work/phonemized_v5/byid"
DEF_NEW = f"{ROOT}/work/phonemized_vernacula/byid"
DEF_OUT = f"{ROOT}/work/ipa_version_diff"

# Characters that should never reach a phoneme vocabulary. Tracked separately from
# the symbol delta because a *new* leak here is a defect, not a convention change.
PUNCT = set(",.!?;:()[]{}\"'«»„“”‘’—–-/\\|@#&*+=<>~$%^_`")


def analyse(lang: str, old_dir: str, new_dir: str) -> dict | None:
    old, new = load_ipa(f"{old_dir}/{lang}.tsv"), load_ipa(f"{new_dir}/{lang}.tsv")
    if not old and not new:
        return None
    text = load_text(lang)
    common = sorted(set(old) & set(new))

    changed: list[tuple[float, str]] = []
    subs: Counter[tuple[str, str]] = Counter()
    ins: Counter[str] = Counter()
    dele: Counter[str] = Counter()
    old_syms: Counter[str] = Counter()
    new_syms: Counter[str] = Counter()
    src_toks = new_toks = 0

    for k in common:
        o_raw, n_raw = old[k], new[k]
        ou, nu = seg(o_raw), seg(n_raw)
        old_syms.update(u for u in ou if u != " ")
        new_syms.update(u for u in nu if u != " ")
        if k in text:
            src_toks += len(text[k].split())
            new_toks += len(n_raw.split())
        if o_raw == n_raw:
            continue
        changed.append((norm_dist(ou, nu), k))

        # Suprasegmentals stripped so the pairs describe segments, not mark placement.
        oa = [u for u in strip_marks(ou, True, True) if u != " "]
        na = [u for u in strip_marks(nu, True, True) if u != " "]
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, oa, na, autojunk=False).get_opcodes():
            a, b = "".join(oa[i1:i2]), "".join(na[j1:j2])
            if tag == "replace" and len(a) <= 16 and len(b) <= 16:
                subs[(a, b)] += 1
            elif tag == "insert" and len(b) <= 10:
                ins[b] += 1
            elif tag == "delete" and len(a) <= 10:
                dele[a] += 1

    changed.sort(reverse=True)
    return dict(
        lang=lang, n_old=len(old), n_new=len(new), n_common=len(common),
        changed=changed, only_old=sorted(set(old) - set(new)),
        only_new=sorted(set(new) - set(old)),
        subs=subs, ins=ins, dele=dele, old=old, new=new, text=text,
        old_syms=old_syms, new_syms=new_syms,
        retention=(new_toks / src_toks) if src_toks else 0.0,
    )


def write_lang(r: dict, out: str) -> None:
    lang = r["lang"]
    # Every changed row, worst-first — this file is the hand-read queue.
    with open(f"{out}/{lang}.changed.txt", "w", encoding="utf8") as f:
        f.write(f"# {lang} — {len(r['changed'])} of {r['n_common']} utterances changed\n")
        f.write("# worst-first by normalised phoneme distance. txt=source, old=v5, new=current\n\n")
        for d, k in r["changed"]:
            f.write(f"## {k}  dist={d:.3f}\n")
            if k in r["text"]:
                f.write(f"txt {r['text'][k]}\n")
            f.write(f"old {r['old'][k]}\nnew {r['new'][k]}\n\n")

    with open(f"{out}/{lang}.subs.tsv", "w", encoding="utf8") as f:
        f.write("count\told\tnew\n")
        for (a, b), c in r["subs"].most_common(200):
            f.write(f"{c}\t{a}\t{b}\n")
        f.write("\n# insertions (current has, v5 lacks)\n")
        for b, c in r["ins"].most_common(60):
            f.write(f"{c}\t—\t{b}\n")
        f.write("\n# deletions (v5 had, current lacks)\n")
        for a, c in r["dele"].most_common(60):
            f.write(f"{c}\t{a}\t—\n")

    with open(f"{out}/{lang}.symbols.tsv", "w", encoding="utf8") as f:
        f.write("status\tsymbol\tcount_v5\tcount_current\n")
        for s in sorted(set(r["old_syms"]) | set(r["new_syms"])):
            o, n = r["old_syms"][s], r["new_syms"][s]
            status = "NEW" if not o else "LOST" if not n else "both"
            f.write(f"{status}\t{s}\t{o}\t{n}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("langs", nargs="*")
    ap.add_argument("--old", default=DEF_OLD)
    ap.add_argument("--new", default=DEF_NEW)
    ap.add_argument("--out", default=DEF_OUT)
    a = ap.parse_args()

    langs = a.langs or sorted(
        f[:-4] for f in os.listdir(a.new)
        if f.endswith(".tsv") and not f.endswith(".errors.tsv")
    )
    os.makedirs(a.out, exist_ok=True)

    rows, all_new, all_lost = [], Counter(), Counter()
    for lang in langs:
        r = analyse(lang, a.old, a.new)
        if r is None:
            print(f"{lang}: absent from both trees, skipped")
            continue
        write_lang(r, a.out)
        rows.append(r)
        for s, c in r["new_syms"].items():
            if s not in r["old_syms"]:
                all_new[s] += c
        for s, c in r["old_syms"].items():
            if s not in r["new_syms"]:
                all_lost[s] += c
        pct = len(r["changed"]) / r["n_common"] if r["n_common"] else 0.0
        print(
            f"{lang:<14} rows={r['n_common']:<6} changed={len(r['changed']):<6} ({pct:6.1%})  "
            f"retention={r['retention']:.3f}  "
            f"syms +{sum(1 for s in r['new_syms'] if s not in r['old_syms'])}"
            f"/-{sum(1 for s in r['old_syms'] if s not in r['new_syms'])}"
        )

    rows.sort(key=lambda r: -len(r["changed"]) / max(1, r["n_common"]))
    with open(f"{a.out}/summary.tsv", "w", encoding="utf8") as f:
        f.write("lang\trows\tchanged\tchanged_pct\tmean_dist_changed\ttoken_retention\t"
                "syms_new\tsyms_lost\tids_only_v5\tids_only_current\n")
        for r in rows:
            ch = r["changed"]
            md = sum(d for d, _ in ch) / len(ch) if ch else 0.0
            f.write(
                f"{r['lang']}\t{r['n_common']}\t{len(ch)}\t"
                f"{len(ch) / max(1, r['n_common']):.4f}\t{md:.4f}\t{r['retention']:.4f}\t"
                f"{sum(1 for s in r['new_syms'] if s not in r['old_syms'])}\t"
                f"{sum(1 for s in r['old_syms'] if s not in r['new_syms'])}\t"
                f"{len(r['only_old'])}\t{len(r['only_new'])}\n"
            )

    with open(f"{a.out}/report.md", "w", encoding="utf8") as f:
        f.write("# IPA corpus version diff — v5 corpus vs current phonemizer\n\n")
        f.write(f"old: `{a.old}`\nnew: `{a.new}`\n\n")
        f.write("| lang | rows | changed | % | mean dist (changed) | token retention | syms +/- |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            ch = r["changed"]
            md = sum(d for d, _ in ch) / len(ch) if ch else 0.0
            f.write(
                f"| {r['lang']} | {r['n_common']} | {len(ch)} | "
                f"{len(ch) / max(1, r['n_common']):.1%} | {md:.3f} | {r['retention']:.3f} | "
                f"+{sum(1 for s in r['new_syms'] if s not in r['old_syms'])}"
                f"/-{sum(1 for s in r['old_syms'] if s not in r['new_syms'])} |\n"
            )
        # New symbols are the retrain's real risk: tokens the live model never saw.
        f.write("\n## Symbols new corpus-wide (v5 model has never seen these)\n\n")
        for s, c in all_new.most_common():
            flag = "  **PUNCT LEAK**" if any(c2 in PUNCT for c2 in s) else ""
            f.write(f"- `{s}` ×{c}{flag}\n")
        f.write("\n## Symbols lost corpus-wide\n\n")
        for s, c in all_lost.most_common():
            f.write(f"- `{s}` (was ×{c})\n")

    print(f"\nwrote {a.out}/summary.tsv, report.md, per-language changed/subs/symbols")


if __name__ == "__main__":
    main()
