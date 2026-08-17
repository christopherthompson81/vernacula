#!/usr/bin/env python3
"""
Compare the two IPA transcriptions of the FLEURS corpus:

  OLD  work/phonemized/byid/<code>.tsv            espeak-ng-portable, canonical IPA mode
  NEW  work/phonemized_vernacula/byid/<code>.tsv  vernacula-phonemizer, phonemizeAsync

The question this answers is *not* "which is right" — espeak is not ground truth.
It is "where do the two engines disagree, and is the disagreement systematic?".
A systematic, high-frequency disagreement is a candidate refinement for
vernacula-phonemizer; a scattered one is usually just a defensible convention
difference (stress placement, syllable boundaries, tie-bar usage).

Outputs, under work/ipa_engine_diff/:
  summary.tsv            one row per language, the headline metrics
  <code>.subs.tsv        top phoneme-level substitutions (espeak → vernacula)
  <code>.words.tsv       top word-level disagreements w/ the source orthography
  <code>.samples.txt     side-by-side utterances, worst-first
  report.md              human-readable roll-up

Usage:
  python3 compare_ipa_engines.py                # all languages
  python3 compare_ipa_engines.py en_us hi_in    # a subset
  python3 compare_ipa_engines.py --sample 1500  # utterances sampled per language
"""
from __future__ import annotations

import os
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

ROOT = "/mnt/data/omnivoice_ipa"
TSV = f"{ROOT}/corpus/fleurs_transcripts/data"
OLD = f"{ROOT}/work/phonemized/byid"
NEW = f"{ROOT}/work/phonemized_vernacula/byid"
OUT = f"{ROOT}/work/ipa_engine_diff"

# FLEURS codes whose NEW run used a closer-matching variety than the base code
# espeak was given, so part of their diff is dialect, not engine disagreement.
# (pt_br was here too until Run 26: espeak now also runs its own pt-br, so both
# sides speak Brazilian and the mismatch note no longer applies.)
VARIETY_MISMATCH = {"ar_eg": "arz vs ar", "es_419": "es-419 vs es"}

# Suprasegmentals we strip to separate "different segments" from "different stress/tone marking".
STRESS = set("ˈˌ")
TONE = set("˥˦˧˨˩꜀꜁꜂꜃꜄꜅꜆꜇ꜛꜜ")
LENGTH = set("ːˑ")


def seg(ipa: str) -> list[str]:
    """Split an IPA string into phoneme-ish units: a base character plus any
    following combining marks / modifier letters, with tie-barred pairs (t͡ʃ)
    held together as one unit."""
    out: list[str] = []
    i, n = 0, len(ipa)
    while i < n:
        ch = ipa[i]
        if ch == " ":
            out.append(" ")
            i += 1
            continue
        unit = ch
        i += 1
        while i < n:
            c = ipa[i]
            cat = unicodedata.category(c)
            # Tie bar MUST be tested before the Mn branch — U+0361/U+035C are
            # themselves Mn, so the generic branch would swallow the bar and
            # leave the second half of the affricate as its own unit.
            if c in ("͡", "͜"):  # tie bar: absorb it and the next base
                unit += c
                i += 1
                if i < n:
                    unit += ipa[i]
                    i += 1
            elif cat in ("Mn", "Me", "Sk", "Lm") or c in LENGTH or c in TONE:
                unit += c
                i += 1
            else:
                break
        out.append(unit)
    return out


def strip_marks(units: list[str], stress: bool, tone: bool) -> list[str]:
    out = []
    for u in units:
        if stress:
            u = "".join(c for c in u if c not in STRESS)
        if tone:
            u = "".join(c for c in u if c not in TONE)
        if u:
            out.append(u)
    return out


def load_ipa(path: str) -> dict[str, str]:
    d: dict[str, str] = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            k, _, v = line.partition("\t")
            if v:
                d.setdefault(k, v)
    return d


def load_text(lang: str) -> dict[str, str]:
    """id → source orthography (col3), first occurrence wins."""
    d: dict[str, str] = {}
    p = f"{TSV}/{lang}/train.tsv"
    if not os.path.exists(p):
        return d
    with open(p, encoding="utf8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[3].strip():
                d.setdefault(c[0], c[3].strip())
    return d


def norm_dist(a: list[str], b: list[str]) -> float:
    """1 - similarity ratio over phoneme units; 0.0 = identical."""
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b, autojunk=False).ratio()


def ctx(units: list[str], i: int, j: int, w: int = 2) -> str:
    left = "".join(units[max(0, i - w):i])
    right = "".join(units[j:j + w])
    return f"{left}_{right}".strip() or "—"


def analyse(lang: str, sample: int) -> dict | None:
    old, new = load_ipa(f"{OLD}/{lang}.tsv"), load_ipa(f"{NEW}/{lang}.tsv")
    common = sorted(set(old) & set(new))
    if not common:
        return None
    text = load_text(lang)

    # Deterministic, spread-out sample (no RNG, so reruns are comparable).
    keys = common if len(common) <= sample else [
        common[round(i * (len(common) - 1) / (sample - 1))] for i in range(sample)
    ]
    keys = list(dict.fromkeys(keys))

    subs: Counter[tuple[str, str]] = Counter()
    subs_ctx: dict[tuple[str, str], Counter[str]] = {}
    ins: Counter[str] = Counter()
    dele: Counter[str] = Counter()
    words: Counter[tuple[str, str, str]] = Counter()
    dists: list[tuple[float, str]] = []
    seg_dists: list[float] = []
    identical = tok_match = 0

    old_syms: Counter[str] = Counter()
    new_syms: Counter[str] = Counter()

    for k in keys:
        o_raw, n_raw = old[k], new[k]
        if o_raw == n_raw:
            identical += 1
        ou, nu = seg(o_raw), seg(n_raw)
        old_syms.update(u for u in ou if u != " ")
        new_syms.update(u for u in nu if u != " ")

        d = norm_dist(ou, nu)
        dists.append((d, k))
        # Segments-only distance: stress + tone stripped.
        seg_dists.append(
            norm_dist(strip_marks(ou, True, True), strip_marks(nu, True, True))
        )

        # Phoneme-level opcodes, suprasegmentals stripped so the pairs are about
        # segments rather than about where each engine puts a stress mark.
        oa = [u for u in strip_marks(ou, True, True) if u != " "]
        na = [u for u in strip_marks(nu, True, True) if u != " "]
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, oa, na, autojunk=False).get_opcodes():
            if tag == "replace":
                a, b = "".join(oa[i1:i2]), "".join(na[j1:j2])
                if len(a) <= 12 and len(b) <= 12:
                    subs[(a, b)] += 1
                    subs_ctx.setdefault((a, b), Counter())[ctx(oa, i1, i2)] += 1
            elif tag == "insert":
                b = "".join(na[j1:j2])
                if len(b) <= 8:
                    ins[b] += 1
            elif tag == "delete":
                a = "".join(oa[i1:i2])
                if len(a) <= 8:
                    dele[a] += 1

        # Word-level pairs, only when both engines produced the same word count
        # AND that matches the source token count (so the mapping is trustworthy).
        ow, nw = o_raw.split(), n_raw.split()
        if len(ow) == len(nw):
            tok_match += 1
            src = text.get(k, "").split()
            if len(src) == len(ow):
                for s, a, b in zip(src, ow, nw):
                    if a != b:
                        words[(s, a, b)] += 1

    dists.sort(reverse=True)
    n = len(keys)
    mean = sum(d for d, _ in dists) / n
    med = sorted(d for d, _ in dists)[n // 2]
    seg_mean = sum(seg_dists) / n

    only_old = {s: c for s, c in old_syms.items() if s not in new_syms}
    only_new = {s: c for s, c in new_syms.items() if s not in old_syms}

    return dict(
        lang=lang, n_common=len(common), n_sampled=n, identical=identical,
        tok_match=tok_match, mean=mean, median=med, seg_mean=seg_mean,
        subs=subs, subs_ctx=subs_ctx, ins=ins, dele=dele, words=words,
        worst=dists[:40], old=old, new=new, text=text,
        only_old=only_old, only_new=only_new,
    )


def write_lang(r: dict) -> None:
    lang = r["lang"]
    with open(f"{OUT}/{lang}.subs.tsv", "w", encoding="utf8") as f:
        f.write("count\tespeak\tvernacula\ttop_context\n")
        for (a, b), c in r["subs"].most_common(150):
            top = r["subs_ctx"][(a, b)].most_common(1)[0][0]
            f.write(f"{c}\t{a}\t{b}\t{top}\n")
        f.write("\n# insertions (vernacula has, espeak lacks)\n")
        for b, c in r["ins"].most_common(40):
            f.write(f"{c}\t—\t{b}\t\n")
        f.write("\n# deletions (espeak has, vernacula lacks)\n")
        for a, c in r["dele"].most_common(40):
            f.write(f"{c}\t{a}\t—\t\n")

    with open(f"{OUT}/{lang}.words.tsv", "w", encoding="utf8") as f:
        f.write("count\tword\tespeak\tvernacula\n")
        for (s, a, b), c in r["words"].most_common(300):
            f.write(f"{c}\t{s}\t{a}\t{b}\n")

    with open(f"{OUT}/{lang}.samples.txt", "w", encoding="utf8") as f:
        f.write(f"# {lang} — most-divergent utterances (normalised phoneme distance)\n\n")
        for d, k in r["worst"]:
            f.write(f"## {k}  dist={d:.3f}\n")
            if k in r["text"]:
                f.write(f"txt {r['text'][k]}\n")
            f.write(f"esp {r['old'][k]}\n")
            f.write(f"ver {r['new'][k]}\n\n")


def main() -> None:
    argv = sys.argv[1:]
    sample = 1200
    if "--sample" in argv:
        i = argv.index("--sample")
        sample = int(argv[i + 1])
        del argv[i:i + 2]
    langs = argv or sorted(
        f[:-4] for f in os.listdir(OLD)
        if f.endswith(".tsv") and not f.endswith(".errors.tsv")
    )

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for lang in langs:
        r = analyse(lang, sample)
        if r is None:
            print(f"{lang}: no overlap, skipped", file=sys.stderr)
            continue
        write_lang(r)
        rows.append(r)
        print(
            f"{r['lang']:<14} n={r['n_sampled']:<5} identical={r['identical']:<5} "
            f"mean={r['mean']:.3f} segments-only={r['seg_mean']:.3f} "
            f"tok-align={r['tok_match'] / r['n_sampled']:.0%}"
        )

    rows.sort(key=lambda r: r["seg_mean"])
    with open(f"{OUT}/summary.tsv", "w", encoding="utf8") as f:
        f.write("lang\tn_common\tn_sampled\tidentical\tmean_dist\tmedian_dist\t"
                "segments_only_dist\ttok_align_pct\tsyms_only_espeak\tsyms_only_vernacula\tnote\n")
        for r in rows:
            f.write(
                f"{r['lang']}\t{r['n_common']}\t{r['n_sampled']}\t{r['identical']}\t"
                f"{r['mean']:.4f}\t{r['median']:.4f}\t{r['seg_mean']:.4f}\t"
                f"{r['tok_match'] / r['n_sampled']:.3f}\t{len(r['only_old'])}\t{len(r['only_new'])}\t"
                f"{VARIETY_MISMATCH.get(r['lang'], '')}\n"
            )

    with open(f"{OUT}/report.md", "w", encoding="utf8") as f:
        f.write("# espeak-ng vs vernacula-phonemizer — FLEURS IPA diff\n\n")
        f.write("`segments_only` strips stress and tone marks, so it isolates disagreement "
                "about *which sounds* from disagreement about *how they are marked*. "
                "espeak is not ground truth — this ranks where to look, not who is wrong.\n\n")
        f.write("| lang | sampled | identical | mean | segments-only | tok-align | note |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['lang']} | {r['n_sampled']} | {r['identical']} | {r['mean']:.3f} | "
                f"**{r['seg_mean']:.3f}** | {r['tok_match'] / r['n_sampled']:.0%} | "
                f"{VARIETY_MISMATCH.get(r['lang'], '')} |\n"
            )
        f.write("\n## Symbols used by only one engine\n\n")
        for r in rows:
            oo = ", ".join(f"`{s}`×{c}" for s, c in
                           sorted(r["only_old"].items(), key=lambda x: -x[1])[:12])
            on = ", ".join(f"`{s}`×{c}" for s, c in
                           sorted(r["only_new"].items(), key=lambda x: -x[1])[:12])
            if oo or on:
                f.write(f"- **{r['lang']}** — espeak-only: {oo or '—'}; vernacula-only: {on or '—'}\n")
        f.write("\n## Top substitutions per language (espeak → vernacula)\n\n")
        for r in rows:
            top = r["subs"].most_common(12)
            if not top:
                continue
            f.write(f"### {r['lang']}\n\n")
            for (a, b), c in top:
                f.write(f"- `{a}` → `{b}` ×{c}\n")
            f.write("\n")
    print(f"\nwrote {OUT}/summary.tsv, report.md, and per-language details")


if __name__ == "__main__":
    main()
