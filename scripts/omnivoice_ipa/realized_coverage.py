"""Realized phone-coverage + greedy set-MULTI-cover for Phase-1 composition.

Driven by what actually OCCURS in the phonemized FLEURS transcripts (not PHOIBLE
inventories). Segments each phonemizer IPA line with panphon, counts phones per
language, then greedily selects the minimal language set so that every phone that
is *coverable at all* reaches the redundancy target:
    each phone in >= K languages AND >= N total occurrences.

Outputs (work/):
  realized_phone_lang_counts.csv   phone x lang occurrence matrix (long form)
  phase1_selection.csv             greedy language pick order + what each unlocks
  coverage_vs_nlangs.csv           curve: #langs -> phones meeting target (the knee)
  realized_summary.txt             headline numbers + the Phase-1 language set
"""
import os, glob, collections
import pandas as pd
import panphon

K_LANGS = 3       # each phone must appear in >= this many languages
N_TOKENS = 300    # ... and >= this many total occurrences
PHON = "/mnt/data/omnivoice_ipa/work/phonemized"
WORK = "/mnt/data/omnivoice_ipa/work"

_FT = panphon.FeatureTable()
# strip stress / boundary marks panphon doesn't segment; keep length, nasalization.
_STRIP = "ˈˌ|."


def segment(line):
    """IPA string -> list of panphon phone segments (diacritics attached)."""
    for ch in _STRIP:
        line = line.replace(ch, " ")
    segs = []
    for tok in line.split():
        segs.extend(_FT.ipa_segs(tok))
    return segs


def main():
    files = sorted(glob.glob(os.path.join(PHON, "*.txt")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    # phone -> {lang: count}
    counts = collections.defaultdict(lambda: collections.Counter())
    lang_phone_set = {}     # lang -> set(phones)
    lang_tokens = {}        # lang -> total phone tokens
    for f in files:
        lang = os.path.basename(f)[:-4]
        c = collections.Counter()
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                for s in segment(line.strip()):
                    c[s] += 1
        for ph, n in c.items():
            counts[ph][lang] = n
        lang_phone_set[lang] = set(c)
        lang_tokens[lang] = sum(c.values())
    langs = sorted(lang_phone_set)

    # long-form matrix
    rows = [(ph, lang, n) for ph, lc in counts.items() for lang, n in lc.items()]
    mat = pd.DataFrame(rows, columns=["phone", "lang", "count"])
    mat.to_csv(os.path.join(WORK, "realized_phone_lang_counts.csv"), index=False)

    all_phones = set(counts)
    # which phones are even *coverable* to target across the full buildable set?
    coverable = {ph for ph, lc in counts.items()
                 if len(lc) >= K_LANGS and sum(lc.values()) >= N_TOKENS}
    rare_uncoverable = all_phones - coverable  # exist but too thin even using all langs

    # greedy: pick lang that most increases the number of phones MEETING target.
    def meets(ph, chosen):
        lc = counts[ph]
        nl = sum(1 for L in chosen if lc.get(L, 0) > 0)
        tot = sum(lc.get(L, 0) for L in chosen)
        return nl >= K_LANGS and tot >= N_TOKENS

    def progress(chosen):
        """Capped progress toward the target, summed over coverable phones:
        min(langs, K)/K + min(tokens, N)/N. Smooth, so it ranks languages
        sensibly even before any phone first crosses the K>=3 threshold."""
        s = 0.0
        for ph in coverable:
            lc = counts[ph]
            nl = sum(1 for L in chosen if lc.get(L, 0) > 0)
            tot = sum(lc.get(L, 0) for L in chosen)
            s += min(nl, K_LANGS) / K_LANGS + min(tot, N_TOKENS) / N_TOKENS
        return s

    chosen, order, curve = [], [], []
    remaining = set(langs)
    while remaining:
        before = sum(1 for ph in coverable if meets(ph, chosen))
        base_prog = progress(chosen)
        best, best_key = None, (-1, -1.0)
        for L in remaining:
            trial = chosen + [L]
            gain = sum(1 for ph in coverable if meets(ph, trial)) - before
            key = (gain, progress(trial) - base_prog)   # tie-break on capped progress
            if key > best_key:
                best, best_key = L, key
        best_gain = best_key[0]
        chosen.append(best)
        remaining.discard(best)
        met = sum(1 for ph in coverable if meets(ph, chosen))
        order.append((len(chosen), best, best_gain, met,
                      round(100 * met / len(coverable), 1)))
        curve.append((len(chosen), met, round(100 * met / len(coverable), 1)))
        if met == len(coverable):      # target fully satisfied
            break

    sel = pd.DataFrame(order, columns=["rank", "lang", "phones_unlocked",
                                       "cum_phones_at_target", "cum_pct_coverable"])
    sel.to_csv(os.path.join(WORK, "phase1_selection.csv"), index=False)
    pd.DataFrame(curve, columns=["n_langs", "phones_at_target", "pct_coverable"]
                 ).to_csv(os.path.join(WORK, "coverage_vs_nlangs.csv"), index=False)

    phase1 = [L for _, L, *_ in order]
    L = []
    P = L.append
    P(f"=== Realized phone coverage / multi-cover (K>={K_LANGS} langs, N>={N_TOKENS} tokens) ===")
    P(f"buildable langs phonemized      : {len(langs)}")
    P(f"distinct realized phones        : {len(all_phones)}")
    P(f"  coverable to target (all 61)  : {len(coverable)}")
    P(f"  too thin even with all 61     : {len(rare_uncoverable)}  "
      f"(need field/low-resource data, not more FLEURS)")
    P("")
    P(f"Phase-1 set that satisfies target: {len(phase1)} languages")
    P(f"  -> {', '.join(phase1)}")
    P("")
    P("--- coverage vs #languages (the knee) ---")
    for n, met, pct in curve:
        if n in (1, 3, 5, 8, 10, 12, 15, 20, 25, len(curve)) or pct >= 99:
            bar = "#" * int(pct / 2.5)
            P(f"  {n:2d} langs: {met:3d}/{len(coverable)} phones at target ({pct:5.1f}%) {bar}")
    P("")
    P("--- diminishing returns: marginal phones unlocked per added lang ---")
    P("  " + " ".join(f"{r[1]}:+{r[2]}" for r in order))
    summary = "\n".join(L)
    with open(os.path.join(WORK, "realized_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
