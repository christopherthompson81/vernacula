"""Per-language training sampling weights — density flattening (Task #3).

The 24 collected languages have naturally uneven utterance counts (2.1k-3.3k) and,
more importantly, each was added to the census-based minimal-25 set (see
espeak-ng-portable's docs/omnivoice-minimal-coverage-set.md) for a SPECIFIC set of
primitives it's the deliberate greedy-cover owner of — English for the 53 generalist
IPA base letters, Zulu for clicks (ǀǁǃ) + breathy voice + ɮ/ɦ, Hausa for ejectives/
implosives, Fula for prenasals, etc. Naive uniform utterance sampling would under-
expose those load-bearing primitives in an epoch relative to generalist languages.

This reconstructs the same population-descending greedy-cover ownership from the
primitive census (so a primitive is attributed to the first, biggest-population
language that carries it), then measures how often each language's OWNED primitives
actually occur in ITS OWN collected FLEURS manifest. Substring counting (not panphon
segmentation) because several owned primitives are combining diacritics that attach
to a base letter rather than segmenting standalone (breathy ̤, unreleased ̚, etc.).

Per-language weight = oversampling factor so the language's scarcest owned primitive
reaches >= N_TOKENS exposures/epoch, capped at MAX_WEIGHT.

Outputs (work/):
  sampling_weights.csv   lang, n_utts, weight, scarcest_owned_primitive, its count
  sampling_summary.txt   headline + full owned-primitive count breakdown per language
"""
import glob
import json
import math
import os
import sys
import pandas as pd

# population-descending order (greedy-cover ownership). 28-lang second pass: the 24
# minimal-set langs + am/om/sd/xh inserted at their population slots (am~57M, om~37M,
# sd~30M, xh~8M) — added to reinforce ejectives (am/om/xh) and implosives (sd: closes
# ɠ, 2nd ʄ source). si excluded: no FLEURS audio.
# ⚠ `en_gb` IS A COVERAGE PATCH, NOT A POPULATION ENTRY. It sits beside en_us because it is the same
# language, and `phon_of` maps both to the census key `en` — so the greedy sweep gives it an OWNED set
# of exactly nothing and a weight of 1.0. That is the intended outcome: it contributes the en-GB vowel
# units en_us never carried (`əᶷ` 192 -> 1,221, `ɛə` 37 -> 429) at natural frequency, without stealing
# ownership from any language or perturbing anyone else's weight. Adding it still makes v7 a DIFFERENT
# experiment than v6 -- 29 languages, not 28 -- which is a deliberate choice, not a free improvement.
POP_ORDER = ["en_us", "en_gb", "cmn_hans_cn", "hi_in", "es_419", "ar_eg", "fr_fr", "pt_br",
             "ru_ru", "de_de", "ja_jp", "tr_tr", "vi_vn", "ta_in", "ko_kr", "ha_ng",
             "th_th", "am_et", "om_et", "sd_in", "ff_sn", "kk_kz", "zu_za", "cs_cz",
             "sv_se", "xh_za", "ca_es", "ga_ie", "cy_gb"]

N_TOKENS = 300     # per-primitive-per-epoch redundancy target (matches realized_coverage.py)
# 2nd-pass cap lowered 8→3: the data additions (sd/xh/am/om) closed the real coverage
# gaps, so oversampling should be a gentle rebalance, not the fix. A high cap made the
# two biggest langs (en/fr) dominate the epoch off borderline-incidental owned phones
# (ʔ=41, ɜ=39). At 3×, anything needing more is a DATA gap (below 300/3=100 occ), and
# only genuinely-thin rare-phone langs (sd/ha/ga ~1.3-1.5×) get a modest 2× via ceil.
MAX_WEIGHT = 3.0
CENSUS = "/home/chris/Programming/espeak-ng-portable/docs/primitive-census.json"
TOKENS = "/mnt/data/omnivoice_ipa/corpus/tokens"
WORK = "/mnt/data/omnivoice_ipa/work"

# ⚠ THE EXCLUSION HAS TO BE APPLIED HERE, NOT ONLY AT SHARD-BUILD TIME. This script sets each
# language's oversampling weight from the count of its scarcest OWNED primitive; counting that over
# pairs which are then discarded targets the wrong number, and nothing downstream would say so. The
# order is: exclude -> patch manifests -> sampling weights -> webdataset. See corpus_filter.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_filter import load_exclusions, load_manifest  # noqa: E402


def owned_primitives():
    """{lang: [primitives it's the greedy-cover owner of]} via population-order sweep."""
    census = json.load(open(CENSUS))
    prim_langs = {p: set(v.get("langs", [])) for p, v in census.items()}
    phon_of = {f: f.split("_")[0] for f in POP_ORDER}
    covered = set()
    owned = {}
    for f in POP_ORDER:
        phon = phon_of[f]
        prims = {p for p, langs in prim_langs.items() if phon in langs}
        new = prims - covered
        covered |= prims
        owned[f] = sorted(new)
    return owned


def main():
    owned = owned_primitives()
    # Below this count, even MAX_WEIGHT× oversampling can't reach N_TOKENS, so the
    # primitive is a DATA gap (needs more source audio), not a sampling problem —
    # excluding it stops a single incidental token (e.g. en_us `r`=1) from pinning a
    # whole language at 8× off noise. It's reported as an under-target gap instead.
    MIN_RESCUABLE = math.ceil(N_TOKENS / MAX_WEIGHT)  # 300/8 = 38

    rows = []
    detail_lines = []
    EXCLUDED = load_exclusions()
    total_dropped = 0
    zero_gaps = []      # (lang, primitive) 0 occurrences — unfixable by resampling
    thin_gaps = []      # (lang, primitive, count) 0<count<MIN_RESCUABLE — data gap, not sampling
    for lang in POP_ORDER:
        mf = f"{TOKENS}/manifest_{lang}.jsonl"
        if not os.path.exists(mf):
            continue
        man, n_dropped = load_manifest(lang, EXCLUDED)
        total_dropped += n_dropped
        ipa_blob = [d["ipa"] for d in man]
        n_utts = len(man)
        blob = "\n".join(ipa_blob)
        counts = {p: blob.count(p) for p in owned[lang] if p != "."}  # "." = pause, not a phone
        for p, n in counts.items():
            if n == 0:
                zero_gaps.append((lang, p))
            elif n < MIN_RESCUABLE:
                thin_gaps.append((lang, p, n))
        # weight is driven ONLY by primitives oversampling can actually rescue to target.
        rescuable = {p: n for p, n in counts.items() if n >= MIN_RESCUABLE}
        if rescuable:
            scarcest, scount = min(rescuable.items(), key=lambda kv: kv[1])
            weight = min(max(N_TOKENS / scount, 1.0), MAX_WEIGHT)
        else:
            scarcest, scount, weight = "-", 0, 1.0
        rows.append(dict(lang=lang, n_utts=n_utts, weight=round(weight, 2),
                          scarcest_owned=scarcest, scarcest_count=scount,
                          n_owned=len(owned[lang]),
                          n_gaps=len([1 for l, p in zero_gaps if l == lang])
                                 + len([1 for l, p, n in thin_gaps if l == lang])))
        detail_lines.append(
            f"{lang} (owns {len(owned[lang])}): " +
            ", ".join(f"`{p}`={counts[p]}" for p in owned[lang] if p != ".")
        )

    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    natural_total = df["n_utts"].sum()
    weight_sum = (df["n_utts"] * df["weight"]).sum()
    scale = natural_total / weight_sum
    df["effective_utts"] = (df["n_utts"] * df["weight"] * scale).round().astype(int)
    df.to_csv(f"{WORK}/sampling_weights.csv", index=False)

    lines = []
    W = lines.append
    W("## Per-language sampling weights (density flattening, Task #3)\n")
    # ⚠ STATED, NOT ASSUMED. The weights below are computed over the corpus AFTER the audio-gate
    # exclusion; printing the count is what makes a missing/stale work/exclusions.tsv visible here
    # instead of silently shifting every weight. 0 dropped with a populated DB means the gate did
    # not run — see exclude_defective.py.
    W(f"Computed over the corpus AFTER exclusions: **{total_dropped} utterances dropped** "
      f"(`work/exclusions.tsv`; audio-side defects, see corpus_filter.py). "
      f"{'⚠ ZERO dropped — has exclude_defective.py been run?' if total_dropped == 0 else ''}\n")
    W(f"Target: every language's DELIBERATE owned primitive (the reason it's in the "
      f"census-based minimal-25 set) reaches >= {N_TOKENS} exposures/epoch. Weight = "
      f"oversampling factor vs. uniform utterance sampling, capped at {MAX_WEIGHT}x. "
      f"Epoch rescaled to the natural total utterance count ({natural_total}) so "
      f"weighting redistributes exposure rather than inflating epoch size.\n")
    W(f"Primitives below {MIN_RESCUABLE} occurrences (= {N_TOKENS}/{MAX_WEIGHT}x) can't reach "
      f"target even at max oversample, so they're treated as DATA gaps (need more source "
      f"audio), not sampling problems — they don't drive the weight.\n")
    W("| lang | n_utts | owns | weight | effective_utts | scarcest rescuable owned | count | gaps |")
    W("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        W(f"| {r.lang} | {r.n_utts} | {r.n_owned} | {r.weight}x | {r.effective_utts} | "
          f"`{r.scarcest_owned}` | {r.scarcest_count} | {r.n_gaps} |")

    if thin_gaps:
        W(f"\n### Thin owned primitives — data gaps, not sampling ({len(thin_gaps)} pairs, "
          f"0 < count < {MIN_RESCUABLE})\n")
        W("Present but too sparse for oversampling to reach target; need more source audio "
          "(the second-pass adds — sd/xh/am/om — target exactly these families). Listed so "
          "they're tracked, not silently 8x'd off a handful of tokens.\n")
        for lang, p, n in sorted(thin_gaps, key=lambda x: x[2]):
            W(f"- **{lang}** `{p}` = {n}")

    if zero_gaps:
        W(f"\n### Genuine zero-count gaps ({len(zero_gaps)} lang/primitive pairs)\n")
        W("Occur **zero times** in the collected corpus — resampling can't fix a zero. Mostly "
          "loanword-only / dialect-mismatch artifacts (e.g. en_us has no RP-only `ɜ`/`ɐ`; "
          "General American uses rhotic `ɚ`). Per the \"first attempt\" plan: don't block, "
          "revisit only if evaluation shows the model can't interpolate them from a neighbour.\n")
        by_lang = {}
        for lang, p in zero_gaps:
            by_lang.setdefault(lang, []).append(p)
        for lang, ps in by_lang.items():
            W(f"- **{lang}**: " + ", ".join(f"`{p}`" for p in ps))

    W("\n### Per-language owned-primitive realized counts\n")
    for dl in detail_lines:
        W(dl + "\n")
    open(f"{WORK}/sampling_summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:100]))
    print(f"\n-> work/sampling_weights.csv, work/sampling_summary.txt")


if __name__ == "__main__":
    main()
