"""Phone-space coverage / greedy set-cover for the OmniVoice IPA corpus.

Inputs:
  - PHOIBLE csv (phone inventories per language)
  - FLEURS->ISO map (audio-available languages)
  - epitran / espeak language lists (phonemizability)

Outputs (to work/):
  - candidate_pool.csv       FLEURS langs resolved to PHOIBLE + G2P flags + #phones
  - setcover_ranking.csv     greedy order maximizing cumulative distinct-phone coverage
  - uncovered_phones.txt     PHOIBLE phones NOT reachable by the FLEURS pool (rare-phone tail)
  - coverage_summary.txt     headline numbers

Run:
  /mnt/data/omnivoice_ipa/venv/bin/python coverage_analysis.py
"""
import os
import pandas as pd
import panphon
from fleurs_iso_map import FLEURS_TO_ISO

_FT = panphon.FeatureTable()
_FVEC_CACHE = {}


def feat_key(phone):
    """Map a phoneme string to a hashable panphon feature-vector signature.

    Collapses narrow diacritic variants that panphon doesn't distinguish
    (e.g. k / kʰ often share a vector) -> 'broad' phone-space coordinate.
    Returns None for phones panphon can't segment.
    """
    if phone in _FVEC_CACHE:
        return _FVEC_CACHE[phone]
    vecs = _FT.word_to_vector_list(phone, numeric=True)
    key = tuple(tuple(v) for v in vecs) if vecs else None
    _FVEC_CACHE[phone] = key
    return key

REF = "/mnt/data/omnivoice_ipa/reference"
WORK = "/mnt/data/omnivoice_ipa/work"
os.makedirs(WORK, exist_ok=True)


def load_phoible():
    df = pd.read_csv(os.path.join(REF, "phoible.csv"), low_memory=False)
    return df


def load_g2p():
    epi = set()
    with open(os.path.join(REF, "epitran_langs.txt")) as f:
        for line in f:
            code = line.strip().split("-")[0]
            if code:
                epi.add(code)  # ISO-639-3
    espeak = set()
    with open(os.path.join(REF, "espeak_voices.tsv")) as f:
        for line in f:
            parts = line.split("\t")
            if parts and parts[0].strip():
                # espeak codes are region-suffixed (en-us, fr-fr); key on the base.
                espeak.add(parts[0].strip().split("-")[0])
    portable = set()
    pp = os.path.join(REF, "portable_phonemizer_langs.txt")
    if os.path.exists(pp):
        with open(pp) as f:
            portable = {ln.strip() for ln in f if ln.strip()}
    return epi, espeak, portable


# ISO-639-3 -> 639-1, just enough to test espeak membership for our candidates.
ISO3_TO_1 = {
    "afr": "af", "amh": "am", "arz": "ar", "arb": "ar", "asm": "as", "ast": "an",
    "azj": "az", "aze": "az", "bel": "be", "bul": "bg", "ben": "bn", "bos": "bs",
    "cat": "ca", "ceb": "ceb", "ckb": "ku", "cmn": "cmn", "ces": "cs", "cym": "cy",
    "dan": "da", "deu": "de", "ell": "el", "eng": "en", "spa": "es", "ekk": "et",
    "est": "et", "pes": "fa", "fas": "fa", "fin": "fi", "fil": "fil", "tgl": "tl",
    "fra": "fr", "gle": "ga", "glg": "gl", "guj": "gu", "hau": "ha", "heb": "he",
    "hin": "hi", "hrv": "hr", "hun": "hu", "hye": "hy", "ind": "id", "ibo": "ig",
    "isl": "is", "ita": "it", "jpn": "ja", "jav": "jv", "kat": "ka", "kaz": "kk",
    "khm": "km", "kan": "kn", "kor": "ko", "kir": "ky", "ltz": "lb", "lug": "lg",
    "lin": "ln", "lao": "lo", "lit": "lt", "lvs": "lv", "lav": "lv", "mri": "mi",
    "mkd": "mk", "mal": "ml", "khk": "mn", "mon": "mn", "mar": "mr", "zsm": "ms",
    "msa": "ms", "mlt": "mt", "mya": "my", "nob": "nb", "nor": "nb", "npi": "ne",
    "nep": "ne", "nld": "nl", "nya": "ny", "oci": "oc", "gax": "om", "orm": "om",
    "ory": "or", "ori": "or", "pan": "pa", "pol": "pl", "pbt": "ps", "pus": "ps",
    "por": "pt", "ron": "ro", "rus": "ru", "snd": "sd", "slk": "sk", "slv": "sl",
    "sna": "sn", "som": "so", "srp": "sr", "swe": "sv", "swh": "sw", "swa": "sw",
    "tam": "ta", "tel": "te", "tgk": "tg", "tha": "th", "tur": "tr", "ukr": "uk",
    "urd": "ur", "uzn": "uz", "uzb": "uz", "vie": "vi", "wol": "wo", "xho": "xh",
    "yor": "yo", "yue": "yue", "zul": "zu",
}


def main():
    ph = load_phoible()
    epi, espeak, portable = load_g2p()

    # phonemes available per ISO-639-3 (union across that lang's inventories)
    iso_to_phones = ph.groupby("ISO6393")["Phoneme"].apply(lambda s: set(s.dropna())).to_dict()
    iso_to_invs = ph.groupby("ISO6393")["InventoryID"].nunique().to_dict()
    all_phonemes = set(ph["Phoneme"].dropna().unique())

    rows = []
    for fleurs, candidates in FLEURS_TO_ISO.items():
        resolved = next((c for c in candidates if c in iso_to_phones), None)
        if resolved is None:
            base = fleurs.split("_")[0]
            rows.append(dict(fleurs=fleurs, iso=None, in_phoible=False,
                             n_phones=0, n_inv=0, epitran=False, espeak=False,
                             portable=(base in portable)))
            continue
        i1 = ISO3_TO_1.get(resolved, resolved)
        base = fleurs.split("_")[0]  # FLEURS base code, e.g. cmn_hans_cn -> cmn
        rows.append(dict(
            fleurs=fleurs, iso=resolved, in_phoible=True,
            n_phones=len(iso_to_phones[resolved]),
            n_inv=iso_to_invs.get(resolved, 0),
            epitran=(resolved in epi),
            espeak=(i1 in espeak or resolved in espeak),
            portable=(i1 in portable or resolved in portable or base in portable),
            phones=iso_to_phones[resolved],
        ))

    pool = pd.DataFrame(rows)
    pool_out = pool.drop(columns=["phones"], errors="ignore")
    pool_out.to_csv(os.path.join(WORK, "candidate_pool.csv"), index=False)

    covered_langs = pool[pool.in_phoible].copy()

    # Greedy set-cover: repeatedly pick the lang adding the most new phonemes.
    remaining = {r.fleurs: r.phones for r in covered_langs.itertuples()}
    covered = set()
    order = []
    while remaining:
        best = max(remaining.items(), key=lambda kv: len(kv[1] - covered))
        fleurs, phones = best
        new = phones - covered
        if not new and order:  # nothing left to add; dump the rest in n_phones order
            for f2 in sorted(remaining, key=lambda k: -len(remaining[k])):
                order.append((f2, 0, len(covered)))
            break
        covered |= phones
        order.append((fleurs, len(new), len(covered)))
        del remaining[fleurs]

    rank = pd.DataFrame(order, columns=["fleurs", "new_phones", "cum_phones"])
    rank["rank"] = range(1, len(rank) + 1)
    rank["cum_pct_pool"] = (100 * rank.cum_phones / len(covered)).round(2)
    rank["cum_pct_phoible"] = (100 * rank.cum_phones / len(all_phonemes)).round(2)
    rank = rank.merge(pool_out[["fleurs", "iso", "epitran", "espeak"]], on="fleurs")
    rank.to_csv(os.path.join(WORK, "setcover_ranking.csv"), index=False)

    uncovered = sorted(all_phonemes - covered)
    with open(os.path.join(WORK, "uncovered_phones.txt"), "w") as f:
        f.write("\n".join(uncovered))

    lines = []
    P = lines.append
    P("=== OmniVoice IPA corpus — phone-space coverage (FLEURS pool) ===")
    P(f"PHOIBLE total distinct phonemes : {len(all_phonemes)}")
    P(f"FLEURS configs mapped           : {len(pool)}")
    P(f"  resolved into PHOIBLE         : {pool.in_phoible.sum()}")
    P(f"  NOT in PHOIBLE                : {(~pool.in_phoible).sum()} "
      f"({', '.join(pool[~pool.in_phoible].fleurs) or '-'})")
    P(f"  epitran G2P available         : {int(pool.epitran.sum())}")
    P(f"  espeak  G2P available         : {int(pool.espeak.sum())}")
    P(f"  neither G2P                   : {int(((~pool.epitran) & (~pool.espeak) & pool.in_phoible).sum())}")
    P("")
    P(f"Phonemes covered by FLEURS pool : {len(covered)} "
      f"({100*len(covered)/len(all_phonemes):.1f}% of PHOIBLE)")
    P(f"Rare-phone tail NOT covered     : {len(uncovered)} "
      f"({100*len(uncovered)/len(all_phonemes):.1f}%) -> needs low-resource/field corpora")
    P("")
    P("--- coverage milestones (greedy order) ---")
    for target in (50, 75, 90, 95, 99, 100):
        hit = rank[rank.cum_pct_pool >= target]
        if len(hit):
            r = hit.iloc[0]
            P(f"  {target:3d}% of pool reachable after {int(r['rank']):3d} langs "
              f"({r.cum_phones} phones)")
    P("")
    P("--- top 15 by marginal contribution ---")
    for r in rank.head(15).itertuples():
        P(f"  {r.rank:2d}. {r.fleurs:12s} {str(r.iso):5s} +{r.new_phones:3d} -> "
          f"{r.cum_phones:4d} ({r.cum_pct_phoible:.1f}% phoible)  "
          f"epi={'Y' if r.epitran else '-'} esp={'Y' if r.espeak else '-'}")
    # --- feature-space (broad) coverage: collapse phones to panphon vectors ---
    all_feat = {feat_key(p) for p in all_phonemes}
    all_feat.discard(None)

    def feat_cov(phone_set):
        s = {feat_key(p) for p in phone_set}
        s.discard(None)
        return s

    covered_feat = feat_cov(covered)
    P("")
    P("--- feature-space (panphon broad) coverage ---")
    P(f"PHOIBLE phonemes -> distinct panphon vectors : {len(all_phonemes)} -> {len(all_feat)}")
    P(f"FLEURS pool covers feature vectors           : {len(covered_feat)} "
      f"({100*len(covered_feat)/len(all_feat):.1f}% of feature space)")
    P(f"Feature-space tail NOT covered               : {len(all_feat)-len(covered_feat)} "
      f"({100*(len(all_feat)-len(covered_feat))/len(all_feat):.1f}%)")

    # --- buildable-today pool: FLEURS ∩ portable phonemizer (user's own IPA G2P) ---
    buildable = covered_langs[covered_langs.portable].copy()
    bcov = set().union(*[r.phones for r in buildable.itertuples()]) if len(buildable) else set()
    bfeat = feat_cov(bcov)
    P("")
    P("--- buildable-today pool: FLEURS ∩ portable phonemizer ---")
    P(f"languages                       : {len(buildable)} / {len(covered_langs)} "
      f"({', '.join(sorted(buildable.iso))})")
    P(f"raw phonemes covered            : {len(bcov)} "
      f"({100*len(bcov)/len(all_phonemes):.1f}% phoible, "
      f"{100*len(bcov)/max(len(covered),1):.1f}% of full FLEURS pool)")
    P(f"feature-space covered           : {len(bfeat)} "
      f"({100*len(bfeat)/len(all_feat):.1f}% of feature space)")
    miss_iso = sorted(set(covered_langs.iso) - set(buildable.iso))
    P(f"FLEURS langs NOT yet phonemizable: {len(miss_iso)} -> {', '.join(miss_iso)}")

    # --- phonemizer bring-up priority: marginal phone gain over the buildable pool ---
    # Greedy: which not-yet-phonemizable FLEURS lang unlocks the most NEW phones next?
    notbuilt = covered_langs[~covered_langs.portable]
    rem = {r.fleurs: r.phones for r in notbuilt.itertuples()}
    cov2 = set(bcov)
    bringup = []
    while rem:
        f2, ph2 = max(rem.items(), key=lambda kv: len(kv[1] - cov2))
        new = ph2 - cov2
        if not new:
            break
        cov2 |= ph2
        iso = pool_out.loc[pool_out.fleurs == f2, "iso"].iloc[0]
        bringup.append((f2, iso, len(new)))
        del rem[f2]
    br = pd.DataFrame(bringup, columns=["fleurs", "iso", "marginal_new_phones"])
    br.to_csv(os.path.join(WORK, "phonemizer_bringup_priority.csv"), index=False)
    P("")
    P("--- phonemizer bring-up priority (marginal NEW phones over buildable-61 pool) ---")
    for r in br.head(15).itertuples():
        P(f"  {r.Index+1:2d}. {r.fleurs:12s} {str(r.iso):5s} +{r.marginal_new_phones:3d} new phones")

    summary = "\n".join(lines)
    with open(os.path.join(WORK, "coverage_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
