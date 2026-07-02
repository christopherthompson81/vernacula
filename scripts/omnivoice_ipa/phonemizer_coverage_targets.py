"""Which languages must @vernacula/phonemizer support to express every
FLEURS-reachable phoneme — and, per language, exactly which phonemes it's for.

Target = the union of PHOIBLE inventories over all FLEURS languages (the phones we
can ever get clean audio for from FLEURS). Already covered = the buildable-60 (FLEURS
∩ current phonemizer). Greedy set-cover over the not-yet-supported FLEURS languages to
close the gap to 100%, attributing each marginal phoneme to the language that covers it.

Outputs (work/):
  phonemizer_targets.csv   lang, iso, n_marginal, marginal_phones
  phonemizer_targets.md    GH-issue-ready table + per-language phoneme lists
"""
import os
import pandas as pd
from fleurs_iso_map import FLEURS_TO_ISO

REF = "/mnt/data/omnivoice_ipa/reference"
WORK = "/mnt/data/omnivoice_ipa/work"

ph = pd.read_csv(f"{REF}/phoible.csv", low_memory=False)
iso2ph = ph.groupby("ISO6393")["Phoneme"].apply(lambda s: set(s.dropna())).to_dict()

buildable = set(open(f"{WORK}/buildable_fleurs_codes.txt").read().split())
buildable.discard("ckb_iq")

# canonical English names (PHOIBLE doculect names are sometimes odd)
ISO_NAME = {
    "xho": "Xhosa", "zul": "Zulu", "mya": "Burmese", "ekk": "Estonian",
    "khm": "Khmer", "ibo": "Igbo", "khk": "Mongolian (Khalkha)", "jav": "Javanese",
    "oci": "Occitan", "lit": "Lithuanian", "hau": "Hausa", "wol": "Wolof",
    "lao": "Lao", "som": "Somali", "luo": "Luo", "umb": "Umbundu", "heb": "Hebrew",
    "hye": "Armenian", "kir": "Kyrgyz", "kat": "Georgian", "tgk": "Tajik",
    "kea": "Kabuverdianu", "sna": "Shona",
}


def resolve(cands):
    return next((c for c in cands if c in iso2ph), None)


fleurs_iso = {f: resolve(c) for f, c in FLEURS_TO_ISO.items()}
fleurs_iso = {f: i for f, i in fleurs_iso.items() if i}

build_iso = {fleurs_iso[f] for f in buildable if f in fleurs_iso}
all_fleurs_iso = set(fleurs_iso.values())

# target phone set = everything reachable from FLEURS audio
target = set().union(*[iso2ph[i] for i in all_fleurs_iso])
covered = set().union(*[iso2ph[i] for i in build_iso]) if build_iso else set()
gap = target - covered

# candidate languages = FLEURS langs not yet phonemizable
cand = {f: iso2ph[i] for f, i in fleurs_iso.items() if f not in buildable}

# family tagging for readability
clicks = set("ǀǁǂǃʘ")
def fam(p):
    tags = []
    if any(c in p for c in clicks): tags.append("click")
    if "ʼ" in p: tags.append("ejective")
    if any(c in p for c in "ɓɗʄɠʛ"): tags.append("implosive")
    if "ː" in p: tags.append("long")
    if "ʷ" in p: tags.append("labialized")
    if "ʲ" in p: tags.append("palatalized")
    if "ˤ" in p or "ˁ" in p: tags.append("pharyngealized")
    if "̃" in p: tags.append("nasal")
    if "ʰ" in p: tags.append("aspirated")
    return tags

# greedy set-cover of the gap
chosen, rows = [], []
remaining = dict(cand)
need = set(gap)
while need:
    best, contrib = None, set()
    for f, phs in remaining.items():
        c = phs & need
        if len(c) > len(contrib):
            best, contrib = f, c
    if not best or not contrib:
        break
    rows.append((best, fleurs_iso[best], sorted(contrib)))
    need -= contrib
    del remaining[best]
    chosen.append(best)

# write CSV
pd.DataFrame([(f, i, len(p), " ".join(p)) for f, i, p in rows],
             columns=["fleurs", "iso", "n_marginal", "marginal_phones"]
             ).to_csv(f"{WORK}/phonemizer_targets.csv", index=False)

# write GH-issue markdown
L = []
W = L.append
W("## Target: express every FLEURS-reachable phoneme\n")
W(f"- FLEURS-reachable phone set (∪ PHOIBLE over {len(all_fleurs_iso)} FLEURS langs): "
  f"**{len(target)}** phones")
W(f"- Already expressible (buildable-{len(build_iso)} = FLEURS ∩ current phonemizer): "
  f"**{len(covered)}**")
W(f"- Gap to close: **{len(gap)}** phones, via **{len(rows)}** additional languages "
  f"(greedy minimal set-cover below)")
resid = need
if resid:
    W(f"- Residual after these langs (phones in no single coverable FLEURS lang / "
      f"thin): {len(resid)}")
W("\n### Languages to support (ordered by marginal phoneme contribution)\n")
W("| # | Language | FLEURS | ISO | new phones | dominant families |")
W("|---|----------|--------|-----|-----------|-------------------|")
for n, (f, i, phs) in enumerate(rows, 1):
    famc = {}
    for p in phs:
        for t in fam(p):
            famc[t] = famc.get(t, 0) + 1
    famtop = ", ".join(f"{k}×{v}" for k, v in sorted(famc.items(), key=lambda x: -x[1])[:4]) or "—"
    name = ISO_NAME.get(i, i)
    W(f"| {n} | {name} | `{f}` | {i} | {len(phs)} | {famtop} |")
W("\n### Per-language phoneme requirements\n")
W("Each language below must be able to emit (at minimum) these phonemes — the ones it "
  "is uniquely responsible for in the cover (others it shares are already expressible):\n")
for f, i, phs in rows:
    W(f"**{ISO_NAME.get(i, i)} — `{f}` ({i})** — {len(phs)} phones:\n")
    W("> " + " ".join(phs) + "\n")
md = "\n".join(L)
open(f"{WORK}/phonemizer_targets.md", "w", encoding="utf-8").write(md)
print(md[:2500])
print("\n...\n[full markdown -> work/phonemizer_targets.md ; csv -> phonemizer_targets.csv]")
print(f"\nSUMMARY: {len(rows)} languages close the gap; residual {len(need)}")
print("Languages:", " ".join(f for f, _, _ in rows))
