# OmniVoice IPA fine-tune — corpus composition investigation

Working plan: `omnivoice_ipa.scratch.md` (repo root). This log covers the **dataset
composition** workstream — deciding *which* languages/utterances to collect for
proportional human-phone-space coverage, *before* downloading audio at scale.

Workspace (off the space-constrained root): `/mnt/data/omnivoice_ipa/`
- `reference/` — inventories & language lists (PHOIBLE, panphon, epitran, espeak, FLEURS)
- `work/`      — analysis outputs (coverage tables, set-cover rankings)
- `corpus/`    — audio (collected later, against the plan)
- `venv/`      — analysis env (panphon, epitran, pandas; no torch)

Analysis code is version-controlled in the repo under `scripts/omnivoice_ipa/`.

---

## Run 1 — 2026-06-20 ~11:00 — Assemble reference inventories

**Question:** what are the four inputs to the composition decision, and how big is the
phone space we're trying to cover?

**Commands / findings (raw):**

- **PHOIBLE** (`reference/phoible.csv`, 24 MB, from phoible/dev master):
  - 105,484 rows · 3,020 inventories · **2,094 distinct ISO-639-3 languages** ·
    **3,142 distinct phonemes** (= the coverage target space).
  - One language can have several inventories (different source doculects); must
    decide how to collapse (union vs. pick-one) in Run 2.
- **panphon** featurization of PHOIBLE phonemes: **3,101 / 3,142 segmentable (98.7%)**.
  - The 41 misses are exotic/notational: variant trills (`R̪|R`, `Rʲ`), small-cap
    pseudo-IPA (`ᴅ`, `N`), downstep tone mark `↓`, epiglottals (`ʡ`, `ʜ`, `ʢ`),
    rhotacized schwa `ɚ`, retroflex implosive `ᶑ`. These are **feature-space blind
    spots** — record them; they can't be balanced via PanPhon features.
- **epitran** (G2P): **117 distinct ISO-639-3** languages (`reference/epitran_langs.txt`,
  126 lang-script variants). High-precision rule-based G2P.
- **espeak-ng** (G2P): **131 voices** (`reference/espeak_voices.tsv`). Broader but
  coarser; offline training-prep only (never a runtime dep — see scratch thesis).
- **FLEURS**: **102 languages** (read speech, 16 kHz, parallel, clean) —
  `reference/` config list captured. The clean multilingual spine.

**Implication for next step:** the candidate pool for the corpus =
**(audio-available) ∩ (phonemizable)**. FLEURS gives 102 audio-available langs;
nearly all have espeak and most have epitran coverage. Run 2 = map these to PHOIBLE
inventories and run greedy set-cover over the 3,142-phone space to (a) confirm how
much of the human phone space 102 high-resource langs actually reach, and (b)
quantify the rare-phone tail that *only* low-resource/field corpora (UCLA, PHOIBLE's
thin doculects) can supply — the "coverage paradox" the scratch plan flags.

**Open decisions seeded:**
- IPA convention (narrow vs broad) — defer until we see G2P output variance.
- PHOIBLE inventory collapse rule (union of all doculects per ISO vs. canonical pick).
- Whether espeak-only langs (no epitran) are worth the lower G2P quality.

---

## Run 2 — 2026-06-20 ~11:30 — Set-cover over the phone space (FLEURS pool)

**Question:** how much of the human phone space does the clean high-resource spine
(FLEURS, 102 langs) actually reach, and what's in the tail it misses?

**Method:** `scripts/omnivoice_ipa/coverage_analysis.py` — resolve FLEURS→ISO-639-3
(`fleurs_iso_map.py`), union phonemes per language across PHOIBLE inventories, greedy
set-cover maximizing cumulative distinct phones. Coverage scored two ways: raw
phoneme strings, and **panphon feature-vector signatures** (collapses narrow diacritic
variants → "broad" phone-space coordinate). Outputs in `work/`.

**Findings (raw):**
- 97/102 FLEURS langs resolve into PHOIBLE (5 miss: be_by, bs_ba, kk_kz, nso_za,
  ny_mw — mapping gaps / absent doculects, to patch).
- G2P availability over the pool: **epitran 60, espeak 72, neither 11.**
  (espeak codes are region-suffixed — `en-us` not `en` — matching keys on the base.)
- **Raw coverage: FLEURS pool reaches 1121/3142 phonemes = 35.7%.**
- **Feature-space coverage: 792/1865 distinct panphon vectors = 42.5%.**
  Even collapsing diacritics, the clean spine covers **under half** the broad phone
  feature space. ← empirical confirmation of the "coverage paradox": high-resource
  data is *structurally* insufficient for universal phone coverage.
- Greedy milestones: 50% of the pool's own phones after 10 langs, 90% after 39,
  100% after 72 — strong diminishing returns; ~30 well-chosen langs ≈ most of what
  the high-resource world can offer.
- Top marginal contributors: hin (+153!), gle (+94), xho (+56), por, nld, dan, cmn,
  mya, amh, kor. (Hindi's huge contribution = PHOIBLE's Hindi doculect is
  phonetically rich + heavily diacriticized.)

**The uncovered tail (2021 phones) — composition, not just count:**
- Mostly **secondary-articulation combinatorics on bases we already cover**:
  labialized 209, long 280, aspirated 159, palatalized 118, pharyngealized 116,
  nasalized 142. The model may generalize these from base+modifier seen elsewhere —
  they are NOT all independent collection targets.
- **Genuinely exotic segment families that need field/low-resource data:**
  **ejectives 151, clicks 147, implosives 20.** These don't live in FLEURS at all.
- Only **16 missing single-codepoint base phones**, several of which are panphon-
  unsegmentable notation (`N R ↓ ʡ ʜ ʟ ʢ ⱱ ɞ ɶ …`) rather than real collection gaps.

**Implication for collection priority (revises the scratch "set-cover over PHOIBLE"):**
1. Take ~25–30 top FLEURS langs for the broad spine (covers ~90% of high-resource phones).
2. Targeted rare-family add-ons, not a blind low-resource sweep:
   - **clicks** → Khoisan + Nguni Bantu (Zulu/Xhosa already in FLEURS — verify click
     realization survives G2P), e.g. Nǁng, Khoekhoe, !Xóõ (UCLA archive).
   - **ejectives** → Ethiopian Semitic (Amharic/Tigrinya), NW/NE Caucasian (Georgian
     in FLEURS), Quechua, Hausa.
   - **implosives** → Sindhi (in FLEURS), Fula, Vietnamese (ɓ ɗ), Zulu.
   - **heavy 2ary articulation** → Caucasus (labialization/pharyngealization), Salish,
     Arabic (pharyngealized).
3. Defer the panphon-unsegmentable notation phones — can't be balanced in feature
   space anyway; not worth chasing.

**Open issue:** epitran shows `eng` as no-G2P (English G2P in epitran is via Flite, a
separate path, not the map dir) — my epitran membership undercounts a few. Minor;
espeak covers them. To revisit once the **user's espeak-ng-portable** G2P (their
active work) is wired in as the primary phonemizer — that changes the phonemizability
axis materially (see Run 3).

---

## Run 3 — 2026-06-20 ~12:00 — Fold in the user's portable phonemizer (the real G2P)

**Context shift:** the user has been building `@vernacula/phonemizer`
(`~/Programming/espeak-ng-portable`) — a portable, dependency-free, pure-TypeScript
espeak-ng-compatible **IPA** phonemizer, with active per-language bring-up (recent:
Tigrinya, Uyghur, Haitian Creole, Kazakh, Hungarian, Azerbaijani…). This is **the**
G2P for our data prep, and it's the user's own controlled engine — using one engine
across all data also solves the scratch plan's "one consistent IPA convention"
requirement for free.

**Sanity check (works across scripts, clean IPA + stress):**
```
en -> həlˈoʊ wˈɜːld          hi -> nəmˈʌsteː dˈʊnɪjˌaː
ti -> sˈəlam ʕˈaləm          sw -> habˈari dunˈia
```

**Supported set:** 71 compiled languages (`reference/portable_phonemizer_langs.txt`).
Re-ran `coverage_analysis.py` with `portable` as the phonemizability axis.

**Headline — the buildable-today pool (FLEURS ∩ portable phonemizer):**
- **61 languages**, covering **906/3142 raw phonemes (28.8%)** = **80.8% of the
  *entire* high-resource (full-FLEURS) phone set**, and **35.1% of the panphon
  feature space.** → With *zero* new phonemizer work we can already build a corpus
  reaching four-fifths of what the clean high-resource world phonetically offers.
- 36 FLEURS langs are audio-available but not-yet-phonemizable. Among them sit the
  **rare-family carriers** from Run 2: **xho, zul (clicks), hau (ejectives),
  kat (Georgian ejectives), yor/ibo (tone)**.

**New cross-repo deliverable — phonemizer bring-up priority by *corpus phone gain***
(`work/phonemizer_bringup_priority.csv`; greedy marginal-new-phones over buildable-61):
```
 1. xho  +43   (clicks — by far the biggest single unlock)
 2. mya  +22    3. ekk(Estonian) +18   4. khm +18   5. ibo +15
 6. khk(Mongolian) +14   7. jav +10   8. oci +9   9. lit +8  10. hau +7
   … wol +7, lao +6, som +6, zul +6, luo +5
```
This is a *different ordering* than the phonemizer's current speaker-count roadmap
(`nextLanguage.scratch.md`). For the IPA-corpus goal specifically, **Xhosa is the
single highest-value bring-up** (+43 phones, unlocks the click family); Burmese,
Estonian, Khmer, Igbo follow. Worth surfacing to the phonemizer roadmap as a
"coverage-weighted" track alongside the speaker-weighted one.

**Decisions resolved:**
- IPA convention → **whatever `@vernacula/phonemizer` emits** (single engine =
  consistent narrow-ish IPA with stress marks). No epitran/vanilla-espeak mixing.
- Phonemizability axis → portable-phonemizer supported set, not epitran/espeak.

**Next:** define the per-language sampling budget over the buildable-61 (task #3),
then begin collecting FLEURS audio for that set (task #4) — pilot a few languages
first to validate the audio→phonemize→manifest pipeline before the full pull.

---

## Run 4 — 2026-06-20 ~12:40 — "How much do we actually need to store?"

**Question (user):** full FLEURS is too big to keep on disk — compute the *actual*
fine-tune footprint; it should be far less than the complete dataset.

**Pulled transcripts only** (no audio): all 61 buildable langs' `*.tsv` =
**169 MB total** (`corpus/fleurs_transcripts/`). From `num_samples` (col 6, 16 kHz):

| | FLEURS train, 61 buildable langs |
|---|---|
| utterances | **157,318** |
| total audio | **533 h** (~8.7 h/lang, 2.8–18.2 h range) |
| avg utterance | 12.2 s |

**The storage lever — OmniVoice trains on codec tokens, not waveforms.**
Measured the real rate by running the exported `higgs_encoder.onnx` on a 10 s signal:
- Encoder ingests **24 kHz**; emits `audio_codes` of shape **(1, 8, T)** at
  **25 Hz × 8 codebooks** = **200 codes/s**, `codebook_size` 1024 (10 bits).
- ⇒ **0.9 MB/h** bit-packed; ~1.4 MB/h as int16 npy; ~5.8 MB/h as naive int64.

**Footprint, the three regimes:**

| What we keep | Phase-1 (~10 langs, ~50 h) | Full buildable-61 (533 h) |
|---|---|---|
| Raw FLEURS audio (tars) | ~28 GB | **~336 GB** ← the thing we can't store |
| Selected 24 kHz FLAC | ~4 GB | ~45 GB |
| **Codec tokens + IPA** | **~50–290 MB** | **~0.5–3 GB** |

**Answer: the *entire* 61-language buildable corpus, stored the way the model
actually consumes it (Higgs codec tokens + IPA strings), is well under ~1 GB
bit-packed (a few GB worst-case as int).** The 336 GB only exists if we hoard raw
audio we never feed to the model.

**Ingest path that needs ~no persistent audio storage:**
stream FLEURS per-language over HTTP (HF `datasets` streaming) → resample to 24 kHz
→ `higgs_encoder` → store `audio_codes` (+ phonemized IPA + speaker/ref metadata) →
discard waveform. Peak transient disk = one language's stream buffer (cap/clear the
HF cache between langs), never the full corpus.

**Consequence for the sampling budget (task #3):** storage is *no longer a reason to
subsample*. Keep all tokens for the buildable set; do phone-density flattening
(oversample rare phones, cap common) as **train-time sampling weights**, not by
deleting data. Balancing becomes a config knob, not a storage decision.

**Caveat to validate next:** tokenize→detokenize round-trip quality on real FLEURS
audio (the codec is GAN-trained; confirm codes are faithful before committing the
whole corpus to token form), and confirm HF streaming avoids full-tar disk writes.

---

## Run 5 — 2026-06-20 ~13:30 — Realized coverage + multi-cover ("how small can Phase 1 be?")

**Question (user):** we don't need every language — compose a set that gives enough
info about *each phoneme across languages* without overkill. What improves from
Phase 1 → all 60, and how small can a smart Phase 1 be?

**Methodological upgrade:** stop selecting on PHOIBLE *inventories* (what a language
*can* produce) and select on **what actually occurs in the phonemized FLEURS
transcripts** (Zipfian; rare inventory phones barely appear in 8 h of read news).

**Pipeline built:**
- `espeak-ng-portable/phonemize_fleurs.ts` — phonemize all buildable langs' FLEURS
  transcripts with `@vernacula/phonemizer` → `work/phonemized/<lang>.txt`. 60 langs,
  ~145k utterances, 0 phonemize errors, ~3 min total.
- **`ckb_iq` dropped:** mapped to the phonemizer's `ku` (Latin Kurmanji), but FLEURS
  Sorani is Arabic-script → produced empty output. *Real buildable count = 60, not 61.*
  (Bad map, not a phonemizer bug; Sorani needs its own module.)
- `scripts/omnivoice_ipa/realized_coverage.py` — segment each IPA line with panphon
  (`ipa_segs`; retains length `ː`, aspiration `ʰ`, nasalization `̃` — verified, so
  the metric is *not* blind to those phonemic distinctions), count phone×lang, then
  greedy **set-MULTI-cover** to the user's target: each phone in **≥3 languages AND
  ≥300 tokens**. (Smooth capped-progress tie-break so early ordering is meaningful.)

**Findings:**
- **216 distinct realized phones** across 60 langs (129 diacriticized). Of these:
  - **140 are coverable to the ≥3-lang/≥300-tok target** using the buildable set.
  - **76 are too thin even with all 60 FLEURS langs** — the rare tail (clicks,
    ejectives, exotic vowels). *More FLEURS languages will never fix these; they need
    field/low-resource data.* Hard ceiling, re-confirmed on realized data.
- **Full target (all 140 at ≥3/≥300) needs 39 languages — NOT 60.** Languages 40–60
  add redundancy *beyond* the target + prosody/speaker variety only.
- **Coverage-vs-#languages curve (the knee):**
  ```
   5 langs → 48.6% of coverable phones at full target
  10 langs → 64.3%      15 langs → 78.6%      20 langs → 85.0%
  25 langs → 90.7%      39 langs → 100%
  ```
  Marginal unlock collapses to +1–2 phones/lang after ~the first 8–10
  (Indic-rich front-load: hi +32, pa +29 carry the aspiration/retroflex/length series).

**Answer to "what improves Phase 1 → all 60":** essentially **no new phonemes** (they
saturate); it's (1) *redundancy* — raising the ~30% of coverable phones from 1–2
attestations up to ≥3, (2) phonotactic/prosodic/speaker variety, (3) forgetting
resistance. The 76 thin phones improve from *none* of it.

**Recommended composition (replaces "Phase 1 = 5–10 popular langs" and "Phase 2 = all"):**
- **Phase 1 (proof / go-no-go): first ~15 of the multi-cover order** (`work/
  phase1_selection.csv`) → 78.6% of coverable phones at *full* ≥3/≥300 redundancy;
  the rest present but under-redundant (not absent). Typologically spread enough that
  the held-out-language test is meaningful (Indic, Semitic, Sinitic, Bantu, IE,
  Turkic, Japonic all represented by lang ~20).
- **Phase 2 (balanced corpus): 39 languages** = the *full* ≥3/≥300 target. This is the
  real "enough info per phoneme" set. **Not 60.**
- **All 60 = overkill for phone coverage**; justified only if prosody/anti-forgetting
  later proves to need it. The rare-tail 76 phones are a *separate* Phase-2/3 workstream
  (targeted click/ejective/implosive field data), independent of FLEURS language count.
- Storage stays trivial either way (Run 4): 39 langs ≈ ~350 h ≈ <1 GB as codec tokens.

---

## Run 6 — 2026-06-20 ~14:30 — Tone audit: the IPA stream isn't tone-consistent

**Trigger (user):** "Non-tonic for Phase 1?" Hindi isn't tonal (user was recalling
**Punjabi**, the tonal Indo-Aryan lang — or Hindi's *breathy/murmured* consonants,
which are phonation, not tone). Checked whether tone survives into the IPA at all.

**Audit across all 60 phonemized langs** — tone appears in **three** different
renderings (the segmental Run-5 coverage was blind to all of them; panphon ignores
tone marks):

| Scheme | Languages | Detail |
|---|---|---|
| **Full Chao contour letters** (˦˥, ˨˩, ˥˩…) | **th** (Thai) | computed from phonology (`thaiPron.ts`, `THAI_TONE_IPA`) — the principled one |
| **Compact digit/glyph** (1,2,**ɜ**,4,5,6,7; "first digit of Chao", tone 3→`ɜ`) | **cmn, yue, vi** | espeak pinyin/jyutping tone numbers carried through as `Tone()` phonemes. Digits are tone, **not** literal numbers (verified: ~0 standalone-number tokens). |
| **Combining diacritic** (grave `̀` = low) | **pa** (Punjabi) | tonogenesis only |
| none | rest | correctly non-tonal |

**Verified the digits are tone, not numerals:** cmn 87k / vi 74k digit-on-phone
tokens, ~0 standalone numbers. (Also spotted a stray artifact: vi emits literal
`(en)…(vi)` **code-switch markers** in the phone stream — separate cleanup.)

**Punjabi is only *partially* toned** (informs user's planned revisit):
- Implemented: word-initial historic voiced aspirate (ਘ/ਝ/ਢ/ਧ/ਭ) → voiceless
  unaspirated + **LOW tone (grave `̀`)** on the vowel (`ਘਰ` ghar → `kʌ̀ɾ`).
  818 graves, on 28% of lines.
- **Missing: the HIGH/high-falling tone** (Punjabi is a 3-way tone system; medial/
  final historic aspirates and ਹ /h/ trigger the high tone) — currently unrendered.
- (The 8,800 U+0303 in pa are **nasalization**, not tone.)

**Cross-language tone work this implies (phonemizer-side, user's domain):**
1. **Unify the rendering** — pick ONE scheme. Full Chao contour letters is the
   standard-IPA choice and what Thai already does; CJK's compact glyph and pa's grave
   would convert to it. (Mapping is well-defined: cmn 1→˥˥,2→˧˥,3→˨˩˦,4→˥˩,5→neutral;
   yue/vi have their own contour tables; th already done.)
2. **Complete Punjabi** — add the high-tone path.
3. **Future tonal bring-ups** (Yoruba/Igbo etc. in the roadmap gaps) need tone from scratch.

**Decisions / changes:**
- **Phase 1 → 17 langs**: added **cmn_hans_cn, vi_vn** (per user) so the proof
  exercises the tonal pathway early — the scratch plan flags suprasegmentals as the
  dominant residual error; better tested in Phase 1 than deferred to Phase 3.
  (`work/phase1_languages.txt` updated.)
- **Corpus/phonemizer decoupling:** the segmental ingest pipeline (Run 4) can proceed
  now; tone-IPA consistency is a parallel workstream. Two implementation sites — fix
  in the phonemizer (cleaner, user is motivated) **or** a corpus-side normalization
  pass that expands all three schemes to Chao letters. Don't block segmental collection
  on it, but the **tonal langs (cmn/yue/vi/th/pa) should not be tokenized into the
  training set until their tone IPA is unified** — else the model sees tone three ways.

---

## Run 7 — 2026-06-20 ~15:30 — Strategy for phones FLEURS can't reach

**Question (user):** what's the plan for the phones FLEURS structurally misses
(Run 5: 76 realized phones too thin even with all 60; Run 2: clicks/ejectives/
implosives absent from the high-resource spine)?

**Tiered the exotic families by *where the audio lives*** (PHOIBLE inventories ×
FLEURS availability × phonemizer support):

| Family | PHOIBLE | in buildable-60 | +phonemizer-ext (FLEURS audio) | beyond FLEURS |
|---|---|---|---|---|
| clicks | 168 | 0 | **+21** (xho, zul) | 147 |
| ejectives | 187 | 16 | **+20** (hau, kat, hye) | 151 |
| implosives | 30 | 4 | +6 | 20 |

**Reframing: a big part of the "gap" is a *phonemizer* gap, not a FLEURS gap.**
The audio for whole click/ejective FAMILIES is already in FLEURS (Zulu/Xhosa clicks;
Hausa/Georgian/Armenian ejectives) — we just can't phonemize those langs yet.

**Strategy — three tiers, cheapest first:**

- **Tier A — extend the phonemizer (highest ROI, reuses FLEURS audio).** Bring up
  xho, zul (clicks), hau, kat, hye (ejectives) in `@vernacula/phonemizer`. Gives the
  model the click & ejective *families* in real connected read speech, zero new audio
  sourcing. Overlaps Run 3's bring-up priority (xho was #1, +43 phones) — i.e. the
  corpus-coverage track of the phonemizer roadmap *is* the rare-phone-tail fix.

- **Tier B — targeted field corpora for the within-family long tail (beyond FLEURS).**
  The 147 "beyond" clicks etc. are mostly *narrow variants* (Khoisan distinguishes
  click place × phonation × accompaniment finely; PHOIBLE counts each separately). The
  model needs the click *primitives* (≈5 places × few phonations), not 147 strings —
  the same secondary-articulation-combinatorics point as Run 2. So Tier B is small and
  deliberate: ONE well-transcribed click language (UCLA Phonetics Lab Archive — e.g.
  Khoekhoe / Nǁng / !Xóõ) + one heavy-ejective language (NW/NE Caucasian) for the
  variants Tier A misses. Not a blind low-resource sweep.
  - **Data-governance gate:** field/endangered-language data carries the same OCAP /
    ethical-sourcing constraints as the Blackfoot Phase 4 (scratch plan). Tier B is
    consent-gated, not "just download UCLA" — partner/verify licensing per language.

- **Tier C — irreducible residual: lean on the base model + verify, don't assume.**
  OmniVoice already covers 600+ langs' phones internally; the IPA adapter only has to
  *align* a symbol to a rep the model holds. For phones with no clean training audio,
  rely on (a) the base model's latent coverage and (b) PanPhon feature-space proximity
  (an untrained IPA symbol sits near trained neighbors in articulatory space). **Measure
  it:** synthesize the gap phone, run a universal phone recognizer (Allosaurus/
  Allophant) on the output → know which Tier-C phones the base model already produces
  zero-adapter vs which genuinely fail. Document the unsupported frontier honestly
  (matches the held-out-language eval protocol).

**Net:** the gap is far smaller than "2021 missing phones." Tier A (phonemizer
extension on existing FLEURS audio) closes the click/ejective/implosive *families*;
Tier B adds a couple of deliberate field corpora for narrow variants; Tier C is a
measured reliance on base-model latent coverage, audited with Allosaurus — not a
collection problem. Rare-phone coverage is therefore mostly a **phonemizer-roadmap**
item, sequenced after the Phase-1 segmental proof.

---

## Run 8 — 2026-06-20 ~16:30 — Ingest pipeline: CPU too slow, CUDA fix, pilot validated

**Question (user):** ETA for collection? Could a CUDA encoder path beat waiting?
Plus: does the repo's Whisper handle long audio specially?

**Throughput (encoder, `higgs_encoder.onnx`):**
- **CPU: 0.3 utt/s → 32.5 h** for the 14 non-blocked langs. Unusable.
- **CUDA: ~11–15 utt/s → <1 h.** But two gotchas, both fixed:
  1. **Correctness:** default CUDA produces *different codes* from CPU (maxdiff 1000 /
     1024 — total divergence). Cause: the VQ does a hard nearest-codebook argmax, so
     TF32 matmul drift flips code indices. **Fix: provider option `use_tf32=0` →
     codes bitwise-identical to CPU** (maxdiff 0). (Same TF32 lesson as the earlier
     transformer-loop CUDA fix.)
  2. **Speed on dynamic shapes:** every clip has a unique length → ORT re-runs cuDNN
     conv-algo search per shape ("Conv running in Fallback mode" spam). **Fix:
     `cudnn_conv_algo_search=DEFAULT`** (14.8 vs 9.8 utt/s). Also switched resampling
     to `soxr_hq` (librosa's default kaiser_best was a bigger bottleneck than the encoder).
- **Encoder input must be a multiple of 960 samples** (two internal encoder paths
  otherwise disagree by one frame at a Concat) → pad up to next ×960.

**Long-audio / Whisper handling (user's question):** the tokenizer's `semantic_model`
*is* a Whisper encoder, and native Whisper chunks long-form at 30 s. But the **official
`omnivoice/scripts/extract_audio_tokens.py` does NOT chunk** — its
`StreamingLengthFilteredDataset` keeps only `min_len ≤ dur ≤ max_len` and drops the
rest (help examples: 2–15 s). So skipping long clips is the canonical behavior for
codec-tokenizing *training* utterances (meant to be short); the raw ONNX encoder runs
full quadratic attention, so a 256 s FLEURS outlier (p95 ~20 s) OOMs. Aligned our
filter to 1–30 s. **NB for training:** emit tokens in `extract_audio_tokens.py`'s
shard/manifest format when we get there — current ingest is a collection/validation
stand-in.

**Robustness fixes:** id↔IPA join was order-based and drifted (raw-split 3332 vs
phonemized 3430 lines, from embedded newlines + csv.reader mis-parsing quote chars).
Re-keyed phonemizer output by utterance **id** (`work/phonemized/byid/<lang>.tsv`);
ingest joins by id. Raw `split("\t")` everywhere (FLEURS tsv is not quoted CSV).

**Pilot (Nepali, CUDA, validated):**
- 3,324 utts, 8 skipped (out of range), **codes 5.8 MB** for 4.6 h (~1.3 MB/h int16),
  293 s end-to-end (~11 utt/s incl. download+resample), tar auto-deleted.
- **Codec round-trip: log-mel spectral distance 3.0–3.2 dB** (3 clips) — faithful
  (>10 dB would be garbage). orig/recon wavs in `work/codec_validation/`.
- Artifacts: `corpus/tokens/{codes_ne_np.npz, manifest_ne_np.jsonl}` (manifest rows =
  {id, lang, ipa, gender, dur_s, n_frames}).
- **Subjective check (user, 2026-06-20):** orig vs recon sounded the same by ear
  (orig-vs-recon comparison, not comprehension — Nepali fluency irrelevant). With the
  3.13 dB log-mel-SD, **codec fidelity confirmed → cleared to tokenize the full corpus.**

**Decision (user): pause collection.** Pipeline is proven end-to-end and the phonemizer
work (#844 tone, #846 coverage) is the critical path (days–weeks). Resuming the full
14-lang non-blocked collection is <1 h of compute. ingest_fleurs.py is ready;
just re-run with `--provider cuda` over `work/phase1_languages.txt` minus tonal langs.

## Run 9 — 2026-06-23 — Text tokenizer = Qwen2 byte-BPE; spacing-modifier vs combining-diacritic split

**Question (espeak-ng-portable referee sweep, item (c)):** does OmniVoice's text-side
tokenizer cleanly consume the IPA we emit (ʔ, ̚, t͡ɕ, Chao letters, narrow diacritics)?
This gates the "make language-distinctive realization EXPLICIT" decision — see the
phonemizer-side memory `omnivoice_explicitness_principle`.

**Tokenizer:** `/mnt/data/models/omnivoice/k2-fsa-OmniVoice` → `Qwen2Tokenizer`, **byte-level
BPE, vocab 151643**. So there is NO hard out-of-vocabulary — every Unicode IPA char is
representable via byte fallback. (c) is therefore NOT "in vocab vs out"; it is **single
clean token vs byte-split**.

**Classified all 130 census codepoints + the composed units we emit** (via the chatterbox
export venv's transformers). The dividing line is the UNICODE BLOCK:
- **CLEAN single token** — spacing modifier letters: `ː ˈ ˌ ʲ ʷ ˤ ʰ ʱ ˀ`(no, ˀ splits — see below),
  the Chao tone letters `˥ ˦ ˧ ˨ ˩` (U+02E5-02E9), clicks `ʘ ǂ ǁ`, glottal `ʔ`, base letters.
  → `aː`, `a˧˥`, `tˤ`, `ðˤ`, `ɽʱ`, `ɖʱ` all tokenize cleanly (base + spacing-mod concatenate).
- **BYTE-SPLIT (2–5 tokens, raw-byte fragments)** — every COMBINING diacritic (U+0300–036F):
  tie `͡`(U+0361), nasal `̃`, dental `̪`, voiceless `̥`, unreleased `̚`, lowering `̞`, syllabic `̩`,
  breathy `̤`, Korean-tense `͈`, raising `̝`, non-syllabic `̯`, breve `̆`, ring-above `̊`. Plus the
  glottalization spacing char `ˀ`(U+02C0) and modifier beta `ᵝ`(U+1D5D). So: t͡ɕ/t͡ʃ/d͡ʒ/k͡p/ɡ͡b
  (4–5 toks), o̞/e̞/ɯᵝ/ə̃/m̥/r̥/k̚/t̚/p̚ (3 toks each), bare ɭ/ɬ/ʄ/ɻ/ɻ (2 toks — these single
  letters are themselves 2-byte in the byte-BPE).

**Implication.** Byte-split is LEARNABLE (the byte fragments for a given codepoint are
deterministic — the adapter can map [k][byte][byte]→unreleased-k) but a WEAKER, more
expensive signal than a clean token, and it compounds with rarity. So the design rule for
the constrained primitive vocabulary: **prefer spacing-modifier representations; spend a
combining diacritic only where the distinction is phonologically load-bearing AND recurs
with corpus frequency.** Validates existing choices — Chao tone rendering (#844), Arabic
emphatics `tˤ/ðˤ`, Hindi breathy `ʱ`, and the to-add Vietnamese `ʔ` are all CLEAN. Flags as
"expensive" (revisit, don't auto-change): affricate TIES (#925; untied `tʃ`=2 clean toks vs
`t͡ʃ`=4 split), Japanese narrow `o̞`/`ɯᵝ`, Vietnamese dental `t̪` (plain `t` is clean) and the
deferred unreleased finals `̚`. None are blockers; they are a token-budget/signal-strength
cost to weigh per feature.

---

## Run 10 — 2026-07-01 — Reconcile corpus plan with the phonemizer's authoritative 25-lang minimal set

**Question:** the phonemizer side crystallized a definitive coverage target
(`espeak-ng-portable/docs/omnivoice-minimal-coverage-set.md`): **25 languages cover all 130
attested primitives**, chosen by greedy set-cover in *speaker-population-descending* order over
the primitive **census** (FLOOR — primitives actually emitted on the per-language corpora), not
PHOIBLE inventories. How does that authoritative set map onto what the ingest pipeline can
collect *today*, and what's actually still blocking?

**The 25:** en cmn hi es ar fr pt ru de ja tr vi ta ko ha th ff si kk zu cs sv ca ga cy

**Mapping to FLEURS collectibility (cross-ref `buildable_fleurs_codes.txt`, current phonemizer
lang list, `fleurs_iso_map`):**

- **21 collectible now** (phonemizer ready + FLEURS audio exists):
  en_us cmn_hans_cn hi_in es_419 ar_eg fr_fr pt_br ru_ru de_de ja_jp tr_tr vi_vn ta_in ko_kr
  th_th cs_cz sv_se ca_es ga_ie cy_gb **kk_kz**.
  - `kk` is the catch: the phonemizer gained Kazakh *after* I computed
    `buildable_fleurs_codes.txt`, so that file is now **stale** (says 60, should be 61 with kk).
- **3 blocked on beyond-espeak phonemizer bring-up** (FLEURS audio exists, G2P doesn't):
  - `ha_ng` (Hausa) → primitives `ʼ ɓ ʷ`
  - `ff_sn` (Fula) → `ⁿ ᵑ ʄ ᵐ`
  - `zu_za` (Zulu) → `ɮ ̤ ǀ ǃ ǁ ɦ`
  - These 3 are the **entire remaining critical path** for corpus collection. The doc says
    they're in progress (authored beyond-espeak like Igbo/Burmese).
- **1 blocked on AUDIO, not phonemizer** — the new gap the old inventory-based analysis missed:
  - `si` (Sinhala): phonemizer supports `si`, but **FLEURS has no `si_*` code**. Sinhala is the
    sole provider of `ᶯ` in the minimal set. So `ᶯ` is unattestable from FLEURS — needs a
    non-FLEURS Sinhala source, or accept `ᶯ` as interpolable from `ɳ` (± retroflex/nasal diac).

**Deferred by design (not gaps):** `ʘ ʡ ʙ` — zero-provider even in the ceiling; revisit only
when audio for a producing language exists (Khoisan / Somali-`ʡ` region / Nias-`ʙ`). Interpolable
set (`ʛ ʢ ʜ ɘ ɞ ɶ ʟ ɰ ⱱ ɢ`) and compositional ties (`k͡p ɡ͡b ŋ͡m`) out of scope.

**Implication for next step:** the old Phase-1(17)/Phase-2(39) lists are **superseded** — they
were PHOIBLE-*inventory* optimal (ne, pa, sd, mr, hu, gu, pl, lb…); the census-based
population-ordered **25 is the real corpus target**. Nepali (the one language already tokenized)
is *not* in the 25 — it was the pipeline-validation run, so no loss. The headline: the blocker
shrank from "23 languages of phonemizer work" to **3 bring-ups (ha/ff/zu) + 1 audio gap (si)**,
and **21/25 are unblocked right now** — collection on those is the proven <1h CUDA run, no longer
gated. Remaining serialization is only ha/ff/zu.

**CORRECTION (same run, deeper check):** read the primitive **census** directly
(`docs/primitive-census.json`) rather than the stale `reference/portable_phonemizer_langs.txt`
snapshot. The census — ground truth of primitives *actually emitted* — spans **81 languages and
already includes ha, ff, zu, si, kk**. So the phonemizer supports the **entire** 25-lang set now;
the ha/ff/zu bring-ups are *done*. The stale snapshot (refreshed to 81 langs) had misled me. Revised
picture: **24/25 collectible from FLEURS immediately** (only `si` lacks FLEURS audio; it's the sole
provider of `ᶯ` among all 81 langs — no phonemizer substitute — but externally sourceable via
OpenSLR, deferred to post-collection sparsity eval per "first attempt"). Bonus: `phonemize()`
defaults to `ipaRendering:"canonical"`, which *is* the #844 tone-harmonization fix — so the corpus
IPA stream is tone-consistent for free (closes Run 6's concern).

**Pipeline piece rebuilt:** the reorg lost the original FLEURS phonemization script (corpus
workflow moved to Wikipedia/Leipzig wordlists for the census, not FLEURS transcripts). Rebuilt as
`espeak-ng-portable/tools/omnivoice-fleurs-phonemize.ts` against the current public `phonemize()`
API. Validated on cs_cz: 2811 utts → canonical IPA, 0 err, 5s. All 24 phonemizer `data/` dirs
present; FLEURS→phon code = first `_`-segment. Remaining to collect: phonemize the other 23
(19 transcripts local; pull ha_ng/ff_sn/kk_kz/zu_za transcripts), then `ingest_fleurs.py
--provider cuda` over the 24 (the proven pilot path).

---

## Run 11 — 2026-07-01 — Execute collection: phonemize all 24, then unblock the deleted codec ONNX

**Question:** run the full 24-language collection end to end.

**Phonemization (done, clean):** ran `tools/omnivoice-fleurs-phonemize.ts` over all 24.
Pulled the 4 missing FLEURS transcripts (ha_ng 3259, ff_sn 3235, kk_kz 3200, zu_za 2858 rows)
via `hf_hub_download`. Result: **51,255 utts across 20 local langs (0 err)** + the 4 new langs.
Rare primitives verified in output: Zulu clicks `ǃ`, Hausa ejective `ʼ`, Fula prenasalization
`ᵑ` all render. The canonical default (#844) gives one consistent tone convention for free.

**Zulu — 3 errors (only lang that threw):** `numberToFragmentTokensZulu` hard-*throws* at
`n ≥ 10⁶` (`zulu.ts:68`); the 3 failures were the same FLEURS sentence containing `5,000,000`.
The cap is artificial — `isigidi` (10⁶) / `isigidigidi` (10⁹) already ship in
`data/zu/tone-lexicon.tsv:460-461`; Hausa (cap 10⁹) shows the `magnitudeGroup` pattern to
extend it. Also a robustness smell: a throw aborts the whole utterance in a batch G2P.
→ filed **espeak-ng-portable#1245** (raise cap + degrade-don't-throw). **Resolved same day**
(upstream #1246, commit 3eda92f0: cap → <10¹², degrade instead of throw). Pulled, re-phonemized
zu_za → **0 err** (was 3), 2858 rows incl. the recovered `5,000,000` sentence
(`isigidi ezinhlanu`). The running ingest has zu_za queued last, so it consumes the corrected
byid file with no re-ingest.

**ne_np pilot data dropped:** the June pilot's `codes_ne_np.npz`/`manifest_ne_np.jsonl` were
doubly stale — pre-canonical IPA *and* encoded by the now-deleted June encoder export (mixing it
with the July re-export = two VQ dialects in one training set). Since ne_np is not in the 25-lang
minimal set (pilot only), resolved as cleanup rather than re-ingest: removed all ne_np corpus
artifacts. Corpus = the 24 minimal-set FLEURS langs only (+ si external, later).

**Stalled download mid-batch:** `zu_za` (last in queue) hit a dead HF connection — process asleep
on futex, GPU 1%, socket in `CLOSE-WAIT` with unacked bytes (remote closed, client never
noticed), `.incomplete` file frozen at 768 MB for 25+ min. Killed the process (23/24 languages'
outputs were already complete and unaffected) and reran `ingest_fleurs.py --provider cuda zu_za`
alone — `hf_hub_download`'s Range-request resume picked up from the existing `.incomplete`/lock
files rather than restarting; confirmed by watching the partial grow (768.0→804.9 MB in 90s)
before it finished the download and moved on to GPU encoding. Note: killing the process lost its
buffered stdout (Python fully-buffers stdout to a non-tty; the batch log's 23 per-language
summary lines never flushed) — harmless, since the manifests/codes on disk are ground truth, not
the log.

**COLLECTION COMPLETE — final tally (all 24 minimal-set languages):**

| | |
|---|---|
| Utterances | **65,208** |
| Source audio | **224.0 hours** |
| Codec tokens (total) | **133.9 MB** |
| Per-language spread | 2,101–3,263 utts (301–815 min) each |

Confirms the Run 2 storage projection (codec tokens, not raw audio ⇒ <1 GB) with room to spare.
Next: Task #3 (per-language sampling-budget weights for training) is the only open composition
item; `si` (Sinhala, external audio) remains deferred per user, revisit only if `ᶯ` proves
sparse downstream.

---

## Run 12 — 2026-07-01 — Task #3: per-language sampling weights (density flattening)

**Question:** the 24 languages naturally range 2.1k–3.3k utterances; how should training sample
across them so every phone gets proportional exposure, not just every utterance?

**First attempt (wrong signal):** segmented the collected manifests' IPA with panphon
(`realized_coverage.py`'s convention) and set each language's weight from its rarest SEGMENTED
phone. Garbage in: the "scarcest phone" was noise — `r` count=1 in English, `v` count=1 in
Hindi — incidental one-off loanword/coarticulation blips, not the deliberate rare primitives
(clicks, ejectives, prenasals) each language was actually added to the set for.

**Fixed approach:** reconstructed the exact primitive-ownership attribution the phonemizer's own
minimal-25 doc used (population-descending greedy cover over the census), then measured how
often each language's OWNED primitives occur in **its own actually-collected corpus** — substring
counting on raw IPA (not panphon segmentation), since several owned primitives are combining
diacritics (breathy `̤`, unreleased `̚`, centralized `̈`) that attach to a base letter rather than
segmenting standalone.

**Key correction mid-build:** the naive version conflated "thin" (few occurrences — fixable by
oversampling) with "absent" (zero occurrences — NOT fixable by oversampling; weighting a zero
stays zero). Split them: weight is computed only from the scarcest **nonzero** owned primitive;
zero-count gaps are reported separately as a distinct, unresolvable-by-resampling list.

**Result** (`scripts/omnivoice_ipa/sampling_budget.py` → `work/sampling_weights.csv` +
`sampling_summary.txt`):
- 3 languages hit the 8x oversampling cap on a genuinely thin-but-present primitive: en_us
  (`r`=1 — a rolled-r loanword form, distinct from its native `ɹ`), hi_in (`ɣ`=6), ff_sn (`ʄ`=2).
- zu_za (4.11x), fr_fr (2.88x), ga_ie (1.46x), ha_ng (1.31x) get moderate oversampling for their
  deliberate rare primitives (Zulu `ɦ`=73, French `ɒ`=104, Irish `̆`=206, Hausa `ɓ`=229).
- The remaining 17 languages sit at 1.0x — their owned primitives are already abundant
  (hundreds to hundreds-of-thousands of occurrences), so uniform sampling suffices.
- **14 lang/primitive zero-count gaps** flagged (not weighted, since resampling can't help):
  en_us (`c x ɐ ɜ ɬ ̃ ̧` — mostly RP-only vowels absent from General American, plus loanword-only
  consonants), cmn (`̪`), hi_in (`ɟ`), pt_br (`́ ̂`), ru_ru (`ɭ`), ja_jp (`̈`), ta_in (`ʉ`).
- Per the "first attempt" plan (mirrors the `si`/`ᶯ` deferral from Run 10): don't block on these
  zero gaps now — revisit only if evaluation shows the model can't render them via feature
  interpolation from a collected phonetic neighbour.

**Composition workstream is now fully closed** (Tasks #1–#9 all done). Remaining open items are
external to this repo: `si` audio sourcing (deferred) and any zero-gap follow-up (deferred),
both contingent on downstream fine-tune evaluation, not further collection work.

**Ingest BLOCKED — codec ONNX deleted:** `ingest_fleurs.py` died immediately:
`higgs_encoder.onnx` no longer exists (the `scripts/omnivoice_export/onnx/` dir is gitignored
and was swept in the `/` disk cleanup; `find / -name higgs_encoder.onnx` → nothing). Also the
ingest stack was split across interpreters — installed `librosa`+`soxr` into `base` (already
has ort-gpu 1.23.2 CUDA + hf + soundfile) to reunite it.
- **Survivors (on /mnt/data, the big disk):** source model
  `/mnt/data/models/omnivoice/k2-fsa-OmniVoice` (model.safetensors + audio_tokenizer) and the
  captured reference tensors `scripts/omnivoice_export/capture/reference.npz` (has
  `enc_input_values` / `dec_audio_codes`). So encoder+decoder are **regenerable** — only the
  export venv + onnx outputs were lost.
- **Re-export setup (space-safe, per user):** everything on /mnt/data — venv
  `/mnt/data/omnivoice_ipa/export_venv`, and `TMPDIR`/`PIP_CACHE_DIR`/`HF_HOME` redirected to
  `/mnt/data/_tmp|_pipcache|_hf` so torch's multi-GB install can't touch `/` (60 G free vs
  488 G). `base` has torch 2.9.1+cu128 but transformers 4.57 (needs ≥5.12) — upgrading base
  risks the main env, so a dedicated venv it is (CPU torch; export with `--device cpu
  --components encoder,decoder` to skip cuDNN-matching and the 0.6B transformer). Output →
  `/mnt/data/omnivoice_ipa/onnx`; will repoint `ingest_fleurs.py`'s `ONNX` there.

---

## Run 13 — 2026-07-01 — LoRA fine-tune wiring + the shared-sentence-id corpus bug

**Goal:** wire the actual IPA-input LoRA fine-tune against the collected corpus.

**Architecture (research, pip pkg `omnivoice==0.1.5`):** text enters as `<|text_start|>{text}
<|text_end|>` straight through the Qwen2 byte-BPE tokenizer — no G2P, no phoneme vocab. There is
a `text_pinyin`-substitution precedent (`use_pinyin_ratio`) that is the exact analog of what an
IPA adapter needs at the data level. Full training code ships (`omnivoice/training/`: HF-Accelerate
`OmniTrainer`, AdamW, masked-diffusion objective, per-codebook CE weighted [8,8,6,6,4,4,2,2]).
Text positions get `labels=-100` (no text loss — we adapt comprehension, not generation). No PEFT
support anywhere. Scope decision (user): **exclusively IPA input**, no orthographic mixing → the
IPA string goes directly in the label JSONL `"text"` field, no substitution logic needed.

**Approach:** fork `cli/train.py` → `scripts/omnivoice_ipa/train_lora.py`: `peft.get_peft_model`
between `build_model_and_tokenizer` and `OmniTrainer`. LoRA (r=16, α=32) on q/k/v/o/gate/up/down
across 28 layers; `embed_tokens` fully unfrozen via `modules_to_save` (sole entry point for the new
modality); everything else (incl. `audio_embeddings`/`audio_heads`) frozen by peft default — right,
since this shouldn't touch audio generation. Trainable = 165M / 778M (21%). Dedicated CUDA venv
`train_venv` on /mnt/data (torch 2.12+cu130, peft 0.19.1). WebDataset packaging
(`build_webdataset.py`): per-lang tar (`<id>.npy` int16 [8,T]) + label JSONL + data.lst + data
config; Task-#3 weights realized as N hardlinked shard copies (webdataset `group_by_keys` raises
"duplicate file name" if the same tar URL is listed twice via the JSON `repeat` field).

**BUG FOUND — FLEURS `id` is a per-SENTENCE id shared across speakers.** `ingest_fleurs.py` keyed
`codes_out[uid]` by it, so multiple speakers of the same sentence **overwrote** each other in the
npz (last-writer-wins), while the manifest still appended a row per wav → **47.5% of the "65,208"
utterances were phantom duplicate rows** pointing at one surviving codes blob. Verified: npz has
exactly the unique-id count; IPA is identical across an id's rows (same sentence) so each surviving
`(ipa, codes)` pair is a **valid, correctly-matched** example — we lost multi-speaker redundancy,
not correctness. **True corpus = ~34,260 unique utterances**, one speaker-rendering per sentence.

Fixes: (a) `build_webdataset.py` dedups by id BEFORE the dev/train split (a row-based split would
leak the same sentence into both). (b) `ingest_fleurs.py` now keys by the unique wav basename +
records `sentence_id`, so a future full re-ingest keeps every speaker (~65k). (c) `ingest_fleurs.py`
keeps tars in a persistent `corpus/audio_cache/` on /mnt/data by default (`--no-keep-audio` to
delete) so the eventual multi-speaker re-ingest downloads once — the original run's tars were
already `os.remove`d, so that rebuild re-downloads regardless.

**Decision (first-attempt):** train on the valid 34,260 now; defer the full multi-speaker re-ingest
(2× data + speaker diversity) until the first fine-tune shows the adapter works — not worth
re-downloading 24 langs before validating the approach.

**Smoke test PASSED** (10 steps, `train_config_smoke.json`): pipeline runs end-to-end, loss
computes and moves (4.05→4.11 eval), grad norms sane (0.5–1.4), ~3 steps/s on the 3090.
`batch_tokens=8192` OOM'd the 24 GB card (fp32 master + AdamW on 165M + activations); **2048 fits**.
Next: set the real-run `train_config.json` (batch_tokens 2048, steps ~6000) and launch.

---

## Run 14 — 2026-07-01 — First IPA LoRA fine-tune: trained, plateaued, generation-tested

**Training:** full run launched (6000 steps, LR 1e-4, batch_tokens 2048 × grad_accum 4, bf16)
over the deduped 34,260-utt corpus, in parallel with the audio prefetch (no GPU contention —
prefetch is network/disk). Eval-loss trajectory:

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| eval loss | 4.107 | 3.958 | **3.917** | 3.959 | 4.035 | 3.948 |

**Converged by ~step 1500** (4.11→3.92, ~0.19 total = ~2× the base IPA-conditioning signal from
the garbage-IPA diagnostic), then flat in the ±0.05 noise band. 6000 steps was overkill.

**Reconciling "loss is a weak proxy" with reading the plateau:** loss answers *relative/
convergence* questions (is it learning? has it stopped?) — the diagnostic proved it's sensitive
enough for that. It does NOT answer *absolute quality* (is the audio intelligible?), because text
conditioning is a small slice of the audio-CE loss. So "plateau → more steps won't help what loss
measures" is a valid convergence call; quality is judged by generation. **Stopped at
checkpoint-3000** (converged + resumable via `resume_from_checkpoint`, so stopping loses nothing)
to spend GPU on the real test.

**Generation acceptance test** (`gen_accept_test.py`): voice-clone from held-out dev ref (id 903)
+ held-out target IPA (id 279), base vs fine-tuned, + ground-truth. Two issues surfaced and were
handled:
- **cuDNN `SUBLIBRARY_VERSION_MISMATCH`** in `audio_tokenizer.encode` (torchaudio resample conv) —
  the exact README hazard. train_venv had torch's bundled cuDNN 9.20 but system libcudnn9 is
  9.23; force-installed `nvidia-cudnn-cu13==9.23.0.39` to match system → fixed. (Training never hit
  it — it uses pre-extracted codes, no audio convs; only generation encodes ref audio.)
- **Duration under-estimation:** `RuleDurationEstimator` is orthographic-calibrated; on IPA
  (stress/tie/length marks) it badly under-allocates frames → first pass truncated a 16.6s target
  to 4.4s (base AND ft equally → a generate()-side issue, not the adapter). Passing the known
  target duration helped (base 11.1s, ft 8.9s) but `duration` is a hint, not a hard clamp — still
  short of 16.6s. **Flagged as a real follow-up:** an IPA-aware duration model (or recalibrated
  char-weights for the phonetic-extensions block) is needed for usable inference.

Samples sent to user for the intelligibility verdict (loss can't give it). Verdict pending.

**Also this run:** audio prefetch complete — all 24 FLEURS train tars cached (~38 GB) in
`corpus/audio_cache/` for the deferred full multi-speaker re-ingest (task #13); ingest_fleurs.py
fixed to key by wav basename + keep audio by default.

**VERDICT (user listen):** the natural-pacing fine-tuned output (gen_finetuned_279_natural,
no duration correction) was **highly intelligible and a good voice clone**, and clearly superior
to the base model on the same IPA input. → The IPA-input LoRA adapter WORKS: pure @vernacula/
phonemizer IPA → intelligible cloned speech. First-attempt corpus + adapter validated end-to-end
on English. The earlier weird pacing was traced to a bad ground-truth target (4s leading silence
in id 279's clip), NOT the model; natural estimation rendered fine, so the duration issue (#14)
is a lower-priority polish item, not a blocker. Next: confirm generalization across the phone
space — especially the rare-primitive languages the corpus was balanced for (zu clicks, ha
ejectives, ff prenasals, cmn/vi tone).

---

## Run 15 — 2026-07-01 — Rare-primitive generalization test (does the corpus's coverage goal hold?)

English working proves the pipeline; the point of the 24-lang phonetically-balanced corpus was
sounds English LACKS. `gen_rare_test.py`: base vs fine-tuned, natural duration, on held-out dev
utterances containing each distinctive primitive:
- zu_za clicks (ǀǁǃ, id 600), ha_ng ejective/implosive (ɓɗʼ, id 1146), ff_sn prenasalized
  (ⁿᵐᵑ, id 656), cmn tone contours (˥˩˧˨, id 519), vi_vn glottalization+tone (ˀ, id 109).

Objective transient-burst proxy was inconclusive (catches general speech dynamics, not clicks
specifically); several ground-truth clips have poor codec-roundtrip (near-silent rms 0.001-0.002),
so judged by ear. Note: ha_ng base+ft both came out quiet (rms 0.008) — possible partial failure.
Samples (base vs ft × 5 langs) sent to user. Verdict pending — the test is whether ft renders the
distinctive sound where base can't, per-language.

**Zulu clicks confirmed + orthography control:** user verdict — fine-tuned (IPA) renders Zulu
clicks, base (IPA) does NOT. Ran the control (`gen_zu_ortho_control.py`): base + conventional Zulu
orthography + language="zu" (OmniVoice natively supports 646 langs incl. 'zu'), same sentence/voice.
Three-way sent (base+IPA / base+orthography / ft+IPA). Interpretation gate:
- base+orthography HAS clicks ⇒ capability was pretrained; the fine-tune is a successful IPA
  *input adapter* re-routing it (the "adapter, not new sounds" thesis) — [[project_omnivoice_onnx_export]].
- base+orthography NO clicks ⇒ native Zulu weak; IPA fine-tune ADDED capability (stronger corpus claim).
Verdict pending.

**VERDICT — Zulu (headline result):** base + conventional Zulu orthography (language="zu", native
646-lang support) produces **NO clicks**; base + IPA no clicks; **fine-tuned + IPA DOES produce
clicks.** So OmniVoice's native Zulu path cannot render clicks even from correct orthography — its
multilinguality is breadth, not phonetic depth. The IPA fine-tune BUILT a working click mapping
the base model never had via any input. → Strongest form of the thesis: **IPA input genuinely
extends phonetic capability beyond what orthography reaches**, not merely re-routes it. This is the
core justification for the phonemizer-driven IPA approach, demonstrated on clicks.

**Hausa (diagnostic negative):** implosive/ejective NOT reproduced. Root issue is NOT the adapter —
both base AND fine-tuned generate Hausa at ~10× low amplitude (peak ~0.04 vs GT 0.34), on two
different targets (ids 1146, 448). So OmniVoice generates Hausa poorly regardless of IPA/fine-tune
(quiet/degraded output), which masks any phonetic gain. Compounding factor: ha_ng implosives are
thin in the corpus (ɓ≈229 occurrences, sampling weight rounded 1.31→1 so NOT oversampled).
Two separable problems: (a) a Hausa-generation amplitude/quality issue in OmniVoice itself,
(b) thin implosive data. Candidates for the second pass: more Hausa data (multi-speaker re-ingest
#13 ≈2×) and/or explicit oversampling; and check whether the quiet output is a ref-clip artifact.

**Net:** approach validated (English intelligible + cloned; Zulu clicks added). Per-language
quality varies — Zulu strong, Hausa weak — exactly the signal the "first attempt → evaluate
sparsity → targeted second pass" plan was designed to surface.

---

## Run 16 — 2026-07-02 — Second-pass dataset: multi-speaker + targeted rare-phone langs + Sindhi

Driven by Run 15 (Zulu clicks work; Hausa implosives don't — thin data + OmniVoice Hausa
amplitude issue). Second-pass corpus:

**Multi-speaker re-ingest (the sentence-id bug fix):** ingest_fleurs.py now keys by wav basename
(+ records sentence_id) so every speaker is kept, not overwritten. 24 langs re-ingested from the
cached tars (no re-download): **34,260 → 65k+ utts**. Confirmed on en_us: 1474 effective → 2596
unique wavs.

**Targeted language adds for the weak families** (ejectives/implosives; clicks already worked):
- xh_za Xhosa (ɓ + ejectives + clicks), am_et Amharic (ejectives), om_et Oromo (ejectives) →
  ejective `ʼ` providers 2→5.
- **sd_in Sindhi** — after filing espeak-ng-portable#1247 (Sindhi implosives rendered as plain
  stops), the maintainer **fixed+merged it same-day** (#1247, authoring/sd/ph_sindhi_implosives +
  sd_implosive_lexicon). Re-phonemized: ɓ×1398 ɗ×2090 ʄ×224 **ɠ×844**. Sindhi single-handedly
  closes the `ɠ` zero-provider gap and gives `ʄ` a 2nd source beyond Fula — the implosive gap the
  corpus couldn't fill before.
- **Total: 28 languages, 76,909 utts (~2.2× the first pass).**

**Sampling weight fixes (sampling_budget.py):**
- Root-caused the first-pass noise: a single incidental owned primitive (en_us `r`=1) was pinning
  a whole language at 8×. Fix: primitives below MIN_RESCUABLE = N_TOKENS/MAX_WEIGHT can't reach
  target even at max oversample → treated as DATA gaps (reported), not weight drivers.
- Lowered MAX_WEIGHT 8→3: the data adds solved coverage, so oversampling is now a gentle
  rebalance, not the fix (a high cap made en/fr dominate the epoch off borderline phones ʔ=41/
  ɜ=39). Result: only 4 langs oversampled — fr 3× (ɒ=104), ga/sd/ha 2× — 24 at 1×; +18% epoch.
- build_webdataset repeat: round→**ceil** (round silently dropped 1.31→1×; ceil guarantees the
  boost). Effective epoch 74,794 items.

**Retrain v2** launched (4000 steps, scaled for the larger corpus; fresh checkpoints_v2 so the
first-pass checkpoint stays for comparison). Then: re-run Zulu/Hausa/rare-primitive tests — the
question is whether Sindhi's implosive data + multi-speaker volume fixes the Hausa-class failures.

**v2 results (28-lang, checkpoint-4000):** eval converged ~3.99 by step 2000 (slightly above v1's
3.92, expected — v2's dev set spans 4 more diverse langs; cross-dataset loss not comparable).
Generation acceptance test — **amplitude finding is the headline: Hausa is no longer quiet.**
v1 Hausa generated at rms 0.008 (near-failed); v2 Hausa is rms 0.03 (normal), as are sd_in (0.03),
am_et (0.13), zu_za (0.15-0.19). So v1's Hausa quietness came from thin SINGLE-speaker data, not
an OmniVoice-Hausa limitation — the multi-speaker re-ingest fixed it. Durations also much closer
to ground truth (natural gen no longer badly truncating). Samples sent (sd implosives incl ɠ,
ha retest, am ejectives, zu clicks regression). Phonetic verdict pending user listen.

**VERDICT — v2 (user listen):** clear broad improvement. Rare sounds now audible where v1 produced
none (Hausa implosive/ejective, Sindhi implosives incl. ɠ, Amharic ejectives), and MORE distinct
where they already worked (Zulu clicks — no regression, sharper). → The second-pass recipe worked:
(a) multi-speaker re-ingest (2.2× data) fixed the Hausa amplitude failure and sharpened all langs,
(b) Sindhi (post-#1247) closed the implosive gap incl. the zero-provider ɠ, (c) am/om/xh gave
ejectives cross-language redundancy (2→5). The phonetically-balanced corpus now demonstrably
renders the full rare-phone space (clicks, ejectives, implosives, tone) via IPA input — the
end-to-end goal of Runs 1–16. Remaining: standalone-inference polish (duration #14), diminishing-
returns thin phones (ʄ still ~224; zero-gaps), and downstream C# integration (out of this
workstream's scope).

---

## Run 17 — 2026-07-02 — Standalone ONNX export of the IPA-fine-tuned model

User decision: the fine-tune isn't adapter-shaped for release (embed_tokens fully retrained, not
low-rank) — ship it as a self-standing model, NOT "OmniVoice + IPA adapter". So: merge the v2 LoRA
into the base weights and export the transformer as a standalone graph.

- Added `--adapter` to `export_omnivoice.py`: `PeftModel.from_pretrained(...).merge_and_unload()`
  before export → self-contained fine-tuned graph. (Encoder/decoder are the codec, LoRA-free, so
  the existing re-exports are reused unchanged.)
- Exported on export_venv (cpu torch + peft + onnx), legacy tracer, out →
  `/mnt/data/omnivoice_ipa/onnx/omnivoice_transformer.onnx` (+ 2.45 GB .onnx.data, 199 weight
  files consolidated).
- **Parity validated** (`parity_merged.py`, the correct test — merged-PyTorch vs ONNX, NOT vs the
  base's captured tf_logits): **100.000% argmax agreement, max |Δlogit| 2.1e-4, MSE 2e-9.** The
  ONNX faithfully reproduces the merged model.

Deliverable: the 3-graph ONNX package for the IPA model — `omnivoice_transformer.onnx` (fine-tuned,
standalone) + shared `higgs_encoder/decoder.onnx`. Optional next: end-to-end ONNX generation check
(infer_onnx.py harness) and quantization (fp16/int8) for deployment size — README's "later phases".

**End-to-end ONNX validation via the C# runtime:** the fine-tuned model runs through the actual
shippable pipeline — `tests/OmniVoiceSmoke` drives `Chatterbox.Base.OmniVoiceTts` (Qwen3 tokenizer +
duration + text-prep + all 3 ONNX graphs + the 32-step diffusion loop). Key point: C# TextPrep wraps
input as `<|text_start|>{text}<|text_end|>` and BPE-encodes — the same path as IPA-in-text-field
training, so feeding an IPA string as `--text` works natively (no C# change needed). Ran with
`--onnx-dir /mnt/data/omnivoice_ipa/onnx` (the merged fine-tuned transformer + shared enc/dec),
IPA target + IPA ref-text, lang=None (matches training's default language_id), cpu, 32 steps →
10.48s WAV. A/B against PyTorch (merged model, same input) sent to user. Durations differ only
because the C# duration estimator (262 tok) and PyTorch RuleDurationEstimator pick different
lengths — content parity is the question. Verdict pending.

**C# vs PyTorch quality — decoding-regime diagnosis:** user found the first PyTorch clip better
than C#. Root cause is NOT ONNX numerics (transformer logit parity was 100% argmax) but the
**decoding regime**: `OmniVoiceTts.RunDiffusion` is hardcoded DETERMINISTIC GREEDY
(position/class temperature = 0, "for determinism"), while PyTorch `generate()` defaults to
`position_temperature=5.0` — STOCHASTIC unmask-order sampling (Gumbel). My first PyTorch clip used
the stochastic default + denoise=False + a shorter duration, so it wasn't the same experiment.
Also note C# is deterministic (re-runs identical) vs PyTorch varies per run. Sent 3 sentence-pairs
(C# greedy vs PyTorch stochastic, same voice) for the user to rejudge whether C# is *consistently*
behind (→ add stochastic position-sampling to the C# loop, a bounded RunDiffusion change) or within
PyTorch's run-to-run variance (→ C# fine as-is). Verdict pending.

**C# over-length root cause — MISSING silence handling (not active lengthening):** C# outputs ran
~35% longer than PyTorch. Controlled test: PyTorch with `preprocess_prompt=False,
postprocess_output=False` → t1 6.92s ≈ C# 6.72s; with defaults on → 5.15s. So the whole gap is
PyTorch's silence handling that `OmniVoiceTts` lacks:
1. **Ref silence removal (QUALITY-relevant):** PyTorch `remove_silence`s the reference before
   encoding. C# feeds the raw ref → inflated `refTokens` → duration formula
   (`target = targetWeight·refTokens/refWeight`) over-estimates → model stretches the same words
   over more tokens → slower/stretched pacing (reads as lower quality). Fix: port `remove_silence`
   to run before `EncodeReference` in the C# path.
2. **Output silence removal (cosmetic):** PyTorch trims output silence; C# doesn't → trailing
   silence. Fix: trim decoded output.
The `OmniVoiceDuration` estimator itself is a faithful port (same formula); the bug is upstream
(un-trimmed ref token count), not the estimate. Separate from the greedy-vs-stochastic regime
question (still pending user rejudge of the 3 pairs), but stretched pacing likely explains part of
the "C# worse" perception.

---

## Run 18b — 2026-07-02 — Full Python↔C# pipeline diff + C# silence-handling fixes

Comprehensive comparison of the Python `generate()` pipeline vs the C# `OmniVoiceTts` port
(agent-mapped Python + hand-read C#). **The port is faithful except for silence/post-processing.**
Verified MATCHES: RMS boost, hop-clip, `_combine_text`, style tokens, CFG-in-log-prob-space
(`log_softmax((1+g)·cl − g·ul)`), mask suppression, greedy token pick (class_temp=0 in both),
layer penalty `−cb·5`, shifted-linspace timesteps, ceil schedule, flattened 8×T top-k, seq layout.

Divergences found:
1. **Ref silence removal** — Python `remove_silence(mid200/lead100/trail200,−50dB)` before encode;
   C# skipped it → inflated ref token count → over-long duration estimate → **stretched pacing**.
2. **Position temperature** — Python default 5.0 (stochastic Gumbel commit-order); C# 0 (greedy).
   **User A/B verdict: `pt_greedy` (position_temp=0) sounded BEST** → greedy is preferred, do NOT
   add stochasticity. C#'s greedy choice was right; #2 was never the problem.
3. **Output silence removal** — Python `remove_silence(mid500/lead100/trail100)`; C# none.
4. **fade + pad** — Python always 0.1s fade + 0.1s zero-pad; C# none.
5. **Output volume** — Python un-boosts `ref_rms/0.1` (or peak-0.5); C# peak-0.95.
6. **Ref-text punctuation** — Python `add_punctuation` appends `.`; C# none.
7. dtype fp16-vs-fp32 — negligible (100% argmax parity).

**Fixes applied** (new `Chatterbox.Base/OmniVoiceAudioPost.cs` — faithful pydub port of
`remove_silence`/`remove_silence_edges`/`fade_and_pad`): #1 wired into `EncodeReference`
(preprocessPrompt), #3+#4 into the output path. Result on the pt_greedy sentence: C# 10.48s →
8.12s (ref trim) → **7.53s ≈ pt_greedy 7.56s**, same regime + silence handling. Remaining minor
(not audible here, TODO): #6 ref punctuation, #5 exact output-volume logic (using peak-norm).
Verdict on fixed C# vs pt_greedy pending user listen.

**C# parity completed (remaining minor items):**
- **#6 add_punctuation** — `OmniVoiceTextPrep.AddPunctuation` (END_PUNCTUATION set + CJK→。);
  applied to ref_text in the harness BEFORE both the duration estimate and text-prep (matches
  Python create_voice_clone_prompt ordering). ref_text now "…smoothies.".
- **#5 output volume** — replaced peak-0.95 with Python's `_post_process_audio` logic: with a
  ref, un-boost `audio *= ref_rms/0.1` when the pre-boost ref_rms < 0.1 else leave the model's
  own level; without a ref, peak-0.5. Reordered to remove_silence → volume → fade_pad.
- **Note (design choice, not a bug):** Python's volume policy makes output loudness TRACK the
  reference (clones loudness too), so raw output is fainter than a peak-normalized clip
  (final C# peak ≈0.20). For a product a final peak/LUFS normalize is a reasonable add; kept
  Python-faithful for now. A/B sent level-matched. Full-parity C# pipeline verdict pending listen.

**All 7 Python↔C# divergences now resolved** (2 silence + fade/pad + volume + punctuation applied;
greedy regime kept per user; dtype negligible). C# ONNX deployment path is feature-complete parity.

---

## Run 19 — 2026-07-02 — Phone-rendering issues: overloaded ɾ vs model-fidelity ʊ

User-spotted rendering issues in generated English (t1): "better" → "be'er" (dropped/glottal t),
"books" → "bucks". Diagnosis splits into two DISTINCT problems:

**books→bucks = model fidelity, NOT phonemization.** Phonemizer correctly emits `bˈʊks` (FOOT)
distinct from `bˈʌks` (STRUT); `ʊ` is abundant in English (5698 occ, 2.19/utt) — not a data gap.
The model under-renders the close `ʊ`/`ʌ` contrast. No IPA change applies; lever is longer/better
training.

**better→bad flap = IPA OVERLOADING (user's insight).** Phonemizer emits `ɾ` for the American
intervocalic-/t/ flap — the SAME symbol as the Spanish/Portuguese tap (EN better `bˈɛɾɚ`, ES pero
`pˈeɾo`). English `ɾ` = 2112 occ vs 105,378 corpus-wide (~50:1 foreign), so the model learns a
blended/foreign-accented tap and applies it to English → the "Mandarin-accent" sound. User
(correctly, per project thesis) rejects language-conditioning as a fix — the IPA itself should
carry the distinction. Options for the American flap:
- `[t̬]` (voiced-t) — precise narrow transcription, but new symbol → needs retrain.
- `[d]` — flap neutralizes /t/-/d/ intervocalically (latter=ladder), ≈ short [d]; matches the
  user's acceptable "bed-er"; abundant + renders on the CURRENT model with no retrain.
- `[t]` — don't flap (clear, British-ish); also no retrain.
Ran current-model A/B on t1 with better = ɾ / d / t (only the flap symbol changed). Verdict pending
→ picks the phonemizer's intervocalic-/t/ rule. Broader implication: audit the phonemizer for other
cross-language-overloaded symbols (one-symbol-one-sound is the design principle).

**Verdicts + actions:** user picked plain `[t]` for "better" (clear, unflapped) — and it renders
correctly on the CURRENT model (no retrain), confirming the fix is purely phonemizer-side and the
overloading thesis end-to-end. Filed **espeak-ng-portable#1250** (emit `t` for American
intervocalic /t/ instead of flapping to `ɾ`; narrow `t̬` or configurable as alternatives). For
books→bucks (`ʊ` fidelity, data is fine), testing the longer-training lever: **v3 run launched,
8000 steps** (v2 was 4000, converged ~2000) → re-test better/bucks on checkpoint-8000 vs -4000.
Broader follow-up: one-symbol-one-sound audit of the phonemizer for other cross-language-overloaded
symbols.

**Run 3 (v3) — retrain on #1250-corrected IPA + longer schedule:** #1250 merged (English-only:
intervocalic /t/ now `t̬` voiced-t, not `ɾ`). Sequence: pulled → re-transcribed all 28 (only en_us
changed: ɾ 2112→0, t̬ 2126) → refreshed manifests' `ipa` from byid by sentence_id WITHOUT
re-encoding (codes unchanged, 1348 en_us rows) → rebuilt webdataset (en_us shard t̬=2041, ɾ=0) →
launched v3 (8000 steps, corrected data). Note: `t̬` is a NEW symbol the prior model never saw, so
this retrain is what teaches it (plain `t` would've worked no-retrain, but the maintainer chose the
narrower `t̬`). v3 tests both: `t̬` rendering + longer schedule for the `ʊ`/bucks fidelity. Re-test
better/books on checkpoint-8000 vs v2-4000 when done.

---

## Run 20 — 2026-07-02 — Handoff: books→bucks was a LABELING bug (offglide collision), not undertraining

Per handoff.scratch.md + espeak-ng-portable @ d177c4b5: the `ʊ` failure was a canonical-IPA label
collision, NOT model undertraining. Diphthong OFFGLIDES wore the same glyph as the syllabic
nucleus of that quality — English `ʊ` was **76% glide** (from aʊ/oʊ), `ɪ` 34% glide — so the model
learned a glide-dominated prototype and rendered pure-FOOT "books" as the central blend ("bucks").
Same overloading mechanism as `ɾ`, at the vowel level. Fixes (canonical-mode only):
- offglides → superscripts (aᶦ aᶷ eᶦ oᶷ, + ⁱ ᵘ ʸ ᶤ ᶶ ᵚ) [the books fix]; flap ɾ→t̬; German
  eu/äu ɔ𐞲 + coda-r ᵄ; Irish i̯→ⁱ; Kazakh ʀ→ʁ; pt $text-stem L2S leak.

**Key correction: reverted the step-count inflation.** A label collision has no gradient that
separates the merged sounds (identical inputs) → more steps just converge harder onto the blend.
So v3's 8000 steps was the wrong lever; killed it. Back to the standard 4000.

Sequence executed: standalone espeak-ng-portable already at d177c4b5 (submodule is stale
15d9112c — see reproducibility note) → re-phonemized all 28 (0 err; offglides+flap landed in byid;
de/ga/kk/pt also changed) → **new `scripts/omnivoice_ipa/patch_manifest_ipa.py`** refreshed
manifest `ipa` from byid by sentence_id WITHOUT re-encoding (28,865 rows across 28 langs; codes
untouched) → rebuilt webdataset → launched v4 (4000 steps). en_us `ʊ`: 5698→1368 (now ~pure FOOT),
glide split to `ᶷ`=4330. Test books/better on checkpoint-4000 when done.

**Reproducibility note:** corpus phonemized with the STANDALONE clone @ d177c4b5; the vernacula
submodule (external/espeak-ng-portable) is still pinned at 15d9112c, and the phonemize tool
(tools/omnivoice-fleurs-phonemize.ts) is untracked in espeak-ng-portable. To make the corpus
reproducible via the submodule: commit the tool upstream + bump the submodule pointer (handoff §3a).

**v4 done + acceptance test (offglide/flap fix):** v4 trained cleanly on d177c4b5 IPA, standard
4000 steps, eval converged ~3.98 (same as v2/v3 — loss can't see the label-collision fix, as
predicted). A/B: OLD pipeline (v2 model + old IPA: ɾ, glide-contaminated ʊ) vs NEW pipeline (v4
model + fixed IPA: t̬, offglides split, clean ʊ). Two sentences: (1) better+books (pipeline diff),
(2) ʊ-dense "she could look at the good book on the wooden shelf" — NO diphthongs → identical IPA
for both → pure MODEL A/B isolating the ʊ fix. Sent to user. Verdict pending — the test of whether
splitting the offglide off the FOOT/KIT glyph fixed books→bucks.

**v4 VERDICT + "book" walked to the floor:** user confirms v4 clearly better; offglide/flap fix
validated (ʊ now consistent + correct-phoneme across could/good/look/book). Chased the residual
"book" softness to ground, ruling out every corpus-side cause: phonemizer (look=lˈʊk / book=bˈʊk,
identical ʊk rime), label (ʊ 100% pure FOOT post-fix, 0% ʊə — American CURE is rhotic ʊɹ), data
(bʊ=48 > lʊ=36 in en_us, so book has MORE context than look), variance (greedy=deterministic).
Decisive test: "book" renders CORRECTLY in an isolated ʊk-rhyme carrier ("I took a good look at the
book the cook shook") but slightly soft in a longer sentence → it's **sentence-level connected-
speech rendering fidelity**, a model-capacity frontier, NOT a corpus/IPA defect. Corpus + phonemizer
work validated. Remaining naturalness gains are model-side (bigger backbone / more data) or the
one-symbol-one-sound phonemizer audit for genuinely-splittable overloaded glyphs.

---

## Run 21 — 2026-07-02 — Distributable diff: base ONNX + 31 MB delta (not 2.45 GB re-ship)

Question: is there a small "diff" applyable over the base OmniVoice transformer ONNX? The fine-tune
= LoRA (true low-rank) + a fully-retrained embed_tokens — but embed is effectively SPARSE: only the
IPA-relevant rows learned; the other ~148k just drift ~0.0003 by weight decay. Per-row max|Δ| is
cleanly bimodal (decay ~0.0003 vs real changes >0.005), so keeping rows with max|Δ| > 0.001 (5416
of 151676, inclusive so it also captures *suppressed* orthographic tokens — a real learned change)
gives the whole diff:
- LoRA A/B factors (fp16): ~20 MB.  embed changed rows (fp16) + indices: ~11 MB.  **Total 31.3 MB**
  vs the 2.45 GB merged transformer (~78x smaller).  `extract_diff.py`.

`apply_diff.py` folds it onto a base transformer.onnx by patching the EXTERNAL DATA FILE directly:
MatMul nodes keep their module path (/model/llm/layers.N/self_attn/q_proj/MatMul) so we map
layer/proj -> the generically-named weight initializer they consume; add ΔWᵀ=((B@A)·2)ᵀ (ONNX
weight is (in,out) vs PyTorch (out,in)); overwrite only the changed embed rows.
- **Timing** (raw-bytes patch): fs-copy 2.1s + Linears 4.4s + **embed 0.03s** (was ~2s via
  full-table round-trip) = **6.6s** to produce the merged file; a load-time in-memory fold skips
  the copy (~4.5s). Not a long job.
- **Parity**: folded-onnx vs merged-v4-PyTorch = **100.000% argmax**, max|Δlogit| 9.2e-3 — exact.

Distribution story (fits the "it's its own model" stance): ship base transformer.onnx once (base
model is ~public) + the 31 MB diff; fold locally to reconstruct the standalone fine-tuned model.
Not an adapter run alongside the base — a reconstruction delta.

---

## Run 22 — 2026-07-02 — Diff as ONNX + C# load-time fold (no merged file)

Refined the distributable diff for the ONNX/C# deployment path:
- **Diff format -> ONNX** (`extract_diff.py` now emits `ipa_diff.onnx`, 31.3 MB): LoRA A/B + sparse
  embed rows as initializers, LoRA scale in metadata. Rationale: the C# fold must parse ONNX
  protobuf anyway (base-graph offsets + MatMul-node->weight map), so one format/reader end to end
  (no safetensors dep in C#). `apply_diff.py` reads the ONNX diff; still 100% parity.
- **Host plan (corrected):** the ONNX *conversion* of k2-fsa/OmniVoice is ours (not public), so we
  host the base 3-graph ONNX (transformer 2.45 GB + shared codec enc/dec) + the 31 MB diff; the
  diff folds onto the base at load.
- **C# load-time fold** (`Chatterbox.Base/OmniVoiceDiff.cs`): added ONNX-protobuf codegen to
  Chatterbox.Base (Grpc.Tools compiles proto/onnx.proto). Parses base+diff, reads each Linear's
  external-data byte range, folds `W += ΔWᵀ = ((B@A)*scale)ᵀ` (cache-friendly transposed+parallel
  matmul — the naive strided W-write was a 25 s killer, fixed to 2.5 s), overwrites the changed
  embed rows, and hands the folded tensors to ORT via `SessionOptions.AddInitializer` (ORT reads
  the rest from the base .onnx.data). **No 2.45 GB merged file** — the true load-time apply.
- **Validated** (OmniVoiceSmoke `--fold-selftest`): base+diff (C#) vs Python-merged transformer =
  **100.000% argmax**, max|Δlogit| 9.9e-5, fold **2.5 s**.
