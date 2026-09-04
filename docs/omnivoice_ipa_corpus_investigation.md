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

## Run 23 — 2026-07-28 — Re-phonemize FLEURS with vernacula-phonemizer; diff vs the espeak corpus

**Question.** The shipped IPA corpus was produced by `espeak-ng-portable` in canonical-IPA mode.
`vernacula-phonemizer` is the intended native replacement. Where do the two disagree on the same
FLEURS text, and which disagreements are *vernacula defects* rather than defensible convention
differences? (espeak is **not** ground truth here — this ranks where to look, not who is right.)

**Setup.** Repo moved to `/mnt/data/Programming/vernacula`; corpus root still `/mnt/data/omnivoice_ipa`.

- New: `scripts/omnivoice_ipa/omnivoice_fleurs_phonemize_vernacula.mts`
  (`.mts`, not `.ts` — this repo has no `type: module`, so tsx compiled the old `.ts` as CJS and
  rejected top-level await). Uses **`phonemizeAsync`**, not `phonemize`: the async entry is what
  restores unwritten vowels on the unpointed abjads (Arabic) and engages the neural OOV/tagger
  models (en, fr, bn, fa, …). Sync would have emitted an Arabic consonant skeleton.
- New: `scripts/omnivoice_ipa/compare_ipa_engines.py` → `work/ipa_engine_diff/`.
- Variety mapping — took the closest registry variety where one exists, since that is what
  production ships: `ar_eg→arz`, `es_419→es-419`, `pt_br→pt-BR` (espeak was run on the bare
  `ar`/`es`/`pt`, so part of those three diffs is dialect, not engine).

**Run.** All 28 FLEURS languages, 76k utterances, **0 errors** (2 EMPTY in `kk_kz`, both the same
duplicate id 1313). Whole corpus re-phonemized in well under a minute — Arabic is the slow path at
116 utt/s (neural diacritizer); everything else is 2–3.5k/s.

Note: FLEURS `train.tsv` repeats a transcript id across speakers, so the ~2–3.4k rows per language
dedupe to ~1.2–1.5k unique ids. Comparison is on unique ids.

**Method caveat found and fixed mid-run.** My IPA segmenter split `d͡ʒ` into two units — U+0361 is
itself category `Mn`, so the generic combining-mark branch swallowed the tie bar before the tie-bar
branch ran. Fixed (tie bar tested first) and re-ran; **metrics moved by <0.01**, so no conclusion
depended on it. Recorded because the same trap will bite any future IPA tokenizer work.

**Headline — `segments_only` distance** (stress and tone stripped, so it isolates *which sounds*
from *how they are marked*):

| tier | languages |
|---|---|
| near-agreement (<0.05) | `zu_za` .008, `ha_ng` .003, `kk_kz` .013, `ff_sn` .029, `vi_vn` .045, `fr_fr` .047 |
| convention-level (.06–.16) | `es_419` .064, `cmn` .067, `th_th` .074, `am_et` .095, `ko_kr` .108, `en_us` .112, `ja_jp` .112, `hi_in` .130, `cy_gb` .134, `ta_in` .139, `de_de` .153, `xh_za` .157 |
| large (.17–.23) | `cs_cz` .174, `ru_ru` .190, `sv_se` .201, `ca_es` .206, `ar_eg` .224 |
| very large (>.3) | `pt_br` .326, `sd_in` .398, `ga_ie` .495 |

**Reading the big three — two are *not* vernacula bugs:**

- **`ga_ie` (.495) — vernacula is more correct.** The entire diff is the broad/slender contrast:
  vernacula marks velarization/palatalization on essentially every consonant (`n`→`n̪ˠ` ×932,
  `s`→`sˠ` ×787, `l`→`l̪ˠ`, `r`→`ɾʲ`), espeak marks it sparsely. Irish consonants are phonemically
  broad/slender, so this is a real phonological distinction espeak drops.
- **`pt_br` (.326) — the *existing corpus* is wrong, not vernacula.** espeak's `pt_br` output is
  **European** Portuguese: `ɨ` reduction (`dɨ`, `nˈɔɾtɨ`), coda `ʃ` (`pɐkiʃtˈɐ̃ᶷ̃`). Vernacula gives
  proper BP — `d͡ʒ/t͡ʃ` palatalization (`nˈɔɾt͡ʃi`), l-vocalization (`sˈuw`). Top subs `ɨ`→`e` ×3192,
  `ʃ`→`s` ×1366 are exactly this. **The shipped fine-tune corpus has EP phonemes on BP audio.**
- **`sd_in` (.398) — this one *is* a vernacula defect.** See below.

**Confirmed vernacula-phonemizer defects (minimal repros verified directly, not inferred):**

1. **Sindhi has no stress at all.** `ˈ`/`ˌ` in **0 of 3443** `sd_in` lines (espeak: 3443/3443).
   Compounded by over-applied schwa epenthesis and lost vowel quality — top subs `ʌ`→`ə` ×3889,
   `eː`→`iː` ×1674, `ɪ`→`ə` ×1251. `نمائش` → `nəmaːʃə` vs espeak `nˈʊmaːˈɪʃ`; nearly every word
   also gains a final `ə`. Sindhi looks materially under-developed.
2. **Amharic and Oromo also emit zero stress** (`am_et`, `om_et`: 0%; espeak 100%).
   (`ja_jp` 0% and `cmn` 17% are *correct* — pitch accent `ꜜ` and tone letters instead.)
3. **Irish has no number expansion.** `phonemizeAsync("25 agus 21","ga")` →
   `"d̪ˠˈoː kˈuːɟ ˈaɡəsˠ d̪ˠˈoː ˈeːn̪ˠ"` — digit-by-digit ("two five"), not `fiche a cúig`.
   `"1998"` → `"ˈeːn̪ˠ n̪ˠˈiː n̪ˠˈiː ˈɔxt̪ˠ"`. English control is fine (`"25 dogs"` → `twˈɛnti fˈaᶦv`),
   so the machinery exists and Irish just isn't wired to it.
4. **Arabic lexicon annotation leaks into output.** `phonemizeAsync("كتب","arz")` →
   **`"katab/[kˈatab"`** — a raw `/[`-separated alternatives entry emitted verbatim. 11 utterances
   in `ar_eg`. Small blast radius, trivially reproducible, clearly wrong.
5. **Punctuation leaks into the IPA string** in every language, and *more* than espeak
   (`ja_jp` 4% vs 0%, `hi_in` 6% vs 0%, `de_de` 12% vs 4%). `phonemizeAsync("これは、テストです！","ja")`
   → `"ko̞ɾe̞wä , te̞sɯᵝto̞de̞sɯᵝ !"`; Catalan leaks `:` and `;`. This matters for the TTS token
   vocabulary — stray `,!.:;?` become phoneme tokens.
6. **Runaway length marks.** `"ああああ"` → `"äːːː"` (triple `ː`); `ja` also shows `e̞ːːː`, `ɯᵝːː`.

**Convention-level differences (no action — but they change the token inventory):** `e`→`ɛ` /
`i`→`ɪ` across cs/xh/ru (vernacula uses lax symbols), gemination written `kk`→`kː` (ta, ar, ru),
`v`→`ʋ` (ta), `r`→`ɾ` (am, tr), `l`→`ɫ` (ca, en), aspiration `kʰ/tʰ/pʰ` marked in en, German
offglides `aᶦ`→`aɪ̯`. Any of these silently changes the phoneme vocabulary the model is trained on.

**Implication for next step.** Three things are now decoupled:
(a) **vernacula fixes** — Sindhi (stress + epenthesis), am/om stress, Irish numerals, the Arabic
`/[` leak, punctuation stripping, length-mark clamping;
(b) **a corpus bug independent of the engine swap** — `pt_br` was phonemized as European
Portuguese and should be regenerated regardless of which engine wins;
(c) **a vocabulary decision** — even the "no action" convention diffs change the token set, so the
IPA token inventory has to be re-derived (and the fine-tune re-run) if we switch engines. That is
the expensive part, not the phonemization itself.

**Artifacts.** `work/phonemized_vernacula/byid/*.tsv` (new IPA),
`work/ipa_engine_diff/{summary.tsv,report.md,<lang>.{subs,words}.tsv,<lang>.samples.txt}`.
Per-language `.samples.txt` is worst-first side-by-side and is the fastest way to eyeball a language.

## Run 24 — 2026-07-28 — vernacula-phonemizer #547 (Sindhi) fixed; `sd_in` corpus re-generated

First of the six Run 23 defects to be fixed upstream. Full write-up lives in the phonemizer repo
(`docs/investigations/sd_native_bringup_investigation.md`, Phase 4); corpus-side effect only here.

- **Stress:** `sd_in` went from **0% → 100%** of utterances carrying a primary stress mark. Sindhi
  was the one Indo-Aryan Perso-Arabic module never wired to the shared `applyWeightStress` rule that
  hi/ur/pa already use.
- **Epenthesis:** default-ə no longer splits a homorganic nasal + stop cluster (سنڌ `sənəd̪ʰə` →
  `sˈənd̪ʰə`). Measured against the 539-word Sindhi lexicon: 53 → 41 split clusters.
- **Corpus metrics** (`compare_ipa_engines.py sd_in`): mean distance **0.565 → 0.518**;
  segments-only **0.398 → 0.395**. The segments-only number barely moves *by design* — it strips
  stress, which is what this change was mostly about. Upstream referee eval 77.0% → 77.5%.
- **Not fixed, and not fixable in code:** Sindhi short-vowel *quality* (`kət̪aːbə` vs attested
  `kɪt̪aːbʊ`). That is the abjad wall — 408 of 539 lexicon words differ on it — and only lexicon
  coverage moves it. Sindhi will remain the weakest language in the corpus after #547 closes.

`work/phonemized_vernacula/byid/sd_in.tsv` regenerated (3443 utterances, 0 errors).

**Script bug found and fixed:** `omnivoice_fleurs_phonemize_vernacula.mts` ignored positional
language args when `--limit` was absent (the `li + 1` skip dropped `argv[0]` because `li` was -1),
so `... sd_in` silently re-ran all 28 languages. Harmless (idempotent) but it made the single-language
turnaround look slower than it is. Guarded on `li >= 0`.

## Run 25 — 2026-07-28 — Two espeak-diff issues were FALSE POSITIVES; how to read the diff

Working #547–#552 (filed from the Run 23 espeak diff) surfaced a systematic error in how I read that
diff. Recording it here, because this doc is what generated the issues.

**#551 "punctuation leaks into the IPA output" — INVALID, it is a designed phrase break.**
`src/core/clauses.ts` implements a clause-pause mechanism: a punctuation token becomes a PENDING pause
rendered between phonemized tokens (`sink.pause(mark)`), deliberately never doubled and never trailing.
Each language declares its own map, and the dispatch is guarded — `const mk = CLAUSE_MARK[m[3]]; if (mk)
sink.pause(mk)` — so an UNMAPPED character is dropped by construction and only declared markers can reach
the output.

Verified empirically across 12 corpus languages: every punctuation character appearing in our FLEURS
output is a declared clause marker, **zero undeclared** (sd `,.;?` vs declared `,.;?`; ca `!.:;` vs
`!,.:;?…`; am `!,.:;?` vs `!,.:;?`). TTS front-ends want phrase-break markers, and that is what these are.

What I read as "leaking more than espeak" was simply that WE emit break markers where espeak does not.
And the `、`→`,` / `।`→`.` mapping I flagged as "normalising but not removing" is the design working:
script-specific punctuation normalised to a canonical break token.

**#548 Amharic "no stress marks" — INVALID.** 915 human referee entries mark no Amharic stress, and
espeak's own marks sit on the first syllable 99.1% of the time — a positional default carrying no lexical
information. (The Oromo half WAS real and is now implemented from a phonetic reference.)

### The reading error, and the correction
The Run 23 write-up said "espeak is not ground truth — this ranks where to look, not who is wrong", and
then I filed issues as if a difference were a defect. Three questions to ask before filing from a diff:

1. **Does the feature exist in the language?** (Amharic stress does not — check the human referees, which
   for am/om/ga carry exactly this evidence.)
2. **Is the difference OUR deliberate design?** (Clause pauses are; `git grep` the mechanism before
   assuming a leak.)
3. **Is espeak's own output informative, or a default?** (99.1%-first-syllable is a filler; measure the
   distribution rather than trusting the presence of a mark.)

Score so far: #547 real, #549 real (though my diagnosis was wrong — it was a documented stub, not
unwired), #550 real, #548 half-real, #551 not real. **#552 (runaway length marks) has not been re-examined
under these questions** and should be before any code is written — some languages contrast overlong vowels.

## Run 26 — 2026-07-28 — pt_br espeak corpus regenerated as BRAZILIAN Portuguese

Fixes the corpus bug found in Run 23: the espeak transcription of `pt_br` was EUROPEAN Portuguese.
`omnivoice_flieurs_phonemize` maps a FLEURS code to espeak's data dir by taking the first `_` segment,
so `pt_br` fell through to bare `pt` — and espeak's `pt` is EP (ɨ-reduction, coda ʃ: `pɐkiʃtɐ̃ᶷ̃`). The
shipped fine-tune corpus therefore had EP phonemes on Brazilian audio.

**Fix:** espeak-ng-portable ships a full `data/pt-br`; verified it produces genuine BP (`nˈɔɾt͡ʃi`,
`ˈĩnd͡ʒjɐ`, no ɨ) before wiring it. The script now carries a `VARIETY` map (`pt_br → pt-br`) with a
header note explaining WHY the fallthrough is wrong, so the next rebuild cannot silently regress.
Checked for other affected languages: espeak-ng-portable has **no** closer variety for `es_419` or
`ar_eg` (bare `es`/`ar` is all it ships), so pt_br was the only one of the three variety-mismatch
languages fixable on the espeak side. (Script renamed `.ts` → `.mts` — same CJS/top-level-await trap
as Run 23's companion script.)

**Result** (`compare_ipa_engines.py pt_br`, 2,793 utterances, 0 errors):

| | Run 23 (espeak=EP) | now (espeak=BP) |
|---|---|---|
| mean distance | 0.385 | **0.274** |
| segments-only | 0.326 | **0.211** |
| tok-align | 81% | 79% |
| `ɨ` in the diff | ×7,094 (top sub) | **0** |

The EP signature is gone entirely; what remains is convention-level notation (z~s voicing assimilation,
ẽn~ẽ nasal spelling, ɐ̃ᶷ̃~ɐ̃w̃ offglide, x~ʁ rhotic — both valid BP), putting pt_br in the same tier as
sv/ca instead of the pathological tier. The two engines now agree they are describing the same dialect.

**Downstream consequence, not yet done:** everything derived from the OLD pt_br espeak IPA — the token
corpus, the IPA vocabulary, and the fine-tune itself — was trained on EP phonemes for Brazilian audio and
needs regeneration. That work is already queued as part of the engine-switch re-fine-tune (Run 23 note c),
so this fix slots into that rebuild rather than triggering its own.

## Run 27 — 2026-07-28 — Post-fix re-diff: one confirmed defect left in the whole table

Regenerated all 28 vernacula transcriptions after the #547–#552 fixes and re-ran the engine diff,
this time judging every large systematic substitution against a HUMAN referee (Run 25 discipline)
instead of against espeak. Verdicts on everything ≥0.1 segments-only:

| lang | seg-only | verdict |
|---|---|---|
| ga_ie | .491 | vernacula MORE correct (broad/slender marked; espeak drops it) — no action |
| sd_in | .400 | espeak is no longer a valid referee here: our vowels are attested-sourced (kaikki/Devanagari), espeak defaults its own — diff ≠ signal |
| ar_eg | .223 | dialect (we run arz, espeak has no Egyptian) — no action |
| pt_br | .211 | convention-level since Run 26 (z~s assimilation, nasal/offglide notation, x~ʁ) |
| ca_es | .206 | **CONFIRMED DEFECT — see below** (plus l~ɫ, r~ɾ notation) |
| sv_se | .201 | judged vs wikipron: the referee writes unstressed e as **ɛ** (vatten `v a tː ɛ n`, efter `ɛ f t ɛ r`) — sides with US; espeak's ə is the outlier. ə→ɛ ×5194 is espeak being wrong. |
| ru_ru | .190 | lax-vowel notation (i~ɪ, ɑ~a, u~ʊ) — convention |
| cs_cz | .174 | e~ɛ, i~ɪ notation — convention |
| xh_za | .157 | e~ɛ, o~ɔ: Xhosa mid vowels are open-mid — vernacula right |
| ta_in | .139 | v~ʋ, geminate CC~Cː notation; ச s~t͡ɕ worth a referee look someday |
| cy_gb | .134 | vowel-notation conventions (Run 23) |
| hi_in | .130 | t→t̪, d→d̪: espeak DROPS dentality — vernacula right |
| om_et | .129 | segmental conventions; stress now implemented (mean .297→.231) |
| ko_kr | .108 | ʌ~ɘ notation + unreleased finals k̚/p̚ — vernacula narrower |

### The one confirmed defect: Catalan clitic vowel reduction
espeak `əl`, we say `ɛɫ` — and the human referee sides with espeak (`em` → `ə m`; Central Catalan
proclitics are [ə]). Affected: el, els, em, et, es, en, del, pel (ə class) and al (should stay a? — to
verify during the fix). Content monosyllables correctly keep their full vowel (mel `mˈɛɫ`).

**Root cause located** (`catalan.ts`): the engine already KNOWS these are unstressed — `FUNCTION_WORDS`
de-stresses them — but the de-stressing is a post-hoc `ipa.replace("ˈ","")` applied AFTER `reduce()`
ran with the word's single nucleus at the stress index. The mark is stripped; the vowel keeps its
stressed quality. Fix shape: for a function word, run reduction with stress = -1 (reduce every nucleus)
instead of stripping the mark afterwards. The `əl→ɛɫ` ×633 and part of `ə→ɛ` ×1184 rows fall out of it.

### Bottom line
After this round, the espeak diff is **mined out**: every remaining systematic difference is either a
notation convention, a documented dialect split, espeak being wrong (sv, hi, ga, xh), or the one Catalan
clitic bug above. Further vernacula improvement has to come from human referees and phonetic references
(the Oromo/Kamisee pattern), not from more espeak comparison.

### Run 27 addendum — the Catalan clitic defect is FIXED (vernacula-phonemizer #558)
`phonemizeWord` gained an `unstressed` flag that sets the stress index to -1 BEFORE reduction (the old
post-hoc ˈ strip removed the mark after the vowel had already kept its stressed quality). el gat →
`əɫ ɡˈat`, ho → `u`; evidence-based exceptions o/no/com keep their vowel (the referee attests o → "o",
em → "ə m"). Citation forms and the referee eval (81.3%) untouched by construction.

**Corpus effect: ca_es segments-only 0.206 → 0.159.** With that, the espeak diff is fully mined out —
every remaining row in the Run 27 table is convention, dialect, or espeak's own error.

## Run 28 — 2026-07-28 — Qualitative read of the vernacula corpus (not metrics: actual reading)

Mechanical sweep first: **zero empty outputs** across all 28 languages, no degenerate repetitions
(the 3 flagged vi_vn lines were the word ở legitimately repeating — but see below), no junk characters
(after discounting my own too-narrow IPA character class).

### The transcripts read WELL
Close-read random samples in en/de/fr/es/hi/ja/cmn/ar. The narrow features that make TTS sound natural
are present and correct: English aspiration + flapping + plausible OOV surnames (huhne → hˈʌn); French
LIAISON (les autorités → le zotɔʁite); Spanish spirantization β/ð/ɣ + seseo + yeísmo; Hindi dentals,
nasalized vowels, geminate d̪ʱː; Japanese sokuon and pitch accent; Mandarin tone letters; German
ɐ̯-vocalization and diphthongs. (One false alarm recorded honestly: en "16" looked like "sixty" in a
truncated display; it is sɪkstˈiːn.)

### Noticeably wrong, ranked
1. **German: prefix destressing misses common irregular participles.** gesagt/gemacht/bekommen/verstehen
   are correctly ɡə-/bə-/fəɐ̯-, but **gegangen → ɡˈeːɡaŋən, bedeutet → bˈeːdɔʏ̯tət, genutzten →
   ɡˈeːnʊt͡stən** — stressed long [eː] on the prefix. These are top-frequency words; a German ear hears
   it immediately. The detector plainly exists and fails on a class of stems — the highest-value fix.
2. **Silent content dropping.** Token retention by language exposed it:
   - **sd_in 93.0%** — BOTH digits and Latin words vanish (اسپتال ۾ 45 → no "45"; facebook → nothing).
     `SindhiPhonemizer` accepts a `foreign` phonemizer but the registry never passes one — a one-line wire.
   - **vi_vn 98.0%** — numbers are PERFECT (25 → hai mươi lăm) but Latin proper nouns vanish
     (paris, sofia, bulgaria dropped whole).
   - **om_et 94.5%** — numbers dropped (dhibbentaa 25 → no 25); Latin names fine.
   For TTS this is worse than a wrong phone: the audio will be missing words the text has.
3. **arz numbers are MSA, not Egyptian**: 80 → θamaːnuːn (with θ, which Egyptian lacks) rather than
   tamanīn. Understandable (numbers route through the MSA compositor) but audible.
4. Minor: French roman numerals (xviie → ksvjj); Arabic foreign names pass through the abjad with no
   vowels (سنترال → sntrˈaːl); cy numbers still digit-by-digit (the Welsh stub, noted at #549);
   ja under-segmentation fusing particles (known, #552 scope).

### Verdict
Qualitatively, yes — these are BETTER transcripts than the espeak corpus: narrower where it matters
(aspiration, liaison, spirantization, dentals, pitch/tone) and now backed by referee-validated fixes.
The remaining defects are enumerable and small: one German stress bug, three wiring gaps for
numbers/foreign words, and a handful of cosmetics. Nothing structural.

### Run 28 addendum — the fixable half SHIPPED (vernacula-phonemizer #560); long-haul logged
- **German prefix stress**: inflection-aware stress lookup + guarded prefix fallback. bedeutet →
  bədˈɔʏ̯tət, gegangen → ɡəɡˈaŋən; roots (beiden, gestern) protected by the same mechanism. 11/14 of the
  most frequent affected words were wrong; all correct now. de referee eval unchanged (78.2% — citation
  lemmas were already covered).
- **No silent content loss**: sd retention 93.0→98.1% (Latin+digits wired to English à la ur/hi, and the
  two dropped PARTICLES added: ۾ [mẽ] ×1068 kaikki-attested, ۽ aẽ ×839), om 94.5→98.5% (digits via
  English as a documented stopgap), vi 98.0→103.9% (non-syllable Latin tokens routed to English —
  paris/sofia no longer vanish).
- **Long-haul logged as issues**: #561 Arabic-variant numerals (arz 80 → θamaːnuːn is MSA with a phoneme
  Egyptian lacks); #562 text-normalization layer (ordinals, roman numerals, initialisms, dates/times,
  units, and the Welsh + Oromo number compositors).
Corpora regenerated for de/sd/vi/om. The Run 28 "noticeably wrong" list is now: fixed (1, 2), logged
(3, 4). Enhanced-corpus state is what the re-fine-tune should build from.

### Run 28 addendum 2 — Arabic foreign-name repair SHIPPED (vernacula-phonemizer #563)
Tier 1 (mater lectionis: و/ي inside an illegal run re-read as the u/i they carry in loan spellings) +
Tier 2 (epenthesis after the run's first consonant — template SELECTED against 57 attested loanword
transcriptions, not assumed). No foreignness detector: the repair keys on the (C)V(C)(C) phonotactic
signature, so native output is untouched by construction.

  سنترال بوكنج  sntrˈaːl bwknɡ → sinitrˈaːl bukinɡ    (bukinɡ = "booking")

Corpus: **931 → 0** phonotactically illegal arz tokens (2.38% → 0.00%); ar_eg regenerated. Tier 3
(diacritizer trained on mined transliterated names) deliberately not logged — future work, approach
obvious from here. The Run 28 list is now fully dispositioned: German stress ✓, silent loss ✓ (sd/vi/om),
Arabic foreign names ✓, arz MSA numerals → #561, normalization layer → #562.

### Run 28 addendum 3 — Egyptian numerals SHIPPED (vernacula-phonemizer #564, closes #561)
`numberToIpa` parameterized by variety table; egyptian.jsonc carries per-form-attested Egyptian numerals
(kaikki/wikipron/Wiktionary-arz, with the fused hundreds 300–900 flagged as pedagogical-literature-only).
80 θamaːnuːn → **tamaniːn**, 25 → xamsa **wi** ʕiʃriːn, 1998 → ʔalf wi tusʕumijːa wi tamanja wi tisʔiːn.
Three homograph traps caught (sˤafːar≠sˤifr, ʔalːif≠ʔalf, majjitiːn≠miteːn). MSA + nine other varieties
byte-unchanged; adding another variety is now a data exercise. ar_eg regenerated — zero θ/ð tokens remain.
Board: #562 (normalization layer) is the only open issue.

### Run 28 addendum 4 — English text normalization SHIPPED (vernacula-phonemizer #565, first #562 consumer)
Pure text→text pass at the top of the en pipeline; every rewrite emits words/digits the existing
number/ordinal/OOV machinery already speaks. Fixed: % and $ (previously DROPPED — silent loss), units
(40 km → kilometers; was "k-e-m"), dates (february 16 → sixteenth), times (12:05 → twelve oh five, no
spurious pause), years (in 1998 → nineteen ninety-eight), roman numerals (world war ii → two; henry
viii → the eighth; closed 2–20 set minus vi/xi, context-gated cardinal-vs-regnal). Corpus-validated:
131/1,476 en_us utterances changed, all sampled changes improvements; en_us regenerated. #562 stays
open for the other languages + the Welsh/Oromo compositors.

### Run 28 addendum 5 — multilanguage normalization SHIPPED (vernacula-phonemizer #566)
Shared symbol layer (%, currency, units — one engine, per-language data, Slavic 3-way agreement,
Turkish prefix percent, Cyrillic units for ru) wired for fr/de/es/pt/ca/cs/ru/sv/tr/ga/hi; French roman
numerals (xviie siècle → dix-septième — the exact Run 23 example — and louis xiv → louis quatorze);
Welsh compositor (decimal, mutation core, feminine mil; every base word referee-attested) and Oromo
compositor (corpus-attested core + [r]-flagged linker), both replacing digit stopgaps. 13 corpora
regenerated; a worktree before/after audit verified ONLY trigger utterances changed. Suite 1493/1493.
(This addendum was briefly committed to the WRONG repo by a wrong-cwd append — twice now with this
doc — removed there, restored here. Check `pwd` before `cat >>`.)

### Run 28 addendum 6 — Japanese particle segmentation SHIPPED (vernacula-phonemizer #567)
The #552 under-segmentation residual, both directions: fused particles let coalescence cross the
bunsetsu boundary (そのうち → so̞no̞ːt͡ɕi; now so̞no̞ ɯᵝt͡ɕi), and stranded particles carried pitch accents
(端では → häɕide̞ wäꜜ — 286 accented strands corpus-wide, a PRE-EXISTING defect the audit surfaced).
Mechanisms: extended particle sets (の/と/も/や/で + から/まで/など + run-start demonstratives, each with
a stated safety argument), particle CHAINING (では/での attach to their content word), and a pitch-layer
guard (bare particle tokens are heiban). ja_jp: 751/1,332 utterances improved; accented strands 286 → 13.
The corpus diff caught two regressions synthetic probes missed (です/できます tearing) — same lesson as
the English round: the corpus pass is the review.

## Run 29 — 2026-07-29 — Fine-tune impact audit of the unwired #562 languages; TWO number bugs found

Question: do the normalization gaps in the unwired languages (am/ar/cmn/ja/kk/ko/th/ta/vi/xh/zu, cy %)
matter for the FLEURS training pairs? Frame: the AUDIO contains whatever the speaker said for those
tokens, so a transcript that drops or misreads them is a misaligned pair.

**Symbol classes (%, currency, time, unit): 288/17,236 utterances = 1.7%** across the 12 unwired
languages (0.3%–3.0% each). In those, the number is READ but the symbol word is missing (cmn "40%" →
sìshí without 百分之; ja without パーセント) — a one-word audio↔text mismatch per occurrence. Real but
small; exclusion or per-language wiring both viable later.

**But probing every language with `phonemize("25", lang)` exposed two pre-existing bugs that dwarf the
symbol question — fixed and merged as #568:**
- **Thai dropped EVERY number**: the tokenizer matched `(\d+)` with no consuming branch. 23.4% of th_th
  utterances contain digits — all had silently lost them. The largest single silent-content-loss found
  in this entire effort. New kaikki-attested Thai compositor (script-words through the g2p).
- **Amharic dropped every TEN**: `String(t)` vs a "20"-keyed table — a one-character fix. 21.7% of am_et
  utterances contain digits; 25 read as "five", 1998 as thousand-nine-hundred-eight.

Neither was visible to the espeak diff (alignment absorbed the gap), the referee evals (no digits in
citation words), or the retention sweep (th's spaceless ratio is meaningless; am's loss fractional).
th_th and am_et regenerated.

**Residual answer for the fine-tune:** after #568, the remaining mismatch surface in unwired languages
is the 1.7% symbol-word utterances plus the year-reading question (languages that read years non-
cardinally in speech — e.g. cmn/ja digit-wise years — get cardinal IPA today; bounded by the ~100
year tokens per language). Both are candidates for utterance EXCLUSION lists at fine-tune time rather
than blockers: filter utterances matching the symbol/year trigger patterns in unwired languages
(~1–3% of data) until the words are wired.

### Run 29 addendum — FLEURS-priority symbol layer SHIPPED (vernacula-phonemizer #569)
The 13 remaining languages wired for %, currency and units, scoped to exactly what their FLEURS text
contains (the Run 29 inventory drove the word list): am በመቶ/ዶላር, arz في المئة (inserted PRE-diacritizer —
post-diacritizer injection reached the g2p as bare skeleton), cmn 百分之 prefix, ja パーセント+キロメートル,
kk пайыз + CYRILLIC км/кг, ko 퍼센트/달러/킬로미터, th เปอร์เซ็นต์ (kaikki-attested), ta சதவீதம்/டாலர்,
vi phần trăm + per-syllable units (ki lô mét), xh/zu class-prefix loans (iipesenti/amaphesenti,
iikhilomitha/amakhilomitha), cy y cant + referee-attested doler/punt/cilogram (singular after numerals),
hi +¥ येन. NOT wired, stated: xh/zu mm/cm/mi/mph, xh ¥, and clock times everywhere.

The 13-language probe caught two latent ENGINE bugs from #566 before merge (NUM space-grouping fusing
any space-separated digits; the %-prefix fallback gluing a currency remnant into a preceding number —
88% $2 → 882) plus the Arabic ordering bug. All regression-tested.

**Fine-tune impact status: the 1.7% symbol-utterance mismatch is now ~0 for the wired classes.** The
remaining exclusion-list candidates are only: xh/zu minor units, clock times (all languages), and the
year-reading nuance for languages with non-cardinal year speech. 13 corpora regenerated.

---

## Run 30 — 2026-07-29 — v5 fine-tune: retrain on the vernacula-phonemizer corpus

The engine switch goes to training. Sequence (the Run 20 recipe, retargeted):

1. **Manifest patch:** `patch_manifest_ipa.py` gained `--byid <dir>` + a per-language MISS report
   (a sentence_id absent from byid would silently keep espeak IPA → mixed-engine corpus).
   Ran with `--byid work/phonemized_vernacula/byid`: **71,964 rows refreshed across 28 langs,
   0 misses** — full engine swap, audio codes untouched.
   - Partial "changed" counts for ff/ha (and some zu/kk/th/vi rows) are NOT misses: checked
     ff_sn/ha_ng row-identity — the two engines genuinely emit identical IPA for most of those
     shallow-orthography sentences.
2. **Offglide-collision check (Run 20 regression risk):** vernacula en_us already writes
   superscript offglides (aᶦ 3403 / aᶷ 1220 / eᶦ 4132 / oᶷ 3177; zero plain aɪ/aʊ), bare ʊ=728
   ≈ pure FOOT. The books→bucks label fix carries over — no collision reintroduced.
3. **Sampling weights re-run** (`sampling_budget.py`, reads the manifests): fr 2.88→1.0 (espeak's
   scarce ɒ isn't a vernacula symbol; scarcest fr-owned is now ɥ=1094), ga 1.46→1.0, only
   **sd_in 1.34 and ha_ng 1.31 (→2× via ceil)** remain oversampled. Three census primitives now
   count 0 in-corpus — kk ʀ (vernacula uses ʁ), ca ɱ, ga ̆ — convention respellings, not lost
   sounds; reported as gaps, not weight drivers.
4. **Webdataset rebuilt** (28 langs, dev holdouts re-split same fractions), `data_config.json`
   regenerated.
5. **v5 launched:** `train_config_v5.json` = v4 verbatim (4000 steps, lr 1e-4 cosine, bf16,
   batch_tokens 2048×4 accum — Run 20 established 4000 is the right schedule; 8000 was the wrong
   lever). Output `checkpoints_v5/`. ~1.13 s/it on the 3090 ⇒ ~75 min.

What v5 has to learn that v4 never saw (Run 23 note c, now live): ɛ/ɪ lax-vowel conventions,
geminate Cː, aspiration kʰ/tʰ/pʰ, dentals t̪/d̪, ʋ, ɫ, ja pitch-accent marks + heiban particles,
sd implosives from the new lexicon/neural tier, all the #547–#569 normalization output (numbers,
%, currency, units as in-language words). Next: eval-loss check, then the acceptance battery
(gen_accept_test / rare-phone / books-better carriers) A/B'd against v4.

**v5 training complete (2026-07-29 10:46, ~2h04 wall):** eval loss 500→4000:
4.132, 4.037, 3.977, 4.008, 3.992, 3.982, 4.022, **3.966** — the same convergence
shape as v4 (3.979 final; different dev IPA so not strictly comparable, but no regression signal
and no instability from the new symbol inventory).

**Acceptance battery (gen_accept_test, checkpoint-4000, GT-duration mode):** base-vs-v5 pairs +
ground truth for three maximally-engine-changed languages, dev-held-out utterances:
- **en_us** — target opens "ˈɑːn sɛptˈɛmbɚ twˈɛnti fˈɔːɹθ sˈɛvəntˈiːn fˈɪfti nˈaᶦn…": a live
  test of the #562 date/year normalization (ordinal day + pair-wise year read) plus the t̬/offglide
  conventions.
- **sd_in** — vernacula's rebuilt Sindhi (implosives ɽ/ɗ…, dentals t̪/d̪, weight stress, 9.9k
  lexicon + neural OOV): tests the deepest single-language rewrite.
- **ja_jp** — pitch-accent marks (ꜜ), mora conventions (ɯᵝ, e̞/o̞), heiban particles: symbols the
  model has NEVER seen in v4's espeak corpus (espeak ja had no pitch marks).
Note (gen harness): stale `/home/chris/Programming/vernacula` ONNX/capture paths in
gen_accept_test.py / apply_diff.py / ingest_fleurs.py updated to /mnt/data after the repo move;
dev ids are now wav-basename keys (multi-speaker re-ingest), so the old 903/279 defaults are dead.
Wavs → train/gen_test/v5_listen/{en,sd,ja}_{base,v5,groundtruth}.wav. **User listening verdict
pending.** After verdict: ONNX export (export_omnivoice.py --adapter, checkpoints_v5) + extract_diff
→ new ipa_diff.onnx + HF re-publish, and re-derive the token-corpus dataset if the verdict holds.

**Listening feedback #1 (en): "st. james" → "street . james" — FIXED (phonemizer PR #570).**
Two defects behind one artifact: the CMU dict maps bare `st`→STREET / `dr`→DRIVE (so saint-type
uses read wrong), and the abbreviation's period leaked into the clause segmenter as a phrase
break. Fix: English normalization step 0 — dotted `st./dr./mt./mr./mrs.` resolved to words with
the dot CONSUMED; st/dr disambiguated by the NEIGHBOR test (following content word → saint/doctor
precedes a name; function word or phrase end → street/drive follows one; FLEURS is lowercased so
capitalization can't be the signal). Undotted `st` + content word → saint (st petersburg).
Known edge, unchanged: stone-weight "1 st of" still reads street (absent from prose).
Suite 1504/1504 (8 new). en_us regenerated: 12 utterances changed, ALL trigger-matched
(st james/louis/petersburg/heliers ×9, dr. tony/lee ×3) — clean isolation; manifest re-patched.
v5 saw 9 "street"-for-saint utterances in training (0.35% of en) — negligible for the model;
the fix matters mainly for live inference phonemization and the NEXT retrain's corpus.

**Listening feedback #2: "weird pauses — is there timescale manipulation?" — YES, and removed.**
gen_accept_test forced `duration = len(GT wav)/SR`, handing the model the HUMAN's pause time +
lead/trail silence as speech budget. Measured: ja GT 11.0s contains only 8.35s speech (17.7%
internal pauses); en GT 15.3s → 13.2s speech. The model doesn't know where the human paused, so
it filled the budget with drag and misplaced pauses. Three-regime A/B confirmed: ja GT-forced
11.2s vs natural ~8.9–14s vs speech-only×1.05 8.96s (tight, zero internal silence — best pacing).
**User decision: remove the force-duration mechanism altogether** — gen() no longer takes a
duration; generate() uses its own estimator. Caveat, stated: RuleDurationEstimator is
orthographic-calibrated, so on IPA it under-budgets en (8.49s for 13.2s GT speech — fast) and
over-budgets ja (14.0s — slow). Removing the forcing removes the FALSE pauses; honest-estimate
pacing is now what the test judges, and an IPA-calibrated estimator is the remaining lever
(same family as the C# duration work in Run 17).
Side findings: sd's dev ref clip is 21.2s — generate() itself warns >20s refs degrade cloning
(pick 5–10s refs for future tests); en target IPA carries no punctuation, so en's residual
clause pauses are model-learned pausing (FLEURS en audio pauses at clauses), not break markers.

**Productionize (the Run 21/22 path, retargeted to v5):**
- `extract_diff.py` → checkpoints_v5/checkpoint-4000: **ipa_diff.onnx 31.4 MB** (392 fp16 LoRA
  initializers; 5,432/151,676 embed rows > 0.001 — v4 was 5,416; v4 diff kept as ipa_diff_v4.onnx).
- `apply_diff.py` fold onto onnx_base: 6.3 s total; **parity folded-vs-merged-v5 PyTorch:
  100.000% argmax, max|Δlogit| 7.95e-3** (v4 was 9.2e-3). Known-harmless peft warning
  (audio_tokenizer semantic-encoder LoRA keys missing = untrained no-op adapters).
- **C# fold self-test** (OmniVoiceSmoke --fold-selftest): base+diff vs Python-merged =
  **100.000% argmax, max|Δlogit| 9.9e-5, fold 2.75 s** — Run 22 numbers reproduced.
- **C# end-to-end smoke** through the shippable runtime (Qwen3 tokenizer → 3 ONNX graphs →
  32-step diffusion, diff folded at load): saint-sentence target with LIVE vernacula IPA
  (ref_text also re-phonemized with the current engine) → 6.11s WAV, clean. Sent to user.
The distributable is unchanged in shape: base 3-graph ONNX (hosted once) + 31.4 MB v5 diff.
NOT yet done (outward-facing, awaiting go-ahead): HF re-publish of the diff (publish_hf.py) and
the token-corpus dataset re-publish (publish_hf_dataset.py — old one is espeak-based AND
pt_br-EP-contaminated).

**HF publish (user go-ahead):** both repos updated in place.
- Model `christopherthompson81/omnivoice-ipa-onnx`: v5 `ipa_diff.onnx` (31.4 MB) replaced v4's;
  base 3-graph ONNX unchanged (HF deduped — only the diff uploaded new bytes). Card rewritten:
  IPA source is now vernacula-phonemizer (linked), with the normalization/stress/pitch story.
- Dataset `christopherthompson81/omnivoice-ipa-corpus`: all 28 manifests re-uploaded (vernacula
  IPA incl. the saint fix; codes .npz byte-identical, deduped). Card: phonemizer credit updated,
  IPA-notes section extended (aspiration, dentals, geminates, ja pitch/mora, sd implosives,
  normalize-before-phonemize contract), example row refreshed to the real current en_us row.
- Post-publish verification: downloaded manifest_en_us.jsonl from the hub — 2,596 rows,
  sentence 46 carries sˈeᶦnt; model repo file list confirmed (5 files + card).
This closes the v5 production loop: corpus → train → parity → C# fold → smoke → published.

---

## Run 31 — 2026-07-30 — ASR × phonemization: the transcript is the SCRIPT, not what was said

**Question:** vernacula-phonemizer #562 had to choose how English reads `i.e.` and `e.g.` — the letter
names, the English gloss, or the full Latin, all interchangeable in speech. Since our output is a training
target paired with audio, the question is answerable rather than a matter of taste: what did the reader
actually say? We ship Parakeet, so ask it.

**Method.** All six `en_us` FLEURS recordings containing either form (four distinct sentences, two of them
read twice by different speakers). ASR is the production pipeline (Sortformer → Parakeet TDT v3), CPU EP.

```sh
# wav name is column 2 of the FLEURS tsv
cd /mnt/data/omnivoice_ipa/corpus/fleurs_transcripts/data/en_us
grep -h 'e\.g\.\|i\.e\.' *.tsv | awk -F'\t' '{print $2}' | sort -u
# extract ONLY those (the archive is 1.4 GB)
tar -xzf /mnt/data/omnivoice_ipa/corpus/audio_cache/data/en_us/audio/train.tar.gz train/<id>.wav
cd src/Vernacula.CLI && dotnet build -c Release -p:EP=Cpu -p:Platform=x64
dotnet run -c Release -p:EP=Cpu -p:Platform=x64 --no-build -- \
    --audio <wav> --model /home/chris/.local/share/Parakeet/models --output <out>.txt --export-format txt
```

**Raw finding.**

| recording | transcript | what the reader said |
|---|---|---|
| 8943036589905798133 | `i.e. 0 or 1` | *"values, zero or one"* — **omitted** |
| 8444646757018174763 | `i.e. 0 or 1` | **omitted** |
| 6335280368099145037 | `(e.g. in the Netherlands` | *"E.g., in the Netherlands"* — letter names |
| 12268645777003278278 | `e.g. the Pennsylvania Wilds` | *"e. g. the Pennsylvania Wilds"* — letter names |
| 9035023492553755712 | `(e.g. visa)` | *"For example, a visa"* — the gloss |
| 9748067524408569243 | `(e.g. visa)` | *"Example given a visa"* |

**Implications.**

1. **The reading question had no answer, and that is the answer.** `e.g.` gets three renderings across four
   recordings, including two speakers reading *the same sentence* differently. No target to match ⇒ the
   choice is free. (The phonemizer went with the English gloss, per preference: *that is* / *for example*.)
2. **`i.e.` was not spoken at all** by either reader — treated as unspoken punctuation. For that
   construction *any* expansion adds phonemes the audio does not contain; the letter names would have been
   exactly as wrong as the gloss.
3. **The real finding, and it is ours not the phonemizer's: the FLEURS transcript is the script the reader
   was given, not a record of what they said.** Every quality gate we have — the espeak diff (Runs 23–28),
   the qualitative read (Run 28), the phonemizer's own corpus diffs — compares text against text. Where
   transcript and audio diverge, the phonemizer can be perfectly correct and the *pair* still teaches a
   wrong alignment. 2 of 6 recordings diverged here, on the token under study, and nothing in the pipeline
   would have flagged it.

**Method caveat.** Parakeet emits normalized orthography, which is what makes it a usable arbiter of
*which* reading — it wrote `E.g.` for the letter names and `For example` for the words, and would not
invent either spelling from the other's sound. It is NOT a phonetic transcription: it cannot distinguish
reduced from full forms, and a token it drops may have been said fast rather than skipped. Two independent
readers omitting `i.e.` is much stronger evidence than one would be.

**Next step, not taken.** A **divergence audit**: ASR a whole corpus, align to the transcript, report the
disagreements — as a data-quality filter on the training pairs and as a source of normalization questions.
Costs real ASR time per language, so it wants one language measured first to find the actual divergence
rate before committing. Two things it must not assume: that divergence means the transcript is wrong (the
reader may have misread), and that a phonemizer change is the fix (often the right response is to drop the
pair). Technique for the one-off case is recorded in the phonemizer playbook, step 5b.

## Run 32 — 2026-08-17 — Re-phonemize on the normalization-layer phonemizer; hand QC before v6

**Question.** vernacula-phonemizer gained text-normalization layers across all languages since the
v5 corpus was cut (592 commits, 2026-07-29 → 2026-08-16). Re-phonemize FLEURS, hand-read every
changed entry, and decide whether the corpus is fit to retrain on.

**Setup.**
- Snapshot of the corpus v5 actually trained on: `work/phonemized_v5/byid` (copied from
  `phonemized_vernacula/byid` before overwriting — the diff has no baseline otherwise).
- Re-ran `omnivoice_fleurs_phonemize_vernacula.mts --all` (`phonemizeAsync`, unchanged):
  **28 languages, 77,584 utterances, 0 errors, 0 empty.** kk_kz's two v5 EMPTYs are gone.
- New: `scripts/omnivoice_ipa/diff_corpus_versions.py`. `compare_ipa_engines.py` is the wrong shape
  for a version bump — it samples 1200/lang and ranks worst-first because *everything* differs
  between two engines. Here almost every row is byte-identical, so this sweeps **every** row and
  dumps the changed set exhaustively (`<lang>.changed.txt` = the hand-read queue), plus per-language
  symbol-inventory deltas (new symbols = tokens the live v5 model has never seen).

**Headline: 7,108 of 40,058 unique utterances changed (17.7%).** Spread is very uneven —
tr_tr 55.6%, ff_sn 47.4%, sd_in 42.7%, ko_kr 35.1%, th_th 34.7%, vi_vn 27.4%, fr_fr 27.0%,
am_et 25.8%, ta_in 24.3% at the top; en_us 4.1%, hi_in 4.5%, ca_es 5.1% at the bottom.

### The single systemic finding: the v5 corpus read thousands separators as the word "zero"

This is not a per-language cosmetic. Reading the changed sets, the *same* defect class shows up in
essentially every language, and the audio contains none of it:

| lang | v5 said | current says |
|---|---|---|
| fr_fr | `zeʁo zeʁo` for a grouped thousand | `mil` / `miljɔ̃` |
| ru_ru | `nolʲ` | `tɨsʲət͡ɕ` |
| am_et | `1,000` → `and , zeɾo` ("one, zero") | `ʃi` |
| kk_kz | `nøɫ` | `məŋ` |
| es_419 | `seɾo` | `mil` |
| de_de | `ʊlnʊlnʊl` ("null null null") | — dropped, correct |
| sv_se | `783,562` → "…åttiotre **komma** fem sex två" | `…ɔtːɪɔtrɛtɵsɛn fˈɛmhɵndrasɛkstɪɔtvɔ` |
| xh_za | `¥2,500` → `kʼuɓˈiːni , amakʰˈuːlu amaɬˈaːnu` | `amawˈaːkʼa amaɓˈiːni … iijˈɛːni` (¥ was silently dropped) |
| cmn | `,` leaked | `t͡ɕʰiɛn` (千) |

Swedish is the sharpest case because comma genuinely *is* its decimal mark: v5 read a
3-digit-grouped number as a decimal ("seven hundred eighty-three **point** five six two"); current
resolves grouping-vs-decimal by group width and gets it right.

Alongside it, the same layer fixed times, units, ordinals, and symbol words in-language:
hi `,`→`bəd͡ʒkəɾ` (बजकर), de `,`→`uːɐ̯` (Uhr), cs `,`→`ɦoɟɪn` (hodin), ru `,`→`t͡ɕɪsof` (часов),
sv `h`→`peːrtɪ̀mːɛ` (per timme), hi `×`→`ɡʊɳaː` (गुणा), tr `ks`→`t͡ʃaɾpɯ` (çarpı),
de `bzw.`→`bəˈt͡siːʊŋsvaɪ̯zə`, fr `av. J.-C.`→`ʒezykʁi`, cy/xh `a.m./p.m.`→`ə prˈənhaᶷn` /
`kʼusˈaːsa` + `ˈɛːmv̤a kʼwɛmˈiːni`, am `ኪ.ሜ`→`kilo metɨɾ` (was `ki . me`, letters + leaked dots).

### Per-language reads worth naming

- **tr_tr (55.6%, the largest mover) — Turkish dot-ordinals, with vowel harmony.** `1.` = "birinci".
  v5 leaked a bare `.`; current emits `-inci/-ıncı/-üncü/-uncu` correctly harmonised. Turkish writes
  ordinals with a dot constantly, which is the whole 55.6%.
- **vi_vn (27.4%) — v5 was spelling Vietnamese syllables out as ENGLISH LETTER NAMES.** Top subs are
  `iːʲeᶦt͡ʃwaᶦ`→`əj`, `tʰiːwaᶦ`→`t̪əj`, `ɛɫwaᶦ`→`ləj`, `d͡ʒiːwaᶦ`→`ɣəj`: v5 read *đây/tây/lây/ghây*
  as "ay-aitch-why", "tee-why", "el-why". The `-ây` rhyme is common, so a meaningful slice of the v5
  Vietnamese corpus was garbage. Now read as Vietnamese.
- **ta_in (24.3%) — Tamil years were digit-composed.** `1980` → v5 "onru aayiram onbadu nuuru
  enbadu"; current `ˈaːjɪɾˌɐt̪ːʊ t̪ˈoɭːaːjˌɪɾɐt̪ːʊ ˈeɳbɐd̪ʊ` (ஆயிரத்து தொள்ளாயிரத்து எண்பது), the
  actual Tamil year form. `1912` → v5 "pattu irandu" (ten two) → `pˈɐnːɪɾˌɐɳɖʊ` (twelve). `-ஆம்`
  now fuses (`ˈaᶦn̪d̪aːm`) instead of splitting.
- **cy_gb — traditional Welsh vigesimal ordinals.** `16eg` → v5 `ˈɨːn dˈeːɡ χwˈeːχ ˈeːɡ` (garbage)
  → `ˈɨnvɛd ˈar bˈəmθɛɡ` ("unfed ar bymtheg"); `14eg` → `pɛdwˈɛrɨð ˈar ðˈeːɡ`.
- **sv_se — century idiom.** `1100-1200-talet` → v5 "ettusen etthundra ettusen tvåhundra-talet"
  → `ˈɛlvahɵndra tɪlː tˈɔlvhɵndratalɛt` (and the dash reads as "till").
- **ko_kr (35.1%) — cross-boundary sandhi, on top of numerals.** Top subs are ㄴ→ㄹ lateralisation
  (`n`→`ɭ` ×27), coda nasalisation (`p̚`→`m` ×35, `k̚`→`ŋ` ×29), intervocalic voicing (`p`→`b` ×24,
  `k`→`ɡ` ×19), tensification (`k`→`k͈` ×12). Also native-vs-Sino numeral selection by counter
  (`15개` → `ˈjɘɭdɐsɘt̚k͈ɛ`, native, correct) and unit idiom (`480km/h` → `sisˈok̚ …`, 시속 first).
- **ff_sn (47.4%) / sd_in (42.7%) / om_et — number compositors now emit words** (Fula
  `ɗiɗi/tati/ɡoː/sapːo`, Sindhi `həzaːɾʊ`, Oromo `kuma`) where v5 emitted digits-as-letters or
  dropped them. ff also resolves a literal `&amp;` HTML entity to "e".
- **en_us (4.1%, all good)** — `sámi` → v5 `ˈɛs mˈiː` (spelled S-M; the acute broke lookup) →
  `sˈæmi`; `müslüm gürses` → v5 `ˈɛm sɫ ˈɛm d͡ʒˈiː ɹsˈɛs` → `məslˈʌm ɡˈʊɹsz`;
  `+30°c` → v5 `θˈɝd̬iː sˈiː ˈɑːɹ` ("thirty C R") → `plˈʌs θˈɝd̬iː dᵻɡɹˈiːz sˈɛɫsiʲəs`;
  `km2` → v5 `kʰˈeᶦəm tʰˈuː` ("K-M two") → `skwˈɛɹ kəlˈɑːmʌt̬ɚz`; `e.g.` → `fɔːɹ ɪɡzˈæmpəɫ`
  (the Run 31 decision, now shipped).

### Confirmed defect, and it is the valuable one: foreign-word delegation uses the SYNC path

Non-Latin-script languages now read their Latin-script proper nouns instead of dropping them —
right call, the audio contains those words and v5's target did not (am_et `national hurricane
center` and `danielle` were both simply absent from the v5 target). But the delegated IPA is
**degraded**, and the reason is exact:

```
                 phonemizeAsync(w,"en")      phonemize(w,"en")      what the corpus contains
liguria          ləɡjˈʊɹiʲə                  lˈaᶦʊɹiʲə              lˈaᶦʊɹiʲə     (ɡ deleted)
adekoya          æd̬əkʰˈɔᶦə                   ˈædŋkoᶷjˌɑː            ˈædŋkoᶷjˌɑː   (illegal dŋ)
riomaggiore      ɹiʲoᶷmˈæd͡ʒiʲɔːɹ              ˈɛɹiʲoᶷmˌæɡɪŋˌɔːɹ      ˈɛɹiʲoᶷmˌæɡɪŋˌɔːɹ
caboolture       kəbˈuːɫt͡ʃɹ                   kʰˈeᶦbuːɫt͡ʃəwɹi        kʰˈeᶦbuːɫt͡ʃəwɹi
sezen            sˈɛzən                      sˈɑːʃɛn                sˈɑːʃɛn
pbs              pʰˈiːbiːz                   pz                     pz
```

The corpus matches **sync byte-for-byte in every case**, including inside a synthetic Thai host
sentence. So the host→English delegation is not calling the async entry, and every delegated token
loses the neural OOV model — the exact downgrade `omnivoice_fleurs_phonemize_vernacula.mts`'s header
warns about for the top-level call. **Blast radius: 1,500 of 15,543 unique non-Latin-script
utterances (9.7%) contain a Latin token** — am_et 8.2%, ar_eg 2.7%, cmn 16.5%, hi_in 3.8%,
ja_jp 8.7%, kk_kz 10.2%, ko_kr 15.2%, ru_ru 8.8%, ta_in 7.4%, th_th 15.2%, sd_in 8.2%. Fixing this
upstream and re-running costs ~1 minute of phonemization, so it should land before v6 is cut.

**Second, smaller defect: initialism reading.** `tb`→`tʰˈiːbˈiː` ✓ and `aol`→`ˈeᶦˈoᶷˈɛɫ` ✓, but
`pbs`→`pʰˈiːbiːz` ("pee-beez") and `xdr`→`ˈɛkdɹ` — both wrong even on the *async* path, so this is
independent of the delegation bug. Should be "pee-bee-ess" / "ex-dee-ar".

### Open design question the diff cannot answer

Delegated foreign words get **American English phonology inside a non-English utterance** —
`sezen aksu` (Turkish) as `sˈɛzən ˈæksuː` in Korean and Thai text. A Korean reader says that name
with Korean phonology. Per Run 31, this is answerable from the audio rather than by taste, and it is
the one place a targeted wav2vec2/ASR probe earns its cost: sample the ~20 utterances where a Latin
proper noun sits in Korean or Thai audio and check which phonology the reader used. Everything else
in this diff is self-evidently better than v5 and does not need an acoustic arbiter.

### Mechanical sweep

0 empty rows in all 28 languages. Degenerate 3-repeats: cmn 12, ta 3 — both checked:
- **cmn is a false alarm** (as in Run 28): `1444年` → `ji sɹ̩ sɹ̩ sɹ̩` is the correct Chinese
  digit-by-digit year reading.
- **ta 157 is a real (3-row) defect:** `us$11.000 … us$22,500` — the dot-grouped `11.000` read as a
  decimal (`pˈʊɭːɪ pˈuːd͡ʒːɪjɐm ×3`, "point zero zero zero") even though the sibling number in the
  same sentence uses comma-grouping. Same family as the phonemizer's current separator work. Also
  `us$` reads as `ˈʌs` with the `$` dropped.

**Punctuation leak (Run 23 defect #5) narrowed but not closed.** `:` (×123) and `;` (×35) are gone
corpus-wide — collapsed into `,`. Remaining stream punctuation is `, . ! ?` only: **5,850 marks
across 4,247 utterances**, worst in am_et (816), ff_sn (663), sd_in (567), de_de (386).
`phonemizeAsync("yahoo!","en")` → `jˈɑːhuː !` and the ja `、`/`。` leaks both reproduce. These still
become phoneme tokens in the fine-tune vocabulary.

**German dot-ordinals are NOT handled** (contrast Turkish, which is): `am 16. februar` →
`am zˈɛçt͡sen . fˈeːbʁuaːɐ̯` — cardinal plus a leaked dot, not "sechzehnten". `der 3. mai` →
`deːɐ̯ dʁaɪ̯ . maɪ̯`. 103 de_de source utterances match `\d{1,2}\.\s`.

### Verdict

The corpus is materially better than v5 and the v5 numeral defect was bad enough that a retrain is
justified on its own. Nothing structural is wrong. Before cutting v6 I would land two cheap upstream
fixes — **foreign-word delegation → async** (9.7% of non-Latin-script utterances) and German
dot-ordinals (103 utterances) — since re-phonemizing is ~1 minute and both change the training
target. The initialism and `us$`/dot-grouping defects are small enough to ship around.

**Artifacts.** `work/phonemized_v5/byid/` (v5 baseline snapshot),
`work/phonemized_vernacula/byid/` (current), `work/ipa_version_diff/{summary.tsv,report.md,
<lang>.changed.txt,<lang>.subs.tsv,<lang>.symbols.tsv}`. `<lang>.changed.txt` is the hand-read queue.

**Not yet done:** manifest patch, sampling-weight re-run, webdataset rebuild, v6 train. Warm-start
vs fresh is still open — see the note below.

## Run 33 — 2026-08-17 — Both upstream fixes landed; and the audio says readers NATIVIZE

Run 32's two blockers fixed in vernacula-phonemizer (branch `norm/foreign-async-oov`, not pushed),
corpus regenerated, and the foreign-word routing question put to the audio.

### Fix 1 — foreign-run delegation now uses the neural English path

Not the one-line change it looked like. `core/foreign.ts` types its reader `(text: string) => string`
**on purpose**: the host stack it maintains is only correct inside one synchronous turn, so the
delegation cannot simply become async. The seam that did work is the one `phonemizeEnNeural` already
uses on itself — resolve OOV readings asynchronously *first*, then hand them to the sync render as an
`oovOverride`. So `phonemizeAsync` now tags the Latin words ahead of the host's render and memoizes
them for the sync reader.

A **plain memo**, not a scoped override: the tagger reads a bare lowercased g2pKey, so a reading is
context-free and there is nothing to restore — none of the interleaving hazard that constrains the
host stack. Consulted only on the foreign path, so `phonemize()` is byte-identical regardless of what
ran before it (asserted in the new test).

**Two invisible failures on the way, both worth recording:**
1. `this.text` inside the new `EnglishPhonemizer.textWithOov` resolved to the **one-argument wrapper
   `getPhonemizer` shadows onto the instance**, which silently drops arguments two and three. No
   error, just the old reading — exactly the trap registry.ts documents for wrapper objects. It calls
   the prototype method instead.
2. **Fixing `setDefaultForeign` alone fixed almost nothing.** `emitUnclaimed` asks the script router
   FIRST and `SCRIPT_TARGET.Latin` is `"en"`, so that is where embedded Latin actually goes; and ~46
   engines that claim Latin themselves (mandarin, hindi, sindhi, amharic, vietnamese, …) are handed
   the reader at construction and reach neither path. One `readAsEnglish` now serves all three.

### Fix 2 — German dot-ordinals: the month test was case-sensitive

Not a missing feature — a **casing** bug. `am 16. Februar` was already correct all along; only
`am 16. februar` failed, because the `ORDINAL_NOUN` test ran case-sensitively against capitalised
month names and FLEURS ships German lowercased. Folded case on that condition only; the second
condition still requires a capitalised noun because it is the one that has to reject a sentence-final
`N.`, and on lowercased text capitalisation is the sole signal — the same wall the English `st./dr.`
work hit in Run 30. Also added a **≤ 31 guard**, which closes an over-fire the fold would have
widened (`im Jahr 1998. Mai war warm` read 1998 as an ordinal): a German day and a century are both
≤ 31, and all 100 such ordinals in the corpus are, so the guard is free — and it fixes the
capitalised form too. Suite **4751/4751** (8 new tests across the two fixes).

### Isolation audit — the fixes changed only what they should

Re-phonemized all 28 (0 errors, 0 empty, 77,584 rows) and diffed against the pre-fix tree
(`work/phonemized_prefix`, `work/ipa_fix_audit`):

- **16 of 28 languages changed by exactly 0 rows** — every Latin-script one (ca cs cy en es ff fr ga
  ha pt sd sv tr xh zu).
- Of the 12 that moved, **100% of changed rows contain a Latin token**: th 103, ko 80, cmn 123, am 52,
  ru 49, ja 42, kk 65, ta 33, hi 26, ar 17, vi 185, om 1. No collateral.
- **de_de: 50 unique rows, 50 of 50 trigger-matched** (`am 6. juli` → zˈɛçstən, `am 10. august` →
  t͡sˈeːntən; leaked ` . ` pauses gone). The one my grep first flagged as untriggered is
  `des 18. jahrunderts` — the corpus's own misspelling, which is deliberately in `ORDINAL_NOUN`.
- Quality of the recovered readings: `maroochydore` mˈɑːɹɔːˌɑːʃˌɔᶦd̬ˌɔːɹ → mɚˈuːt͡ʃid̬ˌɔːɹ, `noosa`
  nɔːoᶷzˈɑː → nˈuːsə, `janissary` jˈænɪəsˌɑːɹ → d͡ʒˈænəsˌɛɹi, `safina` sˈeᶦfinˌɑː → səfˈiːnə, and
  `hesperonychus` recovers a dropped initial h. Two small regressions, both Turkish names read by an
  English model (`fatih` fətʰˈiː → fˈeᶦt̬ɪ, `erkoç` ˈɝkʰɑː → ˈɝkɑːk) — which is the routing question
  below, not the reader's quality.
- Punctuation after the fixes: 5,749 marks in 4,242 utterances, still only `, . ! ?`.

### The routing probe — wav2vec2, and the answer is NATIVIZATION

New: `scripts/omnivoice_ipa/probe_foreign_phonology.py`. Parakeet is European-only, so this uses
**`facebook/wav2vec2-xlsr-53-espeak-cv-ft`** — a multilingual IPA phone recognizer, language-agnostic
by construction and so not presupposing either answer. 21 utterances across ko/ja/cmn/ta/ru/am/th,
selected because the transcript **begins** with the Latin token, which makes the region of interest
the head of the phone string and removes the need for any alignment.

(Mechanics note: `Wav2Vec2PhonemeCTCTokenizer` hard-requires the `phonemizer` package in its
constructor, but only for the *encode* direction. Loading it `do_phonemize=False` skips that backend —
so the probe needs no espeak, and no second phonemizer inside the loop it is refereeing.)

| lang | token | what the corpus now targets | what the reader said | |
|---|---|---|---|---|
| ja | global running | ɡ**l**ˈoᶷbə**ɫ** **ɹ**ˈʌnɪŋ | ɡ**ɾ**oːba**r**dan**ɲ**iŋɡ**oː** | Japanese: ɾ/r for l, oː, epenthesis |
| ja | modern education | mˈɑːd̬**ɚ**n ˌɛd͡ʒəkʰˈ**eᶦ**ʃən | muːdan eduk**eː**ʃʊn | no ɚ, no d͡ʒ, eː |
| ko | atlanta thrashers | ætlˈæntə **θ**ɹˈæʃ**ɚ**z | atlɑnta **t**rɛnʃ**ro**s | θ→t, æ→a, no ɚ |
| ko | palm / commons | pʰˈɑːm / kʰˈɑːmənz | pam / komos | short vowels |
| cmn | metroplus | mˈɛtɹoᶷpləs | mei**5**ts.ou**5**plɑ**5**s | Mandarin syllables, **tone-marked** |
| cmn | cell / lockwood | sˈɛɫ / lˈɑːkwʊd | siɛ5 / lɑu5xu5t | tone-marked |
| th | fernando alonso | f**ɚ**nˈændoᶷ əlˈɑːnsoᶷ | f**oː**nand**oː** alans**oː** | no ɚ, long monophthongs |
| th | lodin | lˈoᶷd̬ɪ**n** | loːdɛ**ŋ** | Thai coda nasal |
| ta | myspace | mˈaᶦspeᶦs | maɪ**ji**speːs**a** | Tamil epenthesis + final a |
| ru | myspace / hokuriku | mˈaᶦspeᶦs / **h**ˌɑːkɚˈiːkʰuː | maɪ**jɪ**speːs / **x**okoriki | Russian epenthesis, x for h |
| am | whistler | wˈɪsl**ɚ** | wist**e**la**r** | Amharic epenthesis |
| th | **kier starmer** | kʰˈɪɹ stˈɑːɹm**ɚ** | kiːəstɑːm**ɚ** | **English — the one clear case** |

**~18 of 21 nativized.** The divergence is not random: what disappears is precisely the set of
English-only phones (ɚ ɝ θ æ, the oᶷ/eᶦ offglides, /l/ vs /ɹ/) and what appears is host machinery
(epenthesis, coda-nasal substitution, Mandarin tone letters, monophthongal long vowels). A
recognizer's noise does not manufacture a systematic pattern in that shape. Caveats stated plainly:
21 utterances, and the recognizer has its own espeak-flavoured biases — this establishes a direction,
not a rate.

**So routing embedded Latin to American English gives the model a target the audio does not contain.**
The right target is host-nativized, which is a real phonemizer feature (per-language loanword
adaptation), not a v6 blocker. Logged as future work; v6 trains with English delegation, now at least
with the *good* English readings.

**Second finding, unlooked for, and it corroborates Run 32's other defect acoustically.** Two of the
21 are initialisms, and the readers said **letter names** while the corpus targets a word:

| token | corpus target | reader | |
|---|---|---|---|
| `acma` (am) | ˈækmɑː | esiːeme | "A-C-M-A" |
| `rspca` (ta) | ɹspkˈɑː | arʌsbilθienuː… | "ar-es-pee-see-ay" |
| `wned` (ja) | **wn**ˈɛd | dabudjoeniːdiː | "double-u-en-ee-dee" |

That upgrades the initialism defect (Run 32: `pbs`→pʰˈiːbiːz, `xdr`→ˈɛkdɹ) from cosmetic to
**measured wrong against the audio** — and `ɹspkˈɑː` / `wnˈɛd` are unpronounceable onsets besides. The
rule the audio supports: an all-consonant or otherwise unpronounceable letter run is read as letter
names.

**Root cause found, and it is NOT a quick fix — deferred past v6 deliberately.** The initialism pass
matches `\p{Lu}{2,}` — **capitals are the signal**, and FLEURS is lowercased, so `pbs`/`xdr`/`rspca`/
`wned` never enter the pass at all. Third instance of the same wall (German ordinals here, English
`st./dr.` in Run 30): *lowercased input has no casing signal*. The alternative signal is already sitting
right there — `isUnreadable` (no vowel, or an illegal onset/coda) is case-independent, and `isRecorded`
guards real dictionary words — so extending the match to lowercase unreadable runs is the right fix.
But `core/initialisms.ts` is **shared by ~190 engines** (it is what reads French TGV and Russian США),
so widening its matcher is a fleet-wide change and wants this repo's fleet-audit treatment, not a
bolt-on. Logged as its own pass. v6 ships with the handful of word-read initialisms; blast radius is
~a dozen utterances corpus-wide, against the ~7,100 rows v6 fixes.

**Artifacts.** `work/phonemized_prefix/` (pre-fix baseline), `work/ipa_fix_audit/` (isolation audit),
`work/asr_probe/{candidates.tsv,phones.tsv,en_readings.tsv,wav/}`.

## Run 34 — 2026-08-17 — Initialism casing: repair the INPUT, and let the phonemizer's own pass fire

**Correction to Run 33's disposition.** I deferred the initialism defect on the grounds that fixing it meant
widening `core/initialisms.ts` (shared by ~190 engines) and that this wanted a fleet audit. That was the
wrong framing, and deferring it unilaterally right after finding acoustic evidence for it was the wrong
call — **v6 was launched before raising it, and had to be stopped and relaunched.** The user's framing:

> fix the input by uppercasing the unpronounceables (the ones that make sense to, anyway) with a QC gate.

The pass is gated on `\p{Lu}{2,}` because **capitals are its signal**. FLEURS destroyed that signal by
lowercasing. So the defect is in our corpus preparation, not in the phonemizer — and the repair belongs in
our pipeline, where it costs the fleet nothing and reuses a pass already tested per language.

### The per-language phonotactics correction

The first scan used `isUnreadableEnglish` for all 28 languages and produced **2,164 candidates / 12,811
occurrences**, mostly ordinary native vocabulary. The user named the reason:

> Shouldn't unpronouncable be language-idiomatic by how each language defines its vowels?

Yes. `makeUnreadableTest` is *parameterized* by `PhonotacticsData` — each language declares its own vowels,
legal onsets and legal codas — and **38 languages ship one**. Welsh spells a vowel ⟨w⟩ (`bwrdd`, `cwmwl`);
Czech has syllabic r/l (`smrt`, `skrz`); Irish has its own clusters (`bheith`). Each was "unreadable" only
because an English test was asked a question about Welsh. Rewired to each host's own test, with English as
the fallback — which for the non-Latin-script hosts is not a compromise but the *correct* question, since a
Latin run there is foreign by definition and is delegated to English.

**2,164 → 1,464 candidates (8,178 occurrences) mechanically**, and the flood drained where predicted:
`mewn`, `roedd`, `oedd`, `sydd`, `bwrdd`, `cwmwl` (cy), `bheith`, `raibh` (ga), `tsarin` (ha), `nder` (ff)
all fall out on their own language's phonotactics.

**Byproduct finding, upstream-reportable: some tables have gaps.** Native words that survive their *own*
language's test point at missing legal clusters — `nicht`/`gibt` (de: `cht`, `bt` are legal German codas),
`jsou`/`kde` (cs: `js`, `kd` are legal Czech onsets), `nhw` (cy: `nh`), `bhfuil` (ga: `bhf`). Separately
`vi`/`xh`/`zu` are Latin-script with **no table at all**, so they still get the English fallback and still
over-select; they contribute 293 (xh), 175 (zu), 80 (vi) of the residue, flagged `approx` in the output.

### The discriminator that works, and the gate

Cross-language spread. An international abbreviation appears as a Latin run in twenty-odd different corpora;
a native word appears in one. `utc` 27 langs, `pbs` 26, `gp` 26, `rspca` 24 — against `mewn` 1, `khi` 1.

Reviewed the 46 candidates at spread ≥4 by hand, in context. **29 accepted, 17 rejected**, and the rejections
are why a predicate cannot do this alone:

| rejected | why |
|---|---|
| `km` `cm` `kg` `kph` `sq` `mbit` | **units** — want "kilometres", not K-M; the unit layer's job |
| `zmapp` | drug name, read "zee-map"; Z-M-A-P-P is wrong |
| `jagr` `angkor` `rossby` `dzong` `bhajan` … | names and words the English test mislabels |
| `rr` | metalinguistic — the sentence is *about* Spanish ⟨rr⟩; a reader trills it |
| `rd` | wants the word "road" |
| `isn` `didn` `wouldn` `wasn` `hadn` `doesn` `couldn` | **a bug in my scan, not the corpus** — FLEURS keeps the apostrophe (`didn't` is intact in col3); my run regex treated `'` as a boundary and split it into `didn` + `t`. I first recorded this as a FLEURS defect; it is not. Fixed with a `(?!'\p{L})` lookahead, one-sided so Catalan `l'adn` still matches. |

Short tokens were checked occurrence-by-occurrence rather than assumed: every `tt` is *audi tt*, every `gp`
is *a1 gp*, every `hk` is *hk management*, every `cg`/`kv` is the code *cg4684*/*kv62*, every `qc` is
*starmer qc*. No word collisions.

**Applied, and audited exactly as Run 33 was: 614 rows changed, 100% of them containing an allowlisted
token — zero collateral.** Readings: `afcfta` ˈæfkftɑː → ˈeᶦ ˈɛf sˈiː ˈɛf tʰˈiː ˈeᶦ; `utc` ˈʌtk →
jˈuː tʰˈiː sˈiː; `rspca` ɹspkˈɑː → ˈɑːɹ ˈɛs pʰˈiː sˈiː ˈeᶦ (the audio-confirmed reading).

**And it nativizes for free, which Run 33 called the harder problem.** Korean `adt` → `ˈeiditʰi` — *Korean*
letter names (에이디티), not English, because uppercasing lets the **Korean** engine's own initialism data
claim the run. The nativization gap closes here for initialisms specifically.

### Judging the remainder with a local model

Reading the residue by hand is not a good use of the reading; discarding it unread is not safe either, since
a language-specific abbreviation is legitimate and has spread 1. Per the user's suggestion, a local
Qwen3-27B (llama-server) triages in batches with **JSON-schema-constrained output**, verdicts
{LETTERS, WORD, UNIT, EXPAND, UNSURE} — the question is lexical rather than phonetic, which is what a model
can actually answer, and the example sentence carries the context that settles most cases.

**⚠ Trap, and it cost a wasted run: Qwen3 is a REASONING model.** Thinking left on, it spends the whole
`max_tokens` budget in `reasoning_content`, returns `content: ""` with `finish_reason: "length"`, and every
row comes back `NO_REPLY`. I launched the sweep without testing one request first, and the user caught a file
of nothing. Fixed with `chat_template_kwargs: {enable_thinking: false}`, tested on one request before
relaunching, and verified against knowns: `nhw` → WORD "Welsh pronoun", `xv`/`xx` → WORD "Roman numeral",
`bzw` → EXPAND "beziehungsweise", `bwlb` → WORD "Welsh bulb".

**The model triages; it does not decide.** Its LETTERS verdicts are a proposal that still gets read before
anything enters `INITIALISM_UPPERCASE`.

**Artifacts.** `scripts/omnivoice_ipa/{scan_initialism_candidates.mts,initialism_casing.mts,judge_initialisms.py}`,
`work/initialism_gate/{candidates.tsv,verdicts.tsv}`, `work/phonemized_preinit/` (pre-repair baseline),
`work/ipa_init_audit/` (the 614-row audit).

## Run 35 — 2026-08-17 — Fix everything the QC found, upstream; second pass finds a bigger one

User direction, after I twice launched v6 on a corpus with known defects: **fix everything first, then a
second QC pass, and confirm before training.** Both launches were killed and their checkpoints deleted.

### Every triage bucket is a work item

I had used the local-model sweep as a `LETTERS` filter and discarded the rest. The user's correction —
each verdict implies an action — turned the ignored buckets into the largest findings of the run:

| verdict | n | action | outcome |
|---|---|---|---|
| LETTERS | 52 | uppercase the input | 29 + 34 accepted after review |
| **WORD** | 1,287 | phonotactics gap / lexicon miss | **the digraph bug, below** |
| UNSURE | 40 | second pass by a bigger model | 5 promoted (`rmn`, `mrt`, `osn`, `bm`, `dda`) |
| EXPAND | 21 | abbreviation dict | 9 dot-fixable, 2 added, rest disposed |
| UNIT | 6 | unit table | 1 added (`zu kma`), rest were already-correct words |
| NO_REPLY | 2 | re-ask | 3 of 5 recovered; resume no longer cements a non-answer |

### The digraph bug — the unreadable test counted LETTERS

`makeUnreadableTest` models phonotactics but reads orthography, and nothing told it which letter pairs
spell ONE phoneme. Diagnosed by parsing each language's own table out of source and reimplementing the
four signals to ask *which* fired — worth doing, because my first guess (missing onsets) was wrong for
Welsh, which already had `ch dd ff ll ph rh th`.

    de  nicht, nacht   `cht` scored a 3-consonant run    ⟨ch⟩ is /x/
    ga  bhfuil         `bhf` likewise                    ⟨bh⟩ is one lenited phoneme
    ca  anys           `nys` likewise                    ⟨ny⟩ is /ɲ/
    ca  lloc           onset `ll` unlicensed             ⟨ll⟩ is /ʎ/
    cs  smrt, skrz     signal 1 "no vowel"               syllabic r/l are nuclei

Added an optional `digraphs` to `PhonotacticsData` doing three jobs: collapsed to one placeholder
consonant before the run test, and automatically legal in onset and coda, since one phoneme needs no
cluster licence. Wired for cy/cs/de/ca/ga/ff/sv/om plus each table's corpus-attested missing clusters.
**Candidates 1,464 → 864; every motivating word clears.** `ghraib` (Abu Ghraib) stays flagged, correctly.

Two over-additions of mine, caught by the suite: Oromo long vowels (aa/ee/ii/oo/uu) are NOT consonant
digraphs and must never collapse to a consonant, and `mr` is not an Oromo onset — it came from one
foreign name and stopped `MRI` being spelled out.

⚠ **This bug does not touch our corpus.** `isUnreadable*` is consumed only by the initialism pass, which
is gated on capitals and never fires on lowercased FLEURS. Latent, real, now fixed.

### xh/zu had no initialism handling at all — and the letters are CLICKS

`UTC` read [ˈuːtʼkǀ], `PBS` read [pʼɓs]: c, q and x are click letters, so an acronym reaching the g2p raw
is confidently wrong rather than mute. Both headers say so; xhosa's also records why nothing was done —
*"no era phrase and NO LETTER NAMES. Both are refusals for want of a source."*

The source question is answerable in the other direction: acronyms in isiZulu/isiXhosa are English
borrowings kept in capitals, so what is needed is the ENGLISH letter series, *adapted* — written in Nguni
orthography and read by this language's own g2p.

**Validated against the audio** (wav2vec2, utterances whose acronym falls early enough to isolate):

| utterance | recognized | reading |
|---|---|---|
| xh `i-usa gymnastics` | `e·ju·e·se·dʒimnestiks` | U-S-A as letter names |
| zu `umboniso we-pbs` | `umbonisowe·pi·pi·es` | P-B-S |
| zu `…bokuthula be-un` | `…be·ju·ena` | U-N |

Every spelling avoids c/q/x (`si` for C, `khyu` for Q, `eksi` for X) and uses aspirated bh/ph/th/kh, since
bare b is implosive /ɓ/ and p/t/k are ejective. Three ordering constraints, all found by the suite: the
rule must run LAST (the currency/era/degree rules own capitals of their own — running first cost
`ku-US$30` its *amadola*); `$` must be in the trailing guard (`US$` is an ENGINE-tier key); and COVID
needs a word-acronym exemption.

**This also solved the Bantu concord problem** I had written off as unfixable-by-casing. The xh/zu rule
deliberately allows a lowercase letter before the capitals, so uppercasing only the acronym half works:
`yepbs` → `yePBS` → *ye* + P-B-S, click-free. All 16 concord forms repaired.

### The casing wall, fourth and fifth sightings

A rule keyed on capitals silently declines on lowercased input, and lowercased input is what corpora
ship. After the German ordinal detector and the English `st./dr.` test: **de `356 v. chr.`** read *f . kʁ*
and **sv `1000 f.kr`** reached the g2p as [kr] — both era rules were case-sensitive. Now folded. A fifth,
left alone deliberately: xh `Mnu.` is already case-insensitive but demands a following CAPITALISED name,
and that capital is the only thing stopping the rule over-firing.

### Corpus state

All repairs are input-side (`initialism_casing.mts`): casing for 68 tokens, abbreviation dots for 11
across de/sv/cs/fr/tr, Nguni concord splits for 16. **755 rows changed vs pre-repair, 100%
trigger-isolated.** Against the v5 corpus the live model trained on: **7,784 of 40,058 unique rows
(19.4%)**. Mechanical: 0 empty, 0 clicks outside xh/zu, punctuation down to 5,726 marks in 4,231
utterances.

### ⚠ The second QC pass found something bigger than everything above

xh/zu are Latin-script, so their tokenizer claims English words outright — there is no foreign-run gap
for `emitUnclaimed` to fill, and no `ForeignPhonemizer` injected. So an embedded English word is read by
the Nguni g2p, and c/q/x become CLICKS:

    national hurricane center   →  … hurrikǀˈaːnɛ kǀˈɛːntʼɛr
    china, manchester city, alexander, arctic, factor, sciences

Measured against the English lexicon (token is English-dict AND contains c/q/x, so a Nguni word with a
genuine click cannot be miscounted): **xh 290/1,509 utterances (19.2%), zu 219/1,478 (14.8%)** — 509
utterances, an order of magnitude more than every other remaining defect combined (~44). The acronym fix
above does not touch it: these are words, not letter runs.

The fix is foreign-word routing for xh/zu — the `ForeignPhonemizer` injection ~46 other engines already
take, gated on "in the English lexicon and not a Nguni word". Not attempted here; surfaced for a decision
before v6.

**Commits (vernacula-phonemizer, branch `norm/foreign-async-oov`, not pushed):** `4f89caa` digraphs,
`5e5ce11` xh/zu letter names, `dedc5b0` era casing + data. Suite 4751/4751 throughout.

### Run 35 addendum — xh/zu foreign-word routing SHIPPED; third QC pass clean

The 509-utterance click defect is fixed. Nguni is Latin-script, so its tokenizer claims embedded English
outright and no unclaimed gap ever reaches `core/foreign.ts` — the mechanism every non-Latin-script host
gets for free. Wired a `ForeignPhonemizer` the way ~46 other engines take one, with the English dict
lookup injected alongside (the shape the registry already uses for Naija's `knownWord`).

**The gate needs BOTH signals, and measuring is what showed it.** Routing on "known English word" alone is
badly unsafe here: the most frequent CMUdict hits in these corpora are ordinary Nguni words — `uma` ×105,
`ngo` ×95, `ama` ×67, `kahle`, `yonke`, `kuba`, `moya` — so that gate would have wrecked the language.
Routing on "contains c/q/x" alone fails the other way, those being native click letters. The conjunction
selects **477 distinct tokens across both corpora — china, atlantic, hurricane, francisco, iraq,
microsoft, xinhua, albuquerque — and not one Nguni word.**

A word *without* a click letter is deliberately left native: `visa`, `asia`, `tsunami`, `europe` read as
reasonable nativisations, which is what the Run 33 probe measured readers doing. This routes only where
staying native is not merely accented but wrong.

**Audit.** Only xh_za (297 rows, 19.7%) and zu_za (226, 15.3%) changed; every other language **0**.
**100% of changed rows contain a c/q/x token, and 100% lost at least one click.**
`electric charge` ɛlˈɛːkǀtʼrikǀ kǀʰˈaːrɡ̤ɛ → ɪlˈɛktɹɪk t͡ʃˈɑːɹd͡ʒ, while native `ngelixa` → ŋɡ̤ɛlˈiːkǁa keeps
its click. (Patching only one engine looked like a no-op: xh emits via `phonemizeWord`, zu via
`phonemizeCompound`, so the branch differs.)

**Third QC pass, final corpus:** 77,584 rows, **0 empty**, **0 clicks outside xh/zu**, 17 three-repeats
(the cmn digit-year false alarm plus the ta dot-grouping case, both previously dispositioned),
punctuation 5,726 marks in 4,231 utterances (`, . ! ?` only). Initialism candidates down to 864 from the
original 2,164.

**Total vs the v5 corpus the live model trained on: 8,223 of 40,058 unique rows (20.5%).**

Upstream commits on `norm/foreign-async-oov`: `4f89caa` digraphs, `5e5ce11` Nguni letter names,
`dedc5b0` era casing + data, `966ad42` xh/zu foreign routing. Suite 4751/4751.

## Run 36 — 2026-08-17 — wav2vec2 over the whole corpus: the first gate that listens

Every gate so far compared TEXT to TEXT. Run 31 named the limit — the FLEURS transcript is the script the
reader was given, not a record of what they said — so a phonemizer can be perfectly correct and the PAIR
still teach a wrong alignment. This is the first gate with audio on one side.

**Pass.** `asr_align_corpus.py`: `facebook/wav2vec2-xlsr-53-espeak-cv-ft` over all 28 languages,
**77,584 utterances, 28/28 complete, 0 errors**, ~28 utt/s on the 3090. Audio streamed straight out of the
per-language `train.tar.gz` (104 GB), nothing extracted; rows land in SQLite (WAL) so the analysis can run
against a partial table while the pass continues. Schema is one row per RECORDING, not per sentence_id:
FLEURS repeats a sentence across speakers, and each recording is a separate observation of what a reader
did.

**Scoring** (`asr_align_report.py`) is relative to each language's own median, never absolute — the
recognizer is systematically closer to some languages than others, so an absolute threshold would rank
LANGUAGES by how well wav2vec2 knows them rather than utterances by how wrong they are. Outliers are
3×MAD (not stdev: the tail we are hunting is exactly what would inflate stdev and hide itself).

### Three methodology bugs, each of which silently disabled a whole language

1. **Recognizer failure had to be split off first, or it owns the worklist.** On some utterances the model
   returns almost nothing — a full Welsh sentence came back as the single phone `k`. Those score ~1.0 and
   say nothing about our IPA. Classified `recognizer_short` by a length ratio: **cy_gb 243, sd_in 229**,
   near-zero elsewhere. That is itself a finding, and it cut cy_gb's apparent tail from 240 to 63.
2. **My `fold()` kept MODIFIER LETTERS.** `ˠ ʲ ʰ ʷ ᶦ` are category **Lm** with combining class 0, so a
   `unicodedata.combining()` test keeps them and each counts as a phone. Irish marks velarisation on
   nearly every consonant, so its IPA carried ~2× the recognizer's phone count: **ga_ie's MINIMUM over
   2,845 utterances was 0.371** and its investigate list came out EMPTY — when everything is uniformly bad,
   nothing looks like an outlier. Fixed: ga_ie median 0.674 → 0.481, investigate 0 → 35.
3. **Tone was compared asymmetrically.** We write tone letters (˥˦˧˨˩, stripped); the recognizer writes
   tone DIGITS (`siɛ5`, `ŋo5`), which were kept. Every tonal utterance carried a fixed penalty.
   Fixed: cmn 0.510 → 0.375, th 0.396 → 0.361, vi 0.611 → 0.590.

### Where the method has power, and where it has none

| | median distance | reading |
|---|---|---|
| strong | fr .086 · es .108 · om .177 · de .179 · en .195 | tight distribution, outliers stand out |
| workable | ff .272 · cy .280 · cs .289 · ha .299 · ja .322 · sv/pt .353 · cmn .375 | |
| weak | ko .562 · sd .553 · tr .511 · kk .496 · ga .481 | high median, flat spread — few detectable outliers |
| **none** | **vi .590** | investigate=2 of 2,994; the recognizer is uniformly mediocre on Vietnamese |

**76,737 scored; 74,449 (97.0%) inside their language's bulk; 2,288 flagged.** The bulk is the useful
half of the result: it says most of the corpus does not need looking at.

### First real finding out of the queue: adjacent numbers merge

en_us #28, `batten was ranked 190th on the 2008 400 richest americans list` — we read
*two MILLION eight thousand four hundred*; the reader said *two thousand and eight … four hundred*. The
two numbers are being joined across the space:

    "the 2008 list"        -> tʰˈuː θˈaᶷzənd ˈeᶦt                    correct
    "the 2008 400 richest" -> tʰˈuː mˈɪɫjən ˈeᶦt θˈaᶷzənd fˈɔːɹ hˈʌndɹəd   2008400

⚠ NOT a blanket bug: space-as-thousands-grouping is CORRECT in fr/cs/sv/ru, and those read `2 008 400` as
2,008,400 properly. English does not group with spaces. Corpus exposure is small (6 en_us utterances, one
of them the date `july 21 356 bce` → 21356) but it is a correctness defect in the flagship language, and
invisible to every text-vs-text gate we have run.

Also confirmed from the audio, en_us #1225: `just not one that looks too expensive` — the reader said
*not just*. Category 1, reader divergence: nothing to fix, and exactly the class Run 31 predicted.

**Artifacts.** `work/asr_align/{align.sqlite,summary.tsv,investigate.tsv,recognizer_short.tsv}`,
`scripts/omnivoice_ipa/{asr_align_corpus.py,asr_align_report.py,nativize_probe.py}`.

### Run 36 addendum — the review columns, and working the queue by language

**Schema.** `utt` gained `status`, `comment` and a cached `dist` (user request). The scoring pass produces a
RANKING; these produce a RECORD, and the difference matters because the scorer changed three times during
this run — a verdict written into the row survives that, a position in a sorted file does not. Automatic
statuses (`verified` 74,446 · `investigate` 1,792 · `recognizer_short` 1,338); hand statuses (`defect`,
`reader_divergence`, `convention`, `artefact`) are only ever set by hand, and **a bulk re-apply never
overwrites one** — verified by re-running and checking the hand rows survived.

**Four defects found by listening, each invisible to every text-vs-text gate we have run:**

| | found | fix |
|---|---|---|
| en `2008 400` → 2008400 | reader said "two thousand and eight … four hundred" | leading group 1–3 digits + not after a month; whole-run match |
| en `u.s.` → the word *ʌs* | reader said "U-S" | uppercase on dot-strip — a dotted run is an initialism by construction |
| es/pt `irm` → *ˈiɾm* | reader spelled "i-ere-eme" | `acronymLetters`; `ɾm` is a legal coda so phonotactics cannot see it |
| de `24 september` → cardinal | reader said *vierundzwanzigSTEN* | a bare number before a month is a date |

Two of the four are the CASING WALL again — sixth and seventh sightings. It is now clearly a property of
the codebase rather than a run of coincidences: *a rule keyed on capitals declines silently on lowercased
input, and lowercased input is what corpora ship.*

**Where the leverage is, measured against each language's own baseline** (a raw share is meaningless when
the corpus is full of the thing):

    4-digit year        10.4% flagged vs  6.7% corpus   1.54x
    grouped/long number 26.3%         vs 18.8%          1.40x
    has a digit         28.8%         vs 21.5%          1.34x
    Latin run           only measurable for non-Latin-script hosts:
                        ar 4.0x · sd 3.4x · hi 3.2x enriched; ko 0.0x · am 0.7x · ta 0.4x NOT

⚠ **The enrichment figures are modest, and that is the result.** After the systemic fixes the queue is a
diffuse long tail rather than one dominant cause — no remaining pattern reaches even 2× its baseline
except delegated Latin in three languages. Continuing to grind row-by-row has visibly diminishing returns;
the biggest remaining lever is the parked per-language nativisation question, which the queue itself
now has evidence for.

**A qualitative finding the metric could not capture, because Latin-script hosts have a 100% baseline:**
French, Catalan and Spanish readers CODE-SWITCH on foreign proper nouns — `birmingham` read in English,
`cinque terre` read in Italian, `washington capitals` in English — where we nativise. Amharic readers do
the opposite. That is direct evidence that nativisation must be a per-language strategy, not a universal
layer, which is where Run 35's probe left the question.

### Run 36 addendum 2 — 585 Welsh audio files are TRUNCATED, and it is upstream FLEURS data

Working the queue by language turned up something that is not a phonemizer defect at all, and is worth
more than everything else found in this pass.

**How it surfaced.** Welsh flagged rows kept showing recognized phones almost unrelated to their text.
The recognizer-reliability check (median heard-phones / our-phones per language) put cy_gb at a healthy
1.04 — but **17.4% of its rows fell under 0.7**, a bimodal shape no median would show. Spanish had a
similar 17.5%, so the first hypothesis (poor Welsh coverage in xlsr-53) was wrong.

**What it actually is.** Seconds-of-audio per phone, against each language's OWN median so a fast-speaking
language is not penalised:

    cy_gb   585 of 3,427 utterances (17.1%)   0.0150 s/phone   vs 0.1509 normal   -- a TENTH
    sd_in    12   ff_sn 9   ar_eg 2   am_et 2   ta_in 1        -- negligible everywhere else

Median 1.5 s of audio for a sentence needing ~96 phones. Spanish, checked, is NOT affected — its
under-0.7 rows have entirely normal duration, so its low ratio has some other cause.

**Read, not just measured** (user's ask): of the 585, **333 decode to nothing at all** and 252 give short
fragments. Where the fragments are intelligible they look like ENGLISH — `ð ə s eɪ m ɪ k s p ɪ ɹ ɪ ə n s ə`
("the same experience"), `h aʊ s`, `ɡ eɪ v ð ə … m eɪ ʃ ə n` — in files whose transcript is Welsh. That is
suggestive but weak on its own: a one-second noisy fragment biases this recognizer toward English whatever
is in it. **The duration is the unambiguous part.**

**Not our download.** Checked the member sizes in the source tar directly: the truncated files really are
small there (median 99,898 bytes vs 954,298 for normal ones; the 2:1 against the decoded duration is just
stereo, which the loader averages). Re-fetching will not help — this is what FLEURS ships.

**Consequence, and it is the important bit.** These are 585 catastrophic TRAINING PAIRS: a full sentence of
IPA against 1.5 s of audio teaches the model to compress a sentence into a tenth of its time. They are
17% of the Welsh corpus. Labelled **`defective_audio`** in the DB (611 rows corpus-wide) — a data defect,
not a QC verdict, and not ours to fix: the action is to exclude the pair and report upstream. Detection
lives in `asr_align_label.py` so it is reproducible rather than ad-hoc.

**⚠ AND IT COSTS COVERAGE, because Welsh is not an interchangeable 17%.** cy_gb was selected into the
corpus as the OWNER of one census primitive, U+0325 (voiceless ring) — and it is the **sole source** of it
across all 28 languages, 1,937 occurrences, every one Welsh. Excluding the defective rows takes 327 of
them (16.9%), leaving **1,610 in 1,148 utterances**, so the primitive survives comfortably. The rest of
the loss is uniform — every Welsh phone loses ~18%, matching the 17.6% of phone tokens dropped, and **no
phone vanishes entirely**. So the exclusion is safe, but only because it was checked: a defect
concentrated on the primitive Welsh was chosen for would have been a different answer.

**Pipeline consequence.** The exclusion has to be applied BEFORE `sampling_budget.py`, not after. That
script sets each language's oversampling weight so its scarcest owned primitive reaches a minimum
exposure per epoch, and computing that over pairs that will then be discarded targets the wrong number.
Order is: exclude `defective_audio` -> patch manifests -> sampling weights -> webdataset.

⚠ **And it argues the audio gate should run BEFORE any future training, not after.** Every text-side gate
we have — espeak diff, qualitative reads, the local-model sweep — is blind to this by construction: the
transcript is fine, the IPA is fine, and only the PAIR is broken.

## Run 37 — 2026-08-17 — The local-model exhaustive judge does NOT work. Abandoned.

**Question.** Run 36's distance heuristic flags 1,782 rows (2.3%) as `investigate` and calls the other
97% `verified`. Reading every row with a local model instead of trusting a heuristic looked like it
would have value. Would it?

**Answer: no.** Recording this in full because the failure is specific and worth not repeating.

**Single-stage judge** (`judge_alignment.py`, Qwen3-27B IQ4_XS, thinking off, constrained JSON,
two-question prompt: Q1 is the IPA correct for the transcript, ignoring the recognizer; Q2 does the
recognizer's output match the IPA). Verdicts were dominated by `reader_diverged` fabrications wherever
the recognizer's phone string diverged for ordinary acoustic reasons. An arithmetic guard
(`nh/ni > 0.85` downgrades the verdict) suppressed the worst of it, but the judgement underneath was
never sound — the guard was doing the work, and the guard is just another heuristic.

**Two-stage cascade** (`judge_cascade.py`, the user's suggestion): a cheap non-thinking screen splits
agree / don't-agree, and only don't-agree goes to a thinking-mode adjudicator. This is the right shape.
It failed at stage 1.

Screen check-rate by language, after three instruction rewrites, one structural fix (make the model
QUOTE the offending token as evidence rather than assert a verdict), and one real bug fix (a bare `?`
placeholder was being counted as evidence in 62% of checks):

    es_419  100.0%      en_us  93.3%      fr_fr  88.0%
    de_de    73.3%      th_th  68.0%      am_et  46.3%

Whole-sample rates across tuning attempts: 98.0% -> 93.7% -> 37.7% check. Even the best is useless:
37.7% of 76,236 rows is ~28,700 adjudications. And note the ORDER — the screen flags MOST where the
model reads BEST. It is not detecting defects, it is detecting how much of the row it can parse.
A screen whose sensitivity is a function of the model's own literacy cannot be calibrated.

**Then: is the adjudicator itself sound?** Tested the four `am_et` screen false positives (all four
hand-verified CORRECT: `1960s`, `2015`, `¥7,000`, `university`) with thinking ON. All four returned
`no_reply`. Retried one with `max_tokens: 8000` — `finish_reason=length`, 7,205 completion tokens,
**empty `content`**, 13,367 chars of truncated `reasoning_content`. It thought for 7,205 tokens about
a two-field comparison and never reached a conclusion.

**Why it truncated, and why that closes the question.** The server ran `-np 8 -c 65536` = 8,192 tokens
per slot, which is exactly where it stopped. Fewer slots buys per-slot context but costs concurrency,
and the two fight each other directly. Feasibility for the 1,782 distance-flagged rows alone at the
measured 118 s/row:

    -np 2  ->  29.2 h        -np 4  ->  14.6 h        -np 8  ->  7.3 h (but 8,192 is not enough)

That is for 2.3% of the corpus. The exhaustive version was never on the table.

**Decision (user's): go back to the heuristic and manual fixes.** The distance heuristic plus reading
the flagged rows by language is the gate. Every systemic defect found so far — the initialism casing
wall, the de ordinals, the xh/zu letter names, the es/pt `irm`, the 585 truncated Welsh files — came
out of that loop, not out of a model verdict. The local model was useful exactly once, for bulk
triage of a 1,464-row candidate list where the answer was a category label and a human reviewed the
output in bulk afterwards (Run 34). That is its shape: pre-sort a list for a human, not adjudicate.

**Kept:** `judge_alignment.py`, `judge_cascade.py` stay in the tree as a recorded dead end.

## Run 38 — 2026-08-17 — Per-language sweep of the `investigate` queue (continued)

Method after Run 37: read the distance-flagged rows per language, by hand, looking for SYSTEMIC
defects rather than adjudicating rows one at a time. The heuristic picks the queue; I read it.

### ff_sn (Fulfulde) — 117 flagged — **two digraphs missing from the rule table**

⚠ **First, the trap in reading this language's queue.** wav2vec2-xlsr-53 has no Fula, so most of the
high-distance rows are the RECOGNIZER failing, not the IPA. Row 28129 is the control: where the
recognizer does decode (`ɡ o t o h a n d e r e iː m e...` vs `ɡˈoto hˈa ⁿdˈeɾ hˈimɓe...`) it matches
cleanly. So the distance ranking is nearly useless here — the finding came from reading the IPA
COLUMN ALONE and noticing sequences that cannot be IPA at all.

    cctv neldinan himɓe ...     -> t͡ʃːtv ...          the initialism casing wall again (known class)
    ... shawwal inji ...        -> shˈawːal ...        LITERAL "sh" sitting in an IPA stream
    kuje waya be komputa        -> ... t͡ʃhaⁿd͡ʒˈata   an impossible cluster t͡ʃ + h

Counted across the whole ff_sn corpus, not just the queue: **280 of 280** rows containing ⟨sh⟩ keep it
as the literal two letters, and **514 of 530** rows containing ⟨ch⟩ emit `t͡ʃh`. Not a sampling
artifact — a total failure, on every occurrence.

**Cause.** Fula writes /t͡ʃ/ as a bare ⟨c⟩, so the rule table never needed a ⟨ch⟩ entry for native
words. But FLEURS ff is Nigerian/Adamawa Fulfulde, which is Hausa-influenced and full of both
spellings. The longest-match scan read `c`→t͡ʃ then `h`→h; ⟨sh⟩ had no rule at any level and fell
through to `s` + `h`. The letter-level fallback was NOT silently deleting anything — `h` is a real
Fula phoneme, so it did exactly what it should. The defect was only ever the two missing digraphs.

**And the manifest already contradicted itself.** `fula.jsonc` declares the Adlam loan letter
U+1E943 → `"sh"` as one of the letters "mapped to the same Boko equivalents the Latin engine already
uses." It did not use it. The second script transliterated straight into the same dead end.

**Third rule, found only by checking the ordering.** ⟨cch⟩ is the geminate of ⟨ch⟩ and fails one level
up: `cc`→t͡ʃː matches first and the `h` is left over. 25 rows (acchugo, yeccheta, picchu, bocchi).
Adding ⟨ch⟩ without ⟨cch⟩ would have fixed 514 rows and left 25 broken in a way the same read had
already walked past.

Evidence, no counterexample either direction: shiri / karshe / kashi / shafi / tasha / shawara + the
English loans spanish / british / washington are all /ʃ/; chede / chaka / chanji / china / march /
charles are all /t͡ʃ/. The one true exception in the corpus is `hesperonychus`, a Latin taxonomic
name where ⟨ch⟩ = /k/ — one word, not worth a rule.

Fixed upstream: `6dea7af` on `norm/qc-backlog`, with regression tests. Full CI green (4,772 tests).

**This is the same SHAPE as the Uzbek case documented at the top of `core/latinPhones.ts`** — a letter
that is unreadable alone but essential in a sequence — running in the opposite direction. Fula reads
the bare letter and misses the digraph. Worth asking of every rule-scan engine, not just this one.

### ha_ng (Hausa) — 109 flagged — **the same digraph, in the sister language**

The ff finding handed over a direct hypothesis: those ⟨sh⟩/⟨ch⟩ spellings in Fulfulde came from
Hausa, so check Hausa. Hausa HAS ⟨sh⟩→ʃ. It does not have ⟨ch⟩, and reads a bare ⟨c⟩ as /t͡ʃ/ —
the exact ff shape. **182 of 182** rows containing ⟨ch⟩ emit `t͡ʃh`.

What differs is the vocabulary: Fulfulde's ⟨ch⟩ words are native (chede, chaka, chanji), Hausa's are
almost entirely foreign proper nouns (china ×32, charles, chile, richard, ovechkin, gingrich). So I
checked whether they should be routing to a foreign reader instead — they do not, and cannot:
`NATIVE_CLASS` in `hausa.ts` accepts all of `a-z`, so a plain-ASCII name carries no letter Hausa
lacks and nothing marks it foreign. That is the existing design (`makeNativiser`, not
`readForeignRun`), so the only question left is what the host makes of the spelling — and /t͡ʃ/ is
what a Hausa reader does with it, which is the nativising Run 33's audio established.

Fixed: `baf5c0b`. ⚠ The ~15 true /k/ readings (orchestra, hesperonychus, maroochydore) are now wrong
in a *different* way than before. They were already wrong, and no rule separates them — nothing in
host text says which orthography a name came from. `core/latinPhones.ts` already documents this.

### Generalising it: `audit_digraph_coverage.py`

Two of these in two languages is a class, so I built the gate instead of hunting language by
language. It enumerates the consonant sequences the corpus actually contains, subtracts the rule
keys, and ranks the remainder — flagging `SCANS-CLEAN` for sequences whose every letter has its own
rule, which is the dangerous kind: it decomposes silently and errors nowhere.

Only **5 of the 182 engines** are auditable this way — `fula hausa hungarian xhosa zulu` are the only
ones with a jsonc `"rules"` table; the other ~177 carry their mapping in code. Four are FLEURS
languages, and two of those were the two bugs.

**Two negative results, which are the point of running it:**

**xh_za / zu_za are clean.** Their tables carry `ntsh thsh tshh ngc ngq ngx nkc nkq nkx ths tsh tyh
bh ch dl dy gc gq gr gx hh hl kh kl mb nc ng nj nk nq nx ny ph qh rh sh th ts ty xh` — every
surviving trigram (`mth`, `thw`, `gcw`, `qhw`, `ncw`, `dlw`) decomposes into a listed digraph plus
`w`/`y`, which is correct. `tyh` never appeared in the uncovered list *because it is already a rule*.
The gate finding nothing in half the languages it can read is what makes the other half credible.

**⚠ And it produced one false lead that the AUDIO killed — worth recording, because I nearly shipped
it.** The audit flagged Fulfulde ⟨dy⟩ ×88 in what looked like native words. `dyam` ×36 sits beside
`jam` ×25, `dyona`/`dyonata` beside `jonta` ×40 — a textbook case for ⟨dy⟩ = /d͡ʒ/, which is a real
Pulaar orthographic convention, and I had the rule half-written. Then the recognizer:

    jirgi je les dyam ...        -> l ɛ s m iː a m      "les dyam" = UNDER WATER (a submarine)
    ko be wala masibo dyam ...   -> m a s i v o ɲ a m i    a disaster, water entered the town
    sikhs do dyona deena ...     -> d o d u n j aː n a
    a dyona ko hatoi ...         -> o d i o n a k a

`dyam` is not `jam` "peace" — it is `diyam` "water" (×65, and `ndiyam` ×100) with the vowel letter
dropped, and `dyona` is `diyona`. ⟨dy⟩ here is a SPELLING VARIANT in the FLEURS text, not a digraph.
One sentence contains both `ndyiam` and `diyam`. The rule would have turned /dijam/ into /d͡ʒam/ on
61 native tokens — a worse defect than the one it was fixing, on evidence that read as conclusive.

**The rule this gives us: the audit proposes, the audio disposes.** Frequency and minimal-pair
co-occurrence in the text were not enough; only listening separated the digraph from the typo.

### pt_br — 118 flagged — clean

Correct Brazilian palatalization (`d͡ʒi`, `t͡ʃi`), nasalization (`ẽj̃`, `ɐ̃w̃`), and the recognizer
broadly agrees. The distance is recognizer drift toward EUROPEAN Portuguese — it hears `ɡ ɔ ʃ t` for
`ɡˈɔstɐ̃w̃`, coda ʃ where BP has s. One real signal, and it is the known Run 33 class, not a defect:
in `a modern education o acusou de...` the reader says the English title IN ENGLISH
(`ɛ d ʊ k eɪ ʃ ʊ ŋ`) while we nativise it to `edukat͡ʃˈiõ` (as if `-ção`).

### ⚠ THE BIGGEST FINDING OF THE SWEEP: six engines leak RAW ORTHOGRAPHIC LETTERS

Rather than eyeball 20 rows for each of the 18 remaining languages, I ran a cheap general scan first:
count every character in each language's IPA and look at the rarest ones, on the theory that a leaked
orthographic letter is rare and odd. No ASCII `g` anywhere (IPA needs U+0261 ɡ) and no stray digits —
clean. But:

    ⟨q⟩ stands in the IPA of FIVE languages that have no /q/:  cy ×38  ga ×33  sv ×13  es ×9  ca ×5

All of them the same handful of items — `Qing`, `Qatar`, `piquet`, `Albuquerque`, `Joaquim`,
Greenlandic `Kalaallit`. `qˈiŋ`, `piqˈɨɛt`, `ˈal̪ˠbˠəqəəɾʲqəə`, `qatˈɑːr`.

**Cause, and it is one line repeated six times.** `core/latinPhones.ts` is the floor under a letter no
g2p can read — it maps `q`→k, `x`→ks, `y`→j — and 46 engines call it. Six do not:

    welsh/g2p.ts:145   else if (/[a-z]/.test(c)) segs.push({ ph: c, ... }); // unknown letter: pass through
    irish/g2p.ts:105   spanish/g2p.ts:210   catalan/g2p.ts:139   swedish/g2p.ts:207   galician/g2p.ts:223

They push the RAW ORTHOGRAPHIC CHARACTER into the phone stream. The letter never fell through to the
floor — it walked straight past it into the output.

**⚠ Why it survived every previous review.** /q/ is a perfectly ordinary phone. No inventory check, no
distance metric and no plausibility read flags it, because the string is only wrong FOR THIS LANGUAGE.
Swedish makes the point sharpest: ⟨qu⟩ already had its own rule, so `square` → `skvˈɑ̀ːrɛ` was correct
all along and ONLY the bare letter leaked — the engine looked like it handled q.

**Irish leaked three — ⟨q⟩, ⟨x⟩ and ⟨y⟩ — and ⟨y⟩ is the one that should worry us.** `y` is a valid
IPA symbol for a close front rounded vowel. An orthographic y did not look out of place in the
output at all; it silently became a VOWEL. A phone-set check would have passed it.

Fixed: `22a2f23`, all six wired to `latinPhone`, each keeping its raw-character path as the last
resort for when the shared reading itself declines (a typed letter is content; the floor exists to
give it a sound, not to delete it). Regression tests in `test/latin-phones.test.ts`, including a
guard pinning `sv square` so the fix cannot reach past the bare letter. Full CI green, 4,777 tests.

**Method note worth keeping.** This was not found by reading the investigate queue — the flagged rows
for cy/ga/sv/es/ca do not concentrate on these words at all. It was found by asking a different
question: *what characters appear in this language's output that this language's engine should not be
able to produce?* That question is cheap, general, and it found a six-engine defect in one pass.

### sv_se — ⟨cc⟩ is a loan cluster, not a geminate — `68a7fc9`

Same pass as the ⟨q⟩ leak, one branch over. The geminate branch fires on any doubled consonant, but
⟨c⟩ is contextual in Swedish and so is absent from `CONS` — which dropped ⟨cc⟩ into that branch's
`else`, and the `else` pushed the raw character: `acceptera` → `acɛptˈeːra`, `piccolo` → `pˈɪ̀cɔlɔ`.
⟨cc⟩ is /ks/ before a front vowel (vaccin, vaccineras, acceptera, acceptabelt) and /k/ elsewhere
(piccolo, cappuccino), and unlike single ⟨c⟩ it is NOT gated on the stressed onset.

### ru_ru / cmn_hans_cn / th_th / pt_br — read and clean

- **ru** (94 flagged): correct reduction (ɐ/ə), palatalization, ɕː. Distance is recognizer noise.
- **cmn** (89): correct Chao tone letters and finals. Checked `yu`→`jy` ×1608 as a possible spurious
  glide — it is a deliberate convention, consistent with `yi`→`ji` and `you`→`jioᵘ`, in a table
  validated against wikipron + epitran. Not a defect.
- **th** (65): tones and length correct; `991` correctly expanded to `kˈaː˥˩w rˈɔː˦˥j kˈaː˥˩w sˈi˨˩p`.

### kk_kz — ⚠ ⟨ь⟩ AND ⟨ъ⟩ WERE BOTH MAPPED TO A GLOTTAL STOP — `fe852f5`

`kazakh.jsonc` had `"ъ": "ʔ"` and `"ь": "ʔ"`. **408 rows** carried a glottal stop that is not in the
word: миль `mˈəjɫʔ`, гольф `ɡˈoɫʔf`, пальма `pɑɫʔmˈɑ`, Нью `nʔjˈu`, премьер `premʔˈer`, Чарльз
`t͡ʃˈɑrɫʔz`. Found by reading the queue — `коньки` came out `kˈonʔkəjʃɪnɪŋ` while the recognizer heard
`k a n ɡ iː t ʃ n ə ŋ`, no glottal anywhere.

Neither sign denotes a sound. Kazakh is Turkic and meets them only in Russian loans, where they do
**opposite** jobs — ь palatalises what precedes, ъ separates with a /j/ — so one entry for both could
never have been right. A palatalised l is light, so ⟨ль⟩ also escapes the dark ɫ that ⟨л⟩ emits for
vowel harmony, while native алма/бала keep theirs.

⚠ **Two existing tests had the defect baked into them** and failed on the fix: съезд expected
`sʔˈezd`, Цельсий expected `t͡sˈelʔsəjj`. Both expectations were simply wrong (/sjest/, /tsɛlʲsij/).
Corrected with the g2p rather than around it — and съезд is a second independent attestation of the
separating ⟨ъ⟩, beyond объектив. **A test can encode a defect and then defend it.**

Backlogged for kk, too small to justify an engine change: `д-р` (2 rows) reads as letter names where
the reader said "doktor", and `ақш` (АҚШ = USA) reads as a word — both the known lowercased-
abbreviation/initialism class.

### hi_in — ज्ञ is not its parts — `3e0900d`

Composed literally the ligature is ज (d͡ʒ) + halant + ञ (ɲ) → `d͡ʒɲ`. Modern Standard Hindi says
/ɡj/: ज्ञान gyaan, विज्ञान vigyaan, वैज्ञानिक vaigyaanik, विशेषज्ञ visheshagya. **73 of 73 rows**
wrong. The recognizer settled it rather than my reading — we wrote `d͡ʒɲˈaːt̪`, the audio came back
`ɡ i a t`. ⚠ Scoped to Hindi deliberately: **Marathi reads the same ligature as `dnya`**, so this
must not move to the shared Devanagari layer; a test pins mr as not containing ɡj.

### ja_jp — the hiragana counter つ — `f55d2df`

`1つには` → `it͡ɕi t͡sɯᵝniwä`, "ichi-tsu". つ is the native (wago) general counter and is wholly
suppletive — 1つ is ひとつ, never いちつ. **89 rows** (1つ ×47, 2つ ×24, 3つ ×12, 5つ ×6).

The cause is a nice one: つ is the ONLY counter written in HIRAGANA, and the number+counter fusion
regex matched `\p{Script=Han}` only — so a digit + つ never reached `readCounter` at all. The counter
table itself was fine; 人 already had its ひとり/ふたり irregulars. Only つ was added beside Han, never
kana generally: a digit is followed by an ordinary particle constantly (3の, 5は), both pinned by test.

### A gate that found nothing (worth recording)

Scanned every language's IPA for SOURCE-SCRIPT characters — Cyrillic, Devanagari, Thai, Han leaking
into the phone stream. **Zero hits across all 28.** Every character that tripped the first draft was
legitimate IPA: θ β χ are the dental/bilabial/uvular fricatives, ⁿ ⁱ are prenasalisation and offglide,
ꜜ is Japanese downstep. Note this gate would NOT have caught the Kazakh ʔ — a glottal stop is
perfectly good IPA, just absent from the word — which is the same reason ⟨q⟩ needed a per-language
question rather than a universal one.

### ar_eg / tr_tr / ko_kr / ta_in / vi_vn / om_et / sd_in — read and clean

Egyptian Arabic reads ق→ʔ and ج→ɡ correctly (the distance is the RECOGNIZER using MSA, not us).
Turkish k→c/ɟ palatalization is right and `2011'de` → `icˈi bˈin ˈon biɾdˈe`. Korean tense/aspirated/
unreleased codas are right and `64세인` → `ˈjuk̚s͈ip̚s͈ɐsein`. Sindhi numbers are right
(`1994` → `hˈɪkʊ həzˈaːɾʊ nˈəwə sˈəʊ t͡ʃoːɾaːnˈoːj`). Oromo ejectives and ᶑ are right.

### xh_za / zu_za — a routing SPLIT inside one phrase (documented, not fixed)

    i-international olympic committee -> ˈiː intʼɛrnatʼiˈɔːnal | oᶷlˈɪmpɪk kəmˈɪt̬i
                                              read as ZULU     |  read as ENGLISH

Not the concord hyphen — tested in isolation, `international` reads Zulu while `olympic` and
`committee` each route to English. The classifier decides per word and splits one English proper name
down the middle. The reader said all of it in English (`i n t ɛ ɾ n a ʃ ə n a l`).

Left alone deliberately. Which half is right is the per-language nativisation question the user
parked after Run 33, and the volume is small — the character scan shows only tens of English-only
phones in xh/zu, so the overwhelming majority of foreign words ARE nativised. What is defensible to
say now is narrower and firmer: **whatever the policy, it should not change mid-phrase.**

⚠ A metric I threw away rather than report: counting rows containing "English-only" phones
(ɝ ʌ ð θ ᵻ oᶷ eᶦ ɹ) gave cy_gb 98.4% and es_419 88.1% — because θ is native Welsh ⟨th⟩. The instrument
was measuring the wrong thing; the per-language character scan is the one that works.

### ⚠ THE CASING WALL, MEASURED — `scan_casing_differential.mts`

Found via sd_in, of all places: `ذاتي vpn ورچوئل` dropped `vpn` from the IPA entirely. Chasing it
landed in the ENGLISH layer, and the failure is not Sindhi's:

    vpn  -> "vpn"     the raw letters, standing in an IPA stream
    vhs  -> "vs"      the h DELETED
    hq   -> "k"       two letters, one phone
    nhs  -> "ns"      h deleted again
    hdmi -> "dmˈɪ"        wto -> "ˈuːt"        vga -> "vŋɡˈʌ"

Every one of them is correct when uppercased (`VPN` → `vˈiː pʰˈiː ˈɛn`). The capital-keyed initialism
rule declines on lowercased FLEURS input, and the ordinary word g2p accepts the token without
complaint — because a letter run IS readable as a word. Fluent, wrong, and sometimes lossy.

**The gate this suggests is much sharper than the one I built in Run 34.** That one asked each
language's phonotactics whether a token was UNREADABLE and produced 1,464 candidates needing bulk
triage. This asks a question that needs no judgement at all:

    does phonemize(token) differ from phonemize(TOKEN)?

Casing is not phonemic, so for an ordinary word the two agree. A disagreement IS the wall, directly
observed — and the uppercase reading is simultaneously the answer, so every hit arrives with its fix
attached. **73 English candidates instead of 1,464.**

**And the yield vindicates the earlier hand review:** of those 73, the genuine initialisms were almost
all already on the allowlist (vpn, xdr, qc, png, afcfta, wned, bce, utc, gmt) or already in EXCLUDED
with a reason (km, cm, kg, kph, sq, zmapp, angkor, jagr, dzong). Only **four** were new — `gps`,
`hiv`, `usaf`, `un` ("waste from the UN camp") — now added.

⚠ The gate misleads in three recorded ways, all now in EXCLUDED: `wwii` (uppercasing gives letter
names, but WWII is "World War Two"), `led` (×13 as the ordinary verb — the homograph loses to
frequency), and `ll` (the tail of we'll / I'll split on the apostrophe — **the tokenizer artifact I
mistook for a corpus defect once already**, showing up in a new instrument). A strong signal is still
not a verdict.

## Run 39 — 2026-08-17 — The corpus-side backlog: exclusion wired, a leak found, corpus rebuilt

Four items were outstanding after the phonemizer merge (`a4717ae`). Doing them turned up a fifth that
is more serious than any of the four.

### 1. `defective_audio` exclusion, wired at the RIGHT step — `corpus_filter.py` / `exclude_defective.py`

The exclusion had to land before `sampling_budget.py`, not just before shard-building: that script
sets each language's oversampling weight from the count of its scarcest OWNED primitive, and counting
that over pairs which are then discarded targets the wrong number *silently*. Both scripts now load
manifests through one shared `load_manifest()`, so neither can forget, and both PRINT the drop count
so a missing `work/exclusions.tsv` is visible rather than silent.

Only `defective_audio` is excluded. `investigate` (1,782) is a QC **queue**, not a verdict;
`recognizer_short` (737) is a fact about the recognizer, not the audio. A status column is a work log,
and only one of its values is a statement that the data is unusable.

**498 utterances dropped** (of 611 flagged — the rest were already absent from the manifests):
cy_gb 480, ff_sn 9, sd_in 6, am_et 2, ar_eg 1. The coverage re-check runs every time rather than
trusting Run 36's note: **no phone vanishes in any language**, and U+0325 — the primitive Welsh is the
SOLE source of — retains 1,557 of 1,846 occurrences across 1,110 utterances.

### 2. ⚠ TRAIN/DEV LEAKAGE: 73–99% OF EVERY DEV SET WAS ALSO IN TRAIN

Found while wiring the exclusion, by reading the code I was about to modify.

`build_webdataset.py` deduped the manifest on `id` before splitting, with a comment stating `id` was
a per-SENTENCE key shared across speakers. **It is not.** `id` is the wav stem — unique per recording
— so the dedup was a no-op and the split was a plain row slice. FLEURS records each sentence with
~2.2 speakers (cy_gb: 3,263 recordings over 1,502 sentences), so slicing rows put the *same sentence*
in both splits, read by a different voice:

    xh_za 99%   ff_sn 94%   cy_gb 95%   en_us 87%   ja_jp 73%   of each dev set's sentences

**Dev loss was scoring recall of sentences already trained on, not generalization** — which means the
v5 run's dev curve does not mean what it appears to mean. The comment shows the author knew the
hazard exactly and keyed the guard to the wrong field; the guard then read as protection for the
whole life of the pipeline.

Fixed by grouping on `sentence_id` and assigning whole groups, with an assert. Dev is still sized in
rows (~80) but now spans ~36 distinct sentences instead of ~80 leaky ones. **Verified on the built
shards, not just the inputs: 0 shared sentence_ids across all 28 languages.**

### 3. ⚠ AND I MADE THE MIRROR-IMAGE MISTAKE MYSELF, ONE COMMAND LATER

`patch_manifest_ipa.py` defaults to `--byid work/phonemized/byid` — the **espeak** tree. I ran it
without the flag and patched all 28 manifests with espeak IPA instead of vernacula.

It was caught only because I checked the *manifests* for the defect signatures rather than trusting
that the byid files being clean meant the manifests were: ff/ha/kk still showed `t͡ʃh`/`sh`/`ʔ` while
hi/cy/ga/sv/es/ca read clean, and that split made no sense under any correct run. Re-run with the
explicit `--byid work/phonemized_vernacula/byid`. **The lesson is the same one as #2 in both
directions: verify the artifact, not the input.**

### 4. Corpus rebuilt on the merged phonemizer

Re-phonemized all 77,584 utterances (0 errors), then re-ran the pipeline in order:
**exclude → patch manifests → sampling weights → webdataset**.

Every defect signature from the sweep is now **zero in the built shards**:

    ff sh 280→0   ff t͡ʃh 514→0   ha t͡ʃh 182→0   kk ʔ 408→0   hi d͡ʒɲ 73→0
    cy q 38→0   ga q 33→0   sv q 13→0   es q 9→0   ca q 5→0   sv `acɛpt`→0

…and the positive side is present: 47 ja `çito̞t͡sɯᵝ`, 109 hi `ɡj`, 252 kk palatalised `lʲ`.

**Final corpus: 74,278 train + 2,133 dev = 76,411 utterances, 28 languages, 0 defective pairs,
0 sentence leakage.**

⚠ **One more incomplete fix, found by re-scanning the REGENERATED output** for the very cluster the
⟨ch⟩ fix targeted instead of assuming it was gone: 5 ff rows and 4 ha rows still had `t͡ʃh`, all of
them ⟨chh⟩ — the Indic transliteration digraph in *Chhatrapati* and *Chhappan*. ⟨ch⟩ matched the
first two letters and left the second h stranded, rebuilding the exact cluster. Fixed upstream in
PR #823 (`28f4d26`). **A fix is not done until the thing it targeted is gone from the output.**

### 5. Upstream report drafted — `docs/fleurs_cy_gb_truncated_audio.md`

585 Welsh train files (17.1%) with median 1.44 s of audio against a median-14.16 s transcript, with
the tar-member evidence that it is not a download artifact, the recognizer findings, and the file
list in `fleurs_cy_gb_truncated_audio.txt`. Ready to file against `google/fleurs`.

## Run 40 — 2026-08-17 — Re-score on the rebuilt corpus: did the fixes actually help?

The corpus was rebuilt on the merged phonemizer, so the alignment DB's `ipa` column went stale while
`phones` — the recognizer output — did not. That makes an OBJECTIVE test available for free: stash
the old IPA in `ipa_prev`, refresh from the new manifests (2,111 rows changed), and re-score. Any
movement is attributable to the fixes, because the audio side never moved.

### The fixes are confirmed, not just plausible

    rows that moved CLOSER to the audio: 1554        further: 373        (4.2 : 1)
    median distance improved in 20 of 27 languages

Per-language, the targeted fixes are decisive — this is the first evidence for them that does not
depend on my reading being right:

    kk_kz  399 better /   7 worse    the ь/ъ glottal-stop fix
    ha_ng  174 /   7                 the ⟨ch⟩ digraph
    hi_in   74 /   1                 ज्ञ → /ɡj/
    ja_jp   73 /   3                 the つ counter
    en_us   35 /   1                 dotted initialisms
    ff_sn  564 / 168                 ⟨sh⟩/⟨ch⟩/⟨cch⟩

### ⚠ THE THREE LANGUAGES THAT GOT WORSE, AND WHAT THEY EACH TURNED OUT TO BE

**xh_za (31 better / 63 worse) — a REAL defect, not mine.** Foreign `c`-initial words are being read
with Nguni CLICKS: `ceo` was `sˈiːʲiːʲˈoᶷ` ("see-ee-oh") and is now `kǀˈɛːɔ`; `china` was `t͡ʃˈaᶦnə`,
now `kǀʰˈiːna`. The recognizer is unambiguous — it heard `s i i o` for *ceo* and `tʃ aɪ n aɪ` for
*china*. No click.

Measured the classifier rather than just complaining about it: **native 12/12 correct** (`cela`,
`caba`, `cha`, `cishe`, `cwaka`, `qaphela`, `ukucela` all keep their clicks), **foreign 16/19
correct** (`city`, `court`, `class`, `crown`, `computer`, `cnn` all correctly refused). The three
failures are exactly the foreign words that are PHONOTACTICALLY LEGAL NGUNI — `canada` is CV.CV.CV
and indistinguishable from a native word by shape; `china`'s ⟨ch⟩ is a legal Nguni digraph. That is
the intrinsic ceiling of a phonotactic test, not a bug in it, and no rule keyed on shape can fix it.
~30 rows in 76k. Documented, not hacked around.

**ff_sn (168 worse) — my `un` addition, and I was wrong about why.** `ha un be` became
`hˈa ˈu nˈa bˈe`. My first read was that I had broken a Fulfulde word by adding `un` to a globally-
applied allowlist on ENGLISH evidence. The audio says otherwise: the recognizer heard `h a j u e n v`
— **the reader spelled it "yu-en"**. Same in Xhosa (`b e j u e n`), and the contexts confirm it
everywhere it fires: xh `be-un`, zu `le-un`, ko `un 회원국`, sd/th, all United Nations. The addition
is correct. The residual distance is a LETTER-NAME CONVENTION mismatch — Fula spells N as `nˈa` and
Xhosa as `ˈɛːni` where the reader used the international `en` — which is a different question
entirely, and a much smaller one.

**zu_za (28/28, median flat)** — the same Nguni click issue as xh.

### The structural hazard the `un` scare exposed — `check_allowlist_collisions.py`

⚠ **`restoreInitialismCasing()` takes no language argument.** Every token in the allowlist is
uppercased in ALL 28 corpora, but each was added on evidence from ONE. `un` is the indefinite article
in French (1,050), Spanish (847), Catalan (778), "one" in Welsh (297), "flour" in Turkish (44) —
**3,064 standalone occurrences against the 6 it was added for.**

So I gated it, and the result is genuinely reassuring in a way I did not expect:

    dda   changes output in 27/28 languages — but NOT in cy, where it is the Welsh word "good"
    un    changes output in 16/28 — but NOT in fr, es, ca, cy, tr, where it is a word

**In both cases the one language that would be damaged is exactly the language that is inert.** Not
luck: a host engine that has its own lexical or rule claim on the string reads it as that word, and
the capital-keyed initialism pass never gets it. The global allowlist is safer by construction than
its design suggests — but only where the host actually knows the word, which is why the gate stays.

Of 76 allowlist tokens, 18 appear in more than one corpus and **zero are real collisions**: the rest
are the same initialism appearing across parallel FLEURS translations (`gmt`, `gps`, `pbs`, `rspca`,
`utc`, `afcfta`, `hk`, `png`), which is correct behaviour.

## Run 41 — 2026-08-17 — Working the bug backlog: a SECOND FLEURS audio defect

### ⚠ es_419 SHIPS 490 SILENT FILES (17.5% of the split) — and it is not the Welsh defect

`recognizer_short` was the one bucket the QC never explained. 490 of its 737 rows are Spanish — 17.5%
of es_419 against ≤0.2% everywhere else — and a concentration in one language is exactly the shape
that found the Welsh problem. So I read it instead of leaving it.

The rows have **empty recognizer output and NORMAL audio duration** — 11.61 s median, slightly
*longer* than the verified rows' 10.74 s. That rules out truncation, which is what Welsh was. So I
measured the audio itself:

    EMPTY-ASR rows    rms 0.000008   peak 0.0000    full length
    verified rows     rms 0.02-0.10  peak 0.24-0.87

**They are digitally silent.** Full-length files containing nothing.

Scanned the entire language rather than a sample, and the correspondence is exact:

    2,796 es_419 files scanned
      490 silent (rms < 1e-4)        490 empty-ASR        overlap 490
        0 silent but not flagged      0 flagged but audible

⚠ **This is a genuinely different defect from Welsh, with the same consequence.** Welsh audio is
TRUNCATED (1.4 s of a 14 s sentence); Spanish audio is FULL LENGTH AND EMPTY. **A duration check
cannot see it** — the files are exactly the right length. Both are catastrophic training pairs, and
both are invisible to every text-side gate because the transcript and the IPA are fine and only the
PAIR is broken.

Relabelled `defective_audio` (490 rows, was `recognizer_short`). Coverage checked: **no phone is lost**,
and es_419's owned primitives survive — ʝ 1456→1190, β 5051→4104. `scan_silent_audio.py` now measures
this directly rather than trusting the recognizer proxy, and is running over all 28 languages.

**And the other empty-ASR rows are three different things, which is why the bucket needed reading:**

    es_419  490   SILENT audio                       -> data defect, excluded
    cy_gb   338   TRUNCATED audio (1.80 s, rms 2e-4) -> the Run 36 defect, already excluded
    sd_in    15   NORMAL audio (5.40 s, rms 0.052)   -> the recognizer simply failed. NOT a defect; kept

A status column is a work log, and one bucket held three unrelated causes.

### ⚠ One of the "unfixable" Nguni click cases was the CASING WALL in disguise

Run 40 measured the xh/zu click classifier (native 12/12, foreign 16/19) and called the three
failures — `china`, `canada`, `ceo` — the intrinsic ceiling of a phonotactic test. **That was right
for two of them and wrong for the third.** `canada` is CV.CV.CV and genuinely indistinguishable from
a native word by shape. But `ceo` is not a word at all — it is an INITIALISM, and uppercasing it
makes xh/zu spell it `sˈiː ˈiː ˈɔː`, which is exactly what the recognizer heard (`s i i o`) where we
were emitting the click `kǀˈɛːɔ`. Inert in the seven other languages that have it.

Calling something an intrinsic limit is a claim worth re-testing per case, not per class.

### Also added to the allowlist, from the corpus-wide differential

`nba` (×97), `fbi` (×72), `cctv` — all read as impossible onset clusters when lowercased (`nbˈa`,
`fβˈi`, `n̪ˠˈəbˠə`, Fula's geminate `t͡ʃːtv`) and correctly as letter names when uppercased. None is
a word in any of the 28 corpora, which is the test that matters for a globally-applied list, and each
pays out across a dozen languages because FLEURS is parallel.

⚠ **`eu` REJECTED and recorded** — the trap the collision gate was built for. It is the European
Union in a few rows and an ordinary WORD in far more: Welsh *eu* "their" ×660, French *eu* ×46,
Portuguese *eu* "I" ×15. Uppercasing it globally would spell out a pronoun in three languages.

### Checked and NOT a bug: the Fula letter names

`un` in Fulfulde reads `ˈu nˈa` where the reader said `uː e n`. Fula's letter-name table is cited to
the **UNESCO Bamako alphabet** — a letter's name is the letter plus -a (ba, ca, da…) — so `n`→"na" is
correct by a documented standard. The discrepancy is that readers **code-switch to English letter
names for international acronyms**. That is a reader behaviour, not an engine error, and it belongs
with the parked nativisation question rather than in a fix.

### The exclusion now takes TWO sources, because they find different things

`exclude_defective.py` unions the DB `status` column with `work/silent_audio.tsv`:

  · the DB carries what the RECOGNIZER-based sweep concluded — including the Welsh TRUNCATION, which
    is audible and would never trip a silence test;
  · the TSV carries what measuring the WAVEFORM found — the Spanish files that are full length and
    empty, which no duration or transcript check can see.

Neither is a superset of the other. Total exclusion is now **988 utterances** (was 498): cy_gb 480,
es_419 490, ff_sn 9, sd_in 6, am_et 2, ar_eg 1. **No phone is lost in any language.**

⚠ **The scan itself had to be rewritten before it could be trusted to finish.** The first version
accumulated results in memory and wrote once at the end; the run was killed partway and left NOTHING
on disk, so all 66 languages would have had to be rescanned. It now appends per language with a
`#done` marker and resumes. A long job that cannot be resumed is a job that has to be run twice.

It is scanning all **66 downloaded languages / 132 GB**, not just the 28 in the corpus — the extra
cost is small next to the rescan risk, and the user plans to expand past 28, so the answer is worth
having in advance.

### ⚠ THE DISTANCE METRIC HAS A RESOLUTION FLOOR, AND A BIAS — do not use it to judge initialisms

Re-phonemized on the extended allowlist (+nba +fbi +cctv +ceo +usa, and the Turkish locale fix):
**113 rows changed across 19 languages**, every visible example a repair — `ga n̪ˠˈəbˠə → ˈɛnʲ…`
(the *nba* case), `zu kǀˈɛːɔ → sˈiː…` (the click), `cs ˈusa → ˈuː ˈɛs ˈaː`, `es fβˈi → ˈefe…`.

Then I scored them against the audio the way Run 40 scored the earlier fixes, and got the opposite of
what I expected:

    moved CLOSER 95      moved further 131      mean delta +0.0011

**That is not evidence against the change. It is the metric failing.** Measured:

    median folded utterance          120 phones
    median size of the edit            2 phones     -> 1.7% of the string

A 1.7% edit sits far below this recognizer's own error rate, so the split is noise. Run 40's result
was trustworthy for the opposite reason — 1,927 rows at 4.2:1 is a ratio no noise process produces —
not because the metric is sharp.

⚠ **And there is a SYSTEMATIC BIAS on top of the noise, in one direction.** Letter-name expansion
*lengthens* the string (`nbˈa` → `ˈɛn bˈiː ˈeᶦ`), and the recognizer under-detects short spelled
syllables. So whole-utterance distance penalises correct spelling. A metric that is biased against
the very change being tested cannot adjudicate it, however much data you give it.

**The right instrument is TOKEN-LEVEL: does the recognizer show letter names where the token sits?**

    ca  nba   ASR  … u n i t s | e m b i eɪ | b a s u s p …    "em-bi-ay"     ✓ spelled
    am  fbi   ASR  … b ə eː f eː v i aɪ | i ɡ ə f i t …        "ef-bi-ay"     ✓ spelled
    de  ceo   ASR  … ɛ p ə l | ts iː aʊ | s t iː f dʒ ɔ p s …  German names   ✓ spelled

Same instrument that carried Run 33 (`rspca` → "ar-es-pee-see-ay"), the `un` question (`j u e n`),
and the Nguni `ceo` (`s i i o`). The additions stand on that evidence, not on the aggregate.

**Rule for the rest of this work: aggregate distance answers "did a WIDE change help?" It cannot
answer "was this ONE token right?" — that needs the audio at the token.**

### The full 66-language silence sweep — es_419 is ALONE

All 66 downloaded languages / 132 GB measured for silent audio:

    es_419  490        ar_eg 1   ckb_iq 1   fa_ir 1   hr_hr 1   lb_lu 2
                       ^ singletons, four of them outside the 28-language corpus

**496 silent files across 66 languages, and 490 of them are Spanish.** That is not a general FLEURS
quality problem — it is one broken split. The other 65 languages are effectively clean, which also
answers the question in advance for the planned expansion past 28.

⚠ And `cy_gb` returns **0 silent**, correctly: Welsh audio is TRUNCATED, not empty. That is exactly
why `exclude_defective.py` unions two sources — a silence test alone would have missed all 585 Welsh
files, and a duration test alone would have missed all 490 Spanish ones.

### Final corpus

    exclude -> patch manifests -> sampling weights -> webdataset

    73,798 train + 2,123 dev = 75,921 utterances, 28 languages
      defective-audio pairs present  : 0
      sentences leaking across splits: 0
      residual defect patterns       : 0

988 utterances excluded (cy_gb 480 truncated, es_419 490 silent, ff_sn 9, sd_in 6, am_et 2, ar_eg 1).
No phone is lost in any language, and U+0325 — the primitive Welsh solely provides — survives at
1,557 occurrences across 1,110 utterances.

## Run 42 — 2026-08-17 — The Nguni click issue: LEXICALISE the loans (PR #824, `a9272d3`)

Run 40 called this the intrinsic ceiling of a phonotactic test and Run 41 peeled `ceo` off it. This
run resolves the rest — not by improving the test, but by accepting that it cannot be improved.

### The defect, measured precisely

Every ⟨c/q/x⟩-initial token in xh/zu, crossed against **cross-language spread** (how many of the 28
parallel corpora contain it — the same discriminator the initialism work validated: an international
name appears in 9-21, a native Nguni word in 1-2):

    CLICKED 59 / REFUSED 64 tokens
    ⚠ clicked with HIGH spread (wrong): 12  — covid xdr cuerden cadwalder canada chhatrapati
                                             corniglia chile choudhary capuzzo carolina congo
    clicked with LOW spread (right):    cishe ×109, xesha ×21, qiniseka, qho, cwaka, cala…
    refused with LOW spread:            congress, crude, container — English, correctly refused

### The 12 fail for TWO different reasons, and only one is the "ceiling"

    fails SIGNAL 3 (looks Nguni)     canada chile carolina congo (+china)   — the documented trade
    fails SIGNAL 2 (not in CMUdict)  covid cuerden cadwalder chhatrapati corniglia choudhary capuzzo

The signal-2 group is phonotactically impossible in Nguni, so signal 3 already clears them; they fail
only because an English pronunciation dictionary carries no proper nouns.

### ⚠ THE OBVIOUS FIX WAS MEASURED AND IT IS WRONG

Relaxing signal 2 (the dictionary check) would fix seven of the twelve in one line. Tested against the
corpus, it also newly routes **real Nguni words** to English: `xakuvakashelwa`, `xawusezantsi`,
`qinsekisa`, `qhwa'`, and — worst — **`compyutha`, the NATIVISED borrowing of "computer"**, which must
stay Nguni. **A lexicon adds words one at a time; loosening a signal removes a guard from all of them
at once.** Signal 2 stays.

### ⚠ AND THE READINGS SPLIT TWO WAYS, which is why no rule could have worked

Against the audio, Nguni readers do not treat these alike:

    canada   ASR `b a s e k a n a d`      -> /kanada/,  NATIVISED with a plain k
    congo    ASR `l i k o o v k o ŋ ɡ`    -> /kongo/,   nativised
    mexico   ASR `u m e ð u k s i k o`    -> /meksiko/, nativised, ⟨x⟩ as its Latin /ks/
    china    ASR `tʃ h aɪ n n a`          -> ENGLISH /tʃaɪna/
    chile    ASR `o d i tʃ aɪ l`          -> English
    carolina ASR `e r o l aɪ n a`         -> English (the ⟨aɪ⟩ is the tell)

Long-established borrowings are nativised; newer names keep English phonology. That is a fact about
each word, not about its shape — **which is the definition of something that must be lexicalised**, and
is how English handles its own loans.

Two verdicts per entry: `declick` (Nguni phonology, click letter read as its Latin value) or `foreign`
(the English reader, the path 435 tokens already take). `src/languages/zulu/nguniLoans.ts`.

### Result

    wrongly-clicked high-spread tokens: 12 -> 1     (that one, `xdr`, is fixed corpus-side already)
    xh+zu rows whose IPA changed: 104
      moved CLOSER to the audio: 73     further: 21     3.5 : 1

⚠ **Note the contrast with Run 41's initialism scoring (95 closer / 131 further, unusable).** The
metric works here and failed there for a structural reason: de-clicking is LENGTH-NEUTRAL (`kǀ`→`kʼ`),
so it carries none of the expansion bias that made whole-utterance distance penalise letter-name
spelling. The instrument is fine when the edit does not change the string length.

Native words verified untouched — `cha`, `cela`, `caba`, `cima`, `coca`, `xhosa`, `cishe`, `xesha`,
`qiniseka`, `cwaka`, `ukucela`, `compyutha`, `qho` all keep their clicks, pinned by test. 2,964 xh and
1,951 zu rows still contain clicks, as they should.

⚠ **One existing test pinned the OLD trade** (`china` keeps a wrong click, "the alternative was reading
`xhosa` as English"). That expectation is obsolete rather than wrong — the lexicon fixes named words
WITHOUT loosening the signal — so it now pins `cuba` and `cima`, still unlisted and still correctly
clicked.

Corpus rebuilt: **73,798 train + 2,123 dev = 75,921 utterances, 0 defective pairs, 0 split leakage.**

## Run 43 — 2026-08-17 — A new instrument, and a negative result across 26 of 28 languages

### First, a durability bug in the QC itself

Re-applying the automatic labels reverted the 490 silent-Spanish `defective_audio` rows straight back
to `recognizer_short`. `defective_audio` is in `AUTOMATIC`, so the pass overwrote a verdict I had set
with a one-off UPDATE — **the comment survived, the status did not**.

**A verdict inside an automatic category is only durable if the automatic pass can reproduce it.**
`apply_auto` now reads `work/silent_audio.tsv` itself, as a SECOND detector beside the existing
seconds-per-phone one. Neither subsumes the other: Welsh is truncated and fails on rate; Spanish is
full-length and empty, so its rate is unremarkable and the rate test cannot see it.

    defective_audio   611 -> 1101      recognizer_short  737 -> 247
    verified idempotent: two consecutive --apply runs now agree, which they did not before

### `confusion_pairs.py` — for defects too thin for the queue

Reading the queue row by row found every large class in this corpus, but it is a per-row instrument.
A wrong mapping costing ONE phone per utterance never reaches a worst-first queue at all: one phone
in a hundred does not move the distance.

So: align our IPA against the recognizer per row, count the 1:1 SUBSTITUTIONS, and — the part that
makes it a test rather than a list — **compare the profile of the `investigate` rows against the
`verified` rows of the same language.** Recognizer noise has the same profile in both. A real defect
class concentrates in the flagged set.

    ratio ~1.0  =  the flagged rows are just noisier rows; no distinct defect class
    ratio >> 1  =  something is different in KIND about what got flagged

### The result: 26 of 28 languages come back at ratio ≈ 1

    hi 1.1   ff 0.4   kk 1.2   cmn 1.2   sv 1.2   ca 1.4   am 1.4   xh 1.6   ja 1.7   de 1.7
    ru 1.8   th 1.8   es 2.0   cs 3.1   cy 4.1   om 4.3   ha 5.6   zu 6.1   pt 6.6
    (vi tr ta sd ko ga ar: too few flagged rows to form a profile at all)

Every top pair is an expected recognizer-convention difference — vowel-quality collapse (`i→ɪ`,
`u→ʊ`, `a→ə`), rhotic notation (`ɾ→r`), aspiration and length unmarked. **No hidden systematic defect
anywhere in those 26.**

### The two outliers are both READER or RECOGNIZER facts, not ours

**en_us, ratio 188.** `ɚ→a`, `ɝ→a`, `ɛ→e`, `ə→e`. Reading the rows: `for the first time` →
`f ɔ ɾ ð e f a s t aɪ m` — a tapped `ɾ` for /ɹ/, `e` for /ɛ ə/, `a` for /æ ʌ/, and **no r-colouring at
all**. That is a non-rhotic, L2/Romance-accented reading against our General American IPA.

Measured across the whole language rather than the flagged tail: of 2,227 rows with ≥4 rhotic
positions, the median rhotic-LOSS rate is 0.20 (ordinary recognizer behaviour) and only **13 rows**
exceed 80% loss, 80 exceed 60%. Reader accent variation, negligible in scale, and not a defect in the
IPA — which is canonical GA by design.

**fr_fr, ratio 31.** `ʁ→ɾ` / `ʁ→r`. Only **1 of 77** flagged rows carries ≥3 of them, and in that row
the recognizer writes `ʁ` word-initially and `r` elsewhere in the same utterance. Its own notation
variance, spread thin. Note fr and en have the two LOWEST median distances in the corpus (0.085,
0.191) — a language that aligns superbly has a tail made of outliers rather than errors, which is why
the enrichment ratio is high there and nowhere else.

### What this establishes

The per-language read plus this instrument now cover both failure shapes: **concentrated** defects
(which the queue finds) and **thin, systematic** ones (which the profile comparison finds). Neither
finds anything further. The corpus's systematic phonemization defects are worked out, and the
remaining queue is recognizer noise, reader accent, and the two documented residues (Nguni
`china`-class ~19 rows, the xh/zu mid-phrase routing split).

## Run 44 — 2026-08-17 — The consonant-skeleton matcher (user's suggestion): built, VALIDATED, and it confirms

The user's observation: comparing our IPA to the recognizer's CONSONANT SKELETON is something a model
could do, but probably a heuristic could too. Run 43's own data argues for it — **every top
substitution in every one of the 28 languages was vowel quality** (`i→ɪ`, `a→ə`, `u→ʊ`, `ɛ→e`, `ɔ→o`).
That is not error, it is two conventions disagreeing about vowels by design, and it is the bulk of the
measured distance. It is the NOISE FLOOR that hid everything smaller — the floor Run 41 hit head-on
when a 2-phone edit in a 120-phone utterance could not be resolved at all.

Consonants are the opposite: both sides agree closely, and a missing or wrong consonant is much more
likely to be ours.

### ⚠ VALIDATED AGAINST THE CORPUS'S OWN KNOWN DEFECTS, and my first version FAILED that test

A new metric that has never been shown to detect anything is not evidence. The DB keeps the pre-fix
IPA in `ipa_prev`, and the merged fixes were largely CONSONANT defects, so the metric can be scored on
2,314 rows where a known fix changed the IPA — it must separate old from new more sharply than the
full distance, which manages 3.5:1.

    FULL distance (baseline)          1651 better /  472 worse   3.5 : 1
    skeleton, my first parameters      555 / 307                 1.8 : 1   ⚠ WORSE
    skeleton + glides                  619 / 488                 1.3 : 1
    skeleton, h and ʔ KEPT            1493 / 322                 4.6 : 1   ✓

**I had excluded `h` and `ʔ`,** reasoning that the recognizer inserts and drops both freely. Sound in
the abstract, and wrong here: the single largest defect this corpus ever had was Kazakh ⟨ь⟩/⟨ъ⟩
emitting a GLOTTAL STOP in 408 rows. Excluding ʔ threw away the evidence for the biggest fix — and it
did so by producing TIES rather than disagreements, which is worse than a wrong answer because it
reads as agreement. Glides genuinely do hurt (`j→ɪ` is a top-ten pair in th and pt), so they stay out.
`--drop-weak` re-runs the losing variant so the claim stays checkable.

**The tuned metric beats the full distance: 4.6:1 against 3.5:1.**

### It finds 1,109 candidates the full-distance queue passed

Concentrated exactly where the full metric was blind — `sd_in` 116, `vi_vn` 59 (all new), `ga_ie` 50,
`ko_kr` 48 — the high-median-distance languages whose vowel noise swamped everything.

### Read across seven languages, and every one is reader or recognizer, not us

    sd vi ga ko   the RECOGNIZER cannot read the language — Irish comes back in English phones
                  (`ð ə k uː n ɑː ʌ v d eɪ t aɪ ɹ v` for `lʲˈɛ kˈuːn̪ˠəw dʲˈeː t̪ˠˈawəɾˠfˠə`)
    es_419        the RECOGNIZER'S DIALECT — it writes Castilian θ where Latin American Spanish
                  correctly has seseo (`t e n θ j o n` vs our `tensjˈon`). Ours is right.
    en_us         reader accent, the non-rhotic/L2 class already measured in Run 43
    fr_fr         READER VARIATION IN YEAR FORM. `1945` — we write `mil nœf sɑ̃ …`, some readers say
                  `dix-neuf cent`. Counted: of 127 rows with a 16xx-19xx year, 44 readers use "mil"
                  and 18 use "dix-neuf". Both are correct French; **we picked the majority form.**
    ha_ng         reader CODE-SWITCHING: `cosmonaut no 11` read as English "number eleven"
                  (`n o m b ə e l e v`) where we read `nˈo` + Hausa *goma sha daya*. `no <digit>` is
                  19 rows across 7 languages and `no` is an ordinary word in several — no rule.

### What it establishes

The suggestion was right and the instrument is a genuine improvement — it is now the sharper of the
two and worth keeping. **It also confirms Run 43 rather than overturning it:** with the vowel noise
removed and 1,109 fresh candidates to read, there is still no systematic phonemization defect left to
find. What remains is recognizer limitation, recognizer dialect, reader accent, reader variation and
reader code-switching — none of which is ours to fix, and all of which are now named.

## Run 45 — 2026-08-17 — All 66 languages aligned; and a BOUND on what this QC can ever claim

38 new languages phonemized (101k utterances, 0 errors) and aligned. The DB now holds **66 languages,
176,526 rows**. Ran the full battery — labeller, skeleton/full ratio, consonant confusion profiling.

### ⚠ THE RECOGNIZER CANNOT HEAR THE PHONES SOME LANGUAGES ARE IN THE CORPUS *FOR*

The skeleton/full ratio test flags a language whose CONSONANTS disagree more than its whole string —
the signature of a language-wide problem that MAD flagging structurally cannot see. Three new
languages came up: kn_in 1.14, te_in 1.12, bn_in 1.04, joining es_419 1.38 and ff_sn 1.15.

Their top substitutions are all one thing: `ɾ→r`, `ɭ→l`, `ʋ→v`, `ʈ→t`, `ɖ→d`, `ɳ→n`, `ɦ→h`. So I
checked the recognizer's actual output alphabet across all 176k utterances:

    phone        ours (66 langs)      recognizer
    ʋ                    136,105               0
    ɦ                     76,815               0
    ʈ / ɖ / ɳ / ɽ   64,953 / 38,818 / 38,543 / 5,116    0
    ɓ / ɗ / ʄ       27,715 / 19,316 / 3,962              0
    ǀ / ǁ / ǃ        4,104 /  2,733 / 4,507              0

**`facebook/wav2vec2-xlsr-53-espeak-cv-ft` HAS NO RETROFLEX STOPS, NO IMPLOSIVES, AND NO CLICKS.** Not
"rarely emits" — zero, across 176,526 utterances.

⚠ **And this lands precisely on the primitives the corpus was BUILT around.** Languages were selected
as OWNERS of census primitives: Fula for ʄ/ɠ and the implosives, Nguni for the clicks, Sindhi for its
implosives, the Indic set for the retroflex series. **The audio gate is blind to exactly the sounds
those languages are present to provide.** Every "the recogniser disagrees" verdict in those languages
was structurally guaranteed and says nothing about our IPA.

    corpus-wide: 3.6% of our phone tokens are invisible to the recognizer
    cmn 12.5%   pa 10.1%   mr 10.1%   hu 9.6%   ta 8.9%   gu 8.7%   te 8.3%   hi 8.0%
    it / pt / ro: 0.0%

This retro-explains every ratio outlier found so far — es_419 (allophones + a Castilian θ bias),
ff_sn (implosives), and now kn/te/bn (retroflexes). **None was ever ours.** It also means the
per-language medians are not comparable across languages in the way a naive reading would suggest:
a language scores badly in proportion to how much of its inventory the instrument lacks.

**What this does NOT undermine:** every defect this QC actually found — the fula digraphs, the kazakh
glottal stop, hindi ज्ञ, the japanese counter, the ⟨q⟩ leak, the silent Spanish audio, the truncated
Welsh audio — was found on phones the recognizer DOES have, or by reading, or by measuring the
waveform. The bound is on what the gate can VERIFY, not on what it has found.

## Run 46 — 2026-08-17 — Cross-word spirantization (the session's strongest fix), and a stress-coverage audit

### ⚠ SPIRANTIZATION STOPPED AT THE WORD EDGE — es/ca/gl, PR #827 (`22ba3bf`)

`spirantize()` guards on `i === 0`, which is WORD-initial, because a per-word function has no other
context. **Its own comment says "except utterance-initial."** So the identical environment was read
two ways depending on which side of a space it fell:

    nada       -> nˈaða            la duda   -> la dˈuða
    cabota(ca) -> kəβˈɔtə          la bota   -> ɫə bˈotə

The engine had already committed to marking allophony; applying it in one environment and not the
same environment across a space is the defect.

**Validated against the audio, and it is the strongest signal this corpus has produced:**

    1,500 es_419 rows re-scored:  1,292 CLOSER   36 further   = 35.9 : 1
    median skeleton distance 0.146 -> 0.103        full 0.108 -> 0.085

For comparison, the previous best was 4.2:1 for seven fixes combined. Corpus-wide after re-phonemizing:
es_419 ratio 1.35 -> 1.22, ca_es -> 0.37.

⚠ **This came out of the user's consonant-skeleton suggestion.** It was invisible until two things were
true at once: vowels removed (they are the noise floor), and the recognizer's blind phones folded out.
es_419 was then one of only two languages whose consonants disagreed MORE than their whole string —
**and it did not move when the blind-phone fold was applied**, which is what proved the cause was ours
and not the instrument's.

`pt` is deliberately untouched: Brazilian Portuguese does not spirantize at all, so it has no
inconsistency. Verified — 0 of 2,793 pt_br rows changed.

**es_419's residual 1.22 is fully recogniser-side:** `s→θ` in 72% of rows (its espeak-European labels
against a seseo variety), `ɾ→r` notation, and `n→m` / `n→ŋ` — Spanish nasal place assimilation, which
our engine deliberately leaves broad and says so.

⚠ **Three mistakes of mine on the way, all in the TEST update rather than the fix:**
  · a blanket regex over test files damaged OTHER languages — Russian `dva`→`ðva`, Welsh `dˈeːɡ`→`ðˈeːɡ`,
    Portuguese `de dˈɔlɐɾɨʃ`. My "IPA-looking" filter matched `ɫ`/`θ`/`ɾ`, which they also use. Reverted.
  · one expectation was INVERTED — `xix` alone is utterance-initial and correctly keeps its stop.
  · **Catalan needed a lookahead, not a capture.** The sibilant test consumed its character, advancing
    past the left context the NEXT word needed, so a second stop in the same clause was silently missed
    (`segons de vídeo` -> `ðə bˈiðəu`). Non-overlapping replacement makes any consumed right-context a
    bug of exactly that shape.

30 existing expectations changed, each verified individually. Two confirm the guards: `802.11n` keeps
`bˈujt` utterance-initially while `ðˈos` spirantizes, and `el Dr. García` keeps `doktˈoɾ` after `el`.

### The 38 new languages: enrichment finds nothing

Consonant-profile enrichment (flagged rows vs the rest) across all 38 returns ratios of 1-4× with tiny
counts — the flagged rows have the same profile as the rest. No distinct defect class, the same
negative the 28 gave.

### ⚠ STRESS COVERAGE — a property of the EXPANSION set, measured per word

Reading the hr_hr queue turned up something no metric flags: its IPA carries **no stress marks at
all**. Measured across all 66:

    ZERO stress:      af_za  hr_hr  is_is  lb_lu  sl_si  sr_rs        (all NEW, none in the 28)
    under 1%:         yue ckb uk as bn bg  (+ cmn/ja, which use TONE instead — correct)

**Of the 28 training languages, only two are stress-poor and non-tonal, and both are principled:**
fr_fr 4.3% is one PHRASE-FINAL accent per rhythmic group, exactly as documented ("French has no
lexical stress"), and am_et 1.4%. **The training corpus is not affected.**

### ⚠ AND "DELIBERATE DEFERRAL" DID NOT SURVIVE BEING CHECKED

I first reported sr/hr/sl/af as principled deferrals because the engines SAY SO. The user pushed back
that deferral is often a label for unfinished work. Checking it changed three of the six verdicts.

**The support I offered was worthless.** I implied "our referee also omits stress." Control across the
whole referee set: **0 of 64 `wikipron-broad` files mark stress on more than half their entries.** That
source strips it universally, so its silence says nothing about any language.

Re-derived per language, on evidence rather than on the docstring:

    af_za   ⚠ NOT justified — THE DATA WAS IN HAND AND DISCARDED. The RCRL dictionary referee has
            stress on 93% of 27,435 entries, and a second source (wiktionary) on 61%. The engine
            SHIPS a lexicon derived from that same RCRL data — af-rcrl-lexicon.tsv, 25,117 entries —
            with stress stripped on import:
                referee  a.fri.ˈkɑːns   mə.ˈny.tə   rə.ˈxiə.rəŋ
                shipped  afrikɑːns      mənytə      rəχiərəŋ
            The header's "Stress … not modelled (folded)" describes the outcome; it does not justify it.

    is_is   ⚠ NOT justified — no data is NEEDED. Icelandic primary stress is word-initial, essentially
            without exception in native vocabulary. It is a rule, not a lexicon, and the engine header
            does not mention stress at all.

    lb_lu   ⚠ NOT justified — the rule ALREADY EXISTS. luxembourgish.ts records a measured stress
            placement ("net +3.9pp over always-first-syllable") and then emits no mark. The work is
            done and the output is suppressed.

    sl_si   constrained by DATA, which is a narrower claim than I made. Slovene has free lexical
    sr_rs   stress; BCS has 4-way lexical pitch accent. Neither is derivable from spelling, and NO
    hr_hr   source in this repo carries it for them (wikipron-broad 0%, epitran 0%, no kaikki file).
            So implementing it needs new data sourced first — not that it is impossible in principle.

**Three of six were unfinished work wearing a deferral label.** The remaining three have a real
obstacle, but the honest statement is "no stress-bearing source is available here", not "not
derivable". None of the six is in the 28, so the training corpus is unaffected — but `af_za` is the
one to fix first, because fixing it is an import change rather than a research problem.

## Run 47 — 2026-08-18 — Run 46's sl/sr/hr verdict was wrong, and a pipeline round that found five defects in itself

### The correction

Run 46 closed the stress audit by calling `sl_si`/`sr_rs`/`hr_hr` "constrained by DATA … NO source in
this repo carries it for them (wikipron-broad 0%, epitran 0%, no kaikki file)." **All three clauses of
that were wrong**, and the reason they were wrong is that the instrument was `grep -c 'ˈ'`.

South Slavic prosody is not written with `ˈ`. It is a tone diacritic on the vowel — `/ǎbdaːl/`,
`/abdǒːmen/`, `/ôːn/`, `/planéːt/`. Counting the wrong character returns zero forever, and "the grep
found nothing" had been promoted to "the data does not exist."

- **kaikki dumps exist**: Slovene 5 499 headwords with IPA, 97.8% accented; Serbo-Croatian 50 692,
  97.8% accented, 28 190 Latin + 24 875 Cyrillic in one unified dump serving all three engines.
- **And the committed referee already carried it.** `sr.wikipron-hbs-latn.tsv` marks the accent as
  `â ǎ ê ô` on 26 126 of 26 486 rows, and *its own header says so*. `referee-eval`'s backbone strips
  exactly those marks, so nothing in the repo had ever looked at them.

Shipped in the phonemizer since: stress position (#832), the four-way pitch accent in the fleet's Chao
notation (#833), and a suffix-conditioned OOV transition tier (#834). Measured 99.3% position / 99.7%
contour against the referee. `af_za` (#828), `is_is` (#829) and `lb_lu` (#830) are also done, so the
whole Run 46 audit is closed.

⚠ **A coverage number in that work was also wrong and is corrected here.** The first estimate of
lexicon coverage said 83–84% of `sr_rs` tokens. FLEURS TSVs have seven columns and **column 5 is
character-separated** (`i m a m o | j e d n o g o…`); reading the whole file with a word regex made every
individual letter a token, and single letters (`i a u o e`) are all in a lexicon as one-letter words.
Measured from column 3 only, the real figure is **43.7%**.

### The round-2 pipeline run, and the five things wrong with it

Extending QC to the languages fetched since the last round. Every one of these was found by checking
rather than by a failure, except where noted.

1. **The fetch was hung, not finished.** 13 h 20 m elapsed against 7 m 28 s of CPU, log silent for
   11.5 h, the in-flight `nso_za` frozen at 335 MB. `hf_hub_download` has a 10 s read timeout (hub
   1.8.0) and it did not fire — a dead-but-open socket leaves the call blocked with no way for the
   caller to notice. Fixed with a **stall watchdog**: the download runs in a child process while the
   parent watches the byte count, and a cache that has not grown in 300 s gets the child killed and
   retried. Watching progress rather than imposing a deadline is the point — a slow link keeps its
   time, only a frozen one is cut. It fired for real on `ny_mw` within the hour and recovered.

2. **Two languages were being skipped permanently.** The "already have it" test was a `listdir` of
   `data/`, which counts a language as cached the moment `hf_hub_download` creates its *folder*.
   `ast_es` and `nso_za` both held empty `audio/` dirs from earlier stalls, so they read as complete
   and would never have been retried. The dry-run said 9 missing; it was 11. Now tests for the tarball.

3. **24 of the 25 alignment-ready languages had no IPA at all.** `load_ipa()` returns `{}` on a missing
   byid file, so the aligner would have logged rows comparing recognizer output against an empty string
   and scored every utterance 1.0 — a full GPU pass producing a uniformly "defective" corpus.
   Phonemization is a prerequisite, and it was not in the plan.

4. **Two FLEURS codes do not match the registry.** `fil_ph` is Filipino to FLEURS and `tl` to the
   registry; `ny_mw` is `ny` (639-1) to FLEURS and `nya` (639-3) to the registry. Both now in the
   `VARIETY` map beside the existing `ar_eg → arz`. Found the second one by pre-checking the batch
   still downloading, rather than waiting for it to fail mid-round.

5. **The label step was missing from the chain, and its order is load-bearing.**
   `asr_align_label.py` writes the `status` column and `exclude_defective.py` reads *that*, not the
   report — without it the new languages get no `defective_audio` at all. It also reads
   `silent_audio.tsv` **at import**, so it must follow the silence sweep, not precede it. Correct order
   is align → sweep → label → report.

### Two traps checked and cleared

- **A CPU venv that imports cleanly.** Of the venvs carrying `soundfile`+`torch`+`transformers`, only
  `train_venv` and `ar-diac-venv` have CUDA; `export_venv` is `+cpu` and would have run the whole
  ~66 k-utterance pass on the CPU with no error at all — days instead of an hour. Checked
  `cuda.is_available()` rather than stopping at the first venv that imported.
- **Today's new stress and tone marks cannot distort the scoring.** `fold()` strips Unicode categories
  Lm (`ˈ ˌ ː`) and Sk (`˥ ˩`), and the `defective_audio` rate test uses `fold(ipa)` rather than
  `LENGTH(ipa)` — with a comment recording that someone already hit that exact bug (608 rows vs 611).
  Verified end to end: `sedamdˈe˩˥setix → sedamdesetix`.

### Coverage after this round

66 languages aligned before; 25 more in flight, 11 more still downloading → 102 of 102 once done. The
silence sweep's `#done` markers matched the alignment set exactly at 66, so the new languages have never
had their **waveforms** measured — which is a separate defect surface, and the reason the two sweeps are
unioned: the recognizer pass caught the 585 truncated Welsh files (audible, invisible to a silence
test), the silence pass caught the 490 full-length-and-empty Spanish ones (unremarkable duration,
invisible to the rate test).

## Run 48 — 2026-08-18 — The recognizer-inventory fold: a monotone win, and two folds the data refused

`km_kh` came out of round 2 at median distance 0.524. Chasing that produced a fold that helps 84
languages and rejected two of the four changes originally proposed — and also produced a wrong diagnosis,
corrected at the end of this run.

### What the DB says, over all 221 469 aligned utterances

Phones we emit at least 2 000 times that the recognizer returns less than 1% as often — 30 of them,
**902 870 tokens = 3.67% of everything we write**, which reproduces the bound established in Run 45 from
a completely different direction:

    ʋ 158956/0   ɫ 90312/0   ɦ 76815/0   ʈ 66306/0   ʂ 46938/0   ɖ 41067/0
    ɳ 38765/0    ɓ 33146/0   ɗ 23654/0   ɽ 5120/0    ʄ 3962/0    clicks 11344/0

Not noise and not our error: `wav2vec2-xlsr-53-espeak-cv-ft` has no symbol for them. Unfolded, a language
dense in these carries a fixed penalty before correctness enters into it, and the 3×MAD test then finds
nothing — the same way `ga_ie` hid behind modifier letters and `vi_vn` behind tone digits.

### Result — nothing got worse

| lang | before | after | Δ |
|---|---|---|---|
| pa_in | 0.442 | 0.356 | +0.086 |
| cmn_hans_cn | 0.374 | 0.289 | +0.086 |
| mr_in | 0.447 | 0.374 | +0.073 |
| ta_in | 0.617 | 0.551 | +0.066 |
| uk_ua | 0.399 | 0.339 | +0.060 |
| **median of 84 languages** | **0.366** | **0.349** | |
| **languages made worse** | | **0** | |

The gains land exactly where the phonology predicts — Indic retroflexes, Mandarin `ʂ ʑ`, Ukrainian `ɫ`.

### ⚠ Two folds the data refused

**`c → tʃ`.** Khmer made `c` look unhearable — ours 1 731 against the recognizer's 10 *there*. Corpus-wide
the recognizer writes it **10 292** times against our 49 987: a fifth, not a hundredth. `tʃ`/`dʒ` are
contrastive across many of these languages and folding globally would have destroyed a distinction the
recognizer does make. Generalising from one language was the error; the DB-wide count is the check.

**Dropping `ʔ`.** The recognizer hears it barely better (737 against our 120 940), so the case looked
identical — but Run 44 already ran this experiment. Dropping `ʔ` scored **1.8:1 against 4.6:1** for
keeping it, because the largest defect this corpus has ever had was Kazakh ⟨ь⟩/⟨ъ⟩ emitting a spurious
glottal stop in 408 rows. Folding it away deletes the evidence for that entire class of fix. The fixed
penalty is the lesser cost, and the reason is now recorded at the fold site so it is not re-proposed.

### Implementation notes

- `COARSEN` moved from `consonant_skeleton.py` into `asr_align_report.py` and re-exported. Defining it in
  the leaf and importing it upward created a circular import the moment the report needed it; the base
  module is where it belongs, since the skeleton tool already depends on the report for `dist`/`fold`.
- `coarsen()` applies **only inside `dist()`**. `asr_align_label.py` therefore picks it up for the
  verified/investigate split, while its `defective_audio` rate test still uses uncoarsened
  `len(fold(ipa))` — so clicks mapping to `""` cannot shift phone counts there.

### ⚠ And the km_kh diagnosis was wrong

I read `km_kh`'s high absolute median as the `ga_ie` degeneracy and proposed building a per-language fold
mechanism for its vowels. Both were wrong, and the check that settles it is whether the TAIL separates,
not where the median sits:

| lang | median | MAD | flagged >3×MAD |
|---|---|---|---|
| km_kh | 0.480 | 0.056 | 62 (**3.7%**) |
| ta_in | 0.551 | 0.059 | 48 (2.0%) |
| gl_es | 0.108 | 0.026 | 60 (2.8%) |
| ro_ro | 0.165 | 0.037 | 84 (2.9%) |
| be_by | 0.316 | 0.040 | 62 (2.5%) |

`km_kh` flags MORE than any of them, with a normal MAD. Nothing like `ga_ie`, whose investigate list came
out genuinely empty. The metric is relative to each language's own distribution by design — that is the
whole reason Run 45 chose 3×MAD over an absolute threshold — so a language the recognizer finds hard is
already absorbed, and a high median is not a defect to engineer away.

**This instrument is a coarse detector of SERIOUS disagreement, not a mechanism for realigning vowels.**
`km_kh`'s top outlier sits at 0.970 against a 0.480 median with recognizer output like
`b aɪ s a n ɛ ɡ p oː v i t uː` — that is the tool working. Whether each flagged pair is our bug, reader
divergence, or recognizer artefact is what triage decides; the metric should not pre-resolve it.

The per-language fold mechanism is therefore **dropped, not deferred**. The global fold stands on its own
merit: it removes a symmetric penalty that carries no information either way, and nothing regressed.

## Run 49 — 2026-08-18 — Round 2 complete: a 66k-utterance control that reframes the Welsh finding

25 languages phonemized, aligned, swept and labelled. The DB goes 66 → 91 languages, 176 526 → 242 894
utterances.

    verified           233360
    investigate          7626
    defective_audio      1133
    recognizer_short      767
    defect                  4   ← hand-set
    reader_divergence       3   ← hand-set, survived a global --apply
    convention              1   ←

### The result that matters is a NEGATIVE one

The 25 new languages contributed **10 `defective_audio` rows out of 66 368 utterances = 0.015%**, and
**one** silent file (`hy_am`, RMS 3.5e-5 over 2.1 s).

    cy_gb   585 / 3427   = 17.1%   truncated
    es_419  490          silent
    new 25   10 / 66368  =  0.015%
              1          silent

Three orders of magnitude. This is the control the Welsh finding never had: before, "17.1% of cy_gb is
truncated" rested on Welsh looking worse than ~40 other languages, which invites "maybe the detector is
too aggressive." Now there is a 25-language, 66 368-utterance baseline where the same detector, unchanged,
finds essentially nothing. Welsh is not at the bad end of a distribution — it is off it.

**This strengthens `docs/fleurs_cy_gb_truncated_audio.md` materially** and the same argument covers the
Spanish silence. Worth adding to that report before filing upstream.

Secondary confirmation: the three hand-set verdicts survived a global `asr_align_label.py --apply`. The
`status IN AUTOMATIC` guard works in practice, not only in its comment — which is the property that made
`defective_audio` durable in the first place (Run 45).

### Two pipeline gaps, both the same shape

Neither produced an error. Both would have produced a pipeline that looked like it had run.

1. **24 of 25 languages had no IPA before alignment was launched.** `load_ipa()` returns `{}` on a missing
   byid file, so the aligner would have compared recognizer output against an empty string and scored
   every utterance 1.0 — a full GPU pass yielding a uniformly "defective" corpus. Phonemization is a
   prerequisite and was not in the plan.

2. **Round 3's rows were left unlabelled.** Round 2's label pass ran before round 3 inserted, so those
   rows carry `status NULL` — invisible to `exclude_defective.py` and indistinguishable from "no defects
   found". Caught only because a status query showed 160 unlabelled rows that resolved to `ast_es`/
   `nso_za`, the two round 3 had just written. Each round now queues its own sweep → label → report.

⚠ **The recurring shape is a stage that quietly does not run.** Not a crash, not a wrong number — an
absence that reads as a clean result. Both are now flagged at the top of the scripts that own them.

### Ordering constraints worth keeping

- `asr_align_label.py` reads `silent_audio.tsv` **at import**, so it must follow the sweep, never precede it.
- Round 3's alignment was gated on round 2's **label** step, not on the whole chain: label does bulk
  `UPDATE`s over the whole table while an aligner `INSERT`s into it, and two SQLite writers is how a bulk
  relabel dies partway through. The report is read-only, so overlapping with that is fine.
- The silence sweep and the aligner both stream `train.tar.gz`; sequencing them is about disk, which only
  costs speed. The DB conflict above is the one that costs correctness.

### Environment traps

- Alignment needs `train_venv`. Bare `python3` has no `soundfile` and dies in one second — loud, harmless.
  The dangerous one is `export_venv`: it imports cleanly and is `torch+cpu`, so it would have run the whole
  pass on the CPU with no error at all. Check `cuda.is_available()`, not just that the import worked.
- Two FLEURS codes do not match the registry: `fil_ph → tl` and `ny_mw → nya`. The second was found by
  pre-checking the batch still downloading rather than waiting for it to fail mid-round.

## Run 50 — 2026-08-18 — Regenerating the stale IPA, and three bugs that all looked like success

After the day's phonemizer fixes, the question was which languages' stored `byid` IPA no longer matches
the engine. Timestamps cannot answer it — a file's mtime says nothing about which PR had landed — so the
check is empirical: re-phonemize a sample and diff against what is stored.

    identical              bs_ba es_419 ca_es zu_za xh_za      (400/400)
    marks only             is_is hr_hr sr_rs                   (400/400 differ only by ˈ ː ˥ ˩)
    SEGMENTAL              af_za 15/400   lb_lu 16/400   id_id 5/400   ms_my 1/400

Splitting marks-only from segmental is the whole point of the check: `fold()` strips suprasegmentals, so
marks-only drift cannot move an alignment score, while segmental drift can and does.

`lb_lu` came back clean on a per-word referee comparison (8 words changed, 0 better / 0 worse). `af_za`
did not — 2 better, 5 worse — which turned out to be #828 silently dropping 198 lexicon entries, fixed in
PR #835. **That fix is the return on this whole question**: without asking "should we regenerate?", the
198 dropped words would have been baked into the training corpus.

### ⚠ Three bugs today, and every one of them reported success

1. **`--redo` was a silent no-op.** `asr_align_corpus.py --redo` gets past the language-level skip, but the
   per-utterance guard `if wav in have: continue` ignores the flag, so every row already present is
   skipped. The run printed `af_za: 0 utterances in 4s` and exited 0. Fixed:
   `have = set() if a.redo else {...}`.

2. **`pkill -f "[a]sr_align_corpus.py"` killed its own shell** (exit 144). The bracket trick protects the
   *pattern*, but the same command carried a heredoc containing the plain filename — and `pkill -f`
   matches the whole command line.

3. **A wait loop deadlocked on itself.** `while pgrep -f "regen.sh"` matched the shell wrapper that had
   *written* the script, because that wrapper's command line still holds the heredoc, literal pattern and
   all. It waited 17 minutes for a process that had already exited.

   (And the status check used to diagnose it had the same flaw: `pgrep -f` for five patterns, all five
   present in the checking command, so all five reported RUNNING when one was.)

**The rule:** `pgrep`/`pkill -f` match the entire command line, and a shell that wrote a script via heredoc
carries that script's text for its whole life. Kill by PID; check with `ps | grep -v " grep "`.

### Verification that the re-align is complete, not partial

Every language's DB row count equals its `byid` row count (af_za 1032, lb_lu 2502, is_is 926, hr_hr 3461,
sr_rs 2944, id_id 2579, ms_my 2667). Worth stating because a partial `--redo` would look identical to a
complete one in the log — which is the same failure shape as the three above.

The marks-only three are re-aligned as well, even though their distances cannot move: a DB whose `ipa`
column disagrees with the shipped corpus is a trap for whoever reads it next.

## Run 51 — 2026-08-22 — The corpus was case-folded all along, and the codes were carrying a policy

### The finding that started it

`refresh_ipa.py --check` reported `de_de: 0 stale / 2987 ⚠ 2985 sentence_id not in byid`, which was my
own smoke test having truncated seven byid files. Repairing that led to asking why a German clock rule
had never fired, and the answer generalized:

    271798/271798 rows are fully case-folded (100.0%)
    102 languages are 100% case-folded

The FLEURS TSV carries BOTH transcripts — col2 raw (case + punctuation), col3 normalized — and the
ingest reads col3. ⚠ `initialism_casing.mts` exists to reconstruct capitals from a hand-reviewed list of
29 entries; the ground truth was in the adjacent column the whole time, for all 199,141 rows that have
one, plus 156,393 rows of punctuation that had no reconstruction at all.

### Neither column is a superset — the merge

A straight swap is wrong. Measured per language:

  yo_ng   the NORMALIZED column adds tone marks and sub-dot vowels raw lacks (`n`→`ń` ×263, `è`→`ẹ`)
          and expands abbreviations (`nn`→`nǹkan`). Tone is lexically contrastive; raw loses meaning.
  ig/ff/lg/so  the normalized column MERGED WORDS by deleting parens without a space
          (`(1040 km)mu` → `1040 kmmu`). There raw is correct.

So: norm supplies word FORMS, raw supplies CASE and PUNCTUATION. Content preserved (diacritics
included) on 100.00% of 198,412 cased-language rows; punctuation 99.3%, capitals 99.4%.

Three bugs found by measuring, not by reading the output:
  - the validity check used a canon() that strips combining marks, so it scored 98.9% "clean" while
    emitting DOUBLED diacritics (`ǹ̀`, `jẹ́́`) — blind to exactly what Yoruba needed;
  - without per-language alignment keys, sr_rs (Cyrillic vs Latin) kept 20.9% of its punctuation and
    emitted 648% of its capitals, Title-Casing whole sentences;
  - norm is not uniformly punctuation-free, so `cunami.` merged to `cunami..` (bs/hr/oc at 110-113%).

⚠ AND ONE FOUND BY TESTING THE CLAIM THE WHOLE THING RESTED ON. Restoring only the FIRST letter's case
destroyed zu_za `iHK` → `ihk` — a Nguni class prefix on an initialism, which starts lowercase so nothing
was capitalised. The engine then read it as a word (`ˈiːhkʼ`) instead of spelling it
(`iɛjˈiːt͡ʃʼi kʰˈɛːji`): the restorer undoing the very repair it claimed to obsolete.

### Predicted in advance: no QC movement. Confirmed.

    PROSODIC change only : 1902 (88.2%)   punctuation -> pauses; fold() cannot see it
    SEGMENTAL change     :  166 ( 7.7%)   ɹˈɑːv -> ˈɑːɹ ˈoᶷ vˈiː ;  el sr -> el seɲˈoɾ
    scored: closer 67  further 56  same 2018   mean Δ -0.00014

A wash, because `notate(units(...))` strips the pauses being restored. The justification is TTS prosody
and segmental correctness, NOT distance. Recorded before running so it could not be rationalised after.

Applied: 249,430 rows, 0 rejected, 209 hand rows untouched, all re-derived, 0 null IPA.

### The codes were carrying a judgement

`patch_manifest_ipa.py --db` reported 988 rows "not in source", clustered in cy_gb (480) and es_419
(490). Those were `defective_audio` rows sitting in TRAINING MANIFESTS. Cause: both languages were
ingested 2026-07-01, before the exclusion logic existed, and every later run skipped them because they
already had an npz. ⚠ 96 languages pruned, 6 not, for two months, with nothing in any log to say so.

⚠ THE CHEAP PATH FOUND IT AND A RE-INGEST WOULD HAVE HIDDEN IT. I had claimed the manifests needed a
5.5-hour re-encode; they did not — the npz is a pure function of the audio, verified byte-identical by
md5. `patch_manifest_ipa.py --db` refreshed 237,173 rows in minutes, and only because it declined to
touch those 988 did they surface at all.

Restructured so the artifacts have separate lifecycles:

    codes_<lang>.npz        write-once, append-only   GPU, only for genuinely new audio
    manifest_<lang>.jsonl   derived from npz ∩ DB     seconds, no GPU, no audio
    align.sqlite            the complete record        the label IS the judgement
    corpus_filter           applies policy at load     free

Appending is only sound because the encoder is reproducible, and that was MEASURED: 40 utterances
re-encoded from `en_us` (written 07-01) and 40 from `mi_nz` (08-22) came back bit-identical, across a
driver change, an onnxruntime change, and an `arena_extend_strategy` change made the same day.

### The long rows, and a threshold that was wrong by 2.6x

3,494 rows exceed the encoder's 30 s window and vanish with only a log count. Two facts, worth keeping
apart: the encoder genuinely cannot take them (a 256 s utterance needs a 7.9 GB attention buffer and
fails — quadratic attention, as the code comment claimed), and some of them are bad pairs anyway.

⚠ MY FIRST CUT WAS WRONG. A global "cps < 7" called 2,756 rows defective. But characters-per-second is
a property of the SCRIPT as much as the speech — per-language 5th percentiles run umb_ao 3.3,
cmn_hans_cn 4.4, en_us 7.8 — so a global cut condemns languages for writing compactly. Against each
language's own 1-30 s distribution: 1,013 anomalous, 2,419 ordinary speech that merely runs long.

Labelled `audio_overlong` (bad pair) vs `uncodeable_length` (fine pair, past the window). 2,541 of them
carried `verified`, which the QC pass never earned — they were skipped BEFORE scoring. Third time this
campaign that `verified` has turned out to mean "unexamined".

### Where it landed

    268,165 manifest rows -> 267,004 usable after the load-time filter
    codes and manifest agree exactly: 0 rows without codes, 0 codes without a row

## Run 52 — 2026-08-22 — v6: the restoration validated by ear, after four instruments called it a tie

v6 fine-tune, deliberately option A: the same 28-language census coverage set as v5, same sampling
weights, same hyperparameters, same LoRA config — the restored corpus as the ONLY variable. 82,258
utterances, 288.2 hours, 4,000 steps, 1:24 wall clock.

⚠ TWO THINGS HAD TO BE FIXED BEFORE IT WAS A CONTROLLED COMPARISON AT ALL:

  - `build_webdataset.py` globbed the tokens dir for its language list, so it silently became a
    102-language build the moment the corpus completed — discarding the census-derived greedy cover
    that `sampling_budget.POP_ORDER` encodes (English owns the 53 generalist base letters, Zulu the
    clicks and breathy voice, Hausa the ejectives, Fula the prenasals). Training on all 102 is a
    legitimate but DIFFERENT experiment; it must be chosen, not inherited from an `ls`. Now reads
    POP_ORDER with `--all` as the explicit opt-in.
  - `use_pinyin_ratio` defaulted to 0.3 from `TrainingConfig` and was never declared. Inert today —
    the branch is gated on a `text_pinyin` key our shards do not carry — but a future shard builder
    adding that field would silently substitute into 30% of samples with nothing to flag it. Pinned
    to 0.0.

### Four instruments, no separation

    recognizer distance   67 closer / 56 further      mean Δ -0.00014
    training loss         v5 and v6 both flat ~3.9
    eval loss @4000       3.9658 vs 3.9777            inside v5's own ±0.04 wobble
    pause-match proxy     0.301 vs 0.348              n=6

⚠ AND THE PROXY WAS WORTHLESS, WHICH I NEARLY REPORTED AS A RESULT. The generator is stochastic —
flow matching from a random initial state — and the same model on the same input three times spread
by up to 0.26, against a v5→v6 difference of 0.047. Noise 5× the effect. The tell was v6@2000 scoring
ABOVE v6@4000, which has no mechanism. The determinism control belonged before the numbers, not after.

### The verdict came from listening

On the punctuation-dense en_us pairs, v6@4000 is clearly better on prosody — unambiguous on the ear,
invisible to every metric available. That is the outcome #871/#873 predicted in advance: no QC
movement, the payoff in prosody alone.

⚠ THE GENERAL LESSON, AND IT IS THE ONE WORTH KEEPING. Every automated check said "no difference".
Had the decision rested on them, the 88.2% prosodic share of the restoration would have been written
off and only the segmental 7.7% credited. The metrics are not merely insensitive here, they are blind
by construction: fold() strips pause marks, next-token loss over 8 weighted codebooks barely moves for
a shifted silence, and an energy-threshold detector counts stop closures as phrase boundaries.

Scope: one listener, one language, six utterances, one checkpoint pair. Enough to justify the change,
not to quantify it.

## Run 53 — 2026-09-02 — Phonemizer drift against the alignment DB: 1.17%, and two defects it exposed

**Question:** the DB's `ipa` was last derived 2026-08-22 (Run 51). 322 phonemizer commits have landed
since. How much of the corpus has drifted, how much of that is inside the 28 training languages, and
does a v7 follow?

### The measurement is only clean on one slice, and that is the slice that was used

`read_text_src` partitions the DB three ways and only ONE supports an isolated comparison:

    fleurs_raw  249,430   ipa derived by rederive_read_text.mts from read_text on 08-22   <- comparable
    auto         22,159   ipa came through the byid / phonemize-fleurs path
    hand            209   human-authored; rederive must never touch them

Re-running `rederive_read_text.mts` on the fleurs_raw rows' own `read_text` therefore varies the
ENGINE and nothing else. The `auto` rows were run separately and, reassuringly, came back
byte-identical on cmn_hans_cn (2,871), am_et (2,501), my_mm (2,290) and th_th (1,947) — so the
pipeline confound I had budgeted for is nil, and their drift (660 rows) is real too.

⚠ **The determinism control ran BEFORE the numbers were read**, per Run 52's lesson. Two full
re-derivations of the 12,051 rows in the five worst-moving languages are byte-identical, so every
difference below is engine change rather than sampling noise. 249k rows re-derive in ~9 minutes,
which is why this was measured whole rather than sampled.

### Where it landed

    WHOLE CORPUS       3,170 / 271,589   1.17%
    28 TRAINING LANGS    340 /  77,507   0.44%

    pa_in  81.1%   ne_np 11.1%   sl_si 5.7%   ckb_iq 5.6%   vi_vn 4.0%   mr_in 3.6%
    mi_nz   2.9%   en_us  2.7%   lo_la 2.5%   xh_za 2.0%    zu_za 1.9%   …56 langs in all

The heaviest movers are languages the model is NOT trained on. Inside the training set only four
exceed 1%: vi_vn 4.0, en_us 2.7, xh_za 2.0, zu_za 1.9. Sixteen of the 28 did not move at all.

### ⚠ THE BIGGEST MOVER IS NOT A DEFECT, AND I CALLED IT ONE FIRST

xh/zu numerals now read in English — `kʼutʰˈaːtʰu kʼuɬˈaːnu` → `θɹˈiː pʰɔᶦnt fˈaᶦv`. That reads as an
obvious regression and is not: `numeral_register.mts` is measured corpus policy (zu 95% closer, xh
91%, scored against a phone recognizer over the whole digit-bearing corpus), and #875 extended it to
CLOCK and DECIMAL forms at 21:37 on 08-22 — hours after the snapshot. The engine called directly
still emits the Xhosa numerals; the register is applied to the text before it. Checking the mechanism
before reporting is what separated this from the next section.

### Defect 1 — #1098 traded a half reading for a leak, and a corpus cannot afford that trade

`afda7429` (#1093/#1098, 08-27) made an unreadable rate DECLINE rather than half-read, on the stated
principle that "a half reading is worse than a visible leak". Fleet-wide it took 290 half readings to
29. In the corpus it does this:

    et_ee   160 km/h        kˈilomeːtrit  ->  km
    mt_mt   160km/siegħa    kɪlɔmɛtru     ->  km
    oc_fr   165 km/h        kilumɛtɾes    ->  km
    ig_ng   83km/awa        kilomita      ->  km
    cs_cz   600 Mbit/s      strˈana       ->  s
    vi_vn   160km/h         kˈi˧ lˈo˧ mˈɛ˧˥t̪  ->  ˈʊkm
    cmn     133 m/s         mi˨˩˦         ->  ˈɛm      (米 → the ENGLISH letter name)

128 changed rows carry a rate; **99 rows gained raw Latin text inside their IPA against 19 that lost
some — net +80.** For several languages every single changed row is a rate row (oc_fr 18/18, ast_es
12/12, ig_ng 8/8, jv_id 5/5).

⚠ **THE PRINCIPLE IS RIGHT FOR THE ENGINE AND WRONG FOR THE CORPUS, and that is the whole finding.**
A leak is loud and a half reading is silent — true when a human reads the output. Here nothing reads
it: the leaked `km` goes into the training tensor as literal Latin characters, teaching the model to
voice them. The corpus wants the half reading, and really wants the full one. cmn is worse still: it
has no leak to decline TO, so it voices `ˈɛm` — precisely the failure the commit message documents
avoiding for a Japanese golden (`12.8 km/秒` → `kʰˈeᶦəm`). The guard is at "ASCII-Latin denominator",
and Vietnamese and Chinese with `/h` and `/s` sit on the wrong side of it. Bisected to the commit;
`160 km` and `5km` still expand correctly, so it is the rate form alone.

### Defect 2 — pa_in's 81% is mostly right, on an arbitrary tie-break that is sometimes wrong

`39050bf7` (#898, 08-23) wired the three Punjabi lexicons into the shipped `text()` path. Most of what
follows is a real gain — ਬਹੁਤ `bˈəɦʊt̪` → `bɔː˩˥t` is the correct tonal reading. But
`build-pa-guru-lexicon.mts` stores `gs[0]`, the FIRST wikipron reading, and **34 of 217 entries come
from a word with more than one distinct reading**, so for those the shipped value is scrape order:

    ਵਿੱਚ  ships bɪt͡ʃːɪ̆   alternatives ɪt͡ʃːɪ̆, ʋɪt͡ʃːɪ̆     — "in", among the commonest words in the language
    ਵੱਲ   ships əllɪ̆      alternative  ʋəllɪ̆              — the initial ʋ dropped entirely

Both are high-frequency function words, which is why one lexicon change moved 81% of the language's
rows. Two further oddities in the same file: bare combining marks (`ਂ`, `ੰ`) are keyed as words, and
four letter-name entries carry alternatives containing a literal U+25CC dotted circle.

### What this does NOT support

**Not a v7 on drift alone.** 340 rows in 77,507 is 0.44%, and 41 of en_us's 71 diffs are a single
comma appearing — prosody, of the class Run 52 established every automated metric is blind to and
which needed an ear to adjudicate. Retraining now would also bake in Defect 1, which touches cmn_hans_cn,
vi_vn and cs_cz inside the training set.

Order, unchanged from the Run 51 restructure and from what the memory of this campaign keeps saying:
fix upstream, re-derive, re-QC, then decide on a fine-tune — never the other way round.

## Run 54 — 2026-09-03 — en-GB enters the corpus as a coverage patch, not a language

**Question.** Upstream #1252 moved `en-GB` onto the superscript offglide convention (`əᶷ eᶦ aᶦ aᶷ`),
which is what the model was already trained on for `en`. But the *units* en-GB produces are not the
units en-US produces, and the v6 model has never seen most of them — hence "smoke" → "smik". What is
the smallest amount of en-GB audio that closes that gap?

**Framing.** The instruction was to add "what's needed to handle the diphthongs, triphthongs or
offglide pairings that didn't appear in en-US, and limit to an accent that matches our
phonemization." So this is deliberately *not* an `en_gb` corpus in the sense the other 28
`POP_ORDER` entries are. It is a patch sized to a measured hole. Two consequences follow:

- Only the two **southern** SLR83 archives are in scope. The other five dialects (Irish, Scottish,
  Welsh, northern, midlands) phonemize to units `en-GB` does not emit — Run 53's r-colouring measure
  separated them cleanly (southern 1.51 vs Irish 2.34 rhotic marks per utterance). Training on audio
  whose realised vowels disagree with the IPA beside them is the exact drift this pipeline keeps
  getting bitten by.
- Utterance selection is greedy set-cover over *vowel units*, not a random sample.

**Method.** Phonemized all 8,492 southern transcripts (4,161 female + 4,331 male) through `en-GB`,
extracted vowel units with `[V]ː?[Vᶦᶷᵊ]ː?[Vᶦᶷᵊ]?`, counted the same units across the 28 `POP_ORDER`
languages in the alignment DB, and greedily picked utterances maximising coverage of units sitting
below `sampling_budget.N_TOKENS` (300).

**Result — 1,281 of 8,492 utterances (759 female, 522 male):**

```
  unit      avail  trained   added   total
  əᶷ        5,807      192   1,029   1,221
  ɛə        1,830       37     392     429
  aᶷə         524       33     267     300
  ʊə          447       69     231     300
  aᶦə         424      200     100     300
  iːə         191      178     122     300
  ɔᶦə         142       39     142     181   <- archive exhausted
  əᶷɪ         141        6     141     147   <- archive exhausted
  ɪɒ          116        0     116     116   <- archive exhausted
```

`əᶷ` — GOAT, the vowel in "smoke", the one that started this — goes from 192 occurrences to 1,221.
`ɛə` (SQUARE) from 37 to 429.

**Negative results worth keeping.**

- **The female archive alone was not enough.** A first pass over the 4,161 female utterances left
  `aᶷə` at 292 and `ʊə` at 287 with the archive exhausted — the units simply are not in those
  transcripts. Adding the male archive closed both to exactly 300. The gender balance in the
  selection (759F/522M) is a *consequence* of that, not a target I set; the male archive got picked
  because it carried units the female one had run out of.
- **43 units cannot reach 300 and never will.** They are rare in English itself (<100 occurrences in
  8,492 utterances). `ɪɒ` has 116 available and 0 trained. No amount of data fixes this, and it
  should not be fixed: a model that sees `əᶷɪ` rarely matches a world where `əᶷɪ` *is* rare.
  The stopping point here is a judgement call, not a computed one.
- **`ingest_dir.py` needs the export venv,** not `/mnt/data/omnivoice_ipa/venv` — that one has
  neither `onnxruntime` nor `soundfile`.
- **The encode was silently running on CPU at 0.69x realtime.** `--provider cuda` asks for
  `CUDAExecutionProvider`, but both venvs had plain `onnxruntime`, not `onnxruntime-gpu` — ORT logs a
  warning and falls back to CPU, and the warning was inside the `grep -v Warning` this script's
  output is normally piped through. Measured: **5.14 s/utterance on CPU vs 0.10 s on GPU, 51x**, or
  110 min vs 2 min for this job. `pip install onnxruntime-gpu` (1.29.0) into `export_venv` after
  uninstalling `onnxruntime` — they collide on the module name, and the GPU wheel is a superset that
  still provides the CPU EP. Worth checking `sess.get_providers()[0]` rather than trusting the flag.

**Ingested.** 1,281 rows, 0 skipped, 2.40 h of audio, 759F/522M, in 133 s on GPU. Manifest ids match
the npz keys exactly, every array is `[8, n_frames]` with `n_frames` agreeing with the manifest, and
all 1,281 rows carry `ipa_src="phonemizer"`. A sample row shows the convention landing as intended —
`kəntɹˈəᶷɫd` for "controlled", `lˈəᶷəd` for "lowered", the superscript offglide `əᶷ` the v6 model
has only 192 examples of.

**Caveat carried forward, in the manifest itself.** These rows are marked `ipa_src="phonemizer"`,
not `"hand"` and not `""` — that is IPA *provenance*, and it stays true regardless of QC. The
alignment verdict lives in `status`, which the pass below fills in.

**Open decision for v7.** Adding `en_gb` takes `POP_ORDER` from 28 languages to 29, which makes v7 a
different experiment than v6 rather than a strict improvement on it. That is a deliberate choice to
put to the user before any fine-tune starts.


## Run 55 — 2026-09-03 — Listening to the en-GB ingest: alignment, and rhoticity as an accent gate

**Pass.** `asr_align_dir.py` — a new sibling of `asr_align_corpus.py`, written for the same reason
`ingest_dir.py` is a sibling of `ingest_fleurs.py`: that script is bound to FLEURS in three places
(streams audio out of `train.tar.gz`, reads a FLEURS TSV for sentence id and transcript, looks IPA up
in `byid/<lang>.tsv`) and a directory-ingested corpus has none of them — it has a manifest already
carrying `id`, `text` and `ipa`. Everything that decides what a score *means* is deliberately
identical: same model, same fp16-on-cuda, same batching, same `INSERT OR REPLACE` into the same `utt`
table, same resume-by-default. Two things differ, both forced by the corpus: the audio is **resampled**
rather than skipped (the FLEURS path skips anything not already 16 kHz because an odd rate there means
a broken member; SLR83 is 48 kHz, so skipping would align nothing), and `sentence_id` is the utterance
id (FLEURS repeats a sentence across speakers; a directory corpus has one recording per id).

**1,281 utterances in 22 s at 57/s. All 1,281 manifest rows matched audio, 0 unreadable, 0 short.**

```
lang     n      short  median  within-3MAD    investigate
en_us    2601   0      0.1795  2436 (93.7%)   165
en_gb    1281   0      0.2048  1265 (98.8%)    16
```

**The headline is that 0.2048 is boring**, and it had to be checked rather than assumed. The
recognizer is en-US flavoured: it writes `oʊ` where we write `əᶷ` and `ɚ` where we write `ə`, and
`fold()` drops modifier letters (category Lm) — so our `əᶷ` folds to one phone while its `oʊ` folds to
two. That asymmetry is exactly the shape of Run 36's `ga_ie` bug, where a uniform per-utterance
penalty flattened the distribution until the investigate list came out EMPTY. It did not happen here:
en_gb sits 0.026 from en_us with a 1.2% tail, so the offset is real but too small to hide outliers.

**The 16 flagged rows are category 3, not our bugs.** Spot-checked six: our IPA is correct in all of
them and the recognizer is what failed — one emits "including bog" *before* "brought", another renders
"snowing" as `s t n ɑː v ɛ n ɪ n`. Labelled `verified`/`investigate` by `asr_align_label.py --apply`;
`corpus_filter` keeps `investigate` in (only `defective_audio` is excluded unconditionally), which is
the right posture for an instrument that is roughly 75% accurate — a hint, not a verdict.

### Rhoticity as an accent gate — the useful thing to do with an unreliable instrument

The target accent is non-rhotic southern; a rhotic reader in the archive would teach the model an
alignment our `en-GB` IPA never writes. Asking "did the recognizer hear a rhotic" is useless (it hears
them everywhere, being en-US flavoured). Asking **which speakers hear more rhotics than their own IPA
predicts, relative to each other**, is not. Per speaker, over `[ɹɻrɚɝ]` (Run 53's fix — the plain
`[ɹɻr]` class is blind to r-coloured vowels): excess = (heard − ours) / utterance, scored by 3×MAD.

**57 speakers, median excess 1.35, MAD 0.19. Exactly one outlier, and it is in the safe direction:**

```
  som_04766   18 utts  heard 4.44  ours 2.28  excess 2.17  z +2.96   (highest, still inside)
  som_00610   21 utts  heard 2.67  ours 2.29  excess 0.38  z -3.48   <- most NON-rhotic
```

No speaker rejected. This is the first *evidence* the two archives are non-rhotic rather than merely
*labelled* southern, which is what "limit to an accent that matches our phonemization" actually needed.

**And a negative result that stops this from being over-read.** The ranking sorts almost perfectly by
SEX: every `som_` (male) sits above every `sof_` (female), male excess 1.5–2.2 against female 1.0–1.5,
with no overlap. Real rhoticity would not sort by sex. That is the recognizer substituting `ɚ` for `ə`
at a sex-dependent rate — instrument bias — and it is why the absolute number cannot be thresholded
and only the within-sex spread carries information. A gate built on the raw count would have
"discovered" that all 29 male readers are rhotic.

## Run 56 — 2026-09-03 — v7: en_gb enters the epoch, and a split that would have dropped it

**Question.** v6 trained on 28 languages. Adding `en_gb` makes v7 a different experiment, not a
strict improvement — so what exactly changes, and does the en-GB data actually reach the model?

### POP_ORDER 28 -> 29, and why it perturbs nothing

`POP_ORDER` is not a list of languages, it is a **greedy-cover ownership order**: a primitive is
attributed to the first, biggest-population language that carries it, and each language's
oversampling weight is set so its scarcest OWNED primitive reaches `N_TOKENS`. Inserting a language
can therefore steal ownership and move everyone's weight.

It does not here, and the reason is worth writing down: `phon_of` maps a corpus key to a census key
with `split("_")[0]`, so `en_gb` -> `en` — **the same key as `en_us`**. The sweep gives `en_gb` an
owned set of exactly nothing, a weight of 1.0, and no influence on any other language's weight.

Measured against the v6 weights: **not one existing language's utterance count or weight changed.**

```
v6 28 langs -> v7 29 langs
  NEW  en_gb: n_utts=1281 weight=1.0 owned=0 effective=1235
  changed languages: 0        dropped: none
  utterances 75,902 -> 77,183     en_gb share of the epoch: 1.60%
```

(`effective_utts` sits below `n_utts` at weight 1.0 because `scale = natural_total / weight_sum`
keeps the epoch the same size — weighting redistributes exposure rather than inflating it. Not a bug;
it caught my eye and is recorded so it does not catch the next reader's.)

### The offglide pairings nothing else was checking

The sampling budget only guarantees exposure for **owned census primitives**, and the census is 142
single phones and diacritics — the offglide PAIRINGS this data exists for are not in it. So no gate
in the pipeline verifies the thing the corpus addition was for. Measured directly, per epoch, with
repeat factors applied:

```
  unit          v6      from en_gb      v7
  əᶷ  GOAT      155      +1,029       1,184
  ɛə  SQUARE     36        +392         428
  ʊə  CURE      106        +231         337
  aᶦə / aᶷə  199 / 32  +100 / +267   299 / 299
  ɔᶦə / əᶷɪ / ɪɒ  41 / 3 / 0  +142/+141/+116   183 / 144 / 116
```

`əᶷ` — the vowel behind "smoke" -> "smik" — goes from 155 exposures per epoch to 1,184.

⚠ **`en_gb` is deliberately left at repeat=1.** Giving it 2 would push the last three units over 300,
but those are archive-exhausted: repeating shows the SAME few clips again rather than adding variety,
which trains memorisation of those recordings instead of the pairing. A number that reaches target by
repetition is not the same number.

### The bug that would have wasted the run

`build_lang` splits train/dev by GROUPING on `sentence_id`, because FLEURS records one sentence with
~2.2 speakers and a plain row slice put the same sentence in both splits (73-99% of every dev set,
Run 40). `ingest_dir.py` wrote **`sentence_id=None` for every en_gb row**. All 1,281 collapse into one
group; groups are assigned WHOLE; the first group fills dev — so **every en_gb row went to dev and the
train set came out EMPTY**. The language would have contributed nothing to v7, silently, and the
existing disjointness assert passes happily on an empty train set.

Fixed at the source (`sentence_id` = the utterance id: one recording per id, nothing to repeat — the
same convention `asr_align_dir.py` uses), the existing manifest patched in place, and a guard added:

    assert train_rows, f"{lang}: split produced an EMPTY train set from {n} rows in {g} group(s)"

Verified the guard fires on the pre-fix shape, then rebuilt: **en_gb train 1,243 / dev 38**.

### v7 corpus

```
29 languages · 75,022 train rows · 259.8 h · 1,009 defective-audio utterances excluded
data_config diff vs v6: en_gb added; every other language's repeat copies IDENTICAL
```

**Process note, twice now.** A `nohup ... &` inside a background wrapper detached and died, leaving a
0-byte log while the harness reported exit 0. Caught by checking `data_config.json`'s mtime rather
than the exit code. Earlier the same day a render was reported as "starting" without being launched.
An exit code is not evidence that work happened; the artifact's timestamp is.
