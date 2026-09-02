# Reference voices

The browser demo offers all 193 languages the phonemizer routes. Generation is **always
voice-cloned** — without a reference, input under ~5 s falls outside the fine-tune's distribution and
can emit noise rather than degrade — so every language needs a reference clip. A language without a
native one is read by a **donor** voice from a near neighbour, which is audible: cloning copies the
speaker's accent along with their timbre.

**165 of 193 languages have a native voice; 28 are on a donor.** This document records where every
one of the 28 stands, what a usable clip is, and what the tooling can do unattended.

## Done — 165 native voices from eight sources

| source | licence | languages |
|---|---|---|
| FLEURS (the fine-tune corpus) | CC-BY-4.0 | 102 |
| Common Voice 22.0 (HuggingFace mirror) | CC0 | 26 |
| Common Voice 26.0 (Mozilla Data Collective) | CC0 | 4 native + 3 donor upgrades |
| Omnilingual ASR corpus | CC-BY-4.0 | 12 |
| Vaani (ARTPARK-IISc) | CC-BY-4.0 | 6 |
| WaxalNLP · OpenBibleTTS · Ravnursson · qirimtatar-tts · WenetSpeech-Wu | CC-BY / BY-SA / Apache | 9 |
| OpenSLR 83 / 158 / 44 | CC-BY-SA-4.0 | 3 — British English, Tibetan, Sundanese |
| LibriVox | public domain | 2 — Latin, Ancient Greek |

Four of these closed gaps this document once called unsolvable. **European Portuguese**: the
`accents` column is nearly empty for `pt`, but `variant` separates the standards and holds readable
labels ("Portuguese (Portugal)"), not codes. **Western Punjabi**: Omnilingual has `pnb_Arab`, in
Shahmukhi. **Sylheti** and **Rangpuri**: both in the main Vaani repo, the second under its other
name, Rajbanshi.

⚠ **Akan was missed for weeks** because the gap analysis matched demo codes literally against corpus
locales, and Common Voice files Twi under `tw`. Check a corpus's own naming before concluding a
language is absent from it.

⚠ **Ancient Greek's reader uses MODERN pronunciation** — the decode shows delta as [v] and upsilon as
[i] — while the phonemizer emits reconstructed Attic. The words match, the pronunciation school does
not. Still closer than the modern Greek news sentence it replaced, but it is the first voice to
replace if a reconstructed-pronunciation recording appears.

### Long clips are cut, but only where the cut is provable

Omnilingual's spontaneous speech runs 25-80 s, and a reference that long lengthens every sentence the
demo later speaks in that language. `tools/trim-to-sentences.mjs` cuts at a pause and keeps the
sentences before it, requiring BOTH that the transcript's sentence count matches the count of speech
runs AND that the result falls in the speaking-rate band measured across the shipped references
(6.4-13.7 IPA characters per second). Equal counts alone are not proof: one Hawaiian clip paired 8
words with 13.2 s of audio. Gulf Arabic found no provable cut and keeps a 22 s reference.

## The 28 still on a donor

Grouped by why, because "no voice yet" means four different things and only one of them is worth
your time.

### 1. Reachable with one manual step (2)

| language | donor | source | what is needed |
|---|---|---|---|
| Karakalpak | Uzbek | Karakalpak Speech Corpus, Mendeley (CC-BY-4.0, 50 h) | a transcript file that matches the audio — see below |
| Kirundi | Kinyarwanda | `DigitalUmuganda/Afrivoice_V2` (CC-BY-4.0, 509 h) | its clips are 15-22 s of image description with no sentence-aligned pauses, so none yields a provable cut; usable only if an ~18 s reference is acceptable |

⚠ **The Karakalpak archive on hand cannot be used.** `DATASET_version2.7z` holds 2,022 clips numbered
`sentence_kaa1000xxxx` while its `train.csv` indexes 17,238 rows numbered `sentence_kaa0000xxxx`,
with **zero** overlap. Every possible offset between the two numbering schemes was tested: the best
correlation between clip duration and transcript length was 0.20, where a true alignment gives above
0.8 — and random pairing scores the same. It is audio from one version with an index from another.

### 2. Accent variants where the donor reads the same written language (4)

**Egyptian Arabic** ← MSA read by an Egyptian speaker (FLEURS `ar_eg`, so arguably already right),
**South Levantine** ← North Levantine, **Standard Malay** ← Malay, **Kirundi** ← Kinyarwanda. None is
wrong today; each would still improve with a native clip.

### 3. Licence-blocked — the data exists and cannot be used (3)

**Xiang** and **Gan** Chinese: MagicData's Changsha and Nanchang corpora are CC BY-**NC-ND**.
**Hakka**: `formospeech` is gated and the Taiwanese government set is under TRAIL, a RAIL-style
licence with behavioural use restrictions. One licence negotiation away, not one dataset away.

### 4. Nothing acceptable found anywhere (19)

Bambara · Bavarian · Bishnupriya · Cherokee · Greenlandic · Hmong · Ilocano · Jin Chinese · K'iche' ·
Lule Sami · Madurese · Min Dong Chinese · Mossi · Nama · Nogai · Papiamentu · Scottish Gaelic ·
Shan · Zhuang

Common Voice 26 has **none** of them, so that avenue is closed rather than unexplored. Notes on the
near misses, so the search is not repeated:

- **Greenlandic** is the sharpest case: acceptable CC-BY-3.0 audio exists on Wikimedia Commons and no
  transcript for it exists anywhere. A native speaker writing out 40 seconds would close it.
- **Scottish Gaelic** and **Bavarian** have Commons clips with real transcripts — a Wikitongues
  interview with Gaelic subtitles, two public-domain spoken Wikipedia articles in Bavarian. Usable,
  but they need hand-cutting, and the Gaelic clips total 240 s.
- **Cherokee**: the community corpus is CC-BY-**NC** except for one CC0 directory, and that directory
  is a tone wordlist.
- **Nama, Shan, Madurese, Ilocano, Zhuang, Bishnupriya** surface only under `mms_ulab_v2`, which is
  CC BY-NC-SA *and* untranscribed.

Three of these keep a donor that is at least a relative rather than a stranger, courtesy of CV 26:
**Mossi** reads with a Dagbani voice, **Greenlandic** with Iñupiaq, **Totontepec Mixe** with
Copainalá Zoque. ⚠ None of those three donor languages has a phonemizer, so `--phon-lang` renders the
reference transcript through the TARGET language's engine, which shares a compatible Latin
orthography. That IPA approximates what the speaker says — a deliberate trade against Fula reading
Mossi and Danish reading Greenlandic.

## What a usable clip is

| | |
|---|---|
| format | WAV, 24 kHz (mono or stereo; 16-bit PCM or 32-bit float) |
| length | 5-12 s of continuous speech, one speaker |
| content | ordinary connected speech, not a word list; no music, no room echo |
| transcript | **required**, exact, in the language's normal orthography |
| licence | must permit redistribution — see below for what is actually redistributed |

The transcript matters as much as the audio: it is phonemized and fed to the model alongside the
codes, and one that does not match what is said degrades the clone.

**What ships is codec output, not audio.** A voice is a few KB of Higgs codec codes; the source WAV
and the 654 MB encoder stay out of the repo. Those codes come from the Higgs codec, which
`k2-fsa/OmniVoice` ships **byte-identical** to
[`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) —
sha256 `fe7c5e8785e0a058…` on both — so they carry Boson's licence rather than the CC-BY-NC term on
the OmniVoice transformer weights. The obligation that DOES follow a voice is whatever its source
audio carried: CC0 asks nothing, CC-BY asks attribution (the `source` block in `voices.jsonc` records
dataset, file and split for every voice), CC-BY-SA asks share-alike.

## Tooling

| tool | for |
|---|---|
| `make-voice-from-commonvoice.mjs` | Common Voice split archives, with `--accent` / `--variant` filters and `--prefix-mb` range fetching |
| `make-voice-from-hf.mjs` | any ungated HuggingFace dataset, through the rows API — a voice costs a few MB rather than the corpus |
| `make-voice-from-openslr.mjs` | an index-plus-audio archive, or any prepared directory via `--dir`, with `--phon-lang` for donor languages |
| `parquet-clips.py` | gated or viewer-disabled repos, reading a shard directly with the CLI token |
| `trim-to-sentences.mjs` | cutting an over-long clip at a provable sentence boundary |
| `find-english-intro.py` | locating where a LibriVox English announcement ends, and checking the clip against its printed text |
| `merge-cv-voices.mjs` | folding any of the above into `voices.jsonc` + `voice-codes.json` |
| `preview-langs.mjs` | one generated clip per language, through the real page, for the listening test |

⚠ Selection is measured, never assumed: every candidate is scored for noise floor, speech fraction
and clipping, and the transcript must be in the language's declared script. **Clipping is measured as
consecutive samples pinned at full scale, not as a peak threshold** — a peak test rejects
*normalised* audio too, which silently discarded the entire Vaani corpus until it was fixed. None of
this can hear a bad read, so the last step is always a listening test.

## Access notes

- **Mozilla Data Collective** (Common Voice 23.0+) needs terms accepted **per dataset in a browser**.
  There is no API route for it, both the full download and the sample refuse without it, and the site
  rate-limits acceptance to roughly one per minute. Slugs are name-derived with a hash suffix, so
  look them up rather than guessing: `GET /api/datasets/{id}` returns name, locale, licence and size.
- **Vaani**: use `ARTPARK-IISc/Vaani` (133 languages), not `Vaani-transcription-part` (64, a subset
  covering nothing the main repo lacks). Transcripts are sparse and unevenly spread — Magahi needed
  three shards before one had any in the length window.
- **HuggingFace gates**: an accepted gate shows as `resolve` returning 200; a repo can be listable
  and still refuse file access ("not in the authorized list").

## Also missing: example text

Six languages have no bundled sample sentence, because neither FLEURS nor the phonemizer's mined
corpora yielded one that passed the filters (single sentence, the language's own script, no digits or
markup): **K'iche', Kalaallisut, Lule Sami, Nama, Nogai, Totontepec Mixe**. The box is empty for those
and the placeholder says so; Greenlandic cannot even be previewed without one. A couple of ordinary
sentences each would close it.
