# Reference voices still to source

The browser demo offers all 193 languages the phonemizer routes. Generation is **always
voice-cloned** — without a reference, input under ~5 s falls outside the fine-tune's distribution and
can emit noise rather than degrade — so every language needs a reference clip. A language without a
native one is read by a **donor** voice from a near neighbour, which is audible: cloning copies the
speaker's accent along with their timbre.

**165 of 193 languages now have a native voice; 28 are still on a donor.** This document records
where each of the 41 stands, and what a usable clip has to be.

## Done — 165 of 193 languages have a native voice

**Common Voice 22.0 (CC0), 26 languages:** Abkhaz, Akan, Albanian, Bashkir, Basque, Chuvash,
Classical Nahuatl, Guaraní, Haitian Creole, Kinyarwanda, Kurmanji, Latgalian, Min Nan, European
Portuguese, Quechua, Santali, Saraiki, Sesotho, Setswana, Castilian Spanish, Tashelhit, Tatar,
Tigrinya, Turkmen, Uyghur, Western Armenian, and **Indian English** (the `en` locale's
accent-labelled rows).

**Beyond Common Voice, 21 more:**

| source | licence | languages |
|---|---|---|
| Omnilingual ASR corpus | CC BY 4.0 | Moroccan, N. Levantine, Iraqi, Hijazi, Sudanese, Libyan and Gulf Arabic; Bhojpuri, Sinhala, Hawaiian, Southern Pashto, **Western Punjabi (Shahmukhi)** |
| WaxalNLP | CC BY / BY-SA 4.0 | Ewe, Kikuyu, Nigerian Pidgin, Malagasy |
| OpenBibleTTS | CC BY-SA 4.0 | Hiligaynon, Chhattisgarhi |
| OpenSLR 83 / 158 / 44 | CC BY-SA 4.0 | British English, Tibetan, Sundanese |
| Ravnursson | CC BY 4.0 | Faroese |
| qirimtatar-tts | Apache-2.0 | Crimean Tatar |

Two of these close gaps this document previously called unsolvable. **European Portuguese**: the
`accents` column is nearly empty for `pt`, but the `variant` column separates the standards and
holds readable labels ("Portuguese (Portugal)"), not codes. **Western Punjabi**: Omnilingual has
`pnb_Arab`, in Shahmukhi.

⚠ **Long clips are cut, but only where the cut is provable.** Omnilingual's spontaneous speech runs
25-80 s and a reference that long slows every later generation in that language.
`tools/trim-to-sentences.mjs` cuts at a pause and keeps the sentences before it, requiring BOTH that
the transcript's sentence count matches the count of speech runs AND that the result falls in the
speaking-rate band measured across the shipped references (6.4-13.7 IPA characters per second).
Equal counts alone are not proof: one Hawaiian clip paired 8 words with 13.2 s of audio. Gulf Arabic
found no provable cut and keeps a 22 s reference.

## The 41 still on a donor voice

Grouped by why, because "no voice yet" means four different things and only two of them are worth
your time.

### 1. Reachable now — a source exists and needs an account, a manual download, or an hour's work (9)

| language | current donor | source | blocker |
|---|---|---|---|
| Awadhi, Magahi, Maithili, Haryanvi | Hindi | Vaani (CC BY 4.0), 6.6 s mean clips; Haryanvi also `bridgeconn/snow-mountain` (CC BY-SA 4.0, single studio speaker) | HuggingFace auto-approve gate — needs an account click |
| Kirundi | Luganda | `DigitalUmuganda/Afrivoice_V2` (CC BY 4.0, 509 h) | same gate |
| Karakalpak | Uzbek | Karakalpak Speech Corpus, Mendeley (CC BY 4.0, 50 h, 25 speakers) | Mendeley's API would not serve the file list anonymously; a browser download works |
| Wu Chinese | Mandarin | `ASLP-lab/WenetSpeech-Wu-Bench` (Apache-2.0, 9.75 h) | the dataset exposes transcripts but no audio column through the rows API; needs a parquet download |
| Latin | Italian | LibriVox (public domain) — Caesar, Vergil, the Vulgate | whole-book recordings; needs a passage cut and its text matched by hand |
| Ancient Greek | Greek | LibriVox (public domain), 45 items | same, plus a choice between reconstructed and modern pronunciation |

### 2. Accent variants where the donor is the same written language (5)

Cheap to improve but not wrong today: **Québécois French** ← France, **Standard Malay** ← Malay,
**Egyptian Arabic** ← the FLEURS `ar_eg` speaker (already Egyptian, so this one is arguably done),
**South Levantine Arabic** ← MSA (Omnilingual has North Levantine but not South), **Aragonese** ←
Latin-American Spanish, though Common Voice 26 has 16.9 h of Aragonese proper.

### 3. Common Voice 26 — done

Sourced through Mozilla Data Collective, which is where Common Voice went after 22.0. Everything
there is CC0, but **terms must be accepted per dataset on the website** — there is no API route for
it, and the site rate-limits acceptance to roughly one per minute.

Native from CV 26: **Aromanian**, **Aragonese**, **Québécois French**, **Balochi** (`bgp`, Eastern
Balochi — a variety of the same language).

⚠ **Donor voices whose speaker's language the phonemizer cannot read.** Copainalá Zoque now reads
for Totontepec Mixe, Dagbani for Mossi, and Iñupiaq for Greenlandic — replacing Spanish, Fula and
Danish respectively. None of the three has an engine, so the reference transcript is rendered
through the TARGET language's engine (Zoque through Mixe, Dagbani through Mossi, Iñupiaq through
Kalaallisut), all of which share a compatible Latin orthography. That IPA approximates what the
speaker says. It is a deliberate trade against a donor from an unrelated family.

### 4. Licence-blocked — the data exists and cannot be used (3)

**Xiang** and **Gan** Chinese: MagicData's Changsha and Nanchang corpora are CC BY-NC-ND.
**Hakka**: `formospeech` is gated and the Taiwanese government set is under TRAIL, a RAIL-style
licence with behavioural use restrictions. These are one licence negotiation away, not one dataset
away.

### 5. Nothing acceptable found anywhere (22)

Bambara · Bavarian · Bishnupriya · Balochi · Cherokee · Greenlandic · Hmong · Ilocano · K'iche' ·
Lule Sami · Madurese · Min Dong Chinese · Mossi · Nama · Nogai · Papiamentu · Rangpuri ·
Scottish Gaelic · Shan · Sylheti · Totontepec Mixe · Zhuang

Notes on the near misses, so the search is not repeated:

- **Greenlandic** is the frustrating one: acceptable CC BY 3.0 audio exists on Wikimedia Commons, but
  no transcript exists for it anywhere. A native speaker writing out 40 seconds would close it.
- **Scottish Gaelic** and **Bavarian** have Wikimedia Commons clips with real transcripts (a
  Wikitongues interview with Gaelic subtitles; two public-domain spoken Wikipedia articles in
  Bavarian). Both are usable but need hand-cutting, and the Gaelic clips are only 64 and 176 s
  total.
- **Cherokee**: the community corpus is CC BY-NC except for one CC0 directory, and that directory is
  a tone wordlist.
- **Nama, Shan, Madurese, Ilocano, Sylheti, Bishnupriya, Zhuang** surface only under `mms_ulab_v2`,
  which is CC BY-NC-SA *and* untranscribed.
- **Ilocano**'s 448 h `sapinsapin/pld` is research-only, and half of it is wordlists.

## What a usable clip is

| | |
|---|---|
| format | WAV, 24 kHz (mono or stereo; 16-bit PCM or 32-bit float) |
| length | 5-12 s of continuous speech, one speaker |
| content | ordinary connected speech, not a word list; no music, no room echo, no clipping |
| transcript | **required**, exact, in the language's normal orthography |
| licence | must be redistributable — the codes ship in the demo |

The transcript matters as much as the audio: it is phonemized and fed to the model alongside the
codes, and one that does not match what is said degrades the clone. The tools are
`tools/make-voice-from-commonvoice.mjs`, `tools/make-voice-from-hf.mjs` (any ungated HuggingFace
dataset, via the rows API) and `tools/make-voice-from-openslr.mjs`; each writes `voices.json` in its
work directory, which `tools/merge-cv-voices.mjs` folds into the shipped files. Only the codes (a
few KB) are published — never the audio, and never the 654 MB encoder.

⚠ Selection is measured, never assumed: every candidate is scored for noise floor, speech fraction
and peak, clipped and half-silent clips are rejected outright, and the transcript must be in the
language's declared script. None of that can hear a bad read, so the last step is always a listening
test.

## Also missing: example text

Six languages have no bundled sample sentence, because neither FLEURS nor the phonemizer's mined
corpora yielded one that passed the filters (single sentence, the language's own script, no digits
or markup): **K'iche', Kalaallisut, Lule Sami, Nama, Nogai, Totontepec Mixe**. The box is empty for
those and the placeholder says so. One or two ordinary sentences each would close it.
