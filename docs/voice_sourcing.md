# Reference voices still to source

The browser demo offers all 193 languages the phonemizer routes. Generation is **always
voice-cloned** — without a reference, input under ~5 s falls outside the fine-tune's distribution and
can emit noise rather than degrade — so every language needs a reference clip. 102 have a native one
from FLEURS. The other 91 are read by a **donor** voice from a near neighbour, which is audible:
cloning copies the speaker's accent along with their timbre.

This is the list of what a native clip would fix, worst first.

## Done — 151 of 193 languages now have a native voice

**Common Voice 22.0 (CC0), 25 languages:** Abkhaz, Akan, Albanian, Bashkir, Basque, Chuvash,
Classical Nahuatl, Guaraní, Haitian Creole, Kinyarwanda, Kurmanji, Latgalian, Min Nan, European
Portuguese, Quechua, Santali, Saraiki, Sesotho, Setswana, Castilian Spanish, Tashelhit, Tatar,
Tigrinya, Turkmen, Uyghur, Western Armenian.

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

## The remaining 42

`tools/make-voice-from-commonvoice.mjs` automates the whole path — download, score, phonemize,
encode, write both files:

    node tools/make-voice-from-commonvoice.mjs --cv eu --lang eu --n 3 --write

⚠ SELECTION IS MEASURED, NOT TRUSTED. Common Voice is crowd-recorded, and its up/down votes only say
the reading matches the sentence, not that the recording is clean. Metadata narrows the field (8-9 s,
two or more up-votes, no down-votes) and then the AUDIO decides: noise floor, speech fraction and
peak are measured on every candidate, clipped and mostly-silent clips are rejected outright. A noisy
reference is cloned faithfully — its noise comes out in every sentence the demo ever speaks — so the
last step is always a listening test. `--clip a.mp3,b.mp3` forces named clips, which is what makes a
shipped voice reproducible.

Common Voice is now exhausted for this purpose: every locale it shares with the demo's gaps has been
tried. The remaining 67 languages need audio from somewhere else, and the tiers below say which of
them matter most.

## Tier A — donor is a stranger (33)

Different family or markedly different phonology. These are where a native clip changes the result
most.

~~Abkhaz ← Georgian~~ (done) · Akan ← Yoruba · Bambara ← Fula · Basque ← Spanish · Cherokee ← English ·
Classical Nahuatl ← Spanish · Ewe ← Yoruba · Hawaiian ← Māori · Hiligaynon ← Cebuano ·
Hmong ← Lao · Ilocano ← Cebuano · K'iche' ← Spanish · Kalaallisut ← Danish · Kikuyu ← Kamba ·
Kinyarwanda ← Luganda · Kirundi ← Luganda · Lule Sami ← Finnish · Madurese ← Javanese ·
Malagasy ← Swahili · Mossi ← Fula · Nama ← Xhosa · Nigerian Pidgin ← Yoruba · Papiamentu ← Spanish ·
Quechua ← Spanish · Santali ← Bengali · Sesotho ← Sepedi · Setswana ← Sepedi · Sinhala ← Tamil ·
Sundanese ← Javanese · Tashelhit ← Arabic · Tibetan ← Burmese · Totontepec Mixe ← Spanish ·
Zhuang ← Thai

⚠ Nama is the one to hear first: it has four click types, and no donor in the set has clicks except
Xhosa, which has three different ones.

## Tier B — related, but a different language (28)

Albanian ← Macedonian · Aragonese ← Spanish · Aromanian ← Romanian · Balochi ← Pashto ·
Bashkir ← Kazakh · Bavarian ← German · Chuvash ← Kazakh · Crimean Tatar ← Turkish ·
Faroese ← Icelandic · Gan ← Mandarin · Guaraní ← Spanish · Haitian Creole ← French ·
Hakka ← Cantonese · Jin ← Mandarin · Karakalpak ← Uzbek · Kurmanji ← Central Kurdish ·
Latgalian ← Latvian · Min Dong ← Cantonese · Min Nan ← Cantonese · Nogai ← Kazakh ·
Scottish Gaelic ← Irish · Shan ← Burmese · Tatar ← Kazakh · Tigrinya ← Amharic ·
Turkmen ← Azerbaijani · Uyghur ← Uzbek · Wu ← Mandarin · Xiang ← Mandarin

## Tier C — same written language, different standard or accent (30)

The donor reads the same orthography; what is wrong is the accent. Cheapest to source and the most
visible to a listener who speaks it.

- **English (British)** and **English (Indian)** ← General American.
- **Spanish (Castilian)** ← Latin American, and **Portuguese (European)** ← Brazilian. FLEURS ships
  `es_419` and `pt_br`, so those two are filed as the native voices for the *varieties they are*;
  the European standards are the ones borrowing.
- **French (Québécois)** ← France French.
- The nine **Arabic dialects** ← FLEURS `ar_eg`, which is MSA read by an Egyptian speaker. Egyptian
  Arabic is therefore well served; Moroccan, Levantine, Iraqi, Gulf, Hijazi, Sudanese and Libyan
  are not.
- **Western Armenian** ← Eastern Armenian (the consonant shift makes this more than an accent).
- The Hindi-belt group — **Bhojpuri, Haryanvi, Chhattisgarhi, Magahi, Maithili, Awadhi** ← Hindi —
  and **Bishnupriya, Rangpuri, Sylheti** ← Bengali.
- **Western Punjabi** ← Punjabi, **Saraiki** ← Sindhi, **Southern Pashto** ← Pashto,
  **Standard Malay** ← Malay.
- **Latin** ← Italian and **Ancient Greek** ← Modern Greek, where there is no native speaker to find
  and the donor is the convention.

## Also missing: example text

Six languages have no bundled sample sentence, because neither FLEURS nor the phonemizer's mined
corpora yielded one that passed the filters (single sentence, the language's own script, no digits
or markup): **K'iche', Kalaallisut, Lule Sami, Nama, Nogai, Totontepec Mixe**. The box is empty for
those and the placeholder says so. One or two ordinary sentences each would close it.
