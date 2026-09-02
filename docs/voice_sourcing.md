# Reference voices still to source

The browser demo offers all 193 languages the phonemizer routes. Generation is **always
voice-cloned** — without a reference, input under ~5 s falls outside the fine-tune's distribution and
can emit noise rather than degrade — so every language needs a reference clip. 102 have a native one
from FLEURS. The other 91 are read by a **donor** voice from a near neighbour, which is audible:
cloning copies the speaker's accent along with their timbre.

This is the list of what a native clip would fix, worst first.

## Done

| language | source | voices |
|---|---|---|
| **Abkhaz** `ab` | Common Voice 22.0, CC0 | 3 (female teens, female thirties, male twenties) |

Abkhaz was read by a Georgian voice, which is a different family entirely. It now has three native
references; the demo defaults to the first and the picker offers the other two.

## What a usable clip is

| | |
|---|---|
| format | WAV, 24 kHz (mono or stereo; 16-bit PCM or 32-bit float) |
| length | ~8 s of continuous speech, one speaker — the corpus references are all 8 s |
| content | ordinary connected speech, not a word list; no music, no room echo, no clipping |
| transcript | **required**, exact, in the language's normal orthography |
| licence | must be redistributable — the codes ship in the demo |

The transcript matters as much as the audio: it is phonemized and fed to the model alongside the
codes, and a transcript that does not match what is said degrades the clone. Encode with:

    node tools/make-voices.mjs <higgs_encoder.onnx> <ref.wav> <ref-ipa.txt> <id> <label>

Only the codes (a few KB) ship; the audio does not, and neither does the 654 MB encoder.

## Common Voice covers 26 of the 90 remaining

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

Available there: Albanian, Aromanian, Bashkir, Basque, Chuvash (`cv`), Classical Nahuatl (`nhi`),
Guaraní, Haitian Creole, Kinyarwanda, Kurmanji, Latgalian, Min Nan (`nan-tw`), European Portuguese,
Quechua (`quy`), Santali, Saraiki, Sesotho, Setswana, Castilian Spanish, Tashelhit (`zgh`), Tatar,
Tigrinya, Turkmen, Uyghur, Western Armenian (`hy-AM`), Western Punjabi (`pa-IN`).

The other 64 need audio from somewhere else — that is where sourcing help is worth most, and the
tiers below say which of them matter most.

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
