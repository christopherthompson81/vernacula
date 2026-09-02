# Reference voices still to source

The browser demo offers all 193 languages the phonemizer routes. Generation is **always
voice-cloned** — without a reference, input under ~5 s falls outside the fine-tune's distribution and
can emit noise rather than degrade — so every language needs a reference clip. 102 have a native one
from FLEURS. The other 91 are read by a **donor** voice from a near neighbour, which is audible:
cloning copies the speaker's accent along with their timbre.

This is the list of what a native clip would fix, worst first.

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

## Tier A — donor is a stranger (33)

Different family or markedly different phonology. These are where a native clip changes the result
most.

Abkhaz ← Georgian · Akan ← Yoruba · Bambara ← Fula · Basque ← Spanish · Cherokee ← English ·
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
