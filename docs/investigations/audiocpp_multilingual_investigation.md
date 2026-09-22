# Supplying phonemes for every language Kokoro speaks

The supplied-phoneme path shipped English-only. The stated reason:

> The phonemizer covers every language this package speaks, but KokoroFormat is an English render
> target: the other five would need their own mapping onto Kokoro's alphabet, and a symbol we got
> wrong is a refused paragraph rather than an accent.

That was a guess dressed as a constraint, and it was wrong twice over — the render targets already
existed, and the cost of being wrong was not what the sentence claims. This log is the correction,
and a second thing that fell out of it.

## Run 1 — 2026-09-21 — how much of the phonemizer's non-English IPA is even sayable?

Question: if we supplied a non-English stream, how much of it lands inside Kokoro's 114-symbol
vocabulary? Measured against the phonemizer's own goldens, 200 rows per language — no rendering,
just raw canonical IPA against the vocabulary:

```
lang     rows  symbols  in vocab  coverage   missing (count)
en        200    26379     24159     91.6%   'ᶦ'×700, 'ɫ'×379, 'ᶷ'×355, '̬'×268, '͡'×216
es        200    30684     30521     99.5%   '͡'×72, 'ᶷ'×50, 'ᶦ'×41
fr        200    22055     22055    100.0%
it        200    33247     32497     97.7%   '͡'×750
hi        200    36367     33556     92.3%   '̪'×1260, 'ɦ'×723, '͡'×521, 'ʱ'×300
pt-BR     200    25635     24226     94.5%   '͡'×510, 'ẽ'×485, 'õ'×183, 'ĩ'×137, 'ũ'×94
```

**The gaps are notation, not inventory.** The tie bar is a combining mark the English target
already drops; `ᶦ`/`ᶷ` are this repo's superscript off-glide convention; and Portuguese writes a
nasal vowel precomposed where Kokoro carries the base vowel plus the combining tilde (U+0303,
token 17) — the same sound spelled two ways.

⚠ **And the Hindi row here is wrong, which is the point of Run 2.** Reading this table I concluded
Hindi was unsafe: the dental diacritic and breathy voice are contrastive, so dropping them merges
words. That conclusion came from modelling the transform in a throwaway script instead of running
the real one.

## Run 2 — 2026-09-21 — the render targets already existed

`KokoroFormat.Render(string ipa, string lang)` — an overload with an arm for `es`, `fr`, `it`,
`pt-BR`, `hi`, `cmn` and `ja`, carrying a `DecomposeUnknown` step for exactly the Portuguese
nasal-vowel case. Its own doc says why nobody noticed:

> ⚠ NOTHING IN THE APP CALLS THIS OVERLOAD YET … the audio.cpp one phonemizes internally with
> eSpeak-ng and accepts no phonemes at all — so this is the render target for a phoneme-input path
> that does not exist on this side of the ABI yet.

That path exists now. Measured through the REAL renderer, same goldens:

```
lang     rows  symbols  coverage   what is left
es        200    30461   100.0%   nothing
fr        200    22050   100.0%   nothing
it        200    31964   100.0%   nothing
pt-BR     200    25512   100.0%   nothing
hi        200    34058   100.0%   nothing
ja        200    25284   100.0%   nothing
cmn       200    26636   100.0%   nothing
```

**All seven, entirely inside the vocabulary.** Hindi included, because the arm maps rather than
drops: `ɦ→h` (Hindi's /h/ is realised breathy anyway, so no contrast), `ʱ→ʰ` (breathy voice only
occurs on voiced stops, so `bʱ→bʰ` stays distinct from `pʰ`), and the dental mark dropped — which
loses nothing, since Kokoro carries `t` and `ʈ` separately and the dental/retroflex contrast
survives as `t` against `ʈ`.

**The lesson is the one from the `aluminium` round, again: measure the real code, not a model of
it.** My script reproduced three transforms and missed the per-language rules, and it produced a
confident wrong answer about a whole language.

## Run 3 — 2026-09-21 — the readback, because coverage is not correctness

Kokoro's non-English voices were trained on misaki's eSpeak output. Ours is an independent
transcription, so landing in the vocabulary says nothing about whether the model has heard this
transcription of these words. Same check as English: render both ways, read both back.

```
es  El puerto estaba tranquilo esta mañana.        both: El puerto estaba tranquilo esta mañana.
es  Los científicos anunciaron el invento.         both: Los científicos anunciaron el invento.
fr  Le port était calme ce matin.                  both: Le port était calme ce matin.
fr  Les scientifiques ont annoncé leur invention.  both: Les scientifiques ont annoncé leur invention.
it  Il porto era tranquillo questa mattina.        both: Il porto era tranquillo questa mattina.
it  Gli scienziati hanno annunciato l'invenzione.  both: Gli scienziati hanno annunciato l'invenzione.
pt  O porto estava tranquilo esta manhã.           both: O porto estava tranquilo esta manhã.
pt  Os cientistas anunciaram a invenção.           both: Os cientistas anunciaram a invenção.
```

**Eight of eight identical**, accents and punctuation included. Group counts match word counts in
seven of eight; the exception is `l'invenzione`, where our phonemizer separates the clitic into two
groups against one written word — which the existing guard catches, falling that segment back to a
proportional spread rather than misaligning it.

Hindi is weaker evidence: Parakeet is not a Hindi recogniser and transliterates
(`Somvarko, vegyaniku nighoshnaki.` against `Somvarko vegyaniko nighosnaki.`). The two readings
agree with each other, which is what was being asked, but this is not the same standard as the
eight above.

## Run 4 — 2026-09-21 — the thing that fell out: thirteen voices that did not work now do

`AudioCppKokoroVoices` shipped 41 of the family's 54 voices, and said why:

> the five Japanese voices — "Kokoro UniDic resources are not bundled in this GGUF";
> the eight Chinese voices — "Kokoro vocab is missing phoneme symbol: H".

⚠ **BOTH REFUSALS ARE ABOUT THE ENGINE'S OWN G2P, AND THAT IS NO LONGER WHAT RUNS.** Every voice
embedding was in the package the whole time — the extracted sidecars hold 54 of 54, `jf_alpha.bin`
through `zm_yunyang.bin`. Only the grapheme-to-phoneme resources were missing. Supplying the
phonemes means no built-in G2P runs at all, so the refusal has nothing to refuse.

Driven through the ABI:

```
jf_alpha    built-in G2P : REFUSED: Kokoro UniDic resources are not bundled in this GGUF
            supplied     : OK 2.38s peak=0.371 groups=2
jm_kumo     built-in G2P : REFUSED: (same)
            supplied     : OK 3.08s peak=0.765 groups=2
zf_xiaobei  built-in G2P : OK 2.98s peak=0.385 groups=0
            supplied     : OK 3.05s peak=0.376 groups=6
zm_yunxi    built-in G2P : OK 3.23s peak=0.535 groups=0
            supplied     : OK 3.30s peak=0.494 groups=10
```

**Five Japanese voices go from unusable to usable.** And the Chinese half of that table was simply
stale — the engine defect it recorded was fixed upstream somewhere in the twenty-odd commits this
repo skipped, so those eight already worked on the text path; what supplying adds there is
alignment (6 and 10 groups against 0).

⚠ **NOT VERIFIED FOR CORRECTNESS, AND THAT GAP IS REAL.** There is no Japanese or Mandarin
recogniser here, so "renders 2.38 s of audio at a plausible peak" is the whole claim for those two.
The European four have a readback; CJK has durations and the absence of a refusal. That is enough
to offer the voices, since the alternative for Japanese is a hard failure, and not enough to say
they are pronounced well.

⚠ **CJK ALIGNMENT IS COARSE, AND "BY CONSTRUCTION" WAS WRONG — see Run 5.** The first version of
this note said the group→word map cannot mean anything for a language without spaces. It can; the
segmentation exists and this repo discards it.

## What changed

- `KokoroFormat.CanRender(lang)` — the list of languages a stream can be supplied for, which is
  what the backend now gates on instead of `lang is "en" or "en-GB"`.
- `KokoroPhonemizer.Phonemize(text, lang)` — the language-taking path the English-only `british`
  flag could not express.
- `AudioCppKokoroVoices` lists all 54 voices, with `j → ja` / `z → zh` for the engine and
  `j → ja` / `z → cmn` for the phonemizer. Those two differ, which is why the tables are separate.
- The tests that pinned the old exclusion now pin the opposite, including that a render target must
  exist for any voice offered — otherwise a voice is listed only to fail the way it used to.

## Run 5 — 2026-09-21 — the trace already segments the languages that do not space

Raised in review: vernacula-phonemizer has a trace function built for exactly this, and it
segments languages that do not use spaces. It does, and both of the places that throw the
segmentation away are this repo's.

`PhonemizeTrace` on the CJK cases, beside what the reader's word model sees:

```
  ja  "科学者たちが発表しました。"     traced=True  tokens=3   whitespace words=1
       [科学者たちが]  -> käɡäkɯᵝɕätät͡ɕiɡä
       [発表しました]  -> häppʲo̞ːꜜɕimäɕitä
       [。]            ->

  cmn "科学家宣布了这项发明。"        traced=True  tokens=1   whitespace words=1
       [科学家宣布了这项发明] -> kʰɤ˥˥ ɕyɛ˧˥ t͡ɕiɑ˥˥ ɕyæn˥˥ pu˥˩ lɤ ʈ͡ʂɤ˥˩ ɕiɑŋ˥˩ fɑ˥˥ miŋ˧˥

  en  "The harbour was quiet."        traced=True  tokens=4   whitespace words=4
       [The] [harbour] [was] [quiet]
```

**Two different shapes, and neither is nothing.** Japanese segments into phrases with real input
spans — bunsetsu-sized rather than word-sized, but two content units where whitespace sees one.
Mandarin returns a single token spanning the sentence, and carries its segmentation in the IPA
instead: one space-delimited group per syllable, ten of them for that sentence.

**Both are discarded here, in two places:**

1. `BlockItemViewModel.FromSegment` builds the reader's clickable words with a hard-coded
   whitespace scan (`while (!char.IsWhiteSpace(extractedText[i])) i++`), so a Japanese sentence is
   one clickable word before any of this is consulted.
2. `KokoroPhonemizer.GroupSourceWords` then maps each traced token to *the whitespace word
   containing its start offset*, which for CJK is always word 0.

So the group→word map comes out as "every group belongs to the single word", the counts happen to
agree, and the highlight covers the whole sentence at once. Nothing misaligns; the granularity is
simply thrown away twice before anything could use it.

### What closing it needs

The word tokenizer has to be able to take boundaries from somewhere other than whitespace.
`WordItemViewModel` already takes an arbitrary `extractedText[start..i]` and a style looked up by
character offset, and the trace's spans are character offsets into the same string, so the shapes
match. The two halves differ in difficulty:

- **Japanese: the token spans are usable as they stand.** Phrase-level highlighting rather than
  word-level, which is a real improvement over sentence-level and is what the trace offers.
- **Mandarin needs an IPA-group → character correspondence that has NOT been validated.** One
  hanzi is one syllable, and the IPA carries one group per syllable, so group *i* ought to be
  character *i* across a run of hanzi — but that is an assumption, and numbers, punctuation and
  latin runs inside Chinese text are exactly where it would break. It needs measuring before it is
  written, not after.

⚠ **AND THE PHONEMIZER WOULD HAVE TO RUN AT DISPLAY TIME, NOT ONLY AT SYNTHESIS TIME.** The reader
builds its words when a document is opened, which today needs no phonemizer at all. Taking word
boundaries from the trace makes the word list depend on it — including for a document that is
never synthesized — so a missing data tree stops being "renders in a different accent" and starts
being "the words are wrong". Whitespace has to remain the fallback.
