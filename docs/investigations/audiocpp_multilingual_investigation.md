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

## Run 6 — 2026-09-21 — building it, and two places the map still collapsed

`WordSegmentation` is the shared unit: whitespace for most languages, the trace for the two that
do not space, and whitespace again whenever the trace cannot do better. The reader, the group→word
map and the aligner all take it, so the segmentation is produced once instead of three times by
three whitespace scans that agreed by construction in English and by accident nowhere else.

Wiring it exposed two defects that the whitespace-only world had hidden.

⚠ **A TOKEN THAT SAYS NOTHING WAS NULLING THE WHOLE MAP.** `GroupSourceWords` returned null for any
token with neither an IPA span nor emitted groups. In English punctuation rides on the preceding
word and never becomes a token, so the branch never ran; a Japanese sentence ends with 。 as its
own token with no IPA, which nulled the map for **every Japanese paragraph** and sent the aligner
to an even split. It contributes nothing now instead of abandoning everything.

⚠ **AND ONE TOKEN COVERING SEVERAL WORD UNITS MAPPED THEM ALL TO THE FIRST.** Mandarin arrives as a
single token spanning the sentence with one group per syllable. The segmenter had already cut that
span into one unit per hanzi — but the map assigned every group to `wordAt[input.Start]`, i.e.
word 0, so all six syllables pointed at 今 and the highlight covered the sentence anyway. When a
token's span covers exactly as many units as it produced groups, the groups are distributed across
them; that check is the same count agreement the segmenter used to cut the span in the first place.

After both:

```
ja   "科学者たちが発表しました。"   2 spans: 科学者たちが | 発表しました    map=0,1        measured
ja   "今日はいい天気ですね。"       2 spans: 今日はいい | 天気ですね        map=0,1        measured
cmn  "今天天气很好。"              6 spans: 今|天|天|气|很|好              map=0,…,5      measured
```

## Run 7 — 2026-09-21 — a cold-call defect in the phonemizer, found by the test

The Japanese test failed at first with one span where two were expected, and the cause was not in
this repo:

```
COLD  PhonemizeTrace("科学者たちが発表しました。", "ja")  tokens=3, InputSpan=null, null, null
WARM  same call again                                      tokens=3, InputSpan=[0,6) [6,12) [12,13)
```

**The FIRST trace for `ja` in a process returns its tokens with no input spans; every later call
returns them populated.** Characterised: Japanese only (`cmn` is correct cold), per-language rather
than global (tracing `en` first does not help), and `IpaSpan` is unaffected — only `InputSpan` is
missing, which points at whatever records input offsets being initialised *during* the first call.

Untreated this is the worst shape a defect can take: the FIRST Japanese document a session opens
silently gets whitespace words and every later one gets phrases — correct on the second look. A
single retry, guarded on the exact signature (tokens present, every `InputSpan` null), makes it
deterministic; it costs one extra phonemization of one block on the path that is already broken and
is deleted the day it is fixed upstream. Reported to the phonemizer session with the repro.

**The test found this, not the app.** A whitespace fallback that silently produces something
plausible is exactly the kind of degradation that never surfaces in use.

### Narrowed, and filed upstream as vernacula-phonemizer#1408

The phonemizer session reproduced it and narrowed it four ways that each would have sent a fix
somewhere else: it is **C# only** (the TypeScript engine is correct cold, so this is a port
divergence), it is **not about tracing** (a plain untraced `Phonemize` also warms it, so the
trigger is first-use initialisation on the Japanese path), tracing another language first does not
help, and it is **not "lazily-loaded languages"** — ten swept, `ja` alone, including among the other
non-spacing scripts.

Mechanism as far as they took it: `Trace.Stop` resolves spans through `Provenance.For(r.Normalized)`
which returns the mapping only when `tracked == normalized`, and something in Japanese first-use
init runs a tracked rewriter over a string that is not the caller's input. ⚠ **Returning null there
is the CORRECT behaviour** — `InputSpan`'s contract is "absent means not known, never identical" —
and it is what saved this. A confident wrong offset would have cut Japanese at the wrong characters
and produced a plausible, silently wrong highlight; the null merely made it coarse.

Two measurements contributed back, one cold process per language:

```
en en-GB es fr it pt-BR hi cmn    cold=present
ja                                cold=ALL NULL  retry=present  third=present

ja : 25 golden rows, one cold process · lost spans on 1 (row 0) · retry failed twice on 0
cmn: 25 rows · lost spans on 0 · hi: 25 rows · lost spans on 0
```

So it is strictly the first call in a process, it warms globally for the language rather than per
text, and one retry has always sufficed — the blast radius is exactly one trace per process, which
is why it survived.

⚠ **AND THE GENERAL GAP IS THAT THE TRACE HAS NO PARITY GATE AT ALL.** The port's parity harness
compares IPA strings, and the trace is not in the goldens, so a divergence this visible was
structurally invisible. Not this repo's to fix, but worth knowing why a defect of this size lasted:
nothing was looking.

## Run 8 — 2026-09-22 — #1408 fixed upstream, and the retry removed against a control

The phonemizer session fixed it (#1417) and reported the cause, which lands exactly where the
narrowing pointed: `Normalize`'s **static constructor** builds its digit-kana table by calling
`ToKatakana("れい")`. A static constructor runs lazily on first use, and for that class first use is
inside `NormalizeJapanese` — *inside the traced window*. So it called `StartTrack("れい")` while the
tracked string was the caller's whole sentence, the mismatch rule correctly refused the mapping, and
the trace came back with every span null. Their `OnPoison` hook named it in one run:

```
POISON tracked="科学者たちが発表しました。" got="れい"
  Provenance.StartTrack ← Rewriter.Rewrite ← Normalize.ToKatakana ← Normalize..cctor()
```

Three copies of that helper exist; only that one goes through the tracked seam, which is why
TypeScript was always clean. ⚠ **The `Provenance.For` null was never the bug** — the equality check
was untouched, and the contract it protects ("absent means not known, never identical") is what kept
this coarse instead of confidently wrong.

**Pin bumped `c6e26698` → `ac60b3c4`** (13 commits; several are the en-GB line descending from
#1385) and the retry in `WordSegmentation.Trace` deleted.

### The control is the part worth recording

A passing check proves nothing here unless it can fail, and this defect is once per process — so
anything running inside a test assembly is already warm. The probe is one trace, one language, one
cold process, run against both pins:

```
ac60b3c4 (fixed)    lang=ja tokens=3 withInputSpan=3   OK
c6e26698 (previous) lang=ja tokens=3 withInputSpan=0   POISONED
```

Same probe, same shape, only the pin differs — so the OK means the fix and not the probe. Then all
nine languages this app speaks, one process each: **9 of 9 populated**, `cmn` 1/1, `ja` 3/3.

### Better: run their gate, not my probe

Reviewing the diff turned up that the fix shipped with a tool — `csharp/tools/trace-cold`, spawned
by `TraceColdInitTests` — so the throwaway probe above is the weaker instrument and nothing in this
repo should point at it. Run against our exact pin:

```
dotnet run --project csharp/tools/trace-cold -c Release -- csharp/goldens
  traced 189 of 189 languages
  no poisons, every traced language carries input spans          exit=0
```

189 languages beats my nine, and it is maintained upstream rather than living in a scratch
directory. ⚠ **But it runs in THEIR suite, not ours** — we consume a pin, not their CI — so what
protects this repo is the pin bump itself plus whatever they gate before tagging. `WordSegmentation`
now says so, and so does the `ja` test, which structurally cannot catch a cold-init regression
because any earlier test in the assembly has already warmed the language.

### One claim in the first draft was wrong

`WordSegmentation.Trace` was documented as "the one place a trace is taken". It is not:
`IpaAnnotator` and the OmniVoice IPA path each call `PhonemizeTrace` directly. Those were exposed
to #1408 too and nobody guarded them — moot now, but the honest scope is narrower, and is what the
comment claims instead: `Segment` builds the words and `KokoroPhonemizer.Phonemize` builds the
group→word map *onto those same words*, so those two must read one trace and not two.

Full suite after removal: 255 + 23 + 362 pass, 0 fail.

**The peer hit the inverse of this and it is the same lesson.** Their first gate lived in the test
assembly and passed *with the fix reverted*, because another test had already warmed `ja`. The
correction was a standalone tool spawned in its own process — and one process covers all 189
languages, since each initializes on its own first trace (~13 s together). My instinct that it had
to be one process per language was half right; the constraint is the process, not the language.

### Still no parity gate on the trace

Their new gate catches cold-init poisoning, not a port divergence in ordinary spans, and the trace
is still absent from the goldens the parity harness compares. Unchanged from Run 7 and still not
this repo's to fix, but this repo now depends on trace spans in two languages, so it is worth
knowing that the thing we depend on is gated for one failure mode and not the other.

They also flagged that their C# suite had been red for thirteen merges — only ever run under a
`--filter` — so "C# 189 byte-identical" reports through that stretch were parity-of-engine-output
claims, not suite-green claims. Nothing that touched this case.
