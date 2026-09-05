# IPA ruby annotation: segmentation for scriptio continua

Ruby (furigana-style) IPA over the reader's karaoke text works by attributing slices of the
phonemizer's reading to written words (PR #110). That relies on written words existing — a
whitespace split. Japanese, Chinese and Thai have no spaces, so the whole sentence is one "word"
and the entire reading stacks over it. This log is the hunt for a segmentation the render can use.

## Run 1 — 2026-09-05

**Question.** Does the phonemizer trace already segment scriptio continua? If so, are the token
spans good enough to place ruby over the right characters?

**Command.** A probe test dumping `PhonemizeTrace` tokens (surface, InputSpan, IpaSpan) for
ja/cmn/th/ko sample sentences.

**Raw finding.**

```
=== ja traced=True tokens=5   私は日本語を勉強しています。
    surf='私わ'      in=null  ipa='wätäɕiwä'
    surf='日本語を'  in=null  ipa='niho̞ŋɡo̞o̞'
    surf='勉強して'  in=null  ipa='be̞ŋkʲo̞ːꜜɕite̞'
    surf='います'    in=null  ipa='imäsɯᵝ'
    surf='。'        in=null  ipa=null

=== ja traced=True tokens=4   東京都に住んでいます。
    surf='東京都に'  in=[0,4)  ipa='to̞ːkʲo̞ːmijäko̞ni'
    surf='住んで'    in=[4,7)  ipa='sɯᵝꜜnde̞'
    surf='います'    in=[7,10) ipa='imäsɯᵝ'
    surf='。'        in=[10,11) ipa=null

=== cmn tokens=1              我喜欢学习中文。
    surf='我喜欢学习中文' in=[0,7) ipa='wo˧˥ ɕi˨˩˦ xuan˥˥ ɕyɛ˧˥ ɕi˧˥ ʈ͡ʂoŋ˥˥ wuən˧˥'

=== th tokens=1               ฉันเรียนภาษาไทย
    surf='ฉันเรียนภาษาไทย' in=[0,15) ipa='t͡ɕʰˈa˩˩˦n rˈia˧n pʰˈaː˧saː˩˩˦tʰˌa˧j'

=== ko tokens=3               저는 한국어를 공부합니다  (spaces already)
    surf='저는' in=[0,2)  / '한국어를' in=[3,7) / '공부합니다' in=[8,13)
```

**What it implies.**

1. Japanese IS segmented — into bunsetsu-like chunks (私は / 日本語を / 勉強して / います), which is
   exactly the unit ruby wants. Korean already has spaces and needs nothing.
2. **But the input spans can be absent for Japanese**: the first sentence reports `in=null` on every
   token, the second reports spans on all of them. The difference is a normalizer rewrite — 私は →
   私わ (the topic particle は read as わ). Provenance is dropped through that rewrite, and null
   means "not known", never "identical". So segmentation is available but *placement* is not,
   text-dependent.
3. Chinese and Thai are NOT segmented: a whole clause is one token. Chinese is still tractable —
   the reading has one space-delimited group per Han character (7 chars → 7 groups) — so a
   per-character ruby is possible. Thai is not: 15 characters → 3 groups, no alignment.

**Next step.** Two things to settle: how often ja loses provenance, and whether the token surface
can substitute for a missing input span (it cannot be displayed as-is — 私わ is not what the reader
typed).

## Run 3 — 2026-09-05 (code review of PR #110, probe harness)

**Command.** A throwaway xunit fact in `tests/Vernacula.Tts.Tests` calling
`IpaAnnotator.Annotate(words, lang)` alongside `Phonemizer.PhonemizeTrace(string.Join(' ', words), lang)`
and dumping both, over ~30 inputs (English abbreviations/currency/percent/fractions/hyphenation,
fr, fr-CA, as, de, es, ru, ar, hi, ko, ja, zh, en-GB). File deleted afterwards.

**Question.** Does the attribution hold on inputs beyond the five unit tests, and does the
annotation always equal what the synthesizer will say?

**Raw findings.**

```
Mr./Smith,/$3.14      -> mˈɪstɚ | smˈɪθ | θɹˈiː dˈɑːlɚz fˈɔːɹtˈiːn     correct
100/km, 5%, 1/2, 3rd  -> all correct
co-operate            -> kʰˈoᶷ ˈɑːpɚˌeᶦt  (stacked on the one word)     correct
10:30/a.m./on/Jan./1st,/2020 -> correct
"$" "3.14" (2 words)  -> <θɹˈiː> | <dˈɑːlɚz fˈɔːɹtˈiːn>                 WRONG, off by a word
fr-CA aujourd'hui     -> annotation oʒuʁdɥˈi   vs reading oʒuʁd͡zɥˈi    WRONG, pre-rewrite reading
ja こんにちは 世界      -> NULL   (matches Run 2: provenance dropped)
zh 你好 世界           -> NULL   (probe artefact: no "zh" alias; the reader picks "cmn")
```

**What it implies.**

1. The `Math.Min(lastWord + 1, lastInSpan)` distribution assumes emission order matches *written*
   word order inside a shared input span. That holds for `Mr. Smith` and fails for a currency sign
   written apart from its amount, where the sign is spoken last. The result is the "silently off by
   a word" case the design says it withholds. A safer rule for a multi-word span with no per-token
   input placement is to stack the whole group over the span's first word (or withhold the span).
2. The `IpaSpan is null → string.Concat(tok.Emitted)` fallback re-introduces exactly what
   `Trace.Stop` refuses to report: for the eight engines with a non-positional post-assembly
   rewrite (fr-CA accent, `as` aspirate collapse) the displayed reading is the pre-rewrite one.
   fr-CA is reproducible above and is a language the reader can pick.
3. The zh null is a probe artefact — `LanguageCatalog` offers `cmn`, which Run 2 traced fine.

**Next step.** Decide per case: stack-on-first-word for the inverted-order spans, and either drop
the `Emitted` fallback or gate it on `trace.Rewrites` carrying no post-assembly entry.

## Run 2 — 2026-09-05

**Question.** How often does Japanese lose input-span provenance, and is there a safe fallback?
Can Chinese be split per character?

**Command.** Probe over 20 hand-written Japanese sentences (particles, numbers, katakana, mixed
digits) and 8 Mandarin ones, reporting per sentence whether every token has an `InputSpan`, and
whether the token surfaces preserve the input's length.

**Raw finding.**

```
ja: 19/20 all-spans, 18/20 length-preserved

ja spans=False lenPreserved=True   私は日本語を勉強しています。
    fallback segments: 私は / 日本語を / 勉強して / います / 。
    token surfaces   : 私わ / 日本語を / 勉強して / います / 。
ja spans=True  lenPreserved=False  値段は3800円でした。      (surfLen 18 vs textLen 12)
ja spans=True  lenPreserved=False  明日の会議は10時から始まります。 (surfLen 18 vs textLen 17)

cmn surf='我喜欢学习中文' han=7 groups=7 match=True
cmn surf='今天天气很好'   han=6 groups=6 match=True
... 8/8 match
```

**What it implies.**

1. Losing provenance is the exception, not the rule: 19 of 20. The one that loses it is the は→わ
   particle rewrite, and it is length-preserving, so cumulative surface lengths reconstruct the
   segments EXACTLY (私は / 日本語を / 勉強して / います / 。).
2. The two sentences where lengths do NOT match are the ones with digits (3800円, 10時), where the
   number expands — and both of those keep their spans. The two failure modes are disjoint, so
   "spans, else cumulative length when the total matches" covers everything seen, and declines
   rather than guesses when neither holds.
3. Mandarin: one IPA group per Han character in all 8 sentences, so an all-Han run whose group
   count equals its character count can be split per character — pinyin-over-hanzi, the exact
   analogue of furigana.
4. Thai remains unsegmentable here (15 characters, 3 groups, one token): its reading stays over the
   whole run. Honest, and visibly different from the CJK case.

**Decision.** Segment inside a whitespace word, not across it: tokens by input span, else by
cumulative length when the total matches, else no split; then split an all-Han piece per character
when its group count matches. Keeping the split *inside* the word means the display word list stays
1:1 with the backend's aligned words, so karaoke timing still attaches by index — the pieces
subdivide their word's own span by IPA weight instead.

## Run 4 — 2026-09-05

**Question.** Does the design from Run 2 hold when built and run: does a Japanese sentence render
segmented, with the reader's own text preserved, and does the highlight follow the audio through
the pieces?

**Commands.** Unit tests over the annotator (16), then the reader driven on `:0` with the OmniVoice
backend, language `ja`, voice `ja`, 16 diffusion steps; xdotool typed the text and clicked
Synthesize, with frames captured every 0.32 s during playback.

**Raw finding.**

```
東京都に住んでいます。   ->  東京都に / 住んで / います / 。
                            to̞ːkʲo̞ːmijäko̞ni  sɯᵝꜜnde̞  imäsɯᵝ

私は日本語を勉強しています。 (the sentence with NO input-span provenance)
                        ->  私は / 日本語を / 勉強して / います / 。
                            wätäɕiwä  niho̞ŋɡo̞o̞  be̞ŋkʲo̞ːꜜɕite̞  imäsɯᵝ

playback frames:  f_3 -> 日本語を highlighted    f_6 -> 勉強して highlighted
synthesis: 2.84 s of audio, 16 steps
```

**What it implies.**

1. The reconstruction works where it matters. In 私は the token surface is 私わ, and the render
   shows 私は — the reader's own characters — while the ruby above reads wätäɕiwä. That is exactly
   the furigana relationship: the writing stays, the reading sits above it.
2. Per-piece karaoke inside one aligned word behaves. The backend times that whole sentence as a
   single word (no spaces to split on), and dividing its span by each piece's IPA weight moves the
   highlight bunsetsu by bunsetsu. It is an estimate within the word, not a measurement — the same
   caveat the OmniVoice word alignment already carries.
3. xdotool cannot type CJK at speed (it produced 私ししししし… at 25 ms/key); 250 ms/key is reliable.
   Recorded here only so the next UI check does not re-derive it.

**Still open.** Thai and other unsegmented scripts keep one reading over the whole run — correct but
not useful; segmenting them needs a dictionary the phonemizer does not carry. Korean already has
spaces and needs nothing.
