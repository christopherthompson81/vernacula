# How fine is Japanese highlighting, and can it be finer?

Reported impression: phrase-level highlighting would feel imprecise to a reader. The question asked
was whether this session and the phonemizer session together could push it closer to word level.

The measurement moved the question. Granularity is already word-ish; the imprecision is a
correctness defect on mixed-script text, and it is worse than anything phrase-level would be.

## Run 1 — 2026-09-22 — is there finer MEASURED structure we are discarding?

Question: the cmn path found one IPA group per hanzi and used it. Does `ja` hide the same
opportunity — more IPA groups than the units we highlight?

Probe over eight hand-written sentences, printing trace tokens, derived word spans, and the Kokoro
IPA's groups (runs between space tokens):

```
科学者たちが発表しました。   traceTokens=3  words=2  ipaGroups=2
今日はいい天気ですね。       traceTokens=3  words=2  ipaGroups=2
私は毎朝七時に起きます。     traceTokens=4  words=3  ipaGroups=3
彼女は新しい本を読んでいます。 traceTokens=6  words=5  ipaGroups=5
```

**`ipaGroups == words` on every row, and `traceTokens` is always exactly one more** — the extra is
the trailing `。`, which produces no spoken group and is correctly not a clickable word.

**Finding: there is no free granularity.** Unlike Mandarin, where the IPA carries one group per
syllable and the characters walk onto them, the Japanese IPA is cut at exactly the units we already
show. Anything finer than a trace token has **no measured boundary behind it** and would have to be
estimated inside a group.

Worth stating plainly because it is the whole constraint: splitting a unit without evidence for
where the audio boundary falls produces confident wrong highlighting, which is the failure mode this
codebase keeps electing to avoid.

## Run 2 — 2026-09-22 — the units are not phrases

Question: how coarse is a unit actually? The impression came from
`科学者たちが | 発表しました`, two units for a whole sentence.

That sentence is among the coarser ones. `彼女は新しい本を読んでいます。` segments as
`彼女は | 新しい | 本を | 読んで | います` — five units, and that is word level by any ordinary
reading. Measured over the phonemizer's own 200-row `ja` golden corpus (123 distinct rows):

```
median chars per highlight unit = 4.27
mean   chars per highlight unit = 5.20
units per sentence: median=12  max=21
```

**Finding: it is not phrase-level segmentation.** A median of 4.3 characters per unit and 12 units
per sentence is close to word level already. The coarse cases are where one dictionary entry spans a
compound plus its particles (`科学者たちが`), not a systematic phrase policy.

⚠ **So the reported impression is real but its cause was not where either of us looked.** Which is
what Run 3 found.

## Run 3 — 2026-09-22 — the actual defect: units that cannot be told apart

Question: if granularity is fine, what makes it feel imprecise?

While probing mixed-script input — ordinary in Japanese, where `PDF`, `Wi-Fi`, `CD` and Latin proper
nouns appear constantly — every token came back claiming **the same input span**:

```
PDFファイルを開いてください。
   token surface=ピーディーエフ   span=PDFファイルを開いてください
   token surface=ファイルを       span=PDFファイルを開いてください
   token surface=開いてください    span=PDFファイルを開いてください
   -> 3 units, 1 distinct span
```

Swept over the golden corpus:

| | |
|---|---|
| rows with **degenerate** spans (≥2 units sharing one span) | **14 / 123** |
| of which mixed-script | **14 / 49** — every single one |
| rows where the **measured tier still engages** on them | **14** |

Worst cases, and they are not marginal (the corpus is news and reference prose, so its rows name
real people and organisations; described by shape rather than quoted):

```
a news sentence with a latin acronym mid-clause
   9 units, 2 distinct; one 30-char span repeated 8x
a news sentence with a latin acronym inside a katakana proper noun
   8 units, 3 distinct; one 22-char span repeated 4x
a reference sentence contrasting two latin-spelled loanwords
   21 units, 8 distinct; one 18-char span repeated 8x
```

**A span covering 30 of 34 characters, offered as eight separate clickable units.** Clicking any of
them lights almost the whole sentence. That is not "less precise than word level" — it is worse than
the one-unit-per-sentence fallback it replaced, and it is the likeliest source of the reported
impression.

⚠ **AND THE EXISTING GATE DOES NOT CATCH IT.** `Align` declines the measured tier when
`groups.Count != map.Count`, because a count disagreement means the engine cut the stream differently
than we did. Here the counts **agree** — 8 groups, 8 map entries — and every entry points at the same
span. The invariant was derived for a different failure and being *degenerate* is invisible to it,
exactly as being *one out* is invisible without a derived check. All 14 rows take the measured tier
and are trusted.

## Where the fix belongs

**Upstream**, primarily. `PDF` → `ピーディーエフ` is a normalizer expansion, and provenance cannot map
the expanded kana back to sub-spans of the original — so it attributes the **whole input** to every
token. Their own contract says `InputSpan` null means "not known, never identical", and a whole-input
span is a *known-looking* answer to an unknown question. This is the same absence/disagreement
collapse the phonemizer session described this week, one level down: a not-known reported as a
coarse-known. Null would have been honest, and this repo already degrades correctly on null.

**Here**, defensively and cheaply: `Align`'s count check should also require the spans to be
distinguishable, and decline the measured tier when they are not. Declining costs granularity on 11%
of Japanese rows and costs nothing anywhere else; trusting them costs correctness on exactly those
rows. Better still, dedupe to the distinct spans and merge the groups that share one — that keeps the
2–8 real boundaries instead of throwing the sentence back to one unit.

## On the original question

Going **finer than a trace token** needs upstream, and needs a real boundary rather than an estimate;
locally there is none to be had. But the corpus says that is not where the win is. Fixing the
degenerate spans moves 14 of 123 rows from "eight units all lighting 30 characters" to honest units,
and that is a far larger improvement in felt precision than subdividing `科学者たちが` would be.

Raised with the phonemizer session with these numbers.

## Run 4 — 2026-09-22 — the local half: two units that claim the same characters are one unit

The spans are wrong at the source and that is upstream's. What is local is that a wrong span is
shown as a confident one, and that part is fixable here.

**Where the fix had to go.** Not in `Align`. The reader pairs displayed words to sidecar words BY
INDEX, and both sides call `WordSegmentation.Segment` — deduping in the aligner alone would give the
two sides different unit counts and shift the whole document, which is exactly the defect review
caught on #237. Fixing it in `FromTrace` keeps them consistent by construction.

**Why merging is the honest answer and not a workaround.** If two tokens cannot say which characters
are theirs, those characters *are* one clickable unit, and its time is the union of their groups.
This is already the answer for an English rewrite where one written word becomes several tokens —
`$3.14` → three, dollars, fourteen — and nothing about Japanese makes it different.

Both downstream pieces already tolerated it, which is why the change is four lines:

- `KokoroAlignment.WordsFromGroups` collects first-start / last-end per unit, so several groups on
  one unit union correctly.
- `GroupSourceWords`' shared-span branch already collapses successive tokens onto one index.

What the old code actually produced, checked rather than assumed: the map was *already* `[0,0,0]`,
so unit 0 got the whole time and units 1 and 2 got **zero-length markers**. Eight duplicate units
did not merely light the same text — seven of them seeked to the end of the sentence.

### Measured, same corpus, before and after

```
                              before     after
degenerate (duplicate spans)    14         0
overlapping pairs                -         0
unordered pairs                  -         0
out-of-range spans               -         0
median chars per unit          4.27      4.33
median units per sentence        12        11
```

**The cost of being correct here is 0.06 characters per unit.** 14 of 123 rows stop offering units
that cannot be told apart, and the granularity everywhere else is untouched.

Mandarin re-measured as a regression check, since the per-hanzi walk now appends through the same
merge: **1.00 chars per unit, median 35 units per sentence, 0 degenerate** — the one-unit-per-hanzi
split is intact.

### The tests were shown to fail first

Per the rule this week keeps teaching: both new tests were run against the reverted fix before being
trusted.

```
Failed  TracedWordsAreOrderedAndNeverOverlap
Failed  MixedScriptJapaneseDoesNotOfferTwoWordsCoveringTheSameCharacters
```

Red without the merge, green with it. Full suite after: 95 + 257 + 23 + 362, 0 failures.

### Still open, and upstream's

The spans remain wrong at the source — `PDF` → `ピーディーエフ` attributes the whole input to every
token rather than withholding. Merging means we no longer *show* a wrong span confidently, but we
also cannot recover the real boundaries, so those 14 rows are coarser than they need to be. If
upstream withholds (null) instead, this repo already degrades correctly; if upstream maps the
sub-spans properly, the merge simply stops firing. Either outcome is strictly better and neither
requires a change here.

## Run 5 — 2026-09-22 — review of the change, and a worse bug found by reading it

Three findings, and the third was not what the review was looking for.

**A silent character loss in the merge.** `Append` extended the previous span with
`Math.Max(last.End, span.End)` and kept `last.Start`. A token arriving with an EARLIER start would
therefore drop the characters ahead of it out of every unit — and a character in no unit is not
clickable and never highlights. Measured as unreachable today (zero unordered pairs across both
corpora) and fixed anyway, to the union of both ends: a silent loss should not rest on an ordering
this code does not enforce.

**Both new tests passed vacuously on an empty result.** `Distinct().Count() == Count()` is true of an
empty list and the ordering loop does not execute, so a regression that returned nothing would have
turned them green. `Assert.NotEmpty` on both. Deliberately still not pinning the unit count for the
mixed-script row — if upstream maps the expansion's sub-spans properly it becomes three distinct
units, and the test should keep passing.

### The one worth the review: two traces, assumed to agree

`KokoroPhonemizer.Phonemize` read a trace for the group→word map and called `Segment`, which traced
the same text AGAIN for the words the map indexes into. Its own comment asserted the two "must read
the same spans" — an invariant nothing enforced, and **#1408 was precisely a case of one trace of a
text disagreeing with the next one**. The assumption had already been wrong once, in the exact way
that would desync a map from its words.

A new `Segment` overload takes the trace the caller already holds, and `Phonemize` hands one trace
to both. Incidentally halves the phonemization work per paragraph for `ja` and `cmn`, but the reason
is correctness: the two cannot disagree if there is only one.

⚠ **THIS WAS NOT FOUND BY A TEST AND NO TEST WOULD HAVE FOUND IT** — the traces agree now that #1408
is fixed, so everything passes either way. It was found by reading a comment that claimed an
invariant and checking whether anything established it.

### A control that was itself invalid

First attempt at re-confirming the tests could fail stashed only the uncommitted review fixes, while
the merge itself was already committed — so the "control" ran against the fix and reported green. The
tests looked vacuous when they were not. Redone against `main`'s segmentation:

```
Failed  TracedWordsAreOrderedAndNeverOverlap
Failed  MixedScriptJapaneseDoesNotOfferTwoWordsCoveringTheSameCharacters
```

Worth recording because it is the same trap one level up: *verifying that a guard can fail is itself
a procedure that can be performed wrongly and report success.* Checking what the control actually
reverted is part of the control.

Corpora re-measured after all three fixes — `ja` 0 degenerate / 4.33 chars per unit, `cmn` 0
degenerate / 1.00 — and the full suite is 95 + 257 + 23 + 362, 0 failures.
