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

> **The mechanism in this section is wrong — see Run 6.** The cause is not the Japanese normalizer
> and not an expansion at all; it is a `\p{L}+` rewrite in `normalizeRomans`, which runs over every
> language. The *shape* of the diagnosis held, the placement did not. The recommendation to decline
> on indistinguishable spans is also withdrawn there.

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


## Run 6 — 2026-09-22 — the cause was one layer up, and not Japanese

The phonemizer session reproduced and fixed it (#1420). **My diagnosis had the right shape and the
wrong placement**, and the difference matters here because it is not `ja`-specific.

I read it as `PDF` → `ピーディーエフ` being unable to map expanded kana back to sub-spans. They dumped
the per-character mapping after each stage and it was **correct throughout** — `ピーディーエフ` mapped
to `[0,3)`, the rest 1:1. Instrumenting the rewriter to report any match of 8+ characters named it
in one run:

```
WIDE 15ch  \p{L}+  on "PDFファイルを開いてください"
  at normalizeRomans (src/core/roman.ts:207)   ← runs over EVERY language
```

`normalizeRomans` rewrites on `\p{L}+` and returns the token unchanged when it is not a Roman
numeral. ⚠ **In a script without spaces there is no word break for `\p{L}+` to stop at**, so the
match is the whole clause, and the rewriter stamped the match's span across every character of the
replacement even though the replacement *was* the match. Fixed by carrying the original
per-character mapping through an identical replacement — and only identity is safe, since an
equal-length but different replacement has no guaranteed correspondence.

**And the fast path explains the 14-of-14 exactly.** `normalizeRomans` returns early when there are
no Roman letters, so a pure-kana sentence never reaches the rewrite. The defect was absent from
precisely the sentences anyone reaches for first when testing Japanese, and present in precisely the
ones with a latin letter. "All 14 mixed-script" was not a correlation to note — it was the mechanism.

```
ja rows where two tokens share a span:  14 → 3
fleet-wide (35,021 rows with 2+ spans): 4,639 → 4,262
```

### Withdrawing the recommendation this document made

Run 3 suggested `Align` should decline the measured tier when spans are indistinguishable. ⚠ **That
would have been a bug.** Two tokens sharing a span is frequently *correct* — a numeral expansion
legitimately produces several tokens from one source span, which is what the other 4,262 fleet rows
are. A blanket decline would throw away good expansions along with bad spans.

Merging, which is what was actually built, does not have that failure: for the two languages that
reach `FromTrace`, several tokens on one span means one clickable unit whose time is the union of
their groups, and that is the right answer whether the shared span came from a legitimate expansion
or from the `\p{L}+` artefact. **The implementation was right and this document's advice was not**,
which is worth recording precisely because the advice reads more confident than the code.

### What remains after the upstream fix

`ja` keeps a residue of 3, which is numeral/unit coarseness — `83 m` gives one token spanning `83 m`
and another spanning `83 mです`. That is *overlap*, not collapse: a different and smaller thing, and
the merge handles it. So the guard stays useful rather than becoming inert, which is not what Run 4
predicted.

### On granularity, the question that started this

They declined to change the tokenizer, on these numbers: median 4.27 chars per unit with
`彼女は|新しい|本を|読んで|います` already word-level is not a tokenizer that needs moving. Nothing
finer is exposed today — `segmentText` works on bunsetsu units, and the readings map it consults does
hold sub-units, but they are used during longest-match and not retained, so there is no API for them.

⚠ **And the decisive reason is the one from Run 1, which they agreed with:** one IPA group per trace
token means a subdivision would have no measured audio boundary behind it, so exposing sub-units
would hand this repo an estimate dressed as a boundary. If the audio side ever gains per-mora groups
the question reopens; until then the answer to "can Japanese highlighting be more word-level" is that
it already is, and the thing that made it feel otherwise is fixed at both ends.

## Run 7 — 2026-09-22 — taking the upstream fix, and what it actually bought

Pin `ac60b3c4` → `6e2165c7`, five commits. Measured rather than relayed, because the interesting
number is the one this repo sees rather than the one upstream reports.

**The upstream defect, counted on the raw trace** — the local merge collapses these downstream
either way, so counting units would have measured our own workaround instead of their fix:

```
                                       ac60b3c4        6e2165c7 (#1420)
ja  rows with 2+ tokens sharing a span      14                3
cmn rows with 2+ tokens sharing a span       3                3
```

`ja` 14 → 3 reproduces their figure exactly.

⚠ **Mandarin was never affected, and the reason is worth keeping.** The `cmn` trace returns ONE
token per sentence carrying per-syllable IPA, so a whole-clause `\p{L}+` match has no second token
to flatten against. The defect needed several tokens to collapse and only `ja` had them — which also
means the `cmn` residue of 3 was never this bug and is unchanged by its fix.

**What the reader gets**, measured through `KokoroPhonemizer` on the same corpora:

```
        median chars per unit    median units per sentence
ja        4.33  ->  4.08              11  ->  12
cmn       1.00  ->  1.00              35  ->  35
degenerate / overlapping / out-of-range: 0 throughout
```

Eleven rows stop being merged into coarser units and get their real boundaries back. Small in the
median because it is 11 of 123 rows, and those rows were the worst ones.

**The merge is not inert and the residue confirms it.** `ja` keeps 3 rows of numeral/unit overlap
(`83 m` against `83 mです`), which is overlap rather than collapse. Run 4 predicted the guard would
stop firing once upstream landed; Run 6 corrected that, and this measures it: still 3 rows relying
on it.

Cold-trace gate re-run on the new pin: **189 of 189 languages, no poisons**. Full suite:
95 + 257 + 23 + 362, 0 failures.

**Scope.** Of the five commits, one is core provenance (`Rewriter`, `Provenance`, `JsRegex`, and the
TypeScript twin) and the rest are English data and lexicon work, including #1418 reverting 26 rows on
re-arbitrated evidence and #1425 porting code-slot unit guards to C#. So this carries English
behaviour changes as well as the fix — the same shape as the previous bump, and the reason the
English goldens are worth a look rather than a nod.
