# Kokoro: a word-final vowel after a flap detaches from its word

Reported from listening to a TTS export: in one sentence `data` sounded "almost date-like", as though the
final vowel were not part of the word. Three earlier reports in the same session turned out to be lexicon
defects in vernacula-phonemizer; this one is not, and the log records the two wrong mechanisms that were
chased first, because both were plausible and both had supporting measurements.

## Run 1 — 2026-09-11 11:15 — the duration hypothesis, and why it was wrong twice

**First wrong answer.** Synthesized the reported phrase as a fragment and measured Kokoro's own `pred_dur`:
the final vowel of `data` came out at 200ms. Concluded duration was not the problem and that the percept was
a synthesis characteristic. That measurement was taken on a 46-rune fragment, which is not the utterance that
was exported.

**Second wrong answer.** Kokoro indexes its style vector by phoneme-string length (`ref_s = pack[len(ps)-1]`),
so a longer input selects a later style row. Re-measuring at realistic lengths:

```
unit                 runes  refRow   data final vowel
fragment                46      45   200ms
the reported sentence  138     137    75ms
a synthetic paragraph  346     345    50ms
```

A clean monotonic effect, and `KokoroTts.ChunkForSynthesis` packs to the 508-token BERT limit rather than to
any prosodic unit — so long paragraphs do get the most compressed style rows. This looked like the answer and
sentence-level chunking looked like the fix. It was still wrong: the real export's chunks are 39–396 chars and
its five `data` tokens run 225–300ms, with no pathological compression. **A measurable effect is not
automatically the reported one.**

## Run 2 — 2026-09-11 11:40 — the real export, and the environment that actually conditions it

Went to the exported job's sidecar (per-word timings) instead of reconstructing. Five `data` tokens; the
reporter identified exactly one as wrong, and it is the LONGEST:

```
#   t        dur   context                    verdict
1   21.43s  275ms  data -> were        (/w/)  fine
2   41.03s  250ms  data -> collection  (/k/)  fine
3   53.00s  300ms  data -> required    (/ɹ/)  BAD
4   67.30s  225ms  data -> collection  (/k/)  fine
5   73.52s  250ms  data -> collection  (/k/)  fine
```

Occurrences 3 and 4 are in the SAME chunk, which rules out every chunk-level explanation including the one
from Run 1. Per-token durations for the two, from the same synthesis:

```
data -> /ɹ/   d=1  ˈ=3  A=3  T=3  ə=2    word 300ms   flap 75ms, vowel 50ms
data -> /k/   d=1  ˈ=2  A=2  T=2  ə=2    word 225ms   flap 50ms, vowel 50ms
```

Before the rhotic the word inflates 33% — and the final schwa is the ONLY segment that does not inflate with
it. The vowel ends up shorter than the consonant preceding it, which is what "too much space between the t
and the a" describes. The reporter's phrasing, not any measurement of mine, is what located this.

## Run 3 — 2026-09-11 11:55 — three candidate fixes, two of them wrong

Substitutions tried in the real chunk, judged by ear:

| variant | flap | vowel | verdict |
|---|---|---|---|
| `T` (current) | 75ms | 50ms | the bug |
| `ː` added after the vowel | 75ms | 100ms | no better — lengthening the vowel is not the lever |
| `ɐ` (near-open, vocab 70) | 75ms | 50ms | no better |
| `ɾ` (tap, vocab 125) | 75ms | 50ms | **good** |
| `d` (vocab 46) | 50ms | 50ms | **good**, but see below |

`ɾ` and `T` have IDENTICAL flap durations, and `ɾ` still sounds right — so the flap:vowel ratio framing from
Run 2 is not the whole story either; the substitution shifts the whole word's duration profile out of the bad
state. Several different edits all work, so "which fixes it" stopped discriminating and the choice became
which is safest.

**`d` is disqualified.** It merges every flap minimal pair at the token level:

```
writer  ɹˈITəɹ -> ɹˈIdəɹ  ≡  rider   ɹˈIdəɹ
latter  lˈæTəɹ -> lˈædəɹ  ≡  ladder  lˈædəɹ
metal   mˈɛTəl -> mˈɛdəl  ≡  medal   mˈɛdəl
atom    ˈæTəm  -> ˈædəm   ≡  Adam    ˈædəm
```

`T` is what preserves the underlying /t/. This is the argument that killed the change `KokoroFormat`'s own
`("d̬", "d")` rule would otherwise have suggested by analogy — and it would not have shown up in any
single-word test.

**A global `T -> ɾ` is also unjustified.** A 10-flap passage (writer, noted, latter, meeting, better, little,
water, metal, city, data) was judged good under BOTH the current `T` and an all-`ɾ` rendering. `T` is not
broken in general, so swapping every flap to fix a rare environment is not warranted.

## Run 4 — 2026-09-11 12:10 — how rare, and what actually conditions it

Scanned every intervocalic flap in the whole export:

```
intervocalic flaps: 17
  flap duration > following vowel: 4 (23.5%)

  səpˈɔɹTɪŋ …   (supporting)  vowel word-INTERNAL, followed by ŋ   — not reported
  ɹᵻlˈATᵻd …    (related)     word-INTERNAL, followed by d         — not reported
  lˈɪməTᵻd …    (limited)     word-INTERNAL, followed by d         — not reported
  dˈATə ɹikwˈ   (data)        vowel WORD-FINAL                     — the reported one
```

The ratio alone does not predict the percept: three rows share it and pass unnoticed. The discriminating
feature is that the post-flap vowel is **word-final** — in the other three it is propped up by following
consonants in the same word. That is the rule's actual condition.

## Run 5 — 2026-09-11 12:20 — the guarded rule, and a scope estimate that was two orders of magnitude out

```csharp
private static readonly Regex WordFinalFlapRe =
    new(@"T([əɐaeiouɑɔɛɪʊʌæɜAIOWYᵻ])(?=[ ,.;:!?…—]|$)", RegexOptions.Compiled);
```

Applied at the very end of `Render`, after the punctuation pass so the lookahead sees an attached mark, and
en-us only. A stressed final vowel cannot match, because its stress mark sits between the `T` and the nucleus.

**Scope was initially estimated at "maybe a dozen words" — reasoning from `data`'s shape without checking what
else ends in flap + word-final vowel. The real figure is 1,356 lexicon rows:**

```
-y / -ie   700    city, pretty, duty, forty, activity, quality
-a         396    data, beta, meta, pita
other      260
```

`-ity` is one of the most productive suffixes in English, so this is a broad change, not a contained special
case — which removed the main argument for preferring it over a global swap, and made `city` (standing in for
700 words) the real acceptance test rather than `data`. Confirmed good by ear on a passage carrying `data`,
`beta` and `city` as targets alongside `writer`, `latter`, `meeting`, `better`, `little`, `water`, `metal`,
`noted`, `limited`, `supporting`, `related` as controls: `T` count 15 → 11, only the three targets moved.

`dotnet test` on Vernacula.Tts.Tests: 184 passed, 4 skipped.

### What this cost, and the lesson worth keeping

Two fixes were proposed and nearly built on mechanisms that were real, measurable, and not the cause —
chunk-length compression most of all, which would have shipped as sentence-level chunking and changed nothing
about this bug, since the good and bad instances share a chunk. What finally isolated it was going to the
actual exported artifact and its per-word sidecar rather than reconstructing the utterance, and taking the
reporter's description of the artifact ("space between the t and the a") as the primary evidence. A/B WAVs
sent for each hypothesis are what kept each wrong turn to one round instead of shipping.
