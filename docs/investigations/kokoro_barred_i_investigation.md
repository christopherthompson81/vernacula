# The reduced prefix vowel: ⟨ᵻ⟩ against ⟨ə⟩ in Kokoro

Reported: "determine" sounds tensed. The phonemizer session established it is not a dictionary
defect — our IPA is already reduced on every path — and handed the question here, because the symbol
survives into Kokoro through `KokoroFormat` and only this repo can run the synthesis.

```
determine   sync dᵻtʰˈɝmən   async dᵻtʰˈɝmən
us_gold     dətˈɜɹmən
```

## Run 1 — 2026-09-22 — what this repo does with the symbol

Question asked by the phonemizer session, which cannot see this code: what does `KokoroFormat` do
with ⟨ᵻ⟩?

**Nothing, deliberately.** It is absent from the replacement table, with a comment saying why:

```csharp
// ᵻ (U+1D7B) is deliberately unmapped: it is Kokoro vocab id 177, not out-of-vocab.
```

So the divergence from the training lexicon reaches the model intact. That is the precondition for
the report being real rather than a phonemizer-side artefact, and it holds.

## Run 2 — 2026-09-22 — does Kokoro render the two symbols differently at all?

The cheapest decisive question, and it comes before any judgement about which is better: if the model
collapses ⟨ᵻ⟩ and ⟨ə⟩ to the same output, the 2,201-word divergence is harmless and nothing needs
changing anywhere.

Same voice (`af_heart`), same word, supplied phonemes, one symbol swapped — so any difference IS the
symbol:

```
determine    dᵻtˈɜɹmən  -> dətˈɜɹmən    rmsDiff=0.064  maxDiff=0.478  DIFFERS
describe     dᵻskɹˈIb   -> dəskɹˈIb     rmsDiff=0.060  maxDiff=0.435  DIFFERS
prefer       pɹᵻfˈɜɹ    -> pɹəfˈɜɹ      rmsDiff=0.065  maxDiff=0.528  DIFFERS
remember     ɹᵻmˈɛmbəɹ  -> ɹəmˈɛmbəɹ    rmsDiff=0.079  maxDiff=0.503  DIFFERS
reduce       ɹᵻdˈus     -> ɹədˈus       rmsDiff=0.049  maxDiff=0.435  DIFFERS
```

**Kokoro treats them as distinct phones.** Not a collapsed pair, so the divergence is capable of
being audible and the report is capable of being right.

⚠ **Two of the five also come out LONGER with ⟨ᵻ⟩** — `determine` by 50 ms and `reduce` by 25 ms,
against identical leading silence in every case, so the extra length is inside the word rather than
before it. That is the direction a less-reduced vowel would move, and "sounds tensed" is what a
listener would call a prefix vowel that is too long and too full. It is consistent with the report;
it is not proof of it.

`decide` and `prepare` carry no ⟨ᵻ⟩ at all (`dɪsˈId`, `pɹipˈɛɹ`) and were dropped from the set.

## What is established, and what is not

| | |
|---|---|
| our IPA is already reduced, so this is not a dictionary defect | phonemizer session, measured |
| ⟨ᵻ⟩ reaches Kokoro unmapped | this repo, read from the code |
| Kokoro renders ⟨ᵻ⟩ and ⟨ə⟩ differently | **measured here** |
| ⟨ᵻ⟩ in PREFIX position is ~absent from Kokoro's training lexicon (9 occurrences in all of us_gold) | phonemizer session, measured |
| ⟨ᵻ⟩ elsewhere is well supported there (2,203 occurrences; see Run 3 for the breakdown) | phonemizer session, measured |
| **which rendering a listener prefers** | **NOT ESTABLISHED — needs ears** |

⚠ **THE LAST ROW IS THE WHOLE DECISION AND NO MEASUREMENT HERE SETTLES IT.** Everything above says
the symbol is out of distribution for the position and that the model responds to it; none of it says
the result is worse. A/B files were rendered per word (⟨ᵻ⟩, half a second of silence, ⟨ə⟩) for a
listener, because that is the only instrument that answers this.

## Where a fix would go, if the listen supports one

Not the phonemizer. Its ⟨ᵻ⟩ is evidenced for the inflectional environment, and changing the engine
would move 2,154 well-supported entries to fix a prefix problem. `KokoroFormat` is the right layer:
it already exists to translate canonical IPA into what Kokoro was trained on — ɝ→ɜɹ, ɚ→əɹ, ɐ→ə are
exactly this class of adjustment — and a rule scoped to the prefix onset leaves the inflectional
cases alone.

⚠ **Scope it to the position, never to the symbol.** A blanket ⟨ᵻ⟩→⟨ə⟩ would hit the 2,154 entries
where gold itself writes ⟨ᵻ⟩, which is the one place the symbol is known to be right.

**Nothing changed yet.** Pending the listen.


## Run 3 — 2026-09-22 — scoping the rule, because "position" is not uniformly safe either

The phonemizer session counted where gold's ⟨ᵻ⟩ actually lives — all 2,203 occurrences:

```
1,287  final -d, inflectional      əbɹˈiviˌAɾᵻd, dˈɛkəɹˌAɾᵻd
  691  word-internal               ˈAblᵻst, ˌæbsᵊntmˈIndᵻdli
  216  final -z, inflectional      kɹˈɪsməsᵻz, dɹˈuzᵻz
    9  PREFIX (de-/re-/pre-)
```

So the prefix position — the case for the rule — is **0.4%** of gold's ⟨ᵻ⟩.

⚠ **BUT THE 9 ARE NOT NOISE, THEY ARE ONE FAMILY**, and it is the family the engine's reduced class
was built around in the first place:

```
dedans deduce deducible deduct deductibility deductible deduction deductive deductively
```

A rule reading "prefix ⟨ᵻ⟩ → ⟨ə⟩" would be right 1,057 times and wrong 9 — and the 9 are exactly the
words where gold and our engine **already agree**, i.e. the strongest evidence in the entire class.
That earns an explicit carve-out rather than being accepted as rounding error: a rule that is wrong
precisely where the evidence is strongest is the wrong rule, whatever its hit rate.

### ⚠ And key the rule on the ONSET, not on what follows

The obvious cheap formulation — ⟨ᵻ⟩ *not* at end-of-word — would sweep up the 691 word-internal
cases, several of which are `-ᵻd-` inside a longer word (`mˈIndᵻdli`, `mˈIndᵻdnəs`). Those are gold's
own spelling and must not move. The rule has to key on *immediately after a de-/re-/pre- onset* and
on nothing about the right-hand context.

This is the same trap as the distinctness mistake earlier in the week, and worth naming as a
pattern: **the general form of a rule looks safer than the specific one, because its counterexamples
live in a different environment and so are invisible from where the rule is being written.**

### A correction carried in

Run 2's table cited "2,154 entries" for gold's inflectional ⟨ᵻ⟩, quoting `englishArpabet.ts`.
Recounted from `us_gold.json` directly: **2,203 total across all environments, of which 1,503 are the
inflectional -d/-z.** The 2,154 in that comment describes something slightly different. Corrected
above rather than left to be inherited by whatever rule gets written.

**Still pending the listen.** None of this changes what the decision rests on.

## Run 4 — 2026-09-22 — the carve-out cannot be expressed as "prefix position"

Checked the scoping against what THIS pipeline actually renders, rather than against gold.

**The carve-out is real here, not just in gold.** All nine of the family come out of our engine with
a prefix ⟨ᵻ⟩, so a naive prefix rule would move every one of them:

```
deduce dᵻdˈus      deduct dᵻdˈʌkt        deductible dᵻdˈʌktᵻbᵊl
deduction dᵻdˈʌkʃən  deductive dᵻdˈʌktəv   deducible dᵻdˈʌkᵻbəl
dedans dᵻdˈæns     deductively dᵻdˈʌktəvli  deductibility dᵻdˌʌktəbˈɪlᵻɾi
```

⚠ **AND THE KEEP AND CHANGE SETS ARE NOT PHONETICALLY SEPARABLE BY THE PREFIX.** `KokoroFormat` sees
an IPA string and has no word identity, and the two sets collide immediately:

```
keep    deduce      dᵻdˈus
change  determine   dᵻtˈɜɹmən
change  reduce      ɹᵻdˈus
```

`deduce` and `determine` share `dᵻ`; `deduce` and `reduce` share `ᵻd`. Neither the onset alone nor
the following segment alone separates them. **"Prefix ⟨ᵻ⟩ → ⟨ə⟩" is not implementable as stated** at
the layer that would implement it.

What does separate them is the pair: all nine of the family are `dᵻd`, and `reduce` is `ɹᵻd`. So a
rule conditioned on `dᵻd` as an exception is expressible — but it is a **lexical** carve-out wearing
a phonological costume, and it is only safe if no word in the 1,057 target set also renders `dᵻd`.
⚠ **That set lives upstream and cannot be checked from here** — asked for rather than assumed.

### The 691 word-internal cases are not at risk here, for an unexpected reason

The warning was that a rule keyed on "not at end-of-word" would sweep them up. It would — but in this
pipeline most of them never arrive as ⟨ᵻ⟩ at all. Where gold writes ⟨ᵻ⟩ we write ⟨ə⟩:

```
            gold                ours
ablest      ˈAblᵻst             ˈAbəlst
absentmindedly  ˌæbsᵊntmˈIndᵻdli  ˌæbsəntmˈIndədli
```

So there is nothing to protect in those rows — **and that is a divergence in the opposite direction,
separate from this one and not investigated here.** Noted rather than followed, because it is a
different question and this one is not settled yet.

The inflectional cases do arrive as ⟨ᵻ⟩ and do match gold (`abbreviated` → `əbɹˈiviˌATᵻd`,
`christmases` → `kɹˈɪsməsᵻz`), which is the set a prefix-keyed rule leaves alone by construction.

**Still pending the listen**, and now also pending a check that no target word renders `dᵻd`.

## Run 5 — 2026-09-22 — the affricate hazard disappears if the rule runs late

The phonemizer session measured the collision set: of the 1,057 target words, **11 match `dᵻd` as a
literal string and 0 match it with a plain /d/**. All 11 are `dᵻd͡ʒ` — degeneracy, degenerate,
deject, dejection and relatives — where the `d` is the first half of /d͡ʒ/, one segment written as
three code points (U+0064 U+0361 U+0292). Their proposed key was therefore `dᵻd` *not followed by
U+0361 U+0292*.

⚠ **That guard is unnecessary HERE, and the reason is the layer.** Their hazard is real for a rule
over canonical IPA; a rule at the end of `KokoroFormat.Render` never sees a tie bar, because the
replacement table strips it and collapses the affricate before anything else runs:

```csharp
("͡", ""),   // tie bar: d͡ʒ → dʒ, consumed just below
("dʒ", "ʤ"),
```

By the end of `Render` an affricate is a single ligature code point. Verified rather than reasoned:

```
judge  raw = d(U+0064) ͡(U+0361) ʒ(U+0292) ˈ ʌ d ͡ ʒ
judge  kokoro = ʤ(U+02A4) ˈ ʌ ʤ(U+02A4)
```

So on the string a Kokoro-side rule actually operates on, the two sets are already disjoint:

```
the 11 affricate collisions, after Render     0 of 11 contain `dᵻd`   (all are dᵻʤ)
the carve-out family, after Render           14 of 14 contain `dᵻd`
```

**`dᵻd` on the rendered string is sufficient, with no affricate guard.** Worth stating explicitly in
whatever gets built, because the guard would look like harmless belt-and-braces and would in fact be
dead code inviting someone to move the rule earlier.

⚠ **And the general hazard is still worth carrying**, because it is not specific to this rule: every
tie-barred segment in canonical IPA — `d͡ʒ t͡ʃ t͡s d͡ʑ` — has this shape, so any rule keyed on a bare
`d` or `t` that runs BEFORE the collapse matches half a segment. The phonemizer session has the same
class documented from the other end, where .NET's code-unit regexes split astral characters that
Node saw whole.

### The carve-out, final shape

```
keep,   gold attests ⟨ᵻ⟩ (7)   deduce deduct deductibility deductible deduction deductive deductively
keep,   no gold entry   (8)    deduced deducted deductibles deducting deductions deducts
                               dedeaux dedeurwaerder
change                         nothing collides
```

Six of the eight unattested are the same paradigm, so this is one family plus two surnames — a
lexical carve-out, and worth calling one rather than dressing it as phonology. The difference from
Run 4 is that it now carries a measured guarantee instead of resting on "the collision happens not
to occur".

### And the opposite-direction divergence is smaller than it looked

Run 4 flagged that we write ⟨ə⟩ where gold writes ⟨ᵻ⟩ word-internally. Counted upstream: of 989 gold
⟨ᵻ⟩ entries we also cover, we match on 966 and differ on 23. So it is 23 words, not the hundreds the
691 figure suggested, it does not threaten a prefix-keyed rule, and it is too small to be worth
chasing on its own. Closed rather than left open.

**Still pending the listen — that is now the only thing gating this.**

## Run 6 — 2026-09-22 — the constraint in Runs 4 and 5 was self-imposed

Runs 4 and 5 reasoned from "`KokoroFormat` sees an IPA string and has no word identity", and built a
`dᵻd` key plus an affricate analysis on top of it. That premise is true of `KokoroFormat` and **false
of the pipeline**, which was the wrong thing to take as a property of the problem.

`PhonemizeTrace` is the full word → IPA trip and it is what this repo already uses. `Phonemize` holds
a trace, the words, and `GroupSourceWords` — a group index → word index map — and it only accepts a
reading whose group count matches that map. So every rendered group can be tied to the word it came
from:

```
text   : We deduce that you determine the deductible to reduce dejection.
kokoro : wi dᵻdˈus ðæt ju dᵻtˈɜɹmən ðə dᵻdˈʌktᵻbᵊl tu ɹᵻdˈus dᵻʤˈɛkʃən.

group          source word    has ᵻ   carve-out?
dᵻdˈus         deduce          yes    KEEP (ded-)
dᵻtˈɜɹmən      determine       yes
dᵻdˈʌktᵻbᵊl    deductible      yes    KEEP (ded-)
ɹᵻdˈus         reduce          yes
dᵻʤˈɛkʃən      dejection       yes
```

**With the word in hand the carve-out is orthographic and exact.** Every one of the 16 keep-words
begins `ded-` — deduce, deduct and paradigm, dedans, dedeaux, dedeurwaerder — and the words that
forced the affricate analysis begin `deg-` and `dej-`. ⚠ **So the affricate hazard never arises**:
it was an artefact of matching IPA where the spelling was available, and `dejection` is excluded by
reading its first three letters rather than by guarding against U+0361.

### What this changes

- The key is the source word, not `dᵻd`. No IPA heuristic, no tie-bar guard, no dead belt-and-braces.
- The rule belongs where the map is — `Phonemize`, walking groups — not inside `Render`, which is the
  layer that genuinely lacks the context.
- ⚠ **And it degrades honestly**: `map` is null or count-mismatched on some readings, and there the
  rule simply does not run, leaving today's behaviour. A word-keyed rule that cannot identify the
  word must do nothing rather than guess.

### The lesson, which is the reusable part

Runs 4 and 5 are not wrong about `KokoroFormat` — they are wrong about which layer the question
belonged to, and every subsequent difficulty (the `dᵻd` collision, the affricate guard, calling it a
lexical carve-out in phonological costume) was manufactured by that choice. **The difficulty of a
rule is evidence about the layer, not only about the rule.** Two runs of increasingly careful work
went into making a bad layer safe, and none of it went into asking whether the information was
available one level up. It was, and this repo already depended on it for karaoke highlighting.

**Still pending the listen.** The design is simpler than it was an hour ago and no more decided.

## Run 7 — 2026-09-22 — the `ded-` key verified from the lexicon end

Checked upstream, in the direction that would actually break it:

```
target set (change)                     1,057
  target words starting `ded-`              0
keep set (gold attests ⟨ᵻ⟩)                 7
  keep words NOT starting `ded-`            0
```

No `ded-` word in the lexicon takes a prefix ⟨ᵻ⟩ except the deduce/deduct family and two surnames,
so the key sweeps in nothing that should change and misses nothing that should not. My 16 against
their 7 is everything we render `dᵻd` against the gold-attested subset — the extra are the rest of
the paradigm plus `dedeaux` and `dedeurwaerder`, which have no gold opinion either way and are
harmless to keep.

⚠ **AND A WORD-KEYED FIX HERE LEAVES EVERY OTHER CONSUMER OF THE LIBRARY UNTREATED.** The phonemizer
would keep emitting ⟨ᵻ⟩ in prefix position; only this repo would correct it. That is an argument for
eventually fixing it upstream as well, not for holding a fix here — recorded so that "we fixed it"
is not later mistaken for "it is fixed".

### The failure mode this exchange produced, which is worth more than the rule

Run 6 named one half: the difficulty of a rule is evidence about the layer, not only about the rule.
The phonemizer session named the other half, having done the same thing from the opposite side — it
accepted "no word identity" as a constraint and began optimising within it, without asking whether
the constraint was real.

⚠ **TWO SESSIONS CAN MANUFACTURE A HARD PROBLEM BETWEEN THEM BY EACH TRUSTING THE OTHER'S FRAMING.**
I asserted a constraint about my own code; they reasoned carefully inside it and produced a
1,057-word sweep, an affricate hazard and a tie-bar guard — all correct, all answering a question
that should not have been asked in that form. Neither of us checked the premise, because it came
from the side that owned the code. The measurement stands and is what makes "0 collisions" a fact
rather than a belief; the question it answered was the wrong one.

**Still pending the listen.** Nothing has been built, and the design has not changed since Run 6.

## Run 8 — 2026-09-22 — the listen, and the rule

**A listener preferred B** — the ⟨ə⟩ rendering — on a connected-reading sample, and called the
difference *"pretty subtle, overall, in this reading"*. That is the size of the claim this change
gets to make, and it is recorded here so nothing downstream inflates it.

The sample was built as the candidate fix rather than a blanket swap, so the listen answered the
actual decision:

```
Before we can determine the cause, we need to describe what happened and reduce
the variables. From there we can deduce the rest.

changed  determine  dᵻtˈɜɹmən  -> dətˈɜɹmən
changed  describe   dᵻskɹˈIb   -> dəskɹˈIb
changed  reduce     ɹᵻdˈus     -> ɹədˈus
KEPT     deduce     dᵻdˈus                    (ded- carve-out)
```

### The rule

`ReduceEnglishPrefixVowel`, in `KokoroPhonemizer.Phonemize`, walking groups against the
group → word map. Three conditions, each earned by an earlier run:

1. **The source word begins `de-`/`re-`/`pre-`** — the word, not the phonemes. Run 6.
2. **Except `ded-`** — one family plus two surnames, the only place gold attests a prefix ⟨ᵻ⟩, and
   verified from the lexicon at 0 collisions either way. Runs 5 and 7.
3. **The ⟨ᵻ⟩ must be the token's FIRST VOWEL.** `represent` is `ɹˌɛpɹᵻzˈɛnt`, whose ⟨ᵻ⟩ is a second
   syllable and nothing to do with the prefix. This one is new here — no earlier run caught it,
   because every example under discussion happened to have the prefix vowel first.

And it does nothing when the map is null or mismatched: a word-keyed rule that has lost the word must
leave the stream alone rather than guess.

⚠ **`dejected` → `dəʤˈɛktᵻd` is the case worth keeping in mind**: the prefix moves and the
inflectional ⟨ᵻ⟩ — the spelling gold itself uses 1,287 times — stays, in one word. Any rule phrased
as "replace ⟨ᵻ⟩" fails there.

### Tests, and what they can and cannot catch

Reverted the rule and ran them: **7 of the target-class cases fail, and the 10 boundary cases pass
either way.** That is correct and worth stating rather than glossing — the boundary cases assert
things are *unchanged*, so they cannot detect the rule's absence; their job is catching
over-application, which is the failure mode that would actually damage a reading.

⚠ **One expectation had to be weakened, and the reason generalises.** `dedans` is rare enough to go
through the neural OOV path, so its tail is `dᵻdˈæns` or `dᵻdˈænz` depending on whether that model
resolved — an exact expectation was testing the model's presence rather than this rule. It now
asserts only the prefix vowel. **Exact-IPA expectations for OOV words are unstable by construction**,
and that applies to any future test in this file.

Suite: 95 + 271 + 23 + 362, 0 failures.

### Still true, and not fixed by this

The phonemizer keeps emitting ⟨ᵻ⟩ in prefix position; only this repo corrects it. Any other consumer
of the library inherits the divergence untreated, which is an argument for eventually fixing it
upstream too — recorded so "we fixed it" is not later read as "it is fixed". vernacula-phonemizer#1445
holds the measurements either way.

`be-` (`before`, `become`, `begin`) shows the same ⟨ᵻ⟩ and is **out of scope**: nobody has measured
gold's spelling for that onset, so it is left alone rather than swept in on the strength of looking
similar.

## Run 9 — 2026-09-22 — review, and the control trap a second time

Four findings.

**The rule disagreed with `Render` about what English is.** `KokoroFormat.Render`'s English arm is
`lang is "en" or "en-GB" or "en-US" or null`; my gate omitted `null`. A call with a null language
would therefore render English phonemes and then skip the English rule on the same call. Aligned to
`Render`'s own definition rather than to a list I wrote from memory.

**`map[group++]` was unbounded.** The count check above it should make an overrun impossible, but
this runs on every synthesised paragraph and an index bug would take the utterance *down* rather
than degrade it. Out of range now means what it already means everywhere else in this rule — cannot
identify the word, so leave the token alone.

**`KokoroVowels.ToCharArray()` allocated on every prefix word.** Held as a static array.

**The tests did not check the output was still in vocabulary.** ⚠ audio.cpp REFUSES a supplied
stream carrying a symbol Kokoro has no id for — it does not drop it, it rejects the request — so a
rule emitting one would take out synthesis on that backend while the ONNX path silently skipped it.
Every case now asserts it.

### Both engines get this, which was worth confirming rather than assuming

`KokoroTts` (ONNX) calls `Phonemize(text, british)`, and `Lang(british)` returns `"en"`/`"en-GB"`,
so the rule applies on both backends. If it had applied to only one, the same document would read
differently depending on the engine — the exact failure `KokoroChunker` exists to prevent.

### ⚠ And the control was invalid, for the second time this week

The first attempt stashed only the uncommitted review fixes while the rule itself was already
committed, so the "control" ran against the rule and reported 17 green. **I wrote this trap up on
#240 and then walked into it again.** Redone against `main`:

```
7 failed, 10 passed
```

The lesson from #240 was "check what the control actually reverted". The lesson now is stronger:
*stash is the wrong instrument for this* whenever any part of the change is committed, and the
question "is the thing I am testing for absence actually absent?" needs an answer that does not
depend on remembering which half is in the index. Reverting a named file to `main` does not.

Suite: 95 + 271 + 23 + 362, 0 failures.

## Run 10 — 2026-09-22 — `pre-` should never have been in the rule

The phonemizer session counted gold's first-vowel spelling per onset, over exactly the words where
our ⟨ᵻ⟩ is itself the first vowel — i.e. the words this rule can reach:

```
be-    ə 160
de-    ə 468   i 30   ɪ 8   ᵻ 7   A 2   ɛ 1
re-    ə 557   i 12   ɪ 2   ɛ 4   A 1   ʌ 1
pre-   ə  32   i 46   ɪ 1
```

⚠ **`pre-` runs the other way, and it shipped in #243.** Gold's majority there is tense `i`, so the
lexicon argument this rule rests on does not hold for that onset. Verified against our own pipeline
rather than taken on the counts — the raw render against what the shipped rule produced:

```
precede      pɹᵻsˈid      -> pɹəsˈid      CHANGED
precise      pɹᵻsˈIs      -> pɹəsˈIs      CHANGED
predict      pɹᵻdˈɪkt     -> pɹədˈɪkt     CHANGED
preclude     pɹᵻklˈud     -> pɹəklˈud     CHANGED
prevent      pɹᵻvˈɛnt     -> pɹəvˈɛnt     CHANGED
predominant  pɹᵻdˈɑmənᵊnt -> pɹədˈɑmənᵊnt CHANGED
precipitate  pɹᵻsˈɪpɪtˌAt -> pɹəsˈɪpɪtˌAt CHANGED
prefer       pɹᵻfˈɜɹ      -> pɹəfˈɜɹ      CHANGED
```

Eight words moved from one non-gold spelling to a different non-gold spelling. ⚠ **AND IT SPLIT A
PARADIGM**, which the counts alone would not have shown:

```
prefer      pɹᵻfˈɜɹ  -> pɹəfˈɜɹ
preferred   pɹifˈɜɹd    pɹifˈɜɹd   (untouched: its first vowel is already tense i)
```

One stem, two prefix vowels, introduced by this rule.

**The listener sample never covered it.** determine, describe, reduce — all `de-`/`re-`. `pre-` was
carried along because it looks like the same class, which is the entire error: the onsets were
treated as a group because they are spelled alike, and the evidence was never per-onset until now.

Removed. The rule is `de-`/`re-` only, and the exclusions are pinned by tests that compare against
the render with no rule applied, so they keep holding if the underlying dictionary reading changes.

### `be-` stays out, on the opposite evidence

Gold is **unanimous** for `be-` — ə 160 times, no counterexample, stronger than either onset in the
rule. It is excluded anyway, because no listener has heard it and this change has been driven by
synthesis rather than by distribution from the beginning. ⚠ **Holding it out is the same discipline
that should have kept `pre-` out**, and it is worth noticing that the discipline was applied to the
onset with the best evidence and skipped for the one with the worst.

### A correction carried in

Their first pass read gold with a plain-IPA vowel class and missed misaki's compact symbols — `A I O
W Y` are vowels too, so `regime ɹAʒˈim` scored as tense `i` when its first vowel is `A`. The table
above is the corrected run, and the error ran in the direction that would have *strengthened* the
`pre-` case they were making. Said plainly rather than quietly re-run, which is the reason the
numbers are usable at all.

### An independent confirmation of the carve-out

⟨ᵻ⟩ survives in gold on exactly 7 `de-` words, and they are precisely the `ded-` family — the
keep-set falling out of a measurement that was not looking for it.

Suite: 95 + 279 + 23 + 362, 0 failures.
