# Kokoro on vernacula-phonemizer — investigation log

Kokoro's G2P frontend was the pure-C# espeak-ng port in `external/espeak-ng-portable`
(`Phonemize.Run` → `KokoroFormat.Render`, docs/kokoro_onnx_investigation.md Runs 9–12). That
submodule is private and is not going to be published; vernacula-phonemizer is. This log is the
move: one Kokoro-alphabet render target over vernacula-phonemizer's canonical IPA, and the
submodule gone.

## Run 1 — 2026-09-05 ~07:50 — what do the two engines actually emit?

Two throwaway console apps (scratchpad; they cannot share a process because both submodules
build an assembly called `Vernacula.Phonemizer`) over 14 sentences chosen for the things a
Kokoro mapping has to get right: the five diphthongs, NURSE/lettER, flaps, syllabic consonants,
affricates, numbers/currency/dates, quotes and dashes, en-GB vs en-US.

Old: `Phonemize.RunWithSourceWords` + `KokoroFormat.Render`. New: `PhonemizeAsync` (`en`,
`en-GB`) and `PhonemizeTrace` (for spans).

Where vernacula-phonemizer's IPA differs from espeak's, i.e. what a new render target has to
re-key:

| feature | espeak-ng-portable | vernacula-phonemizer |
|---|---|---|
| offglides | digraphs `oʊ eɪ aɪ aʊ ɔɪ`, GB `əʊ` | superscript `oᶷ eᶦ aᶦ aᶷ ɔᶦ`, GB `əᶷ` |
| affricates / aspiration / l | `dʒ tʃ t l` | `d͡ʒ t͡ʃ tʰ ɫ` |
| en-us flap | `ɾ`, or `ʔn̩` for button | `t̬` and `d̬` (voicing diacritic) |
| NURSE / lettER | `ɜː` / `ɚ` | `ɝ` / `ɚ` |
| clause punctuation | collapsed to `\n`, re-injected from source text | kept, as its own token: `dˈɔːɡ , dˈʌzənt` |
| en-GB SQUARE | `eə` (rendered `Aə` — the `e→A` rule fired: a bug) | `ɛə` |
| palatal glide | `ðɪʲ` | `iʲə` in uranium |

`ᵻ` appears in both (remember); secondary stress `ˌ` is rarer in the new stream (espeak put
one on "over", "about"). `PhonemizeTrace` reports `Traced=true` for both `en` and `en-GB`, with
`InputSpan` (into the caller's text) and `IpaSpan` (into the IPA) on every token, including the
normalizer's expansions: `$3.14` → three tokens all with input span [23,28), `24` → one token
emitting two groups. That is a better source-word map than the old engine's, which only counted.

The neural (`PhonemizeAsync`) and traced (sync) readings differ only on OOV words — "Kokoro"
reads `koᶷkʰˈɔːɹoᶷ` vs `kʰˈɑːkʰɔːɹoᶷ` — never in word count, on this set.

## Run 2 — 2026-09-05 ~08:10 — the new render target, old vs new word parity

`Vernacula.Tts.Base.KokoroFormat.Render` re-keyed on the table above (tie bar, aspiration, ʲ
dropped; `t̬→T`, `d̬→d`; `ɫ→l`; superscript diphthongs → `O A I W Y`, GB `əᶷ→Q`; leftover
`ᶦ ᶷ → ɪ ʊ`; `ɝ→ɜɹ`, `ɚ→əɹ`; GB `ɛə→ɛː`; en-us strips `ː`; detached punctuation re-attached
to the preceding word). Compared to the old engine's Kokoro output, per word, punctuation
stripped, over the 14 sentences × 2 accents:

    word parity old-vs-new: 137/178 = 77.0%

Every one of the 41 differences is the phonemizer's reading, not the rendering; sorted:

- **new is closer to misaki's lexicon** (which is what Kokoro heard in training): `dˈɔɡ` (old
  `dˈɑɡ`), `mˈWntən`, `jəɹˈAniəm` (old `jʊɹɹˈAniəm`, a doubled r), `ˈɛləmənts`, `bˈʌTən` (old
  `bˈʌtn` from `ʔn̩`), GB `ðˈɛː`/`ʃˈɛː` (old `ðˈAə` — the bug above), `ˈA ˈɛm` for a.m. (old
  `ə ˈɛm`), `ˈɪzənt ðæt ðə` (old ran `ðætðə` together: 7 groups for 8 words).
- **arguable either way**: `wˈɑz`/`ðæn`/`ənd` — vernacula-phonemizer gives citation forms where
  espeak gave the reduced `wʌz`/`ðən`/`ænd`; `lˈɛŋkθ`; `ɪɡzˈæmpəl`; `θˈɑɹiəm`; no `ˌ` on "over".
- **group-count differences**, all from normalization: `$3.14` old "three dollars and fourteen
  cents" (18 groups) vs new "three dollars fourteen" (16) — worth an upstream issue, the "and …
  cents" is the natural reading; `2024` old "two thousand twenty four" vs new "twenty twenty
  four"; `Mr.` both "mister".

Render-level checks all hold: every output codepoint over the set is in `KokoroVocab` (the
test `EveryOutputCodepointIsInTheVocab` fixes three of the sentences), and the Kokoro
alphabet symbols land where misaki puts them (`ʧ ʤ T O A I W Y Q ɜɹ əɹ ɛː ᵊ`-less).

## Run 3 — 2026-09-05 ~08:30 — source-word map, audio, the CUDA runtime, and CI

**Map.** `KokoroPhonemizer.Phonemize` builds the group→source-word map from `PhonemizeTrace`
(each token's `InputSpan.Start` → the whitespace-delimited word containing it, repeated once per
group in its `IpaSpan`) and applies it to the `PhonemizeAsync` reading when the two have the same
number of spoken groups, else to the traced reading. First pass mapped `Mr. Smith` to `0,0`:
both tokens carry the input span [0,9), which covers two written words. Fixed by advancing to
the next word inside a shared span; `$3.14` stays `5,5,5` because its span is one word.

    Mr. Smith arrived at 10:30 a.m. on Tuesday, March 3rd, 2024.   0,1,2,3,4,4,5,5,6,7,8,9,10,10,10
    I thought about it for $3.14 and the 2nd time — really… she said "no".   0,1,2,3,4,5,5,5,6,7,8,9,11,12,13,14

(Word 10 "—" and word 10 "…" are unpronounceable; `SpeakAligned` gives them zero-length markers.)
All 14 sentences map; no fallbacks.

**Audio.** `vernacula-tts-backends --backend kokoro` on CPU, data dir auto-resolved from the
submodule: af_heart 54 tokens → 3.8 s in 750 ms (5.1× real-time); bf_emma with the GB reading
3.0 s in 652 ms. Both play.

**The CUDA runtime.** vernacula-phonemizer references the plain CPU `Microsoft.ML.OnnxRuntime`
and it now flows through `Vernacula.Tts.Base` into every consumer. Checked rather than assumed:
rebuilt `Vernacula.Tts.Backends.CLI` with `-p:EP=Cuda` after deleting its `runtimes/`, and the
shipped `libonnxruntime.so` is sha `1aacefdf…` = the `microsoft.ml.onnxruntime.gpu.linux`
package's, not the CPU package's `d132535d…`; `libonnxruntime_providers_cuda.so` present. The
direct `ExcludeAssets="all"` reference (the trick Vernacula.Tts.CLI already used) went into the
three Cuda consumers of Kokoro: Backends.CLI, Avalonia, KokoroPerf.

**CI simulation.** Deleted the four test projects' `bin/`, ran the workflow's two commands
(`dotnet build Vernacula.slnx -p:EP=Cpu`; `dotnet test <proj> --no-build -p:Platform=x64`):
22 / 55 (+4 skipped) / 63 / 23, all passed. Two false alarms on the way, both worth knowing:
`dotnet build` of the solution wrote nothing for `Vernacula.Tts.Tests` at all, and the cause was
that **something is rewriting `Vernacula.slnx` while I work** — twice now the four renamed
projects were silently dropped from it (an editor with the pre-rename solution open, most
likely). `git checkout -- Vernacula.slnx` and the step passes. Also `Assert.Skip`-gated tests
here need the submodule's `data/`, which the workflow now checks out.

**Removed.** `external/espeak-ng-portable` (submodule deinit + `git rm`), every ProjectReference
to it, and the `--data-dir`-required check in the Backends CLI (the data dir now resolves from
the vernacula-phonemizer submodule; the reader re-resolves a saved pre-migration path). CI
needs only vernacula-phonemizer now — which is the repo being made public.

Not done: `$3.14 → "and … cents"` upstream; a re-listen of the reader's word highlighting with
the new map (it is exercised by `KokoroPhonemizerTests`, not by ear).

## Cross-language rendering — 2026-09-15

`KokoroFormat.Render` was written for English and applied to English, so nobody
had asked what it does to the other languages Kokoro speaks. It does damage, and
the damage is invisible to the obvious check.

### The measurement that missed it

First pass asked "is every rendered symbol in Kokoro's vocabulary?" and reported
**0.00% out-of-vocab for es, fr, it** — clean. That is membership, not
correctness, and the two come apart completely: a rule can rewrite one in-vocab
symbol into another in-vocab symbol and destroy a phonemic contrast while the
number stays green.

Asking instead which rules FIRE, over 60 golden sentences per language:

| lang | rule | count | what it destroys |
|---|---|---|---|
| es | `x→k` | 62 | the jota — *jamón* becomes *kamón* |
| es | `r→ɹ` / `ɾ→T` | 59 / 458 | trill and tap merge; *perro* and *pero* stop contrasting |
| fr | strip `̃` | 400 | nasal vowels, phonemic in French |
| it | `r→ɹ` | 502 | the trill |
| hi | strip `ʰ` | 125 | aspiration — PHONEMIC in Hindi (क/ख), allophonic in English |
| pt-BR | `ɐ→ə` / strip `̃` | 382 / 271 | a phonemic vowel, and the nasals |

Plus two unconditional rules further down: `o→ɔ`, which merges Portuguese *avô*
/o/ with *avó* /ɔ/, and dropping `ː`, which is phonemic length in Hindi.

⚠ **Every one of those rewrites a symbol Kokoro's vocabulary already carries.**
`r`, `ɹ`, `ɾ`, `x`, `ɐ` and U+0303 are all in it. The collapses are English
conveniences — English has no trill, and its `ɾ` really is an allophone of /t/ —
not limits of the model.

### The split

Two paths rather than one parameterised one. The English path is left EXACTLY as
it was, because it is correct and byte-verified; the other languages get the
alphabet conventions without the allophone collapses:

- **AlphabetConventions** — tie bar, offglide diphthongs, `dʒ→ʤ`, `tʃ→ʧ`. These
  are properties of the alphabet Kokoro was trained on, not of English:
  audio.cpp applies the same tie-collapsing table to every eSpeak language it
  drives (`espeak_text()`).
- **LanguageRules** — only what Kokoro's alphabet genuinely cannot carry. Hindi
  alone so far: `ɦ→h`, `ʱ→ʰ`, and the dental bridge dropped.
- **DecomposeUnknown** — a codepoint with no id, decomposed if Unicode yields
  pieces the vocabulary does hold. This is the whole of Portuguese: we write
  nasal vowels precomposed (õ ĩ ũ ẽ), Kokoro carries base + U+0303. All or
  nothing, since half a decomposition is a different sound.

Result: **all seven eSpeak-driven languages clean, English byte-identical**
(114/114 rows on both en and en-GB via the old and new entry points), and the
contrasts restored:

```
es    : pˈeɾo el pˈero ... xamˈon.     tap vs trill; jota survives
it    : kˈorre vˈerso
fr    : œ̃ bɔ̃ vɛ̃ blˈɑ̃
pt-BR : o avˈo e a avˈɔ ... estˈɐ̃w̃
hi    : pˈʊlɪs nˈeː kˈəhaː
```

### ja and zh are a different problem — NOT done

| lang | out of vocab | missing |
|---|---|---|
| ja | 23.96% | `ä`×872, `̞`×1139, `ꜜ`×222, `ʑ`×72 |
| cmn | 38.71% | tone letters `˥˦˧˨˩`×4348, `ʐ`×180, `ᵘ`/`ⁱ`, `̩` |

These need mapping TABLES, not the removal of collapses. The target is knowable
exactly rather than guessable — the bundled multilingual GGUF carries the
engine's own tables, and their output inventories are:

```
ja : abdehijkmnopstvzçɕɡɨɯɲɸɾʣʥʦʨʲβᵝ   (no ː, and NO pitch-accent marks)
zh : aefhijklmnopstuwxyŋɔɕəɚɛɤɥɨɻʂʦʨʰ→↓↗↘ꭧ
     a1 -> a→   a2 -> a↗   a3 -> a↓   a5 -> a
```

So Mandarin tone letters map onto the arrows Kokoro carries for exactly this,
and Japanese pitch accent has nowhere to go — Kokoro's Japanese does not encode
it. Cantonese is irrelevant: Kokoro's `zh` is Mandarin.

### ja and zh — done, and scored rather than eyeballed

Both now render clean. Japanese was mostly mechanical: affricates become the
single ligatures their kana table uses (つ = ʦɨ, ち = ʨi, じ = ʥi), the
centralised ä and the lowering diacritic go, ɴ becomes n, and a bare ʑ — which
is NOT in the vocabulary although ɕ is — becomes ʥ, because that is how their
table spells じ. Matching what the model was trained on beats matching the IPA
more closely.

⚠ Japanese pitch accent is dropped. Kokoro's Japanese does not encode it: no
downstep anywhere in the kana table, and none of → ↓ ↗ ↘, which the vocabulary
carries for Mandarin tone. Lossy and deliberate — inventing a token the model
never saw in Japanese would be worse than losing a distinction it never learned.

Mandarin needed a structural transform, not a table: **the tone mark goes after
the nucleus, not after the syllable** (`zhong1 = ꭧʊ→ŋ`, `yan1 = jɛ→n`), where
our transcription writes tone letters at the end. So the contour is lifted off
and reinserted after the last vowel, with 55 → `→`, 35 → `↗`, 214 → `↓`,
51 → `↘`, and the neutral tone carrying no mark.

**Scored, because eyeballing is what produced the 0.00%-clean mistake above.**
993 syllables from the golden corpus, their pinyin from pypinyin and their
phonemes from the engine's own g2p/zh.json:

| | exact |
|---|---|
| first attempt, rules reasoned from the inventory | 627/993 (63.1%) |
| after the classes the score named | 894/993 (90.0%) |
| after refining two over-applied rules | 978/993 (98.5%) |
| after fixing an ordering bug of my own | 991/993 (99.8%) |

Every step came from the score naming a mismatch class, not from thinking
harder. Two of the four rounds fixed rules I had just written: a glide-dropping
rule that also ate the onset of `wang`, and a `yə → y` replace keyed on a symbol
the loop below it had not yet produced. Both were invisible to the in-vocab
check and obvious to the score.

The last 2 of 993 are `-iong` syllables (穷, 熊), where our transcription differs
structurally rather than by a symbol. Left alone.

**Result: all nine of Kokoro's languages render clean, English byte-identical.**

```
en-us 10,905  en-gb 11,143  es 14,351  fr-fr 10,826  hi 15,661
it    14,809  pt-br 12,320  ja 10,738  zh   13,447      all 0.00%
```
