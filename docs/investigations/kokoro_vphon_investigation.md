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

## Run 12 — 2026-09-16 09:47 — misaki vs our render on a real document, then on misaki's whole lexicon

Every earlier run scored the render against sentences *we* chose. This one starts from a
document nobody wrote for a test: a business/technical response document in markdown, ~21k
characters, already run through the app's TTS path (56 blocks — 31 paragraphs, 22 list items,
3 headings — 1,951 words). Abstracted here on purpose; the content is day-job material and
none of it belongs in this repo.

Question: where do misaki and `vernacula-phonemizer → KokoroFormat` diverge on ordinary prose,
and does the divergence name any defect worth fixing?

Setup: misaki 0.9.4 in a venv on /mnt/data, `en.G2P(trf=False, british=False)`. Our side is a
throwaway console app over `KokoroPhonemizer.Phonemize`, which is exactly what the app calls.
Both sides get the same per-block text; results are aligned per whitespace-delimited source word
using misaki's token offsets and our `GroupSourceWords`. All 56 blocks aligned, 1,951 slots.

**First attempt was wrong and the numbers from it are discarded.** I ran misaki with
`fallback=None`, so every out-of-vocabulary word came back *empty* — 20 of them — and I nearly
wrote that up as "misaki silently drops unknown words". It does no such thing; Kokoro's real
pipeline passes `EspeakFallback`, which I had omitted. With the fallback in place the
misaki-empty class disappears entirely. Lesson: when the reference implementation looks
catastrophically broken, suspect the harness first.

    word slots 1,951   identical 1,156 (59.3%)   differing 795 (40.7%)

Classified, largest first:

| n | class | example (M = misaki, V = ours) |
|---|---|---|
| 149 | stress placement only | `You` M `jˌu` V `ju` |
| 92 | segmental | `January` M `ʤˈænjəwˌɛɹi` V `ʤˈænjuˌɛɹi` |
| 71 | `ə`→`ᵻ` | `details` M `dətˈAlz` V `dᵻtˈAlz` |
| 55 | `æ`→`ə` | `and` M `ænd` V `ənd` |
| 53 | `ᵊ`→`ə` | `Measurement` M `mˈɛʒəɹmᵊnt` V `mˈɛʒəɹmənt` |
| 42/17 | `to` unreduced | M `tə`/`tʊ` V `tu` |
| 40 | `ɐ`→`ə` | `a` M `ɐ` V `ə` |
| 33 | word count | `2025` M "twenty twenty-five" V "two thousand twenty five" |
| 20 | grouping | an all-caps initialism, M one group, V three |
| 14 | `T`→`ɾ` | the word-final flap rule, deliberate |
| 5 | we emit nothing | `→` M "right arrow" V *(silence)* |

Most of that is prosody or free variation. Three things in it are ours and wrong, and the
document was too small to size any of them — so the same comparison was re-run against
misaki's own gold lexicon, which is the thing Kokoro was actually trained on.

### The lexicon sweep

`us_gold.json`, 90,201 entries, 80,222 of them single-reading lowercase words. Our side ran
over all of them (12 shards in parallel — the first, serial, pass took 5m45 of pure wall time
for a *seventh* of the words, which was a waste). Both sides normalised the same way before
comparison, including misaki's own v1.0 rewrite (below).

    80,222 words   exact 27,670 (34.5%)   ignoring stress 36,639 (45.7%)

That is a lexicon-vs-lexicon number, not a quality score — much of the gap is legitimate
variation, and the reference has its own noise (`legal` plain vs `illegal` syllabic;
`national` `nˈæʃənᵊl` vs `international` `ˌɪntəɹnˈæʃᵊnəl`, which place the syllabic schwa on
different syllables; `ɪllˈiɡᵊl` and `ɪmmˈɔɹɾᵊl` carry doubled consonants). What the sweep is
good for is sizing specific classes.

**1. `-able`/`-ible` read as "-AY-bul" — ⚠ MEASURED ON THE WRONG PATH. See Run 13; the
numbers below describe the n-gram fallback, which the app does not run.**

    -able/-ible words where gold reduces the suffix   1,183
       we reduce it too (correct)                       790
       we read FACE instead                             393  (33.2%)

       of the 393 broken:  0%  are in g2p-dict.tsv
       of the 790 correct: 53% are in g2p-dict.tsv

Zero versus fifty-three percent. The dictionary is fine; the **letter-to-sound fallback** reads
a final `-able` as `eᶦbəɫ`.

⚠ The sentence that used to stand here — "`PhonemizeTrace` and `PhonemizeAsync` return the
identical string, so this is rules, not the BiLSTM guesser" — was true of my harness and false
of the phonemizer. The two agreed because my console app had **no ONNX Runtime at all**, so the
tagger silently failed to load and the async entry returned the sync reading. With the tagger
present the BiLSTM fixes essentially all of this. Run 13 has the corrected numbers.

**2. The syllabic schwa `ᵊ` is never emitted at all.**

    gold entries containing ᵊ            7,778  (9.7%)
    our outputs containing ᵊ                 0  (of 80,222)

`ᵊ` is Kokoro vocab id **42** — a trained token, not an exotic. In gold it appears only before
`l` (5,441), `n` (2,449) and `m` (8): the syllabic-consonant slot, where we write plain `ə`.
Not cleanly derivable, though — neither phonetic context (`l`+word-end is 3,280 syllabic vs
1,809 plain) nor spelling (`-al` is 1,309 vs 1,157) separates the two, and the reference is
partly idiosyncratic. So this is worth adopting in the consistent positions, not worth chasing
to byte parity.

**3. Unstressed initial vowels are not reduced: 664 of 2,365 (28.1%)** where gold opens with
`ə` — `acceptable` `əksˈɛptəbᵊl` vs our `æksˈɛptəbəl`, `abhorrent` `əbhˈɔɹənt` vs `æbhˈɔɹənt`.

Smaller, still real: **954 words carry a spurious secondary stress on a final vowel**
(`ability` gold `əbˈɪləTi`, ours `əbˈɪlᵻTˌi`); **45 doubled rhotics** at morpheme seams
(`underreporting` → `ˈʌndəɹɹɪpˈɔɹTɪŋ`, the `jʊɹɹˈAniəm` shape from Run 1 again); **49 `-ize`
words** read with KIT for PRICE. And `-ity` takes `ᵻ` where gold takes `ə` in 1,074 of 1,329 —
systematic, but `ᵻ` is vocab 177 and a near-schwa, so it is a difference rather than an error.

Isolated words and in-sentence readings were checked against each other for the top classes and
are byte-identical, so none of this is an artifact of probing single words.

### One thing the sweep changed my mind about

misaki's gold lexicon writes the flap as `ɾ` (`sˈɪɾi`, `kəpˈæsəɾi`), but `en.py:710` rewrites
**every** `ɾ` to `T` unless `version == '2.0'`, and the default is `None`. So Kokoro v1.0's
English training stream contains `T` in these positions and essentially never `ɾ`, even though
`ɾ` is vocab 125.

Our word-final flap rule deliberately emits `ɾ` there, and
kokoro_word_final_flap_investigation.md measured that it sounds better. That measurement
stands — it was audio, not symbols. But the *reason* may not be the one recorded: `ɾ` may be
behaving well word-finally because it is a weakly-trained token that the duration predictor
under-allocates, rather than because it is the right symbol. Worth knowing before that rule is
extended anywhere else on the theory that `ɾ` is "more correct".

### Where this leaves things

The document comparison was worth doing, but only as a way of finding classes — its 1,951 words
could not size any of them, and its headline 59.3% is mostly prosody and free variation. The
lexicon sweep is what turned "`auditable` sounds wrong" into "33.2% of `-able` words, entirely
in the letter-to-sound fallback, zero of them in the dictionary".

Not fixed here; this run is the measurement. ⚠ The fix list that stood here named the `-able`
fallback rule first; Run 13 withdraws it. The classes measured on the dictionary — the `ᵊ` gap,
the spurious final secondary stress, `seasonality`-style stress — are unaffected by that
correction, because dictionary entries never reach either fallback.

## Run 13 — 2026-09-16 10:15 — the harness had no ONNX Runtime, so Run 12 measured the fallback

Run 12's headline defect is withdrawn. The finding that `-able` words are misread came from a
console app that shipped **no ONNX Runtime at all**, so the English BiLSTM tagger could not
load, `PhonemizeAsync` silently degraded to the n-gram sync path, and the two entries returning
identical strings — which I read as "the neural path agrees, so this is a rule defect" — was
just both calls landing in the same fallback.

Why the harness was built that way: it referenced `Vernacula.Tts.Base`, which puts
`ExcludeAssets="native"` on its phonemizer edge and marks its own ORT package
`PrivateAssets="all"`. Both are deliberate and correct for the app (they are what stops a CPU
`libonnxruntime.so` landing beside the GPU one, #131). Their combined effect on a *new
consumer* is that no ORT arrives at all. `find bin/ -name '*onnxruntime*'` returned nothing.

Re-run with a direct project reference so the native ships, over the same 80,222 words:

| | exact vs gold | OOV words only | `-able`/`-ible` read as FACE |
|---|---|---|---|
| n-gram (Run 12 measured this) | 34.5% | 19.9% | 393 / 1,359 (28.9%) |
| **neural (the app runs this)** | **41.1%** | **31.0%** | **3 / 1,359 (0.2%)** |

The BiLSTM fixes every probe word individually too: `auditable` `ˈɔːd̬it̬ˌeᶦbəɫ` → `ˈɔːd̬ət̬əbəɫ`,
`unlinked` `ənlˈaᶦŋkt` → `ənlˈɪŋkt`, `unlinking` `ənlˈaᶦŋkɪnd͡ʒ` → `ənlˈɪŋkɪŋ`, `totalizer`
`tʰˈɑːt̬ɑːliʲɚ` → `tʰˈoᶷt̬əlˌaᶦzɚ`, `metadata` `mˈiːt̬əd̬ˌeᶦt̬ə` → `mˈɛt̬əd̬ˌɑːt̬ə`. That last one
now matches misaki's `mˈɛTədˌATə` exactly. The OOV improvement, 19.9% → 31.0% exact, is the
tagger's documented "roughly halves OOV error" showing up end-to-end.

`seasonality`, `underreporting` and `evidentiary` are unchanged between the paths, because they
are **in** `g2p-dict.tsv` — those remain dictionary defects and Run 12's readings of them stand.

### Which path does the app actually take?

Checked every call site rather than assuming. Synthesis is on the neural entry everywhere:

- `KokoroPhonemizer.Phonemize` → `PhonemizeAsync` (Kokoro ONNX backend)
- `OmniVoiceIpaTts.cs:91` → `PhonemizeAsync`
- `Vernacula.Tts.CLI/Program.cs:308,311` → `PhonemizeAsync`

Three sync call sites remain, and all three are **annotation, not speech**:

- `TtsEngine.cs:409` — the AudioCpp-Kokoro backend's `CreatePhonemizer`, whose own docstring
  already says it is "informative rather than a record of what was spoken" (audio.cpp
  phonemizes internally with eSpeak).
- `TtsExportService.cs:114,119` — the export CSV's `ipa` column.
- `IpaAnnotator.cs:53` — `PhonemizeTrace`, which is the only entry that reports spans, so the
  reader's per-word highlighting needs it.

So nothing *speaks* from the n-gram path. But the IPA shown above a word in the reader, and the
`ipa` column in an export, are produced by it — which means they can disagree with what was
synthesised, by about 11 points of OOV accuracy. `KokoroPhonemizer` already handles the
shape-mismatch case (it falls back to the traced reading when the group counts differ); the
annotation sites have no such reconciliation because they never see the neural reading at all.

### The part worth fixing

`Languages/English/EnglishTagger.cs:34` and `:43` are bare `catch { return null; }`. The second
one wraps `Onnx.LoadOrt`, which goes to real trouble to throw a diagnosable error —

    English neural OOV G2P needs the ONNX runtime (Microsoft.ML.OnnxRuntime), whose native
    library failed to load: <reason>

— and that message is discarded. `PhonemizeEnNeural` then sees `tagger is null` and returns the
sync path (`EnglishNeural.cs:68`). A caller cannot distinguish "neural reading" from "the model
never loaded", and neither can a test. The degradation is documented ("silent no-op without a
model / `onnxruntime-node`") and intentional as a *policy* — the phonemizer should not take an
utterance down over a missing optional model — but silence about *which* path ran is separable
from that, and is what cost this investigation a full sweep and a wrong conclusion.

Worth having: something like `EnglishNeural.TaggerAvailable` (or a reason string), so callers
that care can assert it, the CLIs can say it under `--verbose`, and a test can pin that the
shipped desktop app really does get the neural path rather than merely referencing a package
that provides it.

### Standing methodology note

Both times this investigation has gone wrong it was the harness, not the code under test:
Run 12 opened with misaki looking like it dropped OOV words (I had passed `fallback=None`), and
closed with our engine looking like it had a rule defect (I had shipped no ONNX Runtime). In
both cases the reference or the subject looked *implausibly* bad and I wrote it up anyway. The
check that would have caught both, and costs nothing: before measuring a difference between two
engines, verify each one is running the configuration you believe it is.

## Run 14 — 2026-09-16 12:05 — every divergence re-measured on the neural path, and triaged

Run 12 sized its classes on the n-gram fallback, so **every number in it was measured against a
path the app does not run** — including the 1,951-word document comparison. Re-run on the neural
entry, with the fixes from #1315/#1316 in place.

The document barely moves: 59.3% → **59.1%** identical, 12 of 56 blocks changed. Its words are
mostly in the dictionary, so the OOV path hardly touches it. The lexicon sweep is where the
classes live.

    80,222 words          exact   ignoring stress
      all                 41.1%        50.9%
      in-dictionary       55.9%        66.5%
      OOV (BiLSTM)        31.0%        40.3%

What the neural path fixed on its own, no work needed: `-able`/`-ible` 393 → **3**; `-ize` with
KIT 49 → **0**; the spurious trailing secondary stress 954 → 275; the doubled rhotic 45 → 21.

Classes were then found by **aligning gold against ours per word and counting every edit
operation**, rather than only re-testing the hypotheses Run 12 happened to form. Two of the top
edits (`insert ɪk`, `delete kᵊ`) turned out to be one alignment artifact of `-ical` (`əkᵊl` vs
`ɪkəl`) and collapse into the classes below; three were new.

### Defects

| n | class | evidence |
|---|---|---|
| ~~1,109~~ **FIXED** | ~~flap before a SECONDARY-stressed vowel~~ — vernacula-phonemizer#1317. Flap errors 2,214 → 1,290, spurious flap tokens 1,785 → 796. The guard now reads the dictionary's own stress digit; `thirty` turned out to be a bad dict row (the only decade written IY2) and was fixed as one. Reading the POST-CLASH stress instead was implemented and rejected: better on whole-word exact, worse on flaps, and it flapped compounds (`sawtooth` → *sˈɔTuθ). docs/investigations/en/en_flap_secondary_stress_investigation.md | we do it 1,112×, gold 47×, and gold agrees with us on **0.4%** of the words. American flapping requires the FOLLOWING vowel to be unstressed. We already never flap before PRIMARY stress (0 occurrences, gold 16) — so the rule has one guard and is missing its twin. 33% dict-sourced, so both the accent lexicon and the rules emit it |
| **16 symbols** | silently read as nothing where misaki speaks them | `→ ← ↑ ↓ × ∞ µ § ¶ © ® ™ € ¥ √ ∑`. Dropping input is worse than mispronouncing it. (`⇒ ÷ † £ · • ∆` are dropped by BOTH — `£` is a currency gap in each) |
| 275 | spurious secondary stress on a final vowel | `ability` `əbˈɪlᵻTˌi`, gold `əbˈɪləTi`. 50% dict-sourced |
| 21 | doubled rhotic at a morpheme seam | `underreporting` `ˈʌndəɹɹɪpˈɔɹTɪŋ`. The `jʊɹɹˈAniəm` shape from Run 1 |
| a few | dictionary stress rows | `seasonality` has primary and secondary INVERTED against gold; `underreporting` carries two primaries. Unaffected by either fallback — these are `g2p-dict.tsv` rows |

### Intentional — divergence is the point, leave them

| n | class | why |
|---|---|---|
| 1,847 | word-final flap emitted as the tap `ɾ`, not `T` | kokoro_word_final_flap_investigation.md, measured on AUDIO. ⚠ Carries Run 13's caveat: misaki v1.0 rewrites every `ɾ` to `T` (en.py:710), so `ɾ` is essentially unseen in Kokoro's English training stream. The audio result stands; the recorded REASON may not |
| — | `r ɾ x ɐ` and the nasal tilde kept in non-English | KokoroFormat's AlphabetConventions. Kokoro's vocabulary carries all of them; collapsing them is an English convenience that merges `perro`/`pero` and strips phonemic aspiration from Hindi |
| — | Japanese pitch accent dropped | Kokoro's Japanese never encoded it |
| — | clause punctuation kept as its own token, re-attached | it is what Kokoro pauses on; reconstructing it from source text was the old engine's problem |

### Open — divergent, defensible both ways, needs a call

| n | class | the tension |
|---|---|---|
| **7,778** | the syllabic schwa `ᵊ` is **never emitted** (0 of 80,222; gold uses it in 9.7%) | vocab id **42**, a trained token. But not derivable: neither phonetic context (`l`+word-end is 3,280 syllabic vs 1,809 plain) nor spelling (`-al` splits 1,309/1,157) separates it, and the reference contradicts itself — `legal` plain vs `illegal` syllabic, `national` `nˈæʃənᵊl` vs `international` `ˌɪntəɹnˈæʃᵊnəl` placing it on different syllables. Adopt in the consistent positions or not at all; byte parity is chasing noise |
| **3,158** | hyphenated compounds SPLIT into two spoken groups | `able-bodied` → `ˈAbəl bˈɑdid`, gold `ˌAbᵊlbˈɑdid`. Ours doubles the primary stress and inserts a word boundary Kokoro reads as a break. But misaki's joining also produces `stˈAtʌvðiˈɑɹt` for "state-of-the-art", which is worse than ours, and our split is what keeps the word→group alignment honest |
| **1,277** | `-ity` nucleus is `ᵻ` where gold has `ə` (96.1% of -ity words) | `ᵻ` is vocab 177 and a near-schwa. A difference, not obviously an error |
| 683 | unstressed initial vowel not reduced | `acceptable` `æksˈɛptəbəl` vs gold `əksˈɛptəbᵊl`. 48% dict-sourced. Ours is a citation form; gold is connected speech. Audible |
| 607 | nasal place assimilation before a velar | `banknote` `bˈæŋknOt` vs gold `bˈænknˌOt`. **Ours is phonetically correct**; gold matches what Kokoro was trained on. Genuinely both ways |
| ~200/1,951 | function words in citation form | `and` `ənd` vs `ænd`, `to` `tu` vs `tə`/`tʊ`, `a` `ə` vs `ɐ`, `the` `ðə` vs `ði`. Noted as "arguable either way" since Run 2 and still undecided |
| 33 | number reading | `2025` as "two thousand twenty five" vs misaki's year reading "twenty twenty-five". Context-dependent; a date column wants the year reading, a quantity does not |
| 20 | initialisms grouped | `VGC` as three groups vs misaki's one. Affects alignment more than sound |

### Note on the reference

misaki's gold is what Kokoro heard, which is why it is the right target — but it is not clean.
It carries doubled consonants (`ɪllˈiɡᵊl`, `ɪmmˈɔɹɾᵊl`), its own doubled rhotic (`ɡɹˈATəɹɹ` for
"greater"), and the `ᵊ` inconsistencies above. Several places where we differ, we are simply
right. The score is a way of FINDING classes, not a target to maximise.

Nothing fixed in this run; it is the triage. Order I would take the defects: the pre-stress flap
(bounded, one condition, 1,109 words, and the guard it needs already exists for primary stress),
then the dropped symbols, then the final-vowel stress, then the dictionary stress rows.

**Update — the pre-stress flap is done (vernacula-phonemizer#1317).** It was not quite the
one-condition change this triage billed: the guard was easy, but the derived `accent-lexicon.tsv`
had to be regenerated for it to reach recorded words at all, and the first design — reading the
stress that survives the clash rule — had to be withdrawn on review because the headline metric
endorsed it and the flap-specific metric refuted it. Next: the 16 dropped symbols.
