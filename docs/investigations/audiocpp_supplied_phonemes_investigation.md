# Supplying vernacula-phonemizer's phonemes to audio.cpp's Kokoro

The audio.cpp Kokoro backend phonemized inside the engine with eSpeak-ng, which made it the one
place in this app where a word is pronounced by something other than vernacula-phonemizer: a
correction the user makes in the dictionary changed the ONNX render and not this one. Upstream
[0xShug0/audio.cpp#577](https://github.com/0xShug0/audio.cpp/pull/577) added a `phonemes` request
option that takes a caller's stream instead. This log is adopting it here.

The engine-side and binding-side halves are already done and merged elsewhere —
audio.cpp#566/#577 for the ABI and the option, AudioCpp-Bindings#87 for `SetOptionArray` and its
tests (`AudioCpp-Bindings/docs/investigations/supplied_phonemes_investigation.md`). What is left
is this repo's: the pins, the frontend, the chunking, and what the supplied stream buys for
alignment.

## Run 1 — 2026-09-20 09:10 — where the three pins actually are

Question: is anything here already in place, and how far behind are the submodules?

```
git submodule status
git -C external/vernacula-phonemizer log --oneline <pin>..origin/main | wc -l
git -C external/AudioCpp-Bindings/external/audio.cpp log --oneline -1
nm -D external/AudioCpp-Bindings/external/audio.cpp/build/bin/libaudiocpp.so | grep set_option
```

| pin | was | is | gap |
|---|---|---|---|
| vernacula-phonemizer | `c59bd8cf` (#1348) | `0dbc7443` (#1377) | 29 commits |
| AudioCpp-Bindings | `766f7187` (#86) | `5a10e50c` (#87) | the supplied-phoneme binding itself |
| audio.cpp (nested) | `31d00b5c` (#565) | `df0e09ef` (#577) | 16 commits |

The built library exported `audiocpp_request_set_option_array` — **not at all**:

```
0000000000312ff0 T audiocpp_request_set_option@@AUDIOCPP_0
```

`libaudiocpp.so.0.1.0` was linked 2026-09-15 19:53, two days before #577 merged. So nothing on
this machine's *vernacula* checkout could serve a supplied stream, and the first step is a
rebuild rather than any code.

The phonemizer's own checkout was already at `origin/main` and clean; only the parent's pin
lagged. The 29 commits are English dictionary work plus one retrained BiLSTM OOV tagger (#1361),
which is a reading change, not an interface change.

**Implication:** move all three pins, rebuild the engine, then write code against an engine that
can actually refuse it.

## Run 2 — 2026-09-20 09:25 — the rebuild, and a ccache that looked broken and was not

Question: `scripts/build-engine.sh --ccache --cuda-arch native -DENGINE_ENABLE_CUDA=ON` reported
the right configuration —

```
backends: CUDA=ON
compiler launcher: CXX=ccache CUDA=ccache
cuda architectures: native (1 entries)      # 86-real
```

— so why was it recompiling every CUDA template instance, when the bindings' own Run 2 had
already paid for a `native`-arch CUDA build two days earlier?

Measured rather than guessed. Over one ~35 s window of the build, `ccache -s` moved by
**hits +0, misses +9**: genuinely missing. But the cache was not full (0.4 GiB of 5.0, 12,629
files), so nothing had been evicted.

```
(default) base_dir =
(default) hash_dir = true
```

And there are two checkouts of the bindings on this machine:

| tree | engine linked | build dir |
|---|---|---|
| `~/Programming/AudioCpp-Bindings` | 2026-09-17 09:00 | 384 MB, CUDA objects present |
| `…/vernacula/external/AudioCpp-Bindings` (this one) | 2026-09-15 19:53 | the one being rebuilt |

**Finding: the cache is warm for the other clone, and ccache keys on absolute paths.** The nvcc
lines carry this tree's include paths (`CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp`), which are
part of the hash, and `hash_dir` is on with no `base_dir`. Entries filed under
`~/Programming/…` cannot hit for `/mnt/data/Programming/…`.

Not a defect and **deliberately not fixed**: `base_dir=/ hash_dir=false` would make the two
clones share, but that is a global config change for a one-off cost, and the earlier work simply
was not done in this working directory. Recorded so the next person does not go looking for a
corrupt cache.

## Run 3 — 2026-09-20 09:40 — what the engine will and will not accept

Question: what shape does the stream have to be in, and who filters it?

From `model_specs/kokoro_tts.json` at `df0e09ef` and the merged `session.cpp`:

- `phonemes` is `string_list` — **one entry per chunk**, rendered in order and merged into one
  buffer. The engine chunks *text* on its own `text_chunk_size`; it cannot chunk phonemes,
  because only the G2P that produced a stream knows where it may be cut. So the caller cuts.
- Each entry ≤ 510 symbols, and **no entry may be empty** — set-but-blank is a caller error, not
  a fallback to the text. An empty list is refused the same way.
- `text` is still required and its language must still match the voice.
- **A caller's stream is validated where the engine's own output is not.** A symbol outside
  Kokoro's vocabulary fails the run, naming the entry: `Kokoro supplied phoneme entry 200:
  Kokoro vocab is missing phoneme symbol: R`.

That last one is the design decision this repo has to answer. Our ONNX Kokoro *drops* an unknown
symbol (`KokoroVocab.Encode`, mirroring misaki/KModel's
`filter(None, map(vocab.get, phonemes))`). audio.cpp *refuses* it. Both are right for their
caller — upstream's reasoning is that you can only demand a correction from someone able to make
one, and a dropped off-glide turns "like" into "lack" byte-identically to having written it.

**Decision: filter here, and say so.** `KokoroVocab.KeepKnown` drops exactly what `Encode` drops
and reports what went, so the two engines say the same thing and one stray diacritic anywhere in
a document cannot be fatal on one backend only. The alternative — send it raw and let the engine
refuse — makes the ONNX path and the audio.cpp path disagree about whether a document is
renderable at all.

## Run 4 — 2026-09-20 10:05 — the chunker had to come out of the ONNX class

Question: the engine wants the chunks; who cuts them?

`KokoroTts.ChunkForSynthesis` already cuts to a *phoneme-token* budget (460 packing, 508 hard,
against Kokoro's 512-token window), descending paragraph → sentence → clause → word, splitting
only at whitespace. That is exactly the cut the supplied-phoneme path needs, and for the same
underlying reason: the model's context window.

It was unreachable — `KokoroTts`'s constructor loads an ONNX session, and `Vernacula.AudioCpp`
deliberately does not reference the ONNX stack at all ("what is being tested is the ABI, not a
hybrid"). Moved verbatim to `KokoroChunker`, which takes only a `KokoroPhonemizer`; `KokoroTts`
delegates. One chunker, so the two engines cut the same document the same way.

The property that makes the word map work — chunks concatenate back to the source word sequence —
was assumed by the ONNX path and is now load-bearing for a second caller, so it is asserted
(`KokoroChunkerTests.ChunkingSplitsAtWhitespaceAndLosesNoWord`) rather than trusted.

## Run 5 — 2026-09-20 10:20 — the alignment this buys, which was not the point but is the gain

Question: the ABI reports no word timings and supplying phonemes does not change that. Does
anything about the reader's highlight improve anyway?

Yes, and by more than expected. The estimator was `OmniVoiceIpaAlignment.Proportional` with an
empty trace, i.e. **weight each word by how many letters it is written with**. Supplying the
stream means `KokoroPhonemizer.Phonemize` has already told us which source word each phoneme
group came from — so a word's share of the paragraph can be its share of the *phonemes*.

The test case is `through spa`: seven letters against three, but three phonemes against three.

    letter-weighted   through / spa  =  2.33
    phoneme-weighted  through / spa  ≈  1.0

Still a proportional spread of one merged buffer, so the sidecar's aligner stays
`audiocpp_proportional` — this is a better weighting, not a measurement. Guards: a map whose
length disagrees with the word count produces **no** words rather than a shifted highlight, a
word that became no phonemes keeps a zero-length marker so the source split stays 1:1, and
all-zero weights fall back to the letter estimate instead of dividing by zero.

## Run 6 — 2026-09-20 10:50 — the engine rebuild, and the first supplied render

Question: with the pin on `df0e09ef` and the library relinked, does a stream we produced actually
render?

```
nm -D build/bin/libaudiocpp.so | grep option_array
0000000000322990 T audiocpp_request_set_option_array@@AUDIOCPP_0
```

(Incidentally the library went 209 MB → 74 MB: the old one was the sticky nine-architecture CUDA
build from before `--cuda-arch native` existed in the script.)

A scratch harness renders each sentence twice on the same session — once with `text` alone, once
with the same text plus our `phonemes` list — and reads both back through Parakeet on CUDA, so
the comparison is about what was SAID rather than about whether audio appeared.

```
[0] The button was forgotten on the cotton coat.
     phonemes : ðə bˈʌTən wʌz fəɹɡˈɑTən ˈɔn ðə kˈɑTən kˈOt.
     engine   : 2.60s      supplied : 2.92s
     heard(e) : The button was forgotten on the cotton coat.
     heard(s) : The button was forgotten on the cotton coat.
```

**It works on the first try, and the syllabic-nasal family — the one that used to be fatal on
this engine (audiocpp_tts_backend_investigation Runs 6-7) — comes through on a stream we wrote.**
`bˈʌTən` is our flap-plus-schwa rendering, not eSpeak's `bˈæʔn̩`; the engine is no longer choosing.

## Run 7 — 2026-09-20 11:05 — twenty sentences, and a verdict rule that was wrong

Question: across an ordinary paragraph's worth of text, does changing the G2P change what is
heard?

First answer was `pass=2 fail=3`, which was the harness's fault, not the stream's. The rule
compared each readback against the SOURCE TEXT, which measures Parakeet's spelling conventions:
"three dollars and fourteen cents" comes back as `$3.14` and "four hundred and twenty" as `420`
on **both** renders. The question is supplied-vs-engine — did the reading change — so the rule
became that, with the text comparison kept as a column rather than as the verdict.

Re-run over 20 sentences, af_heart:

    same reading 17/20

The three that differ:

| # | engine heard | supplied heard | verdict |
|---|---|---|---|
| aluminium | "aluminium" | "aluminum" | **a real reading difference** |
| 9 a.m. | "9AM" | "9 a.m." | the ASR's formatting; both correct |
| …24 March invoice | "invoice" | "Invase" | suspect — chased in Run 8 |

## Run 8 — 2026-09-20 11:20 — one of the two is ours, and the other is not

Question: are `invoice` and `aluminium` both defects in our reading?

`invoice` — no. Our IPA is `ˈɪnvYs`, i.e. /ˈɪnvɔɪs/, which is the word. Put through seven carrier
sentences:

```
The invoice is due.                    heard(e) "invice"    heard(s) "invoice"
Send the invoice today.                same
She filed the invoice yesterday.       same
Dr. Smith wrote about the invoice.     same
Dr. Smith wrote to ... 24 March invoice.  heard(e) "invoice"  heard(s) "Invase"
The March invoice was paid.            same
We received an invoice for the work.   same
```

**Six of seven agree, and the ENGINE is the one that fails a different carrier.** The word is
marginal for the readback near a phrase boundary on both G2Ps; it is not a property of the
supplied stream. Negative result, kept because the first sweep made it look like one.

`aluminium` — this run said yes; **Run 9 retracts that**. Read on, but the conclusion below is
superseded. Reproducible in isolation and in two carriers:

```
aluminium                              heard(e) "Aluminium"   heard(s) "Aluminum"
The aluminium sheet was thin.          heard(e) "aluminium"   heard(s) "aluminum"
Uranium and aluminium are both elements.  same split
```

Our reading is `əlˈumɪnəm` — four syllables, i.e. the pronunciation of the American spelling
⟨aluminum⟩, for the British/IUPAC spelling ⟨aluminium⟩, which has five (/ˌæljʊˈmɪniəm/). eSpeak
reads the spelling as written and is right here.

**This is a vernacula-phonemizer dictionary defect, not an audio.cpp one**, and it was invisible
before today because the engine's own G2P was covering for it. Worth a row upstream; it does not
block this change, because the same word already reads the same way on the ONNX Kokoro — this
run is the first time the two could be compared at all.

### Conclusion

Supplying the stream is sound: 17/20 identical readings, one difference that is the ASR's
spelling, one (`aluminium`) that is CMUdict's deliberate GenAm normalization and correct — see
Run 9 — and none attributable to the transport. The syllabic-nasal class that used to kill this
backend now comes through a stream we control.

## Run 9 — 2026-09-20 — Run 8 was wrong about which locale has the problem

Question, raised in review: is normalizing British ⟨aluminium⟩ to GenAm "aluminum" not simply
**correct** for an `en` voice?

Yes. Run 8 called it a dictionary defect without checking the provenance, and the provenance
settles it — `data/languages/english/g2p-dict.tsv` carries both spellings:

```
aluminium   AH0 L UW1 M IH0 N AH0 M
aluminum    AH0 L UW1 M AH0 N AH0 M
```

That is CMUdict's own row. A GenAm reference deliberately maps the British spelling onto the
American pronunciation, and `en` is GenAm, so `əlˈumɪnəm` for af_heart is the reference reading
rather than a lapse. eSpeak reading the spelling as written is the outlier here. **Run 8's finding
is retracted: there is no defect in the GenAm path, and no upstream row to file.**

It is also the same normalization the sweep accepted without comment elsewhere — `harbour` →
`hˈɑɹbəɹ`, `travelled` → `tɹˈævəld`. Those went unremarked because the ASR's readback spelled them
American too, so nothing stood out; `aluminium` stood out only because the syllable count changes.
Which is the actual signal, and it points one locale over.

### The gap is in en-GB, and it is structural

`data/languages/english-gb/` holds **no lexicon** — six accent-transform tables (BATH, CLOTH, LOT,
MARRY, PALM, YOD) applied over the GenAm dictionary. `aluminium` appears once, in
`en-gb-yod.tsv`. So the British render is CMUdict's four-syllable stem with a /j/ inserted:

```
bf_alice  "aluminium"                        → əljˈuːmɪnəm   heard(s) "Tell you madam."
                                                             heard(e) "Aluminium"
bf_alice  "The aluminium sheet was thin."    → ðə əljˈuːmɪnəm ʃˈiːt wʌz θˈɪn.
                                               heard(s) "Their aluminum sheet was thinned."
```

British is /ˌæljʊˈmɪniəm/ — five syllables, stress on MIN, first vowel /æ/. `əlˈjuːmɪnəm` is
neither word, and Parakeet's "Tell you madam" is a fair description of it.

**Finding: the en-GB layer handles ACCENT differences and structurally cannot handle LEXICAL
ones.** A yod table can insert a consonant; it cannot add a syllable or move the stress. Words
where GenAm and British differ in form rather than in accent — aluminium/aluminum, and by
extension the maths/math, whilst/while family — have nowhere to live in that design. Fixing it
means a GB lexical-variant layer (a small override lexicon consulted before the transforms), not
an edit to any existing row.

Out of scope here and **not introduced by this change**: both backends share the frontend, so the
ONNX Kokoro has read it this way all along. This change only makes it audible in one more place.
The sweep in Run 7 was af_heart only, which is why it took a review question to surface.
