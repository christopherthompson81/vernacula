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
