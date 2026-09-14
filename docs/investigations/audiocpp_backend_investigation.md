# audio.cpp as an ASR backend

Can Vernacula run its transcription pipeline on audio.cpp's C ABI, through
AudioCpp.Bindings, alongside the ONNX backends rather than instead of them?

The interest is not "another ASR model". It is whether the published ABI is
enough to back a real application's pipeline — segments in, text and word
timings out, language reported — when that application was built around a
completely different inference stack.

## Run 1 — 2026-09-14 10:10 — survey before writing anything

**Question:** what does a new `AsrBackend` actually have to satisfy, and does
the ABI expose enough to satisfy it?

Read `docs/dev/asr_backend_dispatch.md` and the dispatch cascade in
`TranscriptionService`.

Sizes, to judge what is reusable:

| | files | lines |
|---|---|---|
| `Vernacula.Avalonia` | 100 | 23,100 |
| `Vernacula.Base` | 36 | 20,382 |
| `Vernacula.CLI` | 1 | 1,420 |

The app is far too large to fork for a variant, and `Vernacula.Base` is the ONNX
inference layer — which is exactly the part audio.cpp would stand beside. So the
variant is a ninth `AsrBackend`, not a second application.

**The shape a backend must produce**, from the Cohere branch (the closest
precedent, being a text-output model rather than a token/timestamp one):

```
per segment: text, text tokens, timestamps, logprobs, optional language
```

Cohere has no per-token timings and calls `BuildSyntheticTokenTimestamps` to
spread them evenly across the segment.

**What the ABI offers**, from `AudioCppResult`:

| member | gives |
|---|---|
| `Text` | `(string Text, string Language)` |
| `Words` | `WordTimestamp(Word, StartSample, EndSample, Confidence)` |
| `Segments` | `SpeechSegment(StartSample, EndSample, Confidence, Text)` |

**Finding:** the ABI gives *real* word timings and a detected language. That is
strictly more than the Cohere path has, which fabricates its timings. So this
backend can populate `timestamps` with measured values rather than synthetic
ones, and feed `language` into the existing LID grouping.

**Implication for the next step:** implement against the Cohere branch's shape,
but source timestamps from `Words` instead of `BuildSyntheticTokenTimestamps`,
and fall back to synthetic only when a model reports no words.

**Open question deferred:** confidence. `WordTimestamp.Confidence` is not a
logprob, and the editor's colouring expects logprobs. Decide once the first
transcript exists rather than guessing the mapping now.

## Run 2 — 2026-09-14 11:05 — let the coverage test enumerate the work

**Question:** which dispatch sites does a new backend actually miss? Rather than
work from the list in `docs/dev/asr_backend_dispatch.md`, add the enum value
alone and let the test say.

```bash
dotnet test tests/AsrBackendCoverage -p:EP=Cpu \
  --filter "FullyQualifiedName~AsrBackendCoverageTests|FullyQualifiedName~ModelRepoSelectionTests"
```

**Raw result:** `Failed: 10, Passed: 76, Total: 86`, naming every site:

```
AsrBackendCoverageTests.LanguageSet_IsNonEmpty
AsrBackendCoverageTests.LanguageOptions_AreNonEmpty
AsrBackendCoverageTests.DisplayName_DoesNotThrow_AndIsNonEmpty
AsrBackendCoverageTests.ModelName_DoesNotThrow_AndIsNonEmpty
AsrBackendCoverageTests.BackendOf_ModelName_RoundTrip
AsrBackendCoverageTests.SettingsService_HasModelsDirGetter
AsrBackendCoverageTests.SettingsViewModel_HasIsAsrProperty
AsrBackendCoverageTests.VocabService_KindOfBackend_IsDefined
AsrBackendCoverageTests.VocabService_Constructor_DoesNotWarnOnRecognizedModel
ModelRepoSelectionTests.EveryBackend_RequiresSomething_AndNoForeignVibeVoicePackage
```

Every message named the doc, e.g. *"ModelManagerService.ActiveRepos has no asset
repos for AudioCpp. See docs/dev/asr_backend_dispatch.md."*

**Finding:** the fan-out backstop works exactly as issue #37 intended — the enum
value alone produced a precise, actionable worklist with no guessing. This is
worth recording as a positive result: the mechanism was built after the Granite
Speech miss, and this is the first new backend since, so it is the first real
test of it.

**Decisions taken while fixing, and why:**

| site | decision |
|---|---|
| `Get` | reuses `ParakeetLangs` rather than copying it — same weights, and a divergence would be a bug |
| `ModelName` | `audiocpp/parakeet-tdt-0.6b-v3`, naming the engine as well as the weights, because "same weights, different engine" is the whole point and the two must not collide in the round trip |
| `ActiveRepos` | Core only. audio.cpp fetches its own weights, but VAD and diarization stay Vernacula's, so the ASR arm is absent on purpose |
| `GetAudioCppModelsDir` | a sibling of the ONNX dirs, not a child — either tool's cleanup would otherwise delete the other's weights |
| `VocabKind.AudioCpp` | runs are built by splitting the stored text, since there are no sub-word ids; `DecodeTokens` throws rather than returning something plausible |

**After:** `Passed! - Failed: 0, Passed: 86, Total: 86`.

**Next:** the four *[manual]* sites the test cannot see — the
`TranscriptionService` branch (the one that does the actual work), the settings
radio button, `HomeViewModel`'s missing-weights text, and the transcript editor.

## Run 3 — 2026-09-14 11:40 — a real transcript through the new backend

**Question:** does the seam actually work end to end, and are the word timings
real?

A harness driving `AudioCppAsr` directly on audio.cpp's own 14 s sample, so the
recognition seam is isolated from Vernacula's segmentation:

```
AUDIOCPP_NATIVE_DIR=.../external/audio.cpp/build/bin \
dotnet run -- assets/resources/sample_16k.wav /mnt/data/models/audiocpp cuda
```

**Raw result:**

```
model:   /mnt/data/models/audiocpp/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf
backend: cuda
audio:   225151 samples, 14.07s
ggml_cuda_init: found 1 CUDA devices ... NVIDIA GeForce RTX 3090
load:    1381 ms
run:     95 ms
lang:
words:   28, frames: 28, conf: 28
text:    Some call me Nature. Others call me Mother Nature. I've been here for
         over four point five billion years twenty two thousand five hundred
         times longer than you.
first 8: Some@0.32s  call@0.96s  me@1.36s  Nature.@1.68s  Others@2.80s
         call@3.44s  me@3.84s  Mother@4.08s
```

**Findings:**

1. It works. A correct transcript, through Vernacula's own backend class, on
   CUDA, with no ONNX Runtime involved in the recognition.
2. Word timings are real and plausible — the gaps track the sentence structure
   (a 1.1 s pause before "Others", which is where the speaker breathes). This is
   the thing the Cohere and Granite paths cannot do.
3. Counts agree three ways: 28 words, 28 frames, 28 confidences. That is the
   property `VocabKind.AudioCpp`'s runs depend on, so it is worth asserting
   rather than assuming.
4. 95 ms for 14.07 s is ~148× realtime on a 3090, after a 1381 ms load. Not a
   like-for-like benchmark against the ONNX Parakeet — different quantisation
   (q8_0), and one whole-file segment rather than VAD segments — so it is
   recorded as a datum, not a comparison.

**Negative result, and a bug it caught:** `lang` came back **empty**. Parakeet
through audio.cpp reports no language. The first version of the branch passed
`language: result.Language` straight to `db.UpdateResult`, which would have
written an empty string over whatever the LID pass had already established —
silently clearing a correct language on every audio.cpp segment. Now passes null
when the engine reports nothing, leaving the existing value alone.

That is the second time in this investigation that the interesting outcome was a
field that came back empty rather than a failure.

**Still open:** the confidence-vs-logprob question from Run 1. The editor colours
runs by log probability; these are confidences in [0,1] stored as-is. On this
clip that means every run carries a value the editor will read on a different
scale than it does for ONNX backends. Needs a decision, and a transcript in the
editor to decide against.

## Run 4 — 2026-09-14 12:05 — an eleventh site, and a contract that does not fit

**Question:** does the wider suite agree, once `VocabServiceSmokeTests` is
included? Runs 2 and 3 filtered to the two classes the dispatch doc names.

**Raw result:** `Failed: 5, Passed: 136, Total: 141`, all five in
`VocabServiceSmokeTests`, with two distinct causes:

```
VocabFixtures has no fixture for this kind. Add one alongside the new VocabKind.
audio.cpp transcripts have no token ids to decode; ...
```

**Finding 1 — the doc is one site short.** `docs/dev/asr_backend_dispatch.md`
lists ten sites. `VocabFixtures` is an eleventh, added after the doc was
written, and nothing cross-checks the two. The doc is now stale in the exact way
it exists to prevent. Worth fixing in the doc, not just in the code.

**Finding 2 — the contract genuinely does not apply.** Those five theories
assert a token-id contract: decode ids against a vocabulary file, one run per
id, runs concatenating to the decoded text. audio.cpp has no vocabulary file and
no ids. A fixture invented for it would exercise a code path that does not exist
in production and would pass while asserting nothing.

**Decision:** split the contract rather than fake a vocabulary, and make the
split typed so it cannot become an exemption:

- `VocabFixtures.HasTokenVocabulary(kind)` classifies each kind and **throws** on
  one it does not name. A new kind must declare which family it is in.
- The five token-id theories iterate `TokenVocabBackends`.
- `WordRunKinds_RebuildTheirRunsFromTheText` and
  `WordRunKinds_TolerateShortLogprobs` assert, for the other family, the same
  properties in the unit that family reports: one run per word, runs
  concatenating to exactly the rendered text, logprobs paired positionally, and
  a short logprob list tolerated.

A kind cannot fall out of both families without `HasTokenVocabulary` throwing,
which is the property the original suite had and the one worth preserving.

**After:** `Passed! - Failed: 0, Passed: 138, Total: 138`.

⚠ This modified the test suite rather than only adding to it. Five theories now
iterate a filtered set. The justification is above; if the preference is that
audio.cpp instead carry a synthetic token vocabulary so the original theories
apply unchanged, that is a reasonable different call and the change is one
commit to revert.

## Run 5 — 2026-09-14 12:40 — making it runnable from the GUI

**Question:** can the backend be selected and used from the desktop app without
setting anything at runtime?

**Two blockers found, both introduced by how the submodule sits:**

1. **The engine was not discoverable.** The binding finds libaudiocpp by walking
   up from the executable for `external/audio.cpp/build/bin`. That holds in the
   bindings' own tree; here the engine is a level deeper, under
   `external/AudioCpp-Bindings/`, so the walk never sees it. Measured: neither
   `vernacula/external/audio.cpp/build/bin` nor the parent's exists.
2. **The models path was empty.** `GetAudioCppModelsDir()` resolves to
   `~/.local/share/Vernacula/models/audiocpp`, which did not exist; the packages
   live on `/mnt/data/models/audiocpp`.

**Fix for (1):** carry the library as a `CopyToOutputDirectory` item rather than
copying it in a target. These flow transitively through a `ProjectReference`, so
it lands beside *the app's* executable where the platform's own default search
finds it. A target copying into `Vernacula.AudioCpp`'s own `OutputPath` put it
somewhere nothing looks — that was the first attempt, and it "worked" in the
sense that the file appeared.

Three roots are tried: `AUDIOCPP_NATIVE_DIR`, the nested submodule build, a
sibling `AudioCpp-Bindings` checkout. A missing engine is a **warning**, not an
error: everything except this one backend builds and runs without it.

**A measurement worth keeping:** globbing `libaudiocpp.so*` put **630 MB** in the
output. `libaudiocpp.so` is a symlink to `.so.0` to `.so.0.1.0` and the copy
dereferences each one, so three identical 210 MB files were written. The loader
only needs the plain name, so the glob now excludes the versioned ones.

**Verified — built once with the variable, then run with it unset:**

```
models:  /home/chris/.local/share/Vernacula/models/audiocpp
package: .../Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf
ggml_cuda_init: found 1 CUDA devices ... NVIDIA GeForce RTX 3090
session: opened with no AUDIOCPP_NATIVE_DIR set
```

So the environment variable is a **build-time** input, not a runtime one. Once
an engine has been found at build time the app needs nothing set.

**Decision recorded:** whole-word highlighting is accepted for this stage, so
the confidence-vs-logprob question from Run 1 is deferred rather than open. The
runs are words and the editor colours them with the reported confidences; that
they are on a different scale to the ONNX backends' logprobs is known and
accepted for now.
