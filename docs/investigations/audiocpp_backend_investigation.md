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
