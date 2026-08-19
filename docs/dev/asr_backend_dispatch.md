# ASR backend dispatch fan-out

Adding a new ASR backend to Vernacula touches roughly ten per-backend
dispatch sites scattered across `Vernacula.Avalonia`. There is no
single registry, and there is no compile-time enforcement that all
sites stay in sync — the C# language considers an enum switch
non-exhaustive even when every declared member is covered, so a
missing case at any site produces a silent fall-through to whatever
default that site defines (almost always Parakeet).

The Granite Speech integration (issue #33 / PR #35) demonstrated the
failure mode: `VocabService` was missed despite a code review and a
per-PR touch-list, and the bug only surfaced when a user opened the
editor on a real transcript and saw raw GPT-2 ByteLevel BPE alphabet
characters instead of decoded text. See issue #37 for the discussion
of approaches; what landed is **B + D**: tighten silent-fallthrough
defaults to throw, plus a coverage test backstop.

## Run the coverage test

Tests run automatically in CI (`.github/workflows/dotnet-test.yml`),
which builds the solution with `-p:EP=Cpu` and runs this test (must
pass) plus `tests/IndicConformerTest` on every push to `main` and on
every pull request. The dispatch-fan-out coverage test therefore blocks
PR merges instead of silently rotting. To run it locally:

```bash
dotnet test tests/AsrBackendCoverage
```

The test iterates `Enum.GetValues<AsrBackend>()` and probes each
dispatch site listed below. Add a new backend → run the test → fix
each failing site one-by-one.

## Dispatch sites

When adding a new `AsrBackend` enum value, every site here must be
updated. Sites marked **[test]** are exercised by
`tests/AsrBackendCoverage/AsrBackendCoverageTests.cs`; sites marked
**[throw]** fail-loud at runtime via an `_ => throw` switch arm; sites
marked **[manual]** require human review during code review or PR
description.

### 1. `AppSettings.AsrBackend` enum *[manual]*

`src/Vernacula.Avalonia/Models/AppSettings.cs`. The source of truth
the rest of the table dispatches on. Adding a new value here is the
trigger for everything below.

### 2. `AsrLanguageSupport` per-backend metadata *[test]*

`src/Vernacula.Avalonia/Models/AsrLanguageSupport.cs`. Five methods,
all switch expressions, defaults throw:

- `Get(AsrBackend)` — language set
- `BackendOf(string)` — model-name → enum
- `ModelName(AsrBackend)` — enum → model-name (round-trip with
  `BackendOf`)
- `DisplayName(AsrBackend)` — UI label
- `LanguageOptions(AsrBackend)` — derived from `Get`

**Behaviour change in this PR:** `Get` previously returned
`FrozenSet<string>.Empty` for unknown backends and `DisplayName`
returned `backend.ToString()`; both now throw
`ArgumentOutOfRangeException`. Callers that ever pass a runtime-cast
`AsrBackend` value (e.g. `(AsrBackend)999` from a corrupt settings
file) will now crash where they previously fell back silently.
That's a deliberate regression — the silent fallback was the original
drift vector — but it does mean defensive callers should validate via
`Enum.IsDefined` before forwarding.

### 3. `ModelManagerService.ActiveRepos` *[throw]*

`src/Vernacula.Avalonia/Services/ModelManagerService.cs`. Switch
expression with explicit Parakeet arm; default throws. VibeVoice is
handled in an early-return above the switch (it can be the
segmentation backend even when the AsrBackend isn't VibeVoice).

### 4. `TranscriptionService` Phase-4 dispatch *[manual]*

`src/Vernacula.Avalonia/Services/TranscriptionService.cs`, around line
820 onward. Long `if/else if` cascade dispatching to per-backend ASR
recognition + LID-grouping. Cannot be made a switch expression because
each branch has substantial unique logic (model loading, batching, LID
grouping). The final `else` corresponds to Parakeet; new backends MUST
add an explicit `else if` branch *before* the Parakeet fall-through,
not after.

There is a separate site in the same file (around line 130) that
reads `SegmentationMode` for the diarization-end-percent reset; that's
not AsrBackend dispatch and is not relevant here.

### 5. `SettingsService.Get<Backend>ModelsDir` *[test]*

`src/Vernacula.Avalonia/Services/SettingsService.cs`. One method per
backend, named `Get<EnumName>ModelsDir()` (PascalCase). The coverage
test reflects on this convention. New backends must add a method whose
name exactly matches `$"Get{backend}ModelsDir"`.

### 6. `SettingsViewModel.IsAsr<Backend>` properties *[test]*

`src/Vernacula.Avalonia/ViewModels/SettingsViewModel.cs`. One
`bool IsAsr<EnumName>` property per backend, plus an entry in the
`[NotifyPropertyChangedFor]` attribute on `_selectedAsrBackend`. The
coverage test reflects on the property; the
`NotifyPropertyChangedFor` entry is **not** caught by the test — it
is a manual-review item. Forgetting it means the radio button does
not update when the backend selection changes.

### 7. `SettingsWindow.axaml` RadioButton block *[manual]*

`src/Vernacula.Avalonia/Views/SettingsWindow.axaml`. One
`<RadioButton>` per backend, bound to the corresponding
`IsAsr<Backend>` property and invoking the `SetAsrBackend` command
with the enum name as a string. XAML cannot be checked at compile
time; this is the most-frequently-missed site after VocabService.

### 8. `HomeViewModel.UpdateStatusText` *[manual]*

`src/Vernacula.Avalonia/ViewModels/HomeViewModel.cs`. `if/else if`
cascade producing the user-facing "weights are missing" string with
an installation path hint. Falls through to a generic localized
message for backends without specific guidance, which is acceptable
behaviour — this site is "graceful degradation, not silent
correctness", so it isn't covered by the test.

### 9. `VocabService` *[test]*

`src/Vernacula.Avalonia/Services/VocabService.cs`. Three sub-sites:

1. **`KindOfBackend(AsrBackend)`** — switch expression, default
   throws. Maps every backend to a `VocabKind`. Covered by the
   `VocabService_KindOfBackend_IsDefined` test.
2. **Constructor cascade** — `if/else if` on the `asrModel` string,
   logs a warning to stderr if no branch matches a non-null name.
   Covered by the
   `VocabService_Constructor_DoesNotWarnOnRecognizedModel` test
   (captures stderr; fails on a fallback warning).
3. **`DecodeTokens` / `GetTokenRuns` / `DecodeToken`** — switch and
   `if/else` chains on the internal `VocabKind`. Default throws.
   New `VocabKind` members must add a case to each method.

### 10. `TranscriptEditorWindow` *[manual]*

`src/Vernacula.Avalonia/Views/TranscriptEditorWindow.axaml.cs`,
roughly lines 100–145 in the constructor. Three sub-sites, all `bool`
flags driven by string equality on `_jobAsrModel`:

1. Backend-specific approximate-timing notice text.
2. Vocab-path resolution (which tokenizer file the editor passes to
   `VocabService`).
3. The `GetModelsDir()` vs `parakeetModelsDir` selector for the
   `VocabService` constructor's first argument.

These cannot be cleanly turned into compile-time-checked switches
without a larger refactor (option C in issue #37, deferred). The
coverage test does not exercise the editor; missing a case here will
present in the editor as wrong vocab-path, missing notice, or both.

## Why no compile-time enforcement?

Switch expressions on C# enums emit CS8509 only when the compiler
cannot prove the input is restricted to the declared members — which
is *always* for enums, since `(AsrBackend)999` is representable. Even
when every declared member is covered, the compiler still requires a
default arm to suppress CS8509. There is no language-level "warn me
when I forget a real enum value but accept unknown ones."

Practical consequence: option B from issue #37 (compile-time
exhaustiveness) is unworkable in C# without an analyzer. The runtime
substitute is `_ => throw new ArgumentOutOfRangeException(...)` at
every dispatch site, plus the coverage test as a backstop. Option A
(loud fallthrough) is what `VocabService`'s constructor uses for the
unknown-string case, since that one site has a meaningful "default to
Parakeet on null" branch that we don't want to throw on.
