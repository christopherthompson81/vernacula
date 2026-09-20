# Measured word timings from audio.cpp's Kokoro

The audio.cpp Kokoro backend estimated its word timings. The ONNX one measures them — it reads the
model's own predicted per-token durations — and the reader's highlight follows the voice exactly
there and approximately here. That was written up as a property of the ABI
(`audiocpp_tts_backend_investigation.md`: "SupportsTimestamps is false on this family and
`result.Words` comes back empty"), which was true and was not the whole story: the durations exist
inside the engine, they were simply never reported.

This log is making the engine report them, and the app read them.

## Run 1 — 2026-09-20 — is the number in there at all?

Question: does audio.cpp's Kokoro compute per-token durations, or does its architecture hide them
inside a fused graph?

It computes them, and holds them in a named field:

```
include/engine/models/kokoro_tts/predictor.h
    struct PredictorOutputs {
        std::vector<int32_t> durations;      // <-- per input token, in decoder frames
        std::vector<float> f0_curve;
        ...
```

`src/models/kokoro_tts/predictor.cpp` fills it (`durations_out.resize(token_count, 1)`) and
`expand_tc_by_durations` / `expand_ct_by_durations` upsample the encoder features by exactly those
counts before the decoder runs. So a group's share of the output is not an estimate at all — it is
the number the decoder was driven with. This is the same `pred_dur` tensor the ONNX path reads.

In `session.cpp`'s chunk loop the three things needed are already in scope together:
`input.input_ids` (so the space tokens, and therefore the group boundaries, are visible),
`predictor.durations`, and the chunk's decoded `audio` for the frames→samples scale.

## Run 2 — 2026-09-20 — how much ABI does this need? None.

Question: what has to be added to the C ABI to carry timings out?

Nothing.

```
include/engine/framework/runtime/session.h
    struct TaskResult { ... std::vector<WordTimestamp> word_timestamps; ... };
    struct WordTimestamp { TimeSpan span; std::string word; float confidence; };
    struct TimeSpan { int64_t start_sample; int64_t end_sample; };

include/audiocpp.h
    audiocpp_result_word_count() / audiocpp_result_word(..., out_word, out_start_sample, ...)
    audiocpp_model_supports_timestamps()
```

The field, the accessor and the capability query all exist and are already bound in
AudioCpp-Bindings (`AudioCppResult.Words`, `AudioCppModel.SupportsTimestamps`). The family simply
never populated the field. **So the bindings need no new code either** — only tests.

One gate, found by following `supports_timestamps` back to its source. It is derived from the model
spec, not set in code:

```
src/framework/model_spec/metadata.cpp:242
    out.supports_timestamps = has_capability(*capabilities, "word_timestamps") || ...
```

and the spec validator restricts which capability tokens each task may declare:

```
src/framework/model_spec/schema.cpp:161
    {"asr", {"word_timestamps", "segments", ...}},
    {"tts", {"speaker_reference", "voice_design", ..., "built_in_voices"}},     // no word_timestamps
```

So `word_timestamps` has to be admitted as a TTS capability. That is a real change to a shared
table rather than a formality, and it is defensible on its own terms: a family whose architecture
predicts durations before vocoding knows where each unit lands. Kokoro does; so does anything
FastSpeech-shaped.

## Run 3 — 2026-09-20 — what is a "word" when the engine has no words?

Question: `word_timestamps` wants a word. What goes in the string?

**Neither path has a token→source-word map.** The supplied-phoneme path obviously does not — there
is no text being spoken. But the TEXT path does not either: `build_kokoro_synthesis_input` runs the
G2P, keeps `std::string phonemes` and `std::vector<int32_t> input_ids`, and records no span back to
the input words. Threading one through eSpeak would be a much larger change than this one.

What IS available on both paths, from `input_ids` alone, is the phoneme GROUP: a run between the
space tokens Kokoro's vocabulary carries. eSpeak separates words with a space and a caller's stream
does the same, so a group is one spoken word on either path.

**Decision: one timing per group, labelled with the group's phonemes.** The unit is right; the
label is honest about being phonemic. A caller whose own G2P produced the stream already knows
which of its words became which group — that is exactly what `KokoroPhonemizer` reports — and a
caller that does not still gets the boundaries.

Rejected: per-token timings (~500 entries per chunk; strictly more information, but every caller
would immediately group them the same way) and a heuristic that zips groups onto the text's
whitespace words on the text path (it breaks precisely where it matters, on numbers and
abbreviations that expand to several groups).

## Run 4 — 2026-09-20 — the change, in three repos and one shared function

**audio.cpp** (`feat/kokoro-word-timings`, off upstream `main`): `append_kokoro_word_timings` walks
the ids, treats pad and space as boundaries, accumulates frames, and converts with a scale derived
from THIS chunk's audio rather than an assumed hop length — so it stays correct if the decoder's
upsampling ratio changes, and absorbs the rounding when a chunk's sample count is not an exact
multiple of its frame count. Chunks are offset by the audio already merged. A duration/token count
mismatch throws rather than degrading: timings that look right and point at the wrong audio are
worse than none.

**AudioCpp-Bindings**: no code.

**Vernacula**: `AudioCppTts.SpeakAligned` returns the groups beside the audio.

The part worth naming is what did NOT get written twice. `KokoroTts.Align` already turned
groups + a group→word map into word timings; the audio.cpp path needed the identical join over
groups from a different source. Extracted to `KokoroAlignment.WordsFromGroups`, and the ONNX path
now calls it too. The engines differ in one sentence — ONNX cuts `pred_dur` at the space tokens
itself, audio.cpp is handed the cuts — and share everything after it, which is the property that
makes a document highlight identically on both.

Alignment is now three tiers, and the sidecar says which was used rather than always claiming
`audiocpp_proportional`:

| tier | when | aligner name |
|---|---|---|
| measured group timings joined to the word map | English voice, phonemizer present, engine reports timings | `audiocpp_duration` |
| proportional, weighted by phoneme count | supplied stream but no timings | `audiocpp_proportional` |
| proportional, weighted by word length | non-English voice, or no phonemizer data | `audiocpp_proportional` |

⚠ The name is chosen once per job from what is knowable before any paragraph renders, so a
per-paragraph fallback lands on a job labelled for the better tier. That is the one direction that
misleads, and it is accepted deliberately: the alternative is renaming the aligner mid-job, and the
fallbacks are rare enough that a per-row audit would be noise.

## Run 5 — 2026-09-20 — two things blocked the measurement, neither of them the change

Question: with the engine rebuilt, does it report timings?

`reports timings=False`.

**The capability is read off the PACKAGE, not the build.** `supports_timestamps` comes from
`capabilities_from_spec`, and the installed `kokoro-82m-q8_0.gguf` embeds a schema-v1 contract
written before `word_timestamps` existed. A published package cannot be edited in place, so
declaring the capability in `model_specs/kokoro_tts.json` does nothing for anyone who already has
the package. Exactly the wall #577 hit with the `phonemes` option, where it was solved by relaxing
the *validation copy*; that reasoning does not transfer here, because a stale capability flag is
not a hard failure — it reads false and a caller falls back. Noted for the PR: existing packages
need regenerating, or upstream decides the runtime's contract should win for capabilities.

**And regenerating is currently broken.** `audiocpp_gguf --input … --model-spec … --overwrite`
reports success —

```
embedded_sidecars=true
embedded_model_spec=true
model_spec_family=kokoro_tts
```

— and the package it writes cannot be loaded:

```
audiocpp_model_load: failed to open binary file: /tmp/audiocpp-gguf/<hash>/voices/zm_yunxia.bin
```

**Zero voices extract from it.** The sidecar manifest is embedded without the payloads. Reproduced
from two different source packages and with `--root` pointing at an extracted tree that definitely
contains the file (`voices/zm_yunxia.bin`, 522,240 bytes). This is the same area #579 fixed once
(a re-encode dropping `voices/*.bin` and exiting 0), so it is either a regression or a second path
with the same shape. Out of scope here; recorded, and worth its own upstream issue.

The measurement went through `--model-spec-override` instead, which is the supported way to load
the runtime's current contract against an older package and what the bindings' own tests use.

## Run 6 — 2026-09-20 — do the two engines agree about where the words are?

Same text, same supplied phonemes, both engines, both joined to source words through the same
`KokoroAlignment.WordsFromGroups` — so what is compared is the timings and not two different
mappings. Word starts are compared as a FRACTION of each engine's own duration: the buffers are not
the same length, and an absolute comparison would report a drift that is only the totals differing.

```
spec override=model_specs/kokoro_tts.json   reports timings=True

[0] The button was forgotten on the cotton coat.
     audio.cpp 2.92s, 8 groups   |   onnx 2.92s, 8 words      worst 0.00%
[1] She read the schedule aloud at a quarter past three.
     audio.cpp 3.15s, 10 groups  |   onnx 3.23s, 10 words     worst 0.76%
[2] Uranium and aluminium are both elements.
     audio.cpp 3.08s, 6 groups   |   onnx 3.02s, 6 words      worst 0.76%
[3] A quarter of the students answered correctly.
     audio.cpp 2.90s, 7 groups   |   onnx 2.90s, 7 words      worst 0.76%
[4] The harbour was quiet this morning.
     audio.cpp 2.38s, 6 groups   |   onnx 2.38s, 6 words      worst 0.76%

rows=37  mismatched sentences=0  worst word-start delta=0.76% of duration
VERDICT: the two engines agree on where the words are.
```

**Finding: ggml and ONNX Runtime produce the same duration prediction.** Sentence [0] is exact —
same total, same word starts to the printed precision — and the worst disagreement anywhere is
0.76% of the utterance, about 18 ms on a 2.4 s sentence. Where the totals differ (3.15 s against
3.23 s on [1]) the word starts still agree as fractions, which says the difference is in the
vocoder's output length and not in the alignment.

That is the evidence the numbers are the MODEL's. Two independent implementations of one duration
predictor, on different runtimes and different hardware paths, landing on the same answer is not
something a bug in either one would produce.

### What this replaces

The reader's highlight on this backend was a proportional spread — first by spelling, then by
phoneme count. Neither was ever going to be right on a word whose pronunciation is longer than its
spelling suggests. It is now the same measurement the ONNX engine has always used, through the same
code.
