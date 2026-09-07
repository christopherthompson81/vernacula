# Preallocated logits binding across ASR backends (issue #46 item 1)

Issue #46 item 1 proposes rolling the PR #45 phase-8 change — swapping the decoder
step loop's `BindOutputToDevice("logits", cpuMemInfo)` (ORT allocates a fresh CPU
buffer per `RunWithBinding`) for `BindOutput("logits", preAllocatedOrtValue)` (one
reused CPU buffer) — across `WhisperTurbo`, `Qwen3Asr`, `VibeVoiceAsr` and
`CohereTranscribe`, predicting "likely 5–10% per-backend wall-time win on
step-heavy workloads".

This log is the measurement behind accepting or rejecting that estimate per backend.

## Benchmark input

`bench.wav`: seven distinct **synthesized** TTS sample clips from the repo
(`scripts/kokoro_export/samples`, `scripts/omnivoice_export/capture`) concatenated
four times, resampled to 16 kHz mono — 89.3 s, 32 VAD segments. Entirely
machine-generated audio; no recorded speech, no PII. Deliberately not a single
looped clip, so the repetition-loop detectors in the decoders do not fire.

Command shape:

```
dotnet src/Vernacula.CLI/bin/Release/net10.0/Vernacula.CLI.dll \
  --asr <backend> --vad --benchmark --audio bench.wav --output out.md
```

Machine: RTX 3090 (24 GB), CUDA EP, Release build.

## Run 1 — 2026-09-07 07:20 — baselines, and which sites are actually hot

Question: before changing four backends, which of the listed `BindOutputToDevice`
sites are on a default code path, and how much of wall time does the step loop
actually own?

### Site survey

Mapping the issue's line numbers onto the methods that contain them:

| Backend | Line | Method | Role | On the default path? |
|---|---|---|---|---|
| WhisperTurbo | 531 | `TranscribeBatch` | prefill (init) | yes |
| WhisperTurbo | 614 | `TranscribeBatch` | **step loop** | **yes** |
| WhisperTurbo | 755 | `Transcribe` | prefill (init) | single-utterance path |
| WhisperTurbo | 816 | `Transcribe` | **step loop** | single-utterance path |
| Qwen3Asr | 665 | static-batch decode | step loop | yes — *already uses `BindOutput`* |
| Qwen3Asr | 1651 | `DecodeOnGpuUnified` | prefill (init) | `--qwen3asr-serial` only |
| Qwen3Asr | 1689 | `DecodeOnGpuUnified` | **step loop** | `--qwen3asr-serial` only |
| Qwen3Asr | 1908 | `DecodeWithIoBinding` | prefill (init) | serial + split-decoder bundle |
| Qwen3Asr | 1937 | `DecodeWithIoBinding` | **step loop** | serial + split-decoder bundle |
| VibeVoiceAsr | 556 | `RunOnce` | prefill **and** step (shared) | yes |
| CohereTranscribe | 353 | init | prefill (init) | yes |
| CohereTranscribe | 458 | step loop | **step loop** | yes |

Two findings from the survey alone, before any measurement:

1. **The prefill/init sites are not worth changing.** They run once per segment (or
   once per batch), not once per token, so there is no allocation to amortise; and
   their logits shape is `[B, S, V]` with `S` varying per segment, so a single
   preallocated buffer would have to be sized for the worst case and re-wrapped per
   call anyway. The win the issue measured is a step-loop property. Only the five
   step-loop sites are candidates.
2. **`VibeVoiceAsr` line 556 is not a step-loop site** in the sense the others are —
   `RunOnce` is shared between prefill and decode, clearing and re-binding outputs on
   every call, so the logits shape varies by call kind. It needs a different change
   from the other four, and there is no VibeVoice bundle on this machine
   (`~/.local/share/Vernacula/models` has no `vibevoice_asr/`) to regression-test it
   against. Deferred.

### Whisper baseline

```
Whisper breakdown (accumulated across all segments):
  Mel          :     179 ms (2.3 %)
  Encoder      :    5573 ms (70.3 %)
  DecoderInit  :     623 ms (7.9 %)
  DecoderStep ORT call     :    1531 ms (19.3 %)
  DecoderStep extract/copy :       0 ms (0.0 %)
  Argmax       :      18 ms (0.2 %)
  Total phases :    7927 ms (ASR swAsr: 35474 ms)
  Step calls   : 57   (avg 26.86 ms ORT / step)
  Segments     : 32
Real-time factor: 0.4040
```

Raw finding: the Whisper step loop is **19.3 % of instrumented phase time but only
4.3 % of measured ASR wall time** (1531 ms of 35474 ms). This workload is
encoder-dominated (70 % of phases), and only 57 decoder steps run across 32
segments because VAD cuts short segments and the batch of 8 finishes on its
longest row.

Implication: even the full 24 % step-loop improvement PR #45 saw on Granite would
be ~370 ms here — **~1 % of ASR wall time, not the 5–10 % the issue predicts.**
The 5–10 % estimate silently assumed every backend is as step-heavy as Granite.
Whisper on VAD-segmented audio is not.

`Total phases` (7927 ms) accounts for only 22 % of `swAsr` (35474 ms). Chased
that down before treating it as a finding: `swAsr` starts at `Program.cs:461`,
*before* the backend is constructed, so it includes ONNX session creation and CUDA
EP initialisation — ~27 s, consistent with the ~28 s Granite spends on the same
thing. Not a mystery and not a regression; it does mean **`swAsr` is the wrong
denominator for a steady-state comparison**, since it is dominated by a fixed
startup cost. The rest of this log compares the instrumented phase totals.

**Run 1 was invalid as a baseline.** Applying the change and re-running produced a
whole-pipeline speedup — Encoder 5573 → 1857 ms, DecoderInit 623 → 72 ms, swAsr
35474 → 4243 ms — which no logits-binding change can explain. Run 1 was the first
CUDA process of the session: cold driver, cold kernel cache, cold page cache. Every
number in it is a cold-start artefact. All subsequent runs discard the first.

## Run 2 — 2026-09-07 07:45 — Whisper, warm, VAD-segmented (B=8, 57 steps)

Change under test, in `TranscribeBatch`'s step loop:

```csharp
var stepLogitsBuf = new float[B * VocabSize];
using var stepLogitsVal = OrtValue.CreateTensorValueFromMemory(
    stepLogitsBuf, [B, 1L, VocabSize]);
...
stepBinding.BindOutput("logits", stepLogitsVal);   // was BindOutputToDevice(..., cpuMemInfo)
...
var stepLogits = new ReadOnlySpan<float>(stepLogitsBuf);   // was curStep[0].GetTensorDataAsSpan<float>()
```

`DecoderStep ORT call`, warm, 3–4 runs each:

| | run a | run b | run c | run d |
|---|---|---|---|---|
| baseline | 125 ms | 128 ms | 126 ms | 125 ms |
| preallocated | 128 ms | 129 ms | 129 ms | — |

No improvement; the two are inside each other's noise, with the preallocated side
marginally *slower*. Transcripts byte-identical.

Objection to this run: 57 steps across 32 segments is not a "step-heavy workload".
Fair — so:

## Run 3 — 2026-09-07 07:55 — Whisper, warm, gapless audio (B=1, 140 steps)

`bench_dense.wav`: `bench.wav` with silences stripped
(`silenceremove=stop_periods=-1:stop_duration=0.15:stop_threshold=-45dB`), 76.2 s,
which VAD returns as **one** segment. Step loop now owns 28–29 % of phase time.

| | run a | run b | run c |
|---|---|---|---|
| baseline | 152 ms | 155 ms | (cold 219) |
| preallocated | 152 ms | 153 ms | 152 ms |

No improvement. Transcripts byte-identical.

## Run 4 — 2026-09-07 08:05 — Whisper, warm, batched *and* step-heavy (B=8, 393 steps)

The strongest form of the proposal's claim: the per-step buffer is at its largest
(B=8 × 51866 floats = 1.6 MB per `RunWithBinding`) *and* the step loop dominates.
`bench_batch8.wav`: `bench_dense.wav` repeated 8× with 0.7 s silences, 615 s, which
VAD returns as 8 long segments filling one batch. Step loop = 32.7 % of phase time.

| | run a | run b | run c |
|---|---|---|---|
| baseline | 798 ms | 797 ms | 800 ms |
| preallocated | 797 ms | 797 ms | 798 ms |

**No improvement — 393 steps, 1.6 MB of per-step allocation removed, and the
difference is under 0.4 %, well inside run-to-run noise.**

### Conclusion on issue #46 item 1

The proposal does not reproduce on WhisperTurbo, in any of the three workload
shapes tested, including the one that maximises the effect it targets. At
~2 ms of GPU work per step, ORT's CPU allocator serving a 1.6 MB buffer is not
measurable; the step is bound by decoder compute, not by output-buffer allocation.

Why PR #45 saw 24 % on Granite and this sees 0 % is not established here, and the
decisive experiment is not available on this machine: Granite's two
`BindOutput(preallocated)` sites live in `TranscribeBatch` (dynamic-MHA bundle) and
`TranscribeStaticGqaBatched` (static-GQA bundle), and the only Granite bundle
present is static-MHA, which dispatches to `TranscribeBatchStatic` — a path that
uses plain `Run()` and no IO binding at all. Reverting Granite's own change and
re-measuring it would need a bundle this machine does not have. The likeliest
explanation is that the 24 % was not the allocation but something else in the same
phase-8 commit — that commit also added the `next_token` GPU-argmax output, which
removes an entire `[B, 1, 49k]` device→host logits copy per step, a far larger
effect than allocating the buffer that copy lands in.

**Recommendation: do not roll this out.** The Whisper change was written,
measured, found to be a no-op, and reverted; it is not in this branch. The
remaining candidate sites are worth even less than Whisper's:

- `Qwen3Asr` line 665 — the **default** path — already uses `BindOutput`. The two
  remaining step-loop sites (`DecodeOnGpuUnified`, `DecodeWithIoBinding`) are
  reachable only via `--qwen3asr-serial` or a split-decoder bundle.
- `VibeVoiceAsr` line 556 is in `RunOnce`, shared between prefill and decode with a
  logits shape that changes between the two, so it needs a different (larger) change
  than the others — and there is no VibeVoice bundle here to regression-test it.
- Every remaining site on the issue's list is a prefill/init binding, which runs
  once per segment rather than once per token.

If someone wants to revisit this, the thing to measure first is whether the
`next_token` GPU-argmax output — not the buffer reuse — is what PR #45 actually
bought, and whether the other backends' decoders can be re-exported to provide it.

## Run 5 — 2026-09-07 08:20 — item 2, `RunAudioPipeline` extraction

Question: do the five `Transcribe*` methods' mel → encoder → projector preambles
really run the same code, or have they drifted?

They had drifted, but only in packaging, not in behaviour:

- three of five assert `projector tokens == NumAudioTokens(waveform)`, two do not;
- single-row paths assign their profile timers (`melLocal = …`), batched paths
  accumulate (`melLocal += …`);
- two spell the `Run` calls across multiple lines, three on one line.

The extracted helper always accumulates (`ref long melMs` with `+=`), which is
equivalent for the single-row callers since their locals start at 0, and returns
the projector's token count so each caller keeps its own assertion verbatim rather
than inheriting one it did not have. That keeps this a pure refactor: no path gains
or loses a check.

Result: 123 lines removed, 70 added (the helper plus five call sites), net −53 with
five copies of the front end collapsed to one. `GraniteSpeech.cs` 2512 → 2459 lines.

Parity, `--asr granite --vad` on `bench.wav` (32 segments, static-MHA bundle →
`TranscribeBatchStatic`): transcripts byte-identical before and after; 8371 ms vs
8792 ms wall, which is load-time jitter on a single run, not a regression in a
refactor that changes no arithmetic.

Only one of the five call sites can be executed on this machine — the other four
need bundle variants that are not here — so the remaining assurance is that the
replaced text was identical and the compiler accepts the substitution. This is
exactly the gap issue #46 item 3 (per-variant smoke coverage) exists to close.
