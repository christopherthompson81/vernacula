# Inference Abstraction Investigation

> **Status: superseded.** The abstraction programme stopped after PR 2. Runs 1 and 2 are preserved as the record of what was actually delivered and why. Run 3 documents the reasoning that closed the plan. See [QWEN3_ASR_PROGRESS.md](QWEN3_ASR_PROGRESS.md) for the Qwen3 runtime details that surfaced during the wind-down.

## Goal

Vernacula ships six ASR backends ([WhisperTurbo](../../src/Vernacula.Base/WhisperTurbo.cs), [VibeVoiceAsr](../../src/Vernacula.Base/VibeVoiceAsr.cs), [CohereTranscribe](../../src/Vernacula.Base/CohereTranscribe.cs), [Qwen3Asr](../../src/Vernacula.Base/Qwen3Asr.cs), [Parakeet](../../src/Vernacula.Base/Parakeet.cs), [IndicConformer](../../src/Vernacula.Base/IndicConformer.cs)). Each evolved independently and now implements overlapping inference-level optimizations in slightly divergent ways. This investigation catalogs the overlap, identifies shared abstractions worth extracting, and sequences the refactor so every backend can pick up the most advanced features in use anywhere in the repo.

The motivating concrete target: generalize VibeVoice's IOBinding + GPU-resident KV cache pattern and the (still-unimplemented) "decoder scheduler" for horizontal batching of autoregressive decode steps across segments, so that Whisper, Cohere, and Qwen3 benefit automatically rather than each reimplementing it.

## Scope

**In:** shared helpers for session construction, VRAM-aware batch planning, IOBinding/KV cache reuse, GPU mel frontend, autoregressive decode-step scheduling, diagnostics.

**Out:** tokenizer/vocab unification (too format-specific), timestamp construction (model-native), export-time graph surgery, model-specific prefix token assembly.

**Excluded backends from AR-specific work:** Parakeet (RNN-T, stateless) and IndicConformer (CTC, stateless) — they benefit from the session builder and GPU mel only.

## Run 1 — 2026-04-21

**Question:** Which inference-level features exist in each backend today, and which of those factor into shared abstractions cleanly?

**Method:** Structured survey of `src/Vernacula.Base/*.cs` entry-points and cross-reference against `docs/dev/*_PROGRESS.md` / `*_investigation.md` notes.

### Feature inventory (per-backend)

| Feature | Whisper-turbo | VibeVoice | Cohere | Qwen3 | Parakeet | IndicConformer |
|---|---|---|---|---|---|---|
| Dynamic/batched decoding | segmented, batch=8 | single-stream | VRAM-budgeted sorted batch | VRAM-budgeted sorted batch | frame-padded | shared encoder |
| KV cache | merged `use_cache_branch` | split fp32, chunked prefill | cross/self KV | packed prefill + step split | n/a (RNN-T) | n/a (CTC) |
| IOBinding | phase 3 (pending) | ✅ GPU-resident KV | ✅ per-step | ❌ | ❌ | ❌ |
| Quantization | FP16 / INT8 / Q4 | FP32 (BF16 KV rejected for parity) | FP32 | FP32 | FP32 | FP32 |
| Beam search | greedy | greedy | greedy | greedy | ✅ TDT beam | greedy CTC |
| Prefix/language conditioning | SOT tokens | prefix+suffix+digit tokens | 10-token context block | role+language prefix | ❌ | per-span vocab |
| ORT opt level | default | `ORT_ENABLE_EXTENDED` + attn-fusion avoidance | default | default | default | default |

Duplicated patterns found:

- `MakeSessionOptions()` is redefined in all six backends: [Parakeet.cs:81](../../src/Vernacula.Base/Parakeet.cs#L81), [CohereTranscribe.cs:167](../../src/Vernacula.Base/CohereTranscribe.cs#L167), [Qwen3Asr.cs:784](../../src/Vernacula.Base/Qwen3Asr.cs#L784), [VibeVoiceAsr.cs:1095](../../src/Vernacula.Base/VibeVoiceAsr.cs#L1095), [WhisperTurbo.cs:1304](../../src/Vernacula.Base/WhisperTurbo.cs#L1304), [IndicConformer.cs:70](../../src/Vernacula.Base/IndicConformer.cs#L70). Each repeats the CUDA → DML → CPU fallback chain and arena config.
- Mel spectrogram frontend duplicated on CPU in [WhisperTurbo.cs:116](../../src/Vernacula.Base/WhisperTurbo.cs#L116) and [Qwen3Asr.cs:58](../../src/Vernacula.Base/Qwen3Asr.cs#L58). Both pay ~250 ms CPU cost per call ([whisper_turbo_investigation.md](whisper_turbo_investigation.md) Run 4, [COHERE_TRANSCRIBE_PERFORMANCE.md:43](COHERE_TRANSCRIBE_PERFORMANCE.md)).
- VRAM-budgeted batch sizing implemented twice, nearly identically, in Cohere and Qwen3 (`EstimateKvBytes`, `EstimateMelFrames`, duration-sorted packing).
- IOBinding + GPU-resident KV cache is a VibeVoice-only pattern that Cohere partially replicates and Qwen3/Whisper lack entirely ([VibeVoiceAsr.cs:254-260](../../src/Vernacula.Base/VibeVoiceAsr.cs#L254-L260)).

**Correction to earlier assumption:** the "VibeVoice Decoder Scheduler" is *not* shipped. It is a deferred optimization in [VIBEVOICE-ASR_EXPORT_PROGRESS.md](VIBEVOICE-ASR_EXPORT_PROGRESS.md). What ships today is the IOBinding pattern and a `decoder_single_static.onnx` static-KV variant. The scheduler itself should be built *as shared infrastructure*, not as a VibeVoice-only feature.

**Implication:** A composition-based set of helpers in a new `src/Vernacula.Base/Inference/` folder is the right shape. Inheritance is the wrong tool because RNN-T (Parakeet) and CTC (IndicConformer) don't share the autoregressive decode loop, but they do share session construction and mel.

## Proposed abstractions

All new code under `src/Vernacula.Base/Inference/`.

### 1. `OrtSessionBuilder`
```csharp
public static class OrtSessionBuilder {
    public static InferenceSession Build(
        string modelPath,
        ExecutionProvider ep,
        SessionOptions? overrides = null);
}
```
Defaults: `ORT_ENABLE_EXTENDED`, attention-fusion avoidance flag (VibeVoice's finding), CUDA → DML → CPU fallback, shared arena config.

### 2. `BatchSizer` + `IBatchCostModel`
```csharp
public interface IBatchCostModel {
    long KvBytesPerToken(int batchSize, int seqLen);
    long ActivationBytes(int batchSize, int melFrames);
}
public static class BatchSizer {
    public static IReadOnlyList<Batch> Plan(
        IReadOnlyList<Segment> segments,
        IBatchCostModel costs,
        long vramBudgetBytes,
        int maxBatch);
}
```
Each AR backend implements a ~20-line cost model. Whisper and VibeVoice adopt; Cohere and Qwen3 retire their bespoke copies.

### 3. `KvCacheBinding`
```csharp
public sealed class KvCacheBinding : IDisposable {
    public KvCacheBinding(OrtIoBinding binding, KvLayout layout, KvGrowthMode mode);
    public void BindForStep(int stepIndex);
    public void ReleaseSegment(int segmentId);
}
```
Generalizes VibeVoice's IOBinding. `KvGrowthMode` covers dynamic-grow vs static-max. Portable to Cohere, Qwen3, Whisper (once Whisper's decoder is re-exported with a split prefill/step graph).

### 4. `GpuMel`
Shared `mel.onnx` parameterized by `(n_fft, hop, n_mels, sr)`. Loaded via `OrtSessionBuilder`. Eliminates the duplicated CPU mel path.

### 5. `AutoregressiveDecoder` + `IArStep`
```csharp
public interface IArStep {
    int[] Prefix(Segment s);
    float[] RunStep(int[] tokens, KvCacheBinding kv);
    bool IsEos(int tokenId);
}
public static class AutoregressiveDecoder {
    public static int[] DecodeGreedy(IArStep step, Segment seg, KvCacheBinding kv, int maxNewTokens);
    public static int[][] DecodeBatchedGreedy(IArStep step, IReadOnlyList<Segment> segs, ...);
}
```
This is where the decode-step scheduler lives. `DecodeBatchedGreedy` batches steps horizontally across segments. Whisper / VibeVoice / Cohere / Qwen3 each implement `IArStep`; scheduler logic is written once.

### 6. `InferenceProfiler`
ORT profile + NVML sampling wrapper so all backends emit comparable traces.

## Sequenced plan

Four PRs, each independently shippable, each narrower in scope than the last:

### PR 1 — `OrtSessionBuilder` + adopt everywhere
Mechanical, zero behavior change aside from universal `ORT_ENABLE_EXTENDED` + attention-fusion avoidance (a known-good upgrade from VibeVoice's investigation).

Concrete file moves:
- **New:** `src/Vernacula.Base/Inference/OrtSessionBuilder.cs`, `src/Vernacula.Base/Inference/ExecutionProvider.cs`
- **Modify (delete local `MakeSessionOptions` + call new builder):**
  - [Parakeet.cs:81](../../src/Vernacula.Base/Parakeet.cs#L81)
  - [CohereTranscribe.cs:167](../../src/Vernacula.Base/CohereTranscribe.cs#L167)
  - [Qwen3Asr.cs:784](../../src/Vernacula.Base/Qwen3Asr.cs#L784)
  - [VibeVoiceAsr.cs:1095](../../src/Vernacula.Base/VibeVoiceAsr.cs#L1095)
  - [WhisperTurbo.cs:1304](../../src/Vernacula.Base/WhisperTurbo.cs#L1304)
  - [IndicConformer.cs:70](../../src/Vernacula.Base/IndicConformer.cs#L70)

Validation: existing unit tests + golden-output transcription tests on one reference clip per backend. No WER change expected; watch for regressions from the opt-level bump.

### PR 2 — `BatchSizer` + adopt in Whisper / VibeVoice
Extract cost-model surface from Cohere and Qwen3 (both already have the logic). Whisper moves from fixed batch=8 to dynamic; VibeVoice gains cross-segment batching where it currently runs single-stream.

### PR 3 — `KvCacheBinding` + port to Qwen3, slot into Whisper phase 3
Extract from VibeVoice. Qwen3 adoption is the single biggest per-backend speedup on the table (VibeVoice saw 5.5× from this pattern). Whisper phase 3 is already queued and should build on this rather than roll its own.

### PR 4 — `AutoregressiveDecoder` + cross-segment decode-step scheduler
Largest change. Depends on PRs 1–3. Likely needs its own investigation doc before starting; specifically, Whisper needs a decoder re-export with split prefill/step graphs (matching the Cohere/Qwen3 pattern) for `IArStep` to plug in. Benefits all four AR backends at once.

## Open questions

- Can `ORT_ENABLE_EXTENDED` be applied uniformly without regressing Parakeet or IndicConformer? VibeVoice's finding was AR-specific — need a validation run per backend in PR 1.
- GPU mel: does it share the `InferenceSession` with the main encoder (one session, multiple subgraphs) or run as a sidecar session? Latency vs memory tradeoff — decide during PR design, not now.
- Whisper decoder re-export for PR 4: can we consume the existing onnx-community split export instead of re-exporting in-house? Needs a separate spike.

## Run 2 — 2026-04-21

**Question:** Extract `BatchSizer` + `IBatchCostModel` from Cohere and Qwen3 as PR 2 outlined.

**Finding — the two backends are structurally different, not merely duplicated.**

- **Cohere** uses an analytical cost model ([CohereTranscribe.cs:118](../../src/Vernacula.Base/CohereTranscribe.cs#L118)): compute exact bytes from architecture constants (layers, heads, head dim, conv channels) and duration-derived frame counts, then greedily grow each batch one segment at a time until peak bytes exceed the VRAM budget.
- **Qwen3** uses an empirical calibration ([Qwen3Asr.cs:421](../../src/Vernacula.Base/Qwen3Asr.cs#L421)): scale a reference batch cap (measured at a known-good VRAM figure) linearly by the current free VRAM. Returns a whole-run `QwenBatchingPlan` record, not per-batch packings.

A unified abstraction covering both shapes would either bloat the return type or force Qwen3 into the analytical mould — which isn't a refactor, it's a reverse-engineering job on Qwen3's cost structure (layers/heads/dim are not currently surfaced the way Cohere's are).

**Implication:** scope PR 2 to the analytical pattern only. Qwen3's migration becomes a separate question ("should Qwen3 move from empirical to analytical cost modelling?") that belongs in its own investigation, driven by calibration-vs-actual data, not by this refactor.

**Also out of scope for PR 2**: adopting the new `BatchSizer` in Whisper and VibeVoice. Both are working-and-tested with their current batching (Whisper fixed batch=8 after phase 6 validation; VibeVoice single-stream with IOBinding). Swapping their batching strategy is a behavioural change that needs its own WER/perf validation, not a drop-in refactor. Ship the shared surface first; defer adoption until each backend is being actively tuned.

**Delivered:**
- [BatchSizer.cs](../../src/Vernacula.Base/Inference/BatchSizer.cs) — `IBatchCostModel` interface, `Batch` record (original-index array), `BatchSizer.Plan` static helper. Preserves Cohere's forward-progress guarantee: the first segment of each batch is always admitted even if it alone would breach the budget.
- [CohereTranscribe.cs](../../src/Vernacula.Base/CohereTranscribe.cs) refactored: inline sort-and-pack loop replaced by `BatchSizer.Plan`, cost-model logic moved into a nested `CohereBatchCostModel : IBatchCostModel`. The existing `Estimate*` private statics stay on `CohereTranscribe` (they encode model-specific architecture constants, which do not belong on the shared abstraction).
- Zero behaviour change: same sort key, same peak-bytes formula, same `MaxBatchSize = 32` cap, same VRAM budget.

**Deferred follow-ups tracked as issues:**
- [#25](https://github.com/christopherthompson81/vernacula/issues/25) — port Qwen3 from empirical to analytical cost modelling, if justified by calibration data.
- [#26](https://github.com/christopherthompson81/vernacula/issues/26) — adopt `BatchSizer` in Whisper and VibeVoice, once each backend is in a validation cycle that can absorb the batching-strategy change.
- [#27](https://github.com/christopherthompson81/vernacula/issues/27) — validate `ORT_ENABLE_EXTENDED` as a universal graph-opt default (carry-over from PR 1 open question).

## Run 3 — 2026-04-21 (closing run)

**Question:** Extract `KvCacheBinding` from VibeVoice and port it to Qwen3 as PR 3.

**Findings that closed the plan:**

1. **Qwen3 already has IOBinding** on its single-stream split-decoder path ([Qwen3Asr.cs:1755](../../src/Vernacula.Base/Qwen3Asr.cs#L1755), `DecodeWithIoBinding`, gated on `_useCudaIoBinding`). The earlier survey that claimed Qwen3 lacked IOBinding was wrong. The PR 3 value proposition — "port VibeVoice's IOBinding to Qwen3 for the biggest backend speedup" — therefore evaporated.

2. **The four backends' KV layouts are structurally incompatible.** Per-layer split 56-tensor (VibeVoice), per-layer split self+cross 32-tensor (Cohere), unified packed `[n_layers, batch, heads, seq, dim]` (Qwen3), merged `use_cache_branch` (Whisper). A shared `KvCacheBinding` class that covers all four either degenerates into a thin facade over `OrtIoBinding` or forces one layout and requires model re-exports of the others. Neither is a refactor.

3. **The real remaining Qwen3 speedup is narrow and Qwen3-specific.** The batched continuous path extracts KV to CPU each step ([Qwen3Asr.cs:1549](../../src/Vernacula.Base/Qwen3Asr.cs#L1549) and siblings), not because it lacks IOBinding infrastructure but because its per-step KV compaction writes into a fresh host buffer. Making that GPU-resident needs a GPU-side compaction kernel — not shared-infrastructure work. Track it against real perf data, not against a plan.

4. **PR 2's Qwen3 "deferral" was based on a similar misreading.** The Qwen3 batch sizer is empirical by design, derived from the CUDA OOM sweep in `scripts/qwen3asr_export/sweep_qwen3_asr_batching.py` (export README line 119: *"Use this to derive a conservative runtime heuristic for choosing Qwen batch counts from free VRAM and planned batch duration"*). Replacing it with an analytical model would discard measured safe-region data for a derived approximation — not a refactor, a regression risk.

**Disposition:**

- **Plan closed.** PRs 3 and 4 are not pursued.
- **Issues #25, #26, #27 closed** as aspirational-but-unjustified work items.
- **Useful surface area retained.** `OrtSessionBuilder` (PR 1) and `BatchSizer` + `IBatchCostModel` (PR 2) remain in place — they were real deduplication wins with zero behaviour change and have value independent of the larger abstraction programme.
- **Qwen3-specific notes consolidated** into [QWEN3_ASR_PROGRESS.md](QWEN3_ASR_PROGRESS.md) so the next reader does not have to re-derive runtime behaviour from the C# source.

**Lesson for future "unify the backends" plans:** sanity-check the survey claims against the actual code before committing to an abstraction. The three specific mis-reads in this programme (Qwen3 lacks IOBinding; Qwen3/Cohere share a cost model; the four KV layouts are variants of one pattern) all survived the initial Explore-agent survey and only collapsed on direct reading. When an abstraction claim rests on "these are duplicates," read both originals end-to-end before writing the shared helper.

## Status

**Programme closed.** Two PRs shipped:

- `b97c29c` — PR 1, `OrtSessionBuilder` extracted from six ASR backends.
- `0da2c80` — PR 2, `BatchSizer` + `IBatchCostModel` extracted from Cohere.

No further PRs. Remaining backend-specific optimisation work is tracked against perf data in the per-backend progress docs, not against this investigation.
