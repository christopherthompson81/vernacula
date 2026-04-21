# Qwen3-ASR C# Runtime Progress

The export side (Python, under `scripts/qwen3asr_export/`) has its own README documenting the export pipeline, the experimental batching artifacts, the empirical VRAM sweep, and the mixed-length parity verifier. This document covers what the **C# runtime** (`src/Vernacula.Base/Qwen3Asr.cs`) does with those artifacts — specifically, how runtime paths are selected and where the known optimisation gaps are. There is no equivalent of this doc in the repo; the only source of truth has historically been the C# source itself.

## Runtime graph selection

At construction, `Qwen3Asr` probes the model directory for three optional ONNX files and builds an internal path matrix ([Qwen3Asr.cs:288-291](../../src/Vernacula.Base/Qwen3Asr.cs#L288-L291)):

| File | Purpose |
|---|---|
| `decoder.onnx` | Unified decoder — prefill and step in one graph. Preferred when present. |
| `encoder_batched.onnx` | Batched encoder with per-row `input_lengths` masking. Required for any batched path. |
| `decoder_init_batched.onnx` | Legacy split-decoder batched prefill. Fallback when the unified decoder is absent. |

`_preferBatched = preferBatched && hasBatchedEncoder && (hasBatchedInit || hasUnified)` — callers ask for batched explicitly via the constructor flag; the runtime only honours it when the required artifacts are present.

Four distinct decode paths exist and are selected dynamically:

| Path | Entry point | Graph | IOBinding |
|---|---|---|---|
| Serial / unified | `DecodeOnCpuUnified` | `decoder.onnx` | ❌ (CPU KV extract-and-rebind each step) |
| Serial / CUDA IOBinding | `DecodeWithIoBinding` ([line 1755](../../src/Vernacula.Base/Qwen3Asr.cs#L1755)) | split init + step | ✅ |
| Serial / CPU split | `DecodeOnCpu` | split init + step | ❌ |
| Batched continuous | `RecognizeUnifiedContinuousBatched` | unified or batched-split | ❌ |

Serial dispatch ([Qwen3Asr.cs:761-765](../../src/Vernacula.Base/Qwen3Asr.cs#L761-L765)):
```
_decoder is not null       → DecodeOnCpuUnified
else _useCudaIoBinding     → DecodeWithIoBinding
else                       → DecodeOnCpu
```

Batched dispatch ([Qwen3Asr.cs:939](../../src/Vernacula.Base/Qwen3Asr.cs#L939)): when the unified decoder is loaded, batched segments go through `RecognizeUnifiedContinuousBatched`; otherwise they fall back to the split-decoder batched path.

## Continuous batching with KV compaction

The batched path is not a static batch — it implements continuous batching with per-step KV compaction ([`CompactStepKvInto`, line 1246](../../src/Vernacula.Base/Qwen3Asr.cs#L1246)):

- Each row in the batch tracks its own `pastLength` and an `activeMask` bit.
- After each decode step, `CompactStepKvInto` re-packs the present KV tensor: each row keeps its real past positions plus the one new token, padded to `maxNextLen = max(pastLengths + activeMask)`.
- Rows that have emitted EOS (cleared in `activeMask`) stop accumulating KV; shorter rows don't keep fake padding positions alive.

This is closer in spirit to vLLM's packed KV than to a fixed-B static batch. The parity property — mixed-length batched output matches serial output on the same inputs — is verified by the export-side `verify_qwen3_asr_unified_batch_parity.py`.

## Batch sizing — empirical, not analytical

Batch sizing is derived from the export-side CUDA OOM sweep, not from an analytical cost model. Runtime helpers:

- `EstimateExperimentalBatchCap(maxSegmentSeconds, freeGpuMemoryMb)` — scales a calibrated reference cap linearly with available VRAM.
- `EstimateExperimentalTotalSecondsCeiling(freeGpuMemoryMb)` — per-run audio-seconds ceiling.
- `ComputeExperimentalBatchingPlan(durations, freeGpuMemoryMb)` — returns a `QwenBatchingPlan` record for the whole run.

The reference point comes from `sweep_qwen3_asr_batching.py`, which runs each `(duration, batch_size)` sweep point in a fresh child process to avoid CUDA allocator pollution (export README line 112). The `QwenBatchingPlan` is therefore a transcription of *measured* safe regions, not a prediction from architecture constants.

When NVML is unavailable (CPU-only builds or missing cudart), `freeGpuMemoryMb` falls back to `FallbackFreeVramMb = 8_000` rather than clamping to batch=1, so CPU runs still get some batching.

## Architecture constants

Read from `config.json → decoder` at construction ([Qwen3Asr.cs:276-281](../../src/Vernacula.Base/Qwen3Asr.cs#L276-L281)):

- `hidden_size`, `num_layers`, `num_key_value_heads`, `head_dim`, `vocab_size`
- `embed_tokens_shape[0]` → base vocab size for the mmap-backed embed lookup
- `special_tokens.eos_token_ids` — multi-valued EOS set

KV shape on the wire: `[n_layers, batch, n_kv_heads, seq_len, head_dim]` — packed layer dim, different from VibeVoice/Cohere (per-layer split tensors).

## Known optimisation gap

**Batched path does not use IOBinding.** `_useCudaIoBinding = false` on every path that loads the unified decoder ([Qwen3Asr.cs:296](../../src/Vernacula.Base/Qwen3Asr.cs#L296)) and IOBinding is implemented only in `DecodeWithIoBinding`, which is the split-decoder single-stream path. The batched continuous path extracts KV to CPU each step via `ExtractTensor` and re-binds as a fresh `DenseTensor` next step ([Qwen3Asr.cs:1549](../../src/Vernacula.Base/Qwen3Asr.cs#L1549), [1576](../../src/Vernacula.Base/Qwen3Asr.cs#L1576), [1627-1662](../../src/Vernacula.Base/Qwen3Asr.cs#L1627-L1662), [1702-1727](../../src/Vernacula.Base/Qwen3Asr.cs#L1702-L1727)).

This is structurally similar to VibeVoice's pre-IOBinding state. The corresponding speedup on VibeVoice was 5.5×, although the analogy is imperfect: Qwen3's KV compaction step writes into a fresh host buffer each step, so GPU-resident KV would need to compose with a GPU-side compaction kernel — not a straight drop-in of the VibeVoice pattern.

The decision to implement this should be driven by a Qwen3 perf profile (not yet run) rather than by a generalised inference-layer refactor. When the profile data exists, tracking the work here is the right place; opening a GitHub issue before then risks another aspirational-but-unpursued task.

## Out-of-scope reminders

- **Analytical cost model.** Replacing the empirical sweep-based sizing with an analytical formula was considered and rejected during the [inference_abstraction_investigation](inference_abstraction_investigation.md): the empirical calibration is the *intended* design, not a hack. A future perf investigation may revisit it with calibration-vs-actual data.
- **Shared `KvCacheBinding` abstraction.** The four ASR backends' KV layouts are too divergent (unified packed vs per-layer split vs per-layer split + cross vs merged-branch) for a non-facade abstraction without model re-exports. See the abstraction investigation doc for the full reasoning.
