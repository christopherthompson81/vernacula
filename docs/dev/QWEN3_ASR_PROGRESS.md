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

Five distinct decode paths exist and are selected dynamically:

| Path | Entry point | Graph | IOBinding |
|---|---|---|---|
| Serial / unified GPU | `DecodeOnGpuUnified` | `decoder.onnx` | ✅ (GPU-resident KV across steps) |
| Serial / unified CPU | `DecodeOnCpuUnified` | `decoder.onnx` | ❌ (CPU KV extract-and-rebind each step) |
| Serial / CUDA IOBinding split | `DecodeWithIoBinding` ([line ~1870](../../src/Vernacula.Base/Qwen3Asr.cs#L1870)) | split init + step | ✅ |
| Serial / CPU split | `DecodeOnCpu` | split init + step | ❌ |
| Batched continuous | `RecognizeUnifiedContinuousBatched` | unified or batched-split | ❌ |

Serial dispatch ([Qwen3Asr.cs:~761](../../src/Vernacula.Base/Qwen3Asr.cs#L761)):
```
_decoder is not null && _useCudaIoBinding → DecodeOnGpuUnified
_decoder is not null                       → DecodeOnCpuUnified
_useCudaIoBinding                          → DecodeWithIoBinding
else                                       → DecodeOnCpu
```

`_useCudaIoBinding` is set in the constructor from `encoderUsesCuda && decoderUsesCuda`. For the unified decoder it is only set in the non-batched branch, because the batched continuous path has its own KV handling and does not consume it.

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

## Remaining optimisation gap

**Batched continuous path still extracts KV to CPU every step.** `RecognizeUnifiedContinuousBatched` outputs `present_keys` / `present_values` to host, runs `CompactStepKvInto` to re-pack per-row with `pastLengths` + `activeMask`, then re-binds from the host buffer on the next step. This is NOT a missing IOBinding — it is the compaction running on CPU, which forces the round trip regardless.

To close this gap, one of:
1. Move compaction to GPU (a small CUDA gather/scatter kernel, or added ONNX ops on the decoder graph).
2. Drop compaction; rely entirely on `attention_mask` to exclude stale positions. Requires re-running `verify_qwen3_asr_unified_batch_parity.py` and accepting the extra attention compute on stale positions indefinitely.
3. Keep compaction on host; bind output to device and read to CPU only for compaction. Saves one transfer per step, not two — marginal.

Option 1 is the right fix but is a meaningful piece of work, not a refactor. Decide by profiling the batched path first to measure how much per-step wall-clock the KV round trip actually costs.

**Resolved in the serial path.** The IOBinding gap for `decoder.onnx` on the single-segment path is closed by `DecodeOnGpuUnified` (see the decode-paths table above). Editor re-transcribe and serial CLI runs now keep KV GPU-resident between decode steps when CUDA is active. The change is untested end-to-end against a baseline — needs a WER + RTF check before declaring it a win.

## Out-of-scope reminders

- **Analytical cost model.** Replacing the empirical sweep-based sizing with an analytical formula was considered and rejected during the [inference_abstraction_investigation](inference_abstraction_investigation.md): the empirical calibration is the *intended* design, not a hack. A future perf investigation may revisit it with calibration-vs-actual data.
- **Shared `KvCacheBinding` abstraction.** The four ASR backends' KV layouts are too divergent (unified packed vs per-layer split vs per-layer split + cross vs merged-branch) for a non-facade abstraction without model re-exports. See the abstraction investigation doc for the full reasoning.
