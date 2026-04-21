# VibeVoice-ASR 3090 Analysis

## Architecture Summary

`microsoft/VibeVoice-ASR-HF` is not a single opaque decoder-only graph. The published Hugging Face implementation cleanly separates into:

- two audio tokenizers:
  - `acoustic_tokenizer_encoder`
  - `semantic_tokenizer_encoder`
- a multimodal projection layer:
  - `multi_modal_projector`
- a Qwen2-family causal language model:
  - `language_model`

During generation:

- audio is encoded only on the first iteration
- audio placeholder tokens in the prompt are replaced with projected audio embeddings
- subsequent decode iterations run through the language model with `past_key_values`

That naturally suggests the following ONNX split:

1. `audio_encoder.onnx`
2. `decoder_prefill.onnx`
3. `decoder_step.onnx`

## 24 GB VRAM Model

Grounded from the published text config:

- layers: `28`
- hidden size: `3584`
- attention heads: `28`
- KV heads: `4`
- intermediate size: `18944`
- vocab size: `152064`

Implications:

- decoder weights dominate memory
- GQA keeps KV cache growth relatively modest
- long-form feasibility depends more on decoder weights plus runtime workspace than on KV cache alone

Initial estimates:

- decoder weights at 16-bit precision: about `14.2 GiB`
- KV cache at FP16/BF16:
  - `16K` tokens: about `0.875 GiB`
  - `32K` tokens: about `1.75 GiB`
  - `64K` tokens: about `3.5 GiB`

Interpretation:

- RTX 3090 24 GB should be treated as plausible but tight
- `30-minute` conversational audio is the first practical success gate
- `60-minute` single-pass operation is a stretch target, not the first acceptance bar

## Runtime Strategy

Baseline order:

1. PyTorch reference
2. ONNX Runtime CUDA
3. TensorRT only if the decoder clearly dominates wall time

Precision policy:

- PyTorch: `bfloat16` if stable, otherwise `float16`
- ONNX Runtime CUDA:
  - `bfloat16` remains the fidelity target because the published checkpoint is stored in `bfloat16`
  - `float16` is currently the stronger observed parity baseline on the single-decoder CUDA path
- TensorRT: `float16`

Quantization policy:

- not part of the first implementation path
- only revisit quantization if FP16/BF16 cannot meet the memory or throughput gates
- if revisited later, prefer selective decoder quantization instead of quantizing the audio encoders first

## Validation Plan

Local AMD laptop:

- export scripts
- graph inspection
- CPU smoke tests

Desktop RTX 3090:

- `5-minute`, `30-minute`, and `60-minute` runs
- record:
  - wall time
  - prefill time
  - tokens/sec
  - real-time factor
  - peak VRAM
  - max stable context length before OOM

## Current Status

### Export

- Model exported as `models/vibevoice_asr_single_bf16_f32kv/`:
  - `audio_encoder.onnx` — BF16 I/O, float32 Conv towers internally (ORT conv-22 compat workaround)
  - `decoder_single.onnx` — segmented interface (`prefix_input_ids / audio_embeddings / suffix_input_ids / past_key_N / past_value_N`), split float32 KV cache (56 separate tensors), `torch.export` at opset 18, no `ScatterND`
- Parity conclusion (f32 KV cache model, `ORT_ENABLE_EXTENDED`, ORT-only):
  - content words match PyTorch across the full tested range (up to 300 s audio)
  - divergences are timestamp second-decimal digits only (≤0.04 s difference); irreducible BF16 tiebreakers
  - `ORT_ENABLE_EXTENDED` is the correct decoder optimization level: avoids the BF16 attention fusion precision regression while still applying GeLU/LayerNorm fusions

### C# runtime (`VibeVoiceAsr.cs`)

- End-to-end working on RTX 3090 (600 s test recording, ~163 segments, full speaker-labeled transcript)
- Integrated into CLI via `--asr vibevoice` / `--vibevoice-model` / `--diarization vibevoice-asr-builtin`
- Key design points:
  - Audio encoder session lifetime is **mode-dependent** (controlled by `persistEncoder` constructor parameter):
    - **Built-in diarization** (`persistEncoder = false`): encoder is created and disposed inside each `Transcribe()` call, freeing its GPU arena (~1–3 GiB) before the long autoregressive decode loop begins — necessary to avoid OOM on 24 GB cards for recordings of 600 s or longer
    - **Segmented mode** (`persistEncoder = true`): encoder session is loaded once in the constructor and shared across all groups — avoids 70+ session-load round-trips (each ~20 s overhead)
  - KV cache is **GPU-resident** via `OrtIoBinding.BindOutputToDevice("present_key_N", cudaMemInfo)`: eliminates PCIe round-trip per decode step
  - Output bindings re-registered each call (`ClearBoundOutputs` + `BindOutputToDevice`) because KV shape grows each step
  - `OrtValue[]` ownership: `DisposableList` from `GetOutputValues()` has no finalizer; OrtValues extracted by reference and disposed pairwise between steps

### Measured performance (RTX 3090, 600 s recording)

| Configuration | RTF | Wall time | Notes |
|---|---|---|---|
| CPU float[][] KV (initial) | 2.78× | 1665 s | baseline |
| **Built-in diarization (whole-recording)** | **0.50×** | **301 s** | **recommended — best quality** |
| Static KV — Python benchmark | 0.38× | 226 s | PCIe was bottleneck; C# path not PCIe-bound |
| Static KV — C# end-to-end | 0.75× | 454 s | net negative: full-window attention over 6144 positions |
| VAD-segmented (70 groups, min 5 s) | 0.226× | 134 s | faster but lower quality; use only when OOM on built-in |
| DiariZen + VibeVoice (30 s groups) | 0.3325× | ~200 s | poor quality — boundary hallucination |
| DiariZen + VibeVoice (60 s groups, 2 s buffer) | ~0.42× | ~250 s | fair quality — still worse than built-in |

**Quality ranking (RTX 3090, 600 s English conversation):**

1. DiariZen + Parakeet — best speaker accuracy and transcription fidelity (English-only)
2. VibeVoice built-in diarization — strong quality, no boundary artifacts, good for multilingual
3. VAD-segmented VibeVoice — acceptable for very long recordings that OOM on built-in
4. DiariZen + VibeVoice segmented — not recommended: boundary hallucination negates diarization benefit

The VAD-segmented speedup (2.2× over whole-recording) comes from shorter KV windows per group — attention cost is quadratic in sequence length. However, group boundaries introduce hallucination artifacts and timestamp discontinuities that the whole-recording path avoids entirely. The built-in path produces noticeably cleaner output: correct speaker attribution at rapid turn-taking boundaries, no duplicate segments, no mid-group hallucinations.

### VRAM ceiling

- Baseline with decoder loaded: ~16.6 GiB weights
- Audio encoder (float32 Conv towers): ~1–3 GiB
  - **Built-in mode**: disposed after encoding, before the long decode loop — required to avoid OOM at 600 s on 24 GB
  - **Segmented mode**: persistent for the `VibeVoiceAsr` instance lifetime — avoids per-group session-load overhead
- KV cache growth per segment (5–9 s audio in segmented mode, ~40–70 tokens): negligible
- Chunked prefill used (`prefillChunkTokens = 512`) in both modes
- Max stable audio (single-pass built-in): 600 s confirmed on 24 GB with encoder disposed before decode

## ORT Profiling Results (2026-04-13)

One run with `SessionOptions.EnableProfiling` on the 600 s test recording (RTX 3090, ORT 1.24.2).
Decoder trace captured ~511 representative mid-run decode steps (from ~58 s to ~80 s wall time) before
ORT's hardcoded 1 M event cap; encoder trace was complete.

### Where time goes inside the decode loop

Of the 20.7 s wall-clock window analysed, **44% is node op dispatch** and the other **56% is ORT
framework overhead** — `SequentialExecutor` dispatch, CUDA kernel launch latency (~7 µs × ~1950 kernel
launches per step ≈ 14 ms/step of pure launch tax), and IO binding machinery.  Because output shapes
grow each step (KV cache), CUDA graphs cannot be used; this overhead is structural to ORT autoregressive
decode with dynamic shapes.

Node dispatch breakdown (of the 44% that is actual ops):

| Op | Time | % of nodes | What it is |
|---|---|---|---|
| MatMul | 1.85 s | 20% | Linear projections — near memory-bandwidth floor |
| Concat | 1.30 s | 14% | KV-cache grow ops (123/step: 56 KV + attention-seq concats) |
| Add | 1.02 s | 11% | Residual connections, biases |
| Mul | 0.94 s | 10% | SwiGLU gate, RoPE |
| Cast | 0.79 s | 9% | BF16↔F32 throughout |
| Transpose | 0.49 s | 5% | KV layout rearrangement |
| Reshape | 0.33 s | 4% | Shape ops |
| FusedMatMul | 0.33 s | 4% | Fused attention projection (good) |
| ReduceMean | 0.32 s | 3% | RMSNorm (unfused components) |
| BiasSoftmax | 0.30 s | 3% | Attention softmax |
| Split/Expand/Pow/etc. | 0.79 s | 9% | RoPE, LayerNorm tail, masking |

### What this confirms and rules out

- **MemcpyToHost: 0.014 s total** — PCIe is not a bottleneck; the 5 ORT-inserted Memcpy warnings at
  startup are negligible at runtime.
- **KV Concat (14%) is real overhead** — 123 Concat ops per step, growing each step, partially explain
  why RTF degrades on longer audio.  The Concat is inside the ONNX graph and cannot be changed without
  a re-export.
- **Cast (9%)** — BF16↔F32 transitions at KV boundaries; the float32 KV cache export causes these
  casts at every layer's K/V store.
- **ORT framework overhead (56%)** — the single largest contributor.  Unreachable without CUDA graphs
  or a different inference engine (TensorRT, vLLM PagedAttention).

### Encoder

Same 44%/56% split.  Conv 21%, MatMul 19%, FusedMatMul 11%.  No actionable hotspot; Conv is the
acoustic tokenizer frontend and is unavoidable.

## Current Conclusions

- The model is end-to-end working in C# on a 3090 at **RTF 0.50×** for 600 s audio in built-in diarization mode.
- Content parity with the PyTorch reference is acceptable across the full tested range.
- The **dynamic decoder** (`decoder_single.onnx`) with GPU IO binding is the production config.
- **Built-in diarization** (`--diarization vibevoice-asr-builtin --asr vibevoice`) is the **recommended pipeline**: whole-recording context produces the cleanest output — correct speaker attribution at rapid turn-taking, no boundary hallucinations, no duplicate segments.
- Static KV (`decoder_single_static.onnx`) was built and benchmarked but is net-negative for the C# GPU IO binding path: the fixed-size 6144-position attention window costs more in attention MACs than the Concat ops it eliminates.
- `ORT_ENABLE_EXTENDED` is used for the decoder (correctness requirement; avoids BF16 attention fusion regression).
- The audio encoder's float32 Conv workaround is stable; true BF16 Conv-22 requires a newer ORT build.
- The dominant per-step bottleneck is the 56% ORT kernel-dispatch overhead (structural to dynamic-shape autoregressive decode; unreachable without CUDA graphs or TensorRT).

### Segmented mode (VAD / DiariZen)

Implemented and available but **not recommended as the default**. Use only when the recording is long enough to OOM in built-in mode:

- `VadSegmenter.MergeShortGroups()` accumulates VAD segments until a minimum span is met (default 5 s), preventing language detection errors on isolated short clips.
- `--min-asr-seconds` and `--asr-buffer` control group size and boundary padding.
- Audio encoder session is kept persistent across groups (`persistEncoder = true`) to avoid per-group session-load overhead.
- Hallucination at group boundaries is the primary quality risk: VibeVoice sees an artificial start-of-audio context for each group, which can produce spurious segments or incorrect timestamps near the cuts. 2 s of padding (`--asr-buffer 2.0`) helps but does not eliminate it.
- DiariZen + VibeVoice segmented is not recommended: DiariZen's shorter segments (~3 s average) create more boundary cuts, worsening hallucination. Use DiariZen + Parakeet instead for English content requiring precise speaker attribution.

## Next Recommended Steps

Performance improvements (in priority order):

1. ~~**Fix decoder `GraphOptimizationLevel` to `ORT_ENABLE_EXTENDED`**~~ — **done (2026-04-13)**. `MakeSessionOptions` now takes an optional `optLevel` parameter; decoder passes `ORT_ENABLE_EXTENDED`, audio encoder defaults to `ORT_ENABLE_ALL`.

2. ~~**Pre-upload audio embeddings to CUDA once**~~ — **retracted**: the ~32 MB figure was the total embedding array size, not per-chunk size. Each prefill chunk copies its own slice (~3.5 MB); every token is transferred exactly once regardless (9 × 3.5 MB ≈ 32 MB total). Pre-uploading would still move the same 32 MB — just in one call instead of nine. At PCIe 4.0 ×16 bandwidth (~64 GB/s), saving 8 copy calls saves ~0.5 ms against a 301 s wall time.

3. ~~**Investigate ORT Memcpy warnings**~~ — **resolved (2026-04-13)** via ORT profiling.
   `MemcpyToHost` = 0.014 s total across the entire 600 s run — not a meaningful cost.  The 5
   ORT-inserted Memcpy nodes at session load time are for shape tensors that ORT keeps on CPU; they
   do not represent a per-step copy bottleneck.

4. **Trim logits to last row before PCIe transfer** — during prefill, the logits output is
   `[1, seq_len, 152064]` BFloat16. At `seq_len=576` (512 audio + 64 prefix/suffix), that's ~175 MB
   per prefill chunk.  Only the last row is needed.  Options: (a) add a `Gather`/`Slice` node to the
   exported decoder graph so logits output is always `[1, 1, 152064]`, or (b) bind logits to CUDA and
   do argmax on-device.  Low priority: profiling confirmed PCIe transfer is not a bottleneck.

5. **Larger prefill chunk size** — current default is 512 audio tokens per chunk.  For 600 s audio
   (~4520 tokens) that's 9 chunks.  If VRAM headroom allows (after audio encoder dispose, ~6–8 GiB
   free), a single 4520-token prefill is feasible and eliminates per-chunk KV re-registration overhead.
   Profile peak VRAM during prefill to find the safe upper bound.

6. ~~**BF16 KV cache re-export**~~ — **tested and rejected (2026-04-13)**.
   Exported `models/vibevoice_asr_single_bf16_bf16kv/` without `--f32-kv-cache` and ran
   `compare-single` parity at 256 tokens (`--dtype bfloat16 --runtime cuda --ort-opt-level extended`).
   - Divergence at position **11** (vs 14 for f32 KV) — 3 tokens earlier
   - Divergence position is the whole-seconds digit of the first `End` timestamp.  Reference picks
     token `"1"` from an exact BF16 tie (both `"3"` and `"1"` at logit 22.125, margin 0.0);
     ONNX picks `"3"` at 22.203125.  Result: first segment boundary `10.28s` (reference) vs `3.38s`
     (ONNX) — a 7-second error that restructures the transcript into an extra spurious segment.
   - Content words still match; divergence is still a BF16 tiebreaker, not accumulated drift.
   - **Verdict**: the f32 attention computation in the f32 KV model pushes this tie 3 tokens further
     out and confines it to sub-second decimals.  BF16 KV accumulation allows it to hit a
     structurally important position earlier.  The parity regression is too impactful for production.
   - f32 KV remains the production config.  The C# runtime change (`_kvCacheIsFloat32` flag read
     from `export-report.json`) is kept as infrastructure for future dtype variants.

7. **Static KV allocation with Slice instead of Concat** — **exported, benchmarked, net-negative for C# (2026-04-14). Not used in production.**
   - New model at `models/vibevoice_asr_static_bf16_f32kv/decoder_single_static.onnx`
   - Graph: `ScatterElements: 56` replaces the 56 KV-growing `Concat` ops; structural `Concat: 342` unchanged
   - `DecoderSingleStaticWrapper` uses `StaticKVCache.update()` → `torch.scatter` → ONNX `ScatterElements`
   - New input: `kv_pos` int64 scalar; KV buffers pre-allocated `[1, 4, 6144, 128]` float32
   - Export artifact: static model `audio_embeddings` and logits are `float16` (dynamic model uses `bfloat16`);
     root cause is `torch.onnx.export(dynamo=True, strict=False)` with varargs KV buffers;
     C# runtime auto-detects via `InputMetadata.ElementDataType` and casts BFloat16 → Float16 on feed
   - Parity vs dynamic decoder (600 s audio, 256 tokens): 96.9% token match;
     divergence is timestamp digit tiebreaker (float16 vs bfloat16 audio_embeddings), not a scatter bug
   - **Python benchmark**: static 0.38× vs dynamic 0.46× RTF — 1.23× faster (static shapes allow ORT plan caching, PCIe was the bottleneck)
   - **C# end-to-end**: static **0.75×** vs dynamic **0.50×** RTF — **1.50× SLOWER**
   - **Root cause**: attention mask uses `kv_length = max_tokens = 6144` (fixed), so Q×K^T runs over all 6144 KV positions every step regardless of fill level. Dynamic model grows from 0 → actual length. At prefill chunk 0 (kv_pos=0), static does 10.7× more attention MACs; at end of decode (kv_pos≈4630), still 1.33× more. C# was already GPU-resident for KV (no PCIe to save), so only the extra attention cost remains.
   - **Verdict**: Static KV helps when PCIe round-trip is the bottleneck (Python / naive ORT). It hurts when inference is already GPU-resident and attention dominates. Dynamic decoder remains the C# production config at RTF 0.50×.
   - Infrastructure kept: `decoder_single_static.onnx` + C# static path are preserved for future use with FlashAttention-style variable-length attention or TensorRT which could gate out unwritten KV positions.

8. **TensorRT EP for the decoder** — TRT would offer better CUDA kernel fusion and potentially
   30–50% improvement on the 44% of time that *is* in node ops.  The 56% ORT dispatch overhead is
   not addressable by TRT via the ORT TRT EP; it requires the TRT native Python runtime or a custom
   C++ harness.  Defer until the model re-export items above are complete.

9. **INT8/INT4 weight quantization** — at RTF 0.50× the throughput may already be adequate.  If
   longer recordings or lower-end GPUs are needed, selective INT8 quantization of the decoder weights
   (keeping audio encoder in BF16) would reduce the 16.6 GiB weight footprint by ~2×, making 60+
   minute recordings on 16 GB GPUs feasible.

10. **Streaming / partial-output mode** — the current implementation waits for the full transcript
    before returning.  For UI responsiveness, expose a progress callback (or
    `IAsyncEnumerable<VibeVoiceSegment>`) that yields segments as they are generated by the greedy
    decoder.  Requires detecting complete JSON segment objects in the token stream mid-generation.
