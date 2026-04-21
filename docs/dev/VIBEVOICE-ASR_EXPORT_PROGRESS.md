# VibeVoice-ASR Export Progress

## Goal

Investigate `microsoft/VibeVoice-ASR-HF` as a standalone ONNX export and benchmark target, with the primary feasibility question:

- Can the model run acceptably on an RTX 3090 with 24 GB VRAM using ONNX Runtime CUDA?
- If CUDA works but the decoder is the bottleneck, is TensorRT worthwhile for the decoder path?

This track is intentionally separate from Vernacula product integration for now. The first milestone is a reproducible export and benchmark harness.

## Current Environment

- Repo branch: `feature/vibevoice-asr-export`
- Local machine used for initial investigation: AMD laptop, no CUDA-visible NVIDIA GPU
- Consequence: export scaffolding, graph inspection, and CPU smoke tests can be developed locally, but real CUDA/TensorRT measurements must happen later on the desktop RTX 3090 machine

## Grounded Model Facts

Sources used during the initial investigation:

- Hugging Face model page: <https://huggingface.co/microsoft/VibeVoice-ASR-HF>
- Transformers docs: <https://huggingface.co/docs/transformers/main/model_doc/vibevoice_asr>
- Transformers implementation:
  - `modeling_vibevoice_asr.py`
  - `processing_vibevoice_asr.py`

Observed architecture shape from the published config and model implementation:

- Top-level model class: `VibeVoiceAsrForConditionalGeneration`
- Audio path:
  - `acoustic_tokenizer_encoder`
  - `semantic_tokenizer_encoder`
  - `multi_modal_projector`
- Text path:
  - `language_model = AutoModelForCausalLM.from_config(config.text_config)`
  - text config is Qwen2-family
- Audio is only consumed on the first generation iteration
- Subsequent decode steps run through the language model with `past_key_values`

This means the intended ONNX split is clean:

1. `audio_encoder.onnx`
2. `decoder_prefill.onnx`
3. `decoder_step.onnx`

## Memory / Feasibility Notes

Grounded from the published `text_config`:

- Hidden size: `3584`
- Layers: `28`
- Attention heads: `28`
- KV heads: `4`
- Intermediate size: `18944`
- Vocab size: `152064`
- Default dtype: `bfloat16`

Initial decoder-side estimate:

- Rough 16-bit decoder weight footprint: about `14.2 GiB`
- KV cache estimate with GQA:
  - `16K` tokens: about `0.875 GiB`
  - `32K` tokens: about `1.75 GiB`
  - `64K` tokens: about `3.5 GiB`

Interpretation:

- `24 GB` appears plausible for FP16/BF16 single-stream inference
- `64K` token operation on a 3090 is not obviously impossible, but it is tight once runtime overhead, projector/audio encoders, activations, and allocator fragmentation are included
- The safest first goal is not "60 minutes at any cost"
- The safest first goal is "prove stable 30-minute operation, then measure the longest stable context window"

## Decisions Locked In

- First pass is `precision-only`
- No quantization in the primary implementation path
- Default decoder investigation target is still `BF16` vs `FP16` parity on CUDA, not quantization
- Batch size target for the first benchmark pass is `1`
- CPU smoke-test default upper bound is `90 seconds`, not `300 seconds`
- TensorRT is deferred until ONNX Runtime CUDA profiling shows the decoder dominates wall time

## Export Design

Planned contracts:

### `audio_encoder.onnx`

Input:

- raw mono audio at 24 kHz

Output:

- projected decoder-ready audio embeddings

Notes:

- This stage wraps the two tokenizer encoders plus the multimodal projector
- It should mirror `get_audio_features(...).pooler_output`

### `decoder_prefill.onnx`

Inputs:

- prompt `input_ids`
- prompt `attention_mask`
- audio embeddings from `audio_encoder.onnx`

Outputs:

- logits for the last prompt position
- initial KV cache

Notes:

- This stage handles the first generation iteration, including replacement of audio placeholder tokens with projected audio embeddings

### `decoder_step.onnx`

Inputs:

- single next-token `input_ids`
- updated `attention_mask`
- prior KV cache

Outputs:

- next-step logits
- updated KV cache

## Immediate Next Steps

1. Add `scripts/vibevoice_export/`
2. Add export requirements and README
3. Implement:
   - `export_vibevoice_asr_to_onnx.py`
   - `inspect_vibevoice_onnx.py`
   - `smoke_test_vibevoice_onnx.py`
   - `benchmark_vibevoice_3090.py`
4. Record exact graph tensor names, dynamic axes, and external-data layout
5. Run local CPU smoke tests with:
   - very short clip: `15-30 seconds`
   - upper default clip: `90 seconds`

## Open Questions

- Whether the Hugging Face implementation exports cleanly with `torch.onnx.export` as-is, or whether Qwen2 cache wrappers need a custom adapter
- Whether ORT TensorRT EP can consume the exported decoder graphs directly with dynamic cache length
- How much memory headroom remains on the 3090 once the real checkpoint, audio encoder path, and ORT workspace overhead are measured
- Whether the ORT CUDA decoder numerics issue can be addressed via session options (e.g. disabling fused attention, forcing higher-precision intermediates) or requires a newer ORT build — the `ScatterND` and audio-replacement hypotheses have been ruled out
- Whether a newer ORT build/provider set can run a true `BF16` audio graph exported at `Conv-22`

## Iteration Log

### 2026-04-13 (third entry) — BF16 KV cache parity test

Tested `models/vibevoice_asr_single_bf16_bf16kv/` (exported without `--f32-kv-cache`).
Parity command: `compare-single --dtype bfloat16 --runtime cuda --ort-opt-level extended --max-new-tokens 256`.

#### Result: rejected — parity regression

| Model | First divergence | Nature of divergence |
|---|---|---|
| f32 KV (`vibevoice_asr_single_bf16_f32kv`) | position 14 | timestamp second-decimal digit (e.g. "4.02" vs "4.04") |
| BF16 KV (`vibevoice_asr_single_bf16_bf16kv`) | position **11** | whole-seconds digit of first segment End timestamp |

Divergence at position 11 is on the first `End` timestamp's whole-seconds digit.  The reference (PyTorch
BF16) has a 0.0-margin exact tie between `"3"` and `"1"` (both at logit 22.125); ONNX picks `"3"` at
22.203125.  This shifts the first segment boundary from `10.28s` to `3.38s` — a 7-second error that
splits the transcript into a spurious extra segment.  Content words still match.

Root cause: the f32 KV model computes attention in float32, slightly shifting the logit distribution so
the 0.0-margin tie is deferred from position 11 to position 14 and confined to a sub-second decimal.
BF16 KV accumulation allows the tie to surface 3 tokens earlier at a structurally important position.

**Decision: f32 KV remains the production config.**  The Cast and Concat overhead attributed to f32 KV
(~23% of node time, ~10% of wall time) is not worth the quality regression.  The C# `_kvCacheIsFloat32`
flag (read from `export-report.json`) is retained as infrastructure.

### 2026-04-13 (second entry) — ORT profiling

Added `SessionOptions.EnableProfiling` support to `VibeVoiceAsr` and the CLI (`--profile <dir>`,
`--profile-steps N`).  Ran a profiling capture on the 600 s test recording (RTX 3090, ORT 1.24.2).

#### Implementation notes

- `ProfileOutputPathPrefix` is not wired to the native `OrtEnableProfiling` call in ORT 1.24.2
  managed bindings — files always land in the process cwd.  Fixed by using `File.Move()` after
  `EndProfiling()` to route each trace to the requested directory as `encoder_*.json` /
  `decoder_*.json`.
- ORT has a hardcoded 1 M event limit (not configurable via session options config keys in 1.24.2).
  At ~1950 events/step this fills in ~511 decode steps.  Added `--profile-steps N` (default 200)
  to cap `maxNewTokens` for profiling runs so the trace is complete and fast.

#### Findings summary

See `VIBEVOICE_ASR_3090_ANALYSIS.md` for the full breakdown.  Key results:

- **56% of wall time** is ORT framework overhead (SequentialExecutor dispatch + CUDA kernel launch
  latency at ~1950 launches/step).  Structural to ORT autoregressive decode with dynamic shapes;
  CUDA graphs cannot be used because KV output shapes grow each step.
- **Concat = 14% of node time** — 123 KV-grow concat ops per step, baked into the ONNX graph,
  growing in cost as the KV cache length increases.
- **Cast = 9% of node time** — BF16↔F32 at every K/V store; consequence of the `--f32-kv-cache`
  export flag needed for content-word parity.
- **MemcpyToHost = 0.014 s total** — PCIe is not a bottleneck; ORT startup-time Memcpy warnings
  are for shape tensors, not per-step copies.  Ruled out as a meaningful cost.
- TensorRT would address only the 44% in node ops; the 56% dispatch overhead requires TRT's own
  runtime or a custom harness, not the ORT TRT EP.

#### Next steps derived from profiling

Two complementary re-export opportunities (not mutually exclusive — additive benefits):

1. **BF16 KV cache** — halves Concat/Transpose/Cast bandwidth; requires parity re-validation
2. **Static KV allocation + Slice** — eliminates Concat overhead entirely; orthogonal to KV dtype

### 2026-04-13 — C# end-to-end integration + IO binding KV cache optimization

#### CLI integration

Wired VibeVoice-ASR into the C# CLI (`Config.cs` / `Program.cs`):

- `--asr vibevoice` backend selector
- `--vibevoice-model <dir>` optional model directory override (falls back to `<modelDir>/vibevoice_asr`)
- `vibevoice-asr-builtin` as a valid `--diarization` value; auto-selected when `--asr vibevoice` is used
- builtin diarization path skips external diarization (Sortformer/DiariZen) and calls `VibeVoiceAsr.Transcribe()` directly

#### First working end-to-end run

On first run (600 s test recording, RTX 3090):

- 121 segments, full transcript, correct speaker labels
- **RTF 2.78×** (total wall time 1665 s)

Errors discovered and fixed during bringup:

| Error | Cause | Fix |
|---|---|---|
| `MatMul not found on CPU EP` | Audio encoder loaded with CPU session options; BFloat16 MatMul nodes in multimodal projector | Use `MakeSessionOptions(ep)` (CUDA) for audio encoder |
| `Tensor element data type discovered: Float, metadata expected: BFloat16` | `input_values` tensor created as float32 | Convert to `BFloat16[]` before creating tensor |
| `Tensor element data type discovered: BFloat16, metadata expected: Bool` | `padding_mask` created as BFloat16 | Use `DenseTensor<bool>` for padding mask |
| Audio encoder OOM (1.84 GB allocation) | Passing all 14.4 M samples (600 s × 24 kHz) to encoder at once | `RunAudioEncoderChunked`: processes 361,600-sample (~15 s) chunks |

#### IO binding optimization for GPU-resident KV cache

The 2.78× RTF was dominated by KV cache PCIe round-trips:

- 56 KV tensors (`past_key/value_0..27`), each `[1, 4, cache_len, 128]` float32
- Old approach: CPU `float[][]` → copy to GPU input → copy from GPU output each step
- At 4,720-token prefill cache: ~543 MB per direction per step, ~1 GB/token transferred, ~4 TB total for 4,000 decode tokens

Fix: `OrtIoBinding` with `BindOutputToDevice("present_key_N", cudaMemInfo)` keeps KV tensors GPU-resident. Key implementation notes:

- `OrtIoBinding.BindOutputToDevice(name, OrtMemoryInfo)` allocates on CUDA each call
- `GetOutputValues()` returns a `DisposableList<OrtValue>` — has **no finalizer**, so skipping `using` is safe; OrtValues stay alive via their SafeHandles
- Must call `ClearBoundOutputs()` + re-register outputs **every call** because KV shape grows each step and ORT cannot reuse pre-allocated buffers with different shapes
- Empty initial KV values: `OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), shape)` with `shape = {1, 4, 0, 128}` is valid
- Audio encoder session made short-lived (created + disposed inside `Transcribe`) so its GPU arena (float32 Conv towers, ~1–3 GB) is freed before the decode loop begins; otherwise OOM at ~step 6780 on a 24 GB card

Result after optimization:

- **RTF 0.50×** (total wall time 301 s for 600 s audio) — **5.5× speedup**
- 120 segments, same quality output

### 2026-04-12 (sixth entry)

- Extended parity tests to longer audio using chunked audio encoder
- Problem: acoustic tokenizer encoders process the full waveform in a single forward pass, causing CUDA OOM above ~150s in `compare-single` mode (8.3B model + large waveform intermediates exceed 24 GB)
- Fix: added `chunked_audio_context(model, chunk_size)` context manager in `smoke_test_vibevoice_onnx.py`:
  - patches `acoustic_tokenizer_encoder.forward` and `semantic_tokenizer_encoder.forward` in-place to process waveform in `chunk_size`-sample pieces (361,600 samples = 15.1s from export report)
  - latents are `[batch, time_frames, channels]`; chunks concatenated along `dim=1`
  - `_LatentsWrapper` proxies all attributes to the last real output, so `padding_cache` and other fields remain accessible
  - applied to the entire PyTorch reference block in `compare-single` mode (covers `pytorch_single_debug`, `pytorch_single_export_style_debug`, `analyze_single_decoder_components`, and `model.generate`)
- Fix: added `run_chunked_ort_audio_encoder()` helper; all three `ort_generate_single`-family functions now call it instead of `run_onnx_session` directly for the audio encoder step — falls through to a single call when input fits within one chunk
- Fix: `analyze_single_decoder_components` skipped for audio > 120s (runs a full quadratic prefill pass; too large for VRAM on long audio)
- Longer audio test results (f32 KV cache model, ORT extended, single-decoder):
  - **90s, 1024 tokens**: 1 divergence in 75 aligned steps (position 75: timestamp digit "2" vs "4", ~0.02s); both ORT and reference cover full 90s
  - **300s, ORT-only**: 2,955 tokens, 77 segments, covers full 5 minutes cleanly; no parity comparison (reference skipped to stay within VRAM)
  - **600s, compare-single**: fails at decoder prefill (ORT BiasSoftmax allocates float32 attention scores = `28 × 4545² × 4 bytes = 2.35 GB`; exceeds available headroom after 16.6 GB model weights)
  - **600s, ORT-only**: same OOM — prefill attention for 4,500 audio tokens is too large regardless of PyTorch model presence
- VRAM ceiling summary for RTX 3090 (24 GB):
  - Baseline with ORT model loaded: ~19.86 GiB
  - Available for prefill activations: ~4 GiB
  - Max audio for ORT-only: ~300–350s (prefill attention ~590 MB for 300s)
  - Max audio for compare-single (PyTorch + ORT both loaded): ~150s
  - Beyond ~5 minutes on a 3090 requires chunked prefill (sliding window KV cache) — deferred
- Parity conclusion: f32 KV cache model passes all feasible parity tests with only timestamp-digit differences; content is identical across the full tested range

### 2026-04-12 (fifth entry)

- Investigated the cause of the position-14 timestamp-digit divergence between ORT and PyTorch (standard BF16 model, extended ORT optimization)
- Tested `allow_bf16_reduced_precision_reduction = False` hypothesis:
  - Hypothesis: PyTorch's default BF16 matmul accumulation (`allow_bf16_reduced_precision_reduction=True`) produces a 5-way BF16 tie at logit 18.25; ORT uses float32 accumulation (`CUBLAS_COMPUTE_32F`) producing unambiguous higher values (~18.95), causing ORT to pick a different token
  - Added `--no-bf16-reduced-precision` flag to smoke test to force PyTorch reference into float32 accumulation mode
  - Result: **hypothesis refuted** — disabling BF16 reduced precision shifted PyTorch's tie from 18.25 to 18.125, not toward ORT's ~18.95; the divergence moved *earlier* (position 13 instead of 14), confirming this is not the cause
  - Added `f32_lm_head_context` and `--f32-lm-head-reference` flag for runtime float32 lm_head patching (export-style reference only, no re-export)
- Confirmed the f32 KV cache model (`models/vibevoice_asr_single_bf16_f32kv`) is the right production target:
  - With `--ort-opt-level basic`: diverges at position 84 out of 256 generated tokens (83 tokens of perfect ORT/reference parity)
  - With `--ort-opt-level extended`: diverges at position 50 out of 256 generated tokens (49 tokens of perfect parity)
  - In **all cases**, divergences are timestamp second-decimal digits only — transcription content is identical in ORT and PyTorch
  - Example (extended, 256 tokens): ORT produces `"End":4.04`, reference produces `"End":4.02` — 0.02 s difference; content `"Mhm."` is the same
- Conclusion:
  - The position-13/14 BF16 tie is an irreducible numerical instability in the standard BF16 model; no PyTorch-side precision flag closes the gap with ORT
  - The f32 KV cache model sidesteps this by computing attention in float32, producing fewer BF16 ties across the decode horizon
  - **Recommended production config: `models/vibevoice_asr_single_bf16_f32kv` + `ORT_ENABLE_EXTENDED`**
  - Content parity across the full 256-token horizon is acceptable; timestamp differences are sub-0.1s rounding noise

### 2026-04-12 (fourth entry)

- Implemented and validated a float32 KV cache + attention fix to close the 256-token content-word divergence
- Root cause: accumulated BF16 matmul drift in the KV cache over ~50-70 decode steps caused ORT's logit distribution to diverge qualitatively from PyTorch — producing different content words ("And so, I mean" vs "And finally")
- Fix: patched `Qwen2Attention.forward` during export via a context manager (`f32_kv_cache_context`) that:
  - upcasts new K/V states to float32 before `DynamicLayer.update()` (avoids dtype mismatch in `torch.cat`)
  - upcasts Q to float32 for Q\*K^T and V\*attn computation
  - casts attn_output back to BF16 before `o_proj`
  - the resulting ONNX graph has explicit Cast nodes baked in — no runtime patching needed
- VRAM impact: ~2× KV cache size only (+50-300 MB for typical 1K-5K token sequences vs 16.6 GB model weights)
- Added `--f32-kv-cache` flag to `export_vibevoice_asr_to_onnx.py`; exported `models/vibevoice_asr_single_bf16_f32kv/`
- 256-token parity result:
  - content words now match: "And finally, I was just getting more and more..." in both PyTorch and ONNX
  - first divergence still at position 14 (a genuine 3-way 0.0-margin BF16 tie: tokens "2"/"4"/"6" all at logit 18.25) — affects timestamp second decimal only
  - ORT top logit at position 14 is now 18.25 (same magnitude as reference), not 19.125 as before the fix
  - remaining minor differences: timestamp second decimals (e.g. 3.32 vs 3.38) and one filler-word split ("Mhm. Oh." vs "Mhm.")
- Conclusion: the float32 KV cache model is the recommended export for CUDA BF16 inference; content parity is achieved up to irreducible BF16 tiebreakers at timestamp digit level

### 2026-04-12 (third entry)

- Diagnosed and confirmed ORT CUDA attention fusion as root cause of BF16 decoder parity gap
- Test matrix (decoder opt level, divergence position, cause):
  - `ORT_ENABLE_ALL` → diverges at position **13** — ORT's fused-attention CUDA kernel replaces the standard Q\*K^T+mask+softmax+V graph pattern with a single op that skips the float32 softmax upcast used by Qwen2 eager
  - `ORT_ENABLE_EXTENDED` → diverges at position **14** — GeLU/LayerNorm fusions are fine; attention fusion is NOT applied at this level; position 14 is a genuine 0.0-margin BF16 tiebreaker (multiple tokens tied at the same logit in BF16), not a systematic error
  - `ORT_ENABLE_BASIC` → same as extended: diverges at position **14**
- Key evidence: the Qwen2 eager attention implementation does `softmax(..., dtype=torch.float32).to(query_states.dtype)`, which exports as Cast(BF16→F32) → Softmax → Cast(F32→BF16). ORT's `ORT_ENABLE_ALL` pattern-matches through these nodes and replaces them with a fused CUDA kernel that performs the softmax in BF16 — different numerics.
- Conclusion:
  - `ORT_ENABLE_EXTENDED` is the optimal level for the decoder: avoids the attention fusion precision regression and still applies useful fusions (GeLU, LayerNorm)
  - the remaining position-14 divergence is irreducible BF16 precision noise (0.0-margin tiebreakers in the second decimal of a timestamp)
  - audio encoder always keeps `ORT_ENABLE_ALL` to stay within VRAM budget (needs memory-layout transforms)
- Applied fix:
  - changed default `--ort-opt-level` in `smoke_test_vibevoice_onnx.py` from `all` to `extended`
  - updated all function-level defaults to match
  - added explanation of the attention fusion root cause in the argparse help and docstrings

### 2026-04-12 (second entry)

- Reworked `DecoderSingleWrapper` to use a segmented input interface, eliminating in-graph audio-token replacement:
  - old signature: `input_ids, audio_embeddings, packed_past_key_values`
  - new signature: `prefix_input_ids, audio_embeddings, suffix_input_ids, packed_past_key_values`
  - the wrapper now builds `inputs_embeds` directly from the three segments via `torch.cat` rather than scatter-replacing audio tokens
- Updated `smoke_test_vibevoice_onnx.py` to handle both the old and new single-decoder signatures:
  - added `split_audio_placeholder_segments()` to split a flat `input_ids` tensor into prefix/suffix around the audio-token run
  - added `build_single_inputs_embeds()` for the matching PyTorch reference path
  - added `decoder_input_names` introspection so old model directories still work
- Export result:
  - the new `decoder_single.onnx` exports cleanly with `torch.export`
  - inspecting the graph confirms no `ScatterND` nodes in the new export
- Parity result after removing `ScatterND`:
  - CUDA `bfloat16` compare-single still diverges at token position `11`
  - CUDA `float16` compare-single still diverges at token position `14`
  - removing `ScatterND` from the single-decoder export did not close the main parity gap
- Isolation experiment: ORT audio embeddings fed into the PyTorch segmented decoder only diverged at token `15`, not token `11`
- Interpretation:
  - audio embeddings are not the primary cause of the early token-11 split
  - the remaining divergence is in ORT CUDA decoder execution itself
  - the decoder issue is worse in `bfloat16` than in `float16`, even after removing the ScatterND path
- Decision:
  - keep the new segmented single-decoder interface going forward because it is structurally cleaner
  - the remaining blocker is ORT CUDA decoder numerics, not audio replacement logic or audio embedding quality
- Added diagnostic tooling to investigate the ORT CUDA numerics root cause:
  - `smoke_test_vibevoice_onnx.py`:
    - `--disable-ort-optimizations` replaced by `--ort-opt-level none|basic|extended|all`
    - `--ort-opt-level basic` = `ORT_ENABLE_BASIC` disables all op fusion (constant folding only)
      → this is the targeted test for whether ORT fused-attention kernels are causing the parity gap
    - `--save-ort-optimized-model PATH` saves ORT's post-optimization graph to disk for inspection
    - `export_style_step_by_step_alignment` section added to `compare-single` output
      → shows per-step logit margin and cross-rank agreement from position 0 to divergence+1
      → reveals whether errors accumulate gradually or one step makes a sudden jump
  - `inspect_vibevoice_onnx.py`:
    - now inspects `decoder_single.onnx` in addition to the split decoder graphs
    - `op_type_counts` added per model (sorted by frequency)
    - `fusion_ops_detected` flags ORT-specific fused ops (Attention, GroupQueryAttention, etc.)
    - `fusion_ops_summary` and `any_fusion_ops_detected` added at the top level
    - use this on a `--save-ort-optimized-model` output to confirm whether ORT fused the attention
- First diagnostic run results (`data/vibevoice-asr_parity/`):
  - Run A (`--ort-opt-level all --save-ort-optimized-model`): crashed during audio encoder session init
    - root cause: `optimized_model_filepath` was applied to ALL sessions including audio encoder
    - ORT's `NchwcTransformer` expanded the audio graph past protobuf's 2GB limit
    - the saved `/tmp/decoder_single_ort_all.onnx` was 0 bytes (failed write)
  - Run B (`--ort-opt-level basic`): OOMed in the audio encoder at ~1.1 GB allocation
    - root cause: `ORT_ENABLE_BASIC` disables memory-layout transforms that keep audio encoder within VRAM budget
    - the audio encoder needs full optimization (`ORT_ENABLE_ALL`) to fit on the 3090
  - Inspect result: the on-disk `decoder_single.onnx` still has the **old interface** (`input_ids`, `attention_mask`) and 2 `ScatterElements` nodes — the segmented re-export with `prefix_input_ids`/`suffix_input_ids` has not been run yet
- Fixes applied based on these findings:
  - `smoke_test_vibevoice_onnx.py`: split session options — audio encoder always uses `ORT_ENABLE_ALL`, the diagnostic `--ort-opt-level` and `--save-ort-optimized-model` now only apply to decoder sessions
  - this prevents both the protobuf overflow and the audio encoder OOM
- Required next step: re-export `decoder_single.onnx` with the new segmented interface (the code changed in the last commit but the model on disk is still the old version):
  ```bash
  python3 scripts/vibevoice_export/export_vibevoice_asr_to_onnx.py \
    --output-dir models/vibevoice_asr_single_bf16 \
    --dtype bfloat16 \
    --device cuda \
    --decoder-graph-mode single \
    --decoder-exporter torch-export \
    --skip-audio-encoder \
    --acoustic-tokenizer-chunk-size 361600 \
    --deterministic-audio \
    --dummy-prompt "Transcribe the meeting audio with speaker labels when available." \
    --overwrite
  ```
- After re-export, verify the new interface with the inspector, then re-run the parity tests:
  ```bash
  # Confirm prefix_input_ids interface and no ScatterElements/ScatterND
  python3 scripts/vibevoice_export/inspect_vibevoice_onnx.py --model-dir models/vibevoice_asr_single_bf16

  # Baseline: full optimization, save ORT-compiled decoder graph for inspection
  python3 scripts/vibevoice_export/smoke_test_vibevoice_onnx.py \
    --model-dir models/vibevoice_asr_single_bf16 \
    --audio data/test_audio/en-US/en-US_sample_01.wav \
    --runtime cuda --device cuda --dtype bfloat16 \
    --mode compare-single --max-new-tokens 32 \
    --save-ort-optimized-model /tmp/decoder_single_ort_all.onnx \
    --json-out data/vibevoice-asr_parity/parity_ort_all_v2.json

  # Inspect what ORT fused (look for Attention/GroupQueryAttention)
  python3 scripts/vibevoice_export/inspect_vibevoice_onnx.py \
    --model-dir /tmp  # adjust to wherever decoder_single_ort_all.onnx landed

  # Key diagnostic: disable all fusion — if divergence moves later, ORT fusion is the cause
  python3 scripts/vibevoice_export/smoke_test_vibevoice_onnx.py \
    --model-dir models/vibevoice_asr_single_bf16 \
    --audio data/test_audio/en-US/en-US_sample_01.wav \
    --runtime cuda --device cuda --dtype bfloat16 \
    --mode compare-single --max-new-tokens 32 \
    --ort-opt-level basic \
    --json-out data/vibevoice-asr_parity/parity_ort_basic_v2.json
  ```

### 2026-04-12 (first entry)

- Updated the audio export investigation around true `BF16`
- Confirmed that the original Hugging Face safetensors store model tensors as `bfloat16`
- Verified that `torch.export` can emit a true `BF16` `audio_encoder.onnx`
- Important standards finding:
  - the exported graph used `Conv` under `opset 18`
  - ONNX `Conv` only admits `tensor(bfloat16)` at `Conv-22`
  - this explains the earlier ORT invalid-graph error for the `BF16` audio graph under `opset 18`
- Tested `torch.export` audio export at `opset 22`
- Result:
  - the `BF16` audio graph became type-valid at the ONNX level
  - but local ONNX Runtime `1.24.4` still could not run it cleanly
  - CPU failed with `NOT_IMPLEMENTED : Conv(22)`
  - CUDA failed during session initialization in `transformer_memcpy` around provider assignment for `Pad`
- Implemented a narrower working fallback for the audio path:
  - keep external `audio_encoder.onnx` inputs and outputs in `bfloat16`
  - promote only the Conv-heavy tokenizer towers to `float32` during legacy export
  - cast tokenizer latents back before the multimodal projector
- Result:
  - ORT now accepts `audio_encoder.onnx` with `tensor(bfloat16)` I/O at `opset 18`
  - this avoids the earlier whole-audio `float16` or `float32` fallback
- Re-exported `models/vibevoice_asr_single_bf16` with this narrower audio conversion
- Parity result:
  - audio export fidelity improved modestly
  - but the main `compare-single` divergence did not move
  - the first mismatch is still at token position `11`, where PyTorch chooses `"1"` and ORT CUDA `BF16` chooses `"3"`
- Interpretation:
  - the audio acceptance problem is now substantially better boxed in
  - the remaining parity issue still points at the CUDA `BF16` single-decoder path, not the audio graph alone

### 2026-04-11

- Reworked `audio_encoder.onnx` to be truly dynamic over waveform length by removing the legacy traced `Split` layout
- Added single-decoder parity tooling:
  - compact `token_divergence`
  - `divergence_step_analysis`
  - optional `--include-token-debug`
- Added ORT Python `BF16` feed/output support through `OrtValue`
- Confirmed current desktop CUDA `BF16` parity gap is real and early:
  - full `30s` compare-single now runs on dynamic audio
  - first divergence occurs at token position `11`
  - the first mismatch is a near-tie in timestamp digits
- Added pure-PyTorch export-style single-decoder diagnostics
- Result:
  - packed-cache round-trip inside PyTorch stays close
  - ONNX CUDA still diverges from both the standard PyTorch path and the export-style PyTorch path
- Ran a temporary CPU `float32` export and compare-single check
- Result:
  - ONNX matched PyTorch exactly on CPU `float32`
  - this ruled out a fundamental single-decoder export logic bug
- Ran a temporary CUDA `float16` single-decoder comparison
- Result:
  - `float16` on CUDA was closer to parity than `bfloat16`
  - the first divergence moved later, from token `11` to token `14`
- Interpretation:
  - remaining mismatch is most likely CUDA low-precision decoder behavior
  - this does not prove `float16` is more faithful to the original checkpoint; it only shows current ONNX-vs-PyTorch parity is better under that runtime dtype

### 2026-04-10

- Created the `feature/vibevoice-asr-export` branch
- Started this progress log
- Confirmed the model split is naturally `audio_encoder + decoder_prefill + decoder_step`
- Confirmed the current laptop cannot validate CUDA/TensorRT locally
- Locked the first pass to FP16/BF16 feasibility work with a `90-second` CPU smoke-test ceiling
- Added the initial `scripts/vibevoice_export/` scaffolding:
  - `export_vibevoice_asr_to_onnx.py`
  - `inspect_vibevoice_onnx.py`
  - `smoke_test_vibevoice_onnx.py`
  - `benchmark_vibevoice_3090.py`
  - `requirements.txt`
  - `README.md`
- Added `docs/VIBEVOICE_ASR_3090_ANALYSIS.md` for the stable architecture and VRAM summary
- Moved the investigation scripts out of `public/` and into root `scripts/` to match the usual private-first workflow
- First live export attempt on CPU failed inside the acoustic tokenizer encoder with:
  - `IndexError: too many indices for tensor of dimension 2`
- Root cause:
  - the export wrapper passed chunk tensors as `[batch, samples]`
  - the tokenizer encoder stem is a causal `Conv1d` stack expecting `[batch, channels, samples]`
- Fix applied:
  - add a singleton channel dimension before calling `acoustic_tokenizer_encoder` and `semantic_tokenizer_encoder`
- Second live export attempt exhausted system RAM during export with the full 8.3B model resident
- Memory-reduction changes applied:
  - export audio and decoder graphs in separate model-load phases
  - drop `language_model` before audio export
  - drop the audio towers before decoder export
  - stop computing dummy decoder audio embeddings through the real audio stack; synthesize shape-only zeros instead
  - disable ONNX constant folding during export to avoid extra export-time memory pressure
- Third live export attempt failed in the Qwen2 decoder during ONNX tracing:
  - mask creation went through the SDPA path and crashed in `sdpa_mask` with `IndexError: tuple index out of range`
- Mitigation applied:
  - force the decoder config onto the `eager` attention backend during export so tracing avoids the SDPA mask helper
- Fourth decoder iteration:
  - eager attention still routed through the shared mask factory and failed in the same traced-shape path
- New mitigation applied:
  - bypass internal mask creation during export
  - build the full 4D causal mask in the wrapper and pass `attention_mask={"full_attention": mask}` directly into Qwen2
- Fifth decoder iteration:
  - `decoder_prefill.onnx` exported successfully
  - `decoder_step.onnx` then failed because Qwen2 now expects a cache object, not a legacy tuple of KV tensors
- Fix applied:
  - wrap the ONNX step inputs into a real `DynamicCache(ddp_cache_data=..., config=...)` before calling the language model
- Audio export then OOMed on the laptop
- Root cause:
  - the exporter still defaulted to the model's production chunk size of `1,440,000` samples
  - that is 60 seconds at 24 kHz, so `--dummy-audio-seconds 5` did not actually keep the dummy audio small
- Fix applied:
  - when `--acoustic-tokenizer-chunk-size` is not provided, derive a smaller export chunk size from the requested dummy duration and the processor pad multiple instead of always using the 60-second production default
- Next audio export attempt got much farther, peaking around `40 GB` RAM before failing on a wrapper shape mismatch
- Root cause:
  - the wrapper tried to compact audio tokens using a derived token-count mask
  - on this short chunked export input, the actual projected sequence length was `37` while the derived mask expected `38`
- Fix applied:
  - for the current batch-size-1 export target, return the actual projected sequence directly with `combined_features.squeeze(0)` instead of masking by a theoretical token count
- Full three-graph export completed successfully after the audio wrapper fix
- First CPU smoke-test attempt failed in ONNX Runtime while loading `decoder_prefill.onnx`
- Root cause:
  - the decoder graphs were exported as `float16`
  - ONNX Runtime CPU lacks kernel coverage for at least one `Where` node in this decoder graph at that dtype
- Mitigation applied:
  - smoke-test tooling now fails early with a clear message when a CPU run is attempted against float16/bfloat16 decoder graphs
- A later smoke-test run still OOMed on the laptop even after freeing the PyTorch reference model first
- Root cause:
  - the remaining memory growth was on the ONNX side, especially when the full decode loop kept multiple ORT sessions and cache tensors alive
- Mitigation applied:
  - split smoke testing into staged modes:
    - `reference-only`
    - `ort-prefill-only`
    - `ort-full`
    - `compare`
  - in the full ORT path, load `decoder_step.onnx` only after prefill completes to reduce overlapping ORT memory usage
- Staged `ort-prefill-only` testing exposed a deeper export issue in `audio_encoder.onnx`
- Root cause:
  - the legacy exporter traced `torch.split(..., chunk_size)` into a fixed split layout
  - so the current audio graph is not truly dynamic over arbitrary waveform lengths
- Immediate runtime mitigation:
  - feed `audio_encoder.onnx` directly from the normalized waveform tensor instead of the processor-produced `input_values`
- Conclusion:
  - the current audio export can still be probed at its exact export-time shape
  - but the audio export itself needs redesign if we want a genuinely dynamic ONNX contract
- Follow-up fix applied to the exporter:
  - round the audio export dummy input length up to a clean multiple of the chosen chunk size
  - record the exact `audio_export_input_samples` in `export-report.json`
- Reason:
  - the legacy audio export is still effectively fixed-shape because traced `Split` nodes bake in the export-time sample layout
  - smoke testing needs the exact export-time input length, not a reconstructed approximation
- Re-exported the audio graph with the fixed-length recording in place
- Successful staged ONNX Runtime CPU validation:
  - command:
    - `python3 scripts/vibevoice_export/smoke_test_vibevoice_onnx.py --model-dir ./models/vibevoice_asr_first_test --audio ./data/test_audio/en-US/en-US_sample_01.wav --runtime cpu --device cpu --dtype float32 --mode ort-prefill-only --max-audio-seconds 30 --json-out data/vibevoice_asr_prefill_only.json`
  - result:
    - `audio_encoder.onnx` loaded and ran on `CPUExecutionProvider`
    - `decoder_prefill.onnx` loaded and ran on `CPUExecutionProvider`
    - `audio_embeddings_shape = [40, 3584]`
    - `prefill_logits_shape = [1, 115, 152064]`
    - `prefill_num_layers = 28`
    - `prefill_next_token = 151644`
    - `legacy_fixed_samples = 128000`
- Interpretation:
  - the current split export is internally compatible through the audio encoder and decoder prefill boundary
  - the decoder side is no longer the immediate blocker for basic ONNX feasibility
  - the current `audio_encoder.onnx` should still be treated as a legacy fixed-shape graph, not a true dynamic-waveform export
- Remaining known gap:
  - `decoder_step.onnx` has not yet been validated in a successful low-memory ORT decode loop on the laptop
  - truly dynamic audio export still needs a redesign, likely away from the currently traced raw-waveform split pattern

### 2026-04-11

- Added BF16-oriented export/runtime fixes in the VibeVoice ONNX tooling:
  - `audio_encoder.onnx` is now exported as `float16` when `--dtype bfloat16` is requested
  - decoder graphs remain in `bfloat16`
  - export metadata now records `audio_encoder_dtype`, `decoder_dtype`, and the mixed-precision note explicitly
- Root cause for that mixed package:
  - ONNX Runtime rejects the current legacy-exported audio `Conv` graph when the audio stage is emitted as `bfloat16`
- Added ONNX Runtime BF16 helper logic to the smoke-test and benchmark scripts:
  - use `OrtValue` feeds for BF16 decoder inputs
  - reconstruct BF16 outputs when ORT Python cannot materialize them through the normal NumPy path
- Investigated the apparent `compare-single` parity gap more closely and found a larger harness issue first:
  - the current `audio_encoder.onnx` still behaves as a fixed-length graph for `361600` samples (`15.067s`)
  - even though the exported input metadata advertises a dynamic `num_samples` axis, the graph still contains a traced `Split` that is baked for the export-time sample count
- Practical consequence:
  - ONNX comparisons against the current single-decoder package must be treated as `15.067s` comparisons, not full `30s` waveform comparisons
- Smoke-test fix applied:
  - `compare` and `compare-single` now align the PyTorch reference waveform to the ONNX audio graph's effective fixed input length when needed
  - the smoke-test payload reports both:
    - `audio_seconds_tested`
    - `reference_audio_seconds_tested`
  - it also reports `legacy_fixed_samples`
- Result after aligning the reference:
  - PyTorch and ONNX now match on the current exported single-decoder package for the tested clip at the fixed `15.067s` window
  - example matched text prefix:
    - `assistant\n[{"Start":0,"End":15.07`
- This is an important interpretation change:
  - the previous larger mismatch was not a pure decoder-precision issue
  - much of it came from comparing ONNX on `15.067s` of audio against PyTorch on `30s`
- Output cleanup:
  - `smoke_test_vibevoice_onnx.py` now hides verbose token-level debug data by default
  - new flag:
    - `--include-token-debug`
  - this restores top-5 candidates and token traces only when deeper parity debugging is needed
- Current understanding:
  - the single-decoder parity harness is now materially more trustworthy
  - the next true export problem is making `audio_encoder.onnx` genuinely dynamic over waveform length instead of only looking dynamic in metadata
- A later `ort-full` attempt still OOMed on the laptop even with `--max-new-tokens 8`
- Interpretation:
  - the current split full-decode path is still too memory-heavy on CPU
  - the most likely cause is duplicated decoder weight residency across `decoder_prefill.onnx` and `decoder_step.onnx` sessions, plus cache growth
- Decision:
  - stop pushing the legacy full-decode smoke test on the laptop for now
  - move the decoder export onto the newer `torch.export`-based ONNX exporter before considering a single-decoder rework
- Implementation change:
  - added `--decoder-exporter auto|legacy|torch-export`
  - `auto` now prefers the newer `torch.export` backend for `decoder_prefill.onnx` and `decoder_step.onnx`
  - `audio_encoder.onnx` intentionally stays on the legacy exporter for now because that path is already fragile and still needs its own redesign
- Follow-up correction after the first `torch.export` attempt OOMed:
  - stop emitting exporter reports and dumped exported-program artifacts by default
  - add `--torch-export-debug-artifacts` for cases where we explicitly want those diagnostics
  - honor `--skip-prefill` and `--skip-step` independently for decoder export so we can test the new path one graph at a time
- `torch.export` decoder prefill now succeeds cleanly at `--opset 18`
- First `torch.export` decoder step attempt failed for a different reason:
  - not memory
  - `torch.export` dynamic-shapes validation rejected the current vararg KV-cache interface with:
    - `inputs has 3 elements, but dynamic_shapes has 58 elements`
- Mitigation applied:
  - disable `torch.export` dynamic-shape annotations for `decoder_step.onnx` for now
  - keep step export fixed at the export-time cache length until the cache interface is redesigned away from varargs
- Successful `torch.export` decoder results:
  - `decoder_prefill.onnx` exports cleanly at `--opset 18`
  - inspection confirms:
    - `input_ids [1, prompt_len]`
    - `attention_mask [1, prompt_len]`
    - `audio_embeddings [num_audio_tokens, 3584]`
    - KV outputs keyed on `prompt_len`
  - `decoder_step.onnx` also exports cleanly at `--opset 18` after removing `dynamic_axes` from the `torch.export` path
- Current decoder state on the new exporter:
  - `decoder_prefill.onnx` is genuinely dynamic over prompt and audio-token lengths
  - `decoder_step.onnx` is currently fixed-shape at the export-time cache contract
  - inspected step shapes:
    - `attention_mask [1, 113]`
    - `past_key_*` / `past_value_*` as `[1, 4, 112, 128]`
- Practical conclusion:
  - `torch.export` is now a viable decoder export path
  - the remaining blocker is not decoder export feasibility
  - the remaining blocker is choosing whether to:
    - keep a mixed package of legacy audio + `torch.export` decoder, or
    - rework the decoder into a single graph/interface before deeper runtime work
- Built a reproducible mixed-package assembler:
  - `scripts/vibevoice_export/assemble_vibevoice_mixed_package.py`
  - purpose:
    - copy legacy `audio_encoder.onnx`
    - copy `torch.export` decoder graphs
    - bring over the required metadata and external-data files
    - write a merged `export-report.json` that records the mixed sources
- Alignment work for the mixed package:
  - re-exported `torch.export` decoder graphs with `--acoustic-tokenizer-chunk-size 32000`
  - result:
    - `audio_export_input_samples = 128000`
    - `prompt_len = 114`
    - `num_audio_tokens = 40`
  - this matches the fixed-sample legacy audio package contract much better than the earlier `121600`-sample decoder exports
- Mixed package assembled successfully in:
  - `models/vibevoice_asr_mixed_test`
- Mixed-package validation results:
  - `ort-prefill-only` succeeded against the mixed package
  - with the export-time prompt text, the mixed prefill path reported:
    - `audio_embeddings_shape = [40, 3584]`
    - `prefill_logits_shape = [1, 114, 152064]`
    - `legacy_fixed_samples = 128000`
- Interpretation:
  - legacy audio export and `torch.export` decoder prefill are compatible in one package
  - the mixed package is therefore a valid transitional artifact
- Final stop condition before the next redesign:
  - even the mixed package still runs into the same structural problem for `ort-full`
  - prefill and step remain two heavyweight decoder sessions
  - that keeps the laptop full-decode path in the same likely-OOM regime
- Decision:
  - stop spending effort on split full-decode testing on the laptop
  - move to a single-decoder rework next
- Export control update for the next 3090 round:
  - added `--decoder-graph-mode split|single|both`
  - `--decoder-graph-mode single` now exports only `decoder_single.onnx`
  - `--export-single-decoder` is retained as a compatibility alias, but now maps to the single-only mode instead of adding a third decoder graph on top of the split pair
- First 3090 single-decoder export attempt with `torch.export` failed in the unified decoder path:
  - `torch.export` treated `seq_len`, `cache_len`, and `total_seq_len` as independent symbolic dimensions
  - Qwen2 attention requires the invariant `total_seq_len = cache_len + seq_len`
  - export failed during fake-tensor shape checking when the attention weights and attention mask no longer had provably equal KV lengths
- Mitigation applied:
  - keep `torch.export` as the preferred path for `decoder_prefill.onnx` and `decoder_step.onnx`
  - temporarily force `decoder_single.onnx` onto the legacy exporter until we redesign the single-decoder contract or find a `torch.export`-compatible way to encode the length relation
- First unified CUDA smoke-test on the legacy-export single decoder:
  - runtime succeeded end-to-end
  - prefill token matched the PyTorch reference
  - first cached decode token also matched the PyTorch reference
  - divergence started on the following token, where ONNX collapsed into repeated newline generation
- Most likely cause:
  - the legacy single-decoder trace forced `packed_past_key_values.shape[4]` through `int(...)`
  - that likely froze `past_length` during tracing even though the ONNX graph still advertises a dynamic cache axis
- Follow-up fix applied:
  - keep `past_length = packed_past_key_values.shape[4]` symbolic in `DecoderSingleWrapper`
  - hardcode batch size `1` in the exported full-attention-mask builder to avoid another unnecessary Python shape cast
- Deeper single-decoder cache rework:
  - stop constructing the unified decoder cache via `DynamicCache(ddp_cache_data=...)`
  - instead, instantiate a cache with the model config and populate each layer's key/value tensors directly from `packed_past_key_values`
  - goal:
    - avoid tracing through `DynamicCache`'s internal empty-cache initialization and append path
    - keep the single-decoder export closer to "explicit packed KV tensors in, explicit packed KV tensors out"
- Follow-up positional fix for the unified decoder:
  - pass explicit `position_ids` derived from `past_length` into Qwen2 during single-decoder export
  - goal:
    - avoid relying on cache-object `get_seq_length()` for rotary positions in the traced single-decoder path
    - test whether the post-step-1 divergence is really a cache-position bookkeeping issue
- Latest 3090 single-decoder result after the positional fix:
  - the catastrophic newline-collapse is gone
  - the unified ONNX decoder now matches the PyTorch reference through token position `11`
  - first observed divergence moves to token position `12`
- Current compare-single snapshot:
  - PyTorch:
    - `assistant\n[{\"Start\":0,\"End\":10.28`
  - ONNX:
    - `assistant\n[{\"Start\":0,\"End\":15.07`
- Interpretation:
  - the single-decoder runtime path is now structurally correct enough to preserve the JSON-style output pattern
  - the remaining issue is later-step fidelity drift, not immediate cache failure
  - this looks more like a legacy-export numerical/graph-fidelity issue than a broken decoder loop
