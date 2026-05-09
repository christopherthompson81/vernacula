# Granite Speech 4.1 → BF16 export & inference investigation

Tracking issue: [#32](https://github.com/christopherthompson81/vernacula/issues/32).

Goal: ship a `granite-speech-4-1-2b-onnx-bf16` HF bundle for hardware
that has native BF16 (Ampere+ NVIDIA tensor cores, AMD Zen 4+
`avx512_bf16`, Intel SPR+ `amx_bf16`). Auto-detect at runtime; FP32
remains the default for hardware without BF16.

The FP32 baseline shipped in PR #31: 10-min en-US clip RTF 0.043 on
RTX 3090. Targets — RTF 0.025–0.030 if BF16 tensor cores hit on GPU,
~0.035 otherwise (faster on CPU only on machines with avx512_bf16).

Each entry is one run or experiment, stamped with local date/time.

## Run 1 — 2026-05-08 — Plumbing audit

**Question:** What hard-coded dtype assumptions block `--dtype bfloat16`
in `export_granite_speech_to_onnx.py`?

**Setup:** Reading the script before running anything.

**Findings:**

- Model loads with the user-selected dtype via `dtype=torch.bfloat16`
  in `load_model_and_processor` (line 189). Encoder/projector/decoder
  weights are BF16.
- **`processor` is loaded without dtype** — its mel/audio_processor
  stays fp32. `make_dummy_processor_inputs` returns `input_features`
  in fp32.
- Hard-coded `torch.float32` survives in two places that participate in
  trace:
  - Lines 1240/1241: `dummy_full`, `dummy_half` — pre-mel waveform
    samples fed to `mel_wrapper`. Mel is DSP-only with no learnable
    params; fp32 in/fp32 out is fine here regardless of model dtype.
  - Lines 1309/1313: `u_past_keys`, `u_past_values` — past-KV dummies
    fed to the unified decoder wrapper. The decoder's projection
    matmuls expect input dtype = weight dtype = BF16 when model is
    BF16, so these MUST follow `args.dtype`.
- `np.float32` audio buffers in `make_dummy_processor_inputs` (lines
  218/219): same story as mel dummies — these go through the
  fp32 audio_processor and never touch BF16 weights. Keep as fp32.

Per-graph dtype boundaries we expect to need:

| Graph | Input | Internal | Output |
|---|---|---|---|
| `mel.onnx` | fp32 audio | fp32 (DSP) | fp32 features |
| `encoder.onnx` | fp32 features | BF16 (with `Cast` at input) | fp32 (with `Cast` at output) — keeps C# side fp32 |
| `projector.onnx` | fp32 enc hidden | BF16 (with `Cast` at input) | fp32 (with `Cast` at output) |
| `decoder.onnx` | int64 ids/masks/cache_pos, **fp32** audio_embeds, **BF16** past_kv | BF16 | **fp32** logits (with `Cast`), **BF16** present_kv |

The "BF16 past_kv" choice on the decoder is load-bearing: keeping KV
BF16 across the chained `Run` steps preserves half the per-step memory
bandwidth and concat cost on KV (Run 7 of the perf doc identified
concat as the #1 ORT op cost). The only consequence on the C# side is
that the *empty-prefill* past-KV OrtValues at the start of decode need
to be created with BF16 dtype rather than the current
`OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), ...)`. ORT
C# exposes `Microsoft.ML.OnnxRuntime.BFloat16` for that. Roughly 5
lines of C# change.

**Implication for the next run:** start with the minimal plumbing fix
(past-KV dummies follow `args.dtype`), attempt the trace, then add
`Cast` nodes inside the encoder/projector/decoder wrappers as failures
surface.

## Run 2 — 2026-05-08 — Conv/Where BF16 gaps force mixed precision

**Question:** Will full BF16 export (encoder + projector + decoder)
load and run on ORT?

**Setup:** With past-KV dummies routed through `args.dtype` (Run 1) and
fp32 round-trip Casts removed from the encoder→projector and
projector→decoder boundaries (per the user's "no double-cast" pushback —
BF16↔fp32 is bit-preserving for BF16 values, so any internal round-trip
is pure bandwidth waste, not a correctness issue), full BF16 export
**traces successfully**. Bundle drops from 8.7 GB to 4.4 GB.

ORT load-time errors block the runtime path:

1. **Encoder Conv at opset 18:** `Type 'tensor(bfloat16)' of input
   parameter (permute_1) of operator (Conv) ... is invalid.` ONNX
   added BF16 to Conv's type constraint at opset 22.
2. **Encoder Conv at opset 22:** `Type parameter (T) of Optype (Conv)
   bound to different types (tensor(bfloat16) and tensor(float))`. The
   Conformer's depthwise convs end up with mixed-dtype inputs/weights
   in the trace — not a single-graph type error but a deeper
   ONNX/PyTorch interaction.
3. **Decoder unified `Where` on CPU EP:** `Could not find an
   implementation for Where(16) node`. ORT CPU EP does not implement
   Where for BF16. The audio merge (`torch.where`) needs Where to
   support BF16 to run on CPU at all.

**Decision:** Mixed precision —
- `model.encoder.to(torch.float32)` and
  `model.projector.to(torch.float32)` after load (encoder accounts for
  ~25% of wall time vs decoder's 60%+; projector is ~0.2%).
- LM decoder stays in `args.dtype`. Audio_embeds boundary cast inside
  `DecoderUnifiedWrapper`: fp32 in → BF16 internal. Logits boundary
  cast: BF16 internal → fp32 out (CPU argmax stays cheap).
- Past-KV stays in `args.dtype` across the chained `Run` loop —
  GPU-resident, never read on the host. C# detects the dtype from the
  decoder's `InputMetadata["past_key_0"].ElementDataType` at construction
  and creates the empty-prefill OrtValues via a small
  `CreateEmptyPastKv(shape)` helper that switches on `Float / BFloat16
  / Float16`.

The CPU `Where(BF16)` gap means **the BF16 bundle is GPU-only in
practice**. Hardware auto-detection should require CUDA EP (Ampere+
preferred), not just `avx512_bf16`. CPU users — including the laptop
target with native CPU BF16 — get the FP32 bundle.

### Verified directly: C# CPU EP fails identically (2026-05-08)

Built `Vernacula.CLI` with `-p:EP=Cpu` and ran the BF16 bundle on the
6.4 s VCTK clip. Same error at `InferenceSession` construction:

```
Microsoft.ML.OnnxRuntime.OnnxRuntimeException:
  [ErrorCode:NotImplemented] Could not find an implementation for
  Where(16) node with name 'node_where_1'
```

This is an ORT op-kernel **registry** miss, not a runtime hardware
gate. The CPU EP's BF16 op coverage is decided at ORT build time;
`avx512_bf16` controls SIMD dispatch *within* an existing kernel,
which means it can speed up a BF16 kernel that ships but cannot
materialise one that doesn't. So:

- `avx512_bf16` in `/proc/cpuinfo` is **not sufficient** for the BF16
  bundle to load on CPU EP.
- Auto-detect must gate on CUDA EP availability with Ampere+ tensor
  cores — period — until ORT closes the CPU op-kernel gaps for
  `Where` and the rest of the LM ops the decoder graph uses.
- The 7840U laptop will use the FP32 bundle today even though its CPU
  has native BF16 in hardware. That capability is wasted on this
  model with current ORT.

## Run 3 — 2026-05-08 — Bundle size and per-stage profile

**Setup:** Mixed-precision export at `--dtype bfloat16 --opset 18`.

**Bundle layout:**

```
mel.onnx              124 KB    (unchanged DSP)
encoder.onnx          2.2 MB    (graph)
encoder.onnx.data     1.7 GB    (fp32 weights, unchanged)
projector.onnx        137 MB    (fp32 weights, inline)
decoder.onnx          7.1 MB    (graph)
decoder.onnx.data     3.5 GB    (BF16 weights — half of fp32's 6.9 GB)
                      ────
total                 5.3 GB    (vs 8.7 GB FP32 → 39% smaller)
```

**C# pipeline change (one helper):**

```csharp
private OrtValue CreateEmptyPastKv(long[] shape) => _pastKvDtype switch
{
    TensorElementType.Float    => OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(),    shape),
    TensorElementType.BFloat16 => OrtValue.CreateTensorValueFromMemory(Array.Empty<BFloat16>(), shape),
    TensorElementType.Float16  => OrtValue.CreateTensorValueFromMemory(Array.Empty<Float16>(),  shape),
    _ => throw new InvalidOperationException(...),
};
```

`_pastKvDtype` is read once at session load from
`_decoder.InputMetadata["past_key_0"].ElementDataType`. The chained
Run-with-OrtValue loop is otherwise unchanged — KV OrtValues come back
from `_decoder.Run` in whatever dtype the graph emits and feed back as
inputs to the next Run without C# ever seeing the data.

## Run 4 — 2026-05-08 — End-to-end RTF on 10 min en-US clip

**Test:** 600 s en-US clip, VAD-segmented to 159 segments, batched
greedy decode through `--asr granite` on RTX 3090.

| Bundle | ASR wall | RTF | Step-loop | Encoder | KV bytes per token (B=16) |
|---|---:|---:|---:|---:|---:|
| FP32 (baseline) | 23.9 s | 0.043 | 13.0 s (60%) | 5.4 s (25%) | 167 MB |
| **BF16 mixed**  | **17.4 s** | **0.032** | **7.8 s** (50%) | 5.2 s (33%) | **84 MB** |

**Step-loop dropped 41%** (13.0 s → 7.8 s) — the dominant LM cost is
where BF16 tensor cores hit hardest. Encoder unchanged at ~5.2 s
(intentional — kept fp32). Combined ASR wall is **27% faster** despite
encoder being unchanged.

Word count: 1514 (FP32) → 1521 (BF16) — within 0.5%. Spot-check diff:
small word-level variations on ~15 lines (e.g. `"Oh, yeah."` vs
`"Uh, uh."`, `"Fred and Nena's"` vs `"Fred and Nanette's"`) typical of
greedy argmax at lower mantissa precision. No catastrophic
regressions, no runaway-loop reappearance, no semantic divergence on
spot-checked passages.

**VRAM-for-batches lever (response to user's concern):** decoder KV
halves at BF16 (167 MB → 84 MB per token at B=16). The cost-model in
`GraniteBatchCostModel.EstimatePeakBytes` still uses
`bytesPerFloat = 4L` — overestimates KV when KV is BF16 — so on a
3090 with 16 hard-cap we're not seeing the unlock yet. On smaller GPUs
the BF16 bundle would actually allow a higher batch cap. Cost model
update to read `_pastKvDtype` and use 2 bytes for BF16 is a follow-up.

**Implication:** ship the BF16 bundle as a sibling HF repo
(`granite-speech-4-1-2b-onnx-bf16`) for hardware that has CUDA EP.
Vernacula's downloader picks one based on hardware capability —
gating on "Ampere+ NVIDIA GPU detected" rather than `avx512_bf16`,
because the CPU `Where(BF16)` gap blocks CPU EP entirely.
