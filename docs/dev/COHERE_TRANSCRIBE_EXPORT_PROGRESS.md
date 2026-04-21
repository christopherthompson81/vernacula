# Cohere Transcribe ONNX Export — Progress & Findings

Model: `CohereLabs/cohere-transcribe-03-2026`  
Architecture: encoder-decoder seq2seq ASR (48-layer Conformer encoder, 8-layer decoder, d_model=1280)

---

## Exported files

| File | Description | Status |
|---|---|---|
| `mel.onnx` | DFT-conv1d mel spectrogram; `waveforms [1,T]` → `features [1,128,F]` | ✅ |
| `encoder.onnx` | Conformer encoder; `input_features [B,128,F]` → `encoder_hidden_states [B,T',1280]` | ✅ batch-dynamic |
| `decoder_init.onnx` | KV-cache init; BOS `[B,1]` + enc hidden `[B,T',1280]` → logits `[B,1,V]` + 32 KV tensors | ✅ batch-dynamic |
| `decoder_step.onnx` | KV-cache step; tokens `[B,1]` + self-KV + cross-KV → logits `[B,1,V]` + updated self-KV | ✅ batch-dynamic |
| `vocab.json` | 16 384-entry token string array (indexed by ID) | ✅ |
| `config.json` | Special token IDs, sampling rate, feature extractor params | ✅ |

Scripts: `scripts/cohere_export/`

---

## Encoder batch-size constraint

### Root cause

The Cohere Conformer encoder contains **data-dependent control flow** that is baked as a
constant during TorchScript tracing, locking the exported graph to the batch size used
during export.  Two confirmed branches in `modeling_cohere_asr.py`:

```
line 310: if pos_emb.size(0) == 1 and batch_size > 1:   # positional embedding expansion
line 170: if self._needs_conv_split(x):                  # convolution split guard
line 118: if projected > max_size_32bit:                 # 32-bit projection guard
```

When exported with a B=1 dummy input, `batch_size > 1` traces as `False` and the
positional embedding expansion path is omitted from the graph entirely.  At runtime with
B > 1, the internal attention reshape (`[B, T, 1280] → [B*H, T, head_dim]`) receives the
wrong shape and raises:

```
Input shape:{1,125,1280}, requested shape:{4,-1,8,160}
Non-zero status code returned while running Reshape node.
Name:'/encoder/layers.0/self_attn/Reshape_3'
```

### Investigation history

**Attempt 1 — export with B=16 dummy, `do_constant_folding=True`** (original export)  
Worked only for B=16.  The value `B*H = 16*8 = 128` was folded into the Reshape constant.
Any other batch size failed.

**Attempt 2 — `dynamo=True` with `torch.export.Dim`**  
Failed at export time:

```
ConstraintViolationError: solving the guards generated for batch resulted in a specialized value of 1.
Suggested fixes: batch = 1
```

`torch.export`'s symbolic analysis correctly identified the data-dependent branch and
refused to export with a dynamic batch dimension.  This was initially misread as proof
that the model is architecturally B=1 only.

**Attempt 3 — `dynamo=False`, B=1 dummy, `do_constant_folding=False`, batch in `dynamic_axes`**  
Export succeeded; parity check passed (max diff 0.001).  Runtime test:

```
B=1: encoder output shape = (1, 63, 1280)   ✅
B=4: RuntimeException — requested shape:{4,-1,8,160}  ❌
```

`do_constant_folding=False` does not help because the control flow is baked by the
**tracer** before constant folding runs.  The `if batch_size > 1` branch is absent from
the traced graph regardless of folding settings.

### Confirmed finding

The Cohere encoder **is** capable of batched inference in PyTorch (the HuggingFace
`transcribe()` API accepts `batch_size=16`).  The limitation is specific to
**TorchScript tracing**: the data-dependent branches are evaluated at trace time and
their outcomes are frozen in the graph.  There is no ONNX export path using the current
`EncoderWrapper` that produces a batch-agnostic model.

### Path to fix

The correct fix is to patch `EncoderWrapper` to **unconditionally execute both branches**
or restructure the forward pass to remove the batch-size-dependent conditionals, so
that the TorchScript trace is batch-agnostic.  Concretely:

- Force `pos_emb` expansion to always run (or pre-expand before export)
- Make `_needs_conv_split` always return a fixed value (or inline the non-split path)
- Verify with B=1 and B=4 parity checks after re-export

This work is tracked as an open item below.

### Resolution — patched EncoderWrapper

`encoder.onnx` is exported with the patched `EncoderWrapper` (`dynamo=False`,
`do_constant_folding=False`) and dynamic axes on both batch and time:

```python
dynamic_axes={
    "input_features":        {0: "batch", 2: "time_frames"},
    "encoder_hidden_states": {0: "batch", 1: "enc_seq_len"},
}
```

Two patches are applied before tracing:

1. Every `ConformerLayer.self_attn.forward` is monkey-patched to always call
   `pos_emb.expand(batch_size, -1, -1)` unconditionally, removing the
   `if pos_emb.size(0) == 1 and batch_size > 1` data-dependent branch.
2. `encoder.pre_encode._needs_conv_split = lambda x: False` removes the conv-split branch.

`do_constant_folding=False` is required: with `True`, the tracer materialises
`pos_emb.expand()` (a no-op for B=1 dummy) into all 48 attention layers separately,
inflating the export from ~7 GB to ~28 GB.  Without folding the PE buffer stays as a
shared graph reference.

#### Overwrite / re-export bug fix

`onnx.save_model` with `save_as_external_data=True` appends to any existing `.data`
file rather than truncating it, causing the file to grow by one model's worth of
weights on every re-export.  Fixed in `_consolidate_external_data` by explicitly
deleting the old `.data` file before calling `save_model`.  Also sweeps stale
`Constant_*` scattered-weight files left by previous partial exports.

### Implications for throughput

All four ONNX models (encoder, decoder_init, decoder_step, mel) support a dynamic
batch dimension.  `mel.onnx` was exported with B=1 dummy and no batch dynamic axis;
mel runs once per segment on CPU.  Encoder and decoder are batch-dynamic and run over
a full segment batch together.

`CohereTranscribe` uses `BatchSize = 8`:
- mel: 8 serial B=1 calls (CPU, negligible)
- encoder: one B=8 call → hidden states `[8, T_enc_max, 1280]`
- decoder_init: one B=8 call → logits `[8,1,V]` + 32 KV tensors shaped `[8,...]`
- decoder_step: one B=8 call per position → 8 next tokens decoded simultaneously

Reading decoder weights once and computing 8 tokens per step is 8× more
HBM-efficient than 8 serial B=1 calls.

#### Parallel decoder lanes (attempted, reverted)

An earlier approach created multiple `(decoder_init, decoder_step)` session pairs,
each with its own CUDA stream, and dispatched segments to lanes in parallel via
`Task.Run`.  This made throughput worse (~90 s vs ~68 s) because:

- Each lane loads decoder weights independently into GPU memory (no sharing between
  ORT sessions).
- Two concurrent streams each issue ~4.3 GB of HBM reads per step, saturating the
  memory bus without proportional compute benefit — weight-read traffic doubles while
  throughput stays flat or falls due to cache thrashing.

Batched decoding (single session, B=8) is the correct solution: weights are read once
and amortised over 8 tokens per step.

#### Throughput measurement (RTX 3090, Release build, 10-minute file, 157 VAD segments)

| Configuration | ASR time | RTF |
|---|---|---|
| B=1 serial decode, constant-folded encoder (baseline) | ~68 s | ~0.117 |
| B=8 batched decode, `GetValue(i)` element copy | ~282 s | ~0.473 |
| B=8 parallel decoder lanes (HBM contention) | ~92 s | ~0.154 |
| Dynamic batching, sorted by duration, bulk tensor copy | 47.3 s | **0.0822** |

The final configuration (0.0822 RTF) uses:
- Segments sorted ascending by duration before batching (minimises straggler waste)
- Batch size determined by `cudaMemGetInfo` free VRAM minus 2 GB safety buffer,
  capped at `MaxBatchSize = 32`
- `DenseTensor<float>.Buffer.Span.CopyTo` for bulk tensor extraction (vs element-by-element `GetValue`)
- Single `(decoder_init, decoder_step)` session pair; weights read once per step
  amortised over all B segments in the batch

The decoder bottleneck breakdown at B=1 (profiled):

| Phase   | Time      | Share |
|---------|-----------|-------|
| Mel     | 259 ms    | 0.4 % |
| Encoder | 6,300 ms  |  9 % |
| Decoder | 50,794 ms | 90 % |

With dynamic batching (B≈32 for most batches on RTX 3090 after models load), decoder
steps process multiple segments per kernel launch.  For 157 segments, batching reduces
the effective number of serial decoder calls roughly 20–30×, each doing proportionally
more useful work per HBM read.

---

## Key bugs found and fixed

### IOBinding output order

`OrtIoBinding.GetOutputValues()` returns tensors in **binding order** (the order
`BindOutputToDevice` was called), not model output order.

The `decoder_init` binding loop was interleaved (`sk0, sv0, ck0, cv0, sk1, sv1, …`) but
the step loop read `initOutputs` assuming grouped model order (`[1..8]` = self_key,
`[9..16]` = self_val, `[17..24]` = cross_key, `[25..32]` = cross_val).  This fed
cross-KV tensors where self-KV was expected, causing a MatMul dimension mismatch on the
first `decoder_step` call.

**Fix**: bind init outputs in four grouped passes (all self_key, then all self_val, then
all cross_key, then all cross_val) and step outputs in two grouped passes (new_self_key,
new_self_val) so `GetOutputValues()` indices match the code's assumptions.  Diagnosed
with a minimal C# standalone that printed output shapes under both binding orders.

---

### Encoder batch padding contamination

When segments of different lengths are batched through the encoder, shorter segments are
zero-padded to match the longest segment's mel length.  The Conformer encoder uses
**global self-attention**, so zero-padded positions contaminate ALL encoder output
positions via attention — not just the padded tail.  The decoder's cross-attention then
attends to these garbage encoder states, causing:

- Repetition loops ("tank tank tank…" ×300, CJK characters)
- Trailing hallucinations (correct content followed by fabricated words)
- Wrong language output (`[笑]`, `嗯`) for short non-speech segments

The contamination is proportional to the padding ratio.  Testing showed:

- 1.0× ratio (no padding): clean
- 1.1× ratio: hallucinations
- 1.3×: severe repetitions
- 2×+: catastrophic loops

Post-hoc zeroing of padded encoder output positions does **not** fix this — the valid
positions themselves are corrupted by attending to zero-padded positions during encoder
self-attention.

**Investigation**: Python B=1 per segment always produced clean output; B=2 with
segments of different lengths reproduced the repetitions.  This confirmed the issue is
the batched encoder, not the ONNX export or decoder.  The decoder at B=157 (all segments
at once) completes in 628 ms — it is not the bottleneck.  Per-segment encoder (B=1
serial) took 53 s for 157 segments — the encoder is the bottleneck.

**Fix**: re-exported `encoder.onnx` with a new `input_lengths [B] int64` input.
`EncoderWrapper.forward` passes this to `self.encoder(input_features, length=input_lengths)`.
The Conformer computes a padding mask from the lengths and the patched
`_patched_rel_pos_attn_forward` applies it to the self-attention scores, masking out
zero-padded positions before softmax.

The ONNX trace uses B=2 dummy inputs with one full-length and one half-length (padded)
item.  This forces the masking branch to be traced into the ONNX graph, making it active
at runtime for any batch with mixed-length segments.

**Result**: batched encoder at B=32 with no `MaxEncFrameRatio` constraint produces clean
output identical in quality to per-segment (B=1) inference.

#### Encoder ONNX I/O (updated)

```
encoder.onnx
  input_features [B, 128, F_max]  float32   — zero-padded mel spectrogram batch
  input_lengths  [B]              int64     — actual mel-frame count per item
  → encoder_hidden_states [B, F_max/8, 1280]  float32
```

**Encoder batch size limit**: the Conformer's positional encoding `Expand` node allocates
~47 MB per batch item.  At B=32 this is ~1.5 GB (safe on RTX 3090 with ~12 GB free after
model loading).  At B=157 it OOMs.  `MaxBatchSize = 32` is the current safe limit.

#### Throughput after fix (RTX 3090, 600 s audio / 157 segments)

| Stage | Time |
|---|---|
| VAD | 2.0 s |
| ASR (B=32 encoder + KV-cache decoder) | 13.5 s |
| **Total** | **15.5 s** |
| **RTF** | **0.026 (38× realtime)** |

For comparison, Parakeet 0.6B (CTC) runs the same file in ~9 s.  The gap reflects model
complexity: 48-layer Conformer encoder + 8-layer autoregressive decoder vs a single CTC
stack.

---

## Precision / dtype notes

### Decoder KV cache (float16) — working

`export_cohere_decoder_kv.py --dtype float16` (default):
- Logits cast back to float32 at graph boundary (C# reads `float[]`)
- KV tensors stay float16 and are passed GPU→GPU via IOBinding; C# never inspects bytes
- Halves KV memory footprint relative to float32

### Encoder float16 — broken

Native `--dtype float16` encoder export produces hallucinations (Arabic/CJK) for short
segments.  The 48-layer Conformer loses numerical precision for low-energy inputs in
fp16.  Post-hoc conversion via `onnxconverter_common` also fails with a type mismatch
in the masking path (`Mul` node sees mixed fp16/fp32 types).  **float32 encoder is
required.**

### Encoder bfloat16 — dead end (upstream ONNX limitation)

bfloat16 is not a valid type for the ONNX `Conv` operator in any current opset (opset
17 allows float16/float32/double only).  The Conformer uses `Conv` in both `pre_encode`
(ConvSubsampling) and in the `ConvolutionModule` of each of the 48 layers, so a
bfloat16 encoder graph is rejected by ORT at load time (`INVALID_GRAPH`).  This is a
spec-level gap — neither the CPU EP nor the CUDA EP can execute it.

The native safetensors weights are bfloat16, so when upstream ONNX adds bfloat16 to the
`Conv` type constraints there will be no overflow risk (unlike float16, which overflows
for large-magnitude Conformer weights).

### Export reproducibility

`--device cuda` introduces CUDA floating-point non-associativity across 48 Conformer
layers.  The variability is in the **export trace**, not at ORT runtime.  For short
ambiguous segments (1–3 s, low energy) the multilingual decoder's language-token
decision is near the en/ar/zh boundary; slight encoder representation shifts across
re-exports can push them to the wrong side.  Use `--device cpu` for the encoder export
to get a deterministic graph.  Use `--language en` for English-only audio to force the
language token regardless of encoder representation.

---

## Context-block tokens

The decoder emits structured metadata tokens before any text tokens.  All context tokens
have IDs in the range `0–254`; text tokens start at ID 255 (byte-fallback range).

| ID range | Meaning |
|---|---|
| 5 / 6 | PNC on / off |
| 8 / 9 | ITN on / off |
| 10 / 11 | Timestamps on / off |
| 12 / 13 | Diarize on / off |
| 16–20 | Emotion: undefined / neutral / happy / sad / angry |
| 21 | Unknown language |
| 22–204 | Language tag `<\|xx\|>` (ISO 639-1) |
| 255–510 | Byte-fallback `<0xNN>` (text) |
| 511+ | BPE text tokens |
| 13764 | `decoder_start_token_id` (BOS) |
| 3 | `eos_token_id` |

The context block always follows a fixed structure:
`<|startofcontext|>` → `<|startoftranscript|>` → `<|emo:*|>` → `<|lang|>` → `<|lang|>` (repeated)
→ `<|pnc|>/<|nopnc|>` → `<|noitn|>` → `<|notimestamp|>` → `<|nodiarize|>`

### Language detection findings

**Python parity test** (`scripts/cohere_export/test_language_detection.py`) on 50
segments of `en-US_sample_01.wav` (known English audio):

- 44/50 segments correctly identified as `en`
- 6/50 misidentified: 3× `ar`, 3× `zh`
- All 50 segments emitted a language token (none missing)
- Misidentified segments are all short (< 3 s) where the model is genuinely uncertain

The same misidentifications occur in the Python reference pipeline — this is **model
behaviour**, not a C# parsing bug.

**Log-probability analysis** of the 6 misidentified segments:

| Seg | Duration | Winner | LP(win) | LP(en) | Δ(win−en) |
|-----|----------|--------|---------|--------|-----------|
| 10 | 2.87 s | ar | −0.498 | −1.036 | +0.538 |
| 13 | 0.67 s | zh | −0.068 | −2.938 | +2.870 |
| 24 | 2.10 s | ar | −0.256 | −1.569 | +1.313 |
| 33 | 2.78 s | ar | −0.592 | −0.810 | +0.218 |
| 37 | 0.80 s | zh | −0.386 | −1.575 | +1.189 |
| 48 | 0.79 s | zh | −0.278 | −1.669 | +1.391 |

Δ ≥ 0.218 for all wrong cases; Δ = 0 for all correct cases (en won).  A threshold on Δ
alone is not sufficient to distinguish correct uncertain predictions from incorrect
confident ones (seg 13, zh, LP=−0.068, is more confident than many correct en segments).

**Solution implemented**: `--language <iso-639-1>` flag forces the language token at the
context-block decode step, substituting the forced token ID before the rest of the decode
proceeds.  This conditions the entire transcript on the specified language.  Zero
additional inference cost — the substitution happens at argmax time.

---

## Known issues / open items

- [x] Language detection missing on some segments — confirmed model behaviour, not a C# bug
- [x] Forced-language decode via `--language` flag — implemented in `CohereTranscribe.Recognize`
- [x] Encoder batch-size > 1 via patched `EncoderWrapper` — `pos_emb` expansion and
      `_needs_conv_split` branches patched out; `do_constant_folding=False` prevents PE tensor
      duplication; B=1 and B=4 parity confirmed at ORT runtime
- [x] Batched decoder — `decoder_init.onnx` / `decoder_step.onnx` exported with batch as a
      dynamic axis; `CohereTranscribe` runs `GreedyDecodeBatch`; parallel lane approach abandoned
      (HBM contention, see above)
- [x] Benchmark batched decode RTF on RTX 3090 — 0.0822 (vs 0.117 baseline) before encoder fix
- [x] IOBinding output order bug — decoder_init outputs must be bound in grouped passes, not
      interleaved; feeds wrong KV tensors to decoder_step otherwise
- [x] Encoder batch padding contamination — re-exported encoder with `input_lengths [B] int64`;
      Conformer masks padded positions in self-attention; B=32 clean at 38× realtime
- [ ] Encoder batch size cap — `MaxBatchSize = 32` limited by PE `Expand` node (~47 MB/item);
      investigate pre-slicing the PE buffer to actual sequence length before expansion
- [ ] Growing self-KV allocation — `decoder_step` outputs concatenated KV (`past‖new`) each step,
      causing N growing allocations; pre-allocating fixed `[B, 8, maxLen, 128]` buffers +
      write-pointer would eliminate this but requires a different ONNX graph
- [ ] Reduced-precision encoder — float16 broken (precision loss), bfloat16 blocked by ONNX spec
      (Conv op does not support bf16); revisit when upstream ONNX adds bfloat16 to Conv type
      constraints

---

## Re-running the export

```bash
# Activate the dedicated venv
source .venv-cohere-asr/bin/activate

# (One-time) Login to HuggingFace for the gated model
huggingface-cli login

# Export mel + encoder (use --device cpu for deterministic encoder representations)
python scripts/cohere_export/export_cohere_transcribe_to_onnx.py \
    --output-dir ./models/cohere_transcribe --device cpu --overwrite

# Export KV-cache decoder_init + decoder_step (float16 KV, float32 weights)
python scripts/cohere_export/export_cohere_decoder_kv.py \
    --model-dir ./models/cohere_transcribe --device cuda --overwrite

# Smoke-test KV-cache vs no-cache
python scripts/cohere_export/smoke_test_kv_cache.py \
    --model-dir ./models/cohere_transcribe \
    --audio data/test_audio/en-US/en-US_sample_01.wav \
    --compare-nocache
```
