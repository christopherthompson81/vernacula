# CoreML ONNX playbook

How to make an ONNX model actually fast under ONNX Runtime's **CoreML execution
provider** on Apple Silicon. Derived from taking Sortformer v2.1 from "fails to
compile at all" to a single-partition CoreML graph.

CoreML matters because it is the only path to the **Apple Neural Engine**. The
WebGPU EP (Dawn → Metal) reaches the GPU and nothing else, so on an M-series
machine WebGPU leaves the ANE idle. WebGPU is also the better default until a
model has been through this playbook, because an un-tuned graph loses to it.

## What this bought on Sortformer

Measured on an M5, ORT **1.24.4**, chunk=992 / spkcache=188 / fifo=124.
Inference figures are from an idle machine; load figures are stable.

| stage | partitions | inference | cold load | warm load | `model.mil` |
|---|---|---|---|---|---|
| shipped dynamic model | *fails to compile* (`error -14`) | — | — | — | — |
| `--coreml-static-batch1` | 71 | 166.0 ms | — | — | — |
| `+ const lengths` | 69 | — | — | — | — |
| `+ all-False `Where` removed` | 35 | 97.0 ms | — | — | — |
| `+ `Pad` → `Concat`` | **1** | 51.3 ms | 101.0 s | 20.8 s | 1.49 GB |
| `+ Gemm pre-transpose` | **1** | 52.3 ms | **2.1 s** | **0.2 s** | **0.00 GB** |

Reference on the same machine: **CPU 163.5 ms, WebGPU 94.0 ms.**
Outputs match the original dynamic model to `3.6e-07` (preds) and `0.0` (embeddings).

Net: **3.2× vs CPU, 1.8× vs WebGPU, and load went from 101 s to 2.1 s.**

## The core mental model

The CoreML EP fuses only the ops it fully supports. Everything else stays on
CPU **and splits the graph at that boundary**. Every boundary is a CPU↔CoreML
copy plus a sync. So the metric that matters is not "percent of nodes
supported" but **partition count**.

Sortformer at 94% supported (1794/1914 nodes) was still barely faster than CPU,
because those 120 stragglers sat inside every encoder layer and carved the graph
into 71 pieces. Getting to 1 partition roughly tripled throughput.

> **Aim for one partition.** Two-thirds supported in one partition beats 95%
> supported across fifty.

## Technique 1 — Static shapes, and no data-dependent slicing

**Symptom:** `Failed to create MLModel ... error code: -14`, preceded by a wall
of `E5RT ... has unbounded dimension which is not supported` and sometimes
`Failed to PropagateInputTensorShapes`.

CoreML's MIL runtime rejects unbounded dimensions outright. Every ASR graph here
has a dynamic time axis, so this hits all of them.

Fixing the *declared input* shapes is necessary but **not sufficient**. Two
traps:

1. **The tracer bakes in shape arithmetic.** `torch.onnx.export` with
   `dynamic_axes` emits `Shape`→`Gather`→`Reshape` chains that compute shapes at
   runtime. Freezing inputs afterwards (`onnxruntime.tools.make_dynamic_shape_fixed`)
   removes the "unbounded" errors but leaves those chains, so shape inference
   still cannot resolve and CoreML still fails. **This must be fixed at export
   time, not post-hoc.** Verified: post-hoc freezing left 352/352 matmuls with
   unresolved shapes; adding constant folding only got it to 86/352, and CoreML
   still refused.

2. **Slicing by a tensor *value* poisons everything downstream.** This pattern:

   ```python
   spk_len = spkcache_lengths[0].to(torch.int64)
   spkcache_trimmed = spkcache[:, :spk_len, :]
   ```

   makes every subsequent shape data-dependent even with fixed input shapes.
   In Sortformer the trim was a **no-op at runtime** — `Sortformer.cs` always
   passes `*_lengths` equal to each buffer's own size — so it was destroying
   compilability for nothing.

**Do:** add an export mode that fixes the input shapes *and* concatenates whole
buffers with no length-based trimming. See `--coreml-static-batch1` in
`scripts/nemo_export/export_sortformer_nemo_to_onnx.py`.

**Check first:** are the "dynamic" axes actually constant at runtime? For
Sortformer they were compile-time constants in `Config.cs`
(`ChunkLength × Subsampling = 992`, `SpeakerCacheLength = 188`,
`FifoLength = 124`), so fixing them cost **zero** padding waste. A model whose
axes genuinely vary needs bucketing, and that trade is real.

## Technique 2 — Bake constant lengths in

If lengths are constant at runtime, make them graph constants
(`--coreml-const-lengths`). Mask construction then folds instead of building
`Range`/`Expand` chains from tensor values.

**Worth knowing: on its own this barely helped** (71 → 69 partitions). Its real
value is enabling Technique 3 — it turns the masks into foldable constants.

## Technique 3 — Delete all-False `Where` masks

**Symptom:** `Where` nodes on CPU, one or more per encoder layer.

With constant lengths and a fully packed sequence, the attention padding mask
folds to a constant that is **entirely False** (Sortformer: shape
`(1,1,436,436)`, 436 = 188+124+124 — the tensor is exactly full).
`Where(false, -10000, x)` is just `x`. These are identities whose only effect is
fragmenting the graph.

**Do:** rewire consumers to the data input and delete the node. Verify the mask
really is all-False first — it is a provable property, so assert it rather than
assuming.

**Trap:** replacing `Where` with `Identity` instead of removing it made things
*worse* (69 → 157 partitions). Remove the node; do not substitute one.

Sortformer: 51 removed, 69 → 35 partitions, 166 ms → 97 ms.

## Technique 4 — Rewrite `Pad` as `Concat`

**Symptom:** `Pad` nodes on CPU.

The CoreML EP declines `Pad` but supports `Concat`, and a **constant zero-pad is
exactly a concat against a zero tensor**. Sortformer had two per layer:

| shape | pads | meaning |
|---|---|---|
| `[1,8,436,871] → [1,8,436,872]` | 1 at start of axis 3 | relative-position shift in self-attention |
| `[1,512,436] → [1,512,444]` | 4 each side of axis 2 | depthwise conv, kernel 9 |

**Do:** for constant-mode, zero-value, single-axis pads, emit
`Concat([zeros, x, zeros], axis)`. Requires static shapes (Technique 1) to size
the zero tensors. Match the zero constant's **dtype** to the padded tensor or
`Concat` fails type inference — this bites after an fp16 conversion.

Sortformer: 34 converted, 35 → **1** partition, 97 ms → 51 ms. Biggest
throughput win of the four.

## Technique 5 — Pre-transpose `Gemm` weights (the load-time fix)

**Symptom:** enormous first-load and warm-load times, and a `model.mil` far
larger than the model's actual weights.

CoreML's `linear` op wants the weight as `[N, K]`. An ONNX `Gemm` with
`transB=0` stores it `[K, N]`, so the EP **synthesizes a transposed copy per
node** — and it writes synthesized constants **inline in `model.mil` as
hex-float text** (`0x1.d040b6p-6`, ~16 chars per float) rather than into the
binary weight blob.

On Sortformer that meant **180 Gemm nodes → 179 inline consts → a 1.49 GB
`model.mil`** holding 0.39 GB of actual weights, re-parsed on every single load.
Meanwhile the MatMul-with-constant weights (10.1 M params) went to `weight.bin`
correctly. 75% of all weights were taking the text path.

**Do:** pre-transpose the constant weight and set `transB=1`. Bit-exact — it is
the same operation. Only do it in place when the initializer has a single
consumer.

```
transB=0:  cold 101.0s   warm 20.8s   cache 2.16 GB   model.mil 1.49 GB
transB=1:  cold   2.2s   warm  0.2s   cache 1.06 GB   model.mil 0.00 GB
```

**46× faster cold compile, 100× faster warm load, zero numerical change.**
Inference is unaffected (52.7 → 54.3 ms, within noise).

This is the highest-leverage item in the playbook and it generalizes to **any**
transformer export, because `transB=0` is what `torch.onnx.export` emits by
default.

## Technique 6 — `ModelCacheDirectory`

Set the CoreML provider option `ModelCacheDirectory` to persist the compiled
model across processes. Necessary but not a substitute for Technique 5:

| | cold | warm |
|---|---|---|
| cache only | 97.5 s | 18.6 s |
| cache + pre-transpose | 2.1 s | **0.2 s** |

## Anti-patterns

**Do not use `ORT_ENABLE_EXTENDED` (or `ORT_ENABLE_ALL`) before CoreML.**
Its fusions produce ops the CoreML EP rejects and it makes partitioning
dramatically worse:

```
raw export        : 69 partitions
ORT_ENABLE_BASIC  : 69 partitions   <- use this
ORT_ENABLE_EXTENDED: 191 partitions  <- much worse
```

`ORT_ENABLE_ALL` additionally emits hardware-specific `NhwcFusedConv` that makes
the saved model **unloadable**. Fold at `BASIC` only.

**Load pre-optimized models with `ORT_DISABLE_ALL`.** Re-optimizing an
already-optimized graph throws
`AddInitializedOrtValue Attempt to replace the existing tensor`.
`OrtSessionBuilder.CreateCachedSession` already does this for cache hits.

**The CoreML provider options do not help load time.** Measured, all identical
to three significant figures: `SpecializationStrategy=FastPrediction`,
`RequireStaticInputShapes`, `AllowLowPrecisionAccumulationOnGPU`. Do not go
looking here — the fix is Technique 5.

**fp16 is not a load-time fix.** It shrank the cache 2.16 → 1.52 GB but made
warm load *worse* (18.6 → 27.5 s). It is an inference lever, and a
double-edged one — see below.

## fp16: unresolved

fp16 gave the fastest inference measured — **22.3 ms** on
`MLComputeUnits=CPUAndGPU` (vs 51 ms fp32) — but at a real accuracy cost:

```
fp32:  preds max=3.58E-07   embs max=0.00E+00
fp16:  preds max=1.26E-03   embs max=9.77E-02
```

`preds` at 1e-3 is likely fine for thresholded diarization logits. `embs` at
9.8e-2 is not obviously safe, because `chunk_pre_encode_embs` feeds back into
the spkcache/FIFO for later chunks — a feedback loop, exactly the structure this
codebase already documents error-compounding through (the TF32/OmniVoice note in
`OrtSessionBuilder.cs`). **A single-chunk parity check cannot see compounding.**
Validate with end-to-end DER on real audio before shipping fp16.

Also note `onnxconverter-common` fights graphs containing explicit `Cast` nodes;
expect to patch mixed-dtype boundaries by hand.

## Should we bypass ONNX and call CoreML directly from C#?

**For a model that reaches 1 partition: no.** Measured on the finished Sortformer
graph, same machine state:

| path | inference | load |
|---|---|---|
| direct CoreML (`ct.models.CompiledMLModel`) | 53.54 ms | 0.16 s |
| ORT CoreML EP | 55.06 ms | 2.77 s cold / 0.2 s warm |

ORT's overhead is ~1.5 ms (≈3%), within noise. That is the expected result: at
one partition ORT hands the entire graph to CoreML as a single fused node, so it
is already a thin wrapper around one CoreML call. A second model format and a
native interop layer would buy ~3%.

Note this is a *consequence* of the playbook. At 71 partitions ORT was adding
real overhead in copies and syncs — but the fix was fixing the graph, not
replacing the runtime.

**Where direct CoreML would genuinely add capability** — all things ORT's CoreML
EP cannot currently express:

1. **Stateful models** (macOS 14+). CoreML natively supports state tensors held
   across predictions — the natural fit for KV-cache decoders and for the
   Parakeet TDT decoder, both of which the CoreML EP handles poorly because they
   are `Loop` subgraphs. This is the strongest argument.
2. **Enumerated / flexible shapes.** CoreML accepts a declared *set* of allowed
   input shapes and compiles for each. That could handle Parakeet's variable
   audio length better than ONNX-side bucketing, without padding waste.
3. **`coremltools` compression.** Its fp16, palettization and quantization are
   better calibrated than `onnxconverter-common`, which fights graphs containing
   explicit `Cast` nodes (see the fp16 section).

**Costs to weigh:** .NET's CoreML bindings require a macOS-specific target
framework (`Microsoft.macOS`), which conflicts with the cross-platform Avalonia
build — it would need a native shim plus P/Invoke. It also means a second model
artifact per model to build, host and version, and it bypasses the
`OrtSessionBuilder` abstraction that currently keeps every EP on one code path.

**Recommendation:** stay on ORT for anything that reaches a low partition count.
Revisit direct CoreML only for the `Loop`-bearing decoders, and treat it as a
capability question (can we express this at all?) rather than a performance one.

## Diagnostic workflow for a new model

1. **Does it compile?** Load with the CoreML EP. `error -14` → Technique 1.
2. **Count partitions.** The EP logs it at warning level:
   ```
   CoreMLExecutionProvider::GetCapability, number of partitions supported by
   CoreML: N number of nodes in the graph: M number of nodes supported by CoreML: K
   ```
   One partition is the goal. More than ~5 means throughput is being eaten by copies.
3. **Find what stayed on CPU.** Set `LogSeverityLevel = ORT_LOGGING_LEVEL_VERBOSE`;
   ORT lists every node and its assigned EP after
   `Node(s) placed on [CPUExecutionProvider]`. Group by op type — the offenders
   are usually two or three types repeated per layer.
4. **Check `model.mil` size** in the cache dir against the model's real weight
   volume. Much larger → Technique 5.
5. **Verify parity at every step**, against the *original* model, not the
   previous step — errors compound quietly otherwise.

## Applying to other models

| model | outlook |
|---|---|
| Sortformer | **Done.** Axes constant at runtime; ideal case. |
| Parakeet encoder | Plausible. Length genuinely varies → needs bucketing, and padding waste is a real cost. Technique 5 applies regardless. |
| Parakeet TDT decoder | Poor. `Loop` subgraphs are badly handled by the CoreML EP; `OrtSessionBuilder` already carries a `.ort` workaround for Loop graphs (issue #56). Likely stays CPU. |
| KV-cache decoders (Cohere, Qwen3, VibeVoice, Granite) | Untested. Fixed cache lengths would help; per-step dynamic KV growth is the obstacle. |
| Conv-heavy (Silero VAD, DeepFilterNet3, WeSpeaker) | Promising — conv is the ANE's strength and these graphs are simpler. Start here for quick wins. |

**Technique 5 is worth applying to every model unconditionally**, including ones
never going near CoreML — it is bit-exact and costs nothing.

## Tooling

- `scripts/nemo_export/export_sortformer_nemo_to_onnx.py` — `--coreml-static-batch1`,
  `--coreml-const-lengths` (Techniques 1–2)
- `scripts/nemo_export/coreml_optimize_sortformer.py` — Techniques 3–5 plus
  `--verify` against the original model. The graph transforms are model-agnostic;
  only the verification harness is Sortformer-specific.

## Caveats

- **ORT-version sensitive.** Validated on **1.24.4**. Python ORT 1.29.0
  partitions the same graph into 194 and diverges at ~1e-2. Re-validate on any
  ORT upgrade.
- **Steady-state only.** The static graph assumes full cache/FIFO and a
  full-length chunk. Warm-up chunks must be zero-padded to full size.
- **Contract changes.** Folding prunes the now-unused `*_lengths`, so the graph
  takes three inputs, not six, at fixed shapes. Callers need a variant path.
