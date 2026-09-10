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
| `+ const lengths` (all three — see Technique 2) | 69 | — | — | — | — |
| `+ all-False `Where` removed` | 35 | 97.0 ms | — | — | — |
| `+ `Pad` → `Concat`` | **1** | 51.3 ms | 101.0 s | 20.8 s | 1.49 GB |
| `+ Gemm pre-transpose` | **1** | 52.3 ms | **2.1 s** | **0.2 s** | **0.00 GB** |

Reference on the same machine: **CPU 163.5 ms, WebGPU 94.0 ms.** ⚠ The model card for this bundle records **196.3 ms / 113.0 ms** for the same machine and ORT; the two have never been reconciled. Measured on an M5 under ORT 1.29.0, the stock graph runs at **171.8 ms** on the CPU EP, which sits between them and settles nothing. Treat both 1.24.4 baselines as indicative (#165 item 8).
Outputs match the original dynamic model to `3.6e-07` (preds) and `0.0` (embeddings).

Net: **3.2× vs CPU, 1.8× vs WebGPU, and load went from 101 s to 2.1 s.**

⚠ **On ORT 1.24.4 — which is not the ORT the macOS build ships.** Read the first
caveat at the bottom before treating any of this as a shipped speedup.

## What this bought on Parakeet

Measured on an M5, ORT **1.29.0**, the 0.6B TDT encoder (24 conformer layers,
d_model 1024, full-context `rel_pos`). Techniques 1 and 6 only; no post-processing.

| stage | partitions | inference (10 s bucket) |
|---|---|---|
| shipped dynamic model | *fails to compile* (`error -14`), 84 | — |
| static bucket + mask hoisted to an input | **1** (1453/1453 nodes) | **58.2 ms** |
| same graph, CPU EP | — | 146.9 ms |
| shipped dynamic model, CPU EP | — | 143.6 ms |

**2.5× vs CPU, holding at every bucket from 4 s to 30 s.** Parity against the shipped
encoder is 2e-7…6e-6 on real speech, with 7/7 identical transcripts through the full
TDT decode — and the mask rewrite is `0.000E+00` in PyTorch, so the residual is
ONNX/EP float noise, not the technique.

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
   In Sortformer the trim was a **no-op at runtime** for `spkcache` and `fifo` —
   `Sortformer.cs` always passes those two `*_lengths` equal to each buffer's own
   size — so it was destroying compilability for nothing. The chunk is the
   exception, and it survives only because it is concatenated *last*: the summed
   logical length still masks a short final chunk's padded tail.

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

> ⚠ **Check each length separately.** In Sortformer only `spkcache_lengths` and
> `fifo_lengths` are genuinely constant. `chunk_lengths` is short for the last
> chunk of every recording, so baking it *does* silently change the diarization at
> the end of every file — measured on the shipped model, up to **0.54** (rms 0.24)
> on `preds`' 0..1 scale, enough to flip speaker assignments. Verify against the
> calling code, not the steady state.

**Worth knowing: on its own this barely helped** (71 → 69 partitions). Its real
value is enabling Technique 3 — it turns the masks into foldable constants.

> ⚠ **And that is the trap, because the two halves of this technique conflict.**
> The attention mask is a function of **all three** lengths, so leaving the one
> non-constant length live keeps the mask data-dependent and **Technique 3 removes
> nothing at all** — 0 of 51 `Where` nodes, four inputs instead of three, and no
> path past 69 partitions. There is no middle setting: you either bake a length
> that is sometimes wrong, or you keep the whole mask. Sortformer's resolution is
> to bake all three behind an explicit opt-in
> (`--coreml-const-chunk-length`) and give the resulting steady-state graph a
> **caller obligation**: route the final, short chunk of each recording to the
> unspecialized graph. Every other chunk is full-length, so that costs one
> inference per recording and keeps the output exact. Expect to find this shape
> wherever "the axes are constant at runtime" turns out to mean "constant except
> at the edges".
>
> **"There is no middle setting" holds only if the mask has to stay derived.** When
> the lengths vary on *every* call there is no one-inference escape hatch and this
> trade has no acceptable answer — take the mask out of the graph instead
> (Technique 6). That is exact at any length and needs neither of these flags.

## Technique 3 — Delete all-False `Where` masks

**Symptom:** `Where` nodes on CPU, one or more per encoder layer.

With constant lengths and a fully packed sequence, the attention padding mask
folds to a constant that is **entirely False** (Sortformer: shape
`(1,1,436,436)`, 436 = 188+124+124 — the tensor is exactly full).
`Where(false, -10000, x)` is just `x`. These are identities whose only effect is
fragmenting the graph.

**Precondition:** *every* length feeding the mask must be a graph constant
(Technique 2, including the awkward one). While any of them is a live input the
mask cannot fold, the condition is not an initializer, and this technique is a
no-op — a silent one, since the transform simply reports 0 removed and the
pipeline continues to a graph that looks finished.

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

> ⚠ **On ORT 1.29.0 this technique is a no-op: the CoreML EP now accepts `Pad`.**
> Verified directly — a one-node `Pad` graph (constant mode, `[0,0,4,0,0,4]`) reports
> `1 partition, 1/1 nodes supported`. Parakeet's encoder carries 48 `Pad`s and reaches
> one partition with them left alone; converting them changed neither the partition
> count nor the inference time (58.1 ms with `Pad`, 59.4 ms with `Concat` — noise).
> The op table shipped in the `onnxruntime` wheel
> (`tools/mobile_helpers/coreml_supported_mlprogram_ops.md`) still omits `Pad`, so it
> is stale; probe rather than trust it. Keep the transform for 1.24.4, and check
> before spending a fold round-trip on it — see the note under Technique 5 about what
> that round-trip costs you.

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

> **But the trigger is `Gemm` specifically, not "transformer".** Parakeet's encoder
> traces to 289 `MatMul` and **zero** `Gemm` — its linears run on 3D `[B,T,D]`
> activations, where the exporter emits `MatMul`, while Sortformer's ran on 2D
> reshapes, where it emits `Gemm`. `MatMul`-with-constant-weight takes the binary
> path already: 2.36 GB of weights produced a **4.5 MB** `model.mil`, versus
> Sortformer's 1.49 GB for 0.39 GB. So check the op mix before assuming the
> pathology, and diagnose it the way step 4 says — by `model.mil` size, not by
> reasoning about the architecture.
>
> That matters more than it sounds, because applying this transform (or Technique 3
> or 4) means folding the graph through ORT first, and **that round-trip is what
> makes the artifact refuse to load above `ORT_ENABLE_BASIC`** (`MatMulAddFusion`;
> see the anti-patterns). A graph that needs none of them never round-trips and has
> no such contract term — Parakeet's ships straight from `torch.onnx.export` and
> loads at every optimization level. Reach for the transforms when the partition
> count says to, not by default.

## Technique 6 — Hoist the mask into a graph input

**Use when:** the "constant except at the edges" trade in Technique 2 has no
acceptable answer, because the lengths genuinely vary on every call.

Techniques 2 and 3 offer only two settings — bake a length that is sometimes wrong,
or keep the whole data-dependent mask and stay fragmented. Sortformer escapes that
because exactly one inference per recording is short, so the runtime can route it to
the unspecialized graph. **A model whose every call has a different length has no
such escape**, and the cost of pretending otherwise is not small: on Parakeet, baking
the length moves the encoder output by 0.17–0.24 on valid frames and changes the
transcript by 2–4% WER on 8–20 s segments, worse on short ones.

There is a third setting. The mask is *derived* data — the only thing the length is
used for. Take it out of the graph and accept it as an input:

```
audio_signal  [1, 128, F]     features, zero-padded to a fixed bucket
pad_keep      [1, F]          1.0 for a real frame, 0.0 for padding
```

Now the shapes are static, the `Range`/`Less`/`Expand`/`ConstantOfShape` chain is
**gone rather than folded**, and the masking is exact at any true length ≤ the
bucket. Parakeet: 4955 → 1453 nodes, `Where` 90 → **0**, one partition, and parity
against the shipped dynamic encoder of 2e-7…6e-6 with identical transcripts.

**Do it in float, not bool.** Feed a keep-mask of 1.0/0.0 and rewrite each masking
site as arithmetic — `Add`/`Mul` are CoreML-supported, and it sidesteps the question
of whether the EP takes a bool input at all:

| NeMo | rewrite | why it is exact |
|---|---|---|
| `scores.masked_fill(mask, -INF_VAL)` | `scores + (keep - 1) * INF_VAL` | masked entries become `s - 10000` instead of `-10000`; both underflow to 0 in the softmax, and both are zeroed by the next line anyway |
| `softmax(...).masked_fill(mask, 0.0)` | `softmax(...) * keep` | identical |
| `apply_channel_mask` | `x * keep` | already a multiply |

**Patch the module instances, not the classes** (`types.MethodType`), so nothing
else in the export process is affected.

> ⚠ **Find every masking site first. There are more than the obvious one.** In NeMo's
> conformer the length reaches masking at **four** resolutions: `_create_masks` builds
> the attention and conv-module masks at the encoder rate, and — easy to miss —
> `MaskedConvSequential` re-masks between *every strided conv in the subsampler*.
> Hoisting only `_create_masks` left a **1e-2 error spread across all frames**, which
> reads like a tolerance question rather than a bug, and does not look like a boundary
> artifact. Do not chase it through ONNX: run the patched and unpatched modules
> side by side in PyTorch on the same input and bisect the substitutions one at a
> time. The target is `0.000E+00`, and it is reachable — anything else means a site
> is still deriving its own mask.

**The masks at different rates are usually slices of each other, not separate
inputs.** Each subsampler stage halves the length as
`L_k = (L_{k-1} - 1) // 2 + 1 = ceil(L_{k-1} / 2)`, and `keep[2i]` is 1 exactly while
`2i < L` — which is `ceil(L / 2)` entries. So stage *k*'s mask is `keep[0::2]` applied
*k* times, and one mel-rate input feeds all four via constant-parameter `Slice`
(supported). Prove the identity exhaustively over every length before relying on it;
it is a five-line check.

**What it costs: bucketing.** Static shapes mean a fixed `F`, so a caller with
variable input pads up to the next bucket and pays for the padding. Measured on an
M5, one bucket per row:

| bucket | CoreML | CPU EP | speedup |
|---|---|---|---|
| 400 frames (4 s) | 34.9 ms | 87.2 ms | 2.50× |
| 1000 frames (10 s) | 58.2 ms | 146.9 ms | 2.52× |
| 2000 frames (20 s) | 110.9 ms | 279.4 ms | 2.52× |
| 3000 frames (30 s) | 165.2 ms | 445.7 ms | 2.70× |

≈ 16 ms fixed + 4.75 ms per second of audio, so the speedup holds at every size and
padding waste is proportional rather than catastrophic. Note the ladder is a
*runtime* cost, not a download one: every bucket traces the same parameters in the
same order, so the weight sidecars come out **byte-identical** — one shared
`.data` file and an ~25 MB graph per bucket. Verify by digest rather than assuming;
a torch or NeMo change that reorders the trace would break it silently.

**The real per-bucket cost is the CoreML cache: ~4.4 GB each**, because the EP stores
the weights twice — once in the `.mlpackage` under `Data/`, once in the compiled
`.mlmodelc`. Four buckets is 18 GB on disk. Budget for it, prune stale entries, and
let it argue for a short ladder.

## Technique 7 — `ModelCacheDirectory`

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

**Load pre-optimized models with `ORT_DISABLE_ALL` — or `ORT_ENABLE_BASIC`, but
never higher.** Re-optimizing an already-optimized graph throws
`AddInitializedOrtValue Attempt to replace the existing tensor`.
`OrtSessionBuilder.CreateCachedSession` already does this for cache hits, but
`Create` defaults to `ORT_ENABLE_ALL`, so a *fresh* session on a pre-optimized
file — which is exactly what the shipped CoreML variant is — has to pass the level
explicitly or it throws at load.

Bisected on ORT 1.26.0 with `disabled_optimizers`: the thrower is
**`MatMulAddFusion`**, an EXTENDED-level pass. It reshapes >2D `MatMul` inputs so
it can emit a `Gemm`, and the already-folded graph is full of the initializers it
generates (`gemm_input_shape_token_N`, `gemm_output_reshape_token_N_new_shape`)
because the fold pass already ran it. Renaming all 340 of those out of the way does
**not** fix it, so it collides on something the fusion mints fresh — treat it as an
ORT defect to route around, not a graph to reshape. The four transforms here are
innocent: a BASIC-folded graph with *none* of them applied fails identically.

**The CoreML provider options do not help load time.** Measured, all identical
to three significant figures: `SpecializationStrategy=FastPrediction`,
`RequireStaticInputShapes`, `AllowLowPrecisionAccumulationOnGPU`. Do not go
looking here — the fix is Technique 5.

**fp16 is not a load-time fix.** It shrank the cache 2.16 → 1.52 GB but made
warm load *worse* (18.6 → 27.5 s). It is an inference lever, and a
double-edged one — see below.

## fp16: the accuracy question is answered, the speed question is per-EP

fp16 gave the fastest inference measured — **22.3 ms** on
`MLComputeUnits=CPUAndGPU` (vs 51 ms fp32) — at an apparent accuracy cost:

```
fp32:  preds max=3.58E-07   embs max=0.00E+00
fp16:  preds max=1.26E-03   embs max=9.77E-02
```

`embs` at 9.8e-2 looked unsafe because `chunk_pre_encode_embs` feeds back into the
spkcache/FIFO for later chunks — a feedback loop, exactly the structure this codebase
already documents error-compounding through (the TF32/OmniVoice note in
`OrtSessionBuilder.cs`) — and a single-chunk parity check cannot see compounding.

**Measured end to end (#172): it does not compound.** Fidelity DER against NeMo is
**0.000%** on three 90 s real-speech samples. The error does grow through the feedback path
— 9.8e-04 on one chunk becomes 1.1e-02 to 3.4e-02 over a recording — but the per-chunk trace
wanders rather than trending, and only 3 frames in 3378 flip their binarized speaker set,
all isolated enough for the median filter to absorb. So the caveat was right to demand the
check and wrong about the outcome.

**What decides fp16 now is throughput, and it is not uniform:**

| EP | fp32 | fp16 |
|---|---|---|
| CPU (x86-64) | 333.0 ms | **415.3 ms** — 25% slower |
| CUDA (3090) | 17.6 ms | **10.6 ms** — 1.66× faster |
| CoreML (M-series) | 51 ms | 22.3 ms *(unre-measured; see below)* |

There are no native fp16 CPU kernels, so ORT casts up and back around every op. fp16 is an
**EP-gated variant**, never a replacement — shipping it as the default slows down every CPU
user. The CoreML figure predates the reproducible converter and should be re-measured before
it is relied on.

`scripts/nemo_export/fp16_convert_sortformer.py` produces the model. Note that a bare
`convert_float_to_float16` call does not: the graph needs its length arithmetic held in
fp32 (it is a frame count, not an activation), its explicit `Cast` nodes reconciled (the
converter rewrites tensor types but not `to` attributes), internal consumers of
`keep_io_types`-cast outputs rewired, and `ScatterElements` held back because the CPU EP has
no fp16 kernel for it. See `docs/investigations/sortformer_fp16_investigation.md`.

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

0. **Ask what the varying axes actually are, and how the caller uses them.** This
   decides which branch of the playbook you are on before you measure anything.
   Constant at runtime, or constant except at the edges → Techniques 1–3, and read
   Technique 2's warning. Genuinely different on every call → Technique 6, and do not
   spend time on the bake-a-length path; it was measured on Parakeet and it costs
   2–4% WER.
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
   volume. Much larger → Technique 5. (Sortformer: 1.49 GB for 0.39 GB of weights.
   Parakeet: 4.5 MB for 2.36 GB — nothing to fix.)
5. **Verify parity at every step**, against the *original* model, not the
   previous step — errors compound quietly otherwise. **Verify in the framework
   before you verify through ONNX**: when a rewrite changes what the model computes,
   a PyTorch-vs-PyTorch comparison isolates it from export and EP noise, and gives
   you a real `0.000E+00` to aim at rather than a tolerance to argue about.
   Bisect substitutions one at a time — the mask hoist looked like a 1e-2 tolerance
   question until that turned it into "three masking sites are still missing".

## Applying to other models

| model | outlook |
|---|---|
| Sortformer | **Done.** Axes constant at runtime; ideal case. |
| Parakeet encoder | **Done, via Technique 6.** One partition, 2.5× over CPU at every bucket, bit-exact. Baking the length (the Sortformer recipe) was measured and rejected — 2–4% WER. |
| Parakeet TDT decoder | **Measured, and not worth it** — and not a `Loop` graph, as recorded here previously. It is 42 nodes with two `LSTM`s, stepped from C#. With its shapes frozen (they are all constant in the greedy path) it puts **23 of 27 nodes on CoreML**, only `LSTM` declining — and runs at **0.652 ms vs 0.643 ms on CPU**. The work per call is too small to pay for two partition boundaries. See "when not to bother" below. |
| `nemo128` preprocessor | **No.** 6 partitions, 31/89 nodes, and CoreML is *slower*: 2.95 ms vs 1.71 ms. `Parakeet.cs` already pins it to CPU, correctly. It is 1% of pipeline time. |
| Silero VAD | **Cannot.** Fails to compile outright — `Error compiling model: Failed to parse the model specification`; it carries three `If` subgraphs. Costs 2.0 ms per second of audio (1.2 s for a 10-minute recording), once per recording. |
| KV-cache decoders (Cohere, Qwen3, VibeVoice, Granite) | Untested. Fixed cache lengths would help; per-step dynamic KV growth is the obstacle. |
| Conv-heavy (Silero VAD, DeepFilterNet3, WeSpeaker) | Promising — conv is the ANE's strength and these graphs are simpler. Start here for quick wins. |

### When not to bother — profile the pipeline, not the model

Partition count tells you whether a graph *can* go fast. It says nothing about
whether the pipeline gets faster, and two of the three checks below are cheaper than
any of the techniques above.

**Is the model actually where the time goes?** Measured over 72.5 s of real speech,
ten segments, timing each ORT session separately:

| stage | share | after the encoder's 2.5× |
|---|---|---|
| `nemo128` preprocessor | 1.0% | 1.8% |
| encoder | 77.9% | 58.5% |
| `decoder_joint` (×20–94 calls per segment) | 21.2% | 39.8% |

A 2.5× on the encoder is **1.88× end to end**. Everything else in the stack was
either slower on CoreML or unable to compile, so that is the whole prize — and it is
worth knowing before, not after.

**Is the per-call work big enough to pay for the boundary?** A partition boundary
costs a copy and a sync each way. `decoder_joint` reaches 23/27 nodes on the EP and
still does not move (0.652 vs 0.643 ms) because each call is one frame and one token.
Small-and-frequent loses to CoreML even when it partitions well; the encoder wins
because each call is 10 s of audio through 24 layers.

**Does a runtime fallback hide the answer?** The first `decoder_joint` probe reported
2 partitions and identical timing — because its CoreML partitions then failed to
compile on unbounded dims and silently fell back to CPU. The capability report and
the timing were both "fine" while nothing ran on the EP. Freeze shapes (Technique 1)
*before* concluding anything from a timing comparison, and check the log for
`has unbounded dimension` rather than trusting a clean partition count.

**Technique 5 is worth applying to every model that has `Gemm` nodes**, including
ones never going near CoreML — it is bit-exact and costs nothing. Check first: a
`MatMul`-only export (Parakeet) does not have the problem, and applying the transform
means a fold round-trip that adds a load-level contract term.

## Tooling

- `scripts/nemo_export/export_sortformer_nemo_to_onnx.py` — `--coreml-static-batch1`,
  `--coreml-const-lengths`, `--coreml-const-chunk-length` (Techniques 1–2). All three
  are needed to reach one partition; the first two alone stop at 69.
- `scripts/nemo_export/coreml_optimize_sortformer.py` — Techniques 3–5 plus
  `--verify` against the original model. The graph transforms are model-agnostic;
  only the verification harness is Sortformer-specific.
- `scripts/nemo_export/export_parakeet_coreml_encoder.py` — Techniques 1 and 6 for the
  Parakeet encoder: static buckets, mask hoisted to a `pad_keep` input, all four
  masking sites rewritten as arithmetic. Needs **no** post-processing — the raw export
  is already one partition — and emits one shared weight sidecar for every bucket.
- `scripts/nemo_export/coreml_partition_probe.py` — the partition/timing/load-level
  probe. `--ep` selects what you measure *and* what you can see: a load-level matrix
  is only meaningful per-EP.

## Caveats

- **The numbers in the table below were measured on ORT 1.24.4; 1.29.0 is what ships
  and has since been measured too.** `Directory.Build.props` pins 1.24.4 only for
  DirectML; an Apple Silicon build is `-p:EP=Cpu` and takes the default, **1.29.0**. It
  was once believed 1.29.0 split this graph into 194 partitions and diverged at ~1e-2 —
  **that does not reproduce.** On an M5 under 1.29.0 the graph reaches **1 partition**
  and `4.470E-07` against the stock model, at **51.5 ms** vs 171.8 ms for stock-on-CPU.
  The variant shipped on that basis; see
  `docs/investigations/sortformer_coreml_publish_investigation.md`.
- **Steady-state only.** The static graph assumes a genuinely full cache/FIFO and a
  full-length chunk. Zero-padding a partly-filled buffer is **not** free: the baked
  lengths claim every frame is real, so padding gets attended to as audio. Both the
  warm-up chunks (cache/FIFO still filling) and the final short chunk of each recording
  must go to the unspecialized graph instead (Technique 2's warning).
- **Contract changes.** Folding prunes the now-unused `*_lengths`, so the graph
  takes three inputs, not six, at fixed shapes. Callers need a variant path.
- **Load level is part of the contract**, not a tuning knob: `ORT_ENABLE_BASIC` or
  lower, or the file does not open. See the anti-patterns above.
