# Parakeet encoder — CoreML variant investigation

Goal: give the Parakeet TDT encoder a CoreML-compilable export, the way
`docs/coreml_onnx_playbook.md` did for Sortformer. The playbook's recipe turns out
not to transfer, and the reason is worth the write-up.

## Environment

- MacBook / **Apple M5**, 24 GB, macOS 25.6.0. The machine that decides this, since
  it is the only one with CoreML.
- export venv `.venv-nemo-export`, python 3.12.14, `nemo-toolkit 2.7.1`,
  `torch 2.14.0`, `onnx 1.22.0`, **`onnxruntime 1.29.0`** — the version an
  `EP=Cpu` osx-arm64 build takes (`Directory.Build.props` pins 1.24.4 for DirectML
  only).
- source checkpoint `nvidia/parakeet-tdt-0.6b-v3` (`parakeet-tdt-0.6b-v3.nemo`,
  2,509,332,480 B).
- reference bundle: the published `christopherthompson81/sortformer_parakeet_onnx`
  `encoder-model.onnx` (41,842,955 B) + `encoder-model.onnx.data`
  (2,435,420,160 B, md5 `2f53c7ed168d73ea305ac2f53bdac097`) — both match the
  manifest.
- speech: `~/Programming/test_speech_audio/en-US_sample_0{1,2,3}.wav`, three ~10 min
  16 kHz mono conversations.

The model: `ConformerEncoder`, 24 layers, d_model 1024, 8 heads, `rel_pos` with
`att_context_size [-1, -1]` (**full context**), `dw_striding` subsampling ×8,
conv kernel 9, 128 mel bins.

## Run 1 — the stock graph, on CoreML

```
CoreMLExecutionProvider::GetCapability ... partitions: 84, nodes: 2325, supported: 2022
Failed to create MLModel ... error code: -14
E5RT ... Failed to PropagateInputTensorShapes: Invalid tensor rank 0 inferred from: ios18.squeeze
```

Exactly the playbook's Technique 1 symptom. The raw graph is 4955 nodes:
289 `MatMul` and **zero `Gemm`**, 90 `Where`, 48 `Pad`, and a large
`Shape`/`Gather`/`Reshape`/`ConstantOfShape` mass from the tracer.

`decoder_joint-model.onnx` was inspected at the same time, because the playbook
records it as a `Loop` graph. **It is not.** 42 nodes, two `LSTM`s, stepped once per
output token from `Parakeet.cs`. `LSTM` is absent from the CoreML EP's op set, so it
stays on CPU — but for a different reason than recorded, and it is the wrong thing to
accelerate regardless.

## Run 2 — does the Sortformer recipe transfer? No.

The playbook reaches one partition by baking every length constant so the attention
mask folds to an all-False constant. Sortformer affords that because exactly one
chunk per recording is short. Parakeet's segments are arbitrary — the question is
what baking costs.

Measured on real speech: pad the features to a bucket, then run the **shipped**
encoder twice, once with the honest `length` and once with the padded length, and
compare over valid frames. Transcripts come from a Python port of `Parakeet.cs`'s
greedy TDT loop (`MaxTokensPerStep` and all), so WER is against the model's own
output, not a reference transcript.

| segment | encoder maxAbs (valid frames) | WER vs. honest-length output |
|---|---|---|
| 8 s | 0.234 | 0.040 |
| 12 s | 0.179 | 0.000 |
| 20 s | 0.167 | 0.024 |
| 3 s | 0.140 | 0.33 – 0.50 |
| 5 s | 0.170 | 0.06 – 0.18 |

The error saturates almost immediately with padding — +10% padding is as bad as
+100% — so it is "the mask is gone", not "the padding is large". **There is no
setting of the Sortformer recipe that is acceptable here**, and no one-inference
escape hatch: every segment pays.

Padding with real trailing *silence* instead of zeros (letting the preprocessor see
it) was measured in the same run and is no better — 0.04 to 0.15 WER.

## Run 3 — the property that makes a third option possible

With an **honest** mask, the stock encoder is padding-invariant:

| segment | pad to +10% | +50% | +100% |
|---|---|---|---|
| 3 s | 3.6e-07 | 3.6e-07 | 3.6e-07 |
| 8 s | 0.0 | 0.0 | 0.0 |
| 20 s | 8.1e-07 | 8.1e-07 | 8.1e-07 |

So a static-shape graph does not need a constant length — it needs the mask, and the
mask can be an *input*. That is the whole design:

```
audio_signal  [1, 128, F]     zero-padded mel features
pad_keep      [1, F]          1.0 real / 0.0 padding
```

Exact at any true length ≤ the bucket, and the `Range`/`Less`/`Expand`/
`ConstantOfShape` chain is gone rather than folded.

## Run 4 — the first attempt was wrong, and wrong in a way that looks like tolerance

Hoisting `_create_masks` and rewriting the three `masked_fill`s per layer as
arithmetic gave a graph with `Where` 90 → 17, static shapes, and parity of **2e-2**
against the reference. Which reads like a tolerance question.

It was not. Bisecting in PyTorch rather than through ONNX (patched vs. unpatched
modules, same input) showed:

* the error was **spread across every frame**, not concentrated at the padding
  boundary — so not a boundary artifact;
* it was **zero whenever the true length was a multiple of 8**, and nonzero otherwise;
* the three arithmetic substitutions were individually exact — control, attention-only,
  conv-only and both-arithmetic all produced the *same* 9.67e-3, i.e. the rewrite was
  innocent;
* `enc.pre_encode(x, lengths=300)` vs `lengths=1000` differ by **2823**.

`length` does not only build the attention mask. `MaskedConvSequential` re-masks
between **every strided conv in the subsampler** — four resolutions in total, not one.
Hoisting one of them and leaving three deriving their own masks is a silent 1e-2.

The fix is that all four masks are exact stride-2 decimations of each other:
`L_k = (L_{k-1} - 1) // 2 + 1 = ceil(L_{k-1} / 2)`, and `keep[2i]` is 1 exactly while
`2i < L`, which is `ceil(L / 2)` entries. Verified exhaustively for every length in
five bucket sizes: 0 mismatches. So one mel-rate input feeds all four through
constant-parameter `Slice`s.

With that, PyTorch parity against the stock encoder is **`0.000E+00`** at true lengths
300, 301, 500, 799, 800, 997 and 1000 — bit-exact, not close.

## Run 5 — the export, and what it did not need

`scripts/nemo_export/export_parakeet_coreml_encoder.py`. Graph: 4955 → 4104 nodes,
`Where` **90 → 0**, `Range`/`Less`/`Equal`/`Expand` all gone, both inputs static.

Running it through the playbook's post-processing (BASIC fold + Pad→Concat) works —
1453 nodes, 48 `Pad`s converted, every shape resolved, 0 unresolved dims — and is
**unnecessary**:

| graph | partitions | inference (10 s bucket) |
|---|---|---|
| raw export | 1 (1453/1453) | 58.1 ms |
| folded + Pad→Concat | 1 (1453/1453) | 59.4 ms |

ORT's load-time BASIC fold does the same work, and the CoreML EP in 1.29.0 **accepts
`Pad`** — confirmed on a one-node graph, `1 partition, 1/1 supported`. (The op table
shipped in the wheel still omits `Pad`; it is stale.)

That is worth more than the 1.3 ms, because the fold round-trip is what gave the
Sortformer artifact its `ORT_ENABLE_BASIC or it will not load` contract term
(`MatMulAddFusion`). Checked here on the CPU EP: the raw export loads at `basic`,
`extended` **and** `all`. No contract term.

Technique 5 also does not apply — 289 `MatMul`, 0 `Gemm`. `model.mil` came out at
**4.5 MB** against 2.36 GB of weights, versus Sortformer's 1.49 GB for 0.39 GB.

## Run 6 — measurements

M5, ORT 1.29.0, one bucket per row, `ModelCacheDirectory` set:

| bucket | CoreML | CPU EP (same graph) | speedup | cold | warm |
|---|---|---|---|---|---|
| 400 (4 s) | 34.9 ms | 87.2 ms | 2.50× | 22.0 s | 0.98 s |
| 1000 (10 s) | 58.2 ms | 146.9 ms | 2.52× | 22.3 s | 0.98 s |
| 2000 (20 s) | 110.9 ms | 279.4 ms | 2.52× | 23.4 s | 1.07 s |
| 3000 (30 s) | 165.2 ms | 445.7 ms | 2.70× | 25 s | 1.1 s |

Shipped dynamic graph on the CPU EP, 1000 frames: **143.6 ms**. So ≈16 ms fixed +
4.75 ms per second of audio, and the 2.5× holds at every size — no crossover where
padding waste eats the win.

Parity through ONNX against the shipped encoder, seven real-speech segments of
3–9.5 s: **2.2e-07 … 5.8e-06**, and **7/7 transcripts identical** through the full
TDT decode.

## Run 7 — the buckets are nearly free to ship

Every bucket traces the same parameters in the same order, so the weight sidecars are
byte-identical — and identical to the **published** `encoder-model.onnx.data`
(md5 `2f53c7ed168d73ea305ac2f53bdac097`). Confirmed by repointing a bucket graph's
external-data locations at the shipped file and running it.

So the variant adds **no weight download**: one `.onnx` per bucket, 10.6 / 25.3 /
49.9 / 74.5 MB for 4 / 10 / 20 / 30 s. The size scales with the bucket because
`linear_pos(pos_emb)` folds to a per-layer constant of `(2T-1) × 1024` floats — that
is a runtime win (one fewer matmul per layer) paid for in file size, and it cannot be
shared because `linear_pos` differs per layer.

The exporter verifies the sharing by digest rather than assuming it, and prints the
md5 so it can be diffed against the published manifest.

**The real per-bucket cost is the CoreML cache: 4.4 GB each**, because the EP stores
the weights twice — `Data/com.microsoft.OnnxRuntime/weights/weight.bin` and
`compiled_model.mlmodelc/weights/weight.bin`, 2,358,295,904 B apiece. Four buckets is
18 GB. That, not download size, is what should decide how long the ladder is.

## Run 8 — is the encoder enough? (the other three models)

Asked directly: does `decoder_joint`, `nemo128` or `silero_vad` also need a CoreML
variant before this stack counts as CoreML-ready? Measured rather than argued.

**Where the time goes.** Ten segments, 72.5 s of real speech, timing each ORT session
separately (`session.run` only, so the ported Python decode loop's own overhead is
excluded):

| stage | ms | share | after the encoder's 2.5× |
|---|---|---|---|
| `nemo128` | 14.8 | 1.0% | 1.8% |
| encoder | 1210.7 | 77.9% | 58.5% |
| `decoder_joint` | 329.3 | 21.2% | 39.8% |

20–94 decoder calls per segment at a very steady **0.66–0.69 ms** each. Amdahl:
2.5× on the encoder is **1.88× end to end**.

**`decoder_joint` — measured no.** The first probe reported 2 partitions / 18 of 26
nodes and timing identical to CPU (0.649 vs 0.657 ms), which looks like "CoreML gains
nothing". It was not a fair test: the log carried
`Input: encoder_outputs has unbounded dimension which is not supported`, so its CoreML
partitions failed to compile and ORT fell back to CPU. Both readings were "fine" while
nothing ran on the EP.

Every one of its axes *is* constant in the greedy path (batch 1, one encoder frame,
one target token, `[2,1,640]` states), so freezing them is legitimate. Frozen:
**2 partitions, 23 of 27 nodes on CoreML, 0 unbounded-dim errors, only `LSTM` left on
CPU** — and **0.652 ms vs 0.643 ms**. No gain. The per-call work is one frame and one
token; it cannot pay for two partition boundaries.

**`nemo128` — no.** 6 partitions, 31 of 89 nodes, and CoreML is **1.7× slower**
(2.95 ms vs 1.71 ms). `Parakeet.cs` already constructs it with plain `SessionOptions`
rather than `OrtSessionBuilder.Create(ep)`; that is correct and should stay.

**`silero_vad` — cannot.** Fails to compile outright:
`Error compiling model: Failed to parse the model specification`. The graph carries
three `If` subgraphs. It costs 0.065 ms per 512-sample window, i.e. **2.0 ms per
second of audio — 1.2 s for a 10-minute recording**, once per recording rather than
per segment.

**Batching, the last open caveat.** The concern was that `Parakeet.cs` batches up to
32 segments while the CoreML path is batch-1, so the 2.5× might not survive. It does
— CPU batching is not a win to give up:

| batch | total | per segment |
|---|---|---|
| 1 | 141.8 ms | **141.8 ms** |
| 2 | 445.3 ms | 222.7 ms |
| 4 | 682.0 ms | 170.5 ms |
| 8 | 1301.6 ms | 162.7 ms |
| 16 | 2618.6 ms | 163.7 ms |

Batch-1 is the *best* CPU case; the CPU EP already saturates its threads on a single
segment. So the batch-1 CoreML design gives up nothing.

Conclusion: **the encoder was the only worthwhile target in this stack.** The next win
after it is not a CoreML variant of anything — it is the decoder's 40%, which wants
batched greedy decoding across segments (NeMo's label-looping decoder) or int8, both
outside this line of work.

## Run 9 — wired into the app, and the answer changes

The runtime path now exists: `Config` carries the ladder and the encoded-length formula,
`Parakeet` opens buckets lazily with a signature check, `ModelManagerService` fetches them
on Apple Silicon only, and the CLI reports which graph each segment ran on.

**Correctness: confirmed.** On a 90 s clip (32 VAD segments) and the full 10-minute file
(132 Sortformer segments), `--ep coreml` produces transcripts **byte-identical** to
`--ep cpu`, with 32/32 and 131/132 segments routed to buckets (the one exception is a
33.9 s segment, longer than the largest bucket, which correctly falls through to the stock
graph).

**Two things found on the way that were not on anyone's list.**

*Parakeet on the WebGPU EP segfaults on macOS — and the cause is ours.*
`webgpu::BufferManager::Release` under `~InferenceSession`, so the process dies at
teardown — **after** transcription and **before** the caller writes the file. The symptom
is a crash and a missing transcript, not a wrong one. It reproduces on the tree before any
of this work, with plain `--ep auto` (Auto resolves to WebGPU on macOS), so the default
macOS ASR path was already losing output.

⚠ **First diagnosed here as an ORT teardown bug and worked around by keeping macOS on the
CPU EP. That was wrong**, and the workaround silently cost macOS the WebGPU speedup. The
cause is `Parakeet`'s constructor building **one `SessionOptions` and handing it to both
sessions**; on WebGPU the second `Dispose` frees already-freed buffers. Reduced to two
sessions and one shared options object with nothing else involved: shared → SIGSEGV, one
options object each → clean, `exit=0`. Fixed at the constructor.

The lesson generalises past this bug: "reproduces before my change" establishes that
something is pre-existing, not that it is someone else's. The stack trace named the EP
because that is where the free happens, not where the mistake is.

⚠ **The same pattern is in at least four other backends** — `WhisperTurbo` (one options
object across 3 sessions), `CohereTranscribe` (3), `GraniteSpeech` (4),
`IndicConformer` (2). Each is the same latent crash on the macOS default path. Not fixed
here.

*The `ep` argument never reached Parakeet.* All three call sites — CLI, `TranscriptionService`,
`TranscriptEditorViewModel` — constructed it with the default `Auto` regardless of the
user's choice, so `--ep coreml` had never reached the ASR encoder. Same class of miss as
the one #164 left in `Sortformer.cs`.

**Performance: the variant does not pay yet.** 10-minute file, 132 segments, warm cache:

| | CPU EP | CoreML buckets |
|---|---|---|
| encoder inference | 13.6 s | **8.7 s** (1.56×) |
| opening buckets | — | **11.7 s** |
| encoder total | **13.6 s** | 20.4 s |
| diarization (Sortformer) | 13.1 s | **6.6 s** (2.0×) |
| whole pipeline | **31.4 s** | 32.0 s |

And with the teardown crash fixed, the option that was invisible while WebGPU could not
finish a run turns out to beat both — same 10-minute file, 159 VAD segments:

| encoder | CPU EP | CoreML buckets | **WebGPU** |
|---|---|---|---|
| | 10.8 s | 19.5 s (10.1 inference + 9.4 load) | **7.1 s** |

WebGPU takes the **stock dynamic graph unmodified**: no buckets, no padding waste, no
per-bucket weight blob, no compiled cache. It reaches the GPU rather than the ANE, so its
ceiling is lower than the buckets' per-inference 2.5× — but it collects its win instead of
spending it on loading. Verified numerically against the CPU EP at arbitrary
non-bucket-aligned lengths (200/517/1000/2311 frames): 4e-7 … 3.6e-6.

Two independent reasons, both invisible from the per-inference benchmark:

1. **Bucket sessions cost ~2.9 s each to open, warm**, charged per bucket per `Parakeet`
   instance, and the app builds one per transcription. Four buckets is 11.7 s against an
   inference saving of 4.9 s.
2. **Real segments are much shorter than the buckets.** p50 is 2.3 s against a 4 s smallest
   bucket; mean fill ~47%, so the padded work is 2.02× the real audio. Run 6's 2.5× assumed
   a full bucket.

Modelling ladders from one to four buckets against the measured distribution puts the best
at **~16.5 s**, still behind the CPU's 13.6 s — fewer buckets means less loading but more
padding and more fall-through. There is no ladder that wins without first removing the
load cost.

For scale: the whole pipeline takes **31.4 s** here against **~7 s on an RTX 3090** — the
M5 is roughly **4.5× slower end to end**. So this is not a case of "the baseline is already
fast enough"; there is real headroom on Apple Silicon. What blocks the buckets from
claiming any of it is specifically the per-session load cost, which is why the fix is to
hide or amortise that rather than to give up on the ANE.

**So `Auto` deliberately does NOT select the buckets** (unlike the Sortformer variant,
where Auto does). An explicit `--ep coreml` still gets them. The gate can be reopened when
the load cost is hidden — opening buckets concurrently with diarization, or holding them
across files — or when the ladder is re-cut for the observed distribution.

Diarization is unaffected and is a clear win: 13.1 s → 6.6 s on the same file.

## Retired — 2026-09-10

**Decision: the encoder buckets are not shipped.** Removed from the download list, from
`Parakeet`, and from `Config`; the published files stay on HuggingFace for anyone
reproducing this, and the exporter and Technique 6 stay in the playbook, because the
technique is sound and generalises — it is this *application* of it that does not pay.

The case against, in one line each:

* WebGPU runs the **stock dynamic graph** at 7.1 s against the CPU EP's 10.8 s on the same
  10-minute file — faster than the buckets manage (19.5 s including load), with no
  re-export, no padding waste, no bucketing and no compiled cache.
* The ANE's 1.56× on inference is real and is spent entirely on opening bucket sessions
  (~2.9 s each, and a static-shape design needs several).
* Real segments fill a bucket ~47%, so half the padded work is wasted before loading counts.

What retiring costs, and why it needed more than deleting a list entry: each shipped bucket
leaves a **~4.4 GB compiled CoreML bundle** in the cache root, and the existing prune only
reclaims superseded versions of a model something still opens — a model that simply stops
being used is never reclaimed at all. Four buckets is ~17.6 GB that would have sat there
forever on every Mac that fetched them. `OrtSessionBuilder.ForgetCoreMLCacheFor` is the
reclaim; `ModelManagerService.RemoveRetiredAssets` runs it on the next download pass.
Measured on this machine: **19.20 GB reclaimed**, Sortformer's own CoreML cache untouched.

## Where this leaves the variant

Done and validated as an artifact. **Not yet consumable from the app**, and the
remaining work is all on the C# side or is a product decision:

1. **The bucket ladder is unchosen.** It should come from the segment-length
   distribution the diarizer and VAD actually produce, weighed against 4.4 GB of
   cache per bucket. Nothing in `Config.cs` caps segment length today.
2. ~~**`Parakeet.cs` has no CoreML path.**~~ Built in Run 9, and behind an explicit
   `--ep coreml` rather than on by default: the ANE wins the inference and the bucket
   session loads hand it straight back. The open item is now **hiding the load cost**
   (open buckets concurrently with diarization, or hold them across files) or **re-cutting
   the ladder** for the measured segment distribution — p50 2.3 s against a 4 s smallest
   bucket. Note Run 8's 1.88× projection was itself optimistic: it counted inference only.
3. **fp16 is untested here** and would halve both the sidecar and the 4.4 GB cache.
   The playbook's fp16 section is Sortformer-specific; on CoreML it was the fastest
   variant measured there.
4. **Not published.** No HF upload, no manifest entry.
