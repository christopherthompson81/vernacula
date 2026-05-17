# Chatterbox long-form perf investigation

Per `chatterbox.scratch.md` Stage 1, the eventual app synthesizes
long-form markdown — many paragraph-sized chunks per CLI invocation,
not the one-shot we ship today. The optimization surface for that goal
differs from the one-shot's: per-chunk costs dominate while load time
amortizes to nothing.

The opening analysis (see PR-comment thread on #64) projected the
following ladder of attack:

| Stage | Strategy | Expected wall reduction for a 5-min audiobook |
|---|---|---|
| Today | Serial chunks | baseline |
| 1 | LM/vocoder pipelined per chunk | ~25% |
| 2 | LM-batched B=4 | ~70% |
| 3 | + LM fp16 quantization | ~85% |

This investigation tackles them in order. Stage 1 first because it's
the smallest code change (one new orchestrator class in
`Chatterbox.Base`) with no risk surface — neither the export nor any
existing pipeline math changes.

Test methodology: drive synthesis via a throw-away C# probe that
constructs one `ChatterboxPipeline`, embeds the voice once, then
synthesizes N text chunks. Measure wall time end-to-end and per-stage
where instrumentation allows. The same chunk text is used N times so
the LM rollout length is identical across runs — isolates the
pipelining lever from any chunking-strategy variability.

Test bed: RTX 3090, ORT 1.24.4, CUDA EP, ChatterboxSmoke's standard
voice (`/home/chris/Downloads/VCTK_p303.wav`), pre-resampled to 24
kHz off-line (see issue #53 / Run 12 in
`docs/chatterbox_investigation.md` for why this matters).

## Run 1 — 2026-05-17 18:35 — Baseline: serial chunks

ChatterboxSmoke gains a `--bench-chunks N` flag that reuses one
pipeline + one speaker embedding across N serial chunk synthesizes
(same text each time so the LM rollout length is identical and
inter-run noise is minimal).

```
$ dotnet run --project tests/ChatterboxSmoke -- \
    --onnx-dir /tmp/cb_dyn5 --voice /home/chris/Downloads/VCTK_p303.wav \
    --bench-chunks 8 --ep cuda

Loaded sessions in 3847 ms total  (requested=cuda, effective=cuda)
Loaded voice /home/chris/Downloads/VCTK_p303.wav: 312936 samples (13.04s)  [15 ms]
  chunk 1/8: LM 1485 ms (174 steps), voc 710 ms, audio 6.92s
  chunk 2/8: LM 1234 ms (174 steps), voc 534 ms, audio 6.92s
  chunk 3/8: LM 1299 ms (174 steps), voc 535 ms, audio 6.92s
  chunk 4/8: LM 1248 ms (174 steps), voc 535 ms, audio 6.92s
  chunk 5/8: LM 1245 ms (174 steps), voc 535 ms, audio 6.92s
  chunk 6/8: LM 1240 ms (174 steps), voc 535 ms, audio 6.92s
  chunk 7/8: LM 1252 ms (174 steps), voc 534 ms, audio 6.92s
  chunk 8/8: LM 1266 ms (174 steps), voc 537 ms, audio 6.92s
Bench: 8 chunks (serial), LM avg 1284 ms, voc avg 557 ms,
  per-chunk avg 1840 ms, chunks-total 14.7s [total wall 19.3s]
```

Per-chunk numbers (steady-state, dropping chunk 1 as warmup):

| | value |
|---|---|
| LM | 1252 ms avg (174 steps × 7.2 ms/step) |
| Vocoder | 535 ms |
| Per-chunk total | 1787 ms |
| Audio produced | 6.92 s |
| **Real-time factor** | **0.26** (vs the opening analysis's 0.35 projection — better than expected) |
| LM share of per-chunk | 70% |

8-chunk wall breakdown: 3.85 s session load + 0.7 s voice/embed +
14.7 s chunk-total = 19.3 s.

### Pipelined target (theoretical)

With LM(N+1) running concurrent with vocoder(N) on the GPU, the
critical path becomes max(LM, voc) per chunk plus a vocoder tail:

```
t_pipelined = LM_1 + Σ max(LM_n, voc_{n-1}) + voc_N
            = LM × N + voc_N             (since LM > voc here)
            = 8 × 1284 + 557
            = 10829 ms
```

Predicted wall: 4.55 s warmup + 10.83 s chunks = **15.4 s**, a
**20% wall reduction** over the 19.3 s baseline (26% reduction on
the chunk-total alone). Vocoder cost essentially disappears inside
the LM time.

If voc were larger than LM (longer chunks → fewer LM steps → less
benefit; OR shorter chunks where the vocoder constant dominates →
more benefit), the savings would shift. For our typical chunk size
the LM is comfortably the bottleneck.

## Run 2 — 2026-05-17 18:50 — Pipelined LM/vocoder: real measurement

New `Chatterbox.Base.ChunkedSynthesizer` runs the LM on a producer
`Task.Run` thread, vocoder on the main thread, with a bounded
`Channel<(idx, speechTokens, lmMs, lmSteps)>` of capacity 1 between
them. Different `InferenceSession` objects → no per-session locking
needed (verified earlier in the SessionLoadObserver discussion).

```
$ dotnet run --project tests/ChatterboxSmoke -- \
    --onnx-dir /tmp/cb_dyn5 --voice /home/chris/Downloads/VCTK_p303.wav \
    --bench-chunks 8 --pipelined --ep cuda

  chunk 1/8: LM 1520 ms (174 steps), voc 779 ms, audio 6.92s
  chunk 2/8: LM 1668 ms (174 steps), voc 589 ms, audio 6.92s
  chunk 3/8: LM 1681 ms (174 steps), voc 590 ms, audio 6.92s
  chunk 4/8: LM 1662 ms (174 steps), voc 592 ms, audio 6.92s
  chunk 5/8: LM 1669 ms (174 steps), voc 591 ms, audio 6.92s
  chunk 6/8: LM 1654 ms (174 steps), voc 592 ms, audio 6.92s
  chunk 7/8: LM 1670 ms (174 steps), voc 591 ms, audio 6.92s
  chunk 8/8: LM 1662 ms (174 steps), voc 536 ms, audio 6.92s
Bench: 8 chunks (pipelined), LM avg 1648 ms, voc avg 608 ms,
  per-chunk-sum-avg 2256 ms, chunks-wall 13.7s [total wall 18.2s]
```

### Versus Run 1 baseline

| Metric | Serial (Run 1) | Pipelined (Run 2) | Delta |
|---|---|---|---|
| LM avg | 1284 ms | 1648 ms | **+364 ms (+28% slower)** |
| Vocoder avg | 557 ms | 608 ms | +51 ms (+9% slower) |
| Per-chunk sum | 1840 ms | 2256 ms | +416 ms (+23% slower) |
| Chunks-wall (8 chunks) | 14.7 s | **13.7 s** | **−1.0 s (−7%)** |
| Total wall | 19.3 s | 18.2 s | −1.1 s (−6%) |

**This is much less than the 25-26% theoretical reduction.** Per-call
operations are 9–28% slower because LM and vocoder now contend for
GPU resources. The pipelining still nets a positive wall delta — but
only ~1/3 of what the no-contention model predicted.

### Why per-call slows down

The LM (memory-bandwidth-bound, 2 GB of weights streaming through PCIe
per step) and the vocoder (compute-heavy, FFT-like ops on the SMs)
share one CUDA context. Concurrent kernel launches go onto separate
streams, but at the hardware level they contend for:

- **Memory bandwidth**: LM step is dominated by KV-cache reads + matmul
  weight reads. Vocoder's STFT/conv ops also pull weights. Adding the
  two together pushes the 3090's ~900 GB/s DRAM bandwidth into
  contention.
- **SM occupancy**: large matmuls (LM) and convolutions (vocoder) both
  want most of the SMs. Co-resident kernels get fewer SMs each.

The result: the LM time grows by ~28% (its weight reads are now
sharing bandwidth with the vocoder's outputs); the vocoder time grows
by ~9% (it was already lighter and proportionally less affected).

### Should we ship Run 2's implementation?

Two reasons in favor, both real:

1. **GUI responsiveness**: in the eventual Avalonia app, the LM
   thread no longer blocks the UI thread. Even a 7% throughput win
   pays off as a much-larger perceived-latency win when the user
   sees the first chunk's audio start playing while the second
   chunk's LM is still rolling.
2. **Net wall is positive**: 6–7% reduction is real, not lost in noise
   (8 chunks × repeatable per-chunk numbers).

One reason against:

- **The throughput ceiling is unaffected**: contention means we
  can't stack this with future optimizations cleanly. Stage 2
  (batched LM) and Stage 3 (fp16) would have to redo their own
  contention modeling — the batched LM occupies the GPU more fully,
  making vocoder overlap *even worse* than what we measured.

### Conclusion + next step

Ship the pipelining for the GUI-responsiveness reason; the throughput
delta is honest signal (small but real). DON'T treat Run 2's pattern
as the load-bearing perf strategy.

The real next probe should be **batched LM (Stage 2)** — where the
export already supports `B>1` dynamic axes (verified above), and the
expected payoff is 2-3× LM throughput. Pipelining a batched-LM with
serial vocoder may give the cleanest combined win because each LM
call now does more work per GPU-context-acquisition.

Other angle worth a one-day probe before Stage 2: explicit
`OrtCUDAStream` priority hints. ORT supports passing a stream handle
to `RunWithBinding`; the LM and vocoder on different priority streams
might let CUDA's scheduler interleave them less destructively. Cheap
to try.


## Run 3 — 2026-05-17 19:30 — Batched-LM feasibility probe

Goal: confirm the LM ONNX honors B>1 at runtime (its `dynamic_axes`
in `export_chatterbox_to_onnx.py:625-633` declare `batch_size` on
every input/output, but we've never exercised it), then measure the
amortization factor `wall(B=N) / wall(B=1)` — anything below N means
batching wins.

New `--bench-batched-lm B` flag in `ChatterboxSmoke` bypasses voice +
vocoder; loads only `embed_tokens + language_model`, replicates the
same prompt across the B dimension (Ezreal sentence, S=82), runs
prefill + 10 autoregressive steps, reports per-step ms.

```
=== B=1 ===  step iter avg: 14.0 ms/call  (14.0 ms/batch-elem)
=== B=2 ===  step iter avg: 21.8 ms/call  (10.9 ms/batch-elem)
=== B=4 ===  step iter avg: 31.3 ms/call  ( 7.8 ms/batch-elem)
```

Prefill (S=82) numbers:

| B | wall | per-batch-elem |
|---|---|---|
| 1 | 31 ms | 31 ms |
| 2 | 43 ms | 21.5 ms |
| 4 | 69 ms | 17.3 ms |

### Amortization

Amortization factor = `wall(B=N) / wall(B=1)`. Theoretical perfect = 1.0
(infinite parallelism); theoretical worst = N (no amortization).

| Stage | wall ratio | % of theoretical wall savings captured |
|---|---|---|
| Prefill B=2 | 1.39× | 61% |
| Step iter B=2 | 1.56× | 44% |
| Prefill B=4 | 2.22× | 59% |
| Step iter B=4 | 2.24× | 59% |

We're capturing 44–61% of the theoretical batching wall savings. The
gap is exactly the bandwidth-bound story: at fp32, each step reads
the full 2 GB of LM weights through PCIe regardless of how many batch
elements share them. Batching amortizes kernel launch overhead but
not the weight read.

### Implication for long-form throughput

Currently (serial B=1): per chunk = LM 1.25 s + vocoder 0.55 s = 1.80 s.
With B=4 batched LM (assuming the per-step ratio holds at full
rollout): LM 1.25 × 2.24/4 = 0.70 s per chunk, vocoder still 0.55 s
serial (or also batched). 4-chunk wall:

| | Serial today | B=4 batched LM | Reduction |
|---|---|---|---|
| LM | 4 × 1.25 = 5.00 s | 1 × 2.80 s | 44% |
| Vocoder | 4 × 0.55 = 2.20 s | 4 × 0.55 = 2.20 s (unchanged) | 0% |
| Total | 7.20 s | 5.00 s | **31%** |

**31% wall reduction projected** at fp32 — vs the ~7% pipelining
yielded. The gap to "theoretical perfect batching" (which would give
~58%) is bandwidth, not compute.

### And this is the lever fp16 unlocks

The amortization factor is bandwidth-ceiling-limited. Halving the LM
weight size with fp16 quantization (Stage 3 in the original analysis)
halves the bandwidth ceiling — at which point the B=4 amortization
should drop from 2.24× toward something like 1.3–1.5×, putting
batched throughput into the 2–3× per-chunk territory.

So the batched-LM work today is architectural: build the C# plumbing
+ orchestration so fp16 lands into a ready-shaped infrastructure
later. The fp32 throughput win is real (31%) but not the headline —
the headline is the structure.

### Next step in this PR series

Implement `AcousticLM.GenerateBatch` (same-length inputs MVP, run to
shared max_steps; per-element early stop is Phase 2). Then a Run 4
that measures full end-to-end batched rollout, including the
realistic STOP_SPEECH-driven variable length per element (which will
cost some efficiency vs the probe's uniform-length steady state).


## Run 4 — 2026-05-17 19:55 — End-to-end batched LM rollout (8 chunks, B=4)

`AcousticLM.GenerateBatch` (same-length MVP, basic Run path, no
IoBinding) wired through ChatterboxSmoke as `--lm-batch B`. Runs
benchChunks chunks in `ceil(benchChunks/B)` groups; each group is one
`GenerateBatch` plus B serial vocoder calls.

```
=== --bench-chunks 8 --lm-batch 4 ===
    batch LM call: 13451 ms total for B=4   (group 1)
    batch LM call: 12891 ms total for B=4   (group 2)
Bench: 8 chunks (lm-batch=4), LM avg 3292 ms, voc avg 558 ms,
  per-chunk-sum-avg 3850 ms, chunks-wall 30.8s [total wall 35.2s]
```

### Surprise vs Run 3's projection

Run 3 predicted 31% wall reduction at B=4. Reality at B=4 in the full
rollout is **worse than serial-IoBinding** in absolute terms:

| Config | LM per chunk | Chunks-wall (8) |
|---|---|---|
| Serial B=1 + IoBinding (Run 1 baseline) | 1284 ms | 14.7 s |
| Batched B=4 + basic Run (this Run 4) | 3292 ms ("share") | 30.8 s |

**Batched is 2.4× SLOWER than serial-IoBinding in absolute time.** The
per-step LM time at B=4 in the full 174-step rollout is ~75 ms vs the
probe's 7.8 ms/batch-elem — what gives?

### Apples-to-apples: basic Run for both

The IoBinding path keeps KV-cache outputs GPU-resident; the basic Run
path host-roundtrips them every step. At step 174 the past_kv per layer
is `[B=4, 16, 255, 64]` × 4 bytes × 60 (30 layers × K+V) = ~250 MB
**copied out of GPU memory per step**. The probe only ran 10 steps;
the full rollout amplifies the host-roundtrip cost.

Re-running serial with `--no-io-binding` for the fair comparison:

```
=== --bench-chunks 8 --no-io-binding ===
  chunk 1/8: LM 5400 ms (174 steps), voc 715 ms, audio 6.92s
  ...
Bench: 8 chunks (serial), LM avg 5219 ms, voc avg 557 ms,
  per-chunk-sum-avg 5776 ms, chunks-wall 46.2s
```

Now we can compare like-for-like:

| Mode (basic Run for both) | LM per chunk | Chunks-wall (8) | vs serial |
|---|---|---|---|
| Serial B=1 basic | 5219 ms | 46.2 s | (baseline) |
| Batched B=4 basic | 3292 ms ("share") | 30.8 s | **−33%** |

The 33% wall reduction matches Run 3's projection. Batching works
exactly as the probe said it would. Just not at IoBinding's absolute
level.

### The real ceiling: IoBinding × batching, both

The two perf wins are independent and stack:

- IoBinding (Run 1 vs basic): 5219 → 1284 ms/chunk = **4.1× LM speedup**
- Batching (basic B=1 vs B=4): 5219 → 3292 ms/chunk = **1.6× LM speedup**

Combined ceiling (assuming the per-step amortization factor of 2.24×
from Run 3 holds when KV stays GPU-resident):
  Serial IoBinding 1284 ms/chunk
  → Batched IoBinding B=4: 1284 × (2.24/4) ≈ 720 ms/chunk
  → ~44% LM reduction per chunk vs serial-IoBinding
  → ~30-35% wall reduction (vocoder share stays constant)

That's the architectural ceiling at fp32. Real IoBinding-for-batch
work is non-trivial:

- `RunLmLoopIoBinding` hardcodes shape `[1, KvHeads, T_kv, HeadDim]`
  for the per-layer KV bindings. Needs parameterization on B.
- Per-element argmax extracts a row from `[B, S, vocab]` GPU memory;
  cheaper if we move argmax to GPU (separate op) or accept one
  `[B, vocab]` last-row CPU copy per step.
- Empty initial KV tensors `[B, KvHeads, 0, HeadDim]` need batch-
  aware allocation.

None of these are blocking; just plumbing work.

### What's shipping in this PR

The `GenerateBatch` MVP itself — even at basic-Run cost — is the
**architectural commit**:

- `AcousticLM.GenerateBatch(condEmbs, textTokenIdsPerChunk, ...)`
  returns `BatchedAcousticLmResult` (per-element results + actual
  steps run).
- Same-length input constraint (Phase 1); STOP_SPEECH triggers
  per-element done tracking but the inner loop still steps in
  lockstep until ALL done or maxSteps. Per-element batch shrinking
  is a follow-up.
- Basic Run path, no IoBinding (Phase 1).

The fp32 throughput is lateral vs serial-IoBinding (slower, in fact),
but the C# shape is now in place for two compounding wins to land
into:

1. IoBinding-for-batch (estimated 30-35% wall reduction at fp32)
2. fp16 LM quantization (halves bandwidth ceiling — combined with
   batched IoBinding, expected 50-60% wall reduction territory)

Neither of these requires re-touching `AcousticLM`'s public surface.

### Honest framing

Pure throughput today: shipping the batched path WITHOUT IoBinding
would be a regression vs serial-IoBinding. Ship it behind the
`--lm-batch` flag (opt-in benchmark/measurement only) until IoBinding-
for-batch lands. The API surface (`GenerateBatch` and
`BatchedAcousticLmResult`) is the durable contract.


## Run 5 — 2026-05-17 20:15 — IoBinding-for-batch: the architecture pays off

Generalized the IoBinding LM-loop path to accept B>1 by parameterizing
the per-layer KV bindings on the leading batch dim. New
`RunBatchedLmLoopIoBinding` is a sibling to the single-batch
`RunLmLoopIoBinding`, sharing the same OrtValue-chaining pattern. The
public `GenerateBatch` gains `useIoBinding` (auto-detected from
`_effectiveCuda`, matching `Generate`); it dispatches between the
basic and IoBinding loop bodies.

Common helpers extracted (`ArgmaxBatchedLastRow`, `AllDone`,
`EmbedBatch`, `GrowBatchedMask`) so both loop bodies share the
correctness-critical bits.

### Batch scaling (8 chunks per run, RTX 3090, CUDA, warm cache)

| Config | LM ms/chunk | LM call wall | Chunks-wall | Total wall | Reduction vs serial-IoB |
|---|---|---|---|---|---|
| Serial IoBinding (Run 1) | 1284 | n/a | 14.7 s | 19.3 s | (baseline) |
| Batched IoB **B=2** | 748 | 1670→1444 ms/group | 10.5 s | 14.9 s | **−29%** |
| Batched IoB **B=4** | 409 | 1783→1494 ms/group | 7.8 s | 12.4 s | **−47%** |
| Batched IoB **B=8** | 236 | 1894 ms (1 group) | 6.4 s | 10.8 s | **−57%** |
| Batched IoB **B=16** (16 chunks) | 152 | 2447 ms (1 group) | 11.2 s | 15.7 s | n/a (different N) |

Amortization (one batched LM call vs B× serial calls):

| B | LM ms/chunk | × B | / single (1284) | Effective LM throughput vs serial |
|---|---|---|---|---|
| 1 | 1284 | 1284 | 1.00× | 1.0× |
| 2 | 748 | 1496 | 1.17× | 1.72× |
| 4 | 409 | 1636 | 1.27× | 3.14× |
| 8 | 236 | 1888 | 1.47× | 5.44× |
| 16 | 152 | 2432 | 1.89× | 8.45× |

### Why this is so much better than Run 3's projection

Run 3's probe predicted 31% chunks-wall reduction at B=4. Reality
delivered **47%**. The probe undershot because it included KV-extract
overhead in its step timing (basic Run path); with IoBinding keeping
KV GPU-resident, the per-step amortization is much cleaner.

The Run 4 finding ("batched-basic is slower than serial-IoBinding")
isn't wrong — it's just the wrong-combination measurement. IoBinding
and batching are independent multipliers and stack better than either
alone.

### Vocoder is now the bottleneck

At B≥8 the vocoder dominates the per-chunk wall:

| | LM ms/chunk | Voc ms/chunk |
|---|---|---|
| B=4 | 409 | 560 |
| B=8 | 236 | 559 |
| B=16 | 152 | 549 |

At B=4 the LM is still 73% of per-chunk time; at B=8 the vocoder is
70% of per-chunk time; at B=16 the vocoder is 78%.

What this means for the next iteration:

1. **Vocoder batching probe**: does the merged `conditional_decoder_loop.onnx`
   accept B>1? The export's dynamic_axes for that graph DO declare
   `batch_size` on inputs/outputs, but we've never exercised it
   (vocoder is single-batch in `Vocoder.cs`).
2. **Or vocoder pipelining**: bring back Run 2's LM/Vocoder overlap
   pattern, applied across batches. Vocoder for batch N runs in
   parallel with LM for batch N+1.

Either lifts the chunk-wall further. Without one of them, going past
B=8 doesn't help much in this run (B=8 chunks-wall is 6.4s; B=16 with
16 chunks is 11.2s — but that's more chunks total, normalized to
chunks-wall/chunk it's only marginally different: 0.80 s/chunk vs
0.70 s/chunk).

### Memory floor

KV cache at B=16, step 174, fp32:
  B × KvHeads × T_kv × HeadDim × 4 bytes × 60 (layers × K+V)
  = 16 × 16 × 255 × 64 × 4 × 60 = 1.0 GB

Plus 2 GB LM weights. 3090 has 24 GB. Plenty of headroom; could
explore B=32 or even B=64 to push the LM further before memory bites.

But — and this is the user's framing — pushing batched LM higher at
fp32 still hits the bandwidth ceiling. fp16 halves both the weight
read AND the KV cache, allowing both bigger batches and faster
per-step (the 2.24× amortization at B=4 should drop to ~1.3× and
chunks-wall should fall to a smaller multiple). The architectural
work in Run 5 is what fp16 plugs into.

### Net delivered in this iteration

| Run | Lever | Wall reduction vs Run 1 |
|---|---|---|
| Run 1 | (serial-IoBinding baseline) | (baseline) |
| Run 2 | LM/Voc pipelining | −7% (lateral) |
| Run 3 | (probe only) | (probe only) |
| Run 4 | Batched basic-Run | NEGATIVE (architectural commit) |
| **Run 5** | **Batched IoBinding B=4** | **−36% (total wall), −47% (chunks-wall)** |
| **Run 5** | **Batched IoBinding B=8** | **−44% (total wall), −57% (chunks-wall)** |

Run 5 is the headline. Per the user's framing it's also where the
fp16 lever will land — the C# shape is in place; the inference
quality story changes when weights halve.


## Run 6 — 2026-05-17 21:30 — Vocoder batching: split-graph fallback

After Run 5 made vocoder the new bottleneck, probe whether the
merged `conditional_decoder_loop.onnx` accepts B>1.

### Run 6a — Probe results

```
Merged conditional_decoder_loop.onnx:
  B=1: 843 ms  (works, single-Run on GPU-side Loop)
  B=2: FAIL  cfm__/estimator/Concat_2: dim 4 vs expected 2
  B=4: FAIL  ...dim 8 vs expected 2
  B=8: FAIL  ...dim 16 vs expected 2

Split graphs (flow_encoder + cfm_estimator + mel2wav):
  flow_encoder B=2: 24 ms
  flow_encoder B=4: 38 ms
  cfm_estimator B=4 (CFG-doubled=8): 173 ms (single step; full CFM = 10 steps)
  mel2wav B=2: 318 ms
  mel2wav B=4: 436 ms
```

The merged graph baked CFG-doubling at trace time — the inner Loop
body's Concat ops have hardcoded dim 2 from the trace. The split
graphs accept B>1 cleanly because their `dynamic_axes` were never
collapsed by a trace-time fold.

**Two paths forward** (decided to go with A; B is a future cleanup):

- **Path A (no export changes)**: use split-graph orchestration in C#
  for B>1 vocoder. Trade GPU-side Loop efficiency for C#-orchestrated
  B>1 batching.
- **Path B (real export refinement)**: rewrite
  `scripts/_export_utils/merge_cond_decoder_loop.py` to make
  CFG-doubling dynamic on input batch. Maybe a day's work +
  parity re-verification. Unlocks single-Run batched-merged vocoder.

### Run 6b — `Vocoder.SynthesizeBatch`

New method always uses the split orchestration path (B>1 requires
split graphs anyway). Constructor change: when `Mode == Merged`,
also load split graphs if present so `SynthesizeBatch` has them.
Added `Vocoder.SupportsBatched` property to surface the capability.

VRAM cost (answering the question explicitly): loading both merged
and split graphs doubles the cond-decoder weight residency. On disk:
~576 MB merged + ~500-600 MB split = ~1.1 GB additional. The 3090
with 24 GB has headroom; a 12 GB card with the 2 GB LM resident
would be tighter. If it bites, gate the split-load on a
`loadSplitForBatch` constructor flag.

One real bug surfaced: `flow_encoder.onnx`'s `z` output stays at
`[1, MelBins, T_mel]` even when input batch is B>1. The pinned
`rand_noise` patch we added in Run 11 of
`docs/chatterbox_investigation.md` for parity reproducibility made
`z` look graph-constant to ORT. Fix in `SynthesizeBatch`: detect
B=1 output and replicate B times in C#. Per-batch-element noise is
intentionally identical (same seed); if we ever want per-element
noise variation, that's a separate export-patch change.

### Run 6c — End-to-end batched LM + batched vocoder

```
=== --bench-chunks 8 --lm-batch 4 --voc-batch 4 ===
Bench: 8 chunks (lm-batch=4+voc-batch=4), LM avg 425 ms,
  voc avg 541 ms, per-chunk-sum-avg 966 ms,
  chunks-wall 7.7s [total wall 13.4s]

=== --bench-chunks 8 --lm-batch 8 --voc-batch 8 ===
Bench: 8 chunks (lm-batch=8+voc-batch=8), LM avg 242 ms,
  voc avg 514 ms, per-chunk-sum-avg 756 ms,
  chunks-wall 6.1s [total wall 11.8s]
```

Headline comparison:

| Config | LM ms/chunk | Voc ms/chunk | Chunks-wall | vs serial baseline |
|---|---|---|---|---|
| Serial IoBinding (Run 1) | 1284 | 557 (merged) | 14.7 s | (baseline) |
| Run 5: lm-batch=4 | 409 | 560 (merged) | 7.8 s | −47% |
| Run 5: lm-batch=8 | 236 | 559 (merged) | 6.4 s | −57% |
| Run 6c: lm-batch=4 + voc-batch=4 | 425 | 541 (split) | 7.7 s | −48% |
| Run 6c: lm-batch=8 + voc-batch=8 | 242 | 514 (split) | 6.1 s | −59% |

Vocoder batching saves **3-5% additional wall** when LM is already
batched. Less than hoped — the C#-orchestrated CFM solve adds 10
host-roundtrips per chunk (one per CFM step) that the merged
graph's GPU-side Loop avoided. So the per-chunk voc-share goes
from 557 (merged, no contention) → 514 (split B=8) = 8% reduction,
not the 30-40% the per-call probe suggested at face value.

### What this tells us about Path B (export refinement)

Path B isn't a free 3× win — even the split-graph batching at B=8
amortizes only modestly. The MERGED graph's superiority is the
GPU-side CFM Loop, not the batch dimension. Path B would let us
keep that GPU-side Loop while ALSO supporting B>1 — the combination
is what would actually pay off. Estimated win for Path B at B=8:
voc/chunk would be ~150-200 ms (rough projection from "what if
GPU-side Loop's amortization at B=8 looked like the merged graph's
B=1 timing × (B=8 amortization factor of ~1.5)").

So Path B is worth doing eventually, but not urgent. Logging it as
a follow-up issue.

### Architectural state at end of Run 6

The C# pipeline now supports B>1 end-to-end. Both `AcousticLM` and
`Vocoder` have `*Batch` methods with consistent same-length-MVP
constraints. ChatterboxSmoke exposes `--lm-batch` and `--voc-batch`
flags. fp16 quantization (when it arrives) plugs into this surface
without further architectural changes.

Cumulative wins:

| Run | What | Chunks-wall (8) | vs Run 1 |
|---|---|---|---|
| Run 1 | Serial baseline | 14.7 s | (baseline) |
| Run 5 | + Batched IoBinding LM (B=8) | 6.4 s | −57% |
| Run 6 | + Batched split vocoder (B=8) | 6.1 s | −59% |

Run 7 (vocoder pipelining: LM(group N+1) overlapped with voc(N)) is
the next iteration target. The contention story should be friendlier
now than in Run 2 — at lm-batch=8 the LM is small enough (242 ms)
that overlapping the 514 ms voc shouldn't starve it.


## Run 6 sidebar — Vocoder export investigation

Question (from the user, mid-Run-7): is Path B (rewriting the merge
script to support batched-merged) worth doing? Or is there no real
headroom to recover even if we fix it?

### What's hardcoded in the merge script

`scripts/_export_utils/merge_cond_decoder_loop.py` was written
assuming B=1 input. Three places hardcode the CFG-doubling shape:

1. **Body input/output shape**: lines 260-269 declare
   `x_carried [1, MEL_BINS, "T_mel"]` and `x_new [1, MEL_BINS, "T_mel"]`.
   For batched, these need `["batch_size", MEL_BINS, "T_mel"]`.

2. **t_in construction inside the body**: line 207 does
   `Concat([t_now_1d, t_now_1d], axis=0)` producing shape `(2,)`.
   At B>1 input, cfm_estimator expects `t_in` shape `(2*B,)` to
   match the CFG-doubled batch. The Concat needs to become a
   `Tile(t_now_1d, [2*B])` where 2*B is computed dynamically from
   the runtime x_carried shape:
   ```
   shape = Shape(x_carried)                # (3,) int64
   batch = Slice(shape, [0], [1], [0])     # (1,) = [B]
   two_b = Mul(batch, [2])                 # (1,) = [2*B]
   t_in  = Tile(t_now_1d, two_b)           # (2*B,)
   ```

3. **x_in CFG-double**: line 209 `Concat([x_carried, x_carried], axis=0)`
   correctly produces (2*B, ...) at any B (axis-0 concat scales) — no
   change needed.

The outer-scope CFG-doublers (lines 344-376) for mu/cond/spks/mask
all use `ConstantOfShape(Shape(mu))` etc. which IS already dynamic
on B — no change needed there.

**Net change for Path B**: 4 nodes need replacement (the 2 fixed
`[1, ...]` shape declarations + the t_in Tile-with-dynamic-2B). Half a
day plus parity re-verification.

### Headroom estimate

What would batched-merged actually deliver vs batched-split?

| | Measured | Source |
|---|---|---|
| Merged B=1 per chunk | 843 ms | Run 6a probe |
| Split B=4 per chunk | 551 ms | Run 6a probe (38+1730+436)/4 |
| Split B=8 per chunk | 514 ms | Run 6c measured |
| **Hypothetical merged B=4** | **~500 ms** | flow 38 + 10 CFM in GPU Loop ~1500 + m2w 436 / 4 |
| **Hypothetical merged B=8** | **~480 ms** | similar arithmetic at B=8 |

The savings vs split come from removing the 10 host-roundtrips
(allocating CFG-doubled arrays in C# per CFM step). Each roundtrip
is maybe 1-3 ms of C# allocation + ORT call setup. For B=8: 10 × ~3
ms = ~30 ms saved per chunk-group, spread across 8 chunks = ~4 ms/chunk.

**Headroom for Path B at fp32**: roughly **5-8% of vocoder time**, or
**~3-5% of total wall time** at our current best configuration
(Run 6c: 6.1 s for 8 chunks at lm-batch=8 + voc-batch=8).

In absolute terms: maybe shave 200-300 ms off the 6.1 s wall. Real
but small.

### Why the headroom is small at fp32

The vocoder is dominated by the actual CFM compute (8 attention
blocks × 10 Euler steps), not the orchestration overhead. The
"10 host-roundtrips" overhead the split path adds is small relative
to the per-step kernel time. Whether the Loop runs on GPU or C#
orchestrates it, the same 10 cfm_estimator computations happen.

### When Path B would matter more

- **fp16 quantization**: vocoder weights halve, per-step CFM time
  drops. The 30 ms of orchestration overhead becomes proportionally
  bigger. Maybe 10-15% headroom recovery at that point.
- **Streaming-within-chunk**: a merged-batched graph keeps the CFM
  solve as one Run, which is a cleaner unit for streaming
  pre-emption / interruption (the C#-orchestrated split path needs
  to interleave per-step checks).
- **Code surface reduction**: SynthesizeBatch is ~150 LOC of CFG
  orchestration in C# that Path B would let us delete in favor of
  one ORT Run call.

### Decision

**Don't do Path B now.** Estimated 3-5% wall reduction at fp32 doesn't
justify the merge-script surgery + parity re-verification. The user
called this correctly — there's some headroom but not much, and
fp16 is the lever that would make it meaningful.

Logged as a follow-up in [issue TODO] alongside the fp16 quantization
work; the two should land together since the latter unlocks the
former's payoff.

### What this means for Run 7

Vocoder pipelining (Run 7, in progress) is still worth doing — it's
orthogonal to merged-vs-split and gives parallelism across groups.
The contention story from Run 2 is friendlier now (LM at lm-batch=8
is only 242 ms; voc is 514 ms — voc dominates, so overlapping with
LM is mostly free for the LM's perspective).


## Run 6 sidebar 2 — Duplicate-node hunt (user-prompted)

Question (from the user): is the export still carrying redundant
nodes we haven't eliminated? The Phase 1 70K→7K cfm_estimator win
came from killing the 10× CFM unroll — are there leftover patterns?

### Value-duplicate initializers

Hashed every initializer in the merged graph and grouped by
(shape, dtype, raw-data-hash):

```
Total initializers:     1545
Unique value-hashes:    1521
Value-duplicate groups: 15
Total waste:            220 bytes (in 575 MB of initializer data)
```

Every duplicate is a tiny shape-arithmetic constant (shape `(1,)` or
`()`, used in trace-time Shape→Gather chains). **0% headroom** here.

### onnxslim pass on the merged graph

Noticed the merged graph never gets the slim treatment — the export
runs `slim_and_externalize` BEFORE the merge step. Ran onnxslim
manually on the produced merged graph:

```
Outer nodes: 1680 → 1676  (−4)
Body nodes:  3008 → 3004  (−4)
File size:   548 MB → 548 MB
```

**−8 nodes out of 4688** (the 4 Identity nodes the merge script adds
at the body boundary plus 4 in outer scope, probably). Negligible
perf impact; not worth wiring an extra slim pass into the export.

### Node-count audit by graph

| Graph | Nodes | Notes |
|---|---|---|
| cfm_estimator standalone | 2988 | per-step, post-Phase-1 |
| Merged body | 3008 | cfm_estimator + 20 boundary ops |
| Merged outer | 1680 | flow_encoder (863) + mel2wav (802) + 15 outer-CFG ops |
| flow_encoder | 863 | dynamic-shape post-Run-10 |
| mel2wav | 802 | scatter_add ISTFT (Run 3 fix) |

The merge script added ~35 nodes total to the union of the three
slimmed sub-graphs. That's a clean stitch.

### Op-type top-5 in the merged body

```
MatMul:               448  (attention + FFN; load-bearing)
Add:                  421
Mul:                  323
Transpose:            310  (attention head reshapes; could in principle be fused)
Reshape:              226
```

The 310 Transposes are the only "huh" — that's a lot. They're mostly
attention-block head-dim reshapes baked in by upstream's
`AttnProcessor → MinimalAttnProcessor` swap and the diffusers
`BasicTransformerBlock` layout. Fusing them would require a custom
attention kernel (TensorRT, ORT custom op) — significant work for
probably <5% gain on attention compute. Not the next lever.

### Conclusion

The export pipeline already eliminated the heavy redundancy. What
remains is essentially compute-bound (matmul + attention) and
memory-bandwidth-bound (weight reads), not redundant work. **No
further dedup-style export optimization will move the needle at
fp32.** fp16 quantization remains the right next lever for the
weight-bandwidth ceiling; merged-graph batching (Path B from the
previous sidebar) remains low priority for the same reason.


## Run 7 — 2026-05-17 22:00 — Pipelined batched groups

`ChunkedSynthesizer.Synthesize` gains a `groupSize` parameter. When >1,
dispatches to new `SynthesizeInGroups` (Channel-based producer/consumer
pattern, same shape as the chunk-at-a-time path from Run 2, but at
the group level). ChatterboxSmoke's `--pipelined` + `--lm-batch B`
combination now routes through this path.

### Measurements (all RTX 3090, CUDA, warm cache)

| Config | Chunks | Groups | Chunks-wall | vs non-pipelined equivalent |
|---|---|---|---|---|
| **Run 7c** lm-batch=4 + pipelined | 8 | 2 | **7.5 s** | vs Run 6c lm-batch=4+voc-batch=4: 7.7 s (−0.2 s, −3%) |
| **Run 7a** lm-batch=8 + pipelined | 16 | 2 | **11.6 s** | vs serial-batched extrapolation 12.6 s (−1.0 s, −8%) |
| **Run 7b** lm-batch=4 + pipelined | 16 | 4 | **14.3 s** | vs serial-batched 16.5 s (−2.2 s, −13%) |

### Why the win is small

Same contention story as Run 2. The chunk timings show it directly —
for Run 7a (2 groups of B=8):

```
chunk 1/16 (group 1): LM-share 245 ms, voc-share 540 ms
chunk 8/16 (group 1): LM-share 245 ms, voc-share 540 ms
chunk 9/16 (group 2): LM-share 704 ms, voc-share 503 ms  ← +2.9× LM!
chunk 16/16 (group 2): LM-share 704 ms, voc-share 503 ms
```

Group 1's batched LM runs alone (no overlap) at 245 ms/chunk. Group
2's batched LM runs concurrent with group 1's vocoder, slows to
704 ms/chunk due to shared SM/bandwidth contention. The vocoder
itself is only slightly affected (540 → 503 ms — actually faster
since it's running in isolation toward the end).

Decomposed wall for Run 7a:
- LM1: 245 × 8 = 1960 ms (alone)
- max(LM2 contended, voc1): max(5632, 4320) = 5632 ms
- voc2 (alone): 503 × 8 = 4024 ms
- Total: 11,616 ms ✓ matches measured 11.6 s

The LM2 inflation (1960 → 5632 ms) cancels most of the overlap
benefit. We save vocoder time but spend it back on slower LM.

### Better at smaller batch sizes

Run 7b (lm-batch=4 → 4 groups) gets a larger % reduction (13%) than
Run 7a (lm-batch=8 → 2 groups, 8% reduction). More groups means more
overlap opportunities, even with contention. But the absolute wall
is worse (14.3 vs 11.6 s) because group=8 has higher batched LM
throughput per chunk.

### Cumulative perf summary (8 chunks, RTX 3090, CUDA)

| Run | Lever | Chunks-wall | vs Run 1 |
|---|---|---|---|
| Run 1 | Serial baseline (IoBinding) | 14.7 s | — |
| Run 2 | + LM/Vocoder pipelining (B=1) | 13.7 s | −7% |
| Run 5 | Batched IoBinding LM (B=4) | 7.8 s | −47% |
| Run 5 | Batched IoBinding LM (B=8) | 6.4 s | −57% |
| Run 6 | + Batched vocoder (B=4) | 7.7 s | −48% |
| Run 6 | + Batched vocoder (B=8) | 6.1 s | −59% |
| **Run 7c** | **+ Pipelined groups (B=4)** | **7.5 s** | **−49%** |

At 16 chunks (more pipelining headroom): Run 7a at 11.6 s vs baseline
extrapolated 29.4 s = **−60% wall reduction.**

### Architectural state at end of Run 7

C# pipeline now has every fp32 perf lever in place:

- **Serial / one-shot**: `AcousticLM.Generate` + `Vocoder.Synthesize`
- **Pipelined one-shot**: `ChunkedSynthesizer.Synthesize(groupSize=1)`
- **Batched**: `AcousticLM.GenerateBatch` + `Vocoder.SynthesizeBatch`
- **Pipelined batched groups (long-form audiobook target)**:
  `ChunkedSynthesizer.Synthesize(groupSize=B)`

ChatterboxSmoke exposes all four via `--bench-chunks`, `--pipelined`,
`--lm-batch`, `--voc-batch`. Future fp16 quantization plugs into
these surfaces without architectural change.

### Where the headroom is now

Roughly speaking:

| Lever | Remaining headroom (fp32) | Notes |
|---|---|---|
| Pipelining contention | 10-20% | Run 2/7 contention is GPU SM + DRAM bandwidth. CUDA stream priority hints might recover some. |
| Vocoder Path B (merged-batched) | 3-5% | Detailed in Run 6 sidebar. Half-day fix; defer. |
| Duplicate-node removal | 0% | Per Run 6 sidebar 2 — already done. |
| fp16 LM quantization | **2-4×** projected | Halves weight bandwidth ceiling. Separate workstream. |
| Custom attention fusion | 5-10% | 310 Transposes in the body could fuse with a custom kernel. Real work. |

The single biggest remaining lever is fp16. Everything we've built
in Runs 1-7 plugs into it; the perf iteration is at a natural
pause point.
