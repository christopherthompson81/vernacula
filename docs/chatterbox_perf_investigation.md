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

## Run 8 — 2026-05-16 22:35 — LM quantization sweep (fp16 / int8-dynamic / int4)

User asked: "sweep through the LM down to int4." Goal: cash in the
fp16 lever called out at the end of Run 7 (projected 2-4× LM
throughput), and survey the broader ORT quantization toolbox while
the rest of the pipeline sits at a natural pause point.

### Strategies tried

| Strategy | Tool | Where it lives in the codebase |
|---|---|---|
| fp16 (keep_io_types=True) | `onnxruntime.transformers.float16.convert_float_to_float16` | `scripts/_export_utils/quantize_lm.py --mode fp16` |
| int8 dynamic (W8A8) | `onnxruntime.quantization.quantize_dynamic` | `--mode int8` |
| int4 RTN MatMulNBits (W4A16, accuracy_level=4) | `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer` | `--mode int4` |

ChatterboxSmoke got a `--lm-path <path>` flag so an alternate
`language_model.*.onnx` can be loaded without rebuilding the bundle.

Strategies surveyed but not run this round: AWQ / GPTQ (need a small
calibration set; deferred unless RTN int4 quality fails the listening
test); static W8A8 with calibration (same — int8 dynamic failed so
hard that static probably won't fix the kernel-selection issue, which
turned out to be the real problem).

### Disk footprint

| Variant | language_model.onnx + sidecar | vs fp32 |
|---|---|---|
| fp32 (original) | 1.9 GB + 600 MB ≈ 2.5 GB | 1.0× |
| fp16 | 0.8 MB + 1024 MB ≈ 1.0 GB | 0.4× |
| int4 | 0.9 MB + 320 MB ≈ 320 MB | 0.13× |
| int8 | 3.4 MB + 512 MB ≈ 515 MB | 0.21× |

### Parity vs fp32 (single-chunk, --io-binding, Ezreal sentence)

| Variant | step0 argmax | step1 argmax | Total steps | Audio length | Tokens diverge at |
|---|---|---|---|---|---|
| fp32 | 1708 | 1736 | 174 | 6.92 s | — (reference) |
| fp16 | 1708 ✓ | 1736 ✓ | 174 ✓ | 6.92 s ✓ | byte 1017 (~token 127) — drift accumulates but stays in-distribution |
| int4 | 1708 ✓ | 1736 ✓ | 176 | 7.00 s | very early, but argmax pattern preserved |
| int8 | 6453 ✗ | 5186 ✗ | 93 (early STOP) | 3.68 s ✗ | step 0 — quality collapse |

int8 dynamic is unusable. The other two preserve the speaker timbre
and content end-to-end on listening (informal check; subjective).

### Perf (RTX 3090, CUDA, warm ORT bake cache)

Single-chunk (B=1), `--io-binding`, 174-176 LM steps:

| Variant | LM total ms | ms/step | vs fp32 |
|---|---|---|---|
| fp32 | 1627 | 9.4 | — |
| fp16 | 1822 | 10.5 | +12% slower |
| int4 | 1432 | 8.1 | **−14% (faster)** |
| int8 | 3009 | 32.4 | +245% (CPU fallback, useless) |

Pipelined 8-chunk (groupSize=1):

| Variant | chunks-wall | vs fp32 |
|---|---|---|
| fp32 | 13.6 s | — |
| int4 | 12.2 s | **−10%** |

Batched 8-chunk (`--lm-batch 4`):

| Variant | chunks-wall | LM ms/chunk | vs fp32 |
|---|---|---|---|
| fp32 | 7.7 s | 403 | — |
| fp16 | 8.1 s | 444 | +5% slower |
| int4 | 10.5 s | 745 | +36% slower |

### What we learned

**fp16 with `keep_io_types=True` is a slight regression, not a win.**

The conversion adds 251 fp16-target Casts and 61 fp32-target Casts —
net +122 Cast nodes vs the fp32 graph (4919 → 5041 total). The
critical-path damage: the LM exposes 30 layers × 2 (key + value) past_kv
inputs, all declared fp32 by the export contract. With `keep_io_types=True`
the converter wraps each in an input-side Cast (fp32 → fp16) and an
output-side Cast (fp16 → fp32). With IoBinding chaining KV across 174
steps, that's ~120 Casts/step on the hot path. Whatever compute we save
in the layer body, we spend back at the I/O boundary.

The fix is true fp16 KV cache (`keep_io_types=False` or a custom
include-set that excludes only `inputs_embeds`/`attention_mask`/`logits`).
That requires the C# IoBinding code to allocate fp16 zero-tensors
for the initial empty past_kv inputs and accept fp16 logits/KV out
— a one-day workstream we deferred.

**int8 dynamic (W8A8) hits a CPU fallback on CUDA EP for this LM.**

3.4× slower at the kernel level, plus quality collapse (step-0
argmax 6453 vs reference 1708). The combo says ORT isn't fusing the
QuantizeLinear/MatMulInteger into a tensor-core kernel for our op
pattern — it's running int8 MatMul on CPU. The quality regression
is the additional per-tensor activation-scale issue you'd see even
on the working CUDA path. Static W8A8 wouldn't fix the kernel issue.
**Mark int8 dynamic dead for this graph.**

**int4 RTN MatMulNBits wins at B=1, loses at B>1.**

The `MatMulNBits` operator has a CUDA kernel optimized for the LLM
decoding case (B=1, T=1, weight-bound). At that operating point we
saw 14% per-step speedup. But the kernel doesn't have an efficient
batched path — `--lm-batch 4` runs 1.85× slower than fp32-B=4
(745 vs 403 ms/chunk). MatMulNBits dequantizes weights per call;
batching amortizes weight bandwidth in the fp32 path but the
dequant cost in the int4 path doesn't share across batch elements.

### Where this leaves long-form synthesis

The Run 7 best (fp32 + `--lm-batch 4` + pipelined groups) is still
the wall-clock champion at **7.7 s for 8 chunks**:

| Run | Best path | 8-chunk chunks-wall | vs Run 1 |
|---|---|---|---|
| Run 1 | Serial IoBinding (fp32, B=1) | 14.7 s | — |
| Run 5/6 | Batched LM+voc (fp32, B=8) | 6.1 s | −58% |
| Run 7c | + Pipelined groups (fp32, B=4) | 7.5 s | −49% |
| **Run 8 int4** | **Pipelined B=1 (int4)** | **12.2 s** | **−17%** |
| Run 8 fp16 | Pipelined B=4 (fp16) | 8.1 s | −45% |
| Run 8 int8 | — | — | unusable |

int4 at pipelined B=1 is the best Run 8 result but it's much worse
than fp32 batched. **None of the off-the-shelf quantization
strategies improved long-form-batched throughput.**

### Where headroom actually lives, post-sweep

| Lever | Projected win | Notes |
|---|---|---|
| True fp16 KV cache (keep_io_types=False + C# fp16 IoBinding) | 30-50% | The natural fp16 win we missed because of boundary Casts. Half-day to a day of C# IoBinding work in AcousticLM. |
| MatMulNBits with B>1-friendly kernel | unknown | Would need a custom ORT op or upstream PR. Not a 1-week project. |
| int4 batched via AWQ/GPTQ + custom kernels | uncertain | Real LLM-stack territory; probably not the right cost for this perf budget. |
| Vocoder Path B (merged-batched) | 3-5% | Still on the deferral list. |
| Custom attention fusion | 5-10% | 310 Transposes in the body could fuse with a custom kernel. Real work. |

The natural next step (if the user wants to keep pushing) is true
fp16 KV cache. The natural pause point is honestly here: we've
covered every off-the-shelf ORT quantization mode for this LM,
documented the quality + kernel-coverage caveats, and the int4 path
is at least a B=1-mode win to bank.

### Artifacts on disk (left intact for follow-up runs)

- `/tmp/cb_dyn5/language_model.fp16.onnx{,_data}` — 1.0 GB total
- `/tmp/cb_dyn5/language_model.int4.onnx{,_data}` — 320 MB total
- `/tmp/cb_dyn5/language_model.int8.onnx{,.data}` — 515 MB total (unusable; keep for forensics)
- `/tmp/cb_dyn5/language_model.{fp16,int4,int8}.opt.cuda.*.onnx{,_data}` — baked optimized graphs
- `/tmp/cb_diag_{fp32,fp16,int4,int8}/cs_tokens.bin` — per-variant token streams for parity diffs
- `/tmp/cb_{fp32,fp16,int4,int8}_*.wav` — audio outputs (subjective listening check)

## Run 9 — 2026-05-16 22:48 — True-fp16 KV cache (boundary-Cast hypothesis test)

Run 8a left a clean hypothesis: fp16's slight regression came from
boundary Casts on the 60 past_kv ports (30 layers × {key, value} ×
{in, out}). The fix was supposed to be true-fp16 I/O — drop
`keep_io_types`, allocate fp16 zero-tensors in C# IoBinding, let
the KV cache live as fp16 end-to-end. Projected: 30-50% LM
throughput win across all batch sizes.

### What we built

**Quantization side**: `quantize_lm.py` gained `--no-keep-io-types`
producing `language_model.fp16io.onnx` with fp16 `inputs_embeds`,
`past_key_values.*`, `present.*`, and `logits` (attention_mask
stays int64).

**C# side**: `AcousticLM` became dtype-aware:

- New `_lmFp16` field detected at construction from
  `_lm.InputMetadata["inputs_embeds"].ElementDataType`.
- Six new helpers: `MakeEmbedsOrtValue`, `MakeEmptyPastKvOrtValue`,
  `LogitsToFloatArray`, `MakeEmbedsNamedValue`,
  `MakePastKvNamedValue`, `PresentToFloatArray`/`PresentDims`.
- All four rollout loops (B=1 / B>1 × IoBinding / basic) thread the
  fp16 dtype through inputs_embeds + past_kv binding and the logits
  readback. Scratch arrays kept alive past `RunWithBinding` via
  explicit `GC.KeepAlive`.
- Pattern adapted from `VibeVoiceAsr.cs::_kvCacheIsFloat32`, which
  already handles the same fp16/fp32-KV switch for its decoder.

### Parity (single-chunk, --io-binding)

fp32 baseline vs true-fp16 LM:

| Metric | fp32 | fp16io |
|---|---|---|
| step0 argmax (pre-penalty) | 1708 | 1708 ✓ |
| step1 argmax (pre-penalty) | 1736 | 1736 ✓ |
| step0 first 10 logits | (reference) | bit-identical to Run 8a fp16 row, fp32 within 1 ULP |
| Total LM steps | 174 | 174 ✓ |
| Audio output length | 6.92 s | 6.92 s ✓ |

Identical to Run 8a's parity story — the dtype-aware C# binding
produces the same tokens as the keep_io_types=True version, just
without the boundary Casts. **The architecture is correct.**

### Perf (RTX 3090, CUDA, warm ORT bake)

| Path | fp32 | fp16io | Δ |
|---|---|---|---|
| Single-chunk B=1 | 1491 ms (8.6 ms/step) | 1713 ms (9.8 ms/step) | **+15% slower** |
| Pipelined B=1 (8 chunks) | 13.6 s wall | 13.9 s wall | +2% slower |
| Batched B=4 (8 chunks) | 7.7 s wall, 403 ms LM/chunk | 7.9 s wall, 427 ms LM/chunk | +6% slower |

**Same 5-15% regression we saw in Run 8a, despite the boundary
Casts being gone.** The hypothesis that boundary Casts were the
problem was wrong.

### What actually happened

I diffed the ORT-baked optimized graphs side-by-side. The fp32 and
fp16io graphs after ORT's CUDA optimization are **structurally
identical**:

```
fp32-opt:    2260 nodes, 272 MatMul, 35 Cast (all to fp16), 61 SimplifiedLayerNormalization, ...
fp16io-opt:  2260 nodes, 272 MatMul, 35 Cast (all to fp16), 61 SimplifiedLayerNormalization, ...
```

ORT applied the same fusions to both. The 35 fp16-target Casts in
**both** graphs say ORT is internally casting fp32 intermediates
to fp16 in 35 places — meaning even the "fp32" baseline does a fair
amount of fp16 compute. The MatMul kernels selected at this dim
(B=1 T=1, 1024 hidden, 64 head_dim) appear not to use Tensor
Cores at all — they're running cuBLAS SGEMM in fp32 in both cases,
just with different load patterns.

The slight regression in fp16io is the cost of:
- C# fp32→fp16 conversion on inputs_embeds (1024 floats/step)
- Cast nodes ORT inserts at fp16 input → fp32 compute boundary
- fp16 logits readback then host fp32 conversion (8194 floats/step)

None of these are big individually, but together they slightly
outweigh whatever bandwidth advantage true-fp16 KV gives.

### Why ORT isn't picking fp16 Tensor Core kernels here

Best guess (haven't verified with verbose kernel logs):
- LLM hidden 1024 / head_dim 64 may be below cuBLAS's heuristic
  threshold for switching to wmma/mma kernels
- B=1 single-token decode is memory-bound; the kernel choice would
  benefit from fp16 weights but not fp16 activations
- ORT's SimplifiedLayerNormalization fusion may force fp32 LN
  internals (the 35 fp16-target Casts)

The "30-50%" projection was wishful — based on what fp16 *should*
deliver on Ampere for transformer LMs at scale, not what ORT
actually delivers on a 30-layer / 1024-hidden Llama-style backbone
at B=1.

### Where this leaves the perf

Same place Run 7c left it. The best long-form path is still:

```
fp32 + lm-batch=4 + pipelined groups → 7.5-7.7 s for 8 chunks (−49%)
```

| Run | Best 8-chunk wall | vs Run 1 |
|---|---|---|
| Run 1 | 14.7 s | — |
| Run 7c | 7.5 s | −49% |
| Run 8 int4 (pipelined B=1) | 12.2 s | −17% |
| Run 9 fp16io (batched B=4) | 7.9 s | −46% |

**Net Run 9: zero perf win, architecturally cleaner code.** The C#
fp16/fp32 dtype-aware machinery is in place — if any future LM
variant comes with a kernel set that *does* benefit from fp16, we
just point `--lm-path` at it and the dtype-aware binding handles
the rest.

### Where the remaining levers actually are

The quantization sweep + fp16 IoBinding work covers ORT-native
quantization for this LM. Things on the table that haven't been
tried:

| Lever | Hypothesis | Cost | Risk of finding zero again |
|---|---|---|---|
| BFloat16 KV cache | bf16 has fp32 range so fewer LN-internals Casts; some ORT kernels are bf16-only-tuned | Same as Run 9 (just regenerate the model + rerun) | Medium — same kernel-selection ceiling may apply |
| ORT TransformerOptimizer's `optimize_by_fusion` with `model_type=gpt2` | Forces explicit MultiHeadAttention / Attention fusion nodes that ORT *does* have fp16 kernels for | Half-day; needs to verify the Llama-derived export matches the gpt2 patterns | Medium — Chatterbox isn't exact gpt2 |
| Onnxruntime-genai conversion | Different runtime that builds attention as a single op; known-fast fp16 kernels | Day+; involves swapping ORT for ORT-GenAI in C# binding | Low if it works at all; high if Chatterbox export doesn't fit GenAI's contract |
| Vocoder fp16 + Path B (merged-batched) | Vocoder is currently 50%+ of long-form wall in some configs; same fp16 + Run-6 deferred work | Day+ | Medium |
| Custom MultiHeadAttention CUDA kernel | The 310 Transposes in the merged body could fuse with a real attention kernel | Week+ | Low payoff probability without dedicated profiling |

The clearest signal: **off-the-shelf ORT quantization has been
exhausted for this graph on this hardware.** Further LM perf needs
either a different runtime (GenAI), a different graph (re-export
with fused-attention op type hints), or custom kernel work — all
substantially larger projects than Runs 1-9 were.

### Artifacts

- `/tmp/cb_dyn5/language_model.fp16io.onnx{,_data}` — 1.0 GB true-fp16 LM
- `/tmp/cb_diag_fp16io/cs_tokens.bin` — token stream for parity diff
- `/tmp/cb_fp16io*.wav` — audio outputs (parity confirmed)
- `quantize_lm.py --no-keep-io-types` — produces the true-fp16 graph
- `AcousticLM._lmFp16` + 6 helpers — dtype-aware IoBinding, ready for any future fp16/bf16 LM variant without further C# changes

## Run 10 — 2026-05-16 23:30 — Path B: merged-batched vocoder

Following the Run 6 sidebar — the merged `conditional_decoder_loop.onnx`
graph baked CFG-doubling at trace time and rejects B>1. PR #66 worked
around this by falling back to the split-graph orchestration at B>1
(measured Run 6c at +3-5% wall when LM was batched; net 8-chunk wall
went from 6.4 s → 6.1 s). Useful but small.

The merged graph's real advantage over split is the GPU-side CFM Loop
— 10 CFM iterations stay on the device; split round-trips through
host between every iter. Path B rewrites the merge script so the
merged graph accepts B>1, then we get the GPU-side Loop *with*
batching. Projected from Run 6c: per-chunk voc could drop from
~514 ms (split B=8) to ~150-200 ms (merged-batched B=8).

### What's hardcoded for B=1 in the current merge script

After reading `scripts/_export_utils/merge_cond_decoder_loop.py`:

| Site | Spec | Status at B>1 |
|---|---|---|
| outer CFG-doubling for mu / spks / cond | `Concat([x, ZerosLike(x)], axis=0)` via `Shape`+`ConstantOfShape` | already dynamic ✓ |
| outer mask CFG-doubling | `Concat([mel_mask, mel_mask], axis=0)` | already dynamic ✓ |
| body `cfm__x_in` | `Concat([x_carried, x_carried], axis=0)` | already dynamic ✓ |
| **body `cfm__t_in`** | `Concat([t_now_1d, t_now_1d], axis=0)` → fixed `(2,)` | **broken — needs `(2B,)`** |
| **body input shape decl** | `x_carried: [1, MEL_BINS, "T_mel"]` | **broken — needs `["batch_size", MEL_BINS, "T_mel"]`** |
| **body output shape decl** | `x_new: [1, MEL_BINS, "T_mel"]` | **broken — same** |
| **z (Loop v_initial)** | flow_encoder emits `z` at fixed `[1, M, T]` due to the pinned-rand_noise patch (Run 6c sidebar) | **broken — need Expand(z, mu_shape) before Loop** |

Three real fixes. The rest of the merge script already handles B>1
correctly because the original author used `Shape`+`ConstantOfShape`
instead of literal-2 leading dims (which is good defensive ONNX
authoring that's now paying off).

### Plan

1. Patch `merge_cond_decoder_loop.py`: t_in via `Tile`, body shape
   decls dynamic, z replication via `Expand(z, mu_shape)`.
2. Re-export `conditional_decoder_loop.onnx`.
3. Re-run `--bench-batched-voc B=4` probe to confirm B>1 acceptance.
4. Update `Vocoder.SynthesizeBatch` so it prefers the merged graph
   when it accepts B>1 (detect via session metadata or just try; the
   split-graph fallback stays as a safety net for older bundles).
5. Bench `--pipelined --lm-batch 4` vs Run 7c's 7.1 s headline.

### Implementation

Three fixes to `merge_cond_decoder_loop.py`:

1. **`cfm__t_in` dynamic sizing**. Old code: `Concat([t_now_1d, t_now_1d])`
   producing fixed `(2,)`. New code: read `B` from `Shape(x_carried)[0]`,
   compute `[2B]`, `Tile(t_now_1d, [2B])` → shape `(2B,)`. At B=1
   produces the same `(2,)` as before (parity preserved).
2. **Body input/output shape decls**. `[1, MEL_BINS, "T_mel"]` →
   `["batch_size", MEL_BINS, "T_mel"]` on `x_carried` and `x_new`.
3. **`z` replication outside the Loop**. `flow_encoder.onnx` emits z
   at fixed `[1, MelBins, T_mel]` (pinned-rand_noise patch, Run 6c).
   Added `Expand(z, mu_shape)` so the Loop's `v_initial` matches B.
   At B=1, Expand is a no-op identity.

The outer graph's CFG-doubling (mu, mask, spks, cond) already used
`Shape`+`ConstantOfShape` for dynamic zeros — no changes needed
there. Run 6's author had already done that piece defensively.

Plus a `metadata_props["supports_batched"] = "true"` flag on the
exported model so `Vocoder.cs` can detect Path B graphs vs older
fixed-B bundles.

C# side, `Vocoder` constructor reads the metadata. When
`_mergedSupportsBatched`:
- Don't load the split graphs (saves ~1.1 GB VRAM, the whole point
  of Path B in Run 6c sidebar).
- `SynthesizeBatch` routes to a new `RunMergedBatched` that packs all
  B chunks into one merged-graph `.Run`.

Older bundles without the metadata flag still work: they fall
through to the split-graph orchestration as before.

### Parity (B=1, new vs old merged graph, seeded random inputs)

```
max abs diff: 3.39e-05
mean abs diff: 4.57e-08
old shape=(1, 144000)   new shape=(1, 144000)
```

Within ORT non-determinism floor. End-to-end Chatterbox synthesis
through the new graph at B=1 produces 174 LM steps, 6.92 s audio —
bit-equivalent to Run 1's baseline.

### Perf (8 chunks, RTX 3090, CUDA, with `--io-binding`)

| Config | chunks-wall | voc/chunk |
|---|---|---|
| Run 7c: split-batched, pipelined groups (B=4) | 7.1 s | ~540 ms (first), 503 (steady) |
| **Run 10: merged-batched, pipelined groups (B=4)** | **7.0 s** | **~557 ms (first), 497 (steady)** |
| Run 10: merged-batched, lm-batch=4 + voc-batch=4 (no pipelining) | 7.4 s | 524 (first), 498 (steady) |
| Standalone vocoder probe B=4 (no LM contention) | n/a | **553 ms/batch-elem** |

Marginal 1% wall reduction (7.1 → 7.0 s). The dramatic Path B
projection in Run 6c ("voc/chunk ~150-200 ms at B=8") didn't
materialize: the batched CFM kernel costs scale almost linearly
with B at both split and merged paths, so per-batch-elem voc time
stays at ~500 ms regardless of which graph we use.

### Why the per-call advantage didn't translate

The standalone Python probe (no .NET, no IoBinding) showed
merged-batched B=4 at 1775 ms = ~444 ms/elem vs split-batched
~540 ms/elem — a real 18% kernel-level win. But in the end-to-end
.NET pipeline:

- ORT inserted 3 outer + 57 body Memcpy nodes on the CUDA EP. The
  warning text explicitly notes "may have negative impact on
  performance (including unable to run CUDA graph)". The body's
  per-iter Memcpys cost ~50-100 ms each — close to the per-CFM-iter
  budget itself.
- Under pipelined-groups, the wall is `max(LM-share, voc-share)`
  per group plus an LM-only first group and a voc-only tail.
  Vocoder isn't on the critical path most of the time; saving
  ~40 ms of voc-per-chunk hides behind the LM contention spike
  (`LM-share` inflates from 421 → 821 ms in group 2 due to SM
  contention with concurrent voc).

So the bottleneck moved from "vocoder graph cost" (where Path B
would help) to "LM SM contention" (where it doesn't).

### What we actually got

| Benefit | Measured | Value |
|---|---|---|
| Wall reduction | −1% (7.1 → 7.0 s) | Marginal |
| VRAM savings | −1.1 GB (split graphs no longer loaded when merged is batched-capable) | Real on 24 GB cards |
| Code path simplification | `Vocoder.SynthesizeBatch` is single-path for capable bundles; orchestration in ONNX, not C# | Real |
| Backward compatibility | Old bundles without metadata fall through to split orchestration | Preserved |

Net assessment: Path B is **worth landing** for the VRAM saving and
architectural simplification, even though the wall-perf win is
small on this hardware. On a memory-constrained box (or future
quantized-LM bundles where the LM also wants more VRAM headroom),
the 1.1 GB matters more.

### Where else could the merged-batched advantage land?

Two scenarios where the per-call advantage would translate:

1. **No pipelining contention**: a single-stream invocation
   (`--lm-batch 4` without `--pipelined`) sees voc-share 524 ms
   merged vs 540 ms split (first call, no contention) — a tighter
   3% win. With more aggressive batching (B=8+), the absolute saving
   grows linearly but stays in single-digit percent.

2. **Beefier GPU (H100, RTX 4090)** where SM contention isn't the
   bottleneck. The kernel-level 18% merged-vs-split advantage from
   the Python probe should show up as actual wall reduction when the
   LM and vocoder can truly run in parallel without contention.

Neither is testable on the 3090 directly. The Path B graph and
metadata flag are in place if/when that hardware shows up.

### Run 10 sidebar — Memcpy elimination, dead-end investigation

Followed up on the "57 body Memcpys block CUDA Graph capture" angle
from the Run 10 conclusion. **Conclusion: dead-end. CUDA Graph
cannot run on this merged graph at all, regardless of Memcpys.**

What we tried:

1. **`session.disable_cpu_ep_fallback=1`**. With CPU EP removed,
   session load failed: "This session contains graph nodes that are
   assigned to the default CPU EP, but fallback to CPU EP has been
   explicitly disabled by the user." Some ops (notably `mel2wav__/STFT`)
   have no CUDA kernel at all — the fallback isn't just heuristic,
   it's mandatory.

2. **`onnxslim` re-pass on the merged graph**. Outer 1681 → 1677
   nodes, body 3015 → 3007 nodes. Per-call timing unchanged (B=4
   stays at ~2000 ms). The 57 body Memcpys aren't redundant —
   they're real data shuffles for shape-derived computations.

3. **ORT-transformers `optimize_model(model_type=...)` attention
   fusion** on `cfm_estimator.onnx`. Tried `unet`, `mmdit`, `clip`,
   `vae`, `sam2`, `conformer`, `bart`. The U-net pattern reduced
   2988 → 2557 nodes (`LayerNormalization` fused to
   `SkipLayerNormalization`), but the shape-pattern op count
   (Shape/Gather/Unsqueeze/Slice/Sqrt/Div/Cast) was unchanged at
   643 across every variant, and **no `Attention` / `MultiHeadAttention`
   nodes were produced**. cfm_estimator's attention is a custom
   conv-like layout that none of ORT's fusion targets recognize.

4. **`enable_cuda_graph=1` provider option**, the actual reason
   CUDA Graph was on the table. Session refused to load:

   ```
   This session cannot use the graph capture feature as requested
   by the user as the model has control flow nodes which can't be
   supported by CUDAExecutionProvider
   ```

   The merged graph's `Loop` op is the blocker, not the Memcpys.
   The earlier "including unable to run CUDA graph" Memcpy warning
   was a red herring for this graph — CUDA Graph wasn't on the
   table regardless.

### What this means

The projected upside from eliminating the Memcpys (~50-100 ms/call
from CUDA Graph capture) was based on a misread of the warning.
With CUDA Graph fundamentally unavailable on a Loop-bearing
session, the Memcpy elimination would yield only the per-call
launch overhead saving — microseconds, not milliseconds.

To make Memcpy elimination actually pay off on this graph, we'd
need EITHER:

- Manually rewrite cfm_estimator's attention export to match an
  ORT-fusion pattern (real upstream change to the export pipeline,
  multi-day work).
- Unroll the CFM Loop into 10 sequential function calls (lose the
  GPU-side Loop, gain CUDA Graph eligibility — but lose the very
  thing that makes the merged graph faster than split).

Neither has a payoff that justifies the work for this branch. Path B
ships as documented: ~1% wall win, ~1.1 GB VRAM savings, simpler
single-vocoder-path architecture.
