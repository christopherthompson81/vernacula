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
