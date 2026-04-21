# DiariZen Performance Progress

Updated: 2026-04-04

## Goal

Reduce DiariZen CPU wall time and real-time factor in `Parakeet.CLI` while preserving the parity result established during the earlier DiariZen correctness work.

Primary benchmark asset:

- Audio: `data/diarizen_parity/en-US_sample_01_first90.wav`
- Duration: `90s`
- Reference: `data/diarizen_parity/reference/first90/reference_segments.json`

## Current Summary

The most useful wins so far came from:

- adaptive CPU segmentation thread selection
- adaptive WeSpeaker CPU inference shape
- lower-overhead WeSpeaker fbank preparation
- streaming/queue-based embedding prep instead of retaining all chunk waveforms

The most important negative result so far:

- worker-layout experimentation alone is not enough anymore
- segmentation batching can be made to work technically, but it did not produce a meaningful CPU speedup in this repo state
- naive cached/streaming overlap-reuse experiments did not beat the current implementation

Current best guidance for desktop CPU inference:

- keep adaptive segmentation threading
- keep adaptive embedding thread/worker heuristics
- bias larger CPUs toward fewer, fatter embedding workers
- do not spend more time on tiny embedding-worker layouts, prep-worker decoupling, or weighted-frame cutoff tuning unless new profiling data points back there

End-state result for this performance pass:

- CPU RTF improved from roughly `1.75` in the older baseline regime to about `0.66`
- the last meaningful win came from segmentation decode/postprocess cleanup, not from more scheduling or batching changes
- this is a reasonable point to stop CPU tuning and move attention to post-processing polish

## Investigative Progress

### 1. Adaptive CPU segmentation threads were worth keeping

The segmentation path previously relied on more static tuning.

The current heuristic now targets roughly:

- `min(12, floor(cpu_count * 0.75))`

with small-memory guardrails to avoid oversubscription on weaker systems.

Key takeaway from local testing:

- on the current `16`-thread machine, the heuristic resolves to `12`
- that matched the earlier best known segmentation setting

Decision:

- keep this adaptive segmentation-thread logic
- treat RAM as a safety guardrail, not the primary tuning signal

Relevant code:

- [Config.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/Config.cs)
- [HardwareInfo.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/HardwareInfo.cs)

### 2. Adaptive WeSpeaker CPU inference shape was the main CPU win

The most important practical speedup came from changing the embedding CPU layout away from many tiny workers and toward fewer workers with higher intra-op thread counts.

Best region observed on the current machine:

- `embed_threads=8`
- `embed_workers=2`

That shape outperformed the older many-worker layout.

Measured effect from the earlier tuning session:

- old `1x14` style run: about `157s`
- improved `8x2` style run: about `91s`

Conclusion:

- fewer, fatter embedding workers were clearly better than many tiny workers
- CPU count predicted the useful shape much better than system RAM

Relevant code:

- [Config.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/Config.cs)
- [WeSpeakerEmbedder.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/WeSpeakerEmbedder.cs)
- [DiariZenDiarizer.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/DiariZenDiarizer.cs)

### 3. WeSpeaker prep and fbank work was reduced successfully

Several implementation changes helped lower the CPU cost of embedding preparation:

- chunk processing was streamed instead of fully materialized up front
- embedding prep moved through a bounded queue
- fbank computation was tightened up
- mel filters were made contiguous
- FFT power was computed once per frame
- the mel accumulation path became more SIMD-friendly
- flattening copies were reduced with `Buffer.BlockCopy`

These were worthwhile implementation improvements and remain part of the branch.

Relevant code:

- [WeSpeakerEmbedder.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/WeSpeakerEmbedder.cs)
- [DiariZenDiarizer.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/DiariZenDiarizer.cs)

### 4. Aggressive segmentation worker/session parallelism did not pay off

An earlier experiment increased outer segmentation parallelism by pushing more chunk-level work concurrently.

Result:

- CPU utilization rose sharply
- wall-clock time got worse rather than better

This matched the repo’s broader pattern:

- simply driving segmentation toward "all cores" increased scheduler and memory pressure
- more segmentation workers were not a reliable path to lower wall time

Decision:

- do not keep pushing segmentation worker/session parallelism
- on this machine, the useful regime mostly collapsed back to `1` worker anyway

### 5. More tiny embedding workers were also a dead end

Several embedding-worker layouts were tested.

The key negative result:

- `1x14` was worse than `1x1`

The best region remained:

- fewer workers
- higher intra-op thread counts per worker

Decision:

- stop spending time on "more tiny embedding workers"
- keep the adaptive heuristic biased toward the `8x2` style on larger CPUs

### 6. Separate embedding prep-worker scaling was not worth keeping

We also tested decoupling prep-worker count from inference-worker count.

Representative reruns:

- forced `prep_workers=2`: about `95.5s`
- forced `prep_workers=2`: about `99.4s`
- current split/default: about `93.2s`
- earlier best: about `90.8s`

Conclusion:

- prep-worker decoupling did not produce a strong or stable enough win
- this is not a promising next lever right now

Decision:

- do not continue prep-worker scaling experiments unless new profiling data says otherwise

### 7. Weighted embedding minimum-frame cutoff was nearly neutral or risky

The weighted-embedding minimum-frame cutoff was tested as another possible CPU lever.

Observed behavior:

- `15` frames was nearly neutral on wall time
- `20` frames saved a little more time
- but `20` also changed output segment count

Conclusion:

- this was not a safe enough speedup

Decision:

- back it out
- do not keep tuning this unless parity tolerance changes

### 8. Segmentation batching experiment: technically possible, not practically compelling

The current session also explored whether fixing the segmentation ONNX export to support real dynamic batching could materially reduce CPU time.

Work completed:

- confirmed the old export advertised dynamic input batch incorrectly
- confirmed it failed at runtime for `batch > 1`
- produced a batch-safe segmentation ONNX export for testing
- verified ONNX Runtime could execute it successfully at:
  - `batch=1`
  - `batch=2`
  - `batch=4`

However, the performance sweep on the `90s` benchmark did not show a meaningful CPU gain.

Measured diarization times with the batch-safe segmentation model:

- `batch=1`: `5052ms`
- `batch=2`: `4890ms`
- `batch=4`: `5061ms`
- `batch=8`: `4925ms`
- `batch=12`: `5282ms`
- `batch=16`: `5497ms`

Approximate diarization real-time factor:

- best observed point in that sweep: about `0.0543`
- baseline in that sweep: about `0.0561`

Interpretation:

- `2` and `8` were only about `2.5%` to `3.2%` better than `1`
- `12` and `16` clearly regressed
- returns flattened almost immediately

More importantly, the older segmentation export was still faster in the repo’s current end-to-end CPU pipeline:

- older export: `4471ms` diarization on the `90s` sample
- newer batch-safe export: `5052ms` diarization at `batch=1`

That means the export rewrite was interesting as a technical investigation, but not a good practical trade for this branch.

Decision:

- revert the export-mechanism experiment
- revert the runtime probing changes tied to it
- keep the original segmentation export in place

### 9. Current bottleneck direction

Given all of the above, the next promising performance direction is not more worker-layout experimentation.

The remaining hot area still appears to be the combined:

- segmentation
- embedding preparation

and especially the amount of repeated work across heavily overlapping chunks.

The strongest next candidates to investigate are:

- reducing repeated preprocessing work across chunk overlap
- reducing segmentation/decode intermediate allocation and copying
- reducing embedding job count more intelligently only if it can be done without harming parity

### 10. Segmentation decode/postprocess cleanup produced a real CPU win

After the batching/export experiment was reverted, the next pass focused on the per-chunk segmentation decode path in [DiariZenDiarizer.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/DiariZenDiarizer.cs).

The original flow did several full passes per chunk:

- softmax
- median filter
- powerset-to-speaker projection
- binarization

That also created multiple intermediate arrays for every chunk.

The current implementation now:

- caches the powerset speaker combinations once
- fuses median-filtered powerset decoding directly into per-speaker score accumulation
- binarizes from a compact linear score buffer
- avoids materializing the extra intermediate `filtered` and `softScores` arrays

Relevant code:

- [DiariZenDiarizer.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/DiariZenDiarizer.cs)

Measured CPU result on the `90s` benchmark slice using the built release CLI with `--skip-asr`:

- diarization time: `59325ms`
- diarization RTF: `0.6592`

Compared to the earlier CPU benchmark on the same `90s` slice:

- earlier diarization time: `70386ms`
- current diarization time: `59325ms`
- improvement: about `15.7%`

Safety check:

- parity rerun remained at `45` perfect matches with speaker-frame F1 `98.6%`
- the same extra trailing C#-only segment remained, so this change did not introduce a new parity regression pattern

Important note:

- the parity harness run that reported about `4289ms` was useful as a correctness check, but not as a CPU timing baseline
- that harness invocation was not the same controlled CPU release-binary benchmark used for the wall-time measurement above

### 11. Streaming/cache overlap reuse experiments were not worth keeping

After the decode cleanup win, two higher-risk experiments were tried to reduce repeated work across heavily overlapping windows.

The first experiment reused overlapping waveform chunk data when building the `16s` segmentation windows.

Result:

- parity stayed effectively unchanged
- clean CPU wall time did not improve enough to justify keeping the change

The second experiment precomputed a global raw WeSpeaker fbank timeline and then sliced chunk-local views with chunk-local mean normalization reapplied.

Result:

- a contaminated run briefly looked promising but was invalid
- the clean CPU rerun came in at `70411ms` diarization on the `90s` slice
- that was worse than the `59325ms` decode-optimized baseline

Interpretation:

- the simple overlap-reuse versions tested here did not reduce real end-to-end CPU time
- the bottleneck is likely deeper than chunk copying or naive chunk-fbank reuse
- these experiments were reverted

Decision:

- stop on this line of investigation for now
- do not keep pushing cached-overlap designs without a much stronger architecture case

## Things Explicitly Ruled Out For Now

Do not re-investigate these unless new profiling data changes the picture:

- full sweep-based release logic
- aggressive segmentation worker/session scaling
- more tiny embedding workers
- prep-worker decoupling/scaling
- weighted embedding minimum-frame cutoff tuning as a speed lever
- naive waveform-window overlap reuse
- naive global-fbank slicing for WeSpeaker chunk prep

## Test Hygiene Lessons

The performance work surfaced a few process lessons that matter enough to preserve:

- do not run multiple wall-time benchmark variants concurrently on the same machine
- host-visible `ps` is more trustworthy than sandbox-isolated process checks when validating live benchmark state
- clean reruns mattered a lot, because concurrency contamination distorted earlier timing results

One useful live-process sanity check from the earlier session showed:

- one real benchmark process
- about `37` matching threads early
- about `22` later
- no runaway process proliferation

## Tooling and Artifacts

Helpful scripts used during the performance work:

- [run_diarizen_parity.py](/home/chris/Programming/parakeet_csharp/scripts/diarizen/run_diarizen_parity.py)
- [benchmark_diarizen_configs.py](/home/chris/Programming/parakeet_csharp/scripts/diarizen/benchmark_diarizen_configs.py)
- [sweep_diarizen_segmentation_threads.py](/home/chris/Programming/parakeet_csharp/scripts/diarizen/sweep_diarizen_segmentation_threads.py)
- [PERFORMANCE.md](/home/chris/Programming/parakeet_csharp/scripts/diarizen/PERFORMANCE.md)

Recent batch-size experiment summary:

- [batch_size_rtf_sweep_summary.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/batch_size_rtf_sweep_summary.json)

Representative recent run artifacts:

- [run_20260404_123150_seg-batch1-newonnx](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260404_123150_seg-batch1-newonnx/run.log)
- [run_20260404_123205_seg-batch4-newonnx](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260404_123205_seg-batch4-newonnx/run.log)
- [run_20260404_123918_old-export-rtf-check](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260404_123918_old-export-rtf-check/run.log)

## Current State Of The Branch

The useful performance-oriented code changes still left in the branch are concentrated in:

- [Config.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/Config.cs)
- [HardwareInfo.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/HardwareInfo.cs)
- [WeSpeakerEmbedder.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/WeSpeakerEmbedder.cs)
- [DiariZenDiarizer.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/DiariZenDiarizer.cs)

The segmentation export experiment itself was reverted after measurement.
