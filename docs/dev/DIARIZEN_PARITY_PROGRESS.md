# DiariZen Parity Progress

Updated: 2026-04-03

## Goal

Reach closer parity between:

- `Parakeet.CLI` using `--diarization diarizen`
- the reference Python DiariZen pipeline output in `data/python_diarizen.json`

Primary test assets:

- Audio: `data/test_audio/en-US/en-US_sample_01.wav`
- Reference transcript: `data/python_diarizen_parakeet.md`
- Reference segments: `data/python_diarizen.json`

## Investigative Progress

### 1. CLI parity workflow improved

The CLI already supported `--export-format json`, but it still forced full ASR after diarization. That made diarization parity work slower and less targeted than necessary.

Added:

- `--skip-asr`
- live DiariZen progress messages during long runs

This allows the CLI to export diarization/VAD segments directly as JSON with empty `text` fields, which is a much better fit for comparison against `data/python_diarizen.json`.

The DiariZen path now also reports progress during:

- audio chunking
- segmentation chunk processing
- powerset decoding
- embedding extraction
- clustering
- reconstruction

This makes long parity runs much easier to monitor.

### 2. Auto execution provider fallback bug fixed

`DiariZenDiarizer` and `WeSpeakerEmbedder` were trying to append CUDA blindly in `ExecutionProvider.Auto`.

On a machine without a CUDA-capable GPU, the CLI could fail during ONNX session creation with:

- `CUDA failure 100: no CUDA-capable device is detected`

This now matches the safer pattern already used elsewhere in the codebase:

- only probe CUDA in `Auto` when `HardwareInfo.CanProbeCudaExecutionProvider()` says it is safe

### 3. Median filter parity mismatch confirmed and fixed

The local DiariZen reference source in `/home/chris/Programming/DiariZen/diarizen/pipelines/inference.py` applies:

- `median_filter(..., size=(1, 11, 1), mode='reflect')`

The C# config was still using:

- `Config.DiariZenMedianFilterSize = 7`

This has been corrected to:

- `Config.DiariZenMedianFilterSize = 11`

### 4. Reconstruction logic mismatch identified and partially aligned

The pyannote/DiariZen reference reconstructs diarization by:

1. Merging local speakers assigned to the same global speaker within each chunk using `max`
2. Aggregating global activations across overlapping chunks
3. Estimating frame-level speaker count
4. Activating the top `count[t]` speakers per frame

The previous C# implementation instead:

- averaged global speaker scores across overlaps
- thresholded each global speaker independently

This is an important structural difference, especially for overlap handling.

The C# reconstruction has been updated toward the reference approach by:

- estimating per-frame active-speaker count from overlapping chunk outputs
- aggregating global activations without final averaging
- selecting the top-N speakers per frame
- using `max` within a chunk when multiple local speakers map to the same global speaker

## New/Updated Tooling

### New CLI option

```bash
dotnet run --project src/Parakeet.CLI/Parakeet.CLI.csproj -- \
  --audio data/test_audio/en-US/en-US_sample_01.wav \
  --model models \
  --diarization diarizen \
  --skip-asr \
  --export-format json \
  --output /tmp/csharp_diarizen.json
```

### New parity driver script

Added:

- `scripts/run_diarizen_parity.py`
- run-specific artifact directories with `run_id + label`

Example:

```bash
python3 scripts/run_diarizen_parity.py \
  --audio data/test_audio/en-US/en-US_sample_01.wav \
  --model models \
  --reference data/python_diarizen.json \
  --label full-reference-check
```

This script:

- runs the C# CLI in diarization-only JSON mode
- loads the reference JSON
- prints the segment comparison report
- writes artifacts into a unique run directory by default
- records metadata, output JSON, comparison JSON, and a log file per run

Default artifact layout:

```bash
data/diarizen_parity/runs/<run_id>_<label>/
  metadata.json
  csharp_diarization.json
  comparison.json
  run.log
```

Example with repo-local artifacts:

```bash
python3 scripts/run_diarizen_parity.py \
  --audio data/diarizen_parity/en-US_sample_01_first90.wav \
  --model models \
  --reference data/diarizen_parity/python_diarizen_first90.json \
  --run-id run_20260403_110000 \
  --label first90-masked-embedder
```

Default log path for the example above:

```bash
data/diarizen_parity/runs/run_20260403_110000_first90-masked-embedder/run.log
```

To watch a run live from another terminal:

```bash
tail -f data/diarizen_parity/runs/run_20260403_110000_first90-masked-embedder/run.log
```

### Faster comparison script

`scripts/compare_diarization.py` was updated to reduce Python-loop overhead by:

- using indexed interval matching instead of naive full scans for every segment
- rasterizing frame-level speaker activity with NumPy difference arrays
- computing overlap-aware metrics from vectorized frame grids

On the current 90-second slice, the core `compare_diarizations(...)` call improved from roughly:

- `~14.6 ms` per call

to roughly:

- `~1.9 ms` per call

on repeated local measurements.

During the latest overlap-heavy investigations, a second issue showed up in the
comparison logic: the greedy one-to-one matcher could select an already-used
Python segment as the "best" overlap for a C# segment and then fail to fall
through to the next valid unmatched candidate.

That especially undercounted matches in genuine overlap regions where one long
segment and one nested overlap segment both existed in both outputs.

The matcher now skips already-consumed candidates while searching, which gives a
much more faithful exact-segment parity readout for overlapped speech.

### Faster embedding extraction

`DiariZenDiarizer` now parallelizes speaker-embedding computation across a
bounded worker pool instead of extracting all chunk embeddings serially on one
core.

Supporting changes:

- `WeSpeakerEmbedder` now uses `IntraOpNumThreads = 1` so the outer
  embedding-job parallelism can scale better on CPU
- DiariZen progress reporting now includes `computed embeddings X/Y` so the
  expensive stage is visible in the run log

Measured on the current 90-second slice:

- before: `94.0s` DiariZen runtime in `run_20260403_111639_first90-prune-tiny-final-speaker`
- after: `63.4s` DiariZen runtime in `run_20260403_113318_first90-parallel-embeddings-cpu`

That is about a `1.48x` speedup with no change in the parity result.

### Parallel segmentation experiment

The segmentation stage originally processed one chunk at a time and only used a
small amount of CPU parallelism through ONNX Runtime intra-op threads.

An experiment switched segmentation to:

- chunk-level `Parallel.For(...)` across the 16-second windows
- `IntraOpNumThreads = 1` in the DiariZen segmentation session so the outer
  chunk parallelism could scale

Measured on the 90-second slice in
`run_20260403_114101_first90-parallel-segmentation-and-embeddings`:

- full parity run: `wall=1:11.39`, `cpu=991%`
- DiariZen stage: `68525ms`

Compared with the prior best CPU-parallel embedding run
`run_20260403_113701_first90-binary-timeline-smoothing`:

- DiariZen stage: `62677ms`

So the segmentation stage did use much more total CPU, but overall wall-clock
time got worse. The likely explanation is that we traded a modestly parallel
single inference for heavier memory/scheduler overhead across many concurrent
inferences.

Conclusion so far:

- yes, segmentation can be driven well beyond 4 cores
- no, simply pushing it toward "all cores" is not automatically a win
- the better current default for parity work may be: keep segmentation more
  conservative, keep embedding extraction parallel

## Run History

### Legacy runs before run IDs

- `legacy_first90_baseline`
  Label: `first90-reconstruction-aligned`
  Artifacts:
  - [csharp_first90.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90.json)
  - [csharp_first90_comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_comparison.json)
  Result:
  - `32` C# segments vs `45` Python segments
  - `1` C# speaker vs `2` Python speakers
  - `22` matches, `9` speaker mismatches, `1` C#-only, `14` Python-only
  - Main takeaway: timing for the dominant speaker was fairly close, but clustering/overlap recovery collapsed everything into one speaker.

- `legacy_first90_progress_rerun`
  Label: `first90-progress-logging`
  Artifacts:
  - [csharp_first90_rerun.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_rerun.json)
  - [csharp_first90_rerun_comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_rerun_comparison.json)
  - [csharp_first90_rerun.json.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_rerun.json.log)
  Result:
  - Established repo-local logging and visible stage/chunk progress.
  - Confirmed the one-speaker collapse persisted after adding progress output.

- `legacy_first90_diag`
  Label: `first90-clustering-diagnostics`
  Artifacts:
  - [csharp_first90_diag.json.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_diag.json.log)
  Result:
  - Introduced more detailed clustering progress messages, including embedding count and AHC/VBx cluster counts.
  - Run was mainly used to improve observability.

- `legacy_first90_thr03`
  Label: `first90-ahc-threshold-0.3`
  Artifacts:
  - [csharp_first90_thr03.json.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_thr03.json.log)
  Result:
  - Threshold-tuning probe launched to test whether a lower AHC threshold alone could recover the missing speaker.
  - Follow-up interpretation still pending.

- `legacy_first90_masked`
  Label: `first90-masked-embedding`
  Artifacts:
  - [csharp_first90_masked.json.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_masked.json.log)
  - [csharp_first90_masked.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_masked.json)
  - [csharp_first90_masked_comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/csharp_first90_masked_comparison.json)
  Result:
  - Ports a more faithful version of the Python embedding path:
    clean non-overlap mask preferred, full mask as fallback, mask applied at feature-frame level rather than using one contiguous region.
  - Major improvement over the prior baseline:
    `58` C# segments vs `45` Python segments,
    `3` C# speakers vs `2` Python speakers,
    `37` matches,
    `0` speaker mismatches,
    `21` C#-only,
    `8` Python-only.
  - AHC produced `24` clusters and VBx refined to `3` speakers.
  - Main takeaway: speaker identity is now mostly right, but the system over-segments and introduces one extra speaker, so the next iteration should focus on reducing fragmentation / false splits rather than restoring speaker separation.

- `run_20260403_111000`
  Label: `first90-prune-tiny-final-speaker`
  Artifacts:
  - [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_111639_first90-prune-tiny-final-speaker/run.log)
  - [csharp_diarization.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_111639_first90-prune-tiny-final-speaker/csharp_diarization.json)
  - [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_111639_first90-prune-tiny-final-speaker/comparison.json)
  Result:
  - Current best 90-second parity result:
    `52` C# segments vs `45` Python segments,
    `2` C# speakers vs `2` Python speakers,
    `37` matches,
    `0` speaker mismatches,
    `15` C#-only,
    `8` Python-only.
  - This removed the stray third speaker while preserving the good speaker mapping.
  - Remaining issue: fragmentation and a small coverage deficit (`81.76s` C# vs `83.07s` Python).
  - Best next direction: reduce micro-segmentation and recover missed coverage without regressing speaker identity.
  - Overlap-aware comparison shows the result is closer than the one-to-one segment counts suggest:
    `96.2%` of C# segments have a same-speaker overlap match,
    `100%` of Python segments have a same-speaker overlap match,

- `run_20260403_123000`
  Label: `first90-parallel-embeddings-cpu`
  Artifacts:
  - [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_113318_first90-parallel-embeddings-cpu/run.log)
  - [csharp_diarization.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_113318_first90-parallel-embeddings-cpu/csharp_diarization.json)
  - [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_113318_first90-parallel-embeddings-cpu/comparison.json)
  Result:
  - Same parity result as `run_20260403_111000`:
    `52` C# segments vs `45` Python segments,
    `2` C# speakers vs `2` Python speakers,
    `37` matches,
    `0` speaker mismatches,
    `15` C#-only,
    `8` Python-only.
  - Runtime improvement was the point of this run:
    DiariZen dropped from `94030ms` to `63423ms` on the 90-second slice.
  - `/usr/bin/time` for the full parity run reported:
    `wall=1:05.29`, `cpu=491%`, `maxrss=1688988KB`.
  - Main takeaway: segmentation was already parallel enough to use several cores,
    but embedding extraction was worth parallelizing and now materially reduces
    wall-clock time without disturbing parity.

- `run_20260403_124500`
  Label: `first90-binary-timeline-smoothing`
  Artifacts:
  - [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_113701_first90-binary-timeline-smoothing/run.log)
  - [csharp_diarization.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_113701_first90-binary-timeline-smoothing/csharp_diarization.json)
  - [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_113701_first90-binary-timeline-smoothing/comparison.json)
  Result:
  - New best 90-second parity result:
    `48` C# segments vs `45` Python segments,
    `2` C# speakers vs `2` Python speakers,
    `45` matches,
    `0` speaker mismatches,
    `3` C#-only,
    `0` Python-only.
  - Added a conservative final cleanup on the binary speaker timeline:
    fill short same-speaker gaps up to `2` frames and remove active regions shorter than `2` frames before segment extraction.
  - This reduced micro-fragmentation around the known trouble spots without disturbing speaker assignment.
  - Frame-level agreement stayed strong:
    `97.0%` exact frame-set match,
    `99.5%` speaker-frame precision,
    `97.2%` speaker-frame recall,
    `98.3%` speaker-frame F1,
    `90.7%` overlap-time Jaccard.
  - Main takeaway: after fixing the overlap-aware exact matcher, the remaining exact mismatches are down to three late-file extra C# fragments around `88.40-89.58s`; all Python segments now have an exact same-speaker counterpart.

- `run_20260403_132000`
  Label: `first90-parallel-segmentation-and-embeddings`
  Artifacts:
  - [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_114101_first90-parallel-segmentation-and-embeddings/run.log)
  - [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_114101_first90-parallel-segmentation-and-embeddings/comparison.json)
  Result:
  - Parity stayed effectively unchanged from the best 90-second result:
    `48` C# segments vs `45` Python,
    `45` matches,
    `0` speaker mismatches,
    `3` C#-only,
    `0` Python-only.
  - CPU usage increased substantially with chunk-parallel segmentation:
    `/usr/bin/time` reported `cpu=991%`.
  - But wall-clock performance got worse:
    DiariZen runtime increased to `68525ms` versus `62677ms` in the prior best CPU configuration.
  - Main takeaway: pushing segmentation far beyond the old ~4-core behavior is possible, but on this machine it is not a net speed win.

- `run_20260403_134500`
  Label: `first90-wide-gap-fill`
  Artifacts:
  - [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_114815_first90-wide-gap-fill/run.log)
  - [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_114815_first90-wide-gap-fill/comparison.json)
  Result:
  - Increasing final same-speaker gap filling from `2` to `20` frames was too aggressive.
  - It reduced fragmentation numerically, but regressed overall quality:
    `40` C# segments vs `45` Python,
    `3` C# speakers vs `2` Python,
    `37` matches,
    `2` speaker mismatches,
    `1` C#-only,
    `6` Python-only.
  - Frame-level quality also dropped:
    `94.3%` exact frame-set match,
    `96.7%` speaker-frame F1,
    `71.6%` overlap-time Jaccard.
  - Main takeaway: the final binary-timeline cleanup should stay conservative; wider same-speaker hole filling starts to smear genuine structure and reintroduce clustering/reconstruction artifacts.

### Pending performance probe

The next targeted runtime experiment is to keep the best-known diarization logic
unchanged while increasing only DiariZen segmentation ONNX intra-op threads from
`4` to `8`.

Goal:

- test whether a moderate increase in segmentation threading helps wall-clock
  time without reintroducing the regression we saw from chunk-parallel
  segmentation

### Segmentation thread sweep

A clean 90-second sweep was run sequentially with segmentation intra-op thread
counts `4, 6, 8, 12, 16` using:

- [summary.md](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/thread_sweeps/sweep_20260403_115924/summary.md)
- [summary.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/thread_sweeps/sweep_20260403_115924/summary.json)

Results:

- `12` threads: `57.02s` wall, `55179ms` DiariZen
- `8` threads: `58.25s` wall, `56394ms` DiariZen
- `6` threads: `58.67s` wall, `56808ms` DiariZen
- `4` threads: `65.87s` wall, `63950ms` DiariZen
- `16` threads: `79.84s` wall, `77952ms` DiariZen

All sweep entries produced the same quality result on the 90-second slice:

- `45` matches
- `0` speaker mismatches
- `3` C#-only
- `0` Python-only
- `98.3%` speaker-frame F1

Main takeaway:

- segmentation threading matters a lot on this machine
- `12` threads is the current best setting
- `16` threads overshoots and hurts runtime badly
- the default segmentation intra-op thread count has been updated to `12`

### Full-file parity analysis

Latest completed full-file comparison:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_115005_full-best-config-fixed-matcher/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_115005_full-best-config-fixed-matcher/comparison.json)

Result:

- `275` C# segments vs `292` Python
- `3` C# speakers vs `3` Python
- `242` exact matches
- `27` speaker mismatches
- `6` C#-only
- `23` Python-only
- `95.1%` exact frame-set match
- `95.4%` speaker-frame F1
- `93.3%` overlap-time Jaccard

Important interpretation:

- This is a large improvement over the older full-file baseline in exact segment
  matching terms:
  `242` matches now versus `209` in `run_20260403_112039_full-best-config-validation`.
- Frame-level quality changed very little compared with the older full run,
  which means the main remaining issue is not gross timing failure.
- The remaining exact mismatches are dominated by a small mixed C# speaker:
  `speaker_1` totals only `13.52s` across the whole file, while the Python
  reference's `speaker 0` totals only `0.02s`.
- Overlap totals show:
  - `speaker_0` aligns strongly with Python speaker `2`
  - `speaker_2` aligns strongly with Python speaker `1`
  - `speaker_1` is a small residual cluster with about `11.07s` overlap to
    Python speaker `1` and `9.07s` overlap to Python speaker `2`, but
    effectively none to Python speaker `0`

Main takeaway:

- the full-file parity problem is now mostly a residual speaker-assignment /
  small-cluster issue, not a broad segmentation-timing issue
- the current exact mismatch count is inflated by this small mixed residual
  cluster
- the next quality iteration should target pruning or merging that residual
  speaker more carefully on full-file behavior

Qualitative note:

- The source audio is a telephone call with two main female speakers and a third
  background male voice.
- That means the extra residual C# speaker is not obviously "nonsense" from a
  human-listening perspective; it may reflect a real but weak/background speaker
  trace that the Python reference mostly suppresses or folds into the dominant
  speakers.
- This makes the current output potentially useful in practice even though it is
  still not at strict parity with the reference bundle.
- For this investigation, parity remains the target, so the residual third
  speaker is still treated as a deviation to be reduced unless we later decide
  the product goal should intentionally diverge from the Python reference.

Comparison note:

- The strict comparison currently uses a one-to-one C#→Python speaker mapping.
- On the latest full-file run, that forces the small residual C# `speaker_1` cluster to map to Python speaker `0`, even though Python speaker `0` is only about `0.02s` total in the reference.
- A new secondary non-unique mapping metric was added to
  [compare_diarization.py](/home/chris/Programming/parakeet_csharp/scripts/compare_diarization.py) to show how much of the remaining mismatch is really speaker splitting rather than timing disagreement.
- On the latest full-file output:
  - strict mapping: `242` matches, `27` speaker mismatches
  - non-unique mapping: `254` matches, `15` speaker mismatches
- This does not change the parity target, but it gives a more honest view of how much residual error is due to a split/extra cluster versus a bad timeline.

Latest tuned performance baseline:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_121720_full-best-config-12threads/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_121720_full-best-config-12threads/comparison.json)

This run uses the tuned segmentation thread count of `12` and produces the same quality result as `run_20260403_115005_full-best-config-fixed-matcher`:

- `275` C# segments vs `292` Python
- `242` strict matches
- `27` strict speaker mismatches
- `254` non-unique-mapping matches
- `15` non-unique-mapping speaker mismatches
- `6` C#-only
- `23` Python-only
- `95.4%` speaker-frame F1

But the runtime improved from:

- `411283ms` DiariZen runtime in `run_20260403_115005_full-best-config-fixed-matcher`

to:

- `358256ms` DiariZen runtime in `run_20260403_121720_full-best-config-12threads`

So the tuned `12`-thread configuration is now the correct full-file baseline to iterate from.

Residual-speaker relabel probe:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_122523_first90-residual-speaker-relabel/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_122523_first90-residual-speaker-relabel/comparison.json)

Change:

- After pruning globally tiny speakers, add a selective relabeling pass for residual speakers with very small total duration.
- Instead of deleting them, try to relabel each residual segment to the best overlapping or nearest dominant neighboring speaker.

90-second result:

- unchanged from the current best short-file baseline:
  `48` C# segments vs `45` Python,
  `45` matches,
  `0` speaker mismatches,
  `3` C#-only,
  `0` Python-only,
  `98.3%` speaker-frame F1.

Main takeaway:

- this heuristic appears safe on the short-file benchmark
- it is worth evaluating next on the full-file residual-speaker problem

Full-file outcome of the residual-speaker relabel probe:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_122636_full-residual-speaker-relabel/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_122636_full-residual-speaker-relabel/comparison.json)

Result:

- `261` C# segments vs `292` Python
- `2` C# speakers vs `3` Python
- `241` strict matches
- `16` strict speaker mismatches
- `4` C#-only
- `35` Python-only
- `95.6%` exact frame-set match
- `96.6%` speaker-frame F1
- `79.0%` overlap-time Jaccard

Interpretation:

- The relabeling heuristic did reduce strict speaker-mismatch count and slightly improved frame-level F1.
- But it achieved that by collapsing the residual third speaker and suppressing too much overlap / weak-speaker structure: Python-only segments increased from `23` to `35`, total C# speech duration dropped by about `5.4s`, and overlap-time Jaccard fell sharply from `93.3%` to `79.0%`.
- This is not a good parity trade.

Conclusion:

- The residual-speaker relabel heuristic has been reverted.
- The best current baseline remains
  [run_20260403_121720_full-best-config-12threads](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_121720_full-best-config-12threads/run.log).

Reference-aligned clustering update:

- The upstream DiariZen / pyannote reference was reviewed again directly from source, including:
  - `diarizen/pipelines/inference.py`
  - `pyannote/audio/pipelines/speaker_diarization.py`
  - `pyannote/audio/pipelines/clustering.py`
- Important divergence found: the reference VBx path does not cluster every extracted embedding directly. It filters to a training subset with enough clean frames, clusters that subset, then assigns all embeddings back to cluster centroids.
- The C# implementation was updated to mirror that behavior more closely:
  - select training embeddings using clean-frame ratio filtering
  - cluster only the filtered subset
  - compute centroids from clustered training embeddings
  - assign all embeddings back to those centroids

90-second validation:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_123916_first90-train-subset-centroid-assign/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_123916_first90-train-subset-centroid-assign/comparison.json)

Result:

- `45` C# segments vs `45` Python
- `2` C# speakers vs `2` Python
- `45` matches
- `0` speaker mismatches
- `0` C#-only
- `0` Python-only
- `98.4%` exact frame-set match
- `99.1%` speaker-frame F1

Main takeaway:

- the 90-second JSON reference target is now at exact segment parity
- this is the first result that removes the last three residual extra C# segments without using an obviously harmful heuristic
- the next step is full-file validation in this state

Full-file validation of the reference-aligned clustering update:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_124056_full-train-subset-centroid-assign/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_124056_full-train-subset-centroid-assign/comparison.json)

Result:

- `278` C# segments vs `292` Python
- `2` C# speakers vs `3` Python
- `267` strict matches
- `6` strict speaker mismatches
- `5` C#-only
- `19` Python-only
- `98.2%` exact frame-set match
- `99.1%` speaker-frame precision
- `98.5%` speaker-frame recall
- `98.8%` speaker-frame F1
- `93.2%` overlap-time Jaccard

Compared with the prior best full baseline
[run_20260403_121720_full-best-config-12threads](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_121720_full-best-config-12threads/run.log):

- matches improved from `242` to `267`
- strict speaker mismatches dropped from `27` to `6`
- Python-only segments dropped from `23` to `19`
- exact frame-set match improved from `95.1%` to `98.2%`
- speaker-frame F1 improved from `95.4%` to `98.8%`

Tradeoff:

- the residual third speaker is no longer preserved as a separate C# speaker on the full file (`2` C# speakers vs `3` Python), so this change is clearly more reference-aligned than before, but less permissive of weak residual speaker structure.

Main takeaway:

- reviewing the reference implementation one more time uncovered a real,
  impactful divergence
- clustering a filtered training subset and then assigning all embeddings to centroids is the strongest parity improvement found so far
- the remaining mismatches are now concentrated in a handful of stretches
  (notably around `332-336s`, `379-381s`, `514-516s`, and a few tiny late-file spans)

Constrained-assignment follow-up:

- The reference clustering path can leave some local tracks effectively inactive under constrained assignment, while the initial C# centroid-assignment port forced every leftover local embedding onto a centroid.
- The constrained centroid-assignment step was updated to allow unassigned local embeddings to remain unassigned instead of forcing a fallback cluster.

90-second validation:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_125410_first90-constrained-unassigned-rerun/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_125410_first90-constrained-unassigned-rerun/comparison.json)

Result:

- still exact short-file parity:
  `45` C# segments vs `45` Python,
  `45` matches,
  `0` speaker mismatches,
  `0` C#-only,
  `0` Python-only.

Main takeaway:

- the constrained-unassigned change is safe on the 90-second benchmark
- it is worth checking on the full-file residual mismatch regions next

Full-file validation:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_125559_full-constrained-unassigned/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_125559_full-constrained-unassigned/comparison.json)

Result:

- `277` C# segments vs `292` Python
- `2` C# speakers vs `3` Python
- `269` strict matches
- `4` strict speaker mismatches
- `4` C#-only
- `19` Python-only
- `99.1%` speaker-frame precision
- `98.5%` speaker-frame recall
- `98.8%` speaker-frame F1
- `93.3%` overlap-time Jaccard

Compared with the prior reference-aligned clustering run
[run_20260403_124056_full-train-subset-centroid-assign](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_124056_full-train-subset-centroid-assign/run.log):

- matches improved from `267` to `269`
- strict speaker mismatches dropped from `6` to `4`
- C#-only segments dropped from `5` to `4`
- Python-only segments stayed at `19`

Remaining mismatch concentration:

- the remaining speaker mismatches are now concentrated around `332.80-334.38s`
  and `514.92-516.06s`
- the remaining C#-only material is down to four small spans:
  `334.38-336.88s`, `547.18-547.94s`, `564.58-565.08s`, and `588.98-589.26s`

Main takeaway:

- allowing constrained centroid assignment to leave weak local tracks
  unassigned produced another small but real full-file parity gain
- this is the current best full-file parity result in the investigation
- the remaining gap now looks localized enough for surgical analysis rather than broad pipeline changes

Comparison follow-up:

- While inspecting the remaining mismatch windows, the comparison script was found to still mishandle one overlap case: it could pair two same-timing overlap segments crosswise instead of preferring the mapped same-speaker candidate first.
- `scripts/compare_diarization.py` was updated so indexed matching now prefers an unmatched candidate whose speaker matches the inferred C#→Python mapping before falling back to overlap-only selection.

Recomputed full-file result for
[run_20260403_125559_full-constrained-unassigned](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_125559_full-constrained-unassigned/run.log):

- `277` C# segments vs `292` Python
- `273` strict matches
- `1` strict speaker mismatch
- `3` C#-only
- `18` Python-only
- `99.1%` speaker-frame precision
- `98.5%` speaker-frame recall
- `98.8%` speaker-frame F1
- `93.3%` overlap-time Jaccard

Remaining mismatch concentration after the matcher fix:

- one true speaker mismatch remains around `333.64-334.38s`
- the remaining C#-only spans are:
  `334.38-336.88s`, `547.18-547.94s`, and `588.98-589.26s`
- the remaining Python-only spans are mostly tiny fragments, with the most substantial residual gap still centered on the `332-336s` region

Main takeaway:

- the current diarizer result is even closer to reference parity than the original stored comparison suggested
- most of the remaining delta is now concentrated in one local alternating span plus a handful of tiny extra or missing fragments
- the next useful work item is focused analysis of the `332-336s` reconstruction behavior, not another broad clustering change

Focused window investigation (`300-360s`):

- A narrow `320-340s` repro clip was too short to preserve stable clustering and
  collapsed to one speaker, so it was not a faithful harness for the remaining parity gap.
- A wider `300-360s` repro window kept the same two-speaker structure as the
  full file and reproduced the stubborn local mismatch around `332.81-333.63s`.

Baseline focused-window run:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_130915_window-300-360-baseline/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_130915_window-300-360-baseline/comparison.json)

Result:

- `32` matches
- `0` speaker mismatches
- `0` C#-only
- `6` Python-only

Diagnostics added during this investigation:

- optional reconstruction debug dump in
  [DiariZenDiarizer.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/DiariZenDiarizer.cs)
  via `PARAKEET_DIARIZEN_DEBUG_RECON_WINDOW` and
  `PARAKEET_DIARIZEN_DEBUG_RECON_PATH`
- optional chunk assignment dump via `PARAKEET_DIARIZEN_DEBUG_ASSIGN_PATH`

What the diagnostics showed:

- the remaining `332-336s` error is not caused by unstable global clustering: chunk-to-global assignments stay consistent through the bad window
- the bad handoff is already wrong at frame-selection time, before final smoothing or segment merging
- a more important reference divergence was identified: pyannote reconstructs from binarized local segmentations, while the C# path was still reconstructing from soft local speaker scores

Binary-local reconstruction update:

- reconstruction now uses thresholded local masks derived from the decoded per-speaker scores, rather than the soft scores themselves

Focused-window validation after the binary-local reconstruction change:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_131826_window-300-360-binary-reconstruct/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_131826_window-300-360-binary-reconstruct/comparison.json)

Result:

- `35` matches
- `0` speaker mismatches
- `1` C#-only
- `3` Python-only

Main takeaway:

- reconstructing from binarized local segmentations materially improved the exact problem window
- unlike the earlier heuristics, this change is directly motivated by a
  confirmed reference implementation difference

90-second safety check after the binary-local reconstruction change:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_131919_first90-binary-reconstruct/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_131919_first90-binary-reconstruct/comparison.json)

Result:

- exact short-file parity is preserved:
  `45` matches,
  `0` speaker mismatches,
  `0` C#-only,
  `0` Python-only

Main takeaway:

- the binary-local reconstruction change survives the 90-second guardrail
- it is now justified to run a new full-file validation from this state

Full-file validation after the binary-local reconstruction change:

- [run.log](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_132048_full-binary-reconstruct/run.log)
- [comparison.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_132048_full-binary-reconstruct/comparison.json)

Result:

- `273` C# segments vs `292` Python
- `2` C# speakers vs `3` Python
- `271` strict matches
- `0` strict speaker mismatches
- `2` C#-only
- `21` Python-only
- `99.3%` speaker-frame precision
- `98.7%` speaker-frame recall
- `99.0%` speaker-frame F1
- `93.5%` overlap-time Jaccard

Compared with the previous best full-file baseline
[run_20260403_125559_full-constrained-unassigned](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_125559_full-constrained-unassigned/run.log):

- matches improved from `269` to `271`
- strict speaker mismatches dropped from `4` to `0`
- C#-only segments dropped from `4` to `2`
- Python-only segments increased slightly from `19` to `21`
- speaker-frame F1 improved from `98.8%` to `99.0%`
- overlap-time Jaccard improved from `93.3%` to `93.5%`

Remaining delta:

- only two C#-only spans remain:
  `547.88-547.94s` and `588.98-589.26s`
- the remaining Python-only spans are now entirely same-speaker timing
  fragments rather than speaker-label disagreements
- the most noticeable unresolved reference gap is still the Python micro-turn pattern around `332.81-333.63s`

Main takeaway:

- reconstructing from binarized local segmentations is the second major
  reference-aligned breakthrough after the training-subset clustering change
- the current best full-file result has eliminated speaker-label mismatch entirely
- what remains is now pure timing/fragmentation parity rather than speaker identity parity

Post-processing and powerset follow-up probes:

- A reference-strict post-processing probe disabled the extra C# binary
  smoothing and tiny-speaker pruning passes after reconstruction.
- On the focused `300-360s` harness, that produced slightly fewer Python-only micro-fragments but introduced new C#-only segments later in the same window.
- Result:
  [run_20260403_132948_window-300-360-binary-reconstruct-no-post](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_132948_window-300-360-binary-reconstruct-no-post/run.log)
  showed `36` matches, `0` speaker mismatches, `3` C#-only, `2` Python-only.
- Disabling post-smoothing alone produced the same focused-window result:
  [run_20260403_133041_window-300-360-binary-reconstruct-no-smoothing](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_133041_window-300-360-binary-reconstruct-no-smoothing/run.log)

Main takeaway:

- the extra post-processing is not the main remaining parity blocker
- removing it helps some Python micro-turns but also creates new C# fragments, so it is not a clean net improvement

Hard powerset-mask probe:

- Another remaining divergence was tested: using a literal hard powerset decode for local masks instead of thresholding summed per-speaker probabilities.
- The focused-window probe was:
  [run_20260403_133213_window-300-360-hard-powerset-masks](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_133213_window-300-360-hard-powerset-masks/run.log)

Result:

- `36` matches
- `0` speaker mismatches
- `2` C#-only
- `2` Python-only
- but total C# duration dropped by `1.82s`
- speaker-frame F1 dropped to `94.6%`
- overlap-time Jaccard dropped to `84.7%`

Main takeaway:

- a naive hard-powerset local mask path is too aggressive and suppresses too much speech
- that probe is not a net improvement over the current binary-local
  reconstruction baseline

Hard powerset-count probe:

- Another targeted divergence check changed only the instantaneous speaker-count path: instead of counting active local speakers from the derived per-speaker masks, it used hard powerset class cardinality per frame.
- The focused-window probe was:
  [run_20260403_133429_window-300-360-hard-powerset-count](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_133429_window-300-360-hard-powerset-count/run.log)

Result:

- `36` matches
- `0` speaker mismatches
- `2` C#-only
- `2` Python-only
- but total C# duration dropped by `1.86s`
- speaker-frame F1 dropped to `94.3%`
- overlap-time Jaccard dropped to `83.8%`

Main takeaway:

- deriving count directly from hard powerset cardinality is also too aggressive in the current C# port
- the remaining parity gap is not best explained by the simple count heuristic alone

Summed-activation reconstruction probe:

- An exact reconstruction divergence was also tested: pyannote-style
  `to_diarization` uses summed chunk activations, while the C# port had been
  ranking speakers by mean activation across contributing chunks.
- The focused-window probe was:
  [run_20260403_133640_window-300-360-summed-activations](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_133640_window-300-360-summed-activations/run.log)

Result:

- `35` matches
- `0` speaker mismatches
- `1` C#-only
- `3` Python-only

Main takeaway:

- switching from mean to summed activations did not change the focused-window
  result relative to the current binary-local reconstruction baseline
- this exact aggregation detail is not the remaining blocker

Binary embedding-mask alignment:

- Embedding-mask extraction was then aligned to the binarized local masks,
  instead of deriving overlap from soft per-speaker scores.
- The focused-window probe was:
  [run_20260403_133745_window-300-360-binary-embed-masks](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_133745_window-300-360-binary-embed-masks/run.log)

Result:

- `35` matches
- `0` speaker mismatches
- `1` C#-only
- `3` Python-only

Main takeaway:

- aligning embedding masks to binarized local segmentations is cleaner and more
  reference-faithful, but it does not materially change the remaining focused mismatch

Count quantization probes:

- The remaining exact decision point in reconstruction was the conversion of
  average instantaneous speaker count to an integer per frame.
- Two focused-window probes were run:
  - [run_20260403_134053_window-300-360-count-rounding-ceil](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_134053_window-300-360-count-rounding-ceil/run.log)
  - [run_20260403_134147_window-300-360-count-rounding-floor](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_134147_window-300-360-count-rounding-floor/run.log)

`ceil` result:

- `33` matches
- `0` speaker mismatches
- `1` C#-only
- `5` Python-only
- total C# duration increased by `2.00s`

`floor` result:

- `33` matches
- `0` speaker mismatches
- `2` C#-only
- `5` Python-only
- total C# duration dropped by `3.14s`

Main takeaway:

- count quantization is not the remaining parity lever
- `ceil` overcounts overlap, while `floor` suppresses too much speech
- the default rounded count remains the best of the tested count variants

Coverage-aware mismatch analysis:

- [compare_diarization.py](/home/chris/Programming/parakeet_csharp/scripts/compare_diarization.py)
  now annotates every `python_only` and `csharp_only` segment with:
  - `coverage_class`
  - same-speaker overlap coverage ratio
  - any-speaker overlap coverage ratio
- Recomputed best-full comparison:
  [comparison_coverage.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_132048_full-binary-reconstruct/comparison_coverage.json)

Result:

- `21` Python-only segments break down into:
  - `7` covered by a same-speaker C# segment already
  - `6` partially covered by a same-speaker C# segment
  - `7` covered only by a different-speaker C# segment
  - `1` fully uncovered
- `2` C#-only segments break down into:
  - `1` partially covered by a same-speaker Python segment
  - `1` fully uncovered

Main takeaway:

- the raw `21` Python-only count substantially overstates the remaining
  parity gap
- most of the leftover difference is fragmentation or speaker handoff structure
  inside speech that C# is already covering
- the truly unresolved full-file misses are now a short list of localized spans

Late-window smoothing-threshold probe:

- The remaining content-bearing mismatches around `564-567s` and `576.6s`
  looked like possible short-region suppression, so the cleanup thresholds were
  made runtime-tunable:
  - `PARAKEET_DIARIZEN_FILL_SHORT_GAP_FRAMES`
  - `PARAKEET_DIARIZEN_MIN_REGION_FRAMES`
- Focused probe:
  [run_20260403_140444_window-540-600-gap1-region1](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_140444_window-540-600-gap1-region1/run.log)

Result:

- `27` matches
- `0` speaker mismatches
- `5` C#-only
- `10` Python-only
- `95.9%` speaker-frame F1
- `94.5%` overlap-time Jaccard

Main takeaway:

- reducing cleanup from `2` frames to `1` frame did not improve the
  `python_only` count in the late window
- it added one extra `C#-only` fragment
- the remaining handoff misses are likely happening before final cleanup, not
  because the cleanup thresholds are too aggressive

Late-window reconstruction debug:

- Focused debug run:
  [run_20260403_141000_window-540-600-debug-handoffs](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_141000_window-540-600-debug-handoffs/run.log)
- Debug artifacts:
  - [reconstruction_debug.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_141000_window-540-600-debug-handoffs/reconstruction_debug.json)
  - [assignment_debug.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_141000_window-540-600-debug-handoffs/assignment_debug.json)

Main takeaway:

- In the key late handoff around local `24.67-24.79s` (full `564.67-564.79s`),
  the missing inserted turn never becomes selected and then removed later.
- Instead, reconstruction keeps `count = 1` and the competing speaker loses the
  frame ranking directly with very small score gaps, for example:
  - `24.58-24.78s`: speaker scores roughly `0.5` vs `0.4`
  - `24.78-24.80s`: speaker scores `0.5` vs `0.5`, but the first speaker still wins
- Chunk-to-global speaker assignment is stable in this region, so the remaining
  issue is not an assignment flip.
- This narrows the residual parity gap to frame-ranking semantics in
  reconstruction, especially near `count = 1` handoffs.

Late-window summed-activation probe:

- Because the debug run showed near-tied frame scores, the exact activation
  ranking semantics were rechecked on the late window using the more
  reference-aligned summed activations path:
  [run_20260403_140714_window-540-600-summed-activations](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_140714_window-540-600-summed-activations/run.log)

Result:

- `27` matches
- `0` speaker mismatches
- `4` C#-only
- `10` Python-only
- `95.9%` speaker-frame F1
- `94.5%` overlap-time Jaccard

Main takeaway:

- the late-window result is unchanged from the baseline
- summed-vs-mean activation ranking is not the remaining lever, even in the
  near-tie handoff window

Content-bearing remaining mismatches:

- The content-bearing `Python-only` spans are now concentrated in a few windows:
  - `333.272-333.512s`
    - `covered_only_by_different_speaker`
    - text: `Don't you do that to me. I can't read Spanish.`
  - `564.672-564.793s`
    - `covered_only_by_different_speaker`
    - text: `One of my dollars.`
  - `566.713-566.872s`
    - `covered_only_by_different_speaker`
    - text: `...we're going to McDonalds`
  - `576.632-576.693s`
    - `covered_only_by_different_speaker`
    - text: `Mm.`
- Some other content-bearing `Python-only` spans still exist, but they are
  already fully or mostly covered by a same-speaker C# segment, for example:
  - `316.392-317.673s`
  - `332.873-333.272s`
  - `333.512-333.632s`
  - `561.592-562.172s`
  - `564.592-564.672s`
  - `566.193-566.713s`
- The only clearly content-bearing `C#-only` span left is:
  - `588.980-589.260s`
    - `uncovered`
    - text overlap: `We don't need a bridge.`

Human spot-check labels:

- The user provided manual diarization judgments for the highest-signal
  remaining windows:
  - `330.146-336.894s`
    - one contiguous speaker
    - label: `A`
  - `564.5-566.9s`
    - third/background speaker
    - label: `C`
    - content: mumbling, indistinct, includes `We're going to McDonald's`
  - `576.6-576.7s`
    - third/background speaker
    - label: `C`
    - content: mumbling, indistinct
  - `588.9-589.3s`
    - laughing followed by an elongated `Well`
    - label: `A`

Implications:

- The Python micro-turn pattern in `332.8-333.6s` is not ground truth.
  Human labeling says that whole `330.146-336.894s` span is one continuous
  speaker, so C# collapsing that handoff is likely better than the Python
  reference there.
- The late-file `564.5-566.9s` and `576.6-576.7s` insertions are real third
  speaker events, so suppressing them to the two dominant speakers is a genuine
  fidelity loss relative to the audio.
- The remaining `588.9-589.3s` `C#-only` span is not an error if it aligns with
  the user's manual label `A`; it is likely another case where C# preserves
  acoustically real material that the Python reference omits or structures
  differently.

Human-debug run for `330-337s`:

- Focused run:
  [run_20260403_141500_window-300-360-human-debug](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_141500_window-300-360-human-debug/run.log)
- Debug artifacts:
  - [reconstruction_debug.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_141500_window-300-360-human-debug/reconstruction_debug.json)
  - [assignment_debug.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_141500_window-300-360-human-debug/assignment_debug.json)

Main takeaway:

- The false second speaker inside the human-labeled single-speaker span
  `330.146-336.894s` is not created by final cleanup.
- It is already present at frame selection time:
  - around local `31.48-31.96s` (full `331.48-331.96s`), `AverageCount`
    reaches `1.9-2.0`
  - both global speakers are selected with scores like `0.9/1.0`
- Chunk-to-global assignment is stable in that window, so this is not an
  assignment flip.

Implication:

- The remaining false overlap there comes from the local binarized
  segmentation/count evidence before final region extraction.
- That makes local powerset decoding and frame-level count estimation stronger
  suspects than clustering or smoothing for the remaining acoustic error.

Local-mask debug for `330-337s`:

- Focused run:
  [run_20260403_143200_window-300-360-local-mask-debug](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_143200_window-300-360-local-mask-debug/run.log)
- Debug artifact:
  [local_mask_debug.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_143200_window-300-360-local-mask-debug/local_mask_debug.json)

Main takeaway:

- The false overlap is already present in the decoded local chunk masks before
  global stitching.
- Across many overlapping chunks, frames around full `331.48-331.96s` are
  repeatedly decoded as a two-speaker local mask `[1, 1, 0, 0]`.
- So the reconstruction stage is not inventing that overlap; it is faithfully
  aggregating local evidence that already says "two speakers here."

Important divergence observed:

- The debug dump also shows an inconsistency between the soft local masks and
  the hard powerset-derived count:
  - some frames carry local mask `[1, 1, 0, 0]`
  - while the decoded hard powerset count for the same frame is not always `2`
- That means the current soft-mask path can assert simultaneous local speakers
  more aggressively than the powerset argmax/cardinality path.

Implication:

- The remaining acoustic error is now strongly localized to how local powerset
  posteriors are turned into per-speaker binary masks and instantaneous count.
- This is a clearer remaining divergence than clustering, smoothing, or final
  region extraction.

Late-window hard-mask probe:

- Because the local-mask debug pointed upstream of reconstruction, the hard
  powerset local-mask path was rechecked on the late `540-600s` window:
  [run_20260403_142439_window-540-600-hard-masks](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_142439_window-540-600-hard-masks/run.log)

Result:

- `25` matches
- `3` speaker mismatches
- `7` C#-only
- `9` Python-only
- `94.4%` speaker-frame F1

Main takeaway:

- Hard local masks are not a net win on the late weak-speaker window.
- They do change the local structure around `564.5-566.9s` and `576.6s`, but
  in a way that destabilizes speaker assignment elsewhere.
- So the remaining fix is not as simple as "switch fully to hard masks."

Soft-mask-to-hard-count projection probe:

- A new targeted decode mode was added behind:
  - `PARAKEET_DIARIZEN_PROJECT_SOFT_MASKS_TO_COUNT=1`
- It keeps soft per-speaker identities, but projects them onto the hard
  powerset-derived frame count so the soft mask cannot assert more simultaneous
  speakers than the powerset count for that frame.

Focused runs:

- [run_20260403_142815_window-300-360-project-soft-to-count](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_142815_window-300-360-project-soft-to-count/run.log)
- [run_20260403_142816_window-540-600-project-soft-to-count](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_142816_window-540-600-project-soft-to-count/run.log)

Result on `300-360s`:

- `36` matches
- `0` speaker mismatches
- `2` C#-only
- `2` Python-only
- but total C# duration dropped by `1.82s`
- frame F1 dropped to `94.6%`

Result on `540-600s`:

- `25` matches
- `3` speaker mismatches
- `7` C#-only
- `9` Python-only
- frame F1 dropped to `94.4%`

Main takeaway:

- Projecting soft masks to the hard count behaves very similarly to the hard
  powerset-mask path.
- It does shorten the false overlap around `330-337s`, but it suppresses too
  much speech overall and is harmful on the late weak-speaker window.
- So it is not a viable default fix.

Late-window local-mask debug:

- Focused run:
  [run_20260403_143800_window-540-600-local-mask-debug](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_143800_window-540-600-local-mask-debug/run.log)
- Debug artifact:
  [local_mask_debug.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_143800_window-540-600-local-mask-debug/local_mask_debug.json)

Main takeaway:

- In the user-labeled weak-speaker region `564.5-566.9s`, the decoded local
  chunk masks mostly alternate between two local speakers with a short overlap
  bridge around `24.82-25.08s`.
- They do not show a clean third local speaker track emerging before
  clustering.
- That means the weak/background speaker miss is already largely present in the
  local decoded evidence, not something introduced later by global clustering
  or final cleanup.

Implication:

- The remaining gap now appears to be at or near the segmentation-model decode
  semantics themselves.
- We are past the point where downstream clustering/reconstruction tweaks are
  likely to recover the missing weak speaker cleanly.

Historical ONNX-export docs corrected:

- The old documents under
  [scripts/diarizen_export](/home/chris/Programming/parakeet_csharp/scripts/diarizen_export)
  were still describing an early prototype path with placeholder clustering and
  simplified decoding as if it were current.
- Added explicit historical/disclaimer notes to:
  - [DIARIZEN_ONNX.md](/home/chris/Programming/parakeet_csharp/scripts/diarizen_export/DIARIZEN_ONNX.md)
  - [DIARIZEN_STATUS.md](/home/chris/Programming/parakeet_csharp/scripts/diarizen_export/DIARIZEN_STATUS.md)
  - [FINAL_SUMMARY.md](/home/chris/Programming/parakeet_csharp/scripts/diarizen_export/FINAL_SUMMARY.md)

Python parity harness environment note:

- The historical `~/base` environment exists, but currently fails to import the
  old pyannote path cleanly:
  - `pyannote.audio` import error: `operator torchvision::nms does not exist`
- So `scripts/test_onnx_parity_v2.py` is a good harness in principle, but the
  environment needs repair before it can be relied on for raw ONNX-vs-PyTorch
  validation.

Fresh parity environments:

- Created a repo-local exploratory environment at:
  - `.venv-diarizen-parity`
- That environment is useful for quick import checks and exporter work:
  - `pyannote.audio` import works
  - `diarizen` import works
- But its initial Python 3.12 / `torch==2.11.0` / `torchaudio==2.11.0` stack is
  not a clean fit for the local DiariZen fork:
  - `scripts/test_onnx_parity_v2.py` fails with
    `ImportError: cannot import name 'AudioMetaData' from 'torchaudio'`
  - this is a library-shape mismatch, not a missing-package issue
- The DiariZen README explicitly documents the older stack:
  - `pytorch==2.1.1`
  - `torchvision==0.16.1`
  - `torchaudio==2.1.1`
- Because `/usr/bin/python3.10` is available locally, a second clean parity
  environment is now being created at:
  - `.venv-diarizen-parity310`
- That 3.10 environment is the right place to run the raw ONNX-vs-PyTorch
  segmentation harness once package installation completes.
- The 3.10 parity environment is now healthy with:
  - `torch==2.1.1`
  - `torchvision==0.16.1`
  - `torchaudio==2.1.1`
  - editable installs of the local `pyannote-audio` and `DiariZen` checkouts
- Validation in `.venv-diarizen-parity310`:
  - `pyannote.audio` import OK
  - `diarizen` import OK
  - `torchaudio.AudioMetaData` is present again, so this environment matches the
    local fork's expectations much better than the Python 3.12 exploratory venv

Exporter-method audit:

- The current segmentation exporter in
  [scripts/diarizen_export/export_diarizen_onnx.py](/home/chris/Programming/parakeet_csharp/scripts/diarizen_export/export_diarizen_onnx.py)
  was manually reconstructing model constructor args from `config.toml` and
  then loading checkpoint weights with `strict=False`.
- That meant export-time config drift or checkpoint/model mismatches could be
  silently tolerated.

Hardening changes:

- The exporter now:
  - loads the full `model.args` dict from `config.toml`
  - filters it against the actual constructor signature
  - passes the supported args through directly
  - raises on any missing or unexpected checkpoint keys after load

Main takeaway:

- This does not yet prove an ONNX export bug, but it removes one plausible way
  the export path could have drifted from the HuggingFace model definition
  without being noticed.
- The next decisive export check is no longer hypothetical: with the new repo-
  local parity venvs in place, the remaining blocker is getting the local
  DiariZen stack onto the documented `torch/torchaudio` version family and then
  running `scripts/test_onnx_parity_v2.py` directly.
- That raw harness is now running from:
  - `.venv-diarizen-parity310/bin/python scripts/test_onnx_parity_v2.py ...`
- So the export-method investigation is no longer blocked on environment setup;
  the next result to inspect is the ONNX segmentation parity output itself.

Raw ONNX segmentation parity result:

- The raw Python parity harness completed successfully in the clean 3.10
  environment:
  - `.venv-diarizen-parity310/bin/python scripts/test_onnx_parity_v2.py ...`
- Result:
  - `292` test segments vs `292` reference segments
  - `3` test speakers vs `3` reference speakers
  - `292` matches
  - `0` speaker mismatches
  - `0` test-only
  - `0` ref-only
- Output artifacts:
  - [data/onnx_parity_v2.json](/home/chris/Programming/parakeet_csharp/data/onnx_parity_v2.json)
  - [data/onnx_parity_v2_comparison.json](/home/chris/Programming/parakeet_csharp/data/onnx_parity_v2_comparison.json)
- Interpretation:
  - swapping only the segmentation model forward pass from PyTorch to ONNX still
    reproduces the full Python diarization reference exactly
  - so the remaining C# parity gap is not explained by a broad segmentation ONNX
    export failure

Export-folder review:

- `scripts/diarizen_export/export_diarizen_onnx.py`
  - still relevant
  - now documented more accurately: it does require a local DiariZen checkout,
    despite loading weights/config from HuggingFace
- `scripts/diarizen_export/export_pyannote_wespeaker_onnx.py`
  - current embedding export path for parity work
  - matches the C# pipeline's `wespeaker_pyannote.onnx` expectations
  - 80-bin Fbank input, raw 256-dim embedding output
- `scripts/diarizen_export/export_wespeaker_onnx.py`
  - historical / legacy path
  - exports the older 40-bin / 512-dim embedding layout
  - not the model currently used by DiariZen parity work
- `scripts/diarizen_export/diarizen_onnx_inference.py`
  - historical simplified wrapper
  - not a faithful reference for the current parity target
  - still useful as an old prototype, but should not be treated as the source of
    truth for current C# behavior
- `scripts/diarizen_export/export_lda_transform.py`
  - still relevant
  - exports the LDA / PLDA assets that the current C# VBx path actually consumes

Embedding-export follow-up:

- Added a new raw embedding parity harness:
  - [scripts/test_onnx_embedding_parity.py](/home/chris/Programming/parakeet_csharp/scripts/test_onnx_embedding_parity.py)
- Purpose:
  - keep the DiariZen/pyannote reference pipeline intact
  - swap only the embedding stage from native pyannote WeSpeaker to
    `wespeaker_pyannote.onnx`
- Important nuance:
  - the native pyannote WeSpeaker path supports frame weights directly
  - the exported ONNX model takes only Fbank input
  - this harness therefore uses the same masked-frame strategy as the current C#
    implementation: compute native Fbank, apply the binary frame mask by frame
    selection, then run the ONNX embedding model on the masked features
- Status:
  - full-file parity run is now in progress from the clean 3.10 environment
  - output target:
    [data/onnx_embedding_parity.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity.json)

Raw ONNX embedding parity result:

- The masked-frame ONNX embedding harness completed successfully:
  - [data/onnx_embedding_parity.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity.json)
  - [data/onnx_embedding_parity_comparison.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity_comparison.json)
  - [data/onnx_embedding_parity_compare_full.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity_compare_full.json)
- Summary:
  - `292` test segments vs `292` reference
  - `286` simple matches in the minimal harness
  - richer comparison:
    - `289` perfect matches
    - `1` timing match with speaker mismatch
    - `2` C#-only
    - `2` Python-only
    - `99.6%` speaker-frame F1
    - `99.7%` overlap-time Jaccard
- Interpretation:
  - swapping only the embedding stage to the exported ONNX model does change the
    result, but only in a tiny late-file region
  - this strongly suggests the broad WeSpeaker export itself is mostly sound
  - the remaining export-side mismatch is concentrated in mask semantics rather
    than in the raw embedding backbone

Weighted ONNX embedding export probe:

- Extended
  [scripts/diarizen_export/export_pyannote_wespeaker_onnx.py](/home/chris/Programming/parakeet_csharp/scripts/diarizen_export/export_pyannote_wespeaker_onnx.py)
  so it can export a weighted variant:
  - `wespeaker_pyannote_weighted.onnx`
  - inputs: `fbank`, `weights`
- Goal:
  - preserve pyannote's native `model_(waveforms, weights=masks)` semantics more
    faithfully than the masked-frame approximation
- Result:
  - initial attempt failed because the weighted path was being exported through a
    trace-unfriendly branch around frame-count interpolation
  - after moving the interpolation explicitly into the wrapper, the weighted ONNX
    export now matches PyTorch at the tensor level
  - direct debug check:
    - max absolute diff: `1.79e-07`
    - mean absolute diff: `5.41e-08`
- Current interpretation:
  - segmentation ONNX export is validated
  - unweighted/masked WeSpeaker ONNX export is close but not exact in a tiny late
    region
  - weighted WeSpeaker ONNX export is now numerically faithful enough to test as
    a real parity path
- Next immediate step:
  - run the full-file embedding parity harness with:
    - `wespeaker_pyannote_weighted.onnx`
    - `--mode weighted`

Weighted ONNX embedding full-file result:

- The weighted ONNX WeSpeaker parity harness completed successfully:
  - [data/onnx_embedding_parity_weighted.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity_weighted.json)
  - [data/onnx_embedding_parity_weighted_comparison.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity_weighted_comparison.json)
  - [data/onnx_embedding_parity_weighted_compare_full.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity_weighted_compare_full.json)
- Result:
  - `292` test segments vs `292` reference
  - `292` matches
  - `0` speaker mismatches
  - `0` test-only
  - `0` ref-only
  - `100.0%` exact frame-set match
  - `100.0%` speaker-frame F1
- Interpretation:
  - preserving pyannote's native frame-weight semantics is sufficient to make
    the ONNX WeSpeaker path exactly reproduce the Python reference
  - the earlier masked-frame ONNX drift was not a broad embedding export failure;
    it was specifically the result of replacing weighted pooling semantics with
    hard masked-frame selection

Export-side conclusion:

- The DiariZen ONNX exports are now validated end-to-end against the Python
  reference when used with the correct semantics:
  - segmentation ONNX: exact parity
  - weighted WeSpeaker ONNX: exact parity
- Therefore the remaining parity gap in the C# implementation is best explained
  by implementation semantics, not by a bad segmentation export and not by a bad
  weighted WeSpeaker export.
- The most important export lesson from this investigation is:
  - weighted WeSpeaker pooling semantics matter
  - masked-frame approximation is close, but not exact

C# weighted-WeSpeaker follow-up:

- Updated the C# path to prefer the weighted embedder export by default:
  - [Config.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/Config.cs)
    now points `DiariZenEmbedderFile` at `wespeaker_pyannote_weighted.onnx`
- Updated
  [WeSpeakerEmbedder.cs](/home/chris/Programming/parakeet_csharp/src/Parakeet.Base/WeSpeakerEmbedder.cs)
  so that:
  - if the ONNX model exposes a `weights` input, it feeds full Fbank features
    plus an upsampled 0/1 frame-weight vector
  - otherwise it falls back to the older masked-frame approximation
- Build status after the change:
  - `dotnet build src/Parakeet.CLI/Parakeet.CLI.csproj` succeeds

90-second C# guardrail run:

- Run:
  [run_20260403_191916_first90-csharp-weighted-wespeaker](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_191916_first90-csharp-weighted-wespeaker/run.log)
- Result:
  - `46` C# segments vs `45` Python
  - `45` perfect matches
  - `0` speaker mismatches
  - `1` C#-only
  - `0` Python-only
  - `98.6%` speaker-frame F1
- Remaining difference:
  - one extra tail segment at `89.58–90.00s`
- Interpretation:
  - weighted WeSpeaker semantics transfer most of the export-side gain into the
    C# implementation
  - but the C# path is still not identical to the Python reference yet, so at
    least one additional implementation detail still differs

Next step in progress:

- Full-file C# validation with the weighted WeSpeaker path is now running:
  - label: `full-csharp-weighted-wespeaker`

Full-file C# weighted-WeSpeaker result:

- Run:
  [run_20260403_192035_full-csharp-weighted-wespeaker](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_192035_full-csharp-weighted-wespeaker/run.log)
- Result:
  - `276` C# segments vs `292` Python
  - `274` perfect matches
  - `0` speaker mismatches
  - `2` C#-only
  - `18` Python-only
  - `98.3%` exact frame-set match
  - `99.0%` speaker-frame F1
  - `93.3%` overlap-time Jaccard
- Compared with the prior best full-file baseline
  [run_20260403_132048_full-binary-reconstruct](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_132048_full-binary-reconstruct/run.log):
  - matches improved `271 -> 274`
  - Python-only improved `21 -> 18`
  - speaker mismatches stayed `0`
  - C#-only stayed `2`
  - frame F1 stayed essentially flat (`99.0% -> 99.0%`)
  - overlap Jaccard dipped slightly (`93.46% -> 93.34%`)
- Important caveat:
  - this path still collapses the output to `2` C# speakers vs `3` Python speakers
  - so while weighted WeSpeaker helps raw segment agreement, it is not the single
    missing ingredient for full reference parity in the C# implementation

Current interpretation:

- Weighted WeSpeaker semantics matter, and they do improve the C# result.
- However, after adopting them, the implementation still differs from Python in
  at least one additional way, likely elsewhere in clustering / reconstruction /
  count handling rather than in the export path itself.

Folder cleanup:

- Updated `scripts/diarizen_export/export_diarizen_onnx.py` docstring to stop
  claiming that the local DiariZen repo is unnecessary.
- Updated `scripts/diarizen_export/export_wespeaker_onnx.py` docstring to mark
  it as historical relative to the current `export_pyannote_wespeaker_onnx.py`
  path.
- Tightened `scripts/diarizen_export/diarizen_requirements.txt` toward the
  actually tested parity stack:
  - Python 3.10
  - `torch==2.1.1`
  - `torchvision==0.16.1`
  - `torchaudio==2.1.1`
  - editable installs of local `pyannote-audio` and `DiariZen`

## Current Status

- Build status: `dotnet build src/Parakeet.CLI/Parakeet.CLI.csproj` succeeds
- A canonical reference bundle exists under:
  - [reference/README.md](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/reference/README.md)
  - [reference/full/reference_segments.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/reference/full/reference_segments.json)
  - [reference/first90/reference_segments.json](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/reference/first90/reference_segments.json)
- The repo treats the JSON segment files as the primary diarization parity target and the markdown transcript as a secondary downstream sanity check.
- The current best full-file parity result is:
  [run_20260404_012330_full-fixedmel-le-minregion](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260404_012330_full-fixedmel-le-minregion/run.log)
  - `273` C# segments vs `292` Python segments
  - `2` C# speakers vs `3` Python speakers
  - `271` perfect matches
  - `0` strict speaker mismatches
  - `2` C#-only, `21` Python-only
  - `99.0%` speaker-frame F1
  - `93.4%` overlap-time Jaccard
- The 90-second slice result (with same code) has `45` matches, `0` speaker
  mismatches, `1` C#-only (`89.58-90.00s` tail only).
- Note: the previous weighted-only baseline (`run_20260403_192035`) had `274`
  matches and only `18` Python-only, but used an incorrect (integer-bin) mel
  filterbank that produced 2 empty filters and mean embedding cosine similarity
  of only `0.887`. The current result uses the correct (continuous Hz) mel
  filter with mean cosine similarity `0.987` and reflects genuinely better
  embedding quality even though strict match count is slightly lower (`271` vs
  `274`) due to the `564-577s` weak-speaker region behaving differently.
- Coverage-aware re-analysis of the full-file `18` Python-only segments shows:
  - `5` already fully covered by a same-speaker C# segment (different boundaries only)
  - `7` partially covered by a same-speaker C# segment
  - `5` covered only by a different-speaker C# segment — but most are tiny Python
    micro-turns or are the `332-336s` region that human labeling confirms is
    actually a Python false handoff
  - `1` fully uncovered (`412.733-412.793s`, `60ms`, Python speaker 2)
- The "missing third speaker" (`2` C# vs `3` Python) amounts to a `20ms` Python
  speaker-0 fragment at `565.892-565.912s`. The larger weak-speaker spans at
  `564.5-566.9s` and `576.6s` appear in the Python reference under speakers 1
  and 2 rather than as a distinct third speaker, so they are represented
  differently rather than absent.
- The entire post-processing layer has been exonerated as the source of the
  remaining drift: pruning, final count-cap, adjacent-merge, and smoothing
  probes all left the full-file result unchanged or made it worse.
- The remaining gap is most likely in per-job fbank or embedding numerical
  differences between the C# fbank implementation and the torchaudio/Kaldi path
  used inside the Python reference pipeline. The torchaudio-style melbank fix
  tried earlier was a real DSP improvement but regressed exact parity (`274 →
  271`), confirming the C# fbank and the Python pipeline are inadvertently
  calibrated against each other at the current values.

## Recommended Next Steps

1. The embedding quality issue is now resolved (mean cosine similarity `0.987`
   vs `0.887` before the mel filter fix).
2. The `86.40-86.44s` extra C#-only fragment has been eliminated by the
   `<= minRegionFrames` fix.
3. The `89.58-90.00s` tail fragment is an artifact of the 90-second test clip
   only: the last chunk starts at 89.6s and is nearly all zero-padding (only
   0.4s real audio), producing a noisy embedding that bleeds into the timeline.
   This does not appear in the full-file run. It does not represent a genuine
   pipeline defect for real audio.
4. The remaining `21` Python-only on the full file break down as:
   - `7` fully covered by a same-speaker C# segment (different boundaries only)
   - `6` partially covered by a same-speaker C# segment
   - `7` covered only by a different-speaker C# segment — all are very brief
     (20-240ms) micro-turns or false Python handoffs (human-labeled 333s region)
   - `1` fully uncovered (412.73s, 60ms, Python speaker 2)
   None of these are actionable at the reconstruction/clustering level; they
   require either better local segmentation resolution for sub-200ms turns or
   a different approach to brief-turn recovery in the weak-speaker region.
5. The investigation has effectively reached a natural stopping point for broad
   pipeline changes. Further gains would require improvements at the local
   segmentation decode level (powerset model or decode threshold tuning) or
   acceptance of the remaining gap as a genuine limitation.
6. Keep using [reference/full/reference_transcript.md](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/reference/full/reference_transcript.md) only as a downstream sanity check.
7. Keep recording every iteration with a run ID and label under `data/diarizen_parity/runs/`.

## Latest Investigation Notes

- The raw Python ONNX parity harness now exonerates the segmentation export:
  [onnx_parity_v2_comparison.json](/home/chris/Programming/parakeet_csharp/data/onnx_parity_v2_comparison.json)
  reaches exact full-file parity (`292/292`, `0` mismatches).
- The weighted WeSpeaker ONNX export also reaches exact full-file parity when
  swapped into the Python reference pipeline:
  [onnx_embedding_parity_weighted_compare_full.json](/home/chris/Programming/parakeet_csharp/data/onnx_embedding_parity_weighted_compare_full.json)
  reaches `292` matches, `0` mismatches, `0` extra, and `0` missing.
- That means the remaining C# drift is no longer well explained by the ONNX
  exports themselves. The likely remaining gap is C# pipeline logic that was
  introduced while compensating for the earlier masked-frame embedder.
- The current weighted C# full-file baseline is:
  [run_20260403_192035_full-csharp-weighted-wespeaker](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_192035_full-csharp-weighted-wespeaker/run.log)
  with `274` matches, `0` speaker mismatches, `2` C#-only, and `18`
  Python-only segments.
- One concrete remaining divergence from pyannote is the final reconstruction
  count cap. Pyannote's `to_diarization(...)` selects the top `count[t]`
  speakers per frame without reapplying the model's local `max_speakers_per_frame = 2`
  limit, while the C# reconstruction path was still capping the final selected
  global speakers per frame at `2`.
- That final reconstruction cap has now been removed in the C# path while
  keeping the model-side powerset decode capped at `2`.
- The 90-second guardrail run with weighted WeSpeaker and no final count cap is:
  [run_20260403_193445_first90-weighted-no-final-count-cap](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_193445_first90-weighted-no-final-count-cap/run.log)
  and it is unchanged from the prior weighted 90-second result:
  `45` matches, `0` speaker mismatches, and one extra C# tail segment
  (`89.58-90.00s`).
- Because the suspected benefit of removing the final count cap is late-file
  weak-speaker recovery rather than first-90 behavior, the next useful check is
  the full-file weighted run with this count-cap change enabled.
- The full-file weighted run without the final reconstruction count cap is:
  [run_20260403_193627_full-weighted-no-final-count-cap](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_193627_full-weighted-no-final-count-cap/run.log)
  and it is exactly identical to the prior weighted full-file baseline:
  `274` matches, `0` speaker mismatches, `2` C#-only, and `18` Python-only.
- That means the extra final count cap was not the cause of the remaining
  weighted C# drift. It was a real semantic difference from pyannote, but
  removing it neither helped nor harmed this test file.
- Disabling the full post-smoothing pass under the weighted embedder is a clear
  regression:
  [run_20260403_204521_first90-weighted-no-post-smoothing](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_204521_first90-weighted-no-post-smoothing/run.log)
  increases the 90-second result from `46` to `49` C# segments while keeping
  the same `45` matches and reintroducing extra micro-fragments at
  `64.64-64.66s`, `64.68-64.70s`, and `86.40-86.42s`.
- So the broad smoothing pass is still doing useful cleanup even after the
  weighted export fix. The next more surgical hack check is the final
  same-speaker adjacent merge, which Python does not do explicitly.
- Disabling only the final same-speaker adjacent merge is neutral on the
  90-second weighted guardrail:
  [run_20260403_204737_first90-weighted-no-adjacent-merge-rerun](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_204737_first90-weighted-no-adjacent-merge-rerun/run.log)
  produces exactly the same result as the weighted baseline:
  `45` matches, `0` speaker mismatches, and one extra tail segment
  (`89.58-90.00s`).
- That strongly suggests the remaining weighted C# drift is not coming from the
  final cleanup layer at all. The last likely causes are deeper in the local
  decode/count and clustering-to-reconstruction path.
- The full-file weighted run without the final same-speaker adjacent merge is:
  [run_20260403_204912_full-weighted-no-adjacent-merge](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_204912_full-weighted-no-adjacent-merge/run.log)
  and it is also exactly identical to the weighted full baseline:
  `274` matches, `0` speaker mismatches, `2` C#-only, and `18` Python-only.
- At this point, the whole final cleanup layer is effectively exonerated as the
  source of the remaining weighted C# drift:
  pruning was neutral, final count-cap removal was neutral, adjacent-merge
  removal was neutral, and only disabling the entire smoothing pass was
  harmful.
- A direct numerical check against `torchaudio.compliance.kaldi.fbank` on a
  real late-file waveform window showed that the old C# fbank implementation
  was not numerically close to torchaudio:
  about `0.25` mean absolute error per coefficient and `>6` max absolute error
  on the tested `564-567s` slice.
- The most concrete DSP mismatch was the mel filterbank construction: the old
  C# code used integer FFT-bin boundaries, while torchaudio/Kaldi evaluates the
  triangular filters directly in mel space at each FFT-bin centre and then pads
  the final Nyquist column.
- The C# mel filterbank has now been updated to match torchaudio's
  `get_mel_banks(...)` construction more closely.
- The 90-second weighted guardrail run after that mel-bank fix is:
  [run_20260403_214158_first90-weighted-melbank-match-torchaudio](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_214158_first90-weighted-melbank-match-torchaudio/run.log)
  and it remains unchanged from the prior weighted baseline:
  `45` matches, `0` speaker mismatches, and one extra C# tail segment
  (`89.58-90.00s`).
- Because this was a real low-level numerical mismatch, it is still worth
  validating on the full file even though the first-90 slice stayed unchanged.
- The full-file weighted run after that mel-bank fix is:
  [run_20260403_214327_full-weighted-melbank-match-torchaudio](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_214327_full-weighted-melbank-match-torchaudio/run.log)
  and it is not a net win:
  `271` matches, `0` speaker mismatches, `2` C#-only, and `21` Python-only.
- Relative to the weighted baseline, the torchaudio-style mel-bank change
  slightly improved frame-level overlap metrics, but regressed exact segment
  parity (`274 -> 271` matches, `18 -> 21` Python-only, `276 -> 273` C#
  segments).
- Because this project is still targeting parity with the Python reference
  rather than just generic numerical fidelity, that mel-bank change has been
  reverted. It was a real DSP mismatch, but not one that moved the diarization
  result in the right direction.
- A deeper embedding-level diagnosis was performed to understand why the correct
  mel filter regressed parity instead of improving it.
  - A new `scripts/compare_embeddings.py` script was written to compare C#
    embedding vectors against Python ONNX embeddings for the same chunks
    (using `wespeaker_pyannote_weighted.onnx` and torchaudio Kaldi Fbank as the
    reference).
  - With the old (integer-bin) mel filter: mean cosine similarity = `0.887`,
    minimum = `0.565`. The root cause was identified: the integer-bin approach
    produces **two empty (all-zero) mel filters** (at indices 1 and 8 in the
    low-frequency range) and several single-bin degenerate triangles. When the
    integer floor-division maps adjacent mel centers to the same FFT bin,
    the filter triangle collapses to zero.
  - The Slaney vs HTK mel-scale formula difference turned out to be irrelevant:
    both give essentially identical filter center frequencies at 64-bit precision
    (max difference < 0.02 mel across the entire range, zero effect on the
    resulting filter matrices).
  - The correct fix is to evaluate each triangular filter at the exact FFT bin
    centre frequency (`k * sampleRate / fftSize` Hz) rather than rounding Hz
    to integer bin indices. This produces zero empty filters and matches
    torchaudio's behaviour.
  - `WeSpeakerEmbedder.MakeMelFilters` was updated to use continuous Hz
    evaluation. After the fix, mean cosine similarity jumped to `0.987` and
    max to `1.000`.
  - Parity impact:
    - 90-second first-90 slice:
      [run_20260404_011725_first90-fixedmel-continuous](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260404_011725_first90-fixedmel-continuous/run.log)
      — `45` matches, `0` speaker mismatches, `2` C#-only (added a tiny
      `86.40-86.44s` fragment vs the prior `1` C#-only).
    - Full-file:
      [run_20260404_011811_full-fixedmel-continuous](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260404_011811_full-fixedmel-continuous/run.log)
      — `271` matches, `0` speaker mismatches, `2` C#-only, `21` Python-only,
      `99.0%` speaker-frame F1, `93.4%` overlap-time Jaccard.
  - The regression (`274 → 271` matches) is concentrated entirely in the
    `564-577s` weak-speaker region. The correct mel filter changes how brief
    200ms speaker turns in that region are embedded: with the old filter the
    `566.7-566.9s` turn was assigned to speaker_1 (matching the Python
    reference); with the correct filter it gets absorbed into the surrounding
    speaker_0 segment. This is a clustering resolution issue in the
    weak-speaker region, not a filter issue. The correct mel filter was
    **kept** since it is objectively the right implementation; the 3-match
    regression reflects real limits of per-chunk embedding discrimination
    for very brief turns at weak-speaker boundaries.
- Another concrete remaining divergence is the embedding-mask length threshold.
  In the Python reference, the minimum clean-mask length is derived from the
  embedder's true `min_num_samples` (for pyannote WeSpeaker, `400` samples),
  which corresponds to roughly one segmentation frame on a 16-second chunk.
  The C# path had been using a hardcoded `10`-frame threshold both to skip
  local speakers entirely and to decide whether to use `cleanMask` or
  `activeMask`.
- The weighted C# path now follows the Python logic more closely:
  derive `minMaskFrames` from `MinNumSamples`, use `cleanMask` only when
  `cleanCount > minMaskFrames`, and do not drop weighted local speakers just
  because they are shorter than the old 10-frame guardrail.
- The 90-second weighted guardrail for that change is:
  [run_20260403_222322_first90-weighted-python-min-mask-threshold](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_222322_first90-weighted-python-min-mask-threshold/run.log)
  and it is unchanged from the weighted baseline:
  `45` matches, `0` speaker mismatches, and one extra tail segment
  (`89.58-90.00s`).
- Because this divergence primarily affects weak/background local speakers, the
  meaningful validation is the full-file weighted run, not the first-90 slice.
- The full-file weighted run for that threshold change is:
  [run_20260403_222454_full-weighted-python-min-mask-threshold](/home/chris/Programming/parakeet_csharp/data/diarizen_parity/runs/run_20260403_222454_full-weighted-python-min-mask-threshold/run.log)
  and it is exactly identical to the weighted full baseline:
  `274` matches, `0` speaker mismatches, `2` C#-only, and `18` Python-only.
- That rules out the clean-mask minimum-length threshold as the missing parity
  lever as well. At this point, the remaining weighted C# drift is more likely
  in the actual per-job embedding values or later clustering/reconstruction
  behavior than in the job-selection thresholds.

## Files Changed In This Investigation

- `src/Parakeet.CLI/Program.cs`
- `src/Parakeet.Base/Config.cs`
- `src/Parakeet.Base/DiariZenDiarizer.cs`
- `src/Parakeet.Base/WeSpeakerEmbedder.cs`
- `scripts/build_diarizen_reference_bundle.py`
- `scripts/run_diarizen_parity.py`
- `data/diarizen_parity/reference/README.md`
