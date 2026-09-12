# cudnn_conv_algo_search across the app's CUDA sessions (#191)

#190 found that ORT's CUDA EP defaults `cudnn_conv_algo_search` to **EXHAUSTIVE**, which
benchmarks every cuDNN algorithm the first time it sees a conv input **shape** — and its algo
cache does not retain every shape, so it re-tunes on every shape *change*. Kokoro paid 2.93x for
that, because every synthesis call is a different chunk length.

Only Kokoro opted in. This log asks the obvious follow-up: which of the app's other CUDA sessions
are paying the same tax, and which would be hurt by "fixing" them.

## Run 1 — 2026-09-11 18:05 — a generic sweep, and why it was invalid

Generic probe: for each model, fill every symbolic axis with a scaled size, cycle five sizes, time
EXHAUSTIVE vs DEFAULT.

⚠ **Half the table was measuring a degenerate graph.** The generic shape-filler set every
`length` input to **0** — parakeet, indicconformer and sortformer all take an explicit length
alongside their features — and gave VoxLingua 640 audio samples, i.e. 40 ms. An encoder told its
input is zero-length does not do the work being timed. `chatterbox flow_encoder` came out at
0.87x (rejected) under those shapes and 1.17x (accepted) under real ones, which is the whole
argument for redoing it.

Generic shape inference is not good enough for this question; shapes are hand-built per model
from here on, with every length input carrying the matching count.

## Run 2 — 2026-09-11 18:15 — the real table

Realistic shapes, five sizes cycled per model (so the shape CHANGES, which is the entire
mechanism — timing one repeated shape measures the case where EXHAUSTIVE amortizes and reports
the opposite answer, which is how `tests/KokoroPerf` misled in #190). RTX 3090, ORT 1.29:

```
model                             EXHAUSTIVE    DEFAULT    gain   verdict
parakeet encoder                      401.7ms   1319.4ms   0.30x  EXHAUSTIVE wins
indicconformer encoder                625.4ms   1129.0ms   0.55x  EXHAUSTIVE wins
sortformer                            321.3ms    547.7ms   0.59x  EXHAUSTIVE wins
diarizen segmentation                 218.5ms    270.8ms   0.81x  EXHAUSTIVE wins
chatterbox flow_encoder                43.9ms     37.7ms   1.17x  DEFAULT wins
chatterbox mel2wav (vocoder)          978.0ms    703.5ms   1.39x  DEFAULT wins
voxlingua107 LID                      161.3ms     92.3ms   1.75x  DEFAULT wins
chatterbox cfm_estimator              407.1ms    187.3ms   2.17x  DEFAULT wins
diarizen wespeaker                    276.7ms     89.6ms   3.09x  DEFAULT wins
```

**The split is architectural, not incidental.** Conformer/transformer ASR encoders — depthwise
separable convs with many distinct shapes — want EXHAUSTIVE and are hurt badly without it
(Parakeet is **3.3x slower** on DEFAULT). Plain CNN stacks, vocoders and speaker-embedding nets
want DEFAULT.

⚠ This retroactively justifies #190's per-caller design. A global flip — the tempting reading of
"Kokoro got 1.55x from this" — would have made ASR transcription over three times slower.

## Run 3 — 2026-09-11 18:25 — does it change the numbers?

Same inputs, both settings:

```
model                          max|diff|     rel L2    cosine
diarizen wespeaker             5.83e-04   2.81e-03    0.999996428
chatterbox mel2wav (vocoder)   1.00e-02   3.05e-03    0.999995360
voxlingua107 LID (logits)      1.09e-03   3.93e-05    0.999999999
chatterbox cfm_estimator       3.12e-02   1.07e-03    0.999999434
```

Ordinary fp32 conv-algorithm variation. For scale: TF32, which the repo already found compounds
into *audible* noise through OmniVoice's diffusion loop, is ~1e-2 — an order of magnitude larger
than this.

## Taken, and deliberately not taken

**Taken** — one-shot graphs, large gain, verified benign:

- `WeSpeakerEmbedder` (3.09x). Runs once per speaker segment, so a long recording makes many
  calls, each a different number of frames. Embedding cosine 0.999996, far inside what clustering
  thresholds care about.
- `VoxLinguaLid` (1.75x). Once per file, each a different clip length. Logits effectively
  identical (cosine 1.000000000).

**Not taken:**

- ⚠ `chatterbox cfm_estimator` (2.17x) is an **iterative** CFM solver, called ~32 times per
  synthesis, so a per-step difference compounds — precisely the failure mode TF32 produced in
  OmniVoice. A single-step numerical check is not evidence about the integrated result. Needs a
  real synthesis and a listen before adopting.
- `chatterbox mel2wav` (1.39x) and `flow_encoder` (1.17x) are the **fallback** split graphs; the
  path that normally runs is the merged `conditional_decoder_loop.onnx`, which was not measured
  and contains the same iterative loop. Changing the fallback alone buys almost nothing.
- The four EXHAUSTIVE-wins models keep ORT's default, which is already what they get.

⚠ Measured with random inputs. That is sound for throughput and for the conv-algorithm delta, but
it is not a quality check on real audio. The two changes taken are one-shot graphs whose outputs
were compared directly; anything iterative needs the real thing.

## Run 4 — 2026-09-11 18:20 — Chatterbox (#193): the merged graph says the opposite

#192 measured the **split** vocoder graphs and deliberately refused to act on them, because the
path that normally runs is the merged `conditional_decoder_loop.onnx` (`supports_batched=true`,
so the split graphs never load). That caution was correct.

Driven through the real pipeline — a real voice prompt, four utterances of differing length, the
speech tokens generated **once** and fed to both vocoder instances so the acoustic LM cannot
confound the comparison. Only the vocoder's session option differs. RTX 3090 / ORT 1.29:

```
speech token counts: 321, 375, 283, 370
  EXHAUSTIVE   2154.1 ms
  DEFAULT      2602.1 ms    0.83x   ← SLOWER
```

⚠ **The merged graph prefers EXHAUSTIVE, while its own components preferred DEFAULT**
(`cfm_estimator` 2.17x, `mel2wav` 1.39x, `flow_encoder` 1.17x). Wrapping the CFM ODE in an ONNX
`Loop` changes which shapes the convs see and how often they change, so a component microbenchmark
does not predict the assembled graph. Acting on the split-graph numbers would have made Chatterbox
synthesis ~20% slower.

### The compounding was real too

Same speech tokens, both settings, audio out:

```
utt0  len  68160   max|diff| 2.40e-02   relRMS 1.67e-02
utt1  len 120000   max|diff| 1.72e-01   relRMS 4.05e-02
utt2  len  31680   max|diff| 3.68e-02   relRMS 2.45e-02
utt3  len 115200   max|diff| 3.48e-02   relRMS 1.59e-02
```

A single `cfm_estimator` step differed by rel L2 **1.07e-3** (Run 3). Integrated over the ~32 ODE
steps the output differs by **1.6e-2 to 4.1e-2** — roughly 40x amplification, which is exactly the
compounding the TF32/OmniVoice precedent warned about. So even had DEFAULT been faster here, it
would have needed a listening test rather than a numeric wave-through.

**Verdict: no change.** Chatterbox keeps ORT's EXHAUSTIVE default, which is already what it gets.

The general lesson, now measured twice in this log: **a conv-algorithm setting is a property of the
assembled graph and its shape traffic, not of a model or a layer type.** Component benchmarks
(Run 2's split graphs) and degenerate shapes (Run 1) both produced confident wrong answers.
