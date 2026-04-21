# DeepFilterNet3 Selective Denoising — Investigation

Goal: apply DFN3 only when it measurably improves ASR transcript quality. Denoising
is an ASR preprocessing step; signal-processing cleanliness is not the objective.

## Run 1 — 2026-04-20 18:10 (lsnr-based wet/dry blend prototype)

Question: does exposing DFN3's own per-frame `lsnr` output and blending
`α·clean + (1−α)·denoised` avoid the quality drop on mostly-clean audio?

File: `~/Programming/test_audio/en-US/en-US_sample_01.wav` (600 s, 16 kHz, telephone-quality conversation).

| Variant | Words vs baseline | Seq-sim |
|---|---|---|
| No denoiser (truth) | 100.0 % | 1.000 |
| α=1 full bypass (resample round-trip only) | 99.9 % | 0.930 |
| Blend default (-15→35 dB) | 90.5 % | 0.861 |
| Blend tight (-5→20 dB) | 92.2 % | 0.832 |
| No blend (old DFN3) | **57.4 %** | **0.594** |

Finding: old DFN3 behaviour destroyed ~43 % of words on this clean file. Per-frame
lsnr blend recovers to ~91 %. Resampling round-trip (16k → 48k → 16k) accounts for
~7 % of remaining variance — a floor you can't beat without skipping the resample.

Implication: file-level bypass is needed to eliminate the resample floor on files
that don't need denoising at all.

## Run 2 — 2026-04-20 18:25 (file-level bypass via lsnr probe)

Question: can a 30 s prefix probe on DFN3's encoder (lsnr only, no decoders) reliably
predict whether to skip the full pipeline?

| Variant | Probe cost | Result on en-US_sample_01 |
|---|---|---|
| `--dfn3-bypass-alpha 1.0` (disabled) | 0 | runs full DFN3 + blend (91 % of baseline) |
| `--dfn3-bypass-alpha 0.9` (default) | 1.6 s probe | doesn't bypass (file-wide mean-α = 0.54) |
| `--dfn3-bypass-alpha 0.5` (forced) | 1.6 s probe | bypasses — **byte-identical to baseline** |

Finding: the short-circuit mechanics work (bypassed output MD5 matches no-denoiser).
But the gating signal is wrong for the stated goal — lsnr measures signal cleanliness,
not "will ASR benefit". Telephone audio got scored lsnr-noisy because DFN3 was trained
on wideband material and sees narrowband as suspicious, even though ASR doesn't care.

Implication: swap the probe off lsnr onto signals that track ASR benefit directly.

## Run 3 — 2026-04-20 18:58 (bandwidth + noise-floor probe)

Question: do cheap FFT-based heuristics (HF energy ratio, noise-floor RMS) gate
the bypass decision better than lsnr?

New probe in `AudioSignalProbe.cs`. Bypass when `HF-ratio ≤ 0.015 OR
noise-floor ≤ -55 dBFS`. Pure DSP, ~500 ms for a 30 s window.

Results on test corpus:

| File | HF-ratio | noise-floor | Decision |
|---|---|---|---|
| `en-US_sample_01.wav` | 0.00 % | -57.4 dBFS | bypass (narrowband + quiet) |
| `fr-FR_sample_01.wav` | 0.35 % | -73.1 dBFS | bypass (narrowband + quiet) |
| `de-DE_sample_02.wav` | 0.00 % | -61.2 dBFS | bypass (narrowband + quiet) |

All test-corpus files are telephony-style narrowband; all bypass correctly. No
wideband-noisy speech in the corpus to exercise the "run DFN3" path.

## Run 4 — 2026-04-20 19:02 (synthesised noisy wideband speech)

Question: when the probe decides to run DFN3 on a wideband+noisy file, does DFN3
actually improve ASR?

Method: mixed white noise into 60 s of `Xanth_audtiobooks/chapter_01.mp3`
(24 kHz mono) via `ffmpeg … amix`. Clean audiobook was 125 words.

| File | HF-ratio | noise-floor | Probe decision | Words | Seq-sim to clean |
|---|---|---|---|---|---|
| `audiobook_clean.wav` | 0.85 % | -∞ dBFS | bypass | 125 | 1.000 |
| `audiobook_noisy.wav` (white, -44 dBFS) | 2.14 % | -44.5 dBFS | **run DFN3** | — | — |
| &nbsp;&nbsp;→ no DFN3 | — | — | — | 125 | **0.992** |
| &nbsp;&nbsp;→ DFN3 on (blend α=19.8 %) | — | — | — | 126 | **0.980** ↓ |
| `audiobook_very_noisy.wav` (white, -32 dBFS, SNR ~8 dB) | 14.12 % | -32.5 dBFS | **run DFN3** | — | — |
| &nbsp;&nbsp;→ no DFN3 | — | — | — | 125 | **0.864** |
| &nbsp;&nbsp;→ DFN3 on (blend α=9 %) | — | — | — | 125 | **0.792** ↓ |
| `audiobook_pink.wav` (pink, SNR ~8 dB) | 1.42 % | -38.9 dBFS | bypass (narrowband) | — | — |

**Finding — unexpected and important**: DFN3 made transcripts *worse* in every
case where it ran, including loud white noise where it should help the most.
The per-frame blend limited the damage (it was aggressive: α as low as 9 %)
but couldn't make DFN3 a net positive. Pink noise at similar energy got
bypassed for narrowband reasons (falls off -3 dB/oct, little HF left).

Possible explanations:
- Parakeet is trained with heavy noise augmentation; it already handles noise
  well and any preprocessing distortion costs more than it saves.
- Synthesised white noise is out-of-distribution for DFN3 (trained on DNS
  Challenge real-world noise: fans, traffic, babble).
- Noise-matched training vs real evaluation distribution mismatch.

**Open question**: is there any class of noisy audio on which DFN3 genuinely
improves Parakeet transcripts? If not, the right conclusion may be to disable
DFN3 by default rather than tune its gating further.

Implication for next run: need real-world noisy speech — DNS Challenge samples,
a field recording, or a recorded voice memo in a loud environment. Synthesised
noise has proven to be a poor test stimulus.

## Run 5 — 2026-04-20 19:15 (resolution: DFN3 removed)

Question: given that every test file so far has either hurt ASR under DFN3 or
been correctly bypassed, is there a positive regime at all?

Discussion with the user reviewed the broader field: modern E2E ASR (Parakeet,
Whisper, Qwen3-ASR etc.) is trained with heavy noise augmentation and absorbs
noise robustness as part of training. Learned denoisers are overwhelmingly
trained on *perceptual* metrics (PESQ, DNSMOS, SI-SDR), not WER. The two
objectives diverge: enhancement artefacts that sound fine to humans are
out-of-distribution for the ASR, and cost more than the removed noise would
have. This matches upstream DeepFilterNet's stated design goals — it targets
real-time video/hearing-aid use cases, never ASR.

**Decision**: remove DFN3 entirely. Replace with always-on deterministic DSP
in `AudioUtils.ApplyCleanup`:
  • 2nd-order high-pass at 75 Hz (removes rumble / HVAC / handling noise)
  • Twin narrow notches at 50 Hz and 60 Hz (removes mains hum)

These target the specific low-frequency interferers modern ASR was not trained
to tolerate, are bit-for-bit deterministic, and have no ML-artefact risk.
A help page (`Help/en/first_steps/audio_input_quality.md`) covers recording
guidance and ASR-backend selection — the higher-leverage strategies for noisy
audio.

Candidates considered and set aside:
  • Silero Denoise — plausible (designed with ASR in mind) but unvalidated and
    not worth integration cost without evidence of Parakeet improvement.
  • RNNoise — conservative but near-zero expected benefit.
  • BSRNN, SGMSE+, FlowSE, BSRNN-Flow — same discriminative/generative training
    concerns as DFN3; no reason to expect different outcomes.

This investigation is closed. If a future real-noisy-speech eval set
(DNS Challenge, URGENT, CHiME) demonstrates a specific denoiser consistently
reduces WER on Parakeet, revisit.
