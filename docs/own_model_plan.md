# Plan — a model of our own: unencumbered weights, precomputed voices, ~10× faster

**Status: SHELVED 2026-09-04.** Written after v7 shipped, then parked the same day once the
timeline was costed honestly (2–4 months part-time, 200–350 GPU hours — see Timeline). Kept because
the analysis and the *rejected* alternatives are the expensive part to redo, not the prose. Nothing
here is being worked on.

Two goals, and it is worth being precise about which is which, because they have different urgency:

1. **Licence.** The OmniVoice transformer weights are CC-BY-NC "due to constraints from its training
   data". Anything built on them inherits that. Training on FLEURS (CC-BY 4.0) and Common Voice
   (CC0) produces weights we own outright.
2. **Speed.** v7 renders in ~40 s per generation on 8 WASM threads. That is a demo, not a product.

The plan reaches both with **one training run**, because the change that buys the speed is the same
change that requires retraining anyway.

---

## What we are NOT changing, and why

The instinct on a project like this is to rebuild the stack. Most of the stack is fine, and the
parts that are fine are the parts that took longest.

| kept | why |
|---|---|
| **The Higgs codec** | See below. The licence is not the problem we think it is. |
| **vernacula-phonemizer** | Every architecture in this space takes a phoneme sequence. The G2P work is portable by construction. |
| **The alignment DB and corpus machinery** | `ingest_dir.py`, `asr_align_dir.py`, `corpus_filter`, `sampling_budget` are all representation-agnostic. |
| **The 530-voice library** | Reference audio decodes from stored codes; voice latents get fitted from it. |
| **The listening harness** | `preview-all.mjs`, the verdict tracking, the rendering path. Judging a new model needs exactly this. |

### ⚠ Do not replace the codec

The obvious move — "escape NC, so replace everything encumbered" — is wrong here. Boson's licence
permits commercial use below **100k monthly active users**. That is not a threshold this project
will meet by accident, and replacing the codec costs:

- a re-encode of the whole corpus (~4 h GPU at the measured 74× realtime, but also a re-derivation
  of every downstream artifact)
- a new decoder to export, quantize, and ship in the browser bundle
- fresh unknowns in the one component that fails *invisibly* — a subtly metallic codec is not a
  crash, it is a mystery

⚠ **The honest asterisk:** a model that emits Higgs codes requires consumers to obtain Boson's codec
under Boson's terms. We would own the weights but not the whole stack. "Unencumbered" is therefore
true of the part that currently blocks us and false of the stack as a whole. Revisit only if the
100k threshold comes into view, or if full redistribution freedom becomes a goal in itself.

### ⚠ Do not switch to mel

Considered and rejected. Mel costs, against 25 Hz discrete codes:

| representation | per second | our ~288 h corpus |
|---|---|---|
| raw 24 kHz 16-bit | 48 KB | ~50 GB |
| mel 80 × 86 fps fp16 | ~14 KB | ~14 GB |
| **Higgs codes 8 × 25 Hz** | **400 B** | **~415 MB** |

That is ~35× the storage, it puts STFT work back inside the training loop (on a single-GPU setup the
data loader is frequently the real bottleneck), it triples the sequence length at inference, **and**
it obliges us to train and quality-bound a vocoder instead of reusing the codec's decoder.

### ⚠ Do not use flow matching

Flow matching is natively continuous and pairs badly with discrete RVQ codes. Since we are staying on
codes, the proven line is **masked-generative** — MaskGIT → SoundStorm → what OmniVoice already does.
It is also the architecture whose behaviour we already have intuitions about, which is worth more
than it sounds when debugging a from-scratch run.

---

## The design

A **masked-generative transformer over Higgs codes**, phoneme-conditioned, with the speaker supplied
as a **precomputed latent** rather than as reference audio in the sequence.

Three independent speed levers, none requiring a new representation:

| lever | from → to | factor | available? |
|---|---|---|---|
| backbone | 600M → ~150M | ~4× | yes |
| sequence length (reference leaves the context) | ~200 → ~110 | ~1.8× | yes, measured |
| decoding steps | 32 → ~12 | ~2.7× | ⚠ **only via distillation** — see below |

⚠ **THE STEP LEVER IS NOT FREE, AND THIS PLAN ORIGINALLY ASSUMED IT WAS.** Lowering the count
naively has already been tried and rejected by ear: web demo **Run 12** set `NUM_STEPS` to 16 to
halve browser latency, and every clip called quirky was at 16 while every clip called fine was at 32
— same text, voice, model and provider, step count the only variable. It was reverted to 32.

So a new model buys **~7×** from the first two levers. Getting the third requires distillation
(consistency or progressive), and that is true of ANY model here — a fresh 150M would need distilling
just as v7 does. Distillation is therefore an independent axis worth ~2–2.7×, not a property of this
design.

The sequence lever is measured, not estimated — `bench-wasm.mjs` on the int4 build at 8 threads:
S=200 → 2279 ms, S=100 → 1295 ms. Reference clips are `refLen` 100–300 frames against a comparable
target, so the reference is roughly half the context.

### Voice conditioning

The library is **fixed and curated**, not user-supplied. We never need zero-shot cloning at
inference — we need replay of known voices. That is a much weaker requirement and it is what makes
the speed lever available.

- **Training:** a small encoder produces a latent per utterance, trained jointly. It only has to be
  good enough to train against.
- **Shipping:** the 530 library voices get their latents **fitted offline by optimisation** against
  their reference audio, with the model frozen. Fitting one speaker well is an easier problem than
  generalising to unseen ones, and it typically beats the encoder's forward pass.
- **Interface:** the model consumes latents and does not care where they came from. Adding
  user-supplied cloning later means adding a producer, not changing the model.

⚠ **Size is not the reason to do this.** A 16 × 512 fp16 latent (~16 KB) is *larger* than the ~1.5 KB
of Higgs codes it replaces. The win is that ~190 frames leave the sequence, and that voices become
fittable offline. Anyone who justifies this on storage has misunderstood it.

**What this unlocks beyond speed:** interpolating between voices, adjusting delivery without
re-recording, and fitting a variant for the 54 single-sex languages and the ~20 where the sourcing
investigation concluded no acceptable open recording exists anywhere.

---

## Phases

### Phase 0 — already done, and it came back negative

⚠ An earlier draft of this plan proposed "run v7 at 8/12/16 steps and listen — free, today". That
experiment had already been run and recorded (web demo Run 12): 16 steps produced audio the listener
called quirky, 32 did not, and the change was reverted. Proposing it again was a failure to read our
own investigation log.

The consequence is recorded above: the step lever exists only through distillation, and the speed
case for a new model is ~7×, not ~19×.

### Phase 1 — the corpus (the big chunk)

See the section below. Expect this to dominate the calendar.

### Phase 2 — prove the conditioning at small scale (days)

Before committing to a long run: ~50M params, 5 languages, and one question — **does a fitted voice
latent reproduce a held-out speaker recognisably?** That is the load-bearing assumption of the whole
design. If it holds, scaling is engineering. If not, we have spent days rather than months.

### Phase 3 — the training run (weeks)

~150M, masked-generative, on the expanded corpus. Calibration from v7: 6000 steps in 2 h 07 m on the
3090 with a 600M backbone. A 150M model on shorter sequences is several times faster per step; the
from-scratch cost is in needing far more steps, not in each step being slow.

### Phase 4 — export and ship (days)

Follows the existing path: `export_omnivoice.py` → quantize (`quantize_lm.py` + `quantize_embedding.py`)
→ `publish_hf.py`. ⚠ Note that the export tooling is currently OmniVoice-shaped — `quantize_embedding.py`
hardcodes `model.llm.embed_tokens.weight`, and the export script's component list assumes the three
OmniVoice graphs. Budget a day for that rather than discovering it at the end.

---

## Phase 1 in detail — CV 26 and the alignment DB

This is expected to be the largest piece of work, and the expectation is right. It is also *not*
compute-bound, which changes how to plan it.

### How much data

FLEURS gives ~288 h across 29 languages and it is not enough for a from-scratch model with speaker
conditioning. Plan for **500–2000 h with many speakers** — speaker *count* matters more than hours
for the latent to generalise, and FLEURS has few speakers per language while Common Voice has many
(the single `es-MX_male` dataset alone: 73,020 clips across 378 contributors).

### The bottleneck is terms acceptance, not bandwidth or GPU

Measured or already recorded:

| step | rate | 2000 h |
|---|---|---|
| MDC download | ~400 MB/min | ~42 GB, a few hours |
| Higgs encode | 74× realtime on the 3090 | ~27 h GPU |
| wav2vec2 alignment | 30–57 utt/s | ~8 h GPU |
| storage as codes | 400 B/s | ~2.9 GB |

⚠ **Mozilla Data Collective requires terms accepted per dataset, in a browser, with no API route and
a rate limit of about one per minute** (web demo Run 24). For dozens of language datasets that is the
schedule driver. Everything after it is automated and cheap.

### ⚠ Common Voice is not FLEURS, and the corpus filter does not know that yet

FLEURS is read speech recorded to a consistent standard. Common Voice is crowd-recorded: levels,
rooms, microphones and reading quality vary enormously. `corpus_filter` currently gates on the
alignment `status` column, which catches *pronunciation* problems but says nothing about a clipped,
noisy, or near-silent recording.

The scoring already exists — `make-voice-from-commonvoice.mjs` computes SNR, speech fraction, peak,
RMS floor and true clipping (flattened runs, not peak ≥ 0.98, which discarded a whole corpus once).
**That logic needs to move from voice sourcing into the corpus path** and run over every candidate
training utterance. Training on unfiltered Common Voice will teach the model the noise.

This is new work, not a port: per-voice scoring runs over dozens of candidates, and this runs over
hundreds of thousands.

### The alignment DB scales, with one caveat

`asr_align_dir.py` already handles directory-shaped corpora and CV layout, and the table is keyed
`(lang, wav)` so it takes new rows without migration. Two things to keep in view:

- ⚠ **wav2vec2 has no clicks, implosives or retroflex stops** (corpus Run 40), so for some languages
  the gate is blind by construction. It is a filter, not a verdict.
- ⚠ **`verified` means the sentence-level distance was unremarkable.** It does not mean every word is
  right — the Croydon case (corpus Run 56) passed as `verified` in all 8 utterances with a
  mispronounced proper noun. At CV scale, expect more of this class and do not read the label as
  stronger than it is.

### Ordering

Do **one language end-to-end first** — download, encode, quality-gate, align, label, sample — and
only then batch the rest. The pipeline has been wrong in a silent way more than once (the wrong ONNX
base, the stale phonemizer pin, `sentence_id=None` emptying a train split), and discovering that
after 2000 h of processing is a different cost from discovering it after 100 h.

---

## Timeline

Calibrating from v7: 600M at S≈200 ran 6000 steps in 2 h 07 m on the 3090 = 0.79 steps/s. A 150M
model at S≈110 is roughly 0.14× the compute per step, so ~4 steps/s after overheads that do not
scale down.

| phase | GPU time | calendar, part-time |
|---|---|---|
| Phase 2 proof (50M, 5 languages, a few runs) | ~20–40 h | 2–3 weeks |
| Phase 1 corpus (terms, download, encode, align, quality gate) | ~35 h | 3–6 weeks |
| Phase 3 (500k–1M steps ≈ 40–80 h **per run**, expect 3–5 runs) | ~150–300 h | 3–6 weeks |
| Phase 4 export and ship | small | days |

**Roughly 2–4 months part-time, 200–350 GPU hours.** ⚠ The single training run is only 2–3.5 days.
What makes this months is that from-scratch models do not work first time, and that Phase 1 is gated
on a person clicking through MDC terms at about one per minute.

That cost, against ~7× rather than the ~19× first assumed, is why this is shelved. The speed case
alone does not carry it. The licence and the offline voice-fitting capability might — that is a
judgement about what the project is for, not a technical one.

## Risks, worst first

1. **From-scratch quality at our data scale.** OmniVoice was pretrained on far more audio than we
   will have. This is the real hazard, not compute. Mitigation: Phase 2's small-scale proof, and a
   willingness to stop.
2. **The voice latent does not generalise.** The design's load-bearing assumption. Phase 2 exists to
   test it cheaply.
3. **Corpus quality at CV scale.** Crowd-recorded audio without an audio-quality gate will degrade
   the model in ways that are audible but hard to attribute.
4. **Calendar, via terms acceptance.** Not technically hard, but it is a person clicking, one per
   minute, and it gates everything downstream.
5. **Codec fidelity ceiling.** We inherit whatever Higgs loses. Acceptable — v6/v7 already sound fine
   through it — but it bounds the best case.

## The cheaper alternative, which is what shelving this chooses

**Distill v7 to fewer steps.** Weeks rather than months, no corpus work, ~2–2.7×, and it changes
nothing else in the stack. It keeps the CC-BY-NC weights, which is the whole thing this plan existed
to escape — so it is a trade, not a strict improvement.

Note the two are not exclusive: distillation applies to a new model as well, and the 7× and the
2–2.7× compose. If this is ever picked up, distillation is worth doing FIRST, on v7, because it is
cheap, it is independent, and it de-risks the step assumption for whatever comes after.

## Open questions

- Exact backbone size. 150M is an estimate, not a measurement — Phase 2 should inform it.
- Number of latent tokens per voice (1 global vs 8–32). A single vector is fastest and most likely to
  lose the prosody and cadence the voice selection has been optimising for.
- Whether the audio heads stay fp32 in the quantized build. Currently they are quantized, which the
  web demo Run 3 argued against for ~28 MB. Still an open one-flag experiment.
- Whether to keep 29 languages or start narrower. Multilingual breadth is a cost multiplier on every
  phase.
