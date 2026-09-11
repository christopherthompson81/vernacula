# Kokoro-82M → ONNX export investigation

Goal: produce a Vernacula-owned ONNX export of hexgrad/Kokoro-82M whose audio output
matches the PyTorch reference (no official conversion script exists upstream). Track the
PyTorch↔ONNX numerical parity hunt here.

Setup: `scripts/kokoro_export/` — `export_kokoro.py` exports
`KModel.forward_with_tokens(input_ids, ref_s, speed)` with `KModel(disable_complex=True)`,
then validates ONNX vs PyTorch on random sample inputs. Python 3.12 venv, torch 2.12.0+cu130,
onnx 1.21, onnxruntime 1.26, opset 17.

## Run 1 — 2026-06-09 18:16

Command:
```
.venv/bin/python export_kokoro.py --out external/kokoro_onnx --opset 17
```
Question: does a naive `forward_with_tokens` export with `disable_complex=True` reach parity?

Finding:
- Export succeeds via the **legacy TorchScript exporter** (`dynamo=False`). The new
  torch 2.12 default (dynamo) needs `onnxscript` which isn't installed; legacy path is also
  the historically battle-tested one for this model.
- `kokoro.onnx` = 325.5 MB.
- Validation **FAIL**: `max|Δ|=3.95e-02`, `corr=0.761943`, both outputs `(76800,)` samples.
- Output lengths match exactly → the duration-based length regulator is deterministic given
  identical inputs (good — rules out a frame-count mismatch).

Tracer warnings worth noting:
- `prim::PackPadded` / `prim::PadPacked` shape-inference-missing warnings — the prosody
  LSTMs use `pack_padded_sequence`, whose ONNX export is historically lossy.
- `istftnet.py:380` `torch.rsqrt(torch.tensor(2))` registered as a constant — benign.

Interpretation: `corr=0.76` (not ~0) means the waveform is *mostly* right, not garbage. The
candidates for the residual: (a) iSTFT framing off-by-N time shift, (b) packed-LSTM export
loss in the duration/prosody predictor, (c) residual STFT divergence despite disable_complex.
Next: characterize the divergence (cross-correlation lag, where the error concentrates)
before assuming it's the STFT.

## Run 2 — 2026-06-09 18:25

Command:
```
.venv/bin/python export_kokoro.py --out external/kokoro_onnx_dynamo --opset 17   # dynamo=True
```
Question: does the torch.export-based (dynamo) exporter fix the packed-LSTM parity gap?

Finding: **dynamo export FAILS outright** (cannot even produce a graph with a dynamic token
length). `GuardOnDataDependentSymNode: Could not guard on data-dependent expression
Ne(u0, 26)` raised inside `transformers/integrations/sdpa_attention.py` →
`scaled_dot_product_attention`. The `26` is the traced token length (seq_len 24 + 2 pad);
with `tokens` made a dynamic Dim, the BERT attention-mask length becomes symbolic and
`torch.export` can't resolve the SDPA branch on it. Suggested `torch._check(...)` fixes
require editing the (vendored transformers / kokoro) model code.

Interpretation: dynamo is a dead end *unless* we either (a) patch the model to insert
`torch._check` / use guard_or_false-friendly APIs, or (b) export with a static token length
and pad at inference (loses the dynamic axis). The legacy exporter is much closer — it
produces a usable graph. Pivot back to diagnosing the Run 1 legacy parity gap rather than
fighting torch.export. User is OK with dynamo in principle, but it doesn't work off-the-shelf
for this model.

Next: diagnose the legacy export divergence — cross-correlation best-lag (time shift?) and
where the error concentrates (onset vs throughout).

## Run 3 — 2026-06-09 18:40

Two diagnostics on the Run 1 legacy graph.

**(a) Random-input parity was misleading.** Cross-correlation of torch-vs-onnx on the random
sample inputs: best lag = 2 samples, corr unchanged (0.758 → 0.760) → NOT a time shift. Error
spread uniformly across all deciles (~0.6 rel-rms) → NOT an onset transient. Conclusion: random
token ids drive the duration predictor + prosody LSTM off-distribution, where tiny numerical
diffs compound. The validation harness itself was wrong.

**(b) Real-input parity is good; `disable_complex` is the real problem.** Captured the actual
`(input_ids[1,79], ref_s[1,256], speed=1.0)` a real `KPipeline("…fox…", voice="af_heart")` run
feeds the model, then compared three signals:

| Comparison | corr | max\|Δ\| | SNR |
|---|---|---|---|
| A complex-STFT torch vs B real-STFT torch (`disable_complex=True`) | 0.884 | 1.56e-1 | **5.5 dB** |
| B real-STFT torch vs C onnx | 0.996 | 7.8e-2 | 21.5 dB |
| A complex-STFT torch vs C onnx (end-to-end) | 0.883 | 1.26e-1 | **5.5 dB** |

Key result: **the ONNX export is faithful to its source (B→C: 21.5 dB), but `disable_complex=True`
itself degrades audio vs the true model (A→B: 5.5 dB).** End-to-end quality is dominated by the
STFT swap, not the conversion. Same output length across all three (pred_dur unaffected by the
decoder STFT), so the divergence is purely in the vocoder iSTFT.

Implication: the README framing ("disable_complex swaps in a real-valued STFT that exports
correctly") is wrong — it exports correctly but is NOT faithful. To get a high-quality export we
must export the **complex** STFT path (opset-17 native STFT or a corrected custom STFT layer),
not disable it. The B→C 21.5 dB residual is a secondary concern behind the 5.5 dB A→B loss.

Next: attempt to export with `disable_complex=False` (complex STFT) and see whether/how the
exporter fails — that failure is presumably *why* disable_complex exists, and is the real
problem to solve.

## Run 4 — 2026-06-09 18:52

Command: `torch.onnx.export(..., disable_complex=False, dynamo=False, opset=17)`
Question: can the faithful complex-STFT path be exported directly?

Finding: **FAILS — `RuntimeError: Unknown number type: complex`.** The legacy TorchScript
exporter cannot represent complex tensors at all. Combined with Run 2 (dynamo dies on the
attention data-dependent guard), neither exporter handles the complex STFT off-the-shelf.
This is the concrete reason `disable_complex` exists.

**Methodology correction (important).** The A→B "5.5 dB SNR" from Run 3 was measured in the
*waveform domain*. iSTFTNet is a GAN vocoder — its output phase is not uniquely determined,
so two perceptually identical renderings can have low sample-wise correlation. Waveform SNR /
corr is therefore an unreliable proxy for quality here; it can flag inaudible phase
differences as large "errors." The Run 1–3 numbers measure *waveform divergence*, NOT
audible degradation. The honest test is to listen (or use a perceptual/mel-domain metric).

Next: render real audio for A (complex/true), B (disable_complex torch), C (onnx) and listen.
Decide acceptability perceptually before chasing the waveform-domain residual. If B and C
sound indistinguishable from A, the disable_complex export is fine and the investigation is
essentially done; if not, the STFT layer needs a faithful real-valued reimplementation.

## Run 5 — 2026-06-09 18:58

Rendered all three to `scripts/kokoro_export/samples/{A_complex_torch,B_disablecomplex_torch,C_onnx}.wav`
(4.7 s each, "The quick brown fox… pauses to think.", voice af_heart) and compared in the
**log-magnitude spectral domain** (phase-invariant — the right domain for a GAN vocoder):

| Comparison | log-spec L1 |
|---|---|
| A complex vs B disable_complex | 0.374 |
| A complex vs C onnx (end-to-end) | 0.372 |
| B disable_complex vs C onnx | 0.195 |
| **reference: A vs A shifted 1 frame** | **0.768** |

The disable_complex spectral difference (0.374) is **half** the magnitude of a single STFT
frame of time jitter (0.768) — i.e. small and perceptually minor. This directly confirms the
Run 4 methodology note: the waveform-domain 5.5 dB A→B "loss" was overwhelmingly vocoder
phase, not audible spectral content. The ONNX export adds almost nothing on top of the
disable_complex source (A→C ≈ A→B).

**Conclusion of phase 1.** The `disable_complex=True` legacy-exporter path produces an ONNX
model that is perceptually close to the true complex-STFT model. Waveform SNR is the wrong
acceptance metric here; log-spectral distance (or listening) is right. Pending a human
listening check on the three WAVs, the export approach is sound. Remaining items are
quality-polish / packaging, not a fundamental fidelity problem:
- Fix the validation harness in `export_kokoro.py` to use **real captured inputs** + a
  **log-spectral** metric instead of random-input waveform SNR (current FAIL is a false alarm).
- Dynamic token axis works under the legacy exporter (it's the dynamo path that can't do it).
- Decide on quantization (fp16/int8) and the C# I/O contract.

## Run 6 — 2026-06-09 19:05

Two confirmations close out phase 1.

**Listening test (human).** User listened to A/B/C: "They all basically sound identical."
This is the perceptual ground truth — confirms the disable_complex ONNX export is faithful and
vindicates dropping waveform SNR for the log-spectral metric.

**Harness fixed + re-run.** `export_kokoro.py` now (a) captures real pipeline inputs for both
tracing and validation, (b) validates with log-spectral L1 (threshold 0.25), (c) defaults to
the legacy exporter (dynamo left as an opt-in flag that currently fails — Run 2). Result:
```
[export] traced shapes: input_ids=(1, 79) ref_s=(1, 256)
[export] wrote external/kokoro_onnx/kokoro.onnx (325.5 MB)
[validate] log-spec L1=0.1322  (threshold 0.25)  PASS
```
(0.1322 < the 0.20-ish B→C residual seen earlier because validation now traces and validates on
the same real input; the prior 0.195 compared two separately-rendered signals.)

**Status: phase 1 complete.** Faithful fp32 ONNX export of Kokoro-82M, validated perceptually
and by log-spectral metric. README corrected (the original "disable_complex is equivalent"
framing was wrong). Open follow-ups for later phases: fp16/int8 quantization, the C# inference
I/O contract, and (optional, low priority) a faithful real-valued STFT reimplementation if we
ever want to close the residual complex-vs-real gap — but the listening test says we don't need
to.

## Run 7 — 2026-06-09 19:25

Command: `.venv/bin/python parity_sweep.py`
Question: does parity hold beyond one utterance — across voices (ref_s vectors) and token
lengths, including the long-input edge?

Finding: **24/24 cells PASS.** 8 English voices (American/British × f/m) × 3 lengths
(short 13 tok / medium ~50 / long ~187). Output length matched **exactly** in every cell.
log-spectral L1 ranged 0.088–0.194 (worst: bm_george long), all under the 0.25 threshold and
all well under one-frame-jitter (~0.77). Mild trend: L1 creeps up with length (STFT-framing /
pred_dur accumulation) and British male voices sit highest — but none near threshold.

Implication: the export generalizes across the style-vector space and length; no voice-pack
indexing bug, no length-dependent divergence. Parity phase is done. `parity_sweep.py` is the
reusable gate (will double as the acceptance test for each quantization level in phase 3).

## Run 8 — 2026-06-09 19:40

Prompted by "we have espeak installed for G2P." The G2P frontend is OUTSIDE the ONNX graph
(we export `forward_with_tokens`), so the C# phase must reproduce phonemization. Question:
can espeak-ng alone drive the C# frontend, or is misaki's lexicon required?

How Kokoro wires English G2P (`KPipeline.__init__`):
```
fallback = espeak.EspeakFallback(british=lang_code=='b')
self.g2p  = en.G2P(british=..., fallback=fallback, unk='')   # lexicon PRIMARY, espeak OOV-only
```
Non-English langs use `EspeakG2P(language=…)` — **pure espeak**.

Measured English divergence (token ids via km.vocab, 114 tokens):

| Path | vs reference | notes |
|---|---|---|
| Raw `EspeakG2P('en-us')` | **~72% token match** | espeak length marks `ː`, rhotic `ɚ`/flap `ɾ`, secondary stress, dialect vowels |
| misaki `EspeakFallback` (normalized espeak) | near-exact | `over` exact; `seashore`/`configuration` off by 1 stress/`ɹ` mark |
| Full `en.G2P` (lexicon + fallback) | exact (the reference) | requires shipping misaki's English lexicon |

All espeak symbols (`ː ɚ ɾ`) ARE in Kokoro's vocab → raw espeak yields *valid* input, just
pronounced differently. Residual EspeakFallback gaps (e.g. `dog` `ɔ`→`ɑ`) are genuine
dialect/lexicon choices espeak can't reproduce, not notation.

Implication — three C# frontend tiers, fidelity vs effort:
1. **Raw espeak-ng** — simplest (just bindings), but ~28% of tokens differ → audibly different
   (vowel length/quality, stress) though still legitimate pronunciations.
2. **espeak-ng + ported misaki normalization** — port EspeakFallback's deterministic string
   rules (`ː` strip, `ɚ`→`əɹ`, etc.) to C#. Closes most of the gap; residual is dialect vowels.
3. **Full parity** — also ship misaki's English lexicon + lookup/stress logic. Exact match,
   most work.

This is a product decision (how close to reference must C# TTS sound) that materially changes
the C# phase scope. Non-English support, if ever wanted, is pure-espeak and thus "free-ish"
under tiers 1–2. Decision pending from user before starting C#.

## Run 9 — 2026-06-09 19:55

Decision: **user chose tier 2 (espeak + ported normalizer).** This run nails down the spec
and the honest parity numbers so the C# port is mechanical.

**Normalizer spec** = `misaki.espeak.EspeakFallback.__call__` (fully deterministic):
1. espeak-ng `en-us`/`en-gb`, flags `preserve_punctuation=True, with_stress=True, tie='^'`.
2. Apply `E2M` map (sorted longest-key-first): diphthong ties `a^ɪ→I a^ʊ→W e^ɪ→A o^ʊ→O
   ɔ^ɪ→Y d^ʒ→ʤ t^ʃ→ʧ`, `ɚ→əɹ`, `r→ɹ`, `x→k ç→k`, `ɐ→ə`, `ɬ→l`, syllabic-n forms, strip `̃`.
3. `re.sub(r'(\S)̩', 'ᵊ\1')` then strip U+0329 (syllabic consonants).
4. US branch: `o^ʊ→O`, `ɜːɹ→ɜɹ`, `ɜː→ɜɹ`, `ɪə→iə`, **strip `ː`**; then `o→ɔ`; then `ɾ→T`, `ʔ→t`;
   strip `^`. (GB branch differs: `e^ə→ɛː`, `iə→ɪə`, `ə^ʊ→Q`, keeps `ː`.)

All output symbols are in Kokoro's 114-token vocab. The only external dep is espeak-ng with
`tie='^'` IPA output.

**Honest tier-2 parity (vs lexicon reference), token ids via km.vocab:**

| Corpus | exact-word | token parity | character |
|---|---|---|---|
| Running prose (freq-weighted, 80 words) | 78.8% | **92.1%** | matches "~90%+" claim |
| Flat lexicon sample (400 words, names/rare-heavy) | 33.0% | 70.8% | worst case |

Dominant divergences on common text: (a) **secondary-stress marks** the lexicon adds but
espeak omits (`tˈuzdˌA` vs `tˈuzdA`) — not recoverable from espeak, perceptually minor; (b)
schwa color (`ə` vs espeak `ᵻ`). **Normalizer gap found:** espeak emits `ᵻ` (U+1D7B) which
`E2M` does NOT map → it's out-of-vocab and gets dropped; the C# port should add `ᵻ→ɪ` (or `ə`).
Proper nouns/rare words are the real weakness (33% exact) — names will sometimes mispronounce;
acceptable for a dictation tool's playback per the tier-2 decision.

Next: C# phase. Ground it in how the existing ONNX models are wired in the Vernacula .NET
solution (`src/`, `Vernacula.slnx`) before writing the Kokoro TTS path + the espeak normalizer.

## Run 10 — 2026-06-09 21:00

C# G2P: the espeak side is already a pure-C# reimplementation in a separate repo,
`~/Programming/espeak-ng-portable/csharp` (`Vernacula.Phonemizer`, golden-tested against the
TS engine). It renders **tieless** IPA (`aɪ`, keeps `ː`, uses `ɚ ɾ ɹ ɐ`). Per the user's
steer, Kokoro is modelled as a **render format** over those IPA phonemes, not a bolt-on
normalizer: added `PhonemeFormat { Ipa, Kokoro }` + `KokoroFormat.Render(ipa, british)` in
`src/Vernacula.Phonemizer/KokoroFormat.cs`. It ports misaki's `EspeakFallback.__call__`
deterministic map, adapted from misaki's tied diphthongs (`a^ɪ`) to this engine's tieless
forms, dropping misaki's tie-dependent syllabic rule in favour of the U+0329 handling this
engine emits, and adding the `ᵻ→ɪ` gap fix from Run 9.

Verified: C# `KokoroFormat.Render(Phonemize.Run(w))` vs Python `EspeakFallback(w)` over a
34-word list → **30/34 exact**. All 4 diffs benign: `remember` is the intentional `ᵻ→ɪ` fix
(C# better — `ᵻ` is out-of-vocab in Python); `example/little/people` are `əl` vs `ᵊl`
(this engine renders the syllabic schwa explicitly, no U+0329 to convert — both valid Kokoro
phonemes). Project builds clean. (espeak-ng-portable repo, file uncommitted — different repo.)

Remaining C# work for a full Kokoro path: `Kokoro.cs` ONNX wrapper in Chatterbox.Base
(mirrors Vocoder.cs), voice-pack loading + `ref_s` indexing, and phoneme→token-id via the
114-entry vocab. The render-format frontend (this run) is the piece that was specced as risky;
it's done and validated.

## Run 11 — 2026-06-09 22:30

Built the C# Kokoro inference path in `src/Chatterbox.Base/`:
- `KokoroVocab.cs` — the 114-entry phoneme→id map (generated from `KModel.vocab`), `Encode()`
  produces `[Pad, …ids…, Pad]`, unknown codepoints dropped (matches KModel).
- `Kokoro.cs` — loads `kokoro.onnx`, lazy-loads voice packs, `Synthesize(phonemes, voice, speed)`.
  `ref_s = voicePack[len(phonemes)-1]` (rune count, matching KPipeline's `pack[len(ps)-1]`).
- `scripts/kokoro_export/export_voices.py` — dumps each voice `.pt` `[510,1,256]` to flat
  `<name>.bin` (`510×256` f32) for the C# loader. 28 English voices exported.

**Correction to Run 9:** `ᵻ` (U+1D7B) IS in the Kokoro vocab (id 177) — misaki feeds it to the
model as a valid token. The Run-9 "gap fix" `ᵻ→ɪ` was therefore a *divergence*, not a fix, and
was removed from `KokoroFormat.cs`. (The Run 8/9 parity *numbers* were unaffected — they used the
real vocab where `ᵻ` maps to 177, not dropped — only the verbal claim was wrong.)

**Parity verification (C# vs Python ONNX, same phonemes + voice af_heart):**
- input_ids: **byte-identical** (38 ids); rune count 36 = Python `len(ps)` → same `ref_s` row 35.
- exported voice row 35 vs `pack[35]`: **max|Δ| = 0.0**.
- audio: length identical (67800); waveform max|Δ| 0.072, corr 0.997, log-spectral L1 0.13.

The waveform difference is NOT a bug: **the kokoro.onnx graph is non-deterministic on CPU ORT** —
two consecutive Python runs with identical inputs differ by max|Δ| 0.070 (parallel FP-reduction
order in the conv/STFT). The C#↔Python difference (0.072) is within that self-noise floor, and
log-spec 0.13 is well under the 0.37 the user already judged "basically identical." Since all
three inputs are provably identical, any residual is ORT execution variance, not the C# code.
**C# Kokoro inference path verified.** Builds clean (Chatterbox.Base, CPU + default EP).

Remaining: wire text→phonemes (the `Vernacula.Phonemizer` repo is separate — needs a project/
package reference decision) into a one-call pipeline, and the playback/UI surface. The core
inference + tokenization + voices are done.

## Run 12 — 2026-06-09 22:55

Wired the G2P frontend end-to-end (user chose **git submodule** for the integration mechanism):
- Added `external/espeak-ng-portable` as a submodule (carries the pure-C# espeak port,
  `KokoroFormat`, and the runtime `data/` language files). `.gitmodules` URL points at the
  canonical GitHub remote; pin is currently a local-only commit (the `KokoroFormat` branch isn't
  pushed yet — push/merge it for fresh clones to resolve).
- ProjectReference Chatterbox.Base → Vernacula.Phonemizer.
- `KokoroTts.cs`: `Speak(text, voice, speed, british)` and `ToPhonemes(text)` compose
  `Phonemize.Run` → `KokoroFormat.Render` → `Kokoro.Synthesize`. Constructor takes the
  phonemizer `data/` dir.

End-to-end verified on CPU: "Hello, this is a Kokoro speech test." → phonemes
`həlˈO` / `ðɪs ɪz ə kəkˈɔɹO spˈiʧ tˈɛst` → 2.6 s audio. Builds clean.

**Known refinement (prosody):** the C# `Phonemize.Run` orchestrator splits clauses with a
**newline** and drops `,`/`.` punctuation, whereas misaki keeps punctuation as phoneme tokens
(ids 1–15) that Kokoro uses for pauses. `KokoroVocab.Encode` silently drops the newline, so
clause boundaries currently carry no pause → slightly rushed pacing vs reference. Candidate
fixes: map the phonemizer's clause newline to a Kokoro pause token, and/or preserve sentence
punctuation through to `Encode`. Deferred pending the listening check; intelligibility is fine.

## Run 13 — 2026-06-09 23:20

User confirmed the missing pause should be there. Characterized the phonemizer's punctuation
handling empirically (no ORT needed):

| source | phonemizer output |
|---|---|
| `Hello, this is a test.` | `həlˈO\nðɪs ɪz ə tˈɛst` (comma→\n, final . dropped) |
| `Wait. Stop! Why?` | `wˈAt\nstˈɑp\nwˈI` (. ! both →\n, final ? dropped) |
| `It cost $3.14 today.` | one clause — the `3.14` dot makes NO break (normalized to words) |
| `Dr. Smith arrived; we left.` | `…\n…\n…` (Dr. and ; both →\n) |

So every clause mark collapses to `\n` (type lost) and the final mark is dropped. Fix in
`KokoroTts.ToPhonemes`: correlate the source text's clause punctuation — regex
`[,;:!?…—] | (?<![0-9])\.(?![0-9])`, the digit-guard excludes decimals so the count stays
aligned — with the `\n` breaks, re-inserting the i-th source mark at the i-th break and the
trailing position. Added `KokoroVocab.Contains`.

Verified reconstruction on all cases: `həlˈO, ðɪs ɪz ə tˈɛst.`, `wˈAt. stˈɑp! wˈI?`,
`…tədˈA.` (3.14 intact), `dˈɑktəɹ. smˈɪθ əɹɹˈIvd; wi lˈɛft.`, `wˈʌn, tˈu, θɹˈi, fˈɔɹ.`.
End-to-end: "Hello, this is a Kokoro speech test." → `həlˈO, ðɪs ɪz ə kəkˈɔɹO spˈiʧ tˈɛst.`
(matches misaki bar the benign `ə`/`ɐ` for "a"), audio 2.8 s vs 2.6 s — pause restored.

**C# Kokoro TTS path is feature-complete:** text → phonemes (espeak port + KokoroFormat +
punctuation) → tokenize (KokoroVocab) → ONNX (Kokoro) → 24 kHz audio, via `KokoroTts.Speak`.
Remaining is non-core: playback/UI surface, and the performance phase (fp16/int8; the
log-spectral harness is the acceptance gate).

## Run 14 — 2026-06-09 23:45 — Performance phase

Built `tests/KokoroPerf` (latency / RTF harness, both EPs; reusable as the quantization gate).
Hardware: RTX 3090, CUDA 12.6, ORT 1.24, fp32 `kokoro.onnx`. RTF = audio_s / inference_s.

| case | phonemes | audio_s | CPU med_ms | CPU RTF | CUDA med_ms | CUDA RTF |
|---|---|---|---|---|---|---|
| short | 11 | 1.52 | 329 | 4.6× | 19 | **80.5×** |
| medium | 48 | 3.23 | 733 | 4.4× | 44 | **73.6×** |
| long | 149 | 9.30 | 2296 | 4.1× | 190 | **48.9×** |

Session load: ~700 ms CPU / ~1.15 s CUDA (one-time). First CUDA call ~400 ms (kernel JIT /
cudnn autotune) → a warmup synth is worth doing so the first user-facing call isn't slow.

**CUDA correctness:** CPU vs CUDA produce **identical durations** (length match across all
cases) and log-spectral L1 0.11–0.20 — within the model's own nondeterminism band and well
under the 0.37 the user judged "basically identical." The `ScatterND reduction=='none'
duplicate-indices` warning is benign here: identical lengths prove the length-regulator indices
aren't duplicated in practice.

**On the user's checklist:**
- *Produce output quickly* ✅ — 49–80× real-time on CUDA, 4× on CPU (CPU is a viable fallback).
- *CUDA testing* ✅ — works and is correct.
- *IO handled (no CPU/GPU transfers)* — **not needed.** Kokoro is a single `Run`: inputs are
  tiny (ids+ref_s+speed < 2 KB) and the output download is < 1 MB (~0.1 ms at PCIe speeds) vs
  ~190 ms compute. IoBinding pays off for iterative graphs (cf. the Vocoder CFM loop), not a
  one-shot graph. Skipped deliberately.
- *Batching* — **not warranted.** 49–80× single-stream RTF already covers real-time and bulk;
  the exported graph is batch=1 (dynamic-batch export hits the Run-2 attention guard). Revisit
  only if a bulk-offline throughput need appears.

**Conclusion:** performance is a non-issue at fp32 — CUDA is 50–80× real-time and correct, CPU
is a 4× fallback. fp16/int8 quantization is now optional (latency, not a bottleneck; could
still cut the 325 MB model / VRAM), gated by the log-spectral harness if pursued.

## Run 15 — 2026-06-10 — Kokoro word alignment (UI backend selector)

To give the Kokoro UI backend karaoke word-highlighting (like Chatterbox), re-exported
`kokoro.onnx` with `pred_dur` as a second output (`output_names=["audio","pred_dur"]`,
`pred_dur` dynamic on `tokens`). Audio output unchanged → validation still PASS (log-spec 0.13).

`pred_dur` is `[tokens]` int64, one per input id (incl. the 2 pad tokens). **Frames→samples is
exactly 600**: `len(audio) = sum(pred_dur) × 600` (67800 = 113 × 600), i.e. 25 ms per duration
unit @ 24 kHz. C# computes the ratio per-utterance as `audio.Length / Σpred_dur` (exact).

**Word boundaries come free from the token stream** (no phonemizer change): a word = a maximal
run of tokens that are neither space (id 16) nor pad (id 0); punctuation (`,` `.` …) stays inside
the preceding word's run so its pause is attributed there. Verified
`həlˈO, ðɪs ɪz ə kəkˈɔɹO spˈiʧ tˈɛst.` → 7 runs ↔ 7 whitespace-split source words. Per-run
cumulative `pred_dur × 600 / 24000` gives word start/end seconds; leading pad = pre-roll silence.

Verified `SpeakAligned("Hello, this is a Kokoro speech test.")` → 7 words, monotonic, "Hello,"
starts at 0.350 s (= 14-unit leading pad), comma/period pauses on the right words, coverage to
2.775/2.800 s.

**UI integration (Chatterbox.Avalonia).** Added a TTS-backend selector. Backend abstraction
`ITtsBackend` + shared streaming records in `Services/TtsStreaming.cs`; `SynthesisService`
(Chatterbox) and new `KokoroSynthesisService` both implement it (chunk via `ParagraphChunker`,
emit `ChunkProducedEvent` with `AlignedWord`s — Kokoro's from `SpeakAligned`). ViewModel gains
`SelectedBackend` + Kokoro model/data dir + voice + speed (voices auto-discovered from
`<modelDir>/voices/*.bin`; en-gb inferred from `bf_`/`bm_`). View shows a backend ComboBox with
backend-specific config panels. Settings persist the new fields. Built clean (CPU + CUDA); app
launches and the Kokoro panel renders correctly (backend picker, populated voice list, speed
slider, panel switching). Word highlighting reuses the existing Chatterbox `AlignedWord` consumer
unchanged, so Kokoro highlights identically.

NB: the re-exported `kokoro.onnx` now has 2 outputs — `Kokoro.SynthesizeWithDurations` requires
the new graph (re-run `export_kokoro.py` if using an older single-output model).

## Run 16 — 2026-06-10 — Alignment precision (spell-out fix)

User reported word-highlight drift ("sometimes ahead, sometimes behind"). Root-caused it to
**spell-out expansions**: the phonemizer's normalizer expands written tokens (`$3.14` → "three
dollars and fourteen cents" = 5 phoneme groups, `2024` → 4, `Dr.` → "doctor", emails/acronyms),
so one grapheme becomes several space-separated groups. The old `SpeakAligned` joined groups to
words *positionally*, so any expansion broke the 1:1 count and the whole sentence fell back to
**even spacing** → every word mistimed. Demonstrated: clean prose timed irregularly (correct);
`"It cost $3.14 … 2024."` timed at a flat 0.77 s/word (the fallback firing).

Fix (chosen over a C#-only heuristic because spell-outs are context-sensitive): thread a
**source-word index through the phonemizer** (espeak-ng-portable @ 93188a0). `InitialTokens`
stamps each token's `text.Split` index; it propagates through the normalize splice helpers
(`MapPlainText`/`RewriteInTokens` + a backfill safety net), `Atom`/`ContextualToken`, and the
`ClauseToken` emit sites; `AssembleClauseIpa` reads it back per output group. New
`Phonemize.RunWithSourceWords` returns IPA + one source-word index per group. Additive metadata
only — **all 158,662 golden parity tests pass** (IPA byte-identical).

`KokoroTts.SpeakAligned` now uses that map: each phoneme group → its source word; consecutive
groups sharing a word merge into one displayed grapheme spanning their combined duration. The
even-split fallback is gone. Verified: `$3.14[0.85–2.27]` spans its full spoken expansion,
`2024.[5.27–6.92]`, `test@example.com[0.90–2.05]`, `ASAP.[2.13–3.28]` — every grapheme maps to
its true audio span while displaying the original source text.

## Run 17 — 2026-06-10 — Markdown-structured karaoke display

User: the karaoke pane wasn't a faithful markdown rendering (uniform size, no line breaks) and
unpronounceable/symbol source words had stopped appearing. Redesigned the display to look like
rendered markdown (heading sizes, paragraph breaks, list bullets, blockquote indent) with
inline **bold**/*italic*/`code`/links, while keeping per-word highlighting.

- **`MarkdownTextExtractor`**: now emits `BlockSpan(Kind, Level, OutputStart, OutputLength)` per
  text block and an `InlineStyle` flag on each `TextRange` (Bold/Italic/Code/Link) — additive;
  existing `.Text`/`.Ranges` consumers unaffected. Verified blocks + styles on a sample doc.
- **Dropped-word fix** (`KokoroTts.SpeakAligned`): iterate source words 0..N-1, emitting one
  `KokoroWord` per source word (zero-length for unpronounceables) → 1:1 with the whitespace
  split, restoring symbols/unpronounceables and keeping the index-zip exact.
- **Display**: `MainViewModel.BuildDisplayStructure` runs the extractor on `Text`, splits the
  extracted text into words with offsets, looks up each word's block (BlockSpan) + inline style
  (TextRange), and builds `DisplayBlocks` (`BlockItemViewModel` → `WordItemViewModel`). Words
  render up front (un-timed, StartSeconds = +∞ so they're never the highlight target); the
  stream attaches real timing to `Words[index]` (both backends emit 1:1 aligned words in order,
  so index-zip holds). Live preview rebuilds on text change. The XAML is now a nested
  ItemsControl (blocks → wrapping word buttons) with heading/bold/italic/code/link/quote styles;
  `.current` highlight composes on top.

Verified by clean build (Debug+Release; Avalonia compiles XAML), extractor + 1:1 unit tests, and
no runtime binding errors when the app launched. Visual layout left for the user to confirm
in-app (GUI screenshot automation was unreliable in this environment).

## Run 18 — 2026-06-10 — Long-form: token-overflow chunking (CLI debug)

App symptom: long-form synthesis incomplete + "not as fast as expected". CLI repro (new
`--backend kokoro` path in Chatterbox.CLI) gave the real error:
`Expand '/bert/embeddings/Expand_1': LeftShape {1,512}, RightShape {1,547}` — a chunk of 547
tokens hit Kokoro's 512 context window, the graph threw, and synthesis aborted partway (the
"incomplete"; the "slow" was the crash, not throughput).

Root cause: `ParagraphChunker` splits by **characters** (≤600, tuned for Chatterbox's LM), not
Kokoro tokens. A 560-char single-sentence paragraph = 545 tokens > 510.

Fix: `KokoroTts.ChunkForSynthesis` — paragraph/char chunk first, then sub-split any over-budget
chunk on sentence → (if needed) word boundaries, all at whitespace so the aligned-word stream
stays 1:1. **Subtle bug found while fixing:** the first word-packer summed *isolated* per-word
token counts, which omit the inter-word **space tokens** (~1 per gap) — so a 90-word paragraph
under-counted by ~90 and never crossed the budget. Fixed by costing +1 per joining space, a
conservative pack budget (460), and an `EmitVerified` safety net that hard-splits any piece whose
*actual* token count still exceeds 508. `KokoroSynthesisService` now uses `ChunkForSynthesis`.

Verified via CLI on a 1.4 KB markdown doc: the 545-token paragraph splits to 433 + 111; all 8
chunks synthesize; **86.2 s audio in 3.4 s = 25.5× real-time** (per-chunk 11–34× — short chunks
lower due to fixed per-call + phonemization overhead). Known refinement: an over-512-token single
sentence is split mid-sentence on a word boundary → a brief prosody seam; could prefer clause
(comma) boundaries. The CLI `--backend kokoro --onnx-dir <model> --data-dir <espeak data> --voice
<name> --text-file <md> --out <wav>` is now the headless long-form test path.

## Run 19 — 2026-06-10 — Batching experiment (model surgery)

Goal: batch multiple chunks in one ONNX Run for higher throughput. Wrote a batched forward
(`scripts/kokoro_export/batched_forward.py`) and validated per-item vs sequential (log-spectral).

**The surgery works — for equal-frame items.** Required: (a) pass real per-item `input_lengths`
→ masks (the model already threads `text_mask`/`attention_mask`); (b) **pack the bare prosody
`predictor.lstm`** (bidirectional, no mask — its backward pass leaked padding into shorter items'
tokens, corrupting `pred_dur`); (c) vectorized duration-expansion length regulator (cumsum/arange
instead of `repeat_interleave`); (d) drop the `.squeeze()`. Result: two **identical** items
batched → both log-spec 0.12 vs sequential. ✓

**Blocker: AdaIN over the frame axis pollutes on length mismatch.** A *variable*-length batch
corrupts the shorter (right-padded) item **throughout** (log-spec 0.83–1.3, not a boundary
effect). Diagnosis: Kokoro's predictor `F0Ntrain` and the iSTFTNet decoder use AdaIN (instance
norm over time); the padding frames pollute the per-item mean/variance. Confirmed it's not the
LSTMs (packing `F0Ntrain.shared` made it *worse*, 0.84→1.0) and not a cliff (replicating the last
real frame into the padding made it worse still, →1.3). The equal-length control (identical items,
no padding) is clean, so the surgery is correct — padding is the sole cause.

**Why bucketing doesn't save it:** the pollution is at the **frame** level, and frame counts vary
continuously — two items with the *same token count* still produce different `pred_dur` sums, so
they'd need frame-level padding and pollute. There's no cheap bucketing key.

**Verdict: not worth pursuing.** Correct variable-length batching would require masking the AdaIN
statistics in *every* norm of the predictor F0/N blocks and the whole iSTFTNet decoder — i.e.
re-architecting the vocoder's normalization — plus a hard ONNX export (masked instance-norm,
pack_padded, and the dynamo attention guard from Run 2). The upside is unproven: PyTorch CUDA is
broken in this venv (cuDNN version mismatch), so the GPU batching speedup couldn't be measured.
Against an already-27×-real-time baseline that streams far ahead of playback, the ROI is poor.
`batched_forward.py` is kept as the record (it correctly batches equal-frame items). If more speed
is ever wanted, fp16 quantization (Run 14's deferred phase 3) is the far better lever.

## Run 20 — 2026-09-11 — Batching, revisited: the two facts Run 19's verdict rested on

Run 19 closed batching as "not worth pursuing" on two supports: (a) the upside was **unproven** —
PyTorch CUDA was broken in the venv, so the speedup was never measured; (b) correct variable-length
batching would need "re-architecting the vocoder's normalization." Both are re-examined here,
because support (a) has since evaporated: `requirements.txt` now pins `nvidia-cudnn-cu13==9.23.0.39`
to match the system libcudnn9, and `torch 2.10.0+cu128` runs conv on the 3090 without complaint.

### The upside, finally measured (RTX 3090, fp32, equal-length batch — the case Run 19 got right)

136 tokens/item, 8.43 s audio/item:

```
B=1  sequential   107.4 ms/item   RTF  78.4x
B=2               73.7 ms/item    RTF 114.4x   speedup 1.46x
B=4               52.7 ms/item    RTF 159.8x   speedup 2.04x
B=8               43.1 ms/item    RTF 195.4x   speedup 2.49x
B=16              40.7 ms/item    RTF 207.0x   speedup 2.64x
B=32              38.5 ms/item    RTF 218.8x   speedup 2.79x
```

**Saturates at ~2.6x by B=8** — not linear. The graph is already wide enough to fill the GPU at
B=1, so batching recovers launch overhead and little else. Any real workload pays a further
fill-efficiency tax on top (a 4-item mixed-length batch below is 64%/64%/0%/52% padding, so ~55% of
the batched work is spent on padding). Realistic expectation is well under 2x, not 8x.

### Where the time goes (answers "isn't the decoder the cheap part that could go on CPU?")

```
total 109.5 ms/call     decoder 51.75 ms (47.3%)   bert 11.40 ms (10.4%)
                        text_encoder 6.33 ms (5.8%)   bert_encoder 0.09 ms (0.1%)
```

The decoder is the **expensive** half, not the cheap part — it is the iSTFTNet vocoder upsampling to
24 kHz. (The absent `predictor` row is a hook artifact: `forward_with_tokens` calls its submodules
directly rather than `predictor.forward`, so that stage is the residual ~36%.) Offloading it to CPU
runs the wrong way: whole-model CPU is 4x RTF against 78x on CUDA (Run 14).

### The blocker is 70 copies of one module, not a re-architecture

All time-axis norms are the same class behind the same attribute:

```
70  InstanceNorm1d   (affine=True, track_running_stats=False, eps=1e-5)  ← all at AdaIN's `.norm`
 3  AdaLayerNorm     ← normalizes over CHANNELS per frame; padding cannot pollute it
 6  LayerNorm
```

So masked instance-norm is one class plus a swap loop, not 70 edits. The decoder changes the frame
rate (stride/upsample), so the mask is rebuilt at each site from the per-item real **fraction**
against whatever length the tensor has there.

### Masking helps, packing reverses sign, neither closes the gap

2x2 over AdaIN masking and packing the frame-level `F0Ntrain.shared` LSTM, log-spec L1 vs sequential:

```
AdaIN mask   F0N packed |  item0 (61% padded)   item1 (unpadded)
     False        False |        0.8367                0.1398     ← Run 19's number, reproduced
     False         True |        1.0133                0.1424     ← Run 19 saw this get worse. It does.
      True        False |        0.5432                0.1391
      True         True |        0.4634                0.1433     ← best
```

⚠ Packing the frame-LSTM **helps once the norms are masked** (0.54 → 0.46) and **hurts when they
are not** (0.84 → 1.01). Run 19 tested it in the unmasked condition and concluded it was harmful;
that conclusion was conditional on a variable it did not hold fixed.

⚠ Calibration the earlier runs lacked: this reimplementation at **zero** padding scores **0.128**
against the true `forward_with_tokens`. So 0.12–0.14 is the **noise floor**, not "parity" — the same
band Run 14 measured for CPU-vs-CUDA and the user judged "basically identical."

### Padding reach is pad-amount-independent — which rules out statistic dilution

B=1 with forced extra frames, isolating padding from batching entirely:

```
extra pad   mask |  mean logspec
        5  False |  0.6773       5 frames of padding does as much damage as 200
        5   True |  0.5603
      200  False |  0.8961
      200   True |  0.5509
```

Five padding frames against 92 real ones cannot move an instance-norm mean by that much. So the
dominant mechanism is **not** dilution of the AdaIN statistics. Stage-by-stage diff, padded vs
unpadded, over the real-frame region only:

```
en     0.000000     ← length regulator exact
asr    0.000000     ← text path exact
F0     0.019337     ← ~2%
N      0.021402     ← ~2%
audio  1.237422     ← 124%
```

**The decoder amplifies a 2% F0/N perturbation into a 124% waveform difference.** That is expected
behaviour for a harmonic vocoder, not corruption: a slightly different F0 moves every harmonic and
drifts the excitation phase cumulatively. It predicts the observed error profile exactly — the first
decile of the item is clean at 0.056, the middle sits at ~0.45, the last decile is 0.997 (error
growing with elapsed time = accumulated drift), and it explains why 5 frames of padding suffices.

**This puts the metric itself in question.** Log-spec L1 is hypersensitive to F0 drift, and Run 19's
entire verdict rested on it. A 2% F0 error is ~0.34 semitones — measurable, near-inaudible. Whether
the residual is audible is not a question log-spec L1 can answer, so it went to the ear (Run 20e:
A = sequential, B = masked+packed, C = unfixed, on the most-padded item of a mixed-length batch).

Pending: the listening verdict. It decides between "masking makes batching viable" and "the residual
is real," and no further surgery is worth designing until it lands.

### Run 20e — the listening verdict

User, on the most-padded item (64% padding) of a mixed-length batch: **"C sounds different, but
not wrong, per se."** C is the *unfixed* batched render — the 0.82 log-spec condition Run 19
described as corrupted throughout. It is audibly acceptable. B (masked+packed, 0.46) sits between
C and sequential, so it is acceptable by implication.

**This retires Run 19's verdict.** Log-spec L1 was measuring F0/phase drift, which the stage diff
above shows is the dominant term and which a harmonic vocoder produces from any F0 perturbation at
all. The number was real; the conclusion drawn from it was not.

What survives from it is a *different* concern the metric was standing in for: the audio is a
function of **batch composition**. The same sentence renders differently depending on its
neighbours. For a tool that caches generations and whose TTS bugs are diagnosed by re-exporting and
comparing clips, non-reproducibility is a real cost even when each individual render is fine.

## Run 21 — 2026-09-11 — The no-surgery alternative, and why it doesn't generalize

Padding is the sole cause of *both* the drift and the wasted work, so the obvious dodge is to not
pad: concatenate sentences into one longer chunk. Four sentences, same content:

```
4 separate chunks : 245 tokens  15.35s audio  205.1 ms  RTF  74.8x
1 joined chunk    : 242 tokens  14.55s audio  107.8 ms  RTF 135.0x     joining = 1.90x
```

1.90x for free — no surgery, no batch axis, no composition dependence. But it does not generalize:
`ChunkForSynthesis` only ever splits *down* to `PackBudgetTokens` (460); it never packs *up* across
paragraphs, and it must not — `ParagraphChunker` output is the unit the app emits per-paragraph
segments and WAVs for (PR #128). Within a paragraph, chunks are already packed. Across paragraphs,
joining is forbidden by the segment contract.

So joining only pays on short paragraphs, which are exactly the ones that may not be merged.
**Batching is the only lever that crosses a paragraph boundary** — which is also where it is most
natural, since paragraphs are already independent output units.

### Fill efficiency: Run 19's "no cheap bucketing key" is wrong

Run 19 dismissed bucketing because frame counts vary continuously. But the key does not have to
*predict* frame count, only correlate with it. Pooled chunk lengths from 2 real export jobs (90
chunks), fill = useful work / padded work:

```
B=4    natural order 69.3%    length-sorted 96.2%
B=8    natural order 59.2%    length-sorted 91.5%
B=16   natural order 53.2%    length-sorted 86.3%
```

Effective speedup = raw x fill: **~2.0x at B=4, ~2.3x at B=8**. Unsorted it would be ~1.1x, which
is the number Run 19 was implicitly assuming. Export is a bulk path with every paragraph known up
front, so sorting is free there; it is not available to an interactive/streaming path.

## Run 22 — 2026-09-11 — It exports, and the first export was silently wrong

Legacy exporter, dynamic `batch` *and* `tokens` axes, masked instance-norm written in plain reduce
ops. Exports clean at 326 MB and loads in ORT with `['batch','tokens']` inputs.

⚠ **The first attempt dropped `pack_padded_sequence` as ONNX-hostile and was wrong at B>1** — and
the export gave no sign of it. Caught only by running in ORT at batch sizes it was not traced with
and comparing per-item frame counts:

```
                item0 frames:  B=1    B=3    B=4
  unpacked                     142    137    125     ← pred_dur shifts with batch composition
  packed                       142    142    142     ← matches the solo run exactly
```

The reasoning error: "packing was only worth 0.54 → 0.46" conflated **two different LSTMs**.
That figure is for `F0Ntrain.shared` (frame-level, optional). The one dropped was
`predictor.lstm` — token-level, bidirectional, and the exact site Run 19 flagged, whose backward
pass reads padding and corrupts `pred_dur`. Packing it is **not optional**, and `torch.onnx` does
export it (ONNX's LSTM op carries a native `sequence_lens` input).

Duration stability matters beyond audio: `pred_dur` drives word-level alignment (Run 15), so stable
durations mean karaoke timing is identical to the sequential render.

```
packed graph, ORT:  B=1 [142]                     logspec 0.139
                    B=3 [142,143,156]             0.575 0.562 0.133
                    B=4 [142,143,156,189]         0.584 0.549 0.599 0.128
```

Padded items land at ~0.55-0.60 — better than the 0.82 the user judged "not wrong" — and unpadded
items sit at the 0.13 noise floor.

## Run 23 — 2026-09-11 — ⚠ The first ORT benchmark measured nothing (CPU fallback)

The export venv has **CPU-only `onnxruntime` 1.27** (`available: ['Azure','CPU']`), so
`providers=["CUDAExecutionProvider",…]` was silently ignored and the "CUDA" benchmark ran on CPU.
Piping through `| tail` buffered the `batched EP:` line that would have said so at once.

⚠ Two process lessons, both cheap: **assert the provider** (`assert
sess.get_providers()[0]=="CUDAExecutionProvider"`) rather than requesting it, and don't pipe a
long-running benchmark through `tail` — it hides the diagnostic line until exit. Redone with a
venv that has the CUDA EP (`.venv-chatterbox-export`), feeding pre-captured inputs from an `.npz`
so the GPU venv needs only numpy + ORT.

### The number, and why one baseline is not trustworthy

8 sentences, 30.3 s audio, interleaved reps so contention drift hits both arms equally:

```
sequential loop (8 Runs)             1238.7 ms   RTF  24.5x
batched B=4  (2 Runs)                 348.3 ms   RTF  87.1x   3.56x
batched B=8  (1 Run)                  279.6 ms   RTF 108.5x   4.43x
sum of shape-warmed single calls      413.6 ms             -> batching vs this floor 1.48x
```

⚠ Those two baselines disagree by 3x on identical work. Per-call latency measured in isolation is
23–69 ms at RTF 66–85x — consistent with Run 14 — yet the same 8 calls in a loop take 1239 ms.
Batching's honest range is therefore **1.5x–4.4x depending on which baseline is fair**, and
resolving that mattered more than the headline.

⚠ Length-sorting did **not** show its predicted benefit here (B=8: 5.25x sorted vs 5.24x natural)
— with 8 items at B=8 there is only one batch, so sorting has nothing to do. Run 21's fill
numbers stand as arithmetic but are **unvalidated end-to-end**; a corpus of many chunks is needed.

## Run 24 — 2026-09-11 — The 3x gap is cuDNN algo search, and it is a one-line fix

The user, watching nvtop: *"it gradually uses more GPU, eventually hits 100."* A staircase ramp
during a loop over cached shapes is ORT re-tuning per shape. ORT's CUDA EP defaults
`cudnn_conv_algo_search` to **EXHAUSTIVE**, which re-runs on every new conv input shape — and every
chunk the app synthesizes is a different length, hence a different shape.

```
cudnn_conv_algo_search=EXHAUSTIVE (ORT default)   1231.3 ms   RTF 24.6x
cudnn_conv_algo_search=HEURISTIC                  1224.4 ms   RTF 24.8x
cudnn_conv_algo_search=DEFAULT                     755.4 ms   RTF 40.1x    ← 1.63x, no model change
```

Also 9,360 `OP Conv(/decoder/…) running in Fallback mode. May be extremely slow.` warnings, all on
decoder convs — consistent with Run 20's finding that the decoder is 47% of runtime.

**The app is paying this.** `OrtSessionBuilder.cs:523` appends CUDA with only `device_id` and
`use_tf32`; `cudnn_conv_algo_search` is never set, so every CUDA session in Vernacula — not just
Kokoro — inherits EXHAUSTIVE. Conv-heavy models (the iSTFTNet decoder, DiariZen, Sortformer,
the vocoder) are the ones that would pay most.

**Not yet verified, and required before changing anything:** this was measured on ORT **1.23.2**
(the chatterbox venv) while the app ships ORT **1.29** — the default and the tuning both may have
moved. DEFAULT being faster is also workload-specific; it must be measured per model, not applied
globally on this one result.

### Where this leaves batching

1.63x for a one-line EP option, against ~1.5–4.4x for a batched graph that needs masked
instance-norm in 70 sites, `pack_padded` on the token LSTM, a new 4-input export, batch-aware
scheduling in `KokoroTts`, and output that varies with batch composition. **The EP option should be
measured and landed first** — it is nearly free, it helps every model, and it shrinks the remaining
gap that batching would have to justify.

## Run 25 — 2026-09-11 — Confirmed on ORT 1.29, and the harness was measuring the wrong workload

C# probe using the app's own ORT 1.29 and the exact provider options `OrtSessionBuilder` passes:

```
cudnn_conv_algo_search=(unset = app today)   1249.2 ms   RTF 24.3x
cudnn_conv_algo_search=EXHAUSTIVE            1278.3 ms   RTF 23.7x    ← unset == EXHAUSTIVE, confirmed
cudnn_conv_algo_search=HEURISTIC             1394.8 ms   RTF 21.7x
cudnn_conv_algo_search=DEFAULT                851.0 ms   RTF 35.6x    ← 1.47x
```

`cudnn_conv_use_max_workspace` made no difference (0 vs 1: 791 vs 788 ms).

⚠ The 9,360 `Conv(...) running in Fallback mode. May be extremely slow.` warnings come from
**DEFAULT**; EXHAUSTIVE emits **zero**. The path ORT warns about is the fast one for this graph.
The warning is not a defect signal here.

**Correctness.** Output lengths identical (durations unchanged). Waveform relRMS 7-9%, which looks
alarming but is the vocoder's F0/phase sensitivity again — log-spec L1 is **0.126-0.142 on all 8
items**, the same noise floor as CPU-vs-CUDA. User on the A/B pair: **"They are indistinguishable."**

### ⚠ KokoroPerf said the opposite, and the harness is wrong for this question

```
                 med_ms DEFAULT   med_ms EXHAUSTIVE
short  (11 ph)        39.2               19.9        ← EXHAUSTIVE 2x faster
medium (47 ph)        64.4               42.2
long  (146 ph)       197.1              164.7
```

The harness times `iters` repetitions of **one** utterance, so the shape never changes. Resolving
the contradiction — same call count, only whether the shape varies between calls:

```
algo           repeat 1 shape    cycle 8 shapes    penalty for variety
EXHAUSTIVE          434.3 ms          1270.6 ms          2.93x
DEFAULT             568.2 ms           789.3 ms          1.39x
```

**ORT's cuDNN algo cache does not retain every shape — EXHAUSTIVE re-tunes on every shape CHANGE.**
So EXHAUSTIVE wins only when one shape repeats, which is what KokoroPerf does and what the app
never does: a document export is a sequence of differently-sized chunks. On that workload DEFAULT
wins 789 vs 1271 (1.61x). The harness's own `warm_ms` (first call on a cold shape) already agreed —
medium 86 ms DEFAULT vs 185 ms EXHAUSTIVE — only the repeated-shape `med_ms` column dissents.

⚠ This also reframes Run 14's CUDA figures: they came from this harness, so they are the
amortized-EXHAUSTIVE best case on a repeated shape, not what a real export sees.

**Shipped:** `cudnn_conv_algo_search` is now a per-caller option on `AppendCuda`/`Create`/
`CreateCachedSession`/`SessionLoader`, defaulting to null (unchanged) everywhere except `Kokoro`,
which passes `"DEFAULT"`. Per-caller rather than global because the right setting depends on shape
stability, not on the model. `VERNACULA_ORT_CONV_ALGO` overrides for measurement.

**Left open:** other conv-heavy CUDA sessions (DiariZen, Sortformer, the vocoder, ASR encoders)
were not measured. Any with varying input shapes are likely paying the same 2.93x re-tune penalty.

## Run 26 — 2026-09-11 — The model is nondeterministic, so "exact" has a floor

Before chasing fidelity, calibrate the target. Same input, twice, CPU, eval mode:

```
dur_equal=True   max|diff| = 1.095e-01   logspecL1 = 0.129265
```

**Kokoro does not reproduce itself.** `SourceModuleHnNSF` draws `torch.randn_like` for the
harmonic-plus-noise excitation on every call (`noise_amp = uv*0.003 + (1-uv)*0.1/3`). So the
0.11–0.20 band Run 14 called "the model's own nondeterminism" is literally that, and the
reimplementation's 0.132 is indistinguishable from running the real model twice.

⚠ Consequence for every earlier run in this log: **0.13 is the floor, not a pass mark.** A batched
render must reach 0.13 to be "no fidelity cost"; 0.46 was real excess, not measurement noise.

## Run 27-31 — 2026-09-11 — Making padding invisible

With `torch.randn_like` stubbed to zeros the model is deterministic (solo vs solo = **0.00000**),
which makes the padding effect measurable on its own.

**Controls first** — batching itself is not the problem:

```
B=1 through the batched code vs solo        0.128   (floor)
batch of 4 IDENTICAL items vs solo          0.141   (floor, 0% padding)
same batch run twice                        0.142   (floor)
batch with 40% padding vs solo              0.273   EXCESS
```

⚠ **The error is a step function in padding, not a gradient.** Sweeping padding on one item, noise
off:

```
pad frames   pad %   logspec   F0 mean abs err (Hz)
         2    1.4%    0.2634                 0.0455
        20   12.3%    0.2633                 0.0455
       160   53.0%    0.2658                 0.0455
```

Two frames of padding do exactly as much damage as 160. **This kills length-bucketing as a fidelity
strategy** (Run 21's fill numbers remain a throughput argument only): padding must be eliminated,
not minimized.

**Where it enters.** Stage diff, batched vs solo, real region only: `bert`, the duration encoder,
`pred_dur`, `en` and `asr` are all at float noise (~1e-6) — only `F0` (3.2e-4) and `N` (1.2e-3)
diverge. That looks negligible until you follow it: `F0_conv` 5e-4 → decoder `encode` 2e-2 →
`generator` 4.4e-1. Each AdaIN divides by a std, so relative error compounds through the stack.

⚠ The masking was incomplete in a way that is easy to miss. `AdaIN1d.forward` is
`(1 + gamma) * self.norm(x) + beta` — re-zeroing inside the InstanceNorm is **undone by the
`+ beta` outside it**, so the padding region carries `beta` into every downstream conv.

```
                                           logspec   F0 err (Hz)
masked statistics only                      0.2634     0.1363
+ re-zero after the whole AdaIN1d           0.1514     0.0060
+ re-zero after ALL convs                   0.2189→1.0239 (worse with more padding)   0.0000
+ re-zero after PREDICTOR convs only        0.0877     0.0000    ← taken
```

⚠ Re-zeroing every conv makes F0 exact but wrecks the audio, and worse the more padding there is:
the generator mixes `[B, T, 1]` layouts and an iSTFT, where a last-dim mask is simply wrong. The
predictor's frame axis is the last dim throughout, so masking is well defined there. Final recipe:
**masked AdaIN statistics + packed `predictor.lstm` and `F0Ntrain.shared` + re-zero after every
`AdaIN1d` + re-zero after predictor convs.** F0 becomes bit-exact and the deterministic residual
(0.0877) sits *below* the model's own run-to-run variance.

## Run 32 — 2026-09-11 — Verified, noise on, worst-case padding

Batch of 8 in natural order (46–77% padding — deliberately not length-sorted):

```
item  tok  pad%   batched vs solo    floor
   0   58   60%            0.1585   0.1382
   2  142    0%            0.1443   0.1494
   4   26   77%            0.1558   0.1234
   7   56   59%            0.1554   0.1266
```

Durations identical to solo for every item. User on the A/B of the worst case (77% padded):
**"Indistinguishable."** Compare where this started: 0.82.

## Run 33 — 2026-09-11 — Scaling: compute saturates long before VRAM

RTX 3090, length-sorted, full-fidelity path:

```
   B   per-item ms      RTF   speedup   VRAM GB
   1         74.1     47.9x         -         -
   8         33.9    117.3x     2.19x      1.40
  16         32.3    123.0x     2.29x      2.22
  64         29.7    133.9x     2.50x      7.09
 128         29.5    134.7x     2.51x     13.60
```

**Throughput saturates at B≈8–16; VRAM never binds** (13.6 GB of 24 at B=128). The graph fills the
GPU by B=8, so ~2.5x is the architectural ceiling here, and B=16 captures nearly all of it at
2.2 GB. Batching beyond 16 buys ~2% for 6x the memory.

## Run 34-36 — 2026-09-11 — Exported, and the two wins overlap

Exported with the full-fidelity recipe (forward hooks are captured by tracing): 326.4 MB, dynamic
`batch` and `tokens` axes. In ORT, durations match the solo render at B=1, 2, 3 and 5, and fidelity
holds at the floor.

⚠ **The cuDNN fix (#190) and batching are not additive — they overlap almost completely**, because
a batched Run is *one* Run and therefore has no shape changes to re-tune:

```
batch=1 graph, sequential, EXHAUSTIVE   1264.1 ms   24.0x   ← what ships today
batch=1 graph, sequential, DEFAULT       818.1 ms   37.1x   ← #190
batched graph B=8, EXHAUSTIVE            347.6 ms   87.2x
batched graph B=8, DEFAULT               350.6 ms   86.5x   ← setting is irrelevant once batched

cuDNN fix alone                  1.55x
batching MARGINAL on fixed base  2.33x
both together                    3.61x        (24.0x -> 86.5x)
```

#190 still earns its place: it is what the B=1 path gets, and it applies to every other CUDA
session in the app.

**The batched graph replaces the batch=1 graph rather than shipping beside it** — at B=1 it is
1.08x *faster* than the current one, so there is no dual-model distribution cost:

```
original graph, 8 chunks sequential    795.4 ms   38.1x
batched graph B=1                      735.5 ms   41.2x   1.08x
batched graph B=8                      350.8 ms   86.5x   2.27x
```

## Run 37 — 2026-09-11 — Wired end-to-end, and the bug only an end-to-end test would find

C# side: `Kokoro.SynthesizeBatch`, `KokoroTts.SpeakAlignedBatch` (alignment extracted into a
shared `Align` so the single and batched paths cannot drift), an optional
`SegmentBatchSynthesizer` on `SegmentedSynthesis.Run`, and `KokoroSynthesisService` flattening
every segment's chunks into one call and regrouping by offset.

`Kokoro` prefers `kokoro_batched.onnx` when the model directory has one and falls back to
`kokoro.onnx`, so existing model downloads keep working.

⚠ **The batched graph broke every single-item caller** — `Speak`, `SpeakAligned`, the CLI:

```
[ErrorCode:Fail] Non-zero status code returned while running Unsqueeze node.
Status Message: Missing Input: input_lengths
```

Loading the batched graph makes `SynthesizeWithDurations` fail, because it does not supply
`input_lengths` and does not expect a leading batch axis. Testing only the new batch API would
have shipped this. `SynthesizeWithDurations` now routes through `SynthesizeBatch` as a batch of
one when the batched graph is loaded.

Streaming semantics are preserved deliberately: the FIRST segment is still rendered alone, so
time-to-first-audio is unchanged and only the tail batches. Results are emitted strictly in order
either way.

End-to-end through the shipping classes (RTX 3090, 8 chunks, CUDA):

```
durations + audio lengths vs one-at-a-time : identical for all 8 items
empty item spliced at its own index        : yes
sequential   697.1 ms   RTF 38.8x
batched      299.6 ms   RTF 90.2x   2.33x
```

Against the 24.0x that ships today, that is **3.8x**.

## Run 38 — 2026-09-11 — Real paragraphs are not uniform, and that halves the win

⚠ **Correction to Run 37's 2.33x.** That was measured on eight phoneme strings of similar
length. Driving the actual Avalonia path (`KokoroSynthesisService` → `ParagraphSegmenter`) on a
markdown document gave **1.21x**, and the raw synthesis call on the same paragraphs gave **0.99x —
slower than sequential.** Phonemization was not the cause (7 ms against 760 ms, 1%).

The cause is length variance. A batch is padded to its longest item, and an ordinary document
mixes one-line headings with long paragraphs:

```
phoneme lengths: 17, 96, 23, 173, 13, 54, 55, 87    max/mean 2.67
fill if run as ONE batch of 8: 37%    <- 63% of the GPU work is padding
```

Run 21's fill-efficiency arithmetic was right and was under-weighted here because Run 23's
8-item test had only one batch, where sorting cannot help. With enough items it is the whole game.
36 paragraphs, phoneme length 7/50/173 (38% fill as one batch):

```
sequential                              3398.6 ms      -
fixed 16, document order                2963.5 ms   1.16x
fixed 16, length-sorted                 2188.5 ms   1.55x
fixed  8, length-sorted                 1929.4 ms   1.76x
fixed  4, length-sorted                 1967.3 ms   1.49x
adaptive sorted, max/min<=1.25, cap 16  1899.4 ms   1.79x
adaptive sorted, max/min<=1.50, cap 16  1852.0 ms   1.84x   ← taken
adaptive sorted, max/min<=2.00, cap 16  2005.0 ms   1.70x
```

Sorting is worth more than batch size: document-order batching never beats 1.21x at any size,
while sorted adaptive grouping reaches 1.84x. Too tight a spread (1.25) fragments into batches too
small to fill the GPU; too loose (2.0) reintroduces padding.

**This is a throughput decision only.** Fidelity does not depend on grouping — the padding error
is a step function and is masked out either way (Run 27-31) — so grouping is free to optimise for
fill. Implemented inside `KokoroTts.SpeakAlignedBatch`, which already phonemizes and so knows the
lengths; callers just hand it work and get results back in input order.

### End to end, the real Avalonia path

37-paragraph markdown document (headings + paragraphs), RTX 3090, CUDA:

```
streaming order        identical (0..36)
paragraph count        identical
word count             identical, 0 text mismatches
max word-timing delta  0.000 ms
wav bytes              17664058 / 17664058  (identical)

warm synthesis   4863 ms -> 2570 ms   1.89x
```

⚠ Both arms already carry the cuDNN fix, so **1.89x is batching's marginal gain**; against what
ships today (EXHAUSTIVE, unbatched) the combined figure is ~2.9x, not the 3.8x Run 37's uniform-
length measurement suggested.
