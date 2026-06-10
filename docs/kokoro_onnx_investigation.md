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
