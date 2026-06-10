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
