# OmniVoice → ONNX export investigation

Tracking the Phase-1 effort to export `k2-fsa/OmniVoice` (Apache-2.0, diffusion-LM TTS) to
ONNX and validate a Python harness against the PyTorch reference. Plan + rationale:
`scripts/omnivoice_export/`. Precedent: `docs/kokoro_onnx_investigation.md`.

Component split (confirmed from `omnivoice/models/omnivoice.py` + HF configs):
- **Transformer** (diffusion denoiser): `_prepare_embed_inputs` → Qwen3 (hidden 1024) →
  `audio_heads`. In `input_ids[2B,8,S]`, `audio_mask[2B,S]`, `attention_mask[2B,1,S,S]`;
  out `logits[2B,8,S,1025]`. Run 32× per generation with CFG.
- **Higgs encoder** (`HiggsAudioV2TokenizerModel.encode`): wav[B,1,T]@24k → codes[B,8,Tc].
- **Higgs decoder** (`.decode`): codes[B,8,Tc] → wav[B,1,Tsamp]@24k. DAC, non-diffusion.

---

## Run 1 — 2026-06-13, environment setup

**Question:** Can we stand up a venv with the OmniVoice model + transformers≥5.12 (for
`HiggsAudioV2TokenizerModel`) on this box (RTX 3090, system cuDNN 9.23.1.3)?

**Commands:**
- `python3 -m venv .venv-omnivoice-export` (Python 3.12.3)
- `pip install torch==2.12.0 torchaudio --index-url .../cu130` then
  `nvidia-cudnn-cu13==9.23.0.39` + `transformers>=5.12` + onnx + onnxruntime-gpu, then
  `git+https://github.com/k2-fsa/OmniVoice.git` (v0.1.5).
- Model: `snapshot_download("k2-fsa/OmniVoice")` → `/mnt/data/models/omnivoice/k2-fsa-OmniVoice`.

**Rationale for the torch/cuDNN combo:** reuse the kokoro_export proven combo (torch
2.12+cu130, nvidia-cudnn-cu13 pinned to system) — see commit "Fix kokoro venv CUDA durably".
The repo-wide kokoro venv has transformers 5.10.2 which predates `HiggsAudioV2TokenizerModel`,
so a fresh venv is required.

**Finding:** Works, with two install gotchas. (1) Installing `transformers`/`onnxruntime`
after the cu130 torch pulled a default PyPI `torch 2.10.0+cu128` and clobbered the cu130
build (and torchaudio 2.11+cu130 → ABI mismatch with torch 2.10). Net result: torch
2.10.0+cu128 / torchaudio 2.10.0, CUDA available on the 3090, bundled cuDNN 9.10. Good
enough — the cu130 + cuDNN-pin durability goal is moot since torch 2.10 carries its own cu12
cuDNN. (2) `omnivoice` must be installed `--no-deps` (its `hatchling` build + `gradio`/
`tensorboardx`/`webdataset` training extras are irrelevant to inference); runtime needs only
`pydub` + `librosa` added on top of what we already had.

**Next:** capture in-distribution reference tensors from a real `generate()` run, then export.

---

## Run 2 — 2026-06-13, capture + first export + first parity

**Commands:** `capture_reference.py` (CUDA), `export_omnivoice.py`, `parity_check.py`.

**Capture shapes** confirm the architecture exactly: enc wav `(1,1,55680)`@24k →
codes `(1,8,58)` (hop 960); transformer `input_ids(2,8,174)` / `audio_mask(2,174)` /
`attention_mask(2,1,174,174)` → `logits(2,8,174,1025)` (the `2` is the CFG cond+uncond
stack); dec codes `(1,8,80)` → wav `(1,1,76800)`.

**Export:** all three graphs exported via the **legacy** TorchScript exporter. Dynamo only
failed for a missing `onnxscript` dep (not a real blocker); legacy is the kokoro-proven path
anyway. SDPA `is_causal` traced as constant False (correct — bidirectional path baked in).
Two follow-ups noted: (a) legacy dumped the transformer's 0.6B weights as **loose external
files** in `onnx/` (e.g. `model.llm.embed_tokens.weight`, `onnx__MatMul_*`) — functional but
messy; re-export with a single `.onnx.data`. (b) Codec export baked several shape-dependent
`if` branches as constants (conv-length checks) → **generalization to other audio lengths is
unverified**; needs a dynamic-axes test with a different-length input.

**First parity (CUDA EP, default TF32):** transformer argmax-agree 0.9964 / max_abs 5.5e-2,
encoder code-match 0.972, decoder log-spectral 0.093. Transformer + encoder looked like
"FAIL" — but that was a tolerance/setup artifact, not an export bug (Run 3).

---

## Run 3 — 2026-06-13, diagnosing the discrepancies

**Question:** are the transformer/encoder discrepancies export bugs, or numerical?

**Finding A — transformer = ORT CUDA TF32.** Re-running the transformer graph with the CUDA
EP option `use_tf32=0` gives **argmax-agree 1.000000, max_abs 1.45e-4, mse 4.3e-10** (CPU EP
identical). With TF32 on it's 0.9964 / 5.5e-2. So the export is mathematically exact; ORT's
default TF32 matmul introduces ~1e-2 logit error that **compounds through the 32-step
diffusion loop** into audibly different (but still valid) speech — this is why the first
end-to-end run measured log-spectral 1.86 vs the PyTorch reference.

**Finding B — encoder = torch CPU-vs-CUDA device sensitivity in the RVQ argmin.** Isolation
test: `ORT-CPU vs torch-CPU = 1.000 across all 8 codebooks` (exact export), while
`torch-CPU vs torch-CUDA = 0.890`. The 11% gap is purely the RVQ nearest-codebook `argmin`
flipping indices when continuous features sit near codebook boundaries — inherent to the
model, independent of ONNX. The "FAIL" came from validating a CPU-ORT graph against a
**CUDA-captured** reference.

**Implication:** validate parity on the **same compute substrate** as the reference. Adopt
the user's strategy explicitly — **parity in full fp32 (CPU / CUDA-with-TF32-off) first,
CUDA/TF32 performance later.** Added a `--provider {cpu,cuda-fp32,cuda-tf32}` switch to
`parity_check.py` and `infer_onnx.py` (default `cpu`); tightened the transformer threshold to
max_abs < 5e-3 so it actually catches TF32.

---

## Run 4 — 2026-06-13, faithful fp32 parity PASS

**Commands:** `capture_reference.py --device cpu`, then `parity_check.py --provider cpu` and
`infer_onnx.py --provider cpu` (apples-to-apples on CPU).

**Finding — PASS:**
- transformer: argmax-agreement **1.00000**, max_abs **1.98e-4**, mse 1.3e-9
- encoder: code-match **1.00000**
- decoder: log-spectral-L1 **0.0008** (waveform corr 1.0)
- **end-to-end ONNX vs PyTorch: log-spectral-L1 0.0012** — perceptually identical.

The three graphs are faithful fp32 exports of OmniVoice's neural components; the Python
harness (`infer_onnx.py`) drives the full diffusion pipeline through them and reproduces the
PyTorch output. **Phase-1 parity milestone met.** WAVs in `capture/` (`py_reference.wav` vs
`onnx_e2e.wav`) for the listen-test.

**Next (later phases):** (1) re-export transformer to a single `.onnx.data`; (2) verify codec
graphs generalize to other audio lengths (the shape-baked conv branches); (3) CUDA/TF32
performance pass + fp16/int8; (4) C# runtime port of the host loop.

---

## Run 5 — 2026-06-13, fixing the silent reference (capture-config bug, not export)

**Symptom:** user reported both `py_reference.wav` and `onnx_e2e.wav` were quiet and
unintelligible. But they were *bit-identical* (rms 0.0195, peak 0.041) → ONNX faithfully
reproduced PyTorch; the **PyTorch reference itself was bad**.

**Root cause:** the Run-1..4 capture used voice-**clone** mode with a Kokoro sample as
`ref_audio` and a *fabricated* `ref_text` ("quick brown fox") that did not match the audio.
OmniVoice prepends `ref_text` + ref-audio-tokens as conditioning and expects them to
correspond; the mismatch derailed the diffusion. (The Kokoro ref's low RMS also scaled the
output down via the ref-RMS post-process.) Diagnostic: design-mode and auto-mode generations
with model defaults — and even my greedy `num_step=16` config — all produced normal-level,
intelligible speech (peak 0.50). User confirmed all three intelligible ("good even").
**The encoder ignores `ref_text`, so graph parity was never affected — only audio quality.**

**Fix:** `make_reference.py` synthesises a matched reference (design-mode clip of a fixed
sentence → `capture/ref_voice.wav` + `.txt`), so `ref_text` provably matches the audio.
`capture_reference.py` / `infer_onnx.py` now default to it and read the transcript from the
sidecar `.txt`. The fabricated-ref / Kokoro-sample default is gone.

**Re-validation (CPU/fp32, matched clone).** New shapes (enc input 78720→codes 82,
transformer S=194, dec codes 74) differ from the Run-2 export trace (enc 55680/codes 58,
S=174, dec 80), so this **also exercises the graphs at unseen lengths**:
- transformer argmax **1.00000** / max_abs 2.06e-4; encoder code-match **1.00000**;
  decoder log-spectral **0.0001**; **end-to-end vs PyTorch log-spectral 0.0033**.
- Output now at speech level (peak 0.299, rms 0.050); py_reference and onnx_e2e bit-identical.

**Implication:** the codec + transformer ONNX graphs **generalize across audio/sequence
lengths** despite the shape-baked-branch tracer warnings (Run 2) — at least across this
range. Follow-up #2 largely resolved; a wider length sweep would fully close it.

User confirmed the corrected clone output (ref_voice → py_reference/onnx_e2e) sounds good.
**Phase-1 complete.**

---

## Run 6 — 2026-06-13, external-data consolidation (follow-up #1)

**Question:** can the transformer graph be made portable (single sidecar instead of ~200
loose external-weight files)?

**Finding:** added `_consolidate_external_data()` to `export_omnivoice.py` — reload the
exported graph and re-save with `all_tensors_to_one_file=True` →
`omnivoice_transformer.onnx.data` (2.45 GB), then delete the loose files. One gotcha:
`onnx.load(load_external_data=True)` *clears* each tensor's `external_data` location, so the
old locations must be collected from a `load_external_data=False` pass FIRST (the initial
attempt found 0 to delete and orphaned the loose files). After the fix, `onnx/` holds only:
`omnivoice_transformer.onnx` (1.5 MB graph) + `.onnx.data` (2.45 GB), `higgs_encoder.onnx`
(625 MB embedded), `higgs_decoder.onnx` (83 MB embedded), `export-report.json`. The .onnx
references exactly one external location; parity re-run (cpu) still **PASS** loading purely
from the consolidated sidecar. **Follow-up #1 done.**

Remaining (future phases): CUDA/TF32 perf + fp16/int8; C# host-loop port.

---

## Phase 2 — 2026-06-13, C# runtime port (CLI-first, greedy)

Ported the host pipeline to C# so OmniVoice runs in Vernacula like Kokoro/Chatterbox.
Greedy regime (position/class temperature = 0) — deterministic, validated to sound good,
enables tight parity. All in `src/Chatterbox.Base/` + `tests/Chatterbox.Tests` +
`tests/OmniVoiceSmoke`. Each stage validated against Python before the next.

- **A — Qwen3Tokenizer** (`Tokenization/Qwen3Tokenizer.cs`): byte-level BPE encoder. No
  runtime Qwen text-encoder existed in the repo (all decode-only / offline-baked), so it was
  built by composing the GPT-2 byte table, the EnTokenizer merge loop, and Qwen3Asr's
  tokenizer.json loader, plus the Qwen split regex + NFC + special/nonverbal handling.
  Exact match vs the Python tokenizer on 34 fixtures.
- **B — OmniVoice.cs** graph wrappers (transformer / codec encode / decode) via
  OrtSessionBuilder. Per-graph parity vs the Phase-1 capture on CPU/fp32: encoder exact,
  decoder waveform max-abs < 1e-2, transformer argmax-agreement > 0.9999.
- **C — OmniVoiceDuration.cs + OmniVoiceTextPrep.cs**: ported RuleDurationEstimator
  (script-weight table + short-text boost) and _combine_text/_prepare_inference_inputs.
  Duration estimate exact (1e-4); cond row0 + audio_mask exact for no-ref configs.
- **D — OmniVoiceTts.cs** greedy diffusion loop (CFG batch, shifted-timestep schedule,
  guidance log-prob mix, layer penalty, top-k unmask, scatter). Cond built from text exactly
  equals the captured step-0 cond; token field matches the PyTorch capture > 98% (the gap is
  ONNX-vs-PyTorch drift through the loop).
- **E — OmniVoiceSmoke** end-to-end console → WAV. Listen-confirmed intelligible cloned
  speech, quality on par with the Python reference. ~14 s / 16 steps on CPU (fp32 path).

Remaining: formal `Chatterbox.CLI --backend omnivoice` flag; CUDA/TF32 perf + fp16/int8;
Avalonia `ITtsBackend` UI backend; sampled (non-greedy) mode.

### Phase 2 perf — diffusion-loop optimization (smoke harness, RTX 3090)

Measurement-driven, via `tests/OmniVoiceSmoke` (added per-phase timers: transformer GPU vs
host scoring). Baseline CPU 16-step diffusion was 14.2 s; CUDA is ~18× (793 ms / 16 steps).
Optimized the CUDA path at 32 steps (3.08 s of audio):

| change | total | transformer | host | notes |
|---|---|---|---|---|
| CUDA baseline | 1803 ms | 1066 | 705 | host scoring single-threaded |
| + parallel scoring | 1090 ms | 976 | 95 | `Parallel.For` over (codebook,pos); CFG softmaxes are independent |
| + IO binding | 1002 ms | 882 | 98 | `OrtIoBinding` reuses pinned input/output buffers — no 12.7 MB logits alloc/step |
| + uncond-split | 880 ms | 762 | 100 | run cond (S=194) and uncond (T=74) as two B=1 passes instead of one [2,S] batch padded to S; the uncond pad was discarded anyway |
| + cond/uncond concurrency | 709 ms | 568 | 123 | `Parallel.Invoke` the two independent passes; they overlap on the GPU |

**2.54×** end-to-end on the loop (1803→709 ms); host scoring 705→~110 ms. Correctness
preserved (all 6 heavy parity tests green across repeated runs; loop token field unchanged).

### What the bottleneck actually is (CUDA-graph + fp16 probe)

Probed with `probe_cuda_graph.py` (device-IO, single forward) to find the real limiter:

- **CUDA graph capture works** (ORT tolerates the shape-massaging CPU nodes) but is only
  **~4% faster** (25.3 vs 26.5 ms/forward). So the loop is **NOT launch-bound**, and the CPU
  shape-fallback nodes are not the wall.
- **fp16 ≈ 2× per forward** on pure GPU compute (13.65 vs 26.47 ms, device-IO, no host
  download). The transformer is **memory-bandwidth bound** — dominated by streaming the
  0.6 B weights from VRAM each forward — so halving weight bytes ~halves time.

This redirects the dynamo question: since the limiter is weight memory traffic (not kernel
launches or CPU-fallback shape ops), neither CUDA graphs nor a dynamo re-export's attention
fusion would help much — they target launch/compute, not weight bandwidth. **fp16 is the
right lever; CUDA graphs and dynamo are not pursued.**

### fp16 transformer — ADOPT (corrects an earlier mis-measurement)

Quantized via `scripts/_export_utils/quantize_lm.py --mode fp16` (keep_io_types=True so the
C# binding is unchanged; 2.45 GB → 1.2 GB). `transformerFile` override on `OmniVoice`/
`OmniVoiceTts` + `--transformer` smoke flag load it. Stable C# numbers (RTX 3090, 32 steps,
warm cache; an initial run had read 569 ms — cold-run variance):

| | transformer | total |
|---|---|---|
| fp32 | ~570 ms | ~697 ms |
| fp16 | ~470 ms | ~610 ms |

- **~18% transformer / ~13% end-to-end win.** (Less than the device-IO 2× because the C#
  loop also spends ~110 ms host scoring + the logits download, and the cond/uncond split
  already trimmed fp32.)
- **Quality identical**: fp16-vs-fp32 output log-spectral-L1 = 0.0000 (greedy token field
  unchanged). No quality risk at fp16.
- **Memory halved** (2.45→1.2 GB), also helping the mobile angle (#151).

Conclusion: adopt fp16 for the CUDA path (CPU path stays fp32 for parity). int8 would cut
memory further for mobile but risks quality and — being weight-bandwidth, not compute,
limited at int8's typical W8A8 — its speed benefit on the 3090 is uncertain; deferred.
Combined with the loop work: original 1803 ms → ~610 ms (~3×).

**fp16 benefit scales with output length** (32 steps, CUDA, transformer ms):

| target tokens | ~audio | fp32 | fp16 | fp16 win |
|---|---|---|---|---|
| 25 | ~1 s | 432 | 434 | ~0% |
| 100 | ~4 s | 597 | 466 | 22% |
| 300 | ~12 s | 1281 | 785 | 39% |
| 600 | ~24 s | 2415 | 1436 | 41% |

Short clips sit on a fixed per-step floor (~400 ms / 32 forwards — launch + weight-load
overhead fp16 can't touch); as length grows, sequence-scaling attention/compute dominates and
fp16 halves it, so the win climbs to ~40% and plateaus (compute-bound). Long-form synthesis is
exactly where fp16 pays off — and the ~15 s chunking target (T≈375) lands in the sweet spot.

---

## Run 7 — 2026-06-13, length sweep closes the shape-branch question (#2/#3)

**Question:** do the codec graphs' frozen data-dependent branches (Run 2 — esp. the encoder's
semantic÷320 vs acoustic÷960 stream-alignment check) stay correct across input lengths, or
only at the 2 lengths tested so far?

**Method:** `sweep_lengths.py` — for 30 hop-multiple lengths (k·960, k∈{5..20, 25, 30, 40,
58, 64, 82, 100, 128, 200, 256, 312, 400, 500, 625}; dense at the low end where alignment is
most likely to flip), compare ONNX-CPU vs torch-CPU on the SAME device: encoder exact
code-match, decoder round-trip waveform max-abs. Real inputs are hop-aligned by
`create_voice_clone_prompt` (it clips `len % hop`), so the hop grid IS the reachable space.

**Finding — PASS across 0.20s..25.00s:** every length gives encoder code-match **1.00000**
and decoder round-trip max-abs ~1e-6; `codes_T == L/960` exactly throughout. The frozen
branches generalize over the full practical range. **Shape-branch question closed.**

All phase-1 work done. Remaining: CUDA/TF32 perf + fp16/int8; C# host-loop port.
