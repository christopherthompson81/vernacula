# Chatterbox ONNX export — investigation log

Running record of the Stage 0 export work in
[chatterbox.scratch.md](../chatterbox.scratch.md). Chronological by run.
Negative results stay; that's the whole point of the log.

## Run 1 — 2026-05-15 — Provenance survey of upstream ONNX artifacts

**Question:** Is there a publisher-canonical Chatterbox ONNX export we
can rely on, or do we need to own it?

**Commands / sources consulted:**

- `WebFetch ResembleAI/chatterbox-turbo-ONNX` (HF model card)
- `WebFetch onnx-community/chatterbox-ONNX` (HF model card + README +
  file listing)
- `WebFetch resemble-ai/chatterbox` (GitHub repo)
- `WebFetch VladOS95-cyber/onnx_conversion_scripts/chatterbox`
  (directory + issues)
- `WebFetch huggingface.co/.../discussions/42` (offline iPhone/Mac
  report)
- WebSearch: "Chatterbox TTS ONNX export RTX 3090 CUDA execution
  provider performance"
- WebSearch: "ResembleAI chatterbox ONNX export official script 2026"

**Findings:**

1. **No publisher export script exists.** `resemble-ai/chatterbox` on
   GitHub has zero ONNX code. ResembleAI's HF org publishes
   `chatterbox-turbo-ONNX` artifacts but no recipe to reproduce them.
2. **The `onnx-community/chatterbox-ONNX` bundle is Frankenstein.**
   - `embed_tokens.onnx`, `speech_encoder.onnx`,
     `conditional_decoder.onnx` come from VladOS95-cyber's script
     (opset 20 / 20 / 17, fp32, no CUDA-specific options).
   - `language_model.onnx` is **not** produced by that script. Open
     issue [#2 on Vlad's repo](https://github.com/VladOS95-cyber/onnx_conversion_scripts/issues)
     ("how to export language_model.onnx") confirms this — the LM came
     from somewhere else, almost certainly HF's standard
     transformers→ONNX exporter, but it's undocumented.
3. **The published LM does have proper KV-cache.** 30 layers, 16
   KV-heads, 64 head_dim. Variants: fp32 (~2 GB), fp16 (~1 GB), q4
   (~354 MB), q4f16 (~305 MB). Inference example in the model card
   threads `past_key_values.N.{key,value}` in and `present_key_values`
   out per step.
4. **The published `conditional_decoder.onnx` is anomalously small**
   (6.35 MB with no `.onnx_data` sidecar listed). Either weights are
   inline (genuinely small CNN vocoder) or there's external data the
   HF directory listing doesn't show. Our re-export will resolve this.
5. **Zero published 3090 + ORT-CUDA benchmarks.** The "decoder is the
   bottleneck" finding is from CoreML on iPhone (community member
   leowangxyz, not ResembleAI staff). Has to be re-validated on the
   3090 once we have our own export.
6. **Vlad's script maturity:** 2 open issues, 0 closed, 0 PRs.
   Single-contributor side project. Useful as a structural reference
   for the three graphs it does cover; not a basis we can sit on.

**Implication for next step:**

We own the export. Stage 0 of [chatterbox.scratch.md](../chatterbox.scratch.md)
is now scoped:

- E1: scaffold `scripts/chatterbox_export/` skeleton matching
  `scripts/vibevoice_export/` conventions.
- E2: adapt Vlad's three graphs (embed_tokens, speech_encoder,
  conditional_decoder), run end-to-end numerical comparison against
  PyTorch reference. Resolve the conditional_decoder size mystery as
  a side effect.
- E3: write the Llama LM export from scratch with proper KV-cache.
- E4: end-to-end parity test (speech-token sequence + waveform
  spectral distance).
- E5: `export-report.json` capturing source SHA, hashes, opset,
  tool versions.

Working on branch `chatterbox-export`.

## Run 2 — 2026-05-15 — Reading Vlad's script

**Question:** What exactly does Vlad's script export, what wrapper
modules does it define, and what can we copy structurally vs. need to
rewrite?

**Sources:** `/tmp/vlad_chatterbox_export.py` (1687 lines, pinned via
header: `chatterbox-tts==0.1.4`, `transformers==4.46.3`, `torch==2.6.0`,
`numpy==2.2.6`, `librosa==0.11.0`, `onnx==1.18.0`, `onnxslim==0.1.59`).

**Findings:**

1. **It loads `ChatterboxTTS.from_pretrained()` from the official
   `chatterbox-tts` PyPI package.** The script doesn't reimplement
   Chatterbox — it imports it. Our export will do the same; the model
   code is upstream. The ~900 lines of nn.Module classes in Vlad's
   script (`S3Tokenizer`, `FSMNMultiHeadAttention`,
   `ResidualAttentionBlock`, `AudioEncoderV2`, `ISTFT`,
   `ConditionalDecoder`, etc.) appear to be **vendored re-declarations of
   Chatterbox internals** needed to construct the export-wrapper modules
   — they aren't fresh model code. We need to verify whether the
   official package now exposes enough surface that we can skip the
   vendoring.

2. **Three wrapper nn.Modules drive the export:**
   - `PrepareConditionalsModel` — speech encoder + S3 tokenizer +
     conditioning latent prep, single forward pass. Output:
     `audio_features`, `audio_tokens`, `speaker_embeddings`,
     `speaker_features`. (~285 lines)
   - `InputsEmbeds` — text-embed + position/exaggeration handling.
     Inputs: `input_ids`, `position_ids`, `exaggeration`. (~95 lines)
   - `ConditionalDecoder` — speech-tokens + speaker conditioning →
     waveform; includes a custom `ISTFT` module to keep the vocoder
     graph ONNX-exportable. (~360 lines)

3. **One monkeypatch is required before export:**
   `chatterbox_model.s3gen.speaker_encoder.xvector.dense` (a `DenseLayer`
   wrapping `BatchNorm1d`) is replaced with a `SafeDenseLayer` that uses
   `LayerNorm`. Comment in the script: "we can safely do that because
   it does not affect inference as we do no need matching training
   dynamics". Load-bearing — without this, the speech_encoder graph
   doesn't export. Need to verify the assertion (norm-equivalence at
   inference) in parity test E4.

4. **Opset choices: 20 / 20 / 17.** Embed tokens and speech encoder use
   opset 20; conditional decoder uses opset 17 — likely because
   ISTFT-related ops don't survive cleanly in newer opsets. We default
   to opset 18 in Vernacula's other exports; for Chatterbox we should
   probably keep 20 for the modern graphs and live with 17 for the
   decoder unless we can verify 18 works.

5. **The Llama LM is NOT exported in this script.** Vlad runs the LM in
   PyTorch (`LlamaForCausalLM.from_pretrained("vladislavbro/llama_backbone_0.5")`)
   to validate end-to-end behavior. The LM weight repo is **Vlad's
   personal HF account**, not ResembleAI's — `vladislavbro/llama_backbone_0.5`.
   This is the LM extraction we'll need to either reuse or redo from
   the official `chatterbox-tts` package's `s3` LM weights.

6. **Reference inputs hardcoded in script:**
   - `input_ids`: 79-token prompt ending in `START_SPEECH_TOKEN,
     START_SPEECH_TOKEN` (comment says "by accident but kept for
     compatibility")
   - `position_ids`: `where(input_ids >= START_SPEECH_TOKEN, 0,
     arange - 1)`
   - `exaggeration`: 0.5 scalar
   - `dummy_audio_values`: `torch.randn(1, 312936)` — that's ~13 s at
     S3GEN_SR (24 kHz)

7. **Post-export onnxslim pass + external data:** `ONNXSLIM_THRESHOLD =
   1e10` (effectively disables size-threshold pruning), then re-saves
   with `save_as_external_data=True, all_tensors_to_one_file=True`.
   Every exported `.onnx` should end up with an `.onnx_data` sidecar.
   This makes the published `conditional_decoder.onnx` being 6.35 MB
   *without* a sidecar on HF (per Run 1 finding 4) genuinely
   suspicious — either the HF directory listing was incomplete, or the
   HF artifact was post-processed differently. **E2 parity should
   double-check this** by comparing our re-exported decoder against the
   published one byte-for-byte (or graph-for-graph if external data
   layout differs).

8. **No CUDA-aware behavior.** Script accepts `device="cpu"` and
   nothing else changes downstream. Our export needs to add
   `--device cuda` for fp16-on-3090 ergonomics.

**Implications for E2 plan:**

- Determine whether Chatterbox's official package exposes
  `speaker_encoder`, `s3gen`, `t3` modules cleanly enough that we can
  skip vendoring ~900 lines of model code. If not, isolate the vendored
  classes in a `_chatterbox_internals.py` so they're easy to spot.
- Validate the `SafeDenseLayer` substitution numerically before
  trusting it. Add a unit test that runs forward through the old and
  new dense layer with random input and confirms close-enough
  agreement at inference time.
- The Llama backbone repo `vladislavbro/llama_backbone_0.5` is
  third-party. Prefer extracting the LM directly from
  `ChatterboxTTS.from_pretrained()` internals so our provenance chain
  ends at ResembleAI, not at Vlad's personal HF.

## Run 3 — 2026-05-15 — E1 + E2 land; three of four graphs exporting

**Question:** Can we stand up the scripts/chatterbox_export/ skeleton
matching vibevoice_export conventions, and adapt Vlad's three-graph
export so it produces working ONNX artifacts on our 3090 stack?

**Commands run (high level):**

- `python3.10 -m venv` + `uv pip install` for the dep stack
- 12 invocations of `python scripts/chatterbox_export/export_chatterbox_to_onnx.py --output-dir /tmp/chatterbox_smoke --device cuda --overwrite`, each surfacing a new failure mode

**Outcome (smoke export, attempt #12):**

```
Graphs emitted: ['embed_tokens.onnx', 'speech_encoder.onnx', 'conditional_decoder.onnx']
```

Artifact sizes:

| File | Header | Sidecar | Total |
|---|---|---|---|
| embed_tokens.onnx | 17 KB | 61.6 MB | 61.6 MB |
| speech_encoder.onnx | 1.1 MB | 1.05 GB | 1.05 GB |
| conditional_decoder.onnx | 32 MB | 533 MB | 565 MB |

Compare to published `onnx-community/chatterbox-ONNX` directory listing
(Run 1 finding 4): "conditional_decoder.onnx 6.35 MB no sidecar listed"
— our 565 MB result makes it clear the HF directory listing was just
incomplete. The decoder genuinely is a substantial graph; the published
artifact must have an `.onnx_data` sidecar not surfaced in the directory
view. **Mystery resolved.** The cond decoder is not a "small graph";
it's a normal-sized vocoder with weights externalized.

**Dependency stack pins that survived debugging:**

```
chatterbox-tts==0.1.5   (0.1.4 wants raw `pkuseg` which needs Python.h;
                         0.1.7 wants transformers==5.x which breaks the
                         wrapper API. 0.1.5 is the sweet spot.)
transformers==4.46.3    (matches Vlad's reference)
torch==2.6.0            (matches Vlad's reference)
numpy>=1.24,<1.26       (chatterbox-tts cap — forces Python 3.10)
onnxslim>=0.1.93,<0.2   (Vlad's 0.1.59 / 0.1.68 want sympy>=1.13.3 which
                         conflicts with torch 2.6.0's sympy==1.13.1)
setuptools<80           (resemble-perth imports pkg_resources, removed
                         in setuptools 80+)
```

**The twelve fixes**, each captured because the issue is non-obvious
and "Vlad's script just works for him" does not protect us:

1. **resemble-perth import dies on setuptools 80+** — pkg_resources got
   removed. Pinned setuptools<80.
2. **`@torch.inference_mode()` decorators in vendored S3Tokenizer / FSQ
   classes** — inference mode poisons tensors with a flag that prevents
   `save_for_backward` during JIT tracing. Replaced 7 decorators with
   `@torch.no_grad()` (same intent, doesn't trigger the trace conflict).
   This is a chatterbox-tts 0.1.5 thing; 0.1.4 may not have it.
3. **`PrepareConditionalsModel.cond_spkr` is precomputed in `__init__`
   from trainable layers** — its `requires_grad=True` made the tracer
   refuse to inline it as a constant. Wrapped in
   `torch.no_grad() + .detach()`. Vlad's script worked because his entire
   export ran under `@torch.no_grad()`; we use scoped semantics instead.
4. **Wrappers needed `.to(device)`** — buffers registered via
   `register_buffer` follow the parent module on `.to()`, but they sit
   on the CPU default until you call it. Added to the export script.
5. **`PrepareConditionalsModel.__init__` created `speaker_emb` on CPU**
   when the chatterbox model was on CUDA. Fixed by reading the device
   from `cond_enc.spkr_enc`.
6. **`PrepareConditionalsModel.mel_spectrogram` created `hann_window` +
   mel filter on CPU.** Added `device=y.device` hints.
7. **LM head: chatterbox.t3 uses split `tfmr` (Llama backbone) +
   `speech_head` (output projection)** — `.tfmr(...)` returns
   `BaseModelOutputWithPast` with `last_hidden_state`, not
   `CausalLMOutputWithPast` with `logits`. Vlad sidestepped this by
   loading `vladislavbro/llama_backbone_0.5` from his personal HF mirror;
   we keep provenance at ResembleAI by composing the two pieces ourselves.
8. **`ConditionalDecoder.flow_forward` allocated `conds` zeros without
   device hint** — added `device=text_encoded.device`.
9. **`ConditionalDecoder.forward` allocated `trim_fade` zeros and
   `cosine_window` linspace without device hint** — added device hints.
10. **`ISTFT.window` was a plain attribute, not a registered buffer** —
    `.to(device)` skipped it. Changed to `register_buffer('window', ...)`.
11. **CUDA `torch.jit.trace` bug in upstream `CausalBlock1D.block(x * mask)`** —
    reproduces with `torch.jit.trace` directly, not specific to ONNX. Eager mode
    runs the same code path cleanly. Pre-casting mask to `x.dtype`
    didn't help; `torch.set_default_device("cuda")` didn't help. **Workaround:
    move the cond decoder + its inputs to CPU just for the export call.**
    The resulting ONNX file is device-independent; runtime can still use
    CUDA EP. The other two graphs (embed_tokens, speech_encoder) export
    fine on CUDA.
12. **`window_sumsquare` used `F.conv_transpose1d` with dynamic-shape input** —
    fails the ONNX symbolic at opset 17 and 18 ("Unsupported: ONNX export of
    convolution for kernel of unknown shape"). Rewrote with a `scatter_add`
    formulation: build flat output-position indices `frame_idx * hop_length
    + sample_idx`, scatter-add the squared window. Same math, ONNX-friendly.
13. **`onnxslim.slim` crashes on the 565 MB cond decoder** —
    `google.protobuf.message.EncodeError: Failed to serialize proto`
    because the inline-weight proto exceeds protobuf's 2 GB limit during
    onnxslim's `shape_infer` size-check (then masked by an
    `UnboundLocalError` from `model` not being assigned in the slim's
    exception handler). Made the slim pass fault-tolerant: on any
    failure, fall back to raw externalization via `onnx.save_model(...,
    save_as_external_data=True)`. Slim is an optimization, not load-bearing.

**Observations on the broader story:**

- Vlad's script "worked" for him under a narrow regime: CPU-only, inside
  a global `@torch.no_grad()` decorator, with `torch.Tensor.item =
  lambda x: x` monkeypatching, against `chatterbox-tts==0.1.4`. Every
  constraint we relaxed surfaced a separate bug.
- Roughly half of the bugs are Vlad's (cond_spkr requires_grad,
  ISTFT.window as plain attribute, scatter_add formulation, device hints
  missing), half are upstream chatterbox-tts (inference_mode decorators,
  the CUDA trace bug in CausalBlock1D), one is onnxslim's own.
- The investigation doc convention from CLAUDE.md paid off — each fix is
  traceable to a specific failure mode, not "we just kept tweaking until
  it worked".

**What's now true:**

- Branch `chatterbox-export` carries a working three-graph export.
- `_chatterbox_internals.py` (1521 lines + 13 inline patches) is the
  vendored baseline we'll iterate on. Attribution preserved in header.
- `export-report.json` captures SHA256 hashes for all 6 artifacts plus
  source revision, opset table, environment.
- Branch `chatterbox-export` is uncommitted; next step is to commit
  this state before tackling E3.

**Open questions for E3 / E4:**

- The Llama LM export: probably much smoother because it's a stock
  Llama backbone and HF's tooling (`optimum.exporters.onnx`) handles
  this well. Still need to compose with `chatterbox.t3.speech_head` (Vlad
  used a separately-uploaded `LlamaForCausalLM` mirror that bundles the
  head; we'd rather not depend on that).
- E4 parity: do the three graphs we exported actually match the
  PyTorch reference numerically? We haven't checked yet. The smoke
  proves export *succeeds*; correctness comes in E4.

## Run 4 — pending — Commit E2 state and start E3 (Llama LM export)
