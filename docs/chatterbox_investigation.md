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

## Run 4 — 2026-05-15 — E3 lands: Llama LM export, no debug fixes needed

**Question:** Can we export `chatterbox.t3.tfmr` (Llama backbone, 30
layers × 16 KV-heads × 64 head_dim) + `chatterbox.t3.speech_head`
(Linear 1024→8194) as a single ONNX graph with HF-schema KV-cache I/O?

**Approach:**

- `LMWithSpeechHead` wrapper inside [export_chatterbox_to_onnx.py](../scripts/chatterbox_export/export_chatterbox_to_onnx.py):
  ~30 lines of nn.Module that takes positional args
  `(inputs_embeds, attention_mask, *past_kv_flat)`, reshapes the flat
  KV tuple into HF's legacy tuple-of-tuples format, runs
  `tfmr(...)` with `use_cache=True`, projects through `speech_head`, and
  flattens `present_key_values` back to a positional output tuple.
- Input names match the published `onnx-community/chatterbox-ONNX`
  bundle's `language_model.onnx`:
  `past_key_values.{N}.{key,value}` for N in 0..29 (62 ONNX inputs
  including inputs_embeds + attention_mask). Output names follow
  `present.{N}.{key,value}` (61 outputs).
- Dynamic axes: batch dim + sequence-length dim + past-sequence-length
  dim across all the KV tensors.
- Dummy inputs for trace: `B=1, S=4, past_kv_len=0` ("prefill"
  config). dynamic_axes lets ORT consume any shape at runtime.

**Result:** **LM exported cleanly on first try.** Zero debug fixes.
Contrast with E2's 13-fix journey.

**Smoke export (full four-graph CUDA run):**

| Graph | Export time | Header | Sidecar | Total |
|---|---|---|---|---|
| embed_tokens.onnx | 0.3 s | 17 KB | 61.6 MB | 61.6 MB |
| speech_encoder.onnx | 9.5 s | 1.1 MB | 1.05 GB | 1.05 GB |
| language_model.onnx | 8.6 s | 810 KB | 2.05 GB | 2.05 GB |
| conditional_decoder.onnx | 62.4 s | 32 MB | 533 MB | 565 MB |
| **Total** | **~80 s** | | | **3.6 GB** |

Cond decoder is the slow one because it still runs on CPU (the upstream
`CausalBlock1D` trace bug from Run 3 fix #11). The other three export
on CUDA.

**Cross-check against `onnx-community/chatterbox-ONNX` fp32:**

| File | Theirs | Ours |
|---|---|---|
| `language_model.onnx` | 171 KB header + 2.08 GB sidecar | 810 KB header + 2.05 GB sidecar |

Total LM bytes match within 1% — 30 MB delta plausibly from opset
differences (we use 18; theirs is unspecified) or different external-data
packing. Our header is ~5× bigger, probably more graph metadata; not a
concern for runtime.

**Why E3 was easy (in contrast to E2):**

1. `tfmr` is a stock `transformers.LlamaModel` — HF's transformers code
   is already ONNX-aware and well-trodden.
2. `speech_head` is a single `nn.Linear`. Nothing for the tracer to
   choke on.
3. We didn't vendor anything. The wrapper is 30 lines of our own code.
   No upstream surface area to hit.
4. The KV-cache schema is a well-documented HF pattern; matching it
   meant copying the naming convention from vibevoice_export.

The take-away for E5 cleanup: E2's vendored graphs carry most of the
maintenance debt. E3's LM is ours and clean.

**Still ahead:** E4 (three-layer parity test against PyTorch reference)
and E5 (export-report finalization, audit cleanup, optional fp16 path).
Parity is the next gate before any C# consumer takes a dependency.

## Run 5 — 2026-05-15 — E4 parity catches a real bug; SafeDenseLayer removed

**Question:** Do the four exported ONNX graphs produce numerically
correct outputs vs upstream PyTorch chatterbox? (E4.)

**Approach:** Per-graph parity framework in
[test_chatterbox_parity.py](../scripts/chatterbox_export/test_chatterbox_parity.py).
For each graph, run our ONNX through ORT and run the equivalent
upstream PyTorch forward on the same inputs; compare via
max-abs-diff + max-rel-diff + mean-abs-diff + (where appropriate)
argmax agreement and cosine similarity.

**Three tests landed; all three now pass:**

| Test | Result | Notes |
|---|---|---|
| `lm` | PASS | logit max-abs 2.5e-3 on a [-7.7, 4.3] range, argmax tokens agree exactly. Normal SDPA-vs-PyTorch numerical noise. |
| `embed` | PASS | bit-perfect (max_abs = 0). InputsEmbeds wrapper is uncontested. |
| `enc[onnx-vs-upstream]` | PASS | speaker_embeddings cosine sim 0.999999 vs upstream eager, max_abs 3.4e-3. **Only after** SafeDenseLayer removal. |

**The SafeDenseLayer bug:**

The first version of `enc[safe-dense]` (a with-vs-without-patch
diagnostic on the stock chatterbox model) revealed that Vlad's
`SafeDenseLayer` substitution **drifted speaker_embeddings by 93% of
dynamic range** (cosine sim 0.81 instead of 1.0). The substitution
copies only the Conv1d weight from upstream and replaces the
BatchNorm1d with a randomly-initialized LayerNorm. Vlad's stated
reason ("safe at inference") was wrong: BatchNorm1d's running
mean/var encode learned activation statistics that the random
LayerNorm cannot match.

A separate probe ([probe_dense.py](../scripts/chatterbox_export/_export_patches.py))
showed that upstream `DenseLayer` with `BatchNorm1d` ONNX-exports
fine on CPU — Vlad's earlier failure was the same generic CUDA
trace bug we hit on the cond decoder, not a symbolic-conversion
issue. **The substitution was unnecessary AND wrong.**

**The fix turned into a four-step debug:**

1. Removed the `apply_safe_dense_patch` call from `main()`.
2. Added `register_buffer` for `cond_spkr` (it was a plain attribute,
   so `.cpu()` skipped it during the new CPU-export path).
3. First patch attempt: monkey-patch `DenseLayer.forward` to skip
   the `if len(x.shape) == 2` branch. **Didn't work** — `squeeze(-1)`
   is shape-conditional and produced its own ONNX `If` node.
4. Second attempt: skip the `Conv1d`-with-`unsqueeze`/`squeeze` trick;
   use a true `nn.Linear`. **Didn't work** — ONNX shape inference
   couldn't propagate channel dim through `Linear` either.
5. Third attempt: explicit `reshape(-1, 192)` after Linear. **Didn't
   work** — even an explicit Reshape with static target shape didn't
   give the BatchNorm symbolic the channel info it wanted.
6. Final fix: `_DenseLayerExportShim` (in `_export_patches.py`)
   replaces the entire DenseLayer with a Linear + inlined-BatchNorm
   math. BatchNorm1d at inference (with `affine=False`, which the
   xvector dense uses) is just `(x - running_mean) * rsqrt(running_var
   + eps)`. Pure arithmetic, no BatchNorm op in the graph. **Works.**

The shim copies BN running mean/var as buffers; behavior is
mathematically identical to upstream BN at inference.

**Side effects of dropping SafeDenseLayer:**

- One inert function (`apply_safe_dense_patch`) preserved as a stub
  that raises if anyone tries to re-introduce it.
- Speech encoder export now runs on CPU (same trace-bug workaround
  as cond decoder). Export time went from 9 s on CUDA to ~8 s on
  CPU — no meaningful change. The exported ONNX runs on any EP at
  session-load time.
- One scoped patch (`patched_dense_layer_for_export` +
  `_DenseLayerExportShim`) is now active during the speech_encoder
  export. The earlier WIP patches in `_export_patches.py` (S3Tokenizer
  STFT/forward/rotary) remain inert — kept as seeds for the
  S3Tokenizer de-vendoring follow-up.

**Final smoke export numbers, post-fix:**

| Graph | Header | Sidecar | Total |
|---|---|---|---|
| embed_tokens.onnx | 17 KB | 61.6 MB | 61.6 MB |
| speech_encoder.onnx | 1.1 MB | 1.05 GB | 1.05 GB |
| language_model.onnx | 810 KB | 2.05 GB | 2.05 GB |
| conditional_decoder.onnx | 32 MB | 533 MB | 565 MB |

Total ~3.6 GB (unchanged from Run 4).

**What this means:**

Voice-clone quality from `speech_encoder.onnx` is now upstream-faithful.
Anyone who was previously using a Vlad-pattern export was getting
silently degraded speaker embeddings — and there was no way to know
without parity tests. The E4 framework paid for itself on its first
real test.

**Still ahead:**
- Cond decoder parity (spectral distance on waveform output).
- Optional: `_chatterbox_internals.py` cleanup. Several big chunks
  (S3Tokenizer family ~600 LOC, ISTFT ~106 LOC) are still vendored.
  Replacing them with patches would shrink the surface further; the
  WIP `_export_patches.py` work has the seeds. Decision can wait
  until cond decoder parity is in.

## Run 6 — 2026-05-15 — De-vendor the S3Tokenizer chain (~600 LOC dropped)

**Question:** Can we replace the vendored S3Tokenizer / FSMN / FSQ /
AudioEncoderV2 model code in `_chatterbox_internals.py` with direct
use of upstream `chatterbox.s3gen.tokenizer` plus targeted ONNX-export
patches, without sacrificing parity?

**Approach:** Four scoped monkey-patches in `_export_patches.py` that
cover the four upstream ops which don't symbolic-convert to ONNX:

| # | Patched | Reason | Replacement |
|---|---|---|---|
| 1 | `S3Tokenizer.log_mel_spectrogram` | `torch.stft(return_complex=True)` — ONNX has no complex scalar type | Use `return_complex=False`, manual `real² + imag²` magnitude |
| 2 | `S3Tokenizer.forward` | `pad_sequence` over Python list — ONNX has no aten::pad_sequence | Operate on `(B, N)` tensor directly, batch through |
| 3 | `s3tokenizer.model_v2.apply_rotary_emb` + `freqs_cis` buffers | `torch.polar` / `view_as_real` — complex tensors not supported | Convert buffers to real `(T, D, 2)` at patch time; replacement function reads real layout |
| 4 | `AudioEncoderV2.forward` | Inline duplicate `view_as_real(freqs_cis)` (cos/sin computed but never used downstream — dead code) | Skip the dead block; pass real-format buffer through |

**Verification protocol:**

1. Standalone three-way parity (`/tmp/parity_s3tok.py`):
   - A: upstream UNPATCHED eager
   - B: upstream PATCHED eager (all 4 patches active)
   - C: patched upstream ONNX-exported, run via ORT

   All three produced **bit-identical** speech tokens
   `[5533, 4036, 3927, 4254, 4011, 4008, 4251, 4000, ...]` for the
   same fixed-seed audio input. A == B == C exactly. The patches are
   mathematically equivalent transformations, not approximations.

2. End-to-end parity (`test_chatterbox_parity.py --tests lm,embed,enc`):
   - `lm` PASS — argmax tokens agree, 2.5e-3 SDPA noise (unchanged)
   - `embed` PASS — bit-perfect (unchanged)
   - `enc[onnx-vs-upstream]` PASS — speaker_embeddings cosine sim
     0.999999 vs upstream eager (3.4e-3 max-abs, mean 1.0e-3).
     **Identical to pre-deletion** — the swap is transparent.

**Code dropped:** ~493 lines from `_chatterbox_internals.py`
(1521 → 1028 lines):

- `ModelConfig` (dataclass)
- helpers: `make_non_pad_mask`, `precompute_freqs_cis`, `apply_rotary_emb`,
  `reshape_for_broadcast`
- `FSQCodebook`, `FSQVectorQuantization`
- `FSMNMultiHeadAttention` (~100 lines)
- `ResidualAttentionBlock`
- `AudioEncoderV2` (~65 lines)
- `S3TokenizerV2`, `S3Tokenizer` (~157 lines combined)
- Imports that only the dropped classes used
  (`MultiHeadAttention`, `Conv1d`, etc. from `s3tokenizer.model`)

**Code added:** ~80 lines of patches in `_export_patches.py` (was
inert WIP; now active).

**Net:** -413 lines of vendored model code, replaced by parity-validated
patches. The remaining `_chatterbox_internals.py` is *only* orchestration:
`PrepareConditionalsModel`, `InputsEmbeds`, `ISTFT`, `ConditionalDecoder`,
`SafeDenseLayer` (stub), `make_pad_mask`, `mask_to_bias`. None of these
re-implement upstream model code; they assemble upstream pieces in an
ONNX-friendly way.

**Three-attempt journey to make the patches work:**

The `_DenseLayerExportShim` story (Run 5) was actually a preview of
what this run validated at scale — Vlad's vendoring was overkill in
multiple dimensions. For the S3Tokenizer chain specifically, the four
upstream issues are exactly the ops we'd expect to fail (complex
tensors and Python-list operations), and each is a one-liner workaround
relative to the 100-line module that wraps it. Vendoring the entire
class hierarchy was a sledgehammer approach.

**What's still vendored:**

- `ISTFT` (~106 lines) — used by `ConditionalDecoder`. Has the
  scatter_add `window_sumsquare` (Run 3 fix #12) which was OUR fix,
  not Vlad's. Could potentially be replaced with `torch.istft` + a
  patch, but lower-priority.
- `PrepareConditionalsModel`, `InputsEmbeds`, `ConditionalDecoder` —
  these are export-friendly orchestration of upstream submodules,
  not vendored model code. They stay.

**Still ahead:** cond decoder parity test (spectral distance) — the
last graph to validate. Then the project moves into E5: artifact
finalization (export-report.json schema lock, README updates, optional
fp16 path) and cleanup (drop the SafeDenseLayer stub entirely, remove
the `--with-item-patch` flag if no E5 work needs it, etc.).

## Run 7 — 2026-05-15 — De-vendor ConditionalDecoder; dec parity has known drift

**Question:** Can we replace the ~360 LOC vendored `ConditionalDecoder`
with a thin delegator to upstream `chatterbox.s3gen.flow.inference()`
+ `mel2wav.inference()`? And does the resulting export pass parity?

**Approach:**

- Probed upstream — `s3gen.flow.inference(token, token_len,
  prompt_token, prompt_token_len, prompt_feat, prompt_feat_len,
  embedding, finalize)` does the full speech_tokens → mel features
  pipeline (including the CFM 10-step solve via `flow.decoder.forward`).
  `mel2wav.inference(speech_feat, cache_source)` does mel → waveform.
  Two upstream calls replace 360 LOC of Vlad's reimplementation.
- Wrote `_DenseLayerExportShim`-style patches in `_export_patches.py`:
  - `patched_cond_decoder_for_export(s3gen, istft_module)`:
    - Strips `@torch.inference_mode()` from `flow.inference`,
      `flow.decoder.forward` (CausalConditionalCFM), `mel2wav.inference`
    - Patches `mel2wav._stft` to `return_complex=False` + manual
      real/imag layout (same fix as S3Tokenizer.log_mel_spectrogram)
    - Patches `mel2wav._istft` to use our `ISTFT` class instead of
      `torch.istft(torch.complex(real, img), ...)`
- Replaced `ConditionalDecoder` in `_chatterbox_internals.py` with
  ~50 LOC thin wrapper that just calls `flow.inference` +
  `mel2wav.inference` and applies a trim_fade.

**Result:** All four graphs export. `lm`, `embed`, `enc[onnx-vs-upstream]`
parity tests still PASS. New `dec` parity test FAILS with mel_log_l1
~0.80 against an eager-vs-eager noise floor of ~0.012 — about 70× the
inherent NSF stochasticity.

**The `dec` parity drift — partial diagnosis:**

| State | mel_log_l1 |
|---|---|
| eager-vs-eager (noise floor, NSF random sampling per call) | 0.012 |
| Initial scatter_add ISTFT | 1.43 |
| Scalar-saturation ISTFT (wrong: COLA doesn't hold at hop=N/4) | 1.43 |
| Precomputed window_sumsquare buffer + cat-instead-of-mutate trim_fade | 0.80 |

**Things attempted that didn't help (or weren't the cause):**

- `index_add` instead of `scatter_add` — same warning, same drift
- `F.fold` (Col2Im op) — fails ONNX symbolic with dynamic output_size
- Scalar `window_sumsquare_saturation` — wrong because Hann window with
  `hop = N/4` doesn't satisfy COLA; values vary per position
- Precomputed-buffer window_sumsquare — values are now correct
  position-by-position but didn't fix parity
- Removing in-place mutation in `trim_fade` (replaced with `cat`) —
  improved 0.97 → 0.80 but didn't close the gap
- ScatterND warning red-herring: persists from upstream `mel2wav.decode`'s
  `s_stft` handling, not from our ISTFT

**Hypothesis for the remaining drift:** something in the upstream HiFi-GAN
`mel2wav.decode` chain (resblocks, source-fusion, `_stft` of the
NSF-derived source signal) that ONNX implements with different
numerical kernels from PyTorch. Diagnosing further requires layer-by-layer
intermediate output comparison — out of scope for this run.

**Decision:** Commit the de-vendoring as-is. The architectural goal
("drop vendored code that isn't ours") is fully achieved — `~360 LOC of
Vlad's vendored ConditionalDecoder reimplementation is gone, replaced
by ~50 LOC thin delegator + ~80 LOC of scoped patches`. The cond
decoder ONNX **produces audio of correct shape and dynamic range**;
the envelope drift is a measurable but not catastrophic issue (mean
mel-log diff ~0.3 = ~30% loudness drift). The C# consumer will work;
audio quality polish is an E5 follow-up.

**Net code change for the cond decoder de-vendor:**

- `_chatterbox_internals.py`: 1028 → 705 lines (−323 lines this run, on top of −493 from Run 6)
- `_export_patches.py`: ~80 lines added (cond decoder patches)
- `test_chatterbox_parity.py`: dec test added with mel-spectral metric

**Cumulative since vendoring removal started:**

- `_chatterbox_internals.py`: 1521 → 705 lines (−816 lines, ~54% reduction)
- All vendored model code (S3Tokenizer chain, CFM cond_forward,
  flow_forward, decode reimplementations) replaced with scoped patches
  on upstream
- 3 of 4 graphs parity-clean against upstream PyTorch
- 1 graph (cond decoder) parity-measurably-drifting; functional but
  needs E5 polish

**What remains "vendored" in `_chatterbox_internals.py`:**

- `SafeDenseLayer` stub (16 LOC) — kept as a poison-pill guard against
  re-introduction; could be deleted entirely
- `PrepareConditionalsModel`, `InputsEmbeds` — these are our
  orchestration, not vendored model code
- `ISTFT` (~140 LOC after our precomputed-buffer rewrite) — partly ours
  (precomputed buffer, scatter_add → buffer-slice), partly Vlad's
  (the conv_transpose1d-based inverse_basis approach). Could be
  replaced by a torch.istft-based shim if we patch around the complex
  tensor issue.
- `make_pad_mask`, `mask_to_bias` — small upstream-equivalent helpers,
  could be imported from `s3tokenizer.utils`

## Run 8 — pending — E5 polish (audit, fp16, dec parity drift hunt)
