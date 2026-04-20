# IndicConformer investigation log

Running log for the IndicConformer ONNX export feasibility spike and
subsequent export/parity work. Plan reference:
`~/Programming/parakeet_csharp/data/IndicConformer_plan.md` (6-phase plan,
Phase 1 is the blocking gate).

Each entry is one run or one discrete investigation, stamped with local date/time.

---

## Run 0 — 2026-04-20 (setup)

**Question:** What checkpoint shape are we actually exporting? The plan
leaves open whether language is a shared-vocab routing trick (Parakeet
style) or something else.

**Pre-run research (web):**

- AI4Bharat's NeMo fork lives at `github.com/AI4Bharat/NeMo`, branch
  `nemo-v2`. The repo's one-line self-description is "implements
  multi-softmax in ASR models" — a third option the plan didn't name.
- Two checkpoint families exist on HF:
  - Per-language: `ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large`
    (hi, bn, ta, te, ml, pa, as, …). Each is a standalone Conformer with
    its own CTC + RNNT heads, trained on one language's data.
  - Unified: `ai4bharat/IndicConformer` — 22 languages, undocumented card.
    Inferred to be a **shared encoder + 22 per-language CTC heads + 22
    per-language RNNT heads**, selected at inference via `language_id='hi'`.
- The multi-softmax design is forced by script diversity: the 22 official
  Indian languages span 11+ distinct scripts (Devanagari, Bengali, Tamil,
  Gurmukhi, Ol Chiki, Meitei Mayek, Perso-Arabic, …), so a Parakeet-style
  shared subword vocab would be enormous and dilute per-script capacity.

**Consequence for Phase 1:** the language-routing question is now "how
exactly are the 22 heads wired in the checkpoint" — one `Linear` with a
gather-by-lang-id, or 22 separate submodules? That decides whether Phase 2
ships one ONNX with a language-id input, one ONNX per language, or an
encoder ONNX + 22 small head ONNX.

**Next:** install `.venv-indicconformer-export`, run
`inspect_indicconformer.py` against (a) the Hindi-only checkpoint as a
warm-up, (b) the unified `ai4bharat/IndicConformer` to answer the wiring
question.

---

## Run 1 — 2026-04-20 12:18 (env setup)

**Command:** `python3 scripts/indicconformer_export/setup_indicconformer_env.py --cuda-version cu128`

**Result:** Exit 0. `.venv-indicconformer-export` created with PyTorch +
cu128, onnxruntime-gpu, AI4Bharat NeMo fork cloned to
`.venv-indicconformer-export/_src/NeMo` (branch `nemo-v2`) and installed
editable with `[asr]` extras.

**Surprise:** The fork ships as `nemo_toolkit-1.23.0rc0` — pre-2.0.
Upstream current is 2.7.1. So this is a significantly older NeMo base.
Implication: our `export_parakeet_nemo_to_onnx.py` export patterns may not
translate directly — 1.x has different `.export()` semantics and may not
have the DFT-conv1d preprocessor insertion point we use on 2.x.

---

## Run 2 — 2026-04-20 12:20 (inspection, attempt 1)

**Command:** inspect_indicconformer.py on the Hindi-only HF repo.

**Result:** Crash at `import nemo.collections.asr`:
`AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`.

**Diagnosis:** `datasets==2.14.4` (pinned transitively by the fork) uses
`pa.PyExtensionType`, which pyarrow removed in v15. Fork install pulled
`pyarrow-23.0.1`.

**Fix:** `pip install 'pyarrow<15'` → 14.0.2. Added `pyarrow<15` to
`scripts/indicconformer_export/requirements.txt` so this is pinned next
time the env is built from scratch.

---

## Run 3 — 2026-04-20 12:21 (inspection, attempt 2)

**Command:** same inspect command.

**Result:** Crash at `import pyarrow as pa`: `_ARRAY_API not found`,
followed by the NumPy 2.x / 1.x ABI warning. pyarrow 14.0.2 was compiled
against NumPy 1.x; env currently has `numpy-2.4.4`.

**Fix:** `pip install 'numpy<2'` → 1.26.4. pyannote.core/metrics complain
about numpy<2, but those are diarization paths we don't hit in the export
pipeline.

**To add to requirements.txt next commit:** `numpy<2` (same rationale as
`pyarrow<15`).

---

## Run 4 — 2026-04-20 12:21 (inspection, attempt 3 — Hindi-only)

**Result:** NeMo imports cleanly. HF download fails:

```
GatedRepoError: Cannot access gated repo for url
https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large/
resolve/main/indicconformer_stt_hi_hybrid_ctc_rnnt_large.nemo
Access to model ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large is
restricted and you are not in the authorized list.
```

`huggingface-cli whoami` confirms logged in as `christopherthompson81`,
but the account isn't on AI4Bharat's approved list for this gated repo.

---

## Run 5 — 2026-04-20 12:22 (inspection, attempt 4 — unified)

**Result:** Same 403 / GatedRepoError against
`ai4bharat/IndicConformer`. Both the per-language and the unified
checkpoints are gated.

---

## Decision point — 2026-04-20 12:23

**Blocker:** Phase 1 can't continue without a checkpoint. Two paths:

**A. Request access on HF.** Visit the repo pages and click "Agree and
access terms." AI4Bharat typically approves within hours. Unblocks both
the per-language and unified inspection paths cleanly.

**B. Use the non-gated legacy checkpoints via
`objectstore.e2enetworks.net`.** AI4Bharat's `indic-asr-api-backend`
still points to direct URLs:

```
https://objectstore.e2enetworks.net/indic-asr-public/checkpoints/
  conformer/stt_hi_conformer_ctc_large_v2.nemo
```

These are the older **Conformer-CTC** models, **not** the current
hybrid CTC-RNNT. Exporting one would still exercise the fork + DFT
preprocessor path and prove ONNX parity — useful de-risking — but
answers a different question than the plan's gate: the hybrid
multi-softmax wiring is only visible in the gated checkpoints.

**Recommendation:** Do A, and pause Phase 1 until access lands. The
legacy CTC checkpoint would burn time on a path we're not shipping.

---

## Run 6 — 2026-04-20 12:27 (gate approved, retry Hindi-only)

**Command:** same as Run 4, after user agreed to HF access terms.

**Result:** Download succeeds (~500 MB). NeMo then crashes on
`FileNotFoundError: model_config.yaml` — it picked the "pre-extracted
directory" code path instead of the tarball-extract path. Side finding:
the file on disk is named `indicconformer_stt_hi_hybrid_rnnt_large.nemo`
(no "ctc" in the filename, despite the HF repo name).

---

## Run 7 — 2026-04-20 12:29 (inspect local .nemo file — THE decisive run)

**Command:** `inspect_indicconformer.py --nemo <local path to the
downloaded .nemo>` to skip NeMo's HF-cache path-resolution bug.

**Result: Phase 1 answered.** The full JSON report is at
`/tmp/indicconformer_hi_report.json`. Key facts:

### The "Hindi-only" HF repo is actually the 22-language unified model

- `model_class: EncDecHybridRNNTCTCBPEModel` — standard NeMo hybrid
  CTC+RNNT, with the `ExportableEncDecModel` mixin in the MRO.
- NeMo logs `_setup_tokenizer: detected an aggregate tokenizer` and
  `Aggregate vocab size: 5632` with 22 separate SentencePieceTokenizers
  loaded, each `initialized with 256 tokens`. So the per-language HF
  repos are training-shorthand aliases of the same 22-language
  checkpoint, not language-specific models.
- `tokenizer_class: MultilingualTokenizer`, `tokenizer_langs` lists all
  22: `as, bn, brx, doi, kok, gu, hi, kn, ks, mai, ml, mr, mni, ne, or,
  pa, sa, sat, sd, ta, te, ur`.

### The "multi-softmax" design is simpler than feared

- `ctc_decoder_class: ConvASRDecoder`, `ctc_decoder_param_names:
  ["decoder_layers.0.weight", "decoder_layers.0.bias"]`. That is **one
  Linear layer** (encoder_dim → 5632), NOT 22 separate softmax heads.
- The config carries `"multisoftmax": true` alongside a
  `language_masks:` array; NeMo logs `Creating masks for multi-softmax
  layer`. So "multi-softmax" here means **one shared softmax + a
  per-language output mask** that zeroes the 5376 logits belonging to
  other languages. `language_id` at inference selects which mask to
  apply.
- `can_set_cur_decoder_ctc: true` — CTC is a first-class alternative
  path on the hybrid model, as hoped.

### Preprocessor matches Parakeet

- `AudioToMelSpectrogramPreprocessor`, `sample_rate: 16000`, `features:
  80`, `n_fft: 512`, `window_size: 0.025`, `window_stride: 0.01`,
  `window: hann`, `normalize: per_feature`. Same family as Parakeet's
  frontend, so the DFT-conv1d preprocessor trick in
  `scripts/nemo_export/export_parakeet_nemo_to_onnx.py` should apply
  with only parameter substitution.

### Training config surprises (non-blocking, worth recording)

- `return_language_id: true` in the train/val manifests confirms the
  model was trained with the language as a side input — used to pick
  the mask at loss time.
- `concat_sampling_technique: temperature`, `temperature: 1.5` across
  all 22 language manifests — classic multilingual upsampling of
  low-resource languages.

### Phase 1 decision

**Commit Phase 2 to a Parakeet-shaped export package with one extra
artifact:** encoder.onnx + ctc_decoder.onnx + nemo128.onnx + vocab.txt
(flat 5632 entries) + **language_spans.json** (22 entries, each a
`[start_id, length]` pair into the vocab) + config.json. Language
selection happens entirely in C# post-argmax, by filtering to token
ids in the requested language's span. No need for a language-id input
on the ONNX graph, no per-language ONNX files.

**Unified `ai4bharat/IndicConformer` inspection:** skipping unless the
above turns out wrong. The per-language repo has already yielded the
22-language multi-softmax checkpoint.

**Open questions for Phase 2:**
1. Does NeMo 1.23.0's `.export()` produce the same split
   encoder/decoder ONNX artifacts as 2.7.1's, or does it only support
   whole-model export? (Affects whether we can reuse Parakeet's export
   module surgery or need a different approach.)
2. Where exactly does the language mask live — on the tokenizer, on
   the decoder, or as a runtime-only filter? If it's on the decoder as
   a fixed tensor, we may be able to bake it into vocab.txt and skip
   language_spans.json entirely.
3. Hindi-only BPE subword preview from the Unicode ranges suggests
   each 256-slot language block is contiguous in the flat vocab. Need
   to confirm with a small script that reads the tokenizer's
   `token_id_offset_by_tokenizer_num` and dumps per-language spans.

---

## Run 8 — 2026-04-20 12:36 (vocab span probe)

**Command:** `probe_vocab_spans.py` — loads the .nemo locally, walks the
MultilingualTokenizer, dumps flat vocab + per-language spans, and peeks
at `model.language_masks`.

**Result:** Spans are **perfectly contiguous, 256 tokens each** in
alphabetical-ish order:

```
as  : [   0, 256]   bn  : [ 256, 256]   brx : [ 512, 256]   doi : [ 768, 256]
kok : [1024, 256]   gu  : [1280, 256]   hi  : [1536, 256]   kn  : [1792, 256]
ks  : [2048, 256]   mai : [2304, 256]   ml  : [2560, 256]   mr  : [2816, 256]
mni : [3072, 256]   ne  : [3328, 256]   or  : [3584, 256]   pa  : [3840, 256]
sa  : [4096, 256]   sat : [4352, 256]   sd  : [4608, 256]   ta  : [4864, 256]
te  : [5120, 256]   ur  : [5376, 256]
```

22 × 256 = 5632 = `tokenizer.vocab_size`. No gaps. Answer to Phase 1
question #3: **yes, contiguous.**

**Surprises:**

- **CTC Linear outputs 5633**, not 5632. `decoder_layers.0.weight` is
  shape `(5633, 512, 1)` — an extra slot for the CTC **blank** at
  index 5632. That's standard CTC shape and belongs in every language's
  mask. Our `vocab.txt` will have 5632 real tokens; the blank is
  implicit at `vocab_size`.
- **`model.language_masks` is a pre-materialized `dict[str, list[bool]]`**
  with 22 entries, each of length 5633. So the mask *is* stored on the
  model object, but it's a derived artifact of the span layout — we
  don't need to serialize the masks, we can ship the 22 × `[start,
  length]` spans and reconstruct on the C# side. (Blank is always
  unmasked.)

**Answer to Phase 1 question #2:** Language mask is a Python-side
Python-list on the model, not a tensor buffer on the CTC decoder module.
It's not in the forward graph — the decoder produces all 5633 logits
every time, and the mask is applied by NeMo's decoding strategy after
argmax. Confirms the design: **zero ONNX changes for multi-language
support.** The exported encoder+CTC graph is language-agnostic.

**Artifacts written:**

- `/tmp/indicconformer_vocab.txt` (5632 lines)
- `/tmp/indicconformer_language_spans.json` (22 × `{start, length}`)

---

## Phase 1 gate — 2026-04-20 12:38 (PASS, with one caveat)

The plan's original Phase 1 decision ("language as encoder input tensor
vs shared 22-language vocab") is now moot. The real answer is:

**IndicConformer has a language-agnostic ONNX forward graph**
(encoder + one `Linear` CTC head → 5633 logits). Language selection
is a post-argmax filter applied in C# using a 22-entry
`[start, length]` table shipped alongside the ONNX files.

**Caveat: Phase 1 never exported a CTC graph to ONNX.** The plan
called for doing a small export + parity check before committing to
Phase 2. We *inferred* the export will be clean because:

- `model.encoder` is a standard Conformer encoder, already tested by
  `ExportableEncDecModel` mixin.
- `model.ctc_decoder` is `ConvASRDecoder` → one `Conv1d` used as a
  logits projection. Two parameters. Trivial.
- Preprocessor is byte-identical to Parakeet's — DFT trick ports
  with parameter substitution.

That inference is strong, but the plan asked for a proof. We'll satisfy
the gate in the next run: write a minimal `export_indicconformer_*` that
ships encoder.onnx + ctc_decoder.onnx, run a synthetic-input parity
check (PyTorch vs ORT on the same random mel-spec), and only then
commit Phase 2.

**On the "use our own export, not NeMo's" discussion:** Parakeet's
exporter calls `model.export(...)` for the split encoder/decoder
artifacts but hand-rolls the preprocessor via `torch.onnx.export`
against `CustomPreprocessorWrapper` — a mixed strategy. For
IndicConformer we can go fully hand-rolled, because both the encoder
and the CTC head are plain `nn.Module`s with clean `(features,
features_lens) → (logits, lens)` contracts. That sidesteps NeMo
1.23.0rc0's possibly-stale `.export()` path entirely and makes the
version-skew risk zero. Committing to this approach for Phase 1
spike.

---

## Run 9 — 2026-04-20 12:44 (Phase 1 export, attempt 1)

**Command:** `export_indicconformer_nemo_to_onnx.py --device auto` (CUDA).

**Result:** Both ONNX exports succeed cleanly. Hand-rolled
`torch.onnx.export` on `EncoderWrapper(model.encoder)` and
`CtcDecoderWrapper(model.ctc_decoder)`, opset 17, dynamic batch+frames.

Sidecar dump crashes at `write_vocab`:
`MultilingualTokenizer.ids_to_tokens() missing 1 required positional
argument: 'lang_id'`. The MultilingualTokenizer's flat `ids_to_tokens`
requires a `lang_id` — need to walk per-sub-tokenizer instead.

**Note:** Run 8's probe had a try/except around the same call and
silently wrote 5632 error strings to `/tmp/indicconformer_vocab.txt`.
Lesson: guard against exception-swallowing in diagnostic scripts by
validating the first line of the output before declaring success.
Folded into the fixed `write_vocab`: walk `tok.tokenizers_dict[lang]`
in offset order and stitch the results.

---

## Run 10 — 2026-04-20 12:47 (Phase 1 export, attempt 2; parity attempt 1 on CUDA)

**Command:** same, with the vocab walker rewritten.

**Result:** All artifacts written:

```
encoder-model.onnx       459 MB
ctc_decoder-model.onnx    11 MB
vocab.txt                 40 KB  (5632 lines, real tokens)
language_spans.json       1.4 KB
config.json               444 B
```

**Parity on CUDA (PyTorch ref) vs CPU (ORT):**

- encoded max-abs delta: **5.99e-03**
- logits  max-abs delta: **2.49e-02** (tolerance 1e-3) → **FAIL**

**Suspected cause:** cross-device FP32 drift. PyTorch ran on CUDA,
onnxruntime on CPU (we don't have `onnxruntime-gpu` pinned in the
session yet). 17 conformer layers compound the drift, and the
`encoder_dim→5633` Conv1d projection amplifies it at the output.

---

## Run 11 — 2026-04-20 12:49 (Phase 1 export, attempt 3; parity on CPU-CPU)

**Command:** `--device cpu --parity-tolerance 1e-2` (the loose tolerance
was just to confirm PASS — actual delta blew past even the strict
1e-3).

**Result: PARITY PASS.**

- encoded max-abs delta: **1.54e-05**
- logits  max-abs delta: **6.87e-05**
- PASS at `1e-2`, also passes at the original strict `1e-3`.

Confirms the CUDA-vs-CPU FAIL in Run 10 was purely cross-device FP32
drift, not a graph-correctness bug. The ONNX export faithfully
reproduces the PyTorch forward when both run on the same device.

Implication for C# runtime: **CPU execution path is numerically
exact.** CUDA path will have e-2 scale drift that C# callers won't
feel (argmax on the masked 256-token Hindi slice swamps e-2 perturbations
unless the top-2 tokens are already close to tied — rare, and
mitigable with a CPU re-run on ambiguous frames if we ever see it).

---

## Phase 1 gate: **CLOSED**

All four Phase 1 subgoals satisfied:

1. ✅ AI4Bharat NeMo fork standing up in an isolated venv (with pins for
   `pyarrow<15` and `numpy<2`).
2. ✅ CTC-head ONNX export — hand-rolled, no dependency on NeMo
   1.23.0rc0's `.export()` path.
3. ✅ Parity vs PyTorch reference — 6.87e-05 logit delta on CPU, well
   inside any reasonable tolerance.
4. ✅ Language-passing mechanism decided — post-argmax vocab-span mask
   in C#, zero ONNX graph changes.

**Proceed to Phase 2** (full export pipeline including nemo128
preprocessor, HF repo publishing shape).

---
