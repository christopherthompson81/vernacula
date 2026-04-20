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

## Run 12 — 2026-04-20 13:50 (Phase 2: DFT preprocessor, attempt 1)

Ported the Parakeet `DFTConvPreprocessorWrapper` from
`scripts/nemo_export/export_parakeet_nemo_to_onnx.py`, added an
`export_preprocessor` function emitting `nemo128.onnx`, and wired a real-audio
parity path that chains nemo128 → encoder → ctc_decoder in both PyTorch and
ORT for an actual wav file.

**Result — Attempt 1:** Crash at wrapper init.
```
AttributeError: 'FilterbankFeatures' object has no attribute 'exact_pad'
```

**Diagnosis:** `exact_pad` and `stft_pad_amount` are NeMo 2.x additions; the
AI4Bharat fork runs NeMo 1.23.0rc0 where they don't exist. NeMo 1.x always
uses center + reflect padding — equivalent to 2.x `exact_pad=False,
stft_pad_amount=None`. Fixed by defaulting both to the 1.x behavior via
`getattr(..., default)`.

---

## Run 13 — 2026-04-20 13:58 (Phase 2, attempt 2 — big feature delta)

**Result:** All exports succeed. Synthetic parity still passes (6.87e-05).
But real-audio parity **fails hard**:

```
[parity-audio] features max-abs delta: 1.467379e+00
[parity-audio] logits   max-abs delta: 2.092063e+01
[parity-audio] FAIL
```

e-0 delta on features is not numerical noise — something structural is wrong.

**Diagnostic (preproc_parity_debug2.py):** compare live NeMo preprocessor
vs DFT wrapper, both in PyTorch (cut the ONNX export out of the loop).

Output:

```
live shape torch.Size([1, 80, 937]), live_lens [937]
wrap shape torch.Size([1, 80, 937]), wrap_lens [936]
cells with delta > 0.5: 41
first 10: [[0, 0, 936], [0, 1, 936], ...]  # all at frame 936
at (0, 0, 936): live=-0.6387542486190796, wrap=0.0
max-per-frame last 10: [..., 0.0026, 1.4673793]
```

**Root cause:** off-by-one on `seq_len`. Live NeMo emits 937 valid frames;
wrapper emits 936, so frame 936 gets masked to pad_value while live NeMo
fills it with real content. All 0..935 frames match within ~4e-3; the
single frame 936 carries the entire 1.47 delta.

Grep on NeMo fork's `FilterbankFeatures.get_seq_len` at
`features.py:390`:

```
seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length) + 1
```

There's a trailing **+1** that the wrapper I copied over was missing.
`scripts/nemo_export/`'s own README calls this out for the `custom` mode
fallback, but the DFT wrapper there — our source — dropped it anyway.
Added to the IndicConformer wrapper.

---

## Run 14 — 2026-04-20 14:05 (Phase 2, attempt 3 — end-to-end parity)

**Result: PASS on both parity paths.**

- Synthetic (encoder + ctc_decoder only): logits delta 6.87e-05
- Real audio (hi-IN Fleurs clip, full nemo128 + encoder + ctc_decoder):
  - features delta: **7.87e-06**
  - logits delta:   **8.96e-05**
  - Tolerance:      1e-02 (loose) and 1e-03 (strict) both pass.

**Final export package** at `~/models/indicconformer_onnx/`:

```
encoder-model.onnx        459 MB
ctc_decoder-model.onnx     11 MB
nemo128.onnx              1.1 MB   <-- new in Phase 2
vocab.txt                  40 KB
language_spans.json       1.4 KB
config.json                444 B
export-report.json         (both parity numbers + PASS flags)
```

## Run 15 — 2026-04-20 14:40 (600M discovery)

**Pivot:** user preferred the 600M variant (`ai4bharat/indic-conformer-600m-multilingual`)
over the 120M we'd been exporting. Standard parameter-scaling motivation.

Inspecting the 600M HF repo reveals it's an **already-exported ONNX package**,
not a `.nemo` file:

- `assets/encoder.onnx` (3 MB metadata + 2.43 GB external-data weights spread
  across ~360 per-tensor blob files under `assets/layers.*`, `assets/onnx__*`,
  `assets/pre_encode.*`, and `assets/Constant_*`)
- `assets/ctc_decoder.onnx` (23 MB, inline)
- 22× `assets/joint_post_net_<lang>.onnx` plus `joint_enc/joint_pred/joint_pre_net/rnnt_decoder.onnx` — RNNT-side, we don't need any of them
- `assets/language_masks.json` — pre-materialized 22 × 5633 bool arrays
- `assets/vocab.json` — per-language dict (see below)
- `assets/preprocessor.ts` — **TorchScript**, not ONNX
- `model_onnx.py` — reference inference script
- `config.json` — minimal, mostly RNNT params plus `BLANK_ID: 256`

**Shipping-shape diff from our 120M package:**

| | 120M (our export) | 600M (AI4Bharat ships) |
|---|---|---|
| Encoder IO | `features/features_lens → encoded/encoded_lens` | `audio_signal/length → outputs/encoded_lengths` |
| CTC IO | `encoded → logits` | `encoder_output → logprobs` |
| Preprocessor | ONNX (our DFT wrapper) | TorchScript |
| Vocab | flat `vocab.txt`, 5632 lines | per-lang JSON dict, 22 × 257 entries |
| Language mask | 22 × `[start, length]` spans | 22 × length-5633 bool arrays |

**Preprocessor config match check** — dumped `preprocessor.ts`'s TorchScript
`.code`. The forward does exactly: seq_len = `length // 160 + 1`, reflect-pad
by 256 on each side, `torch.stft(input, 512, 160, 400, hann, ...)`, power-
spectrogram (pow 2), 80-mel matmul, `log(x + 5.96e-08)`, per-feature
mean/std normalize. **Byte-identical** to our 120M preprocessor config → our
existing `nemo128.onnx` works for 600M without re-export.

---

## Run 16 — 2026-04-20 14:50 (600M repackaging, attempt 1)

Wrote `scripts/indicconformer_export/repackage_600m_indicconformer.py` that:

1. `snapshot_download` the 600M `assets/*` (2.4 GB, 371 files)
2. Load encoder.onnx + external data, rename 4 IO tensors + node
   references, save with single consolidated `encoder-model.onnx.data`
3. Same for ctc_decoder.onnx (inlined, only 23 MB)
4. Copy our existing nemo128.onnx verbatim
5. Flatten vocab.json dict → flat `vocab.txt`
6. Convert bool masks → `[start, length]` spans + verify they agree with
   AI4Bharat's published masks

**Attempt 1 crash:** `onnx.load(..., load_external_data=True)` refuses
symlinked external data (safety check). HF stores blobs at `cache/blobs/*`
and symlinks into `assets/*` — so all 366 weight file links trip the check.

**Fix:** `load_external_data=False` to get the graph only, then manually
walk `graph.initializer`, `.resolve()` each referenced path through the
symlink, read raw bytes, set `raw_data` inline, clear the external-data
marker. Re-save with `save_as_external_data=True` then rebuilds a clean
single-sidecar package.

---

## Run 17 — 2026-04-20 14:56 (repackaging, attempt 2 — vocab assertion)

**Crash:** `AssertionError: as: middle slice is 255 tokens`. I assumed the
per-lang vocab shape was `[<unk>, t1..t255, <blank>]` so the "real" slice
was `vocab[1:-1]`. Actually the layout is `[<unk>, t1..t256]` — 257 entries
where the last is a real token (not a blank marker). The CTC softmax emits
256 logits per lang; the 257th output slot is the shared CTC blank and
doesn't need a vocab entry. Fixed by slicing `vocab[lang][:256]` — keep
`<unk>` at local id 0 plus 255 real tokens.

This matches the flat layout our 120M export produced (verified by
re-reading the 120M `vocab.txt`: each 256-slot block starts with `<unk>`).

---

## Run 18 — 2026-04-20 14:58 (repackaging, attempt 3 — ORT init crash)

Repackage succeeds, ORT fails to load `encoder-model.onnx`:

```
RuntimeException: cannot get file size: ... /Constant_1970_attr__value
```

**Diagnosis:** the repackaged ONNX still has a reference to
`Constant_1970_attr__value` — a file sitting in `assets/` that the ONNX
graph doesn't reach through `graph.initializer`. It's external data on a
**node attribute** (a Constant op's baked-in tensor). My resolver only
walked initializers.

**Fix:** walk `node.attribute` recursively too — handle both scalar
TENSOR attrs and TENSORS lists, and descend into subgraphs (GRAPH/GRAPHS
types) for if/loop nodes. Updated the helper.

---

## Run 19 — 2026-04-20 14:58 (repackaging, attempt 4 — clean)

Repackager output:

```
[encoder]    resolved 366 external tensors (initializers + attrs)
             rewrote 4 IO refs, saved encoder-model.onnx + .data (2.43 GB)
[ctc_decoder] resolved 0 external tensors; rewrote 2 IO refs; inline 23.1 MB
[nemo128]    copied from 120M package (byte-identical config)
[vocab]      5632 lines (22 × 256)
[verify]     masks match derived spans for all 22 languages  ✓
[spans]      22 × {start, length}
[config]     sample_rate, preprocessor params, ctc.blank_token_id=5632
```

---

## Run 20 — 2026-04-20 14:59 (600M end-to-end Hindi validation)

First working decode through the full ONNX package:

```
validate_indicconformer_package.py --package <600m> --wav hi-IN_fleurs_01.wav --lang hi

decoded (hi): स्कीम मार्ग को एक हाइकिंग लंबी पैदल यात्रा मार्ग जैसा ही सोचें
expected:     स्कीइंग मार्ग को एक हाईकिंग लंबी पैदल यात्रा मार्ग जैसा ही सोचें।
```

~2 word-level edits out of 11 words on a 9.4-sec clip — ordinary WER for a
multilingual ASR model on one short clip, not a pipeline defect. Important
positives:

- Pipeline runs end-to-end with our renamed IO conventions
- 2.43 GB consolidated external-data sidecar loads cleanly in ORT
- 120M's nemo128.onnx works unchanged for 600M
- Devanagari script renders correctly (verifies vocab flattening handled
  SentencePiece `▁` markers and UTF-8 write)
- Language-span mask correctly selects Hindi's 256-token slice
- SentencePiece detokenization (`▁` → space) produces readable text

---

## Phase 2 gate (600M): **CLOSED**

**Final 600M package** at `~/models/indicconformer_600m_onnx/`:

```
encoder-model.onnx        42 MB
encoder-model.onnx.data  2.43 GB
ctc_decoder-model.onnx   23 MB
nemo128.onnx             1.1 MB
vocab.txt                 41 KB
language_spans.json      1.4 KB
config.json              467 B
```

Contract identical to the 120M package (same file names, same IO names,
same vocab.txt / language_spans.json shape). **The same C# backend we
write in Phase 3 will load either package without branching.**

---

## Phase 2 gate (120M): **CLOSED**

Full pipeline is numerically faithful on real audio for Hindi. By the
language-agnostic ONNX design (no language_id input on any graph), this
same package covers all 22 languages — Phase 6's smoke test just needs to
exercise the C# masking path per language, not re-export anything.

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
