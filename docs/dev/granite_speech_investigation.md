# Granite Speech 4.1 investigation log

Running log for the IBM Granite Speech 4.1 ASR backend feasibility spike.
Issue reference: [#28](https://github.com/christopherthompson81/vernacula/issues/28).

Each entry is one run or one discrete investigation, stamped with local
date/time. The goal of this doc is to land architectural clarity *before*
writing the export, so the script in [scripts/granite_export/](../../scripts/granite_export/)
is shaped by what the model actually is rather than what the model card
implies.

---

## Run 1 — 2026-05-08 (architecture probe, no execution yet)

**Question:** What does `ibm-granite/granite-speech-4.1-2b` actually look
like at the tensor level? The issue describes "encoder + Granite LLM
decoder, similar shape to Cohere/VibeVoice/Qwen3-ASR" and a "dual-head
CTC encoder (graphemic + BPE)" that the runtime must fuse. Before I copy
the [cohere_export/](../../scripts/cohere_export/) skeleton I want to know which of those
analogies actually hold.

**Method:** No code yet. Read the upstream model card, the top-level
[`config.json`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/resolve/main/config.json),
the [`preprocessor_config.json`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/resolve/main/preprocessor_config.json),
and the underlying [`granite-4.0-1b-base/config.json`](https://huggingface.co/ibm-granite/granite-4.0-1b-base/resolve/main/config.json).
Cross-reference against existing exports to anchor patterns.

### Findings

**Top-level architecture: `GraniteSpeechForConditionalGeneration`,
`model_type: granite_speech`.** Three sub-configs:

| Block | Type | Shape |
|---|---|---|
| `encoder_config` | `granite_speech_encoder` | 16-layer Conformer, hidden 1024, 8 heads × 128 dim, conv kernel 15, input_dim 160, output_dim 348, max_pos_emb 512, context_size 200 |
| `projector_config` | `blip_2_qformer` | 2 hidden layers, 16 heads, hidden 1024, intermediate 4096, cross-attn every layer, `max_position_embeddings 2048` |
| `text_config` | `granite` (`GraniteForCausalLM`) | 40 layers, hidden 2048, 16 heads, num_kv_heads 4 (GQA), head_dim 128, intermediate 4096, vocab 100353, RMS-norm |

**Audio token id:** `100352` (vocab is 100353; the audio placeholder is the
last id). `<|audio|>` is inserted into the prompt before the projector
embeddings get spliced in. Mirrors Qwen3-ASR's `<|audio_pad|>` pattern but
single-token rather than padded-span.

**Mel frontend:** sr 16 kHz, n_fft 512, hop 160, win 400, 80 mels.
Encoder `input_dim 160 = 80 × 2` ⇒ adjacent-frame stacking before the
encoder, which is where the "2× downsampling at the encoder" claim from
the card maps onto the tensor shapes. So the runtime mel produces
`[batch, 80, T]`, the export prepends a frame-stack reshape, and the
Conformer sees `[batch, T/2, 160]`. We can either (a) bake the stack into
the encoder ONNX or (b) keep it on the host. Cohere and Qwen3 both bake
the analogous frontend reshapes into the encoder graph; doing the same
here keeps the C# side simpler.

**Dual-head CTC reality check.** The card says
"graphemic head (348) + BPE head (100,353), fused at decode time via
frame-posterior weighting." The encoder config exposes `output_dim: 348`
(the graphemic head). The BPE head with vocab 100,353 is **not** an
auxiliary CTC output the encoder ONNX needs to expose — it is the
*decoder* LM head, which always projects to vocab 100,353. The "fusion"
the card describes is internal to the projector / training loss, not
something the runtime sees as two parallel encoder outputs.

> **Implication:** the export shape is closer to Cohere/Qwen3-ASR than
> the issue suggested. We expose **one** encoder ONNX, **one** projector
> ONNX (or fused into the encoder), and a split prefill/step decoder
> pair. There is no "dual-head encoder graph" to design.

This is the single biggest reframing from this run. The issue text needs
an update — see Open questions below.

**Context window: 4096, not 128k.** `text_config.max_position_embeddings`
is 4096. The base Granite-4.0-1b-base config (the one the card links to)
shows `131072` because that LLM is trained for long context, but the
*speech* checkpoint reset the position-embedding budget to 4096 during
modality alignment. RoPE theta is 10000 (default), trained over 4096
positions. Practical implication: **don't promise long-form decoding
beyond ~4k tokens in the C# runtime UX**. Audio tokens count toward the
4096 budget — at 10 Hz acoustic embedding rate, 4096 tokens minus prompt
overhead is roughly 6–7 minutes of audio before degradation, before any
generated transcript tokens. Long files need segmentation (which Vernacula
already does for every other backend).

**Granite micro-architecture knobs.** The decoder applies four scalars
that aren't in vanilla Llama-style transformers:

- `attention_multiplier: 0.0078125` (= 1/128, *not* 1/√head_dim ≈ 0.088)
- `embedding_multiplier: 12.0`
- `logits_scaling: 8.0`
- `residual_multiplier: 0.22`

These come along automatically when the export traces through
`GraniteForCausalLM`. Listed here as a parity check item: if the ONNX
output diverges from the PyTorch reference by a constant scale, suspect
one of these dropped through the cracks during a graph-rewrite pass.

**LoRA already merged.** `has_lora_adapter: false` at the top level
means the released checkpoint is a merged-weights snapshot; no separate
adapter files to load before export. Simpler than the VibeVoice case.

**Tied embeddings: split.** `tie_word_embeddings: false`. The decoder has
distinct embedding and LM-head matrices. The Cohere-style external-data
sharing trick (renaming `decoder_init` weights to share storage with
`decoder_step`) still applies between the two decoder graphs, but there's
no intra-graph tie to exploit.

**Projector is a real BLIP-2 Q-Former.** `model_type: blip_2_qformer`,
not a custom "Window Query Transformer." The card's "windowed Q-Former"
phrasing means *the same Q-Former is applied repeatedly over 15-frame
windows of encoder output, with 3 trainable queries per window* — the
windowing is in how it's *invoked*, not in the module class. `transformers`
ships the Q-Former; we get it for free if we trace the public model.

### Pattern anchor: what the export script will look like

Given the above, the closest reference is [cohere_export/](../../scripts/cohere_export/) (encoder
+ split KV-cache decoder) merged with [vibevoice_export/](../../scripts/vibevoice_export/)'s
`_common.py` helpers. Concrete graph layout:

| ONNX | Inputs | Outputs |
|---|---|---|
| `mel.onnx` | `audio [batch, samples]` | `mel [batch, 80, T]` |
| `encoder.onnx` | `mel [batch, 80, T]` | `acoustic [batch, T/2, 1024]` (with frame-stack baked in) |
| `projector.onnx` | `acoustic [batch, T/2, 1024]` | `audio_embeds [batch, T/10, 2048]` (Q-Former, 5× downsample, into LLM hidden dim) |
| `decoder_init.onnx` | `input_ids`, `audio_embeds`, `audio_mask` | `logits`, `present_kv` (40 × 2 × [batch, 4, seq, 128]) |
| `decoder_step.onnx` | `input_id`, `past_kv` | `logits`, `present_kv` |

Whether `projector.onnx` stays separate or fuses into the encoder is a
post-spike call — fusing saves one session creation; keeping them split
matches the Q-Former's distinct shape and makes word-timestamp surgery
for the `-plus` variant easier.

### Sequenced plan (matching the project's standard workflow)

The user's standard cadence — **python export → python parity → C# CLI
+ parity → performance → C# GUI** — maps to issue splits. Filing as
sub-issues of #28 keeps each PR-sized:

1. **Python export skeleton** (this PR): scripts/granite_export/ with
   the export script stubbed, README, requirements, layout decisions
   captured here. Lands runnable enough to produce *some* ONNX bundle on
   the base 4.1-2b checkpoint, even if parity is only spot-checked.
2. **Python parity** (next issue): a Cohere-style smoke test that runs
   the exported bundle against the reference `transformers` pipeline on
   a fixture clip and asserts max-token agreement and mel/encoder/decoder
   tensor diffs under thresholds set by the cohere_export parity table.
   Likely needs a `_common.py` lift from vibevoice_export.
3. **C# CLI + parity** (next issue): `GraniteSpeech` backend in
   [Vernacula.Base/](../../src/Vernacula.Base/), wired through `OrtSessionBuilder`
   and (where applicable) `BatchSizer`. CLI smoke + golden transcript on
   one Vernacula benchmark clip. **Note:** the abstraction programme
   closed without `KvCacheBinding` being shared (see
   [inference_abstraction_investigation.md](inference_abstraction_investigation.md)
   Run 3) — Granite's KV layout (40 layers × split K/V × GQA 4-head)
   is a fifth structurally distinct shape, so it gets its own IOBinding
   path, not a shared one. Cohere's split self/cross pattern is the
   closest reference for the C# decode loop.
4. **Performance** (next issue): IOBinding + GPU-resident KV after the
   CLI is correctness-validated, batching sweep modeled on
   `sweep_qwen3_asr_batching.py`, WER on Vernacula's benchmark vs
   Parakeet TDT v3 / Cohere to decide default-backend status.
5. **C# GUI** (next issue): Avalonia integration, Settings exposure,
   and the open UX question of where keyword-biasing input lives
   (per-transcription input field vs Settings preset).

The `-plus` (speaker-attributed + word timestamps) and `-nar` (non-AR)
variants are deferred until after step 3 lands on the base checkpoint.
`-plus` adds an output graph or post-processing path; `-nar` is a
different decoder topology with no AR KV cache and is effectively a
parallel pipeline.

### Open questions / things this run did *not* answer

- **BPE-head exposure.** I'm asserting the BPE head is the LM head, not
  a separate encoder output. This is the most-likely interpretation given
  the config layout, but the upstream training description ("frame-level
  posterior-weighted pooling using window 4 for BPE classification")
  hints at an intermediate auxiliary head used during *training*. If
  inference also reads frame-level BPE posteriors (e.g. for confidence
  scoring), the encoder ONNX would need a second output. **Verify in
  Run 2 by tracing the public model's `forward` and checking what the
  generation path actually consumes.**
- **Projector fusion vs split.** Decided as part of Run 2/3 once we have
  a concrete graph in hand and can measure session-creation overhead vs
  graph clarity.
- **Audio mask.** The `<|audio|>` token expands to a variable-length
  span of projector outputs at decode time. Mirror Qwen3's `audio_offset`
  + `audio_lengths` pattern for the prefill input shape, or keep it
  implicit by splicing on the host? Decide alongside the prefill graph.
- **Word-level timestamps for `-plus`.** Not designed yet; defer until
  the base export is parity-validated.
- **Issue text.** [#28](https://github.com/christopherthompson81/vernacula/issues/28)
  describes the encoder as "dual-head CTC (graphemic + BPE)" needing
  both heads exposed. Per this run that's a *training-time* fact, not
  an export-time one. The issue body should be updated (or a comment
  added) once Run 2 confirms.

**Next:** lay down the skeleton (this PR), then start Run 2 — actually
load the model and trace `forward` to confirm or refute the BPE-head
interpretation before writing the export script proper.

---

## Run 2 — 2026-05-08 (model loaded, forward traced)

**Question:** Does the Run 1 hypothesis hold under direct inspection of
the public checkpoint? Specifically: is the encoder a single-output
module (no exposed BPE head), and what is the actual prefill contract
that `decoder_init.onnx` needs to mirror?

**Method:** [`scripts/granite_export/inspect_granite_speech.py`](../../scripts/granite_export/inspect_granite_speech.py) walks the
module tree, runs each submodule on the processor's dummy-audio output,
and reads `GraniteSpeechForConditionalGeneration.forward` source for
the audio-merge logic.

**Environment:** transformers 4.57.6 (matches the upstream config's
`transformers_version`), torch 2.11, torchaudio 2.11. Discovered during
this run: the feature extractor uses **torchaudio**, not librosa
(initial requirements.txt listed only librosa); torchaudio added to
requirements.

### Observed shapes (2 s dummy audio @ 16 kHz)

```
input_features          [1, 100, 160]     float32   from GraniteSpeechFeatureExtractor
input_features_mask     [1, 21]           bool      from GraniteSpeechFeatureExtractor
encoder(input_features) [1, 100, 1024]    float32
projector(enc_out)      [1, 21, 2048]     float32
lm logits               [1, 39, 100353]   float32
lm DynamicCache         40 layers; layer0.keys [1, 4, 39, 128]  (GQA: 4 KV heads, head_dim 128)
```

### Run 1 hypotheses confirmed

- **Encoder is single-output.** `forward(hidden_states) -> Tensor`.
  No `aux_logits` / `ctc_logits` field on the model output.
  [`GraniteSpeechCTCEncoder.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/modeling_granite_speech.py)
  does use `self.out` (348-dim) and `self.out_mid` at the **middle
  layer** for *self-conditioning* — softmax of mid-layer CTC logits is
  projected back into the hidden stream — but those tensors are
  internal to the forward pass and not exposed as a return value. The
  "dual-head CTC encoder" wording in [#28](https://github.com/christopherthompson81/vernacula/issues/28)
  refers to a training-time loss / mid-layer self-conditioning trick,
  not two parallel ONNX outputs. The encoder ONNX is a single graph
  with one output of shape `[B, T, 1024]`.
- **Projector is a real BLIP-2 Q-Former + projection.** Internal
  modules: `qformer: Blip2QFormerModel` + `linear: Linear`. The
  windowing happens inside `forward`: pad seq_len to a multiple of
  `window_size=15`, reshape to `[batch * nblocks, 15, 1024]`, run
  Q-Former with 3 trainable queries per block, reshape back to
  `[batch, nblocks * 3, 1024]`, project to 2048. Single-input,
  single-output module — exports cleanly.
- **Decoder accepts `inputs_embeds`** with `past_key_values` as a
  `Cache` or list. Standard `GraniteForCausalLM` contract — same as
  any other Granite LLM export.

### Run 1 hypotheses revised

- **Mel + frame-stack live in the feature extractor, not the encoder.**
  The processor produces `input_features [B, T_post_stack, 160]`
  directly. The encoder's `input_dim 160` is the *post-stack* dim, not
  the pre-stack `[80, T]` shape I'd planned. Update to the planned
  layout: `mel.onnx` (or its C# replacement) outputs
  `[batch, T_stacked, 160]`, **not** `[batch, 80, T]`. Either bake the
  stacking into mel.onnx or replicate
  `GraniteSpeechFeatureExtractor` logic on the host.
- **Audio splicing is `masked_scatter`, not concat.** `forward`'s
  merging logic ([modeling_granite_speech.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/modeling_granite_speech.py)
  `get_merged_audio_embeddings`):

  ```python
  is_audio_index = input_ids == config.audio_token_id  # 100352
  llm_input_ids = where(is_audio_index, 0, input_ids)
  inputs_embeds = embeddings(llm_input_ids)
  if input_features_mask is not None:
      audio_features = audio_features[input_features_mask]
  inputs_embeds = inputs_embeds.masked_scatter(
      is_audio_index.unsqueeze(-1), audio_features)
  ```

  The prompt ships with N copies of `<|audio|>` (one per projector
  output frame), and the projector's masked output replaces those N
  positions in the embedding stream. So **the prompt construction is
  audio-length-aware** — the C# side has to pre-tokenize a prompt with
  the right number of `<|audio|>` placeholders before tokenization.
  Concretely, the processor inserts
  `nblocks(T_input/15) * 3 / projector_downsample_rate * something`...
  actually simpler: the processor inserts as many `<|audio|>` tokens
  as the projector will eventually emit (21 in the 2s example).
  C# runtime needs the same arithmetic.

### Final decoder cache contract

```
DynamicCache with 40 layers
  layer.keys   [B, 4, S, 128]
  layer.values [B, 4, S, 128]
```

That's 40 × 2 = 80 KV tensors per step. Total KV bytes per token (fp32):
`40 layers * 2 (K/V) * 4 KV heads * 128 head_dim * 4 bytes = 163,840
B/token = 160 KiB/token`. For a 4096-position cache budget that's
640 MiB per stream — fits in any modern GPU but nontrivial. Matches the
Cohere split-self-cross pattern in spirit; the Cohere KV layout is
distinct enough that it shouldn't reuse a `KvCacheBinding` (per
[inference_abstraction_investigation.md](inference_abstraction_investigation.md)
Run 3, KV layouts are not cross-portable).

### Resulting export plan (revised from Run 1)

| ONNX | Inputs | Outputs |
|---|---|---|
| `mel.onnx` (TBD) | `audio [B, samples]` | `input_features [B, T, 160]`, `input_features_mask [B, T_proj]` |
| `encoder.onnx` | `input_features [B, T, 160]` | `encoder_hidden [B, T, 1024]` |
| `projector.onnx` | `encoder_hidden [B, T, 1024]` | `audio_embeds [B, T_proj, 2048]` |
| `decoder_init.onnx` | `input_ids [B, S]`, `audio_embeds [B, T_proj, 2048]`, `audio_mask [B, T_proj]`, `attention_mask [B, S]` | `logits [B, S, 100353]`, `present_kv` (40×2 × `[B, 4, S, 128]`) |
| `decoder_step.onnx` | `input_id [B, 1]`, `past_kv` | `logits [B, 1, 100353]`, `present_kv` |

Note `decoder_init` keeps the audio-merging *inside* the graph (matches
Qwen3's pattern; Cohere doesn't need this because it's seq2seq).
That's a `masked_scatter` + an `index` from `input_features_mask` — both
ONNX-expressible.

`mel.onnx` is annotated TBD because torchaudio's mel may or may not
trace cleanly under `torch.onnx.export(dynamo=True)`. If it doesn't,
the fallback is to ship the algorithm in C# as `Vernacula.Base/Inference/Mel*.cs`
and skip the ONNX. Decided in Run 3.

### Open questions that survive into Run 3

- Will `torch.onnx.export(dynamo=True)` accept torchaudio's
  `MelSpectrogram` + the frame-stacking helper as a single graph? If
  not, fall back to a host-side C# implementation.
- The projector's `nblocks = ceil(T/15)` padding produces a
  data-dependent shape. The dynamo exporter is strict about
  `GuardOnDataDependentSymNode` errors here (per cohere_export README
  notes). If it rejects the projector, we patch with a fixed-shape
  wrapper or fall back to the legacy TorchScript exporter.
- Does the prompt construction need to happen in C# string-space, or
  can we bake it into a tokenizer-level helper? Punt to the C# CLI
  step.

**Next:** start writing the actual exports in
[`export_granite_speech_to_onnx.py`](../../scripts/granite_export/export_granite_speech_to_onnx.py)
— encoder first (simplest), then projector, then the decoder pair,
then the mel question.

---

## Run 3 — 2026-05-08 (export landed, four stages parity-green)

**Question:** Land the four ONNX graphs for the base 4.1-2b checkpoint
and confirm parity vs the reference `transformers` forward.

**Method:** Replaced the stub with a full export script. Iterated on each
piece until parity held within fp32 noise. Parity harness:
[`test_parity.py`](../../scripts/granite_export/test_parity.py).

### Final parity (B=1, 2 s dummy audio, CPU fp32, ORT 1.25)

| Stage | max-abs-diff |
|---|---|
| `encoder.onnx`      | 3.4e-4 |
| `projector.onnx`    | 1.4e-6 |
| `decoder_init.onnx` | 3.9e-5 (logits), 4.4e-5 (KV) |
| `decoder_step.onnx` | 9.1e-6 (logits), 2.3e-5 (KV) |

Reference yardstick: `cohere_export` modern-path encoder lands ~3e-6.
Our encoder is two orders looser at 3e-4 because of patch (1) below.
Acceptable for fp32 LM consumption; worth a re-look if the C# CLI flags
WER deltas vs PyTorch.

### Three patches were necessary

**(1) 5-D SDPA in the encoder → manual math.** `GraniteSpeechConformerAttention.forward`
runs `F.scaled_dot_product_attention` on
`[bsz, num_blocks, num_heads, ctx, head_dim]` — five dimensions for
block-windowed attention. Torch 2.11's `aten.scaled_dot_product_attention`
ONNX adapter only handles 4-D Q/K/V and fails with
`only 4D query, key, and value are supported`. The original code uses the
MATH backend, so we inline its math equivalent:
`out = softmax((Q @ K^T) * scale + attn_mask) @ V`. That's mathematically
identical — but the manmade matmul accumulation order differs from
SDPA's, which is the (~3e-4 vs ~3e-6) discrepancy we see in the encoder
parity number.

Lives in
[`_patch_encoder_attention_for_export`](../../scripts/granite_export/export_granite_speech_to_onnx.py).
Mirrors the cohere_export Conformer-attention patch in spirit, though
the shape mismatch is different — Cohere's was a B=1 specialization of
`pos_emb.expand`, ours is the SDPA arity limit.

**(2) `attn_implementation="eager"` for the language model.** The default
SDPA path goes through `transformers.integrations.sdpa_attention.sdpa_attention_forward`,
which contains a data-dependent branch `attention_mask.shape[-1] != q.shape[-1]`
to switch between causal-only and explicit-mask SDPA. The dynamo
exporter cannot guard symbolic dims through this branch and raises
`GuardOnDataDependentSymNode` on the decoder_step trace. Loading with
`attn_implementation="eager"` keeps both prefill and step on the eager
attention path, which has no such branch and traces cleanly. No
numerical impact (eager and SDPA are mathematically identical at fp32).

**(3) Audio-merge: `masked_scatter` → cumsum + gather + where.** This was
the load-bearing one. The reference's `get_merged_audio_embeddings` uses:

```python
valid = audio_features[input_features_mask]   # NonZero + GatherND
embeds = embeds.masked_scatter(is_audio.unsqueeze(-1), valid)  # ScatterND
```

This traced WITHOUT raising any export error, but the resulting graph
produced **wrong logits** at every position from the first audio token
onward. Per-position max-diff:

```
positions 0-2 (pre-audio):    < 5e-6   ✓
positions 3-23 (audio span):  ~10-14   ✗
positions 24-38 (post-audio): ~5-15    ✗
```

In PyTorch the same wrapper's output matched the reference forward to
**0.000e+00** — proving the bug was strictly in the dynamo-to-ONNX
conversion of the boolean-index/masked_scatter pair, not in the wrapper's
logic. Inspecting the produced ONNX showed:

```
NonZero(audio_mask) -> indices
GatherND(audio_embeds, indices) -> flat_audio
ScatterND(text_embeds, scatter_indices, flat_audio) -> merged
```

The most likely cause is that PyTorch's `masked_scatter` walks the
source tensor sequentially in *flat* order while ScatterND requires
*explicit per-element [b, s] indices* — and the translator must derive
them from the bool mask in a way that doesn't match `masked_scatter`'s
flat-walk semantics for B>=2 with mixed mask densities. (This explains
why step parity stayed clean: the step graph has no scatter.)

Workaround: replace the boolean-index + masked_scatter pair with

```python
audio_idx = is_audio.long().cumsum(dim=1) - 1
audio_idx = audio_idx.clamp(min=0)
gathered = torch.gather(audio_embeds, 1, audio_idx.unsqueeze(-1).expand(-1, -1, D))
merged = torch.where(is_audio.unsqueeze(-1), gathered, text_embeds)
```

This avoids both NonZero and ScatterND, traces to plain Gather + Where,
and gives 3.9e-5 logits parity (down from 14.6).

Side benefit: the reformulation does not need an `audio_mask` graph
input at all — `cumsum(is_audio)` derives the per-position audio index
directly from `input_ids`, and the padding slots in `audio_embeds` are
never gathered because no audio token in `input_ids` points at them.
The C# runtime contract simplifies from
`(input_ids, audio_embeds, audio_mask, attention_mask)` to
`(input_ids, audio_embeds, attention_mask)`.

Filing a minimal-repro issue against PyTorch is on the to-do list once
the C# CLI lands — until then this workaround is the right answer
even if the upstream bug gets fixed, since the cumsum form is also
cheaper at runtime (no NonZero, no Gather-with-indirection).

### Other shape decisions made during this run

- **B=2 mixed-length dummy inputs.** `make_dummy_processor_inputs`
  builds a B=2 batch with one full-length and one half-length audio
  segment, padding both `input_features` and `input_ids` to the longer
  length. Single-batch dummies cause the dynamo exporter to specialize
  `batch=1` and reject the dynamic-shape contract. This is the same
  pattern Cohere uses; the Granite Speech twist is that audio token
  count varies between batch items, so prompt padding is also needed.
- **`cache_position` is 1-D.** HuggingFace convention: `cache_position`
  is `[seq_len]`, NOT `[B, seq_len]`. Passing 2-D triggers RoPE
  position_ids broadcast errors before the export even reaches dynamo.
  decoder_step exposes `cache_position` as a non-dynamic 1-element
  input.
- **Varargs + nested dynamic_shapes.** `decoder_step.forward(input_id,
  attention_mask, cache_position, *past_kv)` exposes 4 top-level
  parameters to torch.export (the varargs collapses), so
  `dynamic_shapes` is a 4-tuple where the 4th entry is itself a tuple
  of 80 dicts (one per K/V tensor across 40 layers).

### Final ONNX package shape

| File | Size | Notes |
|---|---|---|
| `encoder.onnx` + `.onnx.data` | 1.68 GB | Conformer encoder, 16 layers |
| `projector.onnx` | 137 MB | Q-Former + linear projection |
| `decoder_init.onnx` + `.onnx.data` | 7.0 GB | LM prefill + audio fuse |
| `decoder_step.onnx` + `.onnx.data` | 7.0 GB | LM step on past KV |
| Tokenizer + processor assets | ~10 MB | tokenizer.json, etc. |
| **Total** | **~16 GB** | fp32 |

The decoder pair currently ships **two full copies** of the 7 GB LM
weights. Cohere's export shares external-data files between init and
step via a rename trick — that optimization is queued for a follow-up
issue alongside fp16 / sharded export. For now correctness > size.

### What's deferred

- **Mel ONNX (`mel.onnx`).** The torchaudio MelSpectrogram + Granite's
  frame-stacking step is not exported. The C# runtime can produce
  `input_features [B, T_stacked, 160]` directly using a torchaudio-
  compatible filterbank. If parity becomes an issue, revisit.
- **Weight sharing across decoder graphs** (see above).
- **fp16 / quantized export.** Today's run is fp32 throughout for
  parity validation. fp16 should mostly come for free via `--dtype
  float16`, but each Granite micro-architecture multiplier
  (`embedding_multiplier=12`, `logits_scaling=8`, `attention_multiplier=1/128`,
  `residual_multiplier=0.22`) needs a parity check at fp16 precision.
- **`-plus` and `-nar` variants.** As planned in Run 1.

### Status

Python export shipping. Next step in the workflow (per the standard
cadence) is the **Python parity → C# CLI + parity → performance →
C# GUI** chain, each tracked under issue #28 or its sub-issues.

---

## Run 4 — 2026-05-08 (encoder full-attention pivot, end-to-end transcription parity)

**Question:** The Run 3 encoder export was numerically green at the
**trace shape** but failed at runtime for any audio whose block count
differed from the dummy. Can the encoder be reshaped to support
variable-length input through ONNX, or do we need an architectural
pivot?

**Method:** Repeated attempts at making `num_blocks =
math.ceil(num_features / context_size)` symbolic through the dynamo
exporter — `unflatten` instead of `reshape`, deriving `num_blocks` from
`hidden_states.shape[1]` after padding (SymInt path), folding
`B*num_blocks` into the leading axis, switching to the legacy
TorchScript exporter. Each attempt fixed one reshape and produced a
new failure further downstream. Full diagnosis in run notes; the short
version is that **dynamo evaluates expressions like `num_blocks * ctx`
at trace time and bakes the result as a static constant in every
downstream `Reshape` target**, so making `num_blocks` itself symbolic
is necessary but not sufficient.

### Cohere precedent

The Cohere encoder export hit a similar shape of problem with
`_needs_conv_split` and `_check_input_shape` — Python control-flow
that bakes shape constants into the trace. The fix was to monkey-patch
the path away (`_needs_conv_split = lambda x: False`); see
[cohere_export L340-343](../../scripts/cohere_export/export_cohere_transcribe_to_onnx.py#L340-L343).
The pattern is "bypass the optimisation, accept a worse worst case."

Cohere/Qwen3/VibeVoice don't have an exact analog because none of them
use **block attention**. Block attention is what creates the
`num_blocks` dim in shape arithmetic; full attention sequences keep T
as a single symbolic dim that the exporter handles cleanly.

### Pivot: full attention with block-diagonal mask

Replaced the block-reshape attention in `_patch_encoder_attention_for_export`
with full attention over the padded T dimension, plus a block-diagonal
additive mask. Mathematically identical to the upstream block
attention; eliminates `num_blocks` from shape arithmetic entirely.

Memory considerations: a naive [T, T, head_dim] rel-pos lookup blows
up to ~5 GB at 60 s of audio. To stay efficient at long T, the
rel-pos bias is computed via the einsum decomposition

```
q_dot_emb[b, h, i, k] = (Q @ rel_pos_emb.weight.T)[b, h, i, k]
pos_bias[b, h, i, j] = q_dot_emb[b, h, i, rel_idx[i, j]]
```

where `rel_idx[i, j] = clamp(i - j, -ctx, ctx) + max_pos_emb`. The
intermediate `q_dot_emb` is [B, H, T, num_indices=1025], ~33 MB at
T=1000 — well bounded. `pos_bias` is then gathered per-pair and
zeroed across blocks via `same_block` mask.

Cost trade-off:

| Audio | Block attn (upstream) | Full attn (this export) |
|---|---|---|
| 4 s (T~200, 1 block) | O(T·ctx) = 40 k | O(T²) = 40 k (equal) |
| 30 s (T~1500, 8 blocks) | 300 k | 2.25 M (~7×) |
| 60 s (T~3000, 15 blocks) | 600 k | 9 M (~15×) |

For typical Vernacula segments (10–30 s) the cost is acceptable. Long
clips are already chunked by the transcript pipeline elsewhere.

### Final parity (full-attention encoder)

| Stage | max-abs-diff | Notes |
|---|---|---|
| `encoder.onnx` | 3.4e-4 | unchanged at trace shape; manual-math accumulation order |
| `projector.onnx` | 1.4e-6 | unchanged |
| `decoder_init.onnx` | 3.9e-5 logits, 4.4e-5 KV | unchanged |
| `decoder_step.onnx` | 9.1e-6 logits, 2.3e-5 KV | unchanged |

**End-to-end transcription on a 6.4 s VCTK clip (multi-block, 2 encoder
blocks at the upstream block-attn rate):**

```
ORT: "Hello, I'm from Ontario. I hope that you will select my voice for your project. Thank you."
Ref: "Hello, I'm from Ontario. I hope that you will select my voice for your project. Thank you."
```

Exact text match. Token streams identical except for the trailing EOS
(reference produces it; the smoke loop terminates on it). Same
end-to-end test on a 3.5 s clip (1 block) also matches.

### Notes on the rel-pos derivation

The upstream encoder pre-computes a `[ctx, ctx]` `attention_dists`
buffer in its `__init__`. The full-attention rewrite ignores that
buffer and computes `rel_idx` from runtime `positions` of shape `[T]`.
The two are equivalent: when restricted to within-block pairs they
produce the same Shaw-style relative distance indices. The buffer is
still registered on the encoder but unused in the patched forward.

### Tooling: `transcribe_smoke.py`

Added [`scripts/granite_export/transcribe_smoke.py`](../../scripts/granite_export/transcribe_smoke.py)
as the end-to-end "Python parity" stage of the workflow:

- Loads ORT sessions for all four ONNX graphs.
- Runs the full pipeline (encoder → projector → decoder_init →
  decoder_step loop) on a real audio clip with greedy decoding.
- Compares output text and token IDs against
  `model.generate(..., do_sample=False, num_beams=1)`.
- Skipping the reference (`--skip-reference`) keeps the test fast for
  ORT-only iteration.

This catches integration bugs (cache_position handoff, KV layout,
projector window alignment) that per-stage parity (`test_parity.py`)
cannot reach. The next workflow stage — C# CLI + parity — uses the
same shape contract but with C#'s ORT runtime in place of Python.

### What's now well-bounded

- Encoder accepts arbitrary T (no num_blocks constraint).
- Projector handles arbitrary T with its window-padding logic.
- Decoder pair has been valid since Run 3.

### What's still deferred

- ~~**Mel ONNX**: still host-side. C# will reproduce torchaudio mel +
  frame-stacking. Decision unchanged from Run 1.~~ **Resolved in Run 5.**
- **Weight sharing across decoder graphs**: the 7 GB LM weights are
  duplicated between init and step. Cohere's external-data rename
  trick should apply.
- **fp16 / quantized export**: today fp32 throughout.
- **Encoder perf**: the full-attention reformulation is ~7-15× more
  attention work than the block attention at long T. If Vernacula's
  benchmark surfaces an encoder bottleneck on longer clips, revisit
  with a chunked-encoder strategy in the C# runtime (split into
  ≤ctx-frame segments, run encoder per segment, concatenate). The
  conv-kernel boundary effect is small (kernel size 15) and could be
  further reduced with overlapped chunking.
- **`-plus` and `-nar` variants**: as planned in Run 1.

---

## Run 5 — 2026-05-08 (mel.onnx, C# CLI parity smoke)

**Question:** Run 4 left the mel frontend as host-side work. Can we
follow the cohere_export / NeMo precedent and export it as ONNX too,
so the C# runtime doesn't need a torchaudio port?

**Method:** Wrap `processor.audio_processor.mel_filters` (a
`torchaudio.transforms.MelSpectrogram`) plus the
`GraniteSpeechFeatureExtractor` post-processing chain
(`log10`, `-8 dB` floor relative to per-clip max, divide by 4 + add 1,
drop-last-frame-if-odd, 2×-frame-stack) into a `MelWrapper` and trace
it through `torch.onnx.export(dynamo=True)`.

**Findings:**

- torchaudio's `MelSpectrogram` decomposes cleanly under dynamo. The
  resulting `mel.onnx` is 0.1 MB (no learned weights — only the static
  HTK mel filterbank and the FFT graph).
- The "drop last frame if odd" step in the upstream code is a
  data-dependent Python branch. Replaced unconditionally with
  `T_even = (T // 2) * 2`. Branch-free; equivalent.
- Parity vs the upstream processor on the 6.4 s VCTK clip: max-abs-diff
  = **4.5e-5** on `input_features` (well below the encoder's downstream
  3.4e-4 noise floor).
- End-to-end `transcribe_smoke.py` with `mel.onnx` in the path produces
  the same transcript as the processor-fed run, so the small mel diff
  doesn't propagate into a different decode.

**C# CLI parity smoke landed:**

- New project [`tests/GraniteSpeechSmoke/`](../../tests/GraniteSpeechSmoke/),
  a console app that drives the bundle from C# end-to-end (NAudio WAV
  load → mel.onnx → encoder → projector → decoder_init → step loop →
  GPT-2 ByteLevel BPE decode).
- Loads the prompt token IDs from `Fixtures/input_ids.bin` (a full BPE
  encoder in C# is a follow-up). Audio comes in via NAudio.
- Asserts the decoded text matches `Fixtures/expected_text.txt`, which
  was produced by `model.generate(...)` in
  [`scripts/granite_export/dump_inputs_for_csharp_smoke.py`](../../scripts/granite_export/dump_inputs_for_csharp_smoke.py).
- **Result on the 6.4 s VCTK clip:** exact text match.

  ```
  ORT  transcript: "Hello, I'm from Ontario. I hope that you will select my voice for your project. Thank you."
  Ref  transcript: "Hello, I'm from Ontario. I hope that you will select my voice for your project. Thank you."
  exact match: True
  ```

This is the "C# CLI + parity" stage of the standard cadence —
proves the export contract is consumable from C# end-to-end. What the
smoke does NOT yet exercise:

- Encoding arbitrary prompts in C# (BPE encoder).
- A real `GraniteSpeech.cs` backend in `Vernacula.Base` integrated
  into `Vernacula.CLI` with batching, IOBinding, etc — that's the
  performance / production stage.

### Final ONNX package shape (with mel.onnx)

| File | Size |
|---|---|
| `mel.onnx` | 0.1 MB |
| `encoder.onnx` + `.onnx.data` | 1.68 GB |
| `projector.onnx` | 137 MB |
| `decoder_init.onnx` + `.onnx.data` | 7.0 GB |
| `decoder_step.onnx` + `.onnx.data` | 7.0 GB |
| Tokenizer + processor assets | ~10 MB |
| **Total** | **~16 GB** fp32 |

### Status

Python export + per-stage parity + end-to-end transcription parity in
both Python and C# are all green on the 6.4 s and 3.5 s VCTK clips.
The export contract is stable enough to start the production
`GraniteSpeech.cs` backend integration into `Vernacula.Base` /
`Vernacula.CLI`.
