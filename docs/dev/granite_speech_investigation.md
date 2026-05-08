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
