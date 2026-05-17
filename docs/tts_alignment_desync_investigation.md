# TTS / Forced Alignment Desync Investigation

User report: words in the Avalonia reader's word-highlight don't track the audio — they drift and eventually no longer line up with what's being spoken. Hypothesis: the EnTokenizer emits `[UNK]` tokens for characters outside the BPE base alphabet (numbers, symbols, abbreviation punctuation, etc.), causing the LM to synthesize audio that doesn't match the text the NFA aligner is then asked to align against.

## Plan

Probe pipeline on a real torture-test input:
1. Run Chatterbox TTS on the input markdown.
2. **Dump the actual LM token IDs**, particularly any `[UNK]` positions.
3. Run Parakeet ASR on the synthesized WAV → transcript.
4. Three-way compare:
   - Original input (post-markdown extraction)
   - LM-decoded token sequence (what the LM "saw")
   - ASR transcript (what the LM "said")
5. Re-run the NFA aligner against each text version; whichever produces correct word timings tells us where the upstream fix needs to go.

Test input: `/home/chris/Downloads/2026-05-XX_PER_Response_short.md` — short government-correspondence-style markdown with dates / numbers / abbreviations (specifics not committed; reference only).

## Run 1 — 2026-05-17 — probe across all 19 chunks of the torture-test markdown

Command:
```
dotnet run --project scripts/TtsAsrProbe -c Release -- \
  --in <torture_test.md> --voice <fry.wav> \
  --onnx-dir /mnt/data/models/chatterbox_export \
  --parakeet-dir ~/.local/share/Vernacula/models/parakeet \
  --out-wav /tmp/tts_asr_probe_out.wav
```

### Headline: the `<UNK>` hypothesis is wrong

**TOTAL: 2 UNK tokens across 19 chunks.** Both were single Unicode characters (looks like fancy quotation marks). UNK is not the desync source.

### What actually showed up

#### Finding 1 (primary): the LM is hitting its `DefaultMaxLmSteps = 256` cap on long chunks, truncating the audio

For every chunk with ≥155 text tokens (155, 157, 180, 190, 246, 261), synthesis produced **exactly 10.24 s of audio in exactly 256 LM steps**. That's the cap firing.

LM-step / text-token ratio observed (smaller chunks not at the cap):
- 12 → 65, 14 → 34/39, 16 → 95, 28 → 79, 43 → 109, 63 → 187, 72 → 171, 85 → 213.
- Roughly 1.4× text tokens → LM steps; ~40 ms audio per LM step.
- A 200-text-token chunk needs ~280 LM steps; the cap is 256. So the audio is truncated mid-chunk.

The chunker (`ParagraphChunker.MaxCharsPerChunk = 600`) hands the synthesizer chunks that systematically exceed what 256 steps can render. The NFA aligner is then asked to map the *full* chunk text onto only the audio that fits in 256 steps — guaranteeing per-word timing collapse over the unspoken tail.

This is the desync the user is seeing.

#### Finding 2 (secondary): Chatterbox occasionally rewords or drops content

Examples from the ASR transcript:
- Input chunk 7: starts "The CRA's response must address all the relevant issues raised by the claimant and explain why the CRA agrees or disagrees. It is not sufficient for the RTA to simply state that…" → ASR heard "…explain why **the additional information did not change the decision as a result**…". The LM either hallucinated or substituted text the input didn't contain.
- Chunk 9 input "Issues." → ASR "Use and". Single-word chunks are surprisingly fragile.
- "Programs" → "Program's", "Larocque" → "LeRoque" (proper-noun pronunciation drift, partly ASR error).
- "(SR&ED)" rendered as "SR and ED" — that's correct TTS behaviour for `&`, not a bug.
- "CRM 7.6.1.3" → "CRM 7.61.3" — multi-decimal numerics get squished by both TTS and ASR.

These are smaller-magnitude issues; the truncation in Finding 1 is the load-bearing one.

#### Side note: the input did contain content that justified care

Two UNK Unicode-quote characters had no audible impact in this run, but they do silently change BPE merge boundaries (UNK acts as a merge barrier per `EnTokenizer`'s comment). Worth normalizing fancy quotes to ASCII at the markdown-extraction stage as a cheap correctness improvement, even though it isn't the desync.

### Implications for the fix

The fix order should be:

1. **Either raise `DefaultMaxLmSteps`** (e.g. to 500+) so 600-char chunks fit, **or lower `ParagraphChunker.MaxCharsPerChunk`** so the chunker only produces chunks the LM can render in ≤256 steps. Lowering chunk size is cheaper (no GPU memory cost) and gives more graceful per-chunk failure modes; raising the cap costs LM memory + step latency. Lowering the chunker target to ~300 chars looks like the right tradeoff — at the observed ~1.4× ratio, that's ~210 LM steps, well under the cap.
2. **The aligner needs guard rails for the "audio shorter than text" case**: detect when chunkAudioDuration is suspiciously short relative to chunkText length and either skip alignment for that chunk or align only the prefix of the text that fits. Avoids smeared word timings even when truncation does occur.
3. Optional: fancy-quote normalization at markdown extraction (cheap; eliminates the small UNK count entirely).

`<unk>` text-normalization (issue #75) is still worth doing for non-spoken-language input like dates and equations, but it is **not** the cause of the current desync.

## Run 2 — 2026-05-17 — Chatterbox internal cross-attention as a replacement for NFA

### Discovery: Chatterbox already does this internally

While digging through chatterbox source (`chatterbox/models/t3/inference/alignment_stream_analyzer.py`), found that Resemble AI already extracts cross-attention alignment for their *own* hallucination-prevention logic. They identified three (layer, head) pairs that carry alignment signal:

```python
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]
```

Mechanism: set `tfmr.config.output_attentions = True` (HF auto-falls-back from SDPA to eager attention so the matrix becomes available), then register a forward hook on each of those three self-attention layers and capture the attention weights tensor. Mean across the three head slices gives a single (T_speech, T_kv) alignment matrix. Resemble uses this for `false_start` / `long_tail` / `repetition` detection and to force early EOS when the rollout goes off-rails — meaning they trust it enough to gate safety logic on it.

For us this means **phase 1 collapses from "discover alignment-bearing heads via heatmap mosaic" to "verify Resemble's pre-selected heads work on our inputs."**

### Phase 1 spike: `scripts/chatterbox_attention_spike/extract_alignment.py`

Standalone Python script (uses `.venv-chatterbox-export`). Loads `ChatterboxTTS.from_pretrained`, attaches forward hooks to the three aligned layers, synthesizes one chunk, captures per-step attention, builds the full alignment matrix, saves as `.npy` + heatmap `.png`.

Ran on three torture-test chunks. All three show clean monotonic-diagonal alignment with no off-diagonal hallucinations:

**Run 1a — date chunk (`"May 25, 2026"`, 96 LM steps, 3.8 s audio)** — `docs/attn_spike_heatmaps/run1_date.png`. Diagonal stripe from (speech row ~50, kv col ~35) down to (~145, ~48). Text-token range cols ~35-48; conditioning prefix (cols 0-35) receives no attention from speech generation.

**Run 1b — long prose chunk (201 chars, 432 LM steps, 17.2 s audio)** — `docs/attn_spike_heatmaps/run1_long_prose.png`. Beautifully clean diagonal from (row ~170, col ~40) to (~595, ~170). 425 decode steps tracking ~130 text positions in strict monotonic order. Thin 1-2 pixel stripe with no off-diagonal blobs.

**Run 1c — multi-decimal numeric chunk (`"CRM 7.6.1.3 reads:"`, 107 LM steps, 4.2 s audio)** — `docs/attn_spike_heatmaps/run1_numeric.png`. Critical test: this is the chunk that produced the worst NFA garbage (`CRM | <unk>.<unk>.<unk>.<unk> | reads<unk>`). Cross-attention shows clean monotonic diagonal from (row ~55, col ~33) to (~150, ~50). The digits get their own attention positions just like any other text token — **the LM model knows exactly where it is in the input regardless of token category.** The vocab/training problems that hobble NFA simply don't apply here, because we're reading the LM's own self-knowledge, not a separate ASR model's interpretation of the audio.

### Implications

- The technique works. Resolution ~40 ms per speech step (~25 Hz), more than enough for word-level highlight.
- No NFA-style vocab/training limitations: anywhere the model produced audio, the alignment knows what text caused it.
- No vocab pre-export work needed; we can use the existing Chatterbox checkpoint.
- The cost at runtime: forcing `output_attentions=True` on the model config switches all 30 layers from SDPA to eager attention. That's a measurable slowdown for full-model inference. For phase 2 we'll want to either (a) only compute attention for the 3 needed layers, or (b) re-export an alignment-specific ONNX graph that outputs just those layers' attention.

### Phase 2 plan (C# side)

1. **Modify `scripts/chatterbox_export/export_chatterbox_to_onnx.py`** to add an optional alignment-output mode for `language_model.onnx`. Either:
   - Add three attention output tensors (one per aligned layer) to the existing graph. Quadratic-in-seq-length per layer; for 1024 max LM steps × 16 heads × fp16, that's ~40 MB per layer per pass. Manageable.
   - OR ship a sidecar graph (`language_model_with_attn.onnx`) used only when alignment is requested. Cleaner separation but larger total bundle size.
2. **C# side: extend `AcousticLM`** to optionally collect attention per layer per rollout step. Build the (T_speech, T_kv) matrix incrementally during the generate loop, slice the text-token cols at the end.
3. **Replace `NemoNfaAligner` in `SynthesisService`** with a `ChatterboxAttentionAligner` that runs after synthesis and converts the attention matrix → word timings via:
   - For each speech step row, argmax → text-token position
   - Group adjacent speech steps that share the same text-token argmax → that's the text-token's audio span
   - Map text tokens back to original input words via the BPE token-to-word boundary tracking
4. **Drop the NFA bundle entirely** from the reader app's required setup. Optional: keep it as a fallback for non-Chatterbox audio inputs in the future.

Open follow-up: text-normalization (issue #75) — still worth doing for spoken quality even though it no longer matters for alignment, since the LM still mispronounces things like `"7.6.1.3"` as `"seven point sixty one point three"`.
