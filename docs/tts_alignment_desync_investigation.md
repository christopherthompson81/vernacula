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
