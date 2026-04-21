# Whisper large-v3-turbo Integration — Investigation

Goal: add a multilingual ASR backend that (a) covers high-speaker-count languages the
existing backends under-serve or serve only via VibeVoice (CUDA + 24 GB VRAM), and
(b) runs on CPU so users without a big GPU can still access those languages.

## Run 1 — 2026-04-20 (ONNX export path research)

Question: what is the best starting point for a Whisper-turbo ONNX? Custom export
vs. community export vs. official tooling?

Candidates surveyed:

1. **[onnx-community/whisper-large-v3-turbo](https://huggingface.co/onnx-community/whisper-large-v3-turbo)**
   (Xenova/onnx-community team). Produced by HF Optimum. Ships:
   - `encoder_model.onnx` in fp32, fp16, int8, q4 variants (external-data for fp32)
   - `decoder_model.onnx` (prefill — no past)
   - `decoder_with_past_model.onnx` (step — with past KV)
   - `decoder_model_merged.onnx` (both behind `use_cache_branch`)
   - `tokenizer.json`, `generation_config.json`, `preprocessor_config.json`
   - Actively used (~9 k monthly downloads), last updated 2025-03, MIT-derived licence.

2. **HF Optimum directly** (`optimum-cli export onnx --task ...-with-past`). Produces
   the same files as onnx-community. No reason to re-run when they've already
   published the output.

3. **Microsoft Olive / `onnxruntime.transformers.models.whisper.convert_to_onnx`**.
   Produces a single merged graph with beam search embedded as an ORT contrib op
   (`WhisperBeamSearch`). Fast for one-shot transcription, but eliminates
   token-by-token control, which is the opposite of what Vernacula needs for its
   segment-based streaming pipeline.

**Decision**: use **onnx-community/whisper-large-v3-turbo** as-is. No custom
exporter needed.

## Run 2 — 2026-04-20 (Cohere comparison: what would a custom exporter actually add?)

Question: Cohere Transcribe has several advanced inference features. Are any of
them things the onnx-community Whisper export doesn't cover, and would a custom
exporter therefore be justified?

Surveyed `CohereTranscribe.cs` and related integration in detail. Findings:

| Cohere feature | Whisper equivalent | Exporter or C# concern? |
|---|---|---|
| `decoder_init.onnx` + `decoder_step.onnx` split | `decoder_model.onnx` + `decoder_with_past_model.onnx` | **Already in onnx-community**. Use these, skip the merged variant. |
| Smaller activation memory → more batch headroom | Same win: two specialized graphs instead of one `use_cache_branch` graph | Comes for free by picking the split pair. |
| IOBinding: cross-KV in CUDA device memory, reused across decode steps | — | **C#-side integration**. Port the pattern from Cohere code. |
| Length-sorted segment batching + VRAM-budgeted batch sizing + OOM fallback | — | **C#-side**. Port from Cohere. |
| `mel.onnx` preprocessor as ONNX graph | *not in onnx-community export* | **Deferred optimisation**. Compute log-mel in C# for v1; export a mel-only graph later if profiling shows it's a bottleneck. |
| Language forcing via context-block tokens (`TokStartOfContext` etc.) | Whisper's native `<\|lang\|>` prefix tokens | Different, simpler on Whisper. No exporter concern. |
| `CohereSyntheticTimestampMode` / `uniform_segment_frames_v1` | Whisper emits real timestamp tokens natively | **Delete workaround** for this backend; Whisper gives us real per-token timestamps. |

**Decision**: still no custom exporter needed for v1. All Cohere's VRAM /
batching advantages transfer to Whisper simply by picking the split-decoder pair
from onnx-community and porting the C# integration patterns. The one real
exporter-side optimisation Cohere has that onnx-community lacks — `mel.onnx` —
is deferred behind a profiling gate. We can add a small custom export script
later that bundles only the mel preprocessor if we need to.

## Integration plan (phases)

Each phase ends at a runnable checkpoint; we validate before moving on.

### Phase 1 — scaffolding and model fetch
- Branch: `feature/whisper-large-v3-turbo` (this one).
- Add `WhisperTurboSubDir`, `WhisperTurboEncoderFile`, `WhisperTurboDecoderInitFile`,
  `WhisperTurboDecoderStepFile`, `WhisperTurboTokenizerFile` constants to
  `Config.cs`.
- Register the file list with `ModelManagerService` so the existing download
  flow fetches it (mirror `Qwen3AsrFiles` / `CohereFiles` patterns).
- No backend code yet. Goal: a user can download the files via the model-manager
  UI.

### Phase 2 — C# backend class (`WhisperTurbo`)
- New `src/Vernacula.Base/WhisperTurbo.cs`. Three `InferenceSession` instances:
  encoder, decoder-init (no past), decoder-step (with past).
- Log-mel computation in C# using `MathNet.Numerics` (STFT + mel filterbank + log).
  Output: `[1, 128, T]` float32 matching Whisper's expected input.
- Greedy decode loop: prefill via `decoder_model`, then iterate on
  `decoder_with_past_model` until `<|endoftext|>` or max-length.
- Tokenizer: load `tokenizer.json` via a lightweight HF-style tokenizer port
  (check existing Qwen3-ASR / Cohere code for reusable byte-level BPE logic).
- Language handling via Whisper's native prefix tokens:
  `<|startoftranscript|><|lang|><|transcribe|>` (with `<|notimestamps|>` or real
  timestamps).
- IOBinding for cross-KV on CUDA EP, ported from `CohereTranscribe.cs`.
- Return a result shape compatible with `TranscriptionDb.UpdateResult()`.

### Phase 3 — batching & VRAM budgeting
- Port Cohere's length-sorted batch scheduling + `EstimateKvBytes` /
  `EstimateEncoderConvBytes` + OOM-halving fallback to `WhisperTurbo.cs`.
- Dynamic batch-size determination based on `cudaMemGetInfo`.

### Phase 4 — UI wiring
- Append `WhisperTurbo` to `AsrBackend` enum (`AppSettings.cs`).
- `SettingsViewModel`: `IsAsrWhisperTurbo`, `ShowWhisperTurboLanguagePicker`,
  `WhisperTurboLanguage` setting field.
- `SettingsWindow.axaml`: radio button + language picker (mirror Cohere /
  Qwen3-ASR).
- `TranscriptionService.cs`: new dispatch branch before the Parakeet fallback.
- `AsrLanguageSupport.cs`: Whisper's 99-language FrozenSet.
- Locale keys for the backend label/description (in `en.json`; translations can
  follow).

### Phase 5 — help-page update
- Add a row for Whisper-turbo to `first_steps/audio_input_quality.md`'s backend-selection
  table.
- Update `first_steps/settings_window.md`'s ASR table.

### Phase 6 — validation
- Transcribe representative files across several high-speaker-count languages
  Whisper should cover (Arabic, Japanese, Korean, Portuguese, Russian, Indonesian,
  Vietnamese) and compare to whatever the existing multilingual backends produce.
- Compare transcript quality vs. VibeVoice on languages where VibeVoice is the
  current "only option on CUDA" — confirms Whisper-turbo is an acceptable
  CPU-capable alternative.
- If CPU mel preprocessing is shown to dominate runtime on GPU, revisit the
  custom-export-of-mel-only deferred optimisation from Run 2.

## Deferred / not doing (v1)

- **Custom ONNX exporter.** Only if `mel.onnx` profiling shows it's needed, or if
  onnx-community ever stops maintaining their export.
- **Beam search.** Whisper can do beam, but greedy at low temperature is a
  well-known strong baseline and matches the per-token streaming pattern we're
  porting. Beam can be added later behind a settings toggle.
- **LM fusion.** Not part of the Whisper tradition; would be an entirely separate
  work item.
- **Word-level timestamps.** `onnx-community/whisper-large-v3-turbo_timestamped`
  exists if we want this; skip for v1.

---

Run log continues below as work progresses.

## Run 3 — 2026-04-20 (Phase 3b batching benchmark, en-US 600 s / 132 segments)

Question: does batching encoder + decoder-init + decoder-step across
segments reduce total ASR time?

Setup: `RecognizeBatched` with length-sorted scheduler, `initialBatchSize=8`,
OOM-halving fallback, per-item EOT tracking with EOT-as-pad for finished
items, no IOBinding. Compared against sequential `Recognize` from Phase 3a.

Results (CUDA, RTX 3090):

| Path | ASR time | Total wall-clock | RTF |
|---|---|---|---|
| Sequential (Phase 3a) | 35.8 s | 45.3 s | 0.072 |
| Batched B=8 (Phase 3b) | 39.8 s | 50.1 s | 0.079 |

Outputs: 0.996 sequence similarity, 0.998 vocab overlap — batched is
numerically identical modulo floating-point nondeterminism.

Finding: batching is **slower** on this workload. Root cause is per-step
memory bandwidth for KV-cache transport. Each decoder-step:
- Copies the 8 decoder-side KV tensors out of ORT into managed float[]
  via `ExtractFloatFromTensor` (≈ 2 MB × 8 = 16 MB per step at B=8).
- Re-wraps them in `DenseTensor<float>` on the next step.
Copies scale with B, so while batch=8 does fewer *step calls* than
sequential, each call moves 8× more bytes. The cross-over favours
sequential on short-variable-segment workloads (this 600 s file
averages 4.5 s/segment with a long tail).

Cohere sidesteps this with IOBinding — cross-attention KVs allocated
in CUDA device memory and bound once across all steps, never crossing
PCIe. We didn't port that optimisation in Phase 3b; doing so is the
natural follow-up if batching becomes load-bearing.

**Decision**: ship Phase 3b as correct-but-opt-in. CLI default remains
sequential; `--whisper-batch N` (N ≥ 2) switches to the batched path.
The batched implementation is a correct foundation for the IOBinding
optimisation when we have a concrete reason to chase it — a workload
with longer, more uniform segments (lectures, podcasts) would be where
we'd expect batching even without IOBinding to break even.

Deferred follow-up: IOBinding for KV transport. Expected to invert the
benchmark above.

## Run 4 — 2026-04-20 (per-phase timing instrumentation, sequential)

Question: where is the 35.8 s of ASR time actually going?

Added timing accumulators around each phase of `Recognize` and printed
the breakdown on `--benchmark`. Same 600 s en-US file, sequential path:

| Phase | Time | % |
|---|---|---|
| DecoderStep ORT loop | 14 991 ms | 46.1 |
| Mel compute (CPU) | 7 645 ms | 23.5 |
| Encoder forward | 7 511 ms | 23.1 |
| DecoderInit | 2 362 ms | 7.3 |
| DecoderStep extract/copy | 24 ms | 0.1 |
| Argmax | 0 ms | 0.0 |
| Token decode | 3 ms | 0.0 |

2113 step calls × 7.09 ms/call ORT → kernel-launch-overhead-dominated
(decoder-step compute is ~10 MFLOPs, sub-µs on a 3090; measured 7 ms is
~99.99 % overhead).

Retroactively explains Run 3: batching Phase 3b was slower not because
of memory copies (extract/copy is 0.1 %) but because B=8 increased
per-kernel compute while the fixed-per-call overhead stayed the same,
and longer stragglers dragged the batch out. IOBinding alone would not
have helped; CUDA Graphs would.

## Run 5 — 2026-04-20 (mel-in-graph optimisation + ONNX STFT trap)

Question: move mel from CPU to GPU via a small mel-only ONNX graph.
Expected 7.6 s → ~1 s.

First attempt: export with `torch.stft`. In PyTorch this is a fast
FFT-based op; after ONNX export it becomes the `ai.onnx.STFT` op, which
turns out to have no CUDA kernel in ORT (falls back to CPU with PCIe
memcpys — ORT warns `2 Memcpy nodes are added to the graph`). Bench on
this graph:

| Provider | ms/call |
|---|---|
| CPU EP | 148.7 |
| CUDA EP (w/ fallback) | 132.6 |
| Hand-rolled CPU FFT (baseline) | ~58 |

End-to-end: mel went from 7.6 s to 17.6 s — a 2.3× **regression**.

Second attempt: replace `torch.stft` with two `Conv1D` ops (cos/sin
kernels pre-windowed, forward-DFT sign). Same trick
`scripts/voxlingua107_export/src/conv_stft.py` used for the LID
pipeline a while back — same issue, same fix. Bench:

| Provider | ms/call | Δ vs STFT-op graph |
|---|---|---|
| CPU EP | 3.24 | 46× faster |
| CUDA EP | 0.90 | 147× faster |

End-to-end on the 600 s file:

| Phase | Before (CPU mel) | After (Conv1D-STFT mel.onnx) |
|---|---|---|
| Mel | 7 645 ms | 145 ms |
| Encoder | 7 511 ms | 7 520 ms |
| DecoderInit | 2 362 ms | 2 354 ms |
| DecoderStep ORT loop | 14 991 ms | 14 982 ms |
| ASR total | 35 759 ms | 28 104 ms |
| RTF | 0.072 | 0.059 |

Saved 7.66 s (21 %). Decoder-step loop is now 60 % of time — the next
target.

**Lesson for future exports**: `torch.stft` → `ai.onnx.STFT` is a trap;
always prefer the two-Conv1D construction when exporting signal-processing
graphs.

Deferred follow-up (now promoted to primary target after mel + unified
decoder): IOBinding + CUDA Graph capture for the decoder step. Kernel
launches dominate; graph capture replays a fixed-shape kernel sequence
with ~1-5 µs launch overhead instead of ~100-200 µs per call. Realistic
expectation: 15 s → ~4 s for the step loop.

## Run 6 — 2026-04-20 (unified decoder swap)

Question: swap the `decoder_model_fp16.onnx` + `decoder_with_past_model_fp16.onnx`
pair for the merged `decoder_model_merged_fp16.onnx` and measure the VRAM /
time delta.

The merged graph has a `use_cache_branch` bool input that selects prefill
(no past) vs step (with past). Prefill must still pass all 16
`past_key_values.*` inputs — the no-cache branch ignores them, but the
ONNX graph requires their tensors to be present. Zero-seq-len shapes
(`[B, 20, 0, 64]`) are valid and avoid any allocation. Similarly the step
call must pass `encoder_hidden_states`; zero-frame `[B, 0, 1280]` works.

| Variant | disk | VRAM peak (approx) | ASR time | RTF |
|---|---|---|---|---|
| Split pair (Run 5) | 662 MB | 662 MB | 28 104 ms | 0.059 |
| Merged single (Run 6) | 344 MB | 344 MB | 28 431 ms | 0.059 |

Savings: ~318 MB VRAM, single weight set. Time delta is within noise
(+326 ms, <2 %). Output is **byte-for-byte identical** — 100.0 %
sequence similarity on the 1589-word transcript.

This VRAM headroom is exactly the slack we'll want when we do the
IOBinding + CUDA Graph work next, because graph capture needs
pre-allocated fixed-shape KV buffers sized to the 448-token decoder
maximum. That pre-allocation is ~B × 4 layers × 20 heads × 448 × 64 ×
2 (key+value) × 4 bytes = ~9 MB per batch-item — not large, but the
unified decoder means we have ~300 MB more room for it and for the
batched encoder / decoder inputs.

Also in this run: fixed the models directory to the canonical
`~/.local/share/Vernacula/models/` (was `~/.local/share/Parakeet/models/`
from earlier ad-hoc testing).

## Run 7 — 2026-04-20 (IOBinding for the decoder loop)

Question: Phase 6a — port Cohere's IOBinding pattern to the Whisper
decoder loop. Per-step PCIe transfer of KV (~60 MB) was hypothesised
to dominate the 7.15 ms/call step cost; IOBinding keeps KV OrtValues
on CUDA across steps, so only logits round-trip for argmax.

Pattern: fresh `OrtIoBinding` per step (needed because KV shape grows
by one and ORT's cached output buffer would shape-mismatch otherwise).
Previous step's output OrtValues are bound as the next step's inputs
— the device memory is referenced, never copied. Logits bound to
`OrtMemoryInfo("Cpu", ...)` so CPU-side argmax reads without a
GPU→CPU transfer.

Results on the same 600 s en-US file:

| Phase | Pre-IOBinding (Run 6) | Post-IOBinding (Run 7) | Δ |
|---|---|---|---|
| Mel | 138 ms | 152 ms | — |
| Encoder | 7 509 ms | 7 823 ms | — |
| DecoderInit | 2 428 ms | **405 ms** | −2 023 ms, 6× |
| DecoderStep ORT | 15 046 ms | **2 118 ms** | −12 928 ms, 7× |
| ASR total | 28 431 ms | **13 283 ms** | −15 148 ms, 2.1× |
| RTF | 0.059 | **0.034** | 1.7× faster |

Per-step cost: 7.15 ms → 1.01 ms (86 % reduction). The remaining 1 ms
is fresh-IoBinding-creation overhead + ORT session bookkeeping — close
to the floor without CUDA Graph capture (which would require re-exporting
the decoder with an explicit attention-mask input, since the current
graph infers seq_len from KV tensor shapes and dynamic shapes are
incompatible with graph capture).

Output: 1.0000 sequence similarity, 1589 words, token-for-token identical.

**Bottleneck has shifted** — encoder is now 74 % of ASR time (was 30 %).
That promotes batched encoder (Path B) to the primary next target. The
unified decoder from Run 6 gave us ~318 MB VRAM headroom, which is the
slack we need for larger encoder batch sizes.

Deferred: IOBinding for the batched path (`TranscribeBatch`). The
non-batched path is already fast enough that the Phase 3b batched path
would not currently win — but if we port batched encoder next, the
batched decoder loop should also adopt IOBinding to stay consistent.

## Run 8 — 2026-04-20 (batched encoder — no-go, compute-bound)

Question: Phase 6b — batch encoder calls (B=8) to amortise weight-load
from HBM. Encoder is now 74 % of ASR time (7 509 ms on 132 segments).
Expected 2-3× reduction if the encoder is memory-bandwidth-bound.

Implementation: extracted `DecodeFromHidden(hidden, lang)` from the
existing IOBinding `Transcribe`, added `RunEncoderBatch(mels, B)`, and
refactored `Recognize` to group work items into batches of 8, run one
encoder call for all, then loop per-segment through the IOBinding
decoder.

Result — negligible win:

| Path | Encoder time | ASR total | RTF |
|---|---|---|---|
| B=1 (Run 7) | 7 823 ms | 13 283 ms | 0.0343 |
| B=8 (Run 8) | 7 582 ms | 13 156 ms | 0.0339 |

Per-call comparison:

| B | calls | ms/call |
|---|---|---|
| 1 | 132 | 59.3 |
| 8 | 17  | 446  |

B=8 takes 7.5× the time of B=1 for 8× the work — nearly linear.
Translation: the Whisper-large-v3 encoder is **compute-bound, not
memory-bandwidth-bound**, at least on RTX 3090. At B=1 we're already
saturating GPU compute; weight reads from HBM are a small fraction
of per-call time so amortising them across 8 items barely changes the
numbers.

Back-of-envelope confirms: weights ≈ 3 GB, HBM bandwidth ≈ 900 GB/s,
so weight-load = ~3.3 ms of a 59 ms call (~5 %). The rest is pure
compute (32 transformer layers × 1500 × 1280² matmuls × 2). Batching
cannot reduce compute.

**Decision**: revert the batched-encoder code. It's ~100 lines that buy
<2 % speedup and introduce FP drift in the output (0.9959 vs 1.0000
sequence similarity — same audio produces slightly different tokens on
edge cases, from the batched matmul's different accumulation order).
Not worth it.

**What this means for the speedup runway:** we've hit a real wall on
this model + hardware. The remaining levers for significant speedup
are architectural, not algorithmic:

- Int8 / q4 quantization (user previously rejected for quality reasons,
  but worth revisiting *specifically for the encoder* — decoder loop
  is already fast, and the encoder is the only remaining target).
- CUDA Graph capture — would require re-exporting the decoder with an
  explicit attention-mask input to remove dynamic KV shapes. Big
  engineering cost; the step loop is now only 20 % of time, so even
  a 10× speedup there only saves ~2 s.
- Shorter-than-30-s encoder input — requires re-export with dynamic
  T_audio axis AND would only help on VAD-sparse files (where most
  segments are ≪ 30 s). Non-trivial.

Current ASR pipeline at RTF 0.034 is already 2.7× faster than Phase 3a
(sequential Transcribe baseline at RTF 0.072) and competitive with the
other backends. Likely the sensible stopping point for optimisation;
ship Phase 4 (Avalonia UI wiring) next rather than chasing diminishing
returns.
