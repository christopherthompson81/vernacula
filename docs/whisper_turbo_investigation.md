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
