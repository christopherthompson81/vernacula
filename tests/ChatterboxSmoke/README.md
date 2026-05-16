# ChatterboxSmoke

C# port of `scripts/chatterbox_export/.../listen_test.py`. Loads the four
ONNX graphs produced by `export_chatterbox_to_onnx.py` and runs them
end-to-end to produce a WAV. Proves the ORT-C# orchestration matches the
PyTorch reference within numerical drift before we factor the pipeline
into `Chatterbox.Base` / `Chatterbox.CLI` / `Chatterbox.Avalonia`.

## What it does

1. Loads ORT sessions via `OrtSessionBuilder.CreateCachedSession` (CUDA EP
   with CPU fallback; pre-optimization cache next to each `.onnx`).
   Auto-detects cond decoder layout — uses the 3-graph **split path**
   (`flow_encoder.onnx`, `cfm_estimator.onnx`, `mel2wav.onnx`) if those
   files are present, otherwise falls back to the monolithic
   `conditional_decoder.onnx`.
2. Reads a voice prompt WAV, resamples to 24 kHz mono via NAudio's
   `WdlResamplingSampleProvider`, pads/crops to 312,936 samples (the
   trace-time canonical input length).
3. Runs `speech_encoder` to get `audio_tokens`, `speaker_embeddings`,
   `speaker_features`, and the conditioning `cond_emb` for the LM.
4. Runs `embed_tokens` with the LM input token sequence. With `--text`,
   the C# `EnTokenizer` (BPE port of upstream `chatterbox.models.tokenizers
   .EnTokenizer`) encodes the sentence and wraps it in the
   `[EXAGGERATION, START, ...bpe..., STOP, START_SPEECH, START_SPEECH]`
   layout the LM expects. Without `--text`, falls back to the hardcoded
   "Ezreal and Jinx teamed up..." sentence for back-compat.
5. Runs the Llama LM autoregressively with a growing KV-cache for up to
   256 steps, applying upstream's repetition-penalty (1.2 if logit > 0
   else no-op). Stops on `STOP_SPEECH_TOKEN = 6562`.
6. Concatenates `audio_tokens` (from the voice clone) with the generated
   speech tokens, then runs the cond decoder. In **split mode**:
   `flow_encoder.onnx` → CFM solve loop in C# (10 cosine-scheduled
   Euler steps, each one call of `cfm_estimator.onnx` on a CFG-doubled
   batch, then a CFG-combine + Euler step) → trim mel prompt prefix →
   `mel2wav.onnx`. In monolithic mode: single `conditional_decoder.onnx`
   call.
7. Writes the result as a 24 kHz mono float32 WAV via `NAudio.WaveFileWriter`.

## Performance notes

Tested on RTX 3090, ORT 1.24.4, warm pre-optimization caches:

| Path | Total | Session load | LM (174 steps) | CFM (10 steps) | mel2wav |
|---|---|---|---|---|---|
| Monolithic (1 cond decoder onnx, basic Run)   | ~180s | ~175s | 5.6s | (inside dec) | (inside dec) |
| Split (3 cond decoder graphs, basic Run)      | 10.0s | 3.2s  | 5.4s | 0.5s | 0.23s |
| Split + `--io-binding` (default)              | **6.1s** | **3.1s** | **1.5s** | 0.5s | 0.23s |

The 50× session-load speedup comes from `cfm_estimator.onnx` containing
ONE Euler-step forward (3K nodes) instead of the 10× unrolled estimator
(70K nodes in the monolithic). Same math; the loop is just in C# now.

The 3.5× LM speedup (`--io-binding`) comes from chaining the LM's KV-cache
outputs as the next step's inputs via `OrtValue` references instead of
copying GPU→CPU→GPU each step. Pattern adapted from
`WhisperTurbo.cs::TranscribeBatch`. Toggle off with `--no-io-binding` if
you want to A/B (both paths produce bit-identical token sequences;
verified per-step logits match to max_abs=0).

## Usage

```bash
dotnet run --project tests/ChatterboxSmoke -- \
    --onnx-dir /path/to/cb_dyn5 \
    --voice ~/Downloads/voice_prompt.wav \
    --text "Hello world. This is fresh text." \
    --out /tmp/chatterbox_out_cs.wav \
    --ep cuda
```

Flags:
- `--onnx-dir` — directory containing the `.onnx` files.
- `--voice` — reference WAV at any sample rate / channel count.
- `--text "..."` — text to synthesize. If omitted, uses a hardcoded
  Ezreal-and-Jinx sentence (back-compat with pre-tokenizer behavior).
- `--tokenizer-json <path>` — chatterbox `tokenizer.json`. If omitted,
  auto-locates from the HuggingFace hub cache at
  `~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/`.
- `--out` — output WAV path (default `/tmp/chatterbox_out_cs.wav`).
- `--ep cpu | cuda` — execution provider. `cuda` requires
  `Microsoft.ML.OnnxRuntime.Gpu`; falls back to CPU if CUDA unavailable.
- `--io-binding | --no-io-binding` — toggle the LM KV-cache IoBinding
  optimization. Default on.
- `--diag <dir>` — dump LM step-0/step-1 + token sequence to `<dir>` for
  the deferred LM-divergence investigation.

## What this proves (and doesn't)

**Proves:**
- All four ONNX graphs load and run via ORT-C# with the expected I/O
  contract.
- The KV-cache step loop in C# produces a valid speech-token stream
  (verified by listening — the output should be acoustically close to
  the Python reference).
- The full pipeline runs end-to-end without any Python in the loop.

**Doesn't prove:**
- Text tokenization. We hardcode the same `InputIds` array the Python
  listen test uses. A real CLI needs an in-process text→tokens pipeline.
- Streaming. The current implementation generates the whole utterance
  before writing audio; chunked synthesis and gapless concatenation are
  Stage 1 steps 8 and beyond.
- Performance. The implementation uses `Run()` with `NamedOnnxValue` for
  readability — every LM step copies KV tensors to/from host memory via
  `.ToArray()`. The Python reference uses CUDA tensor chaining; the
  Python listen test takes ~3 minutes per LM run on a 3090 because of
  the unbatched step loop, and this C# version will be in the same
  ballpark or slower. `IoBinding` + `OrtValue` chaining (see
  `WhisperTurbo.cs::TranscribeBatch`) is the optimization path.
- Forced alignment. The eventual app needs word-level timestamps for
  highlighting; that's a separate pass via the ASR aligner.

## Cross-implementation parity: voice-prompt resampling

C# uses NAudio's `WdlResamplingSampleProvider` to convert audio to 24 kHz;
Python's `listen_test.py` uses `librosa.load(..., sr=24000)` which goes
through librosa's kaiser_best resampler. The two resamplers produce
*different* 24 kHz samples from the same 16 kHz input, which yields
different speaker embeddings → different LM token sequences (without
changing perceived audio quality).

For reproducibility / cross-implementation regression testing, pre-resample
the voice prompt to 24 kHz once (any standard tool — librosa, ffmpeg, sox)
and feed the 24 kHz file to both pipelines. They'll then agree to within
ORT cross-version drift (1 token for the Ezreal sentence in our testing;
your inputs may vary). See issue #53 for the full investigation.

## Next steps (in `chatterbox.scratch.md` order)

1. Text tokenizer port (so we can pass `--text "any sentence"`).
2. Factor the orchestration into `Chatterbox.Base` as `SpeakerEmbedder` +
   `AcousticLM` + `Vocoder` classes.
3. `Chatterbox.CLI` consumes the Base library; `--text-file` for
   markdown input; chunked synthesis.
4. `IoBinding` perf pass on the LM step loop.
