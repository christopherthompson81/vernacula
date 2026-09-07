# VibeVoice-ASR-Streaming investigation

Issue #138 asks for `microsoft/VibeVoice-ASR-Streaming-7B`. Vernacula already ships the
non-streaming `microsoft/VibeVoice-ASR-HF` (see `VIBEVOICE-ASR_EXPORT_PROGRESS.md` and
`src/Vernacula.Base/VibeVoiceAsr.cs`). This log follows the usual porting ladder: run it in
Python as upstream distributes it, export to ONNX, prove parity, then performance, then the C#
CLI, then the desktop app.

## Run 0 — 2026-09-07 10:20 — architecture reading, no code run yet

**Question.** What is this model, how does upstream intend it to be run, and how much of the
existing VibeVoice-ASR port carries over?

**Sources.** HF card + `config.json` + `preprocessor_config.json` + `added_tokens.json` of the
streaming repo; upstream `microsoft/VibeVoice` at `1541f59` (2026-09-03), files
`vibevoice/modular/modeling_vibevoice_asr.py`, `docs/vibevoice-asr-streaming.md`,
`demo/vibevoice_asr_streaming_inference_from_file.py`; arXiv 2609.02812 abstract.

**Raw findings.**

- Not a transformers-native class. `config.json` names `VibeVoiceForASRStreamingTraining`; the
  inference class is upstream's `VibeVoiceASRForConditionalGeneration` from the GitHub repo
  (`pip install -e .`, `transformers>=4.51.3,<5`). Model is not gated. 8 safetensors shards,
  16.16 GiB total.
- Same skeleton as the non-streaming model: causal acoustic tokenizer (vae_dim 64) + causal
  semantic tokenizer (vae_dim 128), one connector each, summed into a Qwen2 7B decoder
  (28 layers, hidden 3584, 28 heads / 4 KV heads, vocab 152064, rope_theta 1e6). Audio at
  24 kHz, 3200 samples per frame (7.5 frames/s, 133.3 ms/frame). A `diffusion_head_config`
  is present but unused for ASR.
- Weight groups: `model.acoustic_tokenizer` (552 tensors, includes the decoder half we do not
  need), `model.semantic_tokenizer` (276), `model.language_model` (338), two connectors (5
  each), `lm_head.weight`.
- Streaming is a *prompt format*, not a different network. From `preprocessor_config.json`:
  `chunk_frames: 22`, `lookahead_frames: 4`, so each step consumes 22 frames of new audio
  (2.933 s) plus 4 frames of lookahead (0.533 s) = 26 frames (3.467 s). One new token,
  `<|text_chunk_end|>` = 151665, sits in the first free slot after `<|file_sep|>`; the
  speech markers reuse `<|vision_start|>`/`<|vision_end|>` (151652/151653) as before.
- `streaming_generate` (the reference algorithm, `modeling_vibevoice_asr.py:429`):
  1. Prefill the text prompt `"You are a helpful assistant that transcribes audio input into
     text output. Please transcribe the following audios streamingly with these keys:
     speaker, content\n"` (hotwords append `and extra info: ...`). No chat template.
  2. Default `encode_mode="split_then_encode"`: cut the waveform into windows
     `[k*chunk, k*chunk + chunk + lookahead)`, zero-pad the last one to full length, and run
     the *whole* audio encoder on each window independently (the tokenizer streaming cache is
     not used here; each 3.47 s window starts cold). The alternative `encode_then_split`
     encodes the whole file once and slices features; upstream's demo uses the default.
  3. Per window: feed `[<speech_start>, 26 frames, <speech_end>]` through the LM with the
     running KV cache, greedy-decode until `<|text_chunk_end|>` or EOS (cap 256 tokens per
     chunk), then feed `<|text_chunk_end|>` itself so the cache always ends on it. Yield the
     decoded chunk text.
  So the KV cache grows monotonically over the whole file: prompt + N × (28 + text tokens).
  No cache eviction anywhere. A 60-minute file is ~1230 chunks ≈ 34k audio positions plus
  text; fits the 131072 position budget but see VRAM note below.
- The acoustic path is stochastic by design: `encode_speech` calls
  `sample(dist_type="gaussian")`, which scales `fix_std` (0.5) by 0.8 and adds
  `randn * randn` noise to the latent means. Upstream's demo does not seed. The existing
  non-streaming export used `--deterministic-audio` (mean only); the same decision is needed
  here and must be justified by measuring transcript variance across seeds against mean-only.
- Output format is speaker-attributed plain text per chunk. There are no timestamps in the
  prompt keys (`speaker, content` only); time comes from the chunk index × 2.933 s. The
  non-streaming port derived word times differently, so the C# result mapping will not carry
  over unchanged.
- Upstream verified environments: NVIDIA PyTorch containers 24.07–25.12; `sdpa` attention is
  the demo default, flash-attn optional. BF16 on CUDA, fp32 on CPU.
- Reusable from the existing port: the acoustic+semantic encoder export (same tokenizer
  configs, same connectors), the Qwen2 decoder export with KV cache, the C# tokenizer and
  KV-cache IO-binding work in `VibeVoiceAsr.cs`. New: chunk windowing, the
  `<|text_chunk_end|>` loop, incremental cache growth across chunks (the non-streaming path
  prefilled once).

**Implications for the next step.**

- Run upstream's own demo unchanged first (`demo/vibevoice_asr_streaming_inference_from_file.py`)
  on upstream's own demo audio (`demo/asr_demo/demo1-chat.mp3`, a two-speaker chat, and
  `demo3-hotwords.wav`) so the reference output and RTF on this RTX 3090 are on record before
  anything is changed.
- VRAM: 7B in BF16 ≈ 14.5 GiB of weights plus tokenizers; the 3090 has 24 GiB. Expect the
  demo to fit for short files; measure peak memory against file length since the cache never
  shrinks.
- Machine setup: `.venv-vibevoice-streaming` (Python 3.12, torch cu128, upstream `-e`),
  checkpoint at `/mnt/data/models/hf/VibeVoice-ASR-Streaming-7B`.

**Addendum (10:40).** Two more facts worth having before the first run:

- A `microsoft/VibeVoice-ASR-Streaming-1.5B` sibling exists (same recipe and the same
  22+4 frame chunking; Qwen2.5-1.5B decoder: hidden 1536, 28 layers, 12 heads / 2 KV heads,
  vocab 151936; 5.24 GiB of safetensors). The paper says both were released together. For the
  desktop app that is the far more plausible default (VRAM, download size, CPU/DirectML
  feasibility), so it is being downloaded alongside the 7B and every measurement below is
  taken on both.
- The existing export script targets the *transformers-native* `VibeVoice-ASR-HF` layout
  (`acoustic_tokenizer_encoder`, `multi_modal_projector`, `audio_token_id`). The streaming
  checkpoints use upstream's own layout (`model.acoustic_tokenizer`, `model.acoustic_connector`,
  `<|vision_start|>` markers, no audio placeholder tokens). Loading them through the
  transformers class will not work without key remapping; the export wrappers must be
  written against upstream's modules. The graph shapes are the same, so the ONNX package
  format the C# side consumes can still match.

## Run 1 — 2026-09-07 11:30 — upstream demo, unchanged, both checkpoints

**Command.** `demo/vibevoice_asr_streaming_inference_from_file.py --model_path <ckpt>
--audio_files demo1-chat.mp3 demo3-hotwords.wav` from upstream `1541f59`, in
`.venv-vibevoice-streaming` (torch 2.11.0+cu128, transformers 4.57.6, sdpa attention, BF16),
RTX 3090. Audio is upstream's own demo material: a 69 s two-person interview clip and a 17 s
Mandarin/English clip naming the product.

**Question.** Does the distributed code run as-is on this machine, and what does its output
look like?

**Raw result.**

| checkpoint | 69 s clip gen time | RTF | 17 s clip gen time | RTF | wall incl. load |
|---|---|---|---|---|---|
| 1.5B | 11.69 s | 0.169 | 1.98 s | 0.115 | 21 s |
| 7B | 12.16 s | 0.176 | 2.00 s | 0.116 | 21 s |

- Both run cleanly, exit 0. Only warnings: the tokenizer-class mismatch notice
  (`Qwen2Tokenizer` file loaded through `VibeVoiceASRTextTokenizerFast`, expected and
  harmless since the streaming token resolves) and the `torch_dtype` deprecation.
- 24 chunks for the 69 s clip, 6 for the 17 s clip, i.e. one per 2.933 s as designed. Text
  boundaries fall mid-sentence and mid-word-group; each chunk's text is a continuation, and
  speaker turns appear as `\n Speaker N:` prefixes inside the chunk text.
- **7B separates the two speakers** (Speaker 0 / Speaker 1 alternating plausibly through the
  interview). **1.5B labels everything Speaker 0** on this clip. Content-wise the two are
  close; 7B is cleaner on punctuation and fixes a few word errors 1.5B makes.
- On the product-name clip neither model gets the name without hotwords: 1.5B writes
  "Y-voice", 7B "Why Voice". 1.5B also writes "dilation" for "diarization". This is the clip
  upstream ships to show `--context_info`, so Run 2 repeats it with hotwords.
- The 7B is no slower than the 1.5B here. Generation is ~25 tok/s either way, so the loop is
  bound by per-token Python/launch overhead, not by weights. Real per-model cost will only
  show up once the decode loop is tight (ONNX/C#), or on GPU-bound long contexts.
- Nothing about the run was seeded, so these transcripts include the acoustic sampling
  noise.

**Implications.** Upstream's code is a valid reference. Next: seed the runs to measure how
much the Gaussian latent sampling moves the transcript (decides whether mean-only export is
acceptable, as it was for the non-streaming port), capture peak VRAM per checkpoint, and
repeat the product-name clip with `--context_info`.

## Run 2 — 2026-09-07 11:45 — seed variance, peak VRAM, hotwords

**Command.** `scripts/vibevoice_streaming_export/run_reference.py` (upstream
`streaming_generate` unchanged, wrapped with `torch.manual_seed`, timing and
`max_memory_allocated`), three seeds per checkpoint on the 69 s interview clip, then the
product-name clip with `--context_info "VibeVoice,diarization"`. Compared with
`compare_runs.py` (word error rate against seed 0, count of byte-identical chunks).

**Question.** How much does the Gaussian acoustic-latent sampling move the transcript, how
much VRAM does each checkpoint really take, and do hotwords work as advertised?

**Raw result.**

| checkpoint | seed | WER vs seed 0 | identical chunks | speaker turns | distinct speakers | RTF | peak VRAM |
|---|---|---|---|---|---|---|---|
| 1.5B | 0 | 0 | 24/24 | 19 | 1 | 0.174 | 4.89 GiB |
| 1.5B | 1 | 0.020 | 21/24 | 20 | 1 | 0.164 | 4.89 GiB |
| 1.5B | 2 | 0.007 | 22/24 | 19 | 1 | 0.163 | 4.89 GiB |
| 7B | 0 | 0 | 24/24 | 18 | 2 | 0.181 | 16.32 GiB |
| 7B | 1 | 0.027 | 11/24 | 17 | 2 | 0.172 | 16.32 GiB |
| 7B | 2 | 0.017 | 19/24 | 18 | 2 | 0.181 | 16.32 GiB |

- Weights resident: 1.5B 4.81 GiB, 7B 16.20 GiB (BF16; the 16.2 includes the unused
  acoustic decoder half and the diffusion head). Peak over a 69 s file is only ~0.1 GiB above
  weights, so the KV cache is negligible at this length. Load from NVMe: 0.9 s / 2.0 s.
- Seed-to-seed drift is 1 to 3 % WER and, on the 7B, more than half the chunks differ
  byte-for-byte between seed 0 and seed 1 (mostly punctuation and where a word lands
  relative to the chunk boundary). Speaker count is stable across seeds. So any parity
  comparison against upstream must be seeded, and a mean-only (deterministic) export needs
  to be shown to sit inside this envelope rather than outside it.
- Hotwords work on both: with `VibeVoice,diarization` in the prompt both models write the
  product name correctly throughout the Mandarin/English clip and 1.5B stops writing
  "dilation". 7B's rendering is the cleaner one (1.5B still garbles the model name once as
  "OSASR"). Cost: +0.3 to +0.5 s on a 17 s clip, i.e. a few extra prompt tokens.

**Implications.** Run 3 replaces the sampling with the latent mean (the choice the
non-streaming export made) and checks (a) it is bit-repeatable, (b) its WER against the
seeded runs is within the seed-to-seed spread, and (c) whether upstream's non-default
`encode_then_split` (one cached whole-file encoder pass, then slice features) produces the
same text as the default per-window cold encoding, since that decides whether the exported
audio encoder needs the streaming conv cache at all.

## Run 3 — 2026-09-07 12:00 — mean-only encoding vs sampling; per-window vs whole-file encode

**Command.** `run_reference.py --deterministic` (monkeypatches
`VibeVoiceTokenizerEncoderOutput.sample` to return the mean; nothing else changes) twice per
checkpoint, plus once with `--encode_mode encode_then_split`. All on the 69 s interview clip.

**Question.** (a) Is mean-only bit-repeatable? (b) Is its transcript inside the seed-to-seed
envelope from Run 2? (c) Does upstream's non-default whole-file encode give the same text as
the default per-window cold encode?

**Raw result** (WER measured against deterministic run A):

| checkpoint | run | WER | identical chunks | speakers | peak VRAM |
|---|---|---|---|---|---|
| 1.5B | deterministic B | 0.000 | 24/24 | 1 | 4.89 GiB |
| 1.5B | encode_then_split | 0.133 | 0/23 | 1 | 7.95 GiB |
| 1.5B | seeds 0/1/2 | 0.007 / 0.013 / 0.000 | 22 / 21 / 24 | 1 | 4.89 GiB |
| 7B | deterministic B | 0.000 | 24/24 | 2 | 16.32 GiB |
| 7B | encode_then_split | 0.070 | 0/23 | 2 | 19.35 GiB |
| 7B | seeds 0/1/2 | 0.040 / 0.033 / 0.023 | 11 / 11 / 12 | 2 | 16.32 GiB |

- (a) Yes: two deterministic runs are byte-identical on every chunk, both checkpoints.
- (b) Yes for 1.5B (0 to 1.3 %, and seed 2 happens to be identical to the mean). For 7B the
  mean-only transcript sits 2.3 to 4.0 % from the seeded ones, versus 1.7 to 2.7 % between
  seeds; slightly wider than the seed spread but the same kind of differences (punctuation,
  chunk-boundary word placement), and speaker count is unchanged. **Decision: export
  mean-only, as the non-streaming port did**, with parity measured against a deterministic
  Python reference rather than a seeded one.
- (c) No. `encode_then_split` produces 23 chunks instead of 24 (it sizes chunks from a
  measured tokens-per-second rather than the sample count), no chunk matches, WER 7 to 13 %,
  and it needs ~3 GiB more VRAM because the entire file is encoded at once. Upstream's demo
  and FastAPI server both use the default, so **the reference behaviour is per-window cold
  encoding**: each 26-frame window (3.467 s, zero-padded at the end of the file) goes through
  the acoustic and semantic encoders independently with no carried conv state. That is the
  simpler export: a fixed-shape audio encoder on 83,200 samples, no streaming conv cache,
  identical to how the model is run in production.

**Implications.** Export design is now fixed on the audio side: one static-shape encoder
graph (1×83200 → 26×hidden) reused per window. Run 4 measures cache growth and stability
over 10 and 30 minute files before the decoder export is designed, since the LM cache never
shrinks.

## Run 4 — 2026-09-07 12:30 — cache growth and stability over 10 and 30 minutes

**Command.** `run_reference.py --deterministic` on 10 min and 30 min files made by looping
the 69 s interview clip (`ffmpeg -stream_loop`, mono 24 kHz). Same content repeated, so this
measures memory, speed and degeneration, not accuracy.

**Question.** The LM cache never shrinks. Does a long file fit, does per-chunk latency grow,
and does the output stay sane (no empty chunks, no runaway speaker labels, no collapse)?

**Raw result.**

| checkpoint | length | chunks | peak VRAM | cache growth | RTF | s/chunk first 10 | s/chunk last 10 | speakers | empty chunks | words |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B | 10 min | 205 | 5.20 GiB | +0.39 | 0.158 | 0.48 | 0.43 | 1 | 0 | 2414 |
| 1.5B | 30 min | 614 | 5.96 GiB | +1.15 | 0.164 | 0.46 | 0.41 | 1 | 0 | 7193 |
| 7B | 10 min | 205 | 16.97 GiB | +0.77 | 0.200 | 0.51 | 0.56 | 2 | 0 | 2462 |
| 7B | 30 min | 614 | 18.47 GiB | +2.26 | 0.221 | 0.54 | 0.72 | 2 | 0 | 7342 |

- No OOM, no exceptions, no empty chunks, word count scales linearly with length (the
  10-minute file is 8.7 loops and yields 8.7× the words), and the 7B keeps exactly two
  speaker labels for the full 30 minutes rather than inventing new ones.
- Cache growth is roughly linear: 7B ≈ 0.75 GiB per 10 minutes, 1.5B ≈ 0.38 GiB per
  10 minutes. Extrapolated, a 60-minute file peaks near 20.7 GiB on the 7B, inside the
  3090's 24 GiB but with the same headroom caveat the non-streaming port hit; the 1.5B
  would sit near 7 GiB. ONNX Runtime will have its own workspace on top of this.
- Per-chunk latency on the 7B creeps from 0.54 s to 0.72 s over 30 minutes (attention over
  ~23k cached positions); the 1.5B does not move. Both stay far under the 2.93 s chunk
  period, so real-time streaming has a wide margin even in this Python loop.
- Overall RTF in Python is 0.16 to 0.22; the loop is still overhead-bound (see Run 1).

**Implications.** The distributed model is understood and runs as intended on this machine;
this closes the "use it as upstream intends" step. Export design that follows from Runs 0-4:

1. `audio_encoder.onnx`: fixed shape, 1×83,200 samples → 26×hidden, mean-only, no conv
   cache, run once per window (exactly upstream's default path).
2. Decoder with a growing KV cache across chunks, the same graph handling the 28-token
   audio block prefill, the per-token decode, and the trailing `<|text_chunk_end|>` feed.
   The existing single-graph decoder export with IO-bound cache from the non-streaming
   port is the starting point; the difference is that prefill happens many times.
3. Cache must be sized for the longest job the app allows, or evicted by policy. Upstream
   never evicts; measuring transcript quality with a sliding window is a separate
   investigation, not part of parity.
4. Start with the 1.5B for export and parity work (fast iteration, 5 GiB), then apply the
   same recipe to the 7B.
