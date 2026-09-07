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

## Run 5 — 2026-09-07 13:30 — first ONNX export (1.5B) and ORT parity

**Command.** `scripts/vibevoice_streaming_export/export_vibevoice_streaming_to_onnx.py
--model_path <1.5B> --output-dir /mnt/data/models/vibevoice_streaming_export/1.5B_bf16_f32kv`
(BF16 decoder, float32 KV cache, float32 audio encoder, legacy exporter, opset 18), then
`test_streaming_parity.py` with ORT 1.29 CUDA at `extended` optimisation on the 69 s clip,
compared against the deterministic Python reference from Run 3.

**Question.** Does the export reproduce upstream's per-window loop, and how close is ORT to the
PyTorch reference?

**Export notes.**

- Wrapper check before tracing: with a BF16 cache the wrapper's logits are *identical* to
  upstream's forward on both the prompt prefill and the first audio window (max diff 0.0000,
  same argmax). With the float32 KV patch logits move (max diff 4.9 on a 150k-vocab logit
  vector) but the argmax is unchanged; this is the same precision path the non-streaming
  package ships. The float32 audio encoder sits 0.56 % relative from upstream's BF16 towers.
- The audio encoder graph is 2.77 GB (float32 conv towers, both encoders) and takes a dynamic
  `num_samples` axis; `decoder_single.onnx` is 3.09 GB. Package total ~5.9 GB for the 1.5B.
- The harness's memory watchdog killed the first full export during external-data
  aggregation of the decoder (host had 111 GiB available; the watchdog looks at free, not
  available). Re-running decoder-only in the foreground completed in a few minutes.
- Verified after export that `past_key_0` feeds a `Concat` in every layer: transformers
  4.57's `DynamicLayer.update` skips the concat when the cache is empty, and the dummy cache
  used for tracing is empty, so this was worth checking rather than assuming.
- `export-report.json` carries the streaming constants (window 83,200 samples, hop 70,400,
  26 frames), the prompt token ids (plain and hotword head/tail variants), and the four
  special ids, so the C# side needs no BPE encoder for the default prompt.

**Parity result (1.5B, ORT CUDA, extended opt, 69 s clip, KV round-tripped through host
memory each step):**

| | WER vs torch deterministic | identical chunks | speaker turns | tokens | final KV length | RTF |
|---|---|---|---|---|---|---|
| ORT 1.5B | 0.007 | 22/24 | 19 (same) | 432 | 1159 | 0.149 |

The two differing chunks are one interjection rendered "Oh my" instead of "Oma" and one
dropped comma. Both are inside the 0 to 1.3 % envelope the seeds produce (Run 3), and well
below the 7 to 13 % that the wrong encode mode produced, so the loop and the graphs are
doing what upstream does. ORT already beats the PyTorch loop (0.149 vs 0.162) despite
copying 56 KV tensors to and from the host every token; IO binding is the obvious perf step.

**Implications.** The 1.5B package is parity-clean. Apply the same export to the 7B, then
compare its parity (the 7B's seed envelope is wider, 1.7 to 2.7 %). After that: measure the
package with IO-bound KV (the perf step), then the C# port against these same records.

**7B parity (same setup):** WER 0.040 against the deterministic torch reference, 11/24
identical chunks, 2 speakers, 428 tokens. That is numerically the same distance as one
seeded torch run sits from the deterministic one (seed 0: 0.040, 11/24), so ORT is inside
the model's own noise. The chunk diffs are commas, `?`/`!` choices, and which side of a
boundary a word lands; in one place the torch reference emits a spurious two-word speaker
turn that ORT does not. Host-KV RTF for the 7B is 0.296 (56 float32 tensors copied both
ways per token, ~20 MB each way by the end of the clip); IO binding is measured next.
Both 7B graphs were checked for the past-cache `Concat` as with the 1.5B.

## Run 6 — 2026-09-07 12:20 — ORT with IO-bound KV cache (the C# execution model)

**Command.** `bench_streaming_iobound.py` (KV `present_*` outputs bound to the CUDA device and
fed straight back as `past_*`; only the last logits row returns to the host), ORT 1.29 CUDA,
extended optimisation, 69 s clip and the 10-minute looped file.

**Question.** What does the package actually cost once the cache stops round-tripping, and is
it still parity-clean?

**Two ORT-Python pitfalls found on the way** (both irrelevant to C#, both recorded so the
next script does not rediscover them): `IoBinding.clear_binding_inputs()` /
`clear_binding_outputs()` segfault in the 1.29 wrapper, so create a fresh binding per step;
and a zero-length tensor created directly on the CUDA device also segfaults, so the empty
initial cache lives on the host and every later cache tensor is a device-resident output.

**Raw result, 1.5B:**

| audio | gen time | RTF | tokens | tok/s | final KV length | GPU used at end |
|---|---|---|---|---|---|---|
| 69 s clip | 6.0 s | 0.087 | 432 | 71.7 | 1159 | 8.66 GiB |
| 10 min | 75.1 s | 0.125 | 3745 | 49.9 | 9721 | 10.91 GiB |

- 1.9× faster than the same graphs with host-side KV (Run 5) and 1.9× faster than the
  PyTorch loop (Run 1/3) on the short clip.
- Throughput falls from 72 to 50 tok/s between 1.2k and 9.7k cached positions. With a
  dynamic cache every step `Concat`s the full past into a new tensor (56 of them), so the
  per-token cost is linear in cache length and the total is quadratic. The non-streaming
  port answered this with `decoder_single_static.onnx` (pre-allocated buffers, in-place
  scatter); the same export is the next perf step once the 7B numbers are in.

**Raw result, 7B (69 s clip):** 12.5 s gen, RTF 0.181, 431 tokens at 34.6 tok/s, final KV
1158, 20.72 GiB in use at the end (ORT arena on top of 15.2 GB of weights plus the
float32 audio encoder). Same speed as the PyTorch loop on this clip; the 7B is
weight-bandwidth-bound per token, so binding buys less than it does on the 1.5B. The
10-minute 7B run is the VRAM test: the cache alone is small, but the arena grows with every
`Concat`-produced tensor.

**7B, 10 minutes: out of memory.** ORT's BFC arena failed a 27 MB allocation inside layer
27's attention at some point in the file (nvidia-smi read 20.7 GiB after the 69 s clip).
The KV cache itself is small (~1 GiB at this length in float32); what fills the card is the
arena's fragmentation under the dynamic-cache pattern, where every step allocates 56 new
`Concat` outputs that are each a little larger than the last. The non-streaming port hit the
same wall and answered it with `decoder_single_static.onnx` (pre-allocated `[1,kv,max,128]`
buffers plus a `kv_pos` scatter), so the 7B needs that export before it is usable past a few
minutes on a 24 GiB card. The 1.5B reached 10.9 GiB at 10 minutes and is fine.

**Run-to-run repeatability (1.5B, IO-bound, identical inputs):** three runs of the 69 s clip
give WER 0.007 to 0.043 against each other (15 to 22 of 24 chunks identical), and the
IO-bound run differs from the host-KV run of the same graphs by 0.017. Traced to the audio
encoder: on CUDA two consecutive runs of the same 83,200-sample window differ by ~1.7e-4
relative (max-abs 8.5e-5 of scale), on every setting tried (`cudnn_conv_algo_search`
HEURISTIC/DEFAULT, `use_tf32=0`); on the CPU EP the output is bit-identical run to run, and
CPU vs CUDA differ by 2.6e-4. There is no random op in the graph. This is cuDNN
accumulation-order noise in the float32 conv towers; it is 20× below BF16 resolution, but
the decoder consumes the frames in BF16, so a handful of rounding flips reach the greedy
argmax and move a comma or a boundary word. The decoder itself is deterministic within a
process for identical inputs. Consequence: ORT parity is a tolerance statement (WER against
the deterministic torch reference inside the seed envelope, speaker count unchanged), never
a byte-equality one, and upstream's own inference is stochastic anyway.

**Quantization (user's question, 12:40).** The decoder is a stock Qwen2 causal LM
(Qwen2.5-1.5B / -7B), the architecture the GGUF ecosystem quantizes as a matter of course;
weight-only INT4/INT8 on its linear layers (ORT `MatMulNBits`) is the obvious lever for the
7B on a 24 GiB card and for CPU/DirectML feasibility, while the KV cache, attention and the
conv tokenizers stay float. That is the perf step after the static-KV export, measured
with the same parity harness: WER against the torch reference must stay inside the seed
envelope and the speaker count must not change.

## Run 7 — 2026-09-07 12:45 — C# backend and CLI

**Code.** `src/Vernacula.Base/VibeVoiceStreamingAsr.cs` (new): reads the `streaming` and
`tokenizer` sections of `export-report.json`, encodes each 83,200-sample window with
`audio_encoder.onnx`, and drives `decoder_single.onnx` with the same IO-bound step the
non-streaming backend uses (KV cache device-resident, only the last logits row read back).
`ToSegments` folds the inline `Speaker N:` markers into `VibeVoiceSegment`s; the model gives
no timestamps, so a boundary inside a chunk is placed proportionally to its character
offset in that chunk's text (approximate to within one 2.93 s chunk, never decreasing).
`Vernacula.CLI` gains `--asr vibevoice-streaming`, `--vibevoice-streaming-model <dir>`, and a
`--hotwords` flag that is parsed but refused for now: the package has no BPE *encoder* on
the C# side, only the byte-level decoder, so hotwords need one before they can be wired.

**Command.** `Vernacula.CLI --audio demo1-chat.wav --asr vibevoice-streaming
--vibevoice-streaming-model <1.5B package> --export-format json` (the CLI reads WAV only on
Linux; NAudio's MP3 path is Media Foundation).

**Raw result (1.5B, 69 s clip).**

| | WER vs torch deterministic | speaker turns | wall incl. session load |
|---|---|---|---|
| CLI | 0.011 | 19 | 15.2 s |
| Python IO-bound (Run 6) | 0.011 | 19 | 6.0 s gen + load |

- Same distance from the reference as the Python ORT loop, and the same turn count, so the
  C# loop reproduces the Python one (the two differ from each other only by the CUDA
  encoder jitter described under Run 6).
- First comparison read 0.18 WER: the Python records keep the `Speaker N:` markers as text
  while the CLI turns them into fields, 38 words on this clip. `compare_runs.py` now strips
  markers before scoring and accepts the CLI's JSON directly.
- Wall time includes two session loads and the whole-file decode; the decode itself is the
  Python figure, ~6 s.

**State of the port after Runs 0-7.** Python reference → ONNX package → ORT parity →
IO-bound perf → C# backend → CLI are all in place for the 1.5B. Open items, in order:

1. `decoder_single_static.onnx` for both checkpoints (pre-allocated KV, `kv_pos` scatter):
   removes the per-token `Concat` and the arena growth that OOMs the 7B at 10 minutes and
   slows the 1.5B from 72 to 50 tok/s over the same span. The non-streaming export script
   already has the wrapper.
2. Weight-only INT4/INT8 on the Qwen2 decoder, measured with the same harness.
3. Desktop app: a `vibevoice/vibevoice-asr-streaming` model entry, download manifest,
   settings row, and `TranscriptionService` branch; chunk callbacks map onto the existing
   progress reporting naturally.
4. Hotwords need a BPE encoder in C# (Qwen2 byte-level BPE; `tokenizer.json` ships the
   merges).
5. The 1.5B labels every speaker `Speaker 0` on the interview clip while the 7B separates
   two; the app's default should be decided on more than one clip.

## Run 8 — 2026-09-07 12:55 — static KV cache: a memory fix, not a speed fix

**Command.** `--static-kv-max-tokens 16384` added to the streaming export (the non-streaming
port's `StaticKVCache` wrapper, adapted to upstream's module layout), then
`bench_streaming_iobound.py --static` on the 1.5B.

**Question.** Does replacing the per-token `Concat` with a `ScatterElements` into
pre-allocated buffers fix both the 7B's OOM and the throughput decay?

**A bug found first.** The static run initially produced nonsense (WER 3.5, 2042 tokens for a
432-token clip). The wrapper was exact in PyTorch step-for-step against the dynamic one, and
the graph consumed `kv_pos` in every layer, so the fault was in my benchmark: it computed
`seq_len` with `len()` on an empty id array of shape `(1, 0)`, which is 1, not 0. `kv_pos`
therefore advanced one position too far on every decode step and the mask drifted off the
cache. Dynamic mode never reads `seq_len`, which is why only static broke. Fixed by using
`.size`.

**Raw result (1.5B, correct):**

| decoder | 69 s RTF | 69 s tok/s | 10 min RTF | 10 min tok/s | 10 min WER | GPU at end |
|---|---|---|---|---|---|---|
| dynamic | 0.087 | 71.7 | 0.125 | 49.9 | 0.023 | 10.91 GiB |
| static (16384) | 0.262 | 24.6 | 0.251 | 24.9 | 0.023 | 12.66 GiB |

Parity is unchanged (0.023 at 10 minutes either way), but static is **2× slower**, and
flat: 24.6 tok/s at a 1.2k cache and 24.9 tok/s at a 9.7k cache. That is the point. Static
attention costs the full `max_tokens` buffer on every step regardless of how much is
filled, so its per-token cost is constant in `max_tokens` while dynamic's is proportional
to the *current* length. Dynamic wins whenever the average cache length over a job is below
the buffer size, which for a 16384 buffer means every job shorter than about an hour.

**Conclusion: static KV is the answer to the 7B's OOM, not to throughput.** It should be
exported with `max_tokens` sized to the longest job the app allows, and preferred only
where the dynamic arena cannot fit. The 1.5B should keep the dynamic decoder.

## Run 9 — 2026-09-07 13:00 — float16 decoder and weight-only quantization

**Question (user's).** The decoder is a stock Qwen2 causal LM; is it resilient to
quantization?

**Getting there.** ORT's `MatMulNBitsQuantizer` skipped 254 of 255 MatMuls with "doesn't
have const weight". Two reasons, both fixed in `quantize_decoder.py`: the exported weights
are BF16 (the quantizer reads float32/float16), so a float16 decoder export was needed; and
`torch.onnx` keeps `nn.Linear` weights as `[out, in]` initializers behind a `Transpose`, so
the MatMul's B input is a node output rather than an initializer. ORT's own constant folding
leaves those alone (the tensors are external data), so the script now transposes the
initializers itself and rewires the MatMuls: 197 folded.

**Raw result (1.5B, ORT CUDA, IO-bound, WER vs the deterministic torch reference):**

| decoder | weights on disk | 69 s WER | 10 min WER | 69 s tok/s | 10 min tok/s | 10 min RTF | GPU at end |
|---|---|---|---|---|---|---|---|
| BF16 (shipped) | 2.9 GB | 0.011 | 0.023 | 71.7 | 49.9 | 0.125 | 10.91 GiB |
| float16 | 2.9 GB | 0.008 | 0.020 | 67.5 | 46.8 | 0.132 | 10.80 GiB |
| **INT8 weight-only** | **1.9 GB** | **0.008** | **0.021** | **79.9** | **55.3** | **0.112** | **9.80 GiB** |
| INT4 (block 128) | 1.2 GB | 0.068 | 0.070 | 80.8 | 58.7 | 0.112 | 8.80 GiB |
| INT4, LM head kept | 1.5 GB | 0.057 | 0.074 | 78.7 | 57.1 | 0.112 | 9.04 GiB |

- **INT8 is free.** Parity is indistinguishable from float16 (0.008 / 0.021 vs 0.008 /
  0.020), it is 11 to 18 % faster than BF16, and it saves 1 GB of weights and 1.1 GB of
  VRAM. This is the configuration to ship.
- **INT4 round-to-nearest is not.** 0.057 to 0.074 WER is 3× the ORT baseline and outside
  the model's own seed envelope (0 to 1.3 % for this checkpoint); the speaker-turn count
  drifts too (179 vs 152 over 10 minutes). Excluding the vocab projection barely helps, so
  the loss is spread across the layers rather than concentrated in the head. A calibrated
  method (GPTQ/AWQ) would be the thing to try before writing INT4 off, but plain RTN at
  block 128 is not usable.
- float16 and BF16 are equivalent in both parity and speed, so the float16 export is a fine
  base for quantization without a separate accuracy argument.

## Run 10 — 2026-09-07 13:07 — the 7B: INT8 is what makes it usable

**Command.** 7B static-KV decoder (`max_tokens` 12288, exported on CPU after the CUDA export
OOMed), 7B float16 decoder, and 7B INT8 weight-only, each on the 69 s clip and the 10-minute
file.

**Raw result (WER vs the deterministic torch reference):**

| 7B decoder | weights | 69 s WER | 69 s tok/s | 10 min | 10 min WER | GPU at end |
|---|---|---|---|---|---|---|
| BF16 dynamic (Run 6) | 15.2 GB | 0.023 | 34.6 | **OOM** | — | 20.72 GiB @ 69 s |
| static KV (12288) | 15.2 GB | — | — | **OOM** | — | — |
| float16 dynamic | 15.0 GB | 0.012 | 33.6 | **OOM** | — | 21.87 GiB @ 69 s |
| **INT8 weight-only** | **7.8 GB** | **0.012** | **45.3** | **completes** | **0.016** | **18.90 GiB** |

- **Only the INT8 build finishes a 10-minute file on a 24 GiB card.** It runs at RTF 0.250
  with 188 speaker turns against the reference's 190, WER 0.016, which is *better* than the
  BF16 build's 0.023 on the short clip and comfortably inside the 7B's own seed envelope
  (0.017 to 0.027).
- Static KV did **not** rescue the 7B. Pre-allocating 12288-position buffers costs about
  2.8 GB for the 56 in and 56 out tensors, and every step's attention scores span the whole
  buffer, so it OOMs sooner than the dynamic cache rather than later. Static remains
  useful only where the buffer can be sized close to the actual job length; for the 7B on
  this card, quantization is the lever that matters.
- The 7B INT8 is 31 % faster than BF16 on the short clip (45.3 vs 34.6 tok/s) and halves the
  weights.

**Recommended shipping configuration after Runs 8-10.**

| | decoder | why |
|---|---|---|
| 1.5B | INT8 weight-only, dynamic KV | parity identical to float16, 11-18 % faster, 9.8 GiB peak at 10 min |
| 7B | INT8 weight-only, dynamic KV | the only build that completes long files on 24 GiB; parity better than BF16 |

Neither needs the static-KV graph; it stays in the export script for cases where a buffer
can be sized to the job (and it is the only way to bound the arena on a smaller card).
Remaining perf idea, not yet tried: calibrated INT4 (GPTQ/AWQ) to see whether the 0.07 WER
of round-to-nearest INT4 is recoverable, which would matter for CPU and 8 GiB cards.
