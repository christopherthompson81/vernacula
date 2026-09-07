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

The two differing chunks differ by one interjection spelled two ways and one dropped comma. Both are inside the 0 to 1.3 % envelope the seeds produce (Run 3), and well
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

## Run 11 — 2026-09-07 13:30 — KV cache design: is there a bounded-VRAM cache with no speed penalty?

**Question (user's).** Is there a KV cache type that gives a dependable VRAM ceiling without
the slowdown static buffers cost?

**First: the non-streaming port already answered half of this, and I rediscovered it the
expensive way.** `VIBEVOICE_ASR_3090_ANALYSIS.md` states plainly that static KV
"was built and benchmarked but is net-negative for the C# GPU IO binding path: the
fixed-size 6144-position attention window costs more in attention MACs than the Concat ops
it eliminates." Run 8 here measured the same thing independently at 16384. **Read the
sibling investigation before repeating an experiment it already ran.**

**What that port also found, which reframes the question:**

- Decode time splits **44 % node dispatch / 56 % ORT framework overhead** (~1950 kernel
  launches per step). Crucially: *"Because output shapes grow each step (KV cache), CUDA
  graphs cannot be used; this overhead is structural to ORT autoregressive decode with
  dynamic shapes."*
- Of the 44 % that is real ops: `Concat` (the cache growth) is 14 %, `Cast` 9 % (BF16↔F32 at
  every K/V store, a direct cost of the float32 KV cache), `MatMul` 20 %.
- `ORT_ENABLE_ALL` was rejected because ORT's **fused attention kernel does the softmax in
  BF16**, skipping the float32 upcast Qwen2's eager path uses, which moved the first
  divergence earlier. `ORT_ENABLE_EXTENDED` is pinned for correctness.

**The mechanism that fits: `com.microsoft.GroupQueryAttention` with a shared cache buffer.**
Verified present in ORT 1.29 on this machine, with inputs `past_key`, `past_value`,
`seqlens_k`, `total_sequence_length`, and `T_CACHE` accepting float32, float16, bfloat16 and
**int8/uint8/float8** (with `k_scale`/`v_scale`). Built a synthetic node at the 1.5B's
geometry (12 heads / 2 KV heads / 128 dim) and ran it:

| cache dtype | CUDA | CPU | buffer shape preserved |
|---|---|---|---|
| fp16 act / fp16 cache | ok | ok | yes |
| fp32 act / fp32 cache | ok | ok | yes |
| fp16 act / fp32 cache | ok | ok | yes |

Then the decisive measurement — one decode token, buffer fixed at 16384, varying how much of
it is valid:

| valid positions | % of buffer | µs / step |
|---|---|---|
| 128 | 0.8 % | 24.1 |
| 1024 | 6.2 % | 76.4 |
| 4096 | 25.0 % | 88.2 |
| 8192 | 50.0 % | 128.1 |
| 15000 | 91.6 % | 175.0 |

**Cost tracks the valid length, not the buffer size.** That is the property our static export
lacks (it was flat at ~O(max_tokens), hence 2× slower). So GQA gives, in one op: a buffer
allocated once at a known ceiling, an in-place cache update (no `Concat`, removing 14 %), no
BF16↔F32 cast at the K/V boundary if the cache is fp16 (removing much of the 9 %), and
attention proportional to what is actually filled.

**And the bigger prize, from the sibling log:** fixed input *and output* shapes are exactly
the precondition CUDA graphs need. The 56 % framework overhead was declared structural
*because the cache grows*; a shared buffer removes that reason. Capturing the decode step as
a CUDA graph is the only lever measured here that attacks the largest single cost.

**The pitfall this must clear, also from the sibling log.** GQA *is* a fused attention
kernel, the same class that caused the `ORT_ENABLE_ALL` parity regression. That regression
was specifically a BF16 softmax replacing a float32 upcast. Our best build is now INT8
weights with float16 activations, and flash-style kernels normally accumulate softmax in
float32, so it may well not reproduce — but it is not safe to assume. Validation is the
harness we already have: WER against the deterministic torch reference must stay inside the
seed envelope and the speaker count must not move.

**Assessment.** Yes, there is a design that is bounded and fast, and it is GQA with a shared
buffer, not a preallocated buffer behind ordinary attention. It is not free: it needs the
per-layer attention subgraph replaced by a GQA node. ORT's own fusion tooling has no Qwen2
GQA path (`MODEL_TYPES['qwen3']` maps to the GPT-2 handler; `fusion_rotary_attention` never
emits GQA), so the realistic routes are (a) emit GQA directly from the export wrapper by
replacing `Qwen2Attention.forward` with a custom symbolic op, or (b) `onnxruntime-genai`'s
model builder, which produces Qwen2 with GQA plus INT4/INT8 natively — but it is
`input_ids`-based and our decoder is fed `inputs_embeds` (audio frames interleaved with
text), so it would need adapting. Route (a) keeps our existing package contract.

Ordering, if this is pursued: GQA + fp16 shared-buffer cache first (parity check against the
harness), then CUDA graph capture on the now-fixed shapes, then optionally int8 KV via
`k_scale`/`v_scale`. Expected: bounded VRAM at a chosen ceiling, per-token cost at or below
today's dynamic cache, and the first real attack on the 56 % dispatch overhead. Not started;
this run is analysis only.

**Addendum — the old fusion pitfall does not apply, measured.** Rather than let the
`ORT_ENABLE_ALL` history rule GQA out, all the attention paths were compared against a
float64 reference on the same inputs (1.5B geometry, 1024-position cache, one decode token):

| attention path | relative error vs float64 |
|---|---|
| eager, bf16 cache, float32 softmax — *the build we shipped* | 2.46e-03 |
| eager, bf16 cache, **bf16 softmax** — *the ORT_ENABLE_ALL regression* | 2.88e-03 |
| eager, fp16 cache, float32 softmax — *the fp16/INT8 build* | 3.04e-04 |
| eager, fp16 cache, fp16 softmax | 3.83e-04 |
| **GQA, fp16 cache** | **4.04e-04** |
| GQA, fp32 cache | 1.23e-06 |

The 2026-04 regression was a **BF16** problem, not a **fusion** problem: dropping the float32
softmax upcast cost only 17 % more error, but on a BF16 baseline already at 2.5e-3, which is
coarse enough that 0.0-margin logit ties flip. GQA with an fp16 cache lands at 4.0e-4 —
**six times more accurate than the BF16 build that shipped**, and level with the eager fp16
path we now run. The precondition that made fused attention dangerous is gone the moment the
model left BF16.

**One trap found in the process:** the fp32 cache is the most accurate option by three
orders of magnitude but falls off the flash-attention kernel and is **60× slower**
(4833 µs/step vs 77 µs at a 1024 cache). GQA must be run with an fp16 cache. That is also
the cheaper one: 0.44 GiB for all 28 layers at a 16384 ceiling, against 0.88 GiB for fp32,
and it removes the BF16↔F32 casts that cost 9 % of node time in the sibling port's profile.

**Revised assessment: GQA with an fp16 shared buffer is the configuration to build**, and the
historical objection to fused attention should not block it. Expected outcome, all of it
measured here in isolation and still to be confirmed end to end: a hard VRAM ceiling chosen
at export time, per-token cost proportional to the filled length rather than the buffer,
better numerics than either shipped build, no `Concat` and no K/V casts, and fixed shapes
that unblock CUDA graph capture against the 56 % dispatch overhead.

## Run 12 — 2026-09-07 14:00 — GQA shared-buffer decoder, built and measured (1.5B)

**Code.** `scripts/vibevoice_streaming_export/export_gqa.py` exports `decoder_gqa.onnx`:
`Qwen2Attention.forward` is replaced by one that projects Q/K/V, applies RoPE exactly as the
eager path does, and then calls a single `com.microsoft::GroupQueryAttention` node per layer
against a pre-allocated fp16 buffer. `bench_streaming_gqa.py` drives it, binding
`past_key_i` and `present_key_i` to the **same** device tensor so the cache is updated in
place.

**Four things that had to be got right, all of them non-obvious:**

1. `torch.autograd.Function` with a `symbolic` staticmethod no longer traces on torch 2.11
   (`RuntimeError: unordered_map::at`). Use `torch.library.custom_op` plus
   `torch.onnx.register_custom_op_symbolic`.
2. A registered custom op may not return a tensor that aliases an input, so the placeholder
   returns `past_k.clone()`. The real aliasing (present *is* past) happens at bind time.
3. transformers 4.57 builds its causal mask with `torch.vmap`, which the TorchScript
   exporter cannot trace. Passing `attention_mask={"full_attention": <dummy>}` skips
   `create_causal_mask` entirely; GQA masks from `seqlens_k` anyway.
4. The head counts arrive at the symbolic as traced `Constant` values, so they need
   `symbolic_helper._parse_arg(v, "i")` before becoming `*_i` attributes.

**And one bug that cost the most time, worth remembering:** `IoBinding.get_outputs()` returns
values in **binding order, not model order**. Binding `logits` after the 56 cache tensors and
then reading `get_outputs()[0]` silently yields `present_key_0`, so the argmax was taken over
a cache tensor and the model "generated" 1536 tokens of garbage for a 17 s clip. The graph
was correct the whole time: an A/B against the dynamic decoder matched step for step
(argmax identical, logit correlation 0.99999). Bind `logits` first.

**Raw result (1.5B, WER vs the deterministic torch reference):**

| build | 69 s RTF | 10 min RTF | 69 s tok/s | 10 min tok/s | decay | 10 min WER | GPU 69 s | GPU 10 min |
|---|---|---|---|---|---|---|---|---|
| dynamic BF16 (Run 6) | 0.087 | 0.125 | 71.7 | 49.9 | −30 % | 0.023 | 8.66 | 10.91 |
| INT8 dynamic (Run 9) | 0.078 | 0.112 | 79.9 | 55.3 | −31 % | 0.021 | 7.80 | 9.80 |
| **GQA fp16** | 0.084 | **0.086** | 74.8 | 71.7 | **−4 %** | 0.020 | 9.16 | **9.17** |
| **GQA + INT8** | **0.065** | **0.066** | 96.3 | 94.6 | **−2 %** | 0.024 | 8.25 | **8.25** |

**Both properties hold at once.** GPU usage is *identical* at 69 seconds and 10 minutes
(8.25 GiB for the INT8 build) because the cache is one allocation of 0.44 GiB made up front;
and throughput decays 2 % over the same span where the dynamic cache loses 30 %. Parity is
unchanged: 0.024 at 10 minutes against the reference's own seed spread of 0 to 1.3 %, and the
speaker-turn count is 155 against 152.

Against the build that started this session, GQA + INT8 is **1.9× faster end to end** (RTF
0.125 → 0.066) on 2.7 GiB less VRAM, with the memory now flat in recording length rather than
growing. The buffer ceiling (16384 positions here) is what bounds the job length, and it is
chosen at export time.

## Run 13 — 2026-09-07 13:41 — GQA on the 7B: the checkpoint that could not finish, finishes

**Raw result (7B, WER vs the deterministic torch reference):**

| 7B build | 69 s RTF | 10 min | 10 min RTF | 10 min tok/s | 10 min WER | GPU 69 s / 10 min |
|---|---|---|---|---|---|---|
| BF16 dynamic (Run 6) | 0.181 | **OOM** | — | — | — | 20.72 / — |
| float16 dynamic (Run 10) | 0.187 | **OOM** | — | — | — | 21.87 / — |
| static KV 12288 (Run 10) | — | **OOM** | — | — | — | — |
| INT8 dynamic (Run 10) | 0.139 | completes | 0.250 | 26.7 | 0.016 | 14.90 / 18.90 |
| GQA fp16 | 0.174 | **completes** | 0.195 | 34.3 | 0.014 | 22.83 / 22.72 |
| **GQA + INT8** | **0.124** | **completes** | **0.142** | 47.2 | 0.016 | 15.16 / 15.68 |

- GQA alone rescues the 7B: the unquantized fp16 build now completes a 10-minute file at
  22.7 GiB, where the same weights with a dynamic cache died in layer 24. Memory is flat
  (22.83 → 22.72 GiB) but the margin on a 24 GiB card is thin.
- **GQA + INT8 is the configuration to ship for the 7B:** 15.7 GiB, RTF 0.142, 47.2 tok/s.
  That is 1.76× faster than INT8 with the dynamic cache (0.250) and leaves 8 GiB of headroom.
  Parity is 0.016 with 188 speaker turns against the reference's 190 — indistinguishable
  from the dynamic INT8 build and inside the 7B's seed envelope.

**The failure mode changed, which matters as much as the speed.** The 30-minute file stopped
with `KV buffer full: 16401 > 16384` — a clean, predictable refusal naming the ceiling,
raised before any work was wasted. Compare the old behaviour: `BFCArena ... Failed to
allocate memory for requested buffer of size 42663936` from inside layer 25, after minutes of
compute, with no way to know in advance whether a given file would fit. A bounded cache turns
an unpredictable crash into a documented limit the caller can check up front:
`ceiling ÷ (tokens per chunk) × 2.93 s` of audio, or simply refuse the job.

Run 14 re-exports both checkpoints at a 32768 ceiling (buffer 1.76 GiB on the 7B, 0.88 on the
1.5B) to cover 30-minute files.

## Run 14 — 2026-09-07 13:52 — 32768 ceiling: 30-minute files on both checkpoints

**Command.** Both checkpoints re-exported with `--max-tokens 32768` (buffer 1.75 GiB on the
7B, 0.88 on the 1.5B), quantized to INT8, run on the 30-minute file.

| build | audio | RTF | tok/s | KV used | GPU | WER vs torch | speaker turns |
|---|---|---|---|---|---|---|---|
| 7B GQA+INT8 | 30 min | 0.177 | 37.7 | 29821/32768 | 15.69 GiB | 0.020 | 585 (ref 588) |
| 1.5B GQA+INT8 | 30 min | 0.078 | 78.0 | 28742/32768 | 8.67 GiB | 0.025 | 439 (ref 436) |

- **The 7B transcribes a 30-minute recording for the first time**, at 15.7 GiB with 8 GiB to
  spare, faster than the reference PyTorch loop (0.177 vs 0.221) and with parity inside its
  seed envelope. Every earlier build OOMed at 10 minutes.
- The 1.5B does the same file at RTF 0.078 in 8.67 GiB — 2.1× faster than the PyTorch loop.
- Cache use at 30 minutes is ~29k positions, so the 32768 ceiling is worth about 33 minutes.
  A ceiling maps to duration as `ceiling ÷ ~16.0 positions per second of audio`.
- One thing to watch: the 7B run emitted **three** distinct speaker labels where the
  reference emitted two, on audio that contains two speakers (the file loops one interview).
  WER is unaffected (0.020) and the turn count matches (585 vs 588), so this is one stray
  label rather than a systematic split, but speaker-label stability over long files deserves
  its own check before the app trusts the labels for anything but display.

**Upstream comparison (checked at the user's suggestion).** `microsoft/VibeVoice`'s own vLLM
streaming server does not stream unboundedly either. It enforces two ceilings in code —
`max_model_len` (default **16384**, the same number chosen here independently) and
`max_audio_windows` (default 512, ~25 minutes) — and on either it raises "session outgrew
the ... context ... Raise --max-model-len, or start a new session." The docs say the same:
"`--max-model-len` is what caps how long one session can run; raise it for hour-long
streams." There is no eviction, sliding window or re-priming anywhere in the repo, and the
design notes state that every chunk reuses the KV cache of every chunk before it.

Their bookkeeping is worth copying in one respect: they count the *expanded* length, warning
that each audio placeholder becomes `window_frames + 2` positions and that counting the
unexpanded prefix runs short by ~20 tokens per window. Our check counts real cache positions,
so it is already on the right side of that.

This says upstream chose a bounded session, not that windowing fails — their product is
real-time sessions rather than long-file transcription, so they had no reason to need it.
The sliding-window trial continues on branch `exp/vibevoice-sliding-window`.

## Run 17 — 2026-09-07 15:10 — closing out the performance spike

Three remaining ideas were tested before moving to the C# port. Two are dead ends worth
recording; one is a clean win.

**1. CUDA graph capture — closed, and permanently.** The sibling port attributed 56 % of
decode time to ORT dispatch overhead and called it unreachable because cache shapes grew.
GQA fixed the shapes, so this was the obvious next lever. ORT refuses:
`This session cannot use the graph capture feature ... as all compute graph nodes have not
been partitioned`. Profiling showed why the first time: **1197 of 4392 node dispatches ran on
the CPU EP** (510 `Gather`, 339 `Unsqueeze`, 171 `Concat`, 168 `Div` — all shape arithmetic
from the traced dynamic dimensions). A decode-only export with every axis fixed
(`--decode-only`, `decoder_gqa_step.onnx`) eliminates them completely: **0 CPU nodes, 2673
CUDA nodes**, 39 % fewer dispatches. Capture is *still* refused, because
`total_sequence_length` is a required CPU input of GroupQueryAttention itself, so a
CPU/GPU boundary always remains. CUDA graphs are not available for this model as exported.

**2. The fixed-shape decode graph is not worth shipping.** Having built it, it is only 3 %
faster per token (5.76 vs 5.95 ms) despite deleting 1197 CPU dispatches — those nodes are
tiny and overlap with GPU work. It would double the package size for 3 %. Rejected.

**But measuring it produced the most useful number of the day.** Raw per-token latency on
the graph is **5.95 ms = 168 tok/s**, while the end-to-end Python bench reports 94.6 tok/s on
the same model. **44 % of measured time is Python harness overhead** — a fresh `io_binding`
per step, numpy↔OrtValue conversions, the logits copy. That overhead does not exist in the
C# backend, which reuses one binding and reads logits from a bound buffer. So the C# port is
itself the largest remaining performance item, worth up to ~1.8× on the decode loop, and the
Python RTF figures in this document are a floor rather than a ceiling.

**3. float16 audio encoder — ship it.** The encoder was exported with float32 conv towers,
inherited from the sibling port's workaround for ORT having no bf16 Conv. fp16 Conv is well
supported in ORT 1.29:

| encoder | size | per 3.47 s window | 10-min encoder total | 10-min WER |
|---|---|---|---|---|
| float32 | 2.77 GB | 30.49 ms | 6.0 s | 0.024 |
| **float16** | **1.39 GB** | **19.73 ms** | **4.0 s** | **0.021** |

Half the size, 35 % faster, and parity is unchanged (0.021 vs 0.024 — if anything nearer the
reference). Numerically it sits 9.4e-4 from the float32 encoder, which is below the fp16
resolution the decoder consumes frames at anyway.

**Final 1.5B configuration**: fp16 audio encoder + GQA INT8 decoder with a shared fp16 cache.
10-minute file at **RTF 0.060 in 7.35 GiB**, against RTF 0.125 in 10.91 GiB for the first
working ONNX build this session — **2.1× faster on 3.6 GiB less**, with memory now flat in
recording length. Package drops from 5.9 GB to 3.3 GB.

**Perf spike closed.** Remaining ideas, none blocking: calibrated INT4 (GPTQ/AWQ) for CPU and
small cards, batching the audio encoder across windows in file mode (the windows are all known
up front, so they need not be encoded one at a time), and the re-priming work parked on
`exp/vibevoice-sliding-window`.

## Run 18 — 2026-09-07 15:40 — C# backend on the GQA package

`VibeVoiceStreamingAsr` now targets the GQA package only (`decoder_gqa.onnx`; a package
without `"attention": "GroupQueryAttention"` is rejected with a message naming the export
script). Changes: one device tensor per layer per side allocated at the ceiling and bound as
both `past_key_i` and `present_key_i`; the two int32 length inputs; float16 throughout
(encoder input, audio embeddings, logits); a `MaxAudioSeconds` property derived from the
ceiling; and an up-front length check that refuses an over-long recording before any
compute, matching upstream's behaviour.

**Two bugs, both about where memory lives.**

1. The first working version ran the 69 s clip in **53.4 s** — 9× slower than Python.
   `OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, ...)` allocates in
   **host** memory, so the whole 0.44 GiB cache was copied to the device and back on every
   one of the 481 steps. Allocating from `new OrtAllocator(_decoder, cudaMemInfo)` instead
   took it to 9.9 s. The IO binding was correct all along; the buffers it pointed at were not.
2. `logits` must be bound **before** the cache tensors, because `GetOutputValues()` returns
   binding order rather than model order — the same trap hit in Python (Run 12) and already
   documented in `CohereTranscribe.cs`.

**Where the time actually goes** (10-minute file, 4118 decoder steps, temporary
instrumentation since removed):

| stage | time |
|---|---|
| decoder `RunWithBinding` | 28.85 s |
| audio encoder | 4.12 s |
| argmax over 152k logits | 1.54 s |
| binding and input construction | 0.32 s |
| session load + tokenizer.json parse (startup) | ~5 s |

**Result:** the C# decode loop runs at **142.7 steps/s against Python's 128.3**, an 11 %
gain, confirming the direction predicted in Run 17 though not its magnitude (the 168 tok/s
figure there was a synthetic single-step measurement and does not survive real workload
variance). End-to-end the CLI is RTF **0.066** on the 10-minute file versus Python's 0.060,
the difference being ~5 s of fixed startup that a longer job amortises.

**Parity is exact:** WER **0.021** on the 10-minute file and **0.008** on the 69 s clip
against the deterministic torch reference — identical to the Python runner on both, with
matching speaker-turn counts (156 and 19).

One deliberate change: log-probabilities are now opt-in (`computeLogprobs`, default off).
The second pass needs a `Math.Exp` per vocabulary entry, 152k per token; measured at ~7 % of
wall time here, and the CLI does not use the values.

## Run 19 — 2026-09-07 16:10 — desktop app integration

`AsrBackend.VibeVoiceStreaming` added and wired through every place the enum is enumerated.
The app's dispatch-coverage suite is what makes that tractable: each unmapped site throws by
design rather than silently inheriting Parakeet's behaviour, so adding the enum member and
running the tests lists the work.

Wired: the model-name mapping (`microsoft/vibevoice-asr-streaming`), the 10-language set from
the model card (two fewer than the non-streaming sibling — it drops Thai and Vietnamese), the
display name, the settings radio with a CUDA-gated label matching VibeVoice's, the
missing-model messages in both the home and settings view models, the download manifest and
asset list, and `VocabService` (same Qwen2 byte-level vocabulary, read from the streaming
package's own folder).

`TranscriptionService` reuses the whole-recording VibeVoice path since this checkpoint also
segments and attributes speakers itself; only the backend class and model directory differ.
Chunk callbacks drive progress, and `ToSegments` produces the rows the existing bulk-insert
code already handles. Both checkpoints force `SegmentationMode.VibeVoiceBuiltin` and hide the
standard segmentation choices.

**The suite caught four real failures**, all one bug: `VocabFixtures` writes its tokenizer by
`VocabKind`, and the two VibeVoice backends share a kind but read different folders, so the
streaming service found no vocabulary and decoded to "". The fixture now writes both folders.

**Verification.** Full suite green (27 / 158 / 178). Headless Xvfb run with isolated
`XDG_*` dirs: app launches with no exceptions, and the new row renders under VibeVoice-ASR in
the Transcription tab, correctly greyed with the CUDA reason on a host whose CUDA EP check
fails (screenshot taken; the check fails inside Xvfb, which is itself the right behaviour to
see).

**Not done, and deliberately.** The HF repo the manifest points at
(`christopherthompson81/vibevoice-asr-streaming-onnx`) does not exist yet, so Download
Missing Models will 404 until the package is published. Publishing it is the next step, along
with a manifest of per-file MD5s built the same way as the other model repos.

## Run 20 — 2026-09-07 16:50 — two sizes, selectable; packages assembled for publishing

**Size selection.** `VibeVoiceStreamingSize { Small1_5B, Large7B }` persists in settings and
picks both the install folder (`vibevoice_asr_streaming_1_5b` / `_7b`, side by side so
switching does not force a re-download) and the download repo. The settings pane shows the
pair only when the streaming backend is selected, each labelled with its measured VRAM peak
and the trade-off that matters:

| | VRAM | 10-min RTF | speakers on the test clip |
|---|---|---|---|
| 1.5B | 7.4 GB | 0.060 | labels every turn Speaker 0 |
| 7B | 15.7 GB | 0.142 | separates two speakers, held over 30 minutes |

A warning line appears when the detected card is smaller than the selected checkpoint needs.
Changing size re-runs the model check, since the missing-file set differs.

**A packaging trap.** The float16 audio encoder fits in a single protobuf (1.39 GB) and so
saved *without* a `.data` sidecar, while every other graph has one. The downloader's asset
list is static, so a package missing a file the list names would 404 at install.
`assemble_package.py` therefore re-saves both graphs with external data, giving every package
the same nine files at both sizes.

**Packages built and verified end to end through the C# CLI, not just the Python harness:**

| package | size | WER vs the torch reference | speakers |
|---|---|---|---|
| 1.5B | 3.2 GiB | 0.008 | 1 |
| 7B | 9.0 GiB | 0.012 | 2 |

Manifests of per-file MD5s built with the shared `scripts/make_manifest.py`, and model cards
written to `scripts/hf_readmes/vibevoice-asr-streaming-{1.5b,7b}-onnx/` in the same house
style as the other exports, stating the deterministic-encoding and INT8 changes, the CUDA
requirement, the cache ceiling, and the speaker-attribution difference between sizes.

**Not uploaded.** The two HF repos do not exist yet and publishing ~12 GiB to public repos is
the user's call, so this stops at locally verified packages under
`/mnt/data/models/vibevoice_streaming_publish/{1.5b,7b}`.

## Run 21 — 2026-09-07 17:20 — does the app actually stream? (it did not)

**Question (user's).** Upstream's doc says the model "transcribes while the audio is still
arriving ... a transcript appears as the speaker talks." Does the desktop app do that?

**No, and the first integration was worse than the backend it sat next to.** The decode loop
was genuinely chunk-by-chunk and the progress line updated per chunk, but transcript rows were
built from `ToSegments` *after* the whole recording finished, so nothing appeared in the
transcript until the end. The non-streaming VibeVoice path already streamed rows through
`onSegment` as they completed, so the streaming backend was the less streaming of the two.

The cause was a real difficulty rather than an oversight, which is why it needs a design
rather than a one-line fix: a speaker turn spans chunks and is only final when the *next*
marker arrives, so there is no completed segment to emit at the moment a chunk lands.

**Fix: `VibeVoiceStreamingAsr.SegmentAssembler`.** It folds chunks into turns incrementally
and keeps the newest turn open so it grows as chunks arrive. Callers re-read `Segments` after
each `Add`: entries beyond what they have shown are turns that just opened, and the last
entry's text may have changed. `ToSegments` is now this class run over a finished list, so the
batch and streaming paths cannot drift apart.

The app adds rows for newly opened turns and rewrites the text of the one in progress, so the
transcript fills in as decoding proceeds. The authoritative rows are still bulk-inserted at
the end, so the provisional end time on an open turn is corrected rather than persisted wrong.

The CLI now prints each chunk as it is emitted, matching upstream's own demo:

```
  [   2.9s] Speaker 0:<first speaker's opening line>   Speaker 0:<start of the next>
  [   5.9s] <continues mid-sentence into this chunk>
  [   8.8s] [Applause]   Speaker 0:<and so on>
```

(Chunk text abstracted; the shape is what matters — a line per hop, turns marked inline, and
a sentence that runs across a chunk boundary rather than being cut at it.)

**Verification.** Parity unchanged (WER 0.008 on the 69 s clip, same as before the change).
Five new tests pin the behaviour the promise depends on: turns visible before the recording
ends, the open turn growing rather than duplicating, incremental output equal to batch,
monotonic turn starts, and text preserved when a recording contains no speaker marker at all.
Suite 163 pass.

**Still not streaming in the strict sense.** The app transcribes a file that already exists,
so it streams *decoding*, not *capture*. Live microphone input would need an audio source
feeding the same loop, which the backend is shaped for — it takes one window at a time and
holds its own cache — but nothing upstream of it currently produces audio incrementally.

## Run 22 — 2026-09-07 17:50 — the ceiling was ours, and raising it costs ~1 MB

**Question (user's).** A "streaming" model that caps at 17 minutes seems odd.

**Two things were tangled in the word.** "Streaming" here is about *latency*, not unbounded
length: the contrast upstream draws is with the non-streaming checkpoint, which emits nothing
until the recording ends. That is the ordinary online/offline distinction and says nothing
about running forever. The length bound is a separate axis, and it is not arbitrary — this
model attributes speakers purely from accumulated context, so evicting context cuts the
thread, which Run 15 measured directly.

**But the 17-minute figure was mine, not the model's.** It came from copying upstream's
server default of 16,384. The checkpoints are trained further:

| checkpoint | trained context | audio | KV cache at that ceiling |
|---|---|---|---|
| 1.5B | 65,536 | ~68 min | 1.75 GB |
| 7B | 131,072 | ~137 min | 3.50 GB |

**And raising it does not touch the weights.** The ceiling exists only as the buffer length
declared on the decoder's `past_key/value` inputs; no node carries it as a constant, because
GroupQueryAttention takes the live length through `seqlens_k` and `total_sequence_length`.
`set_cache_ceiling.py` rewrites those 56 dimensions and the export report with
`load_external_data=False`, so the `.onnx.data` file is never read or rewritten — verified
byte-identical by MD5 on both packages.

**Consequence for publishing:** re-publishing the raised ceiling moved **0.9 MB** of graph
plus two small JSON files per repo, instead of re-uploading 12.2 GB of weights. Both repos
updated in seconds.

**Verified after patching:** both packages transcribe the 69 s clip with output identical to
before (WER 0.008 and 0.012 against the torch reference, same speaker counts), and the C#
backend reports the new limits.

**The remaining cost is speed, not memory.** Attention tracks the filled cache, so a token
late in a two-hour file costs several times an early one; VRAM stays flat. Documented in the
help text and both model cards rather than left for a user to discover.

## Run 23 — 2026-09-07 18:10 — selecting Streaming downloaded the wrong model

**Reported.** With the 1.5B streaming backend selected, the download showed
`vibevoice_asr/audio_encoder.onnx.data` and a 17.2 GB total — the *non-streaming* package.

**Cause.** `ActiveRepos()` opened with an early return that pre-dated this backend:

```csharp
if (AsrBackend == VibeVoice || Segmentation == VibeVoiceBuiltin)
    return [ the non-streaming VibeVoice repo ];
```

The streaming backend also forces `VibeVoiceBuiltin` (it segments itself), so that condition
matched first and the switch arm added for it below was never reached. Selecting Streaming
queued a different model, 17.2 GB of it, into `vibevoice_asr/`.

**Fix.** The streaming backend is now resolved *before* that test, with a comment saying why
the order matters. The unreachable switch arm is removed.

**Why nothing caught it.** Every existing check asks whether a mapping *exists* — a models
directory getter, an `IsAsr…` property, a language set — not whether the mapping is the
*right* one. The dispatch suite would have been just as happy with a backend that downloaded
someone else's weights.

`ModelRepoSelectionTests` closes that gap by asserting on the resolved file list: the
streaming backend requires only files under its own package folder and never
`decoder_single.onnx`; the non-streaming backend is unaffected; built-in segmentation *alone*
(a non-VibeVoice ASR backend using VibeVoice for segmentation) still pulls the non-streaming
package; and no backend pulls another's package. Negative-checked by restoring the original
ordering, which fails with `"vibevoice_asr/decoder_single.onnx" ... expected start
"vibevoice_asr_streaming_7b"` — the reported bug exactly — and passes again once fixed.

**One trap inside the test itself:** an early version matched the substring `"streaming"` and
flagged Parakeet, because Sortformer's diarization model is named
`diar_streaming_sortformer_4spk-v2.1.onnx`. It now matches the package folder prefix.

Suite 168 pass.

## Run 24 — 2026-09-07 16:30 — the app segfaulted at the end of a job

**Reported.** The app crashed when a transcription finished. No console output — just KDE's
crash notifier, which means a signal rather than a managed exception.

**Getting the evidence.** `coredumpctl` had the dumps. The first backtrace was a jump to
`0x0000000100000012` — not an address, a corrupted function pointer — with the frames beneath
it inside `libonnxruntime.so`. A later dump from the same crash site showed live `libcuda`
threads, confirming the decoder really was on the GPU, and an ORT worker still initialising
on another thread. Symbolising went nowhere: the shipped runtime exports only its C API, so
the offsets resolve to nothing useful.

That was enough to characterise it: memory the runtime still expected to own, freed
underneath it. Two things in the new backend could do that, and both were mine.

**Cause 1 — the cache buffers' lifetime was handed to the runtime.** Each step called
`binding.GetOutputValues()` to read the logits. That returns wrappers for *every* bound
output, including the 56 cache tensors — the same buffers reused on every later step. Their
release is then the runtime's business, not ours, while the loop still depends on them. The
fix removes the call entirely: the logits go into an `OrtValue` allocated and owned here and
bound by name, so nothing is fetched back and nothing else's lifetime is in question.

**Cause 2 — a null data pointer on every decode step.** Audio embeddings were wrapped with
the `Memory<T>` overload unconditionally, and a decode step passes no audio. A zero-length
`Memory<T>` over an empty array has nothing to pin, so the tensor carried a null pointer into
the runtime. The non-streaming backend has an explicit branch for exactly this case, using
the plain-array overload; that branch was lost when the code was adapted. Restored.

Neither reproduced in the CLI, which is why they survived: the CLI's short-lived process and
lighter allocation pressure never collected the wrappers.

**Confirmed fixed** by the reporter on a rebuilt binary.

**Two other defects found while chasing it**, both real, neither the crash:

- The transcript editor had no branch for `microsoft/vibevoice-asr-streaming`, so a finished
  streaming job resolved *Parakeet's* vocabulary path and Parakeet's model-availability
  check. Degraded silently — no confidence highlighting — rather than failing loudly.
- Text emitted before the model names anyone arrives as speaker −1, and went to the results
  database as `speaker_-1` with a diarization id of 0. Folded onto speaker 0.

**A note on the harness.** An opt-in end-to-end test was written to run the real pipeline in
process, and then deleted: this repository's build guard refuses the GPU runtime in that test
project (it is marked CPU-only), so the test could never exercise the path that crashes. A
test that cannot run is worse than no test. The reproduction that did work was the app itself
under Xvfb with an isolated profile and a job seeded into the control database.


## Run 25 — 2026-09-07 19:00 — the 7B's "third speaker" is the crowd

Run 14 flagged a third speaker label on the 7B's 30-minute run against two-speaker audio.
Resolved: it is not drift and not a defect. Every occurrence is the same thing —

```
Speaker 2:Applause And Cheering
```

— and there are exactly three, at chunks 332, 379 and 426, one per repetition of the looped
source. The model gives the audience its own label, and gives it the *same* label each time
the same audio comes round, which is evidence the attribution is stable rather than wandering.

**Consequence for the app, which is real if minor:** a speaker in the transcript need not be a
person. A segment may arrive attributed to a speaker whose content is a non-speech event. The
editor already lets a speaker be renamed, so this needs no code; it is documented in the help
text so the label is not read as a transcription error.

## Run 26 — 2026-09-07 19:20 — per-word confidence for the streaming backend

The editor colours words by confidence for every other backend. This one produced none: the
decode loop could compute log-probabilities but no caller asked, and the assembler dropped
them anyway when folding chunks into turns.

**The hard part is attribution, not computation.** A speaker turn is a slice of a chunk's
text, so a segment must claim exactly the tokens that produced *its* characters. Tokens are
byte-level, so byte offsets are not character offsets the moment anything is non-ASCII — and
this model handles ten languages including Chinese, Japanese and Korean. `DecodeWithOffsets`
therefore decodes each prefix to get true character offsets, and a chunk carries one offset
per token. The assembler claims a token when any of its characters fall inside the range it is
taking, and the speaker marker's own tokens go to neither side, since a marker is structure
rather than speech.

Confidences stay opt-in. The app asks for them because the editor shows them; the CLI asks
only under `--benchmark`, since it prints text.

**Verified** on the 69 s clip: 19 of 19 segments carry per-word confidence, and parity is
unchanged at WER 0.008. Four tests pin the attribution: tokens follow text across a marker,
they accumulate across chunks within one turn, marker tokens belong to neither side, and a
segment reports *no* confidences rather than a mismatched count when they were not computed.

## Run 27 — 2026-09-07 19:45 — hotwords, and a tokenizer that moved a layer down

Hotwords were parsed but refused: they are user text spliced into the model's prompt, so they
must be tokenized with the model's own vocabulary, and the C# side had only a *decoder*.

**No new tokenizer was written.** `Qwen3Tokenizer` already implements the full byte-level BPE
encoder for exactly this vocabulary — it was just in `Vernacula.Tts.Base`, because when it was
written the ASR backends were all decode-only. Its header said so. That is no longer true, so
the class moved to `Vernacula.Base`, where an ASR backend can use it without dragging in the
TTS stack (the command-line tool references only `Vernacula.Base`, and should stay that way).
Two callers inside the TTS library and one test needed a `using`; nothing else changed, and
its parity tests still pass.

**Effect, on the clip upstream ships to demonstrate the feature:**

| | without hotwords | with `VibeVoice,diarization` |
|---|---|---|
| product name | "Y-voice" | "VibeVoice" |
| technical term | "dilation" | "diarization" |

Seven tokens of prompt. Wired in the CLI as `--hotwords` and in the app as a text box under
the size picker, saved with the other settings. Bad hotwords are logged and ignored rather
than failing the transcription, since losing a job to a typo in an optional field would be a
poor trade.
