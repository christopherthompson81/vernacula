# Granite Speech 4.1 perf investigation log

Running log for the perf-iteration loop on the Granite Speech 4.1 ONNX
bundle. The export and parity work is in
[granite_speech_investigation.md](granite_speech_investigation.md);
this doc is its sibling, focused entirely on runtime cost and
optimization decisions.

Each entry is one run or one discrete experiment, stamped with local
date/time. Negative results stay in the log — the whole point is the
trail of "we tried this, here's what we measured, here's what it
implies for the next step."

Issue reference: [#28](https://github.com/christopherthompson81/vernacula/issues/28).

---

## Run 1 — 2026-05-08 (baseline question)

**Question:** What does the export pipeline actually cost at runtime,
and where is the time going? The Run 4 export reformulation traded the
upstream's O(T·ctx) block attention for O(T²) full attention; the
[granite_speech_investigation.md Run 4](granite_speech_investigation.md)
note flagged this as ~7-15× more attention work at 30-60 s but offered
no wall-clock numbers. We need a baseline before deciding whether the
full-attention encoder is a real bottleneck or just a cosmetic concern.

**Method:** Build `profile_granite_speech.py` after the
[`profile_qwen3_asr_pipeline.py`](../../scripts/qwen3asr_export/profile_qwen3_asr_pipeline.py)
pattern. Time each ORT stage independently and report tokens/sec for
the autoregressive loop. Run on CPU first (deterministic, easy to
reason about), then CUDA (the realistic deployment target on a 3090,
25 GB VRAM).

**Pre-run hypotheses to falsify:**

- **H1**: The encoder's full-attention will dominate at long T (90 s)
  on CPU. The naive expectation: O(T²) at T=4500 → ~20 M scores per
  layer per head → 16 layers × 8 heads × 4500² × 4 bytes = 2.6 GB of
  attention scores per encoder pass.
- **H2**: On GPU, encoder is fast (matmul-bound, GPU strengths) and
  the autoregressive decoder dominates. Each step has 40 layers × KV
  attention against the full-prompt KV cache; at T=900 audio embeds +
  ~25 prompt tokens that's ~1000-token attention per layer per step.
- **H3**: The mel.onnx and projector are negligible (<1% of total).

### Baselines (CPU + GPU, fp32 throughout)

Both runs use the standard fp32 export from
[granite_speech_investigation.md Run 5](granite_speech_investigation.md).
Warm-up = 1 run, timing = 2nd run. Decoder cap 32 tokens (6.4 s clip),
256 tokens (90 s clip). Hardware: NVIDIA RTX 3090 (25 GB VRAM).

#### Stage timings (ms)

| Stage | CPU 6.4 s | GPU 6.4 s | CPU 90 s | GPU 90 s |
|---|---:|---:|---:|---:|
| mel | 10 | 12 | 97 | OOM |
| encoder | 644 | 21 | 21 105 | OOM |
| projector | 33 | 1 | 171 | OOM |
| decoder_init (prefill) | 547 | 37 | 5 131 | OOM at first try; passes with `arena_extend_strategy=kSameAsRequested` |
| decoder_step / token | 223 | 30.5 | 275 | OOM |
| **Total** | 6 144 | 747 | 96 435 | (see below) |
| **RTF** | 1.04× | 8.6× | 0.93× | — |

Tokens decoded: 22 (6.4 s clip), 254 (90 s clip).

#### Findings

- **H1 (full-attention encoder dominates at long T)**: partially confirmed
  on CPU, falsified on GPU at the audio lengths we care about.
  Encoder at T=321 (~6.4 s): 21 ms on GPU, 644 ms on CPU. At T=4500
  (~90 s): 21 s on CPU (factor of ~33× from T=321; cleanly quadratic in
  T). On GPU we couldn't measure the 90 s case directly because the
  decoder weight footprint OOMs first, but extrapolating from the 6.4 s
  GPU number with O(T²) scaling: 21 ms × (4500/321)² ≈ 4 s. Big in
  absolute terms; small compared to the autoregressive decode cost.
- **H2 (AR decoder dominates on GPU)**: confirmed. At 6.4 s, decoder_step
  is 91% of GPU runtime (677 ms / 747 ms). At 30.5 ms/token / 32.8 tok/s
  on a 3090 for a 1.84 B-param LM with ~84-token KV cache, this is
  on the slow side (we'd hope for ~80-100 tok/s on a tuned fp16 path).
- **H3 (mel + projector negligible)**: confirmed. Both <2 ms on GPU.
- **CPU is unusable at long form**: 90 s of audio takes 96 s of wall
  clock (RTF 0.93×). The encoder full-attention is part of it (21 s)
  but the AR decoder at 70 s for 254 tokens is the larger share.
  Whatever ships, it ships on GPU (or DirectML / Metal).
- **GPU OOMs at the 90 s clip with the fp32 bundle.** The 1.84 B-param
  LM appears in BOTH `decoder_init.onnx` and `decoder_step.onnx` —
  two separate copies of the LM weights at 7.0 GB each. Plus the
  encoder at 1.7 GB. Total resident weights: **15.7 GB**, leaving
  only ~9 GB for activations on a 25 GB GPU. The `decoder_init`
  prefill at T_prompt=918 + audio_embeds[1, 900, 2048] eats it. Cohere
  and Qwen3 both ship duplicate decoder weights in this same shape;
  it's a generic problem at this LM size, not Granite-specific.

  > For a 2 B-param model this is *absurd* — fp16 weights for one copy
  > of the LM should be ~3.7 GB. We're sitting at 4× that on GPU.

- **`arena_extend_strategy=kSameAsRequested`** was a partial fix:
  decoder_init no longer OOMs at the prefill step, but decoder_step
  OOMs on its first call (8.28 MB allocation failed). Confirms that
  the binding constraint is *weight footprint*, not arena
  fragmentation.

#### Decisions for Run 2

- **fp16 conversion of the decoder pair only.** Halves each from 7 GB
  → 3.5 GB; total resident weights drop from 15.7 GB to 8.7 GB. Should
  clear 90 s OOM with ~16 GB headroom for activations.
- **Encoder + projector stay fp32.** Granite's encoder is a 16-layer
  Conformer with Shaw relpos attention; per @user "conformer models
  don't tolerate quantisation well" — relpos additive bias scores can
  exceed fp16 dynamic range. We won't touch the encoder until we have
  a parity check that proves fp16 there is safe.
- **Decoder unification (single graph, `use_cache_branch` switch)** is
  the proper fix for the duplicate-weights problem but a much larger
  rewrite. Defer until the fp16 path is shipping.
- **Decoder_step latency is the next optimisation target after fp16.**
  fp16 will already give a ~2× kernel speedup for free; further wins
  likely live in attention fusion + IOBinding for the KV cache.

---

## Run 2 — 2026-05-08 (fp16 decoder pair)

**Question:** Does converting just the decoder pair to fp16 (encoder
stays fp32) unblock 90 s GPU inference, and does it hold transcription
parity?

**Method:** Add
[`scripts/granite_export/convert_decoders_to_fp16.py`](../../scripts/granite_export/convert_decoders_to_fp16.py)
that runs `OnnxModel.convert_float_to_float16(keep_io_types=True)` on
just `decoder_init.onnx` and `decoder_step.onnx`. Encoder and projector
are untouched per @user's note about Conformer fp16 sensitivity. Then
re-run per-stage parity, end-to-end smoke, and the perf profiler.

**Implementation gotcha:** The fp16 decoder LM weights are still ~3.7 GB
which exceeds the protobuf 2 GB single-file limit, so the post-convert
save **must** use `use_external_data_format=True` and consolidate to a
single `<name>.onnx.data` sidecar. Qwen3's example doesn't surface
this because their LM is smaller.

**Findings:**

#### Size

| File | fp32 | fp16 |
|---|---:|---:|
| `decoder_init.onnx` (+ data) | 7 015 MB | 3 686 MB |
| `decoder_step.onnx` (+ data) | 7 016 MB | 3 686 MB |
| Total resident weights on GPU | 15.7 GB | **8.7 GB** |

#### Parity

Per-stage max-abs-diff (vs PyTorch eager fp32 reference):

| Stage | fp32 | fp16 |
|---|---:|---:|
| encoder      | 3.4e-4 | 3.4e-4 (unchanged) |
| projector    | 1.4e-6 | 1.4e-6 (unchanged) |
| decoder_init logits | 3.9e-5 | **4.9e-2** |
| decoder_init KV    | 4.4e-5 | **2.6e-1** |
| decoder_step logits | 9.1e-6 | **2.8e-3** |
| decoder_step KV    | 2.3e-5 | **1.7e-1** |

The decoder logit diffs are ~3 orders looser than fp32 — consistent
with fp16's ~3-decimal-digit precision applied to logits in the
±10..30 range. The real question is argmax stability, validated end-to-end:

| Clip | Tokens generated | text match | divergence |
|---|---:|---|---|
| 6.4 s VCTK | 22 | **exact** | none |
| 90 s en-US conversation | 252 vs 255 | semantically same | 1 token skipped at position 173 |

The 90 s divergence is a single skipped filler-word fragment
("ele-" before "electronic"). Both transcripts say "…he'll be working
on um air some kind of electronic…" vs "…air ele- some kind of
electronic…". This is fp16 argmax noise nudging past a near-tie on a
partial-word filler — semantically lossless. Acceptable for greedy
decoding; could be tightened with beam search if it ever matters.

#### Perf (GPU 3090, fp16 decoders, fp32 encoder)

| Stage | fp32 6.4 s | fp16 6.4 s | Δ | fp32 90 s | fp16 90 s | Δ |
|---|---:|---:|---:|---:|---:|---:|
| mel | 11 | 10 | -10% | OOM | 427 | — |
| encoder | 21 | 21 | 0% | OOM | 897 | — |
| projector | 1.2 | 1.5 | — | OOM | 40 | — |
| decoder_init | 37 | 25 | **-33%** | OOM | 880 | — |
| decoder_step / token | 30.5 ms | **19.2 ms** | **-37%** | OOM | 164 ms | — |
| Total | 747 ms | 483 ms | -35% | OOM | **43.8 s** | — |
| Tokens/sec | 32.8 | 52.1 | **+59%** | — | 6.1 | — |
| RTF | 8.6× | 13.3× | +55% | — | 2.05× | — |

#### ORT op-level breakdown (fp16, decoder_step over 22 steps + 22 warm-up steps)

Top kernel ops, total ms:

| op | calls | total ms |
|---|---:|---:|
| MatMul | 10 604 | 481 |
| Concat | 10 868 | 405 |
| **Cast** | 7 172 | **337** |
| FusedMatMul | 5 324 | 256 |
| Mul | 8 844 | 187 |
| Transpose | 7 084 | 186 |
| Add | 8 800 | 182 |
| Reshape | 12 496 | 155 |

Cast at #3 with 337 ms is fp16↔fp32 boundary churn — most of it from
`keep_io_types=True` casting KV cache I/O back to fp32 every step. Fix:
expose KV cache as fp16 directly so neighbouring step calls don't
round-trip through Cast. (Requires a graph-edit pass that strips the
boundary Casts on the KV inputs/outputs only.)

Concat at #2 with 405 ms over ~10 870 calls is the per-step
`past_kv = concat(past_kv, new_k_or_v)` for 40 layers × 2 K/V × ~22
calls × 2 runs ≈ 3 520 expected; the 3× excess is concats elsewhere in
the graph (likely RoPE rotation construction). The per-step
allocate-and-copy of the KV cache is exactly what `IOBinding` on the
KV tensors avoids — VibeVoice saw 5.5× from this same pattern.

#### What's now well-bounded

- **fp16 decoder pair is parity-acceptable for greedy ASR.** Ship this
  as the default GPU bundle.
- 90 s GPU inference works end-to-end at 2× realtime.
- 6.4 s GPU inference at 13× realtime is comfortably interactive.

#### Next levers (Run 3+)

- **IOBinding the KV cache.** Eliminates the per-step
  numpy↔ORT copies that the profiler doesn't show but are visible as
  Memcpy node warnings (`2 Memcpy nodes are added to the graph`).
  Should also reduce Concat overhead by allowing in-place append
  rather than alloc-and-copy. VibeVoice precedent suggests a large
  speedup at long context, where the cost actually bites.
- **Strip boundary Casts on KV inputs/outputs.** Right now the fp16
  graph casts each KV layer back to fp32 at the boundary, then back
  to fp16 inside the next step's kernel. With `keep_io_types=False`
  on the KV ports specifically, a pure-fp16 KV cache stays in fp16
  across step boundaries. Saves ~337 ms / 22 calls = 15 ms/step on
  6.4 s.
- **Memcpy node warnings.** The current bundle reports 2-3 Memcpy
  nodes per session at load time. Some ops aren't on the GPU and ORT
  shuttles tensors host↔device per call. Worth diagnosing which
  kernels fall back to CPU.
- **Decoder unification (single graph, `use_cache_branch`).** The
  bigger structural win — collapses the duplicate 3.7 GB LM copy
  between init and step into one. Halves resident weights from 8.7 GB
  to ~5.0 GB on GPU. Bigger graph rewrite; defer until IOBinding +
  Cast cleanup are landed.

---

## Run 3 — 2026-05-08 (decoder unification, fp32, **shipping target**)

**Question:** @user objected to fp16's 1-token parity loss at 90 s.
Can we get the memory savings of Run 2 *without* the precision tradeoff
by collapsing `decoder_init.onnx` + `decoder_step.onnx` into a single
graph that handles both modes?

**Method:** A single `decoder.onnx` that takes `past_kv` as a graph
input always, with variable past length. Prefill runs with zero-length
past_kv and `cache_position=[0..S-1]`; step runs with populated
past_kv and `cache_position=[past_len]`. HF's
`GraniteForCausalLM.forward` was probed first to confirm it handles a
zero-length `DynamicCache` correctly — yes, it just runs prefill.

The audio merge runs unconditionally on every call. At step time,
`input_ids` is the next-token id which never matches
`audio_token_id`; the cumsum-gather + `torch.where` pattern collapses
to a no-op (text_embeds wins everywhere). Caller passes a 1-row
dummy `audio_embeds` at step time so the gather index stays in range.

Trace dummy: B=2, S=2 (mid-prompt), past_len=2 (mid-cache). Both seq
and past_len are non-zero so dynamo doesn't specialise either; runtime
accepts past_len=0 (prefill) and S=1 (step) once the graph is dynamic.

### Parity (vs PyTorch fp32 reference)

| Mode | logits | KV |
|---|---:|---:|
| Prefill (unified) | 3.3e-4 | 5.4e-4 |
| Step (unified) | 9.1e-6 | 2.3e-5 |

Step is bit-identical to the split `decoder_step.onnx`. Prefill is ~8×
looser than the split `decoder_init.onnx` (different attention path
through the same graph) but still well within fp32 noise — and **100×
tighter than the fp16 prefill** (4.9e-2). The 1-token "ele-" filler
that fp16 dropped on the 90 s clip is preserved.

End-to-end smoke vs `model.generate()`:

| Clip | text match | tokens (ORT vs ref) |
|---|---|---|
| 6.4 s VCTK | exact | 22 vs 23 (trailing EOS) |
| 90 s en-US | **exact** | 254 vs 255 (trailing EOS) |

### Perf — fp32 unified vs fp16 split (GPU 3090)

| 6.4 s | fp32 split | fp16 split | **fp32 unified** |
|---|---:|---:|---:|
| mel | 11 | 10 | 9 |
| encoder | 21 | 21 | 22 |
| projector | 1.2 | 1.5 | 1.3 |
| decoder_init | 37 | 25 | 30 |
| decoder_step / token | 30.5 ms | 19.2 ms | **19.8 ms** |
| Total | 747 ms | 483 ms | **502 ms** |
| Tok/s | 32.8 | 52.1 | **50.4** |
| RTF | 8.6× | 13.3× | **12.8×** |

| 90 s | fp32 split | fp16 split | **fp32 unified** |
|---|---:|---:|---:|
| mel | OOM | 427 | 178 |
| encoder | OOM | 897 | 619 |
| projector | OOM | 40 | 14 |
| decoder_init | OOM | 880 | 372 |
| decoder_step / token | OOM | 164 ms | **61 ms** |
| Total | OOM | 43.8 s | **16.7 s** |
| Tok/s | OOM | 6.1 | **16.4** |
| RTF | OOM | 2.05× | **5.38×** |

### Findings

- **fp32 unified is competitive with fp16 split at short audio (50.4
  vs 52.1 tok/s) and 2.6× faster at long audio (16.4 vs 6.1 tok/s).**
  Counterintuitive at first but reasonable in retrospect:
  - Single 7 GB session beats two 3.5 GB sessions on cache locality —
    one weight set resident, no inter-session memcpys.
  - fp16's Cast-at-boundary overhead (337 ms across 22 step calls in
    Run 2's profile) cancels half its kernel speedup.
  - `decoder_init` is also a single session here (vs being a separate
    7 GB load in the split case), so no duplicate weight pressure on
    activations during prefill at long audio.
- **No OOM at 90 s.** Resident weights drop from 15.7 GB (fp32 split)
  to 8.8 GB (fp32 unified). Plenty of headroom on a 25 GB 3090.
- **Greedy parity is exact at 90 s.** The "ele-" filler that fp16
  dropped is preserved in the unified fp32 path — confirming that
  parity loss in Run 2 was fp16-induced, not graph-induced.
- **Single 7 GB graph (vs 14 GB split) is the right shipping target.**
  Lower memory, faster inference, no parity loss.

### Decision

**Make the unified decoder the default.** The split init/step pair
remains supported via `--no-unified-decoder` (or by omitting
`--unified-decoder` if we keep that as the trigger), but new
deployments target `decoder.onnx` only. Update the C# smoke and the
production backend to use the unified contract.

### Final shipping numbers (fp32 throughout, 3090)

| Audio | RTF | Tok/s | Total wall |
|---|---:|---:|---:|
| 6.4 s | 12.8× | 50.4 | 0.50 s |
| 90 s | 5.38× | 16.4 | 16.7 s |

### Open follow-ups

- **decoder_step at long context** is still the dominant cost (61
  ms/tok at the 90 s clip vs 20 ms/tok at 6.4 s — KV cache attention
  scales with cache size). IOBinding the KV cache should remove the
  per-step host↔device numpy copy that the Memcpy node warnings
  surface. VibeVoice precedent: ~5× from this same pattern.
- **Memcpy node warnings** persist (2-3 nodes per session). Would
  benefit from a focused diagnostic run.
- **Encoder full-attention scaling** is now visible at 619 ms for
  90 s — but at 4% of total runtime, not yet a priority.

---

## Run 4 — 2026-05-08 (C# tracing, IOBinding attempt)

**Question:** Where does C# overhead live on top of ORT execution, and
can IOBinding for the KV cache eliminate it?

**Method:** Instrument [`tests/GraniteSpeechSmoke/Program.cs`](../../tests/GraniteSpeechSmoke/Program.cs)
with per-stage and per-step inner timings (`ORT.Run()`, output extract,
input build, argmax, bookkeeping). Add `--ep cuda` and run on a 3090.
Then implement IOBinding behind `--io-binding` to keep KV cache
GPU-resident across step calls.

### C# baseline (no IOBinding)

| | 6.4 s GPU | 90 s GPU |
|---|---:|---:|
| Total wall | 527 ms | 22 264 ms |
| RTF | 12.18× | 4.04× |
| decoder_step / token | 21.0 ms | 82.4 ms |
| Tok/s | 47.6 | 12.1 |

#### Per-step breakdown at 90 s (254 steps, growing KV cache)

| | ms | % |
|---|---:|---:|
| ORT.Run() | 14 704 | **70.3%** |
| Output extract (`.ToArray()` ×80 KV/step) | 6 200 | **29.6%** |
| Input build (DenseTensor + NamedOnnxValue list) | 5 | 0.0% |
| Argmax over 100 353 vocab | 8 | 0.0% |
| Bookkeeping | 0 | 0.0% |

The 29.6% on output extract is the C#-specific overhead the Python
profiler doesn't surface. Each step copies ~190 MB (40 layers × 2 K/V ×
[1, 4, ~1100, 128] fp32) from the GPU back to managed heap arrays just
to re-pin them as the next step's inputs. Across the 90 s run that's
~48 GB of host↔device round-trip.

C# is ~25-35% slower than the Python profiler (16.7 s vs 22.3 s on
90 s). Most of the gap is this output-extract overhead — Python's
`.numpy()` apparently lands less aggressively on the GPU↔host copy
path, or amortises better.

Compared to the Python profiler:
| | Python 90 s | C# 90 s |
|---|---:|---:|
| Total | 16.7 s | 22.3 s |
| decoder_step / token | 61.1 ms | 82.4 ms |
| Tok/s | 16.4 | 12.1 |

### IOBinding attempt: implementation works, transcript broken

Wrote an IOBinding decode loop following the VibeVoice and Qwen3 patterns:

- One binding per step (fresh `OrtIoBinding` each iteration).
- `BindOutputToDevice("present_key_<L>", cudaMem)` so the new KV stays
  on GPU.
- `BindInput("past_key_<L>", presentKvOrtValueFromPreviousStep)` —
  no host roundtrip.
- Logits bound to CPU memory for argmax.

Result: **decode_step latency drops to 14.5 ms/token (≈1.45× faster)
but the transcript is garbage from token 2 onward.**

```
ORT: "Hello,null<|fim_suffix|>'s. and.-dron for_<|fim_suffix|> -in trouble--<|fim_suffix|>..."
Ref: "Hello, I'm from Ontario. I hope that you will select my voice ..."
```

#### Diagnostic findings

- **KV cache shapes ARE growing correctly.** Logged at each step:
  ```
  [step 0] cachePos=84 totalLen=85 input_past_k0=(1, 4, 84, 128) output_present_k0=(1, 4, 85, 128)
  [step 1] cachePos=85 totalLen=86 input_past_k0=(1, 4, 85, 128) output_present_k0=(1, 4, 86, 128)
  [step 2] cachePos=86 totalLen=87 input_past_k0=(1, 4, 86, 128) output_present_k0=(1, 4, 87, 128)
  ```
  Shapes are bit-correct; `pastLen` advances; `cache_position` is
  passed correctly.
- **Per-step wall time stays constant at ~14.5 ms across all steps,
  regardless of the cache size.** A correct KV-attention should grow
  ~linearly with cache size (Python sees 19 ms → 61 ms across the
  same range). Constant per-step time strongly suggests the model is
  not actually attending over the larger cache — it's seeing the same
  effective-size context every step.
- **Same garbage with several "fix" attempts:**
  - Deferring disposal of old past_kv until next iteration's
    `ClearBoundInputs` releases binding refs.
  - Fresh `OrtIoBinding` per step (rules out output-buffer aliasing
    across reused bindings).
  - Copy-out-and-rewrap each step — read present_kv via
    `GetTensorDataAsSpan<float>().ToArray()` and re-create OrtValues
    from those buffers. Even with all I/O on CPU memory and zero
    binding-internal lifetime concerns, transcript is still garbage.

So the bug is **not** in OrtValue lifetime, output-buffer aliasing,
or input construction. The most likely remaining hypothesis: the
binding path interacts differently with the unified decoder graph's
Memcpy nodes (the warning at session-load: `2 Memcpy nodes are added
to the graph`) than `Run()` does. Possibly some implicit
host↔device transition happens in `Run()` that `RunWithBinding`
expects the caller to have already arranged.

#### Why the precedents don't immediately apply

VibeVoice and Qwen3 both use IOBinding successfully on their decoder
step graphs. Two structural differences from the unified Granite
decoder:

- **Granite has 80 separate per-layer K/V outputs** (`present_key_<L>`,
  `present_value_<L>` for L in 0..39). Qwen3 packs all KV into a
  single pair (`present_keys`, `present_values`) of monolithic
  tensors. Different binding accounting load.
- **Unified Granite handles BOTH prefill and step in one graph.**
  VibeVoice and Qwen3 use separate prefill/step graphs. The unified
  graph traces with "always-present past_kv" semantics and may have
  graph-level structures (audio cumsum-merge, position-derived RoPE)
  that interact with binding's memcpy-bypass behaviour.

### Disposition

**IOBinding deferred until a focused investigation can isolate the
binding-vs-Run divergence.** The C# smoke ships with `--io-binding`
gated behind a warning (perf measurement only, not for correctness).

The fp32 unified decoder from Run 3 remains the shipping
configuration:

| Audio | RTF | Tok/s |
|---|---:|---:|
| 6.4 s | 12.8× | 50.4 (Python) / 47.6 (C#) |
| 90 s | 5.38× / 4.04× (C#) | 16.4 / 12.1 |

That's already comfortably real-time at long form. The C# 25-35%
overhead is real but doesn't change the deployment story.

#### Next leverage points (for a future Run)

- **Isolate the IOBinding bug** with a minimum-repro on the unified
  decoder — bind ALL inputs as CPU-side OrtValues, log what ORT
  reports about node placement, see if the Memcpy nodes are silently
  bypassed under binding.
- **Try IOBinding on the SPLIT decoder pair.** If binding works
  there (matching Qwen3's success on its split path), the unified
  decoder's per-layer KV layout or audio-merge structure is the
  culprit, and the fix is graph-shape rather than lifetime. Cost:
  resurrect the split bundle (still supported) for benchmarking
  alongside unified.
- **Reduce the C# `.ToArray()` overhead even without IOBinding.**
  Use `Buffer.BlockCopy` or `Marshal.Copy` from a `Memory<float>` view
  of the OrtValue's tensor. Should at least cut the host-side copy
  cost by removing the per-element bounds-check overhead in
  `Span<T>.ToArray()`.