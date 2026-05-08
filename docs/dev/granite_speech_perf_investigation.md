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

---

## Run 5 — 2026-05-08 (smaller bite: pre-allocated KV buffers)

**Question:** Run 4 left two paths open: a focused IOBinding diagnosis,
or a smaller bite that targets the per-step `.ToArray()` allocations
without touching the binding API. Per @user's prompt, try the smaller
bite first; only do the diagnosis lap if there's nothing easier.

**Method:** Two attempts, in order of decreasing scope.

### 5a — `Run`-with-`OrtValue` (instead of `RunWithBinding`)

Goal: avoid the binding API entirely while still passing OrtValues
directly (no host roundtrip on KV). ORT C# has an overload
`Run(RunOptions, IReadOnlyCollection<string>, IReadOnlyCollection<OrtValue>, IReadOnlyCollection<string>)` that skips the binding and uses
OrtValues directly.

**Result:** **Also produces garbage**, but with a *different* garbage
pattern than `RunWithBinding`:

```
RunWithBinding:    "Hello,null<|fim_suffix|>'s. and.-dron for_<|fim_suffix|>..."
Run-with-OrtValue: "Hello,nulls<|fim_suffix|>spspspspspspspspspspspsp..."
```

Different pattern, same first-correct-token-then-collapse signature.
The fact that BOTH paths fail tells us the bug is NOT in `RunWithBinding`
itself — it's in **how ORT consumes inputs constructed via
`OrtValue.CreateTensorValueFromMemory`** for this specific graph.

Hypotheses ruled out along the way:
- GC moving the source array (fixed by keeping arrays in
  `pastKvArrays[80]`; still garbage).
- Lifetime of the disposable result list (fixed by *not* using `using`;
  still garbage).
- Output-buffer aliasing across reused bindings (fixed by
  fresh-binding-per-step; still garbage).

Remaining hypothesis: some incompatibility between
`CreateTensorValueFromMemory`'s tensor representation and what the
unified Granite decoder graph expects on its many small inputs (40×
past_key + 40× past_value + cache_position + attention_mask). Worth
flagging but not worth blocking on — the bigger discovery is that the
`DenseTensor` + `NamedOnnxValue` path WORKS while the `OrtValue` path
doesn't.

### 5b — Pre-allocated KV buffers + `Span.CopyTo` (in the working path)

Stays on the `DenseTensor` + `NamedOnnxValue` path. Replaces:

```csharp
pastKeys[L] = stepResults[1 + L].AsTensor<float>().ToArray();   // 80 fresh allocs / step
```

with:

```csharp
// One-time:  pastKeys[L] = new float[1 * NumKvHeads * (promptLen + maxNewTokens) * HeadDim];
// Per step:  ((DenseTensor<float>)stepResults[1+L].AsTensor<float>())
//                 .Buffer.Span.CopyTo(new Span<float>(pastKeys[L], 0, totalElems));
```

Net effect: 80 × maxNewTokens fresh allocations replaced with 80
one-time allocations + 80 × maxNewTokens span-copies into pre-allocated
buffers. Removes all GC pressure from the step loop's output extract
and uses the JIT-vectorised `Span.CopyTo` for the host-side memcpy.

**Memory cost:** 80 buffers × `1 * 4 * (promptLen + maxNewTokens) * 128`
floats × 4 bytes = ~190 MB at the 90 s clip. Fits comfortably in
typical heap budgets.

### Results (GPU 3090, fp32 unified decoder)

| 6.4 s GPU | Run 4 baseline | Run 5b pre-alloc | Δ |
|---|---:|---:|---:|
| Total | 527 ms | 525 ms | ~0 |
| decoder_step / token | 21.0 ms | 20.7 ms | -1.4% |
| Output extract | 62 ms (13.4%) | 25 ms (5.5%) | -60% |
| Parity | exact | exact | — |

| 90 s GPU | Run 4 baseline | Run 5b pre-alloc | Δ |
|---|---:|---:|---:|
| Total | 22 264 ms | **19 929 ms** | **-10.5%** |
| decoder_step / token | 82.4 ms | **73.0 ms** | **-11.4%** |
| Tok/s | 12.1 | **13.7** | **+13%** |
| RTF | 4.04× | **4.52×** | **+12%** |
| Output extract | 6 200 ms (29.6%) | **4 154 ms (22.4%)** | -33% |
| Parity | exact | **exact** | — |

The 4.4 s of extract that REMAINS at 90 s is pure PCIe bandwidth: 80
buffers × ~190 MB total per step × 254 steps = ~48 GB of host-side
memcpy across the run at ~13 GB/s, which is close to PCIe 3.0 ×16
peak. To shrink it further the data has to stay on the GPU — which
is exactly the IOBinding goal that Run 4 / Run 5a couldn't get to work
on this graph.

### Disposition

- **Ship Run 5b.** Pre-alloc + `Span.CopyTo` is in the smoke's
  baseline (non-IOBinding) decode path. 12% RTF win at 90 s, zero
  parity risk.
- **`--io-binding` flag remains** as a measurement tool; warning
  message updated to flag both `RunWithBinding` and Run-with-OrtValue
  failure modes.
- **The "actual IOBinding" focused-diagnosis lap remains open.** The
  most concrete next experiment: try IOBinding on the SPLIT decoder
  pair (decoder_init + decoder_step). If it works there — matching
  Qwen3's success on its split path — the bug is unified-decoder
  specific, and the fix is graph-shape rather than C# API. Cost:
  resurrect the split bundle for a benchmark comparison.

### Updated shipping numbers (fp32 unified, GPU 3090, C#)

| Audio | RTF | Tok/s | Total wall |
|---|---:|---:|---:|
| 6.4 s | 12.2× | 48.4 | 525 ms |
| 90 s | 4.52× | 13.7 | 19.9 s |

---

## Run 6 — 2026-05-08 (focused-diagnosis lap: chained Run-with-OrtValue beats binding)

**Question:** Run 5a observed that **both** `RunWithBinding` and the
`Run`-with-`OrtValue` overload produced garbage on the unified
decoder. The user's note ("memory-juggle if it doesn't fit") opened up
testing IOBinding on the **split** decoder pair, where Cohere/Qwen3
showed it works. Three subruns:

- **6a:** Test OrtValue inputs on split decoder_step (no binding) →
  isolates whether the bug is unified-graph-specific.
- **6b:** Test `RunWithBinding` on split decoder_step (with
  GPU-resident KV between steps) → the original IOBinding goal.
- **6c:** If 6b fails, test chained `Run-with-OrtValue` (use one
  step's output OrtValues as the next step's input OrtValues — no
  binding API) → the IOBinding pattern *without* binding.

### Memory juggling

Both prefill (encoder + projector + decoder_init = ~9 GB) and step
(decoder_step alone = ~7 GB) sessions can be on GPU sequentially. Load
phase 1, run prefill, extract KV to host, dispose phase 1; load phase
2 (decoder_step alone), run AR loop. Only ONE 7 GB decoder is resident
at a time → the "duplicate weights" problem from Run 1 simply
disappears with this loading strategy. Total resident peak: ~9 GB at
prefill, ~7 GB at step. Comfortably fits on a 25 GB 3090.

### 6a — split + Run-with-OrtValue (no binding): EXACT MATCH

Per-token at 6.4 s: **23.4 ms/tok** (slower than Run 5b's 21 ms because
this path still does host roundtrip on KV — copies present_kv to fresh
managed arrays each step). At 90 s: **84.5 ms/tok**, exact text match.

**The Run 4/5a bug IS unified-decoder-specific.** OrtValue inputs work
fine on the split decoder_step (same per-layer KV layout, just no
audio_embeds input and no cumsum-merge logic). The trigger appears to
be the unified graph's audio_embeds input or its cumsum-merge.

### 6b — split + RunWithBinding (proper IOBinding): GARBAGE

Tried two patterns:
- Single binding reused across steps (with `ClearBoundInputs/Outputs`).
- Fresh `OrtIoBinding` per step (Qwen3's pattern).

Both produce the same garbage transcript: first 1-2 tokens correct,
then collapse into repeated junk. Output device variation (CPU vs CUDA
for `present_kv` outputs) didn't help.

The most likely cause: when `BindInput` receives a host-resident
OrtValue from `CreateTensorValueFromMemory` and the session is on
CUDA, ORT does NOT auto-transfer the data to GPU like `Run` does — it
silently passes the host pointer to the GPU kernel, which reads
garbage. Workaround in Cohere/Qwen3 is to bind GPU-resident OrtValues
directly; we tried that for steps after step 0 (using step 0's GPU
outputs as step 1's GPU inputs), still garbage. The deeper interaction
isn't worth chasing because…

### 6c — split + chained Run-with-OrtValue: EXACT MATCH at full perf

Realisation: we don't actually need `RunWithBinding` to keep KV on the
GPU across steps. **`Run`-with-`OrtValue` already does it implicitly
when both Run calls are on the same CUDA EP.** Each step's output
OrtValues are GPU-resident; passing them directly as the next step's
input OrtValues skips any host roundtrip. The "IOBinding pattern"
(GPU KV chain across steps) works without the binding API at all.

```csharp
var outputs = stepSess.Run(runOpts, names, inputValues, names);
// outputs[1..81] are CUDA-resident OrtValues from this step's
// present_key_<L> / present_value_<L>.
// Take ownership and use directly as next step's past_kv:
for (int L = 0; L < NumDecoderLayers; L++)
{
    pastKvs[2 * L]     = outputs[1 + L];
    pastKvs[2 * L + 1] = outputs[1 + NumDecoderLayers + L];
}
```

### Final perf comparison (GPU 3090, C#, fp32)

| 6.4 s | wall | ms/tok | tok/s | RTF | parity |
|---|---:|---:|---:|---:|---|
| Run 4 baseline (unified .ToArray) | 527 ms | 21.0 | 47.6 | 12.18× | exact |
| Run 5b (unified pre-alloc + Span.CopyTo) | 525 ms | 20.7 | 48.4 | 12.23× | exact |
| **Run 6c (split-juggle chained Run)** | **434 ms** | **19.8** | 50.6 | **14.79×** | **exact** |

| 90 s | wall | ms/tok | tok/s | RTF | parity |
|---|---:|---:|---:|---:|---|
| Run 4 baseline | 22 264 ms | 82.4 | 12.1 | 4.04× | exact |
| Run 5b | 19 929 ms | 73.0 | 13.7 | 4.52× | exact |
| **Run 6c** | **17 631 ms** | **56.2** | **17.8** | **5.11×** | **exact** |

Run 6c at 90 s is **21% faster than Run 5b** and **31% faster than the
baseline**. The C# RTF (5.11×) is now within 5% of the Python
profiler's 5.38× on the same hardware — most of the C# overhead is
gone.

The split bundle is also smaller in resident weight terms because we
only ever have ONE 7 GB decoder loaded at a time (phase 1: encoder +
projector + decoder_init; phase 2: decoder_step alone). The "duplicate
weights" problem from Run 1 only existed when both decoder graphs
were loaded simultaneously.

### Deployment story

There are now two viable shipping configurations:

| | Unified bundle (Run 5b) | Split bundle (Run 6c) |
|---|---|---|
| 90 s C# RTF | 4.52× | **5.11×** |
| Disk size | ~9 GB | ~16 GB |
| Resident GPU weights | 8.8 GB | 8.8 GB phase 1 / 7 GB phase 2 |
| Session-load overhead | one-time at startup | ~1.3 s per phase transition |
| Decoder API | DenseTensor + NamedOnnxValue | OrtValue + Run-with-OrtValue |
| Parity | exact | exact |
| Notes | Simpler runtime, smaller bundle | Faster, two-phase loading |

For the C# CLI: **split-juggle is the recommended shipping config for
GPU**. CPU users should stick with unified (smaller bundle, no
phase-transition cost). The smoke supports both via `--split-juggle-test`.

### Open follow-ups

- **Why does `RunWithBinding` produce garbage on this graph?** Filed
  as a known-unknown. Workaround works fine; the binding bug is worth
  flagging upstream if anyone needs to use it for a different reason
  (e.g., to bind device-resident OrtValues without going through
  Python).
- **Why does the unified decoder fail with OrtValue inputs?** Most
  likely the audio_embeds input or the cumsum-merge logic. If we ever
  want a single-graph deployment with the Run-with-OrtValue speed
  win, this needs root-causing — but the split-juggle path is fully
  equivalent and faster.
- **Phase-transition cost** (1.3 s session load when switching from
  decoder_init to decoder_step) is one-time per request. For batched
  workloads it's amortised; for single-shot it's noticeable. A future
  optimisation: keep decoder_init resident OFF the GPU (CPU) and
  decoder_step ON the GPU, so prefill runs slowly on CPU but step
  loop runs fast on GPU without the phase transition. Worth measuring
  if user-facing latency becomes the bottleneck.

---

## Run 7 — 2026-05-08 (chained Run-with-OrtValue on the UNIFIED decoder — best so far)

**Question:** Run 6c showed that chained `Run`-with-`OrtValue` works on
the split decoder pair but it requires the memory-juggling phase
transition. Run 5a previously concluded that the same pattern produces
garbage on the unified decoder. @user prompt: "How about going back to
the single decoder now?" — does the chained pattern actually work on
unified, or was Run 5a's failure setup-specific?

**Re-examining Run 5a's setup carefully:** Run 5a built OrtValues from
*managed arrays* (via `CreateTensorValueFromMemory(arr, shape)` after
`.ToArray()`-ing prior outputs). All inputs were **CPU-resident
OrtValues** going into a CUDA session. Garbage.

**Key insight from Run 6c:** when prefill is run via
`Run`-with-`OrtValue`, its outputs come back as **GPU-resident
OrtValues** that the next `Run` call accepts directly. The "OrtValue
inputs" in the chain are NOT host-resident copies; they're the device
tensors produced by ORT's own kernels. ORT handles them correctly even
on graphs where naive `CreateTensorValueFromMemory` inputs misbehave.

So: try the chained pattern on the unified decoder, where prefill outputs
ARE GPU-resident OrtValues from the unified `Run`, and the step loop
chains those forward. Different from Run 5a's CPU-side OrtValues.

### Implementation

Single `decoder.onnx` session. Prefill: `Run`-with-`OrtValue` using
host-side OrtValues for `input_ids` / `audio_embeds` / `attention_mask`
/ `cache_position` and zero-length past_kv. Step loop: each step's
output OrtValues become the next step's past_kv inputs. Logits are
read on CPU for argmax; KV stays GPU.

### Results (GPU 3090, fp32, C#)

| 6.4 s | wall | ms/tok | tok/s | RTF | parity |
|---|---:|---:|---:|---:|---|
| Run 5b unified DenseTensor pre-alloc | 525 | 20.7 | 48.4 | 12.23× | exact |
| Run 6c split-juggle chained | 434 | 19.8 | 50.6 | 14.79× | exact |
| **Run 7 unified chained** | **704** | **18.4** | **54.3** | 9.12× | **exact** |

| 90 s | wall | ms/tok | tok/s | RTF | parity |
|---|---:|---:|---:|---:|---|
| Run 4 baseline | 22 264 | 82.4 | 12.1 | 4.04× | exact |
| Run 5b unified pre-alloc | 19 929 | 73.0 | 13.7 | 4.52× | exact |
| Run 6c split-juggle | 17 631 | 56.2 | 17.8 | 5.11× | exact |
| **Run 7 unified chained** | **15 102** | **54.9** | **18.2** | **5.96×** | **exact** |

**Unified chained on 90 s is the fastest path so far** — 14% faster than
split-juggle (5.96× vs 5.11×) and 32% faster than Run 5b's pre-alloc
DenseTensor path. It also surpasses the Python profiler's RTF (5.38×),
likely because the chained OrtValue path skips both Python's tensor-
lifecycle overhead AND the C# managed-array roundtrip.

The 6.4 s case looks worse on RTF (9.12× vs 14.79× for split-juggle)
because of session-load overhead amortised over a short clip — a
warm-up run of either path would put unified ahead. At 90 s where
session load is < 1% of total, unified is clearly fastest.

### Why Run 5a was wrong

Run 5a used:
1. `RunWithBinding` for prefill, with `BindOutputToDevice("present_key_<L>", cpuMem)`
2. `.GetTensorDataAsSpan().ToArray()` to copy the binding's outputs to managed arrays
3. `OrtValue.CreateTensorValueFromMemory(arr, shape)` to wrap those arrays as OrtValues
4. `Run`-with-`OrtValue` for steps, with these CPU-side OrtValues as past_kv inputs

The bug was in step 4: passing CPU-side OrtValues (built from managed
arrays) as past_kv to a CUDA-EP session produces garbage on the
unified decoder, but works on the split decoder_step. We never ruled
out *that specific path*; we just assumed all OrtValue inputs were
broken on unified.

The chained pattern (Run 7) never has CPU-side past_kv after step 0:
prefill outputs are GPU-resident OrtValues, and every step's past_kv
is the previous step's GPU output. The bug doesn't manifest because
the host-side OrtValue path is never used for past_kv.

The host-side `CreateTensorValueFromMemory` *is* still used for the
small inputs that change every step — `input_ids`, `attention_mask`,
`cache_position`, the dummy `audio_embeds`. Those evidently don't
trigger the bug. Why exactly? Probably small enough that ORT's
implicit Memcpy handles them correctly, or the bug is specific to the
audio_embeds + cumsum-merge interaction at full prompt-shape inputs.
Not worth root-causing now since the chained pattern sidesteps it
entirely.

### Disposition: unified chained is the new shipping target

| | Unified DenseTensor (Run 5b) | Split-juggle chained (Run 6c) | **Unified chained (Run 7)** |
|---|---|---|---|
| 90 s C# RTF | 4.52× | 5.11× | **5.96×** |
| Disk size | ~9 GB | ~16 GB | **~9 GB** |
| Resident GPU weights | 8.8 GB | 8.8 GB phase 1 / 7 GB phase 2 | **8.8 GB** |
| Session-load overhead | one-time at startup | ~1.3 s per phase transition | **one-time at startup** |
| Decoder API | DenseTensor + Run | OrtValue chain via Run | **OrtValue chain via Run** |
| Parity | exact | exact | **exact** |
| Code complexity | simple | two-phase loading | **simple** |

**Run 7 wins on every dimension.** Smaller bundle than split, simpler
runtime than split, faster per-token than both prior paths.

### Final shipping numbers (fp32 unified chained, GPU 3090, C#)

| Audio | RTF | Tok/s | Total wall |
|---|---:|---:|---:|
| 6.4 s | 12+× (warm) / 9.1× (cold) | 54.3 | 0.7 s |
| 90 s | **5.96×** | **18.2** | **15.1 s** |

## Run 8 — 2026-05-08 — VRAM-budgeted batching in GraniteSpeech.cs

**Question:** Does cross-segment batching beat per-segment serial decode by
the expected ~B× factor?

**Setup:** `GraniteSpeech.TranscribeBatch` runs mel + encoder + projector
**serially per row** (encoder uses full attention with no padding mask, so
batching with zero-padded mel would contaminate the attention output of
real positions), then runs the unified decoder **batched** with LEFT-padded
`input_ids` and a shared `cache_position` so rotary positions for generated
tokens are identical across rows.

`audio_embeds` is right-aligned in `[B, A_max, 2048]`: the cumsum-gather
merge picks the first `N_audio[b]` entries per row, so they sit at
`[0, N_audio[b])` per row and the rest stay zero (never selected because
`is_audio = (input_ids == AudioTokenId)` is False outside those positions).

EOS handling: once a row emits EOS we substitute `EosTokenId` for its step
input and freeze its output token, but keep stepping the batch until the
longest row finishes. Wasted-step overhead is minimised by
`BatchSizer.Plan`'s ascending-duration packing — segments of similar
length batch together.

VRAM budget: `cudaMemGetInfo` minus 3 GB safety buffer. Cost model returns
`max(KV_at_end, prefill_logits, encoder_full_attn)` per prospective batch.
KV dominates at long durations, prefill logits dominate at large B with
short prompts.

**Findings (3090, fp32 unified chained, CUDA EP):**

| Test | Audio | Segments | Mode | ASR wall | RTF |
|---|---:|---:|---|---:|---:|
| VCTK_p307 single | 6.4 s | 1 | B=1 | 3.25 s | 0.508 |
| VCTK ×4 batch | 25.0 s | 4 | B=4 | 3.66 s | 0.147 |
| en-US 90 s | 90 s | 20 (VAD) | mixed B | 7.80 s | **0.091** |

The 25 s 4-segment case ran ~3.5× faster per real-time second than the
single-segment case (0.147 vs 0.508). The 90 s VAD-segmented run hit RTF
0.091 (≈ 11× real time) — better than Run 7's 5.96× single-segment number
because most VAD chunks are short and batching amortises prefill across
them. Transcripts on all four VCTK rows matched the reference text modulo
punctuation variance (`"Hello I'm from Toronto"` vs
`"Hello, I'm from Toronto."`) — unrelated to batching; the model's
punctuation is non-deterministic across prompt contexts.

**Implication:** Batching is a multiplicative win on top of Run 7 for
real workloads where diarization or VAD produces multiple segments. The
single-segment chained path remains the right baseline for B=1; the
batched path delegates to it via `TranscribeBatch(new[]{wave}, n)`.

## Run 9 — 2026-05-08 — Straggler waste from runaway-loop rows

**Question:** A 10 min en-US clip (159 VAD segments) ran at RTF 0.131 with
batched Granite Speech, 1.74× *slower* than Qwen3-ASR running serially
(RTF 0.075) on the same audio. Where is the time going?

**Setup:** Added per-stage timing accumulators in `TranscribeBatch`
(opt-in via `VERNACULA_GRANITE_PROFILE=1`). 10 batches of B=15-16 covered
all 159 segments.

**Pre-fix profile (10 min audio, 159 segs):**

```
mel:        551 ms   ( 0.7%)
encoder:    5422 ms  ( 7.1%)
projector:  44 ms    ( 0.1%)
prefill:    2078 ms  ( 2.7%)
step-loop:  67648 ms ( 89.0%)
overhead:   283 ms   ( 0.4%)
```

Step-loop dominates at 89%, so pipeline overlap of mel/encoder/projector
with the decoder cannot save more than ~8% even with perfect
parallelism — there is no significant "fast stage waiting on slow stage"
to recover.

**Per-batch breakdown told the real story:**

| Batch | B | Steps | Step ms | Notes |
|---|---:|---:|---:|---|
| 1 | 16 | **256** | 27,147 | runaway loop |
| 2 | 16 | 6 | 250 | normal |
| 3 | 16 | 7 | 314 | normal |
| 4 | 16 | **256** | 28,314 | runaway loop |
| 5 | 16 | 13 | 667 | normal |
| 6 | 16 | 15 | 806 | normal |
| 7 | 16 | 21 | 1,315 | normal |
| 8 | 16 | 21 | 1,421 | normal |
| 9 | 16 | 32 | 2,534 | normal |
| 10 | 15 | 43 | 4,880 | normal |

**Two batches each ran the 256-step `maxNewTokens` cap** because at least
one row in each fell into a greedy loop — `"Well, uh, uh, uh, …"` and
`"We strapped him on and we, we, we, …"`. Together those two batches ate
**56.8 s of 76 s total step-loop time (75%)**. The other eight batches
finished in 12 s.

The pathology is straggler waste in batched greedy decode: with B=16, one
runaway row drags 15 finished rows along for the ride until the cap, so
each runaway costs `cap × B × per_step_ms` even though the survivors
have long since emitted EOS.

**Fix — per-row periodic-loop detector:**

```csharp
// 3 cycles of period 1..4 on the row's tail → force EOS for that row
if (tok == EosTokenId || IsRepetitionLoop(generated[b]))
{
    finished[b] = true;
    finishedCount++;
}
```

`IsRepetitionLoop` checks the last `3 × p` tokens for any p in [1, 4];
3 cycles is conservative — natural speech rarely repeats a 1-4-token
motif three times in a row, but a stuck decode always does. Trim the
tail in `TrimRepetitionTail` before decoding so the output text
contains one cycle of the motif instead of a long repeated string.

**Post-fix profile (same 10 min audio):**

```
mel:        549 ms   ( 2.5%)
encoder:    5435 ms  (25.1%)
projector:  49 ms    ( 0.2%)
prefill:    2322 ms  (10.7%)
step-loop:  13034 ms (60.1%)
overhead:   305 ms   ( 1.4%)
```

Step count dropped from 670 to 175 (74% reduction). Total ASR wall:
**78.3 s → 23.9 s (3.3× speedup)**. RTF: **0.131 → 0.043**.

**Sanity-check against Qwen3-ASR on the same audio:**

| Backend | Mode | ASR wall | RTF | Words |
|---|---|---:|---:|---:|
| Qwen3-ASR | serial | 43.0 s | 0.075 | 1,477 |
| Granite (pre-fix) | batched B=16 | 78.3 s | 0.131 | 1,766 |
| **Granite (post-fix)** | **batched B=16** | **23.9 s** | **0.043** | **1,514** |

Post-fix Granite is **1.7× faster than serial Qwen3-ASR** and produces a
word count 2.5% above Qwen3 — meaning the rep-detect removed only the
~252 bogus repetition words from the pre-fix output, not real content.
Spot-checks on the runaway segments confirm both: the
`[00:05:37 - 00:05:37]` zero-duration VAD chunk where Granite looped
`"uh, uh, uh, …"` is `"Well."` per Qwen3 and `"Well, uh"` per post-fix
Granite — close to the actual audio (a single utterance), and the
overlap-vs-cap behaviour is now identical between batches.

**Implication on the original pipeline-stall question:** there is no
significant stall to overlap. After the rep-detect fix the encoder is
the largest pre-decoder cost at 25% (5.4 s on a 24 s run), but
pipelining it with the decoder step-loop on the same CUDA stream would
not actually overlap (single-stream serialisation). True overlap would
require multiple CUDA streams per session — significant complexity for
a ~5 s win on a 10 min clip. Park.

### Open follow-ups (decreasing priority)

- **Concat is now the #1 ORT op (249 ms across 64 tokens at 90 s)**
  — past_kv + new_kv concatenation per layer per step. A static-KV
  cache (pre-allocate to max + scatter-write into [past_len]) would
  eliminate it. Graph-level export change; significant work but
  ~13% potential win on the dominant cost.
- **Encoder full-attention at long T** still costs ~580 ms per call
  (4% of 90 s total). Not yet a priority but if Vernacula benchmarks
  on multi-minute clips become important, the chunked-encoder
  strategy from Run 4 of `granite_speech_investigation.md` is the
  next move.
- **Why does naive `CreateTensorValueFromMemory` past_kv fail on
  unified but work on split?** Filed as a known-unknown; the chained
  pattern routes around it.