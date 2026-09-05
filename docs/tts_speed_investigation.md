# OmniVoice-IPA speed: reader vs web demo — investigation log

The question: the reader (Vernacula.Tts.Avalonia, C# + ONNX Runtime) seems to generate much
faster than the web demo (browser, onnxruntime-web). Is it, and by how much, and why?

Same utterance for every run — two sentences, 170 IPA chars, 190 target tokens (7.06 s), the
library's `en` reference voice (83 codes), 32 diffusion steps. Machine: RTX 3090, i7-10700K
(8 physical / 16 logical cores).

## Run 1 — 2026-09-05 ~10:40 — the C# engine, CUDA and CPU

`OmniVoiceIpaTts` (what the reader drives), fp32 base + v7 diff, via the scratch harness.

    ep=Cuda load=3775ms
      run0: 7.06s audio in 3874ms  (1.82x RT)     ← first call: CUDA kernel warm-up
      run1: 7.06s audio in 2690ms  (2.62x RT)
      run2: 7.06s audio in 2861ms  (2.47x RT)

    ep=Cpu  load=6119ms
      run0: 7.06s audio in 67608ms  (0.10x RT)    ← partly contended by Run 2's first attempt
      run1: 7.06s audio in 53533ms  (0.13x RT)

So the C# engine on CUDA is ~2.5× faster than real time; the same engine on the CPU, fp32 on 16
threads, is ~8× slower than real time. Twenty-fold between the two, from the execution provider
alone — the reader defaults to `EP=Cuda`, which is why it feels fast.

## Run 2 — 2026-09-05 ~10:50 — one WASM transformer forward, 8 threads, S=350

`tools/bench-wasm.mjs` on the int4 web build (the same wasm ORT the browser runs, in Node; only
the thread count differs). S=350 ≈ text 170 + ref 83 + target 190 minus the pieces that share
positions — close enough to the browser's real sequence for this utterance.

    first attempt, while Run 1's CPU pass was still running:  7910 ms  → 253 s / generation
    uncontended:                                               3965 ms  → 127 s / generation
                                                               (runs: 3933, 3974, 3987)

The contended number is the one a user with a busy machine gets, and it is the kind of number
that makes the demo look broken. Uncontended, 32 forwards is ~127 s for 7 s of audio: ~0.06× real
time — roughly half the speed of the C# CPU path, which runs fp32 with native kernels against
int4 weights that WASM has to dequantize on the fly, and with 16 threads to WASM's 8.

## Run 3 — 2026-09-05 — the browser itself, headless Firefox, full generation

`tools/smoke-tts.mjs` serves the demo's own pipeline with COOP/COEP (so threads are on) and
drives it in headless Firefox. First attempt failed in the tool, not the demo: `_keys.json`
became a `{engine, dirs, languages, foreign}` manifest when per-language data loading landed and
the tool still read it as a flat list. Fixed to resolve the language's keys the way
`phonemizer.ts` does.

Second attempt failed the same way one layer down: the tool's `voicesUrl` was the pre-library
`voices.json`, and the transpiled engine it serves from `build-smoke/` was a Sep 1 copy that
predates `voice-codes.json`. Rebuilt `build-smoke/` from the current `src/inference/*.ts` (tsc,
`.ts` imports rewritten to `.js`) and pointed the tool at `voices.jsonc` + `voice-codes.json`.
Third attempt's chain died after its first run because `pkill -f "firefox --headless"` matched
the shell running it. Four defects in a tool that had not been run since the day it was written —
worth knowing before trusting any "smoke" that is not in CI.

    firefox, WASM (forced), 8 threads, COOP/COEP on:
      ep=wasm  targetTokens=266  audio=10.09s
      generate 181.8s  (transformer 174.9s, host 5.1s)        → 0.055x real time

⚠ targetTokens is 266 here against 190 in the C# runs: the tool picks the language's voice with
its own `voiceFor` default rather than the library's `default` entry, so a different reference
length drove the duration estimate. The per-forward cost is what matters and it agrees with Run 2's
extrapolation (174.9 s / 64 forwards = 2.7 s at S≈430 vs 3.97 s at S=350 in Node — the browser's
transformer share is in the same band). The audio is 10 s rather than 7 s; the ratio holds either
way.

## Run 4 — 2026-09-05 ~11:30 — WebGPU: broken since the threading change, then measured

Both WebGPU runs failed before the first forward:

    [webgpu] DataCloneError: Failed to execute 'postMessage' on 'Worker': GPUDevice object
    could not be cloned.  →  Error: no available backend found.

Cause: 21c65d1 ("8 WASM threads and run the session off the main thread") set
`ort.env.wasm.proxy = true` unconditionally. In proxy mode ORT posts its `env` to a worker, and
`useMaxLimitsDevice` had put a GPUDevice in `env.webgpu.device` — a GPUDevice cannot cross a
worker boundary. `pickExecutionProvider` chooses WebGPU whenever an adapter exists and there is
no fallback to WASM on session failure, so from that commit **the deployed demo could not
generate at all on Chrome, or on any Firefox with WebGPU enabled**. That is what "it doesn't
happen with WebGPU on the demo" was. Nothing measured it because the smoke tool that exercises
WebGPU had itself been broken (Run 3) since before the change.

Fix: the EP is decided in ortInit.ts, before ORT is configured, and `proxy` follows it — on for
WASM (which needs the worker to keep the tab responsive), off for WebGPU (whose work is
asynchronous on the GPU anyway). Then:

    chrome / webgpu (Dawn, RTX 3090), decoder on wasm:
      ep=webgpu  targetTokens=266  audio=10.09s
      generate 40.2s  (transformer 34.8s, host 3.5s)      → 0.25x real time

    firefox / webgpu: killed after the transformer session came up — Runs 6 and 8 established
    Firefox's WebGPU is flat at ~3 s per forward whatever the work (≈190 s here), and no one
    deploys to it.

## The table

Same two sentences, 32 steps, one `en` reference voice (the browser tool picked a longer
reference, hence 10.09 s of audio to the C# runs' 7.06 s — compare rates, not totals).

| path | audio | wall | × real time |
|---|---|---|---|
| C# engine, CUDA fp32 (the reader's default)      | 7.06 s | 2.7–2.9 s | **2.5×** |
| browser, Chrome WebGPU, int4                      | 10.09 s | 40.2 s | 0.25× |
| C# engine, CPU fp32, 16 threads                   | 7.06 s | 54–68 s | 0.12× |
| browser, Firefox WASM, int4, 8 threads            | 10.09 s | 181.8 s | 0.055× |

So: confirmed, and by more than it looks like from the reader — the reader runs the fp32 model
on the GPU through CUDA and is ~10× faster than the browser's best path on the same GPU, and
~45× faster than the browser path most visitors actually get.

**Where the 10× between CUDA and Chrome WebGPU goes.** Both are the same GPU. The C# path is
fp32 with cuBLAS/cuDNN kernels and ORT's CUDA graph optimisations; the browser path is
`MatMulNBits` int4 weights dequantised inside WGSL shaders, fp32 activations (no shader-f16),
per-op dispatch through Dawn, and 32 forwards (the CFG pair is batched as B=2 in one forward,
in both implementations). Per forward: 34.8 s / 32 = 1.09 s at S≈430 on WebGPU vs ~87 ms on
CUDA. (An earlier draft of this paragraph counted 64 forwards; the diffusion loop batches the
pair — see diffusion.ts.)

## Run 5 — 2026-09-05 ~12:20 — what can ORT's own fusion passes do to this graph?

Goal 1× real time on Chrome WebGPU (from 0.25×). The budget: 32 forwards for 7 s of audio →
≤220 ms per B=2 forward at S≈430; measured 1.09 s. The graph has 4,768 non-Constant nodes, every
one a WebGPU dispatch, and Dawn's per-dispatch cost is 0.1–0.3 ms — so dispatch count alone could
be most of the forward. First question, therefore: how many of those nodes fuse for free?

`onnxruntime.transformers.optimizer` over the fp32 v7 graph (`onnx_base/…_v7.onnx`):

    model_type=gpt2, no geometry:        7371 → 5727 nodes (29 s)
      SimplifiedLayerNormalization 57, SkipSimplifiedLayerNormalization 56
      removed: Cast 344, Mul 226, Add 169, Pow 113, ReduceMean 113, Sqrt 113, Div 113, Reshape 57
    model_type=gpt2, num_heads=16, hidden_size=1024 (Qwen3-0.6B):  identical — no Attention,
      MultiHeadAttention or RotaryEmbedding fusions
    model_type=phi:  TypeError inside FusionOptions — not applicable to this export

So the RMSNorms fuse (113 of them, ~14 nodes each — a third of the graph's dispatches), and
nothing else does: Qwen3's q/k RMSNorm inside the attention block, GQA, and the boolean 4-D mask
this export takes as an input match none of ORT's attention patterns. Fusing attention and rotary
into GroupQueryAttention/RotaryEmbedding — both of which ORT-web 1.29's WebGPU bundle carries —
would be hand graph surgery or a re-export through contrib ops. Deferred until the profile says
whether dispatch count is actually the bottleneck.

Building the RMSNorm-fused graph through the shipped recipe (int4 MatMulNBits block-32 →
int8 per-row embedding) into `onnx_web/v7-fused/` for a like-for-like bench. Also noted: the
Python tooling on this box is the system interpreter's onnxruntime 1.24.4 — there is no
`export_venv` for this pipeline any more.

## Run 6 — 2026-09-05 ~12:50 — the kernel profile of the real Chrome path

ORT's WebGPU profiler (timestamp queries) hooked into the demo's own engine in `smoke-tts.mjs`
(`PROFILE=1`; the module is the CDN one, so setting `env.webgpu.profiling` on it before
`OmniVoice.load` reaches the session). Two dead ends first: `bench-webgpu.mjs` reported 0 kernels
because its self-made device lacked `timestamp-query`, and — more important — it measured
3.3 s per forward against the smoke's 1.09 s on what should be the same GPU: its adapter reports
maxBufferSize 2147 MB where Run 8 recorded 4295 MB for Chrome/Dawn on the NVIDIA card, so that
bench is not on the device the demo gets. Left as a known-bad tool for now; the smoke is the
instrument.

    generate 43.3s  (transformer 38.0s, host 3.5s)   [profiling costs ~3 s over Run 4's 40.2 s]
    117,888 kernels, 15.4 s GPU time per generation → per forward: 3,684 kernels, 480 ms GPU,
    1.19 s wall → 0.7 s per forward NOT in kernels

    per forward, top kernels by GPU time:
       403.9 ms    394×  MatMulNBits     ← 84% of GPU time
        25.5 ms    114×  MatMul          (the attention QK/PV products)
         9.1 ms    620×  Mul
         8.0 ms    506×  Add
         5.1 ms     60×  Where
         4.7 ms    286×  Transpose
         3.3 ms     56×  Softmax
         2.6 ms    226×  Pow / Div / ReduceMean / Sqrt  (RMSNorm, ×4 kernels each)

Two conclusions, both load-bearing:

1. **60% of a forward is dispatch, not compute.** 3,684 dispatches at Dawn's ~0.19 ms each is
   0.7 s — the "not in kernels" figure to within noise. Kernel count is the first lever, and the
   RMSNorm fusion (Run 5) removes ~1,600 of the 3,684.
2. **The matmuls are running at ~7% of the card.** 404 ms for ~1 TFLOP of int4 weight-matmul at
   B·S = 860 tokens is ~2.5 TFLOPS on a 35-TFLOPS fp32 part; CUDA does the whole forward in
   87 ms. ORT-web's `MatMulNBits` has a faster path for fp16 activations (packed math, the
   DP4A/subgroup variants), which this fp32-activation build never reaches. fp16 activations are
   the second lever — and the one with the precision question from the TF32 finding attached.

Attention itself (MatMul + Softmax + Transpose ≈ 35 ms) is not the problem; fusing it would be
worth doing only for its dispatch count.

## Run 7 — 2026-09-05 ~13:20 — the fused build: 1,240 fewer dispatches bought 6%

Same text, Chrome WebGPU, 32 steps, profiler on:

    v7 (shipped):  generate 43.3s (transformer 38.0s)  per forward 3,684 kernels, 480 ms GPU, 1.19 s wall
    v7-fused:      generate 41.1s (transformer 35.8s)  per forward 2,442 kernels, 462 ms GPU, 1.12 s wall
                   LayerNormalization 114× 3.0 ms, SkipLayerNormalization 112× 2.5 ms — the
                   fusion landed and the RMSNorm cost went from ~10 ms to ~5.5 ms per forward.

So Run 6's conclusion 1 was WRONG in its arithmetic: removing a third of the dispatches removed
~6% of the wall time, which puts Dawn's per-dispatch cost near 0.03 ms, not 0.19. The 0.66 s per
forward that is not kernel time is something else: ORT's CPU-side work in wasm (the ~1,900
Shape/Gather/Concat/Unsqueeze nodes run on the CPU, plus uniform packing and buffer management)
and the per-step GPU→CPU readback of the [2,8,S,1025] fp32 logits (28 MB at S=430) with the sync
it forces. Attention fusion and graph capture would still help, but they are the smaller lever;
the readback and the MatMulNBits kernel (397 ms, unchanged) are the larger ones.

The fp16 timing that followed hung: node waited on a page that never reported, and no Chrome
process was alive — the browser died on the fp16 build. Diagnosed next with the one-forward
logit tool, which reports the page's error instead of waiting.

## Run 8 — 2026-09-05 ~14:10 — fp16: built, runs on WASM, cannot be measured on this adapter

Tooling first, because it changed what was findable. The http+spawn harnesses report only at the
end and hang silently on a dead page; three hours went to timeouts. Replaced for this work by
`tools/pw-tts.mjs` (Playwright driving the system Chrome, headed on :0 with the same flags; the
page's console, errors and progress stream live; a deadline instead of a timeout) and
`tools/pw-bench.mjs` (the same, for N forwards with session options from env). The demo also now
logs a lost WebGPU device to the console instead of hanging on the next run.

The fp16 build (`v7-fused-f16/`, 424 MB: activations fp16 via ORT's converter, int4 weights and
the int8 embedding untouched):

- Runs on WASM in Node (993 ms/forward at S=64) — the graph is valid.
- On WebGPU the session creates in 2 s and the first forward raises
  `Error while parsing WGSL: 'f16' type used without 'f16' extension enabled`, then a cascade of
  invalid pipelines, and the loop "completes" on whatever the failed kernels left behind.
- The adapter here (Chrome 152, ANGLE/Vulkan, RTX 3090) offers `subgroups`,
  `subgroup-size-control`, `timestamp-query`, `chromium-experimental-subgroup-matrix` — and NOT
  `shader-f16`. ORT-web requests f16 itself when the adapter has it (`i("shader-f16")` in the
  1.29 bundle), so the build should work on an adapter that offers it (Metal on a Mac, most
  likely) and is untestable on this one.
- ⚠ ORT-web 1.29 creates its own device and IGNORES `env.webgpu.device`: the device in use had
  ORT's own feature set, not ours. `useMaxLimitsDevice` has been a no-op since the move to the
  native WebGPU EP; it works because the largest tensor (155 MB) fits the default 268 MB limit.

Like-for-like on the same 190-token text under the new harness (the library's default `en`
voice, as the C# runs): fp32-fused **7.3 s of audio in 26.2 s, 0.28× real time, 694 ms per
forward at S≈350**. (The 10 s/40 s figures earlier in this log were the smoke tool's longer
voice; ratios agree.)

## Run 9 — 2026-09-05 ~14:40 — readback, capture, static shapes: three suspects, three acquittals

`pw-bench.mjs`, fused fp32 build, B=2, Chrome/Dawn on the RTX 3090:

    S=350  dynamic graph                          647 ms/forward
    S=350  + preferredOutputLocation: gpu-buffer  627 ms   → the 22 MB logits readback is ~20 ms
    S=512  dynamic graph                         1083 ms
    S=512  static graph, ORT offline BASIC       1124 ms   → the 2,300 CPU-side shape nodes cost nothing
    S=512  static graph, ORT offline EXTENDED    2684 ms   ← EXTENDED inserts FusedMatMul (28×), which
                                                            the WebGPU EP does not run; do not use it
    enableGraphCapture                            refused: "not all compute graph nodes have been
                                                  partitioned to the JsExecutionProvider", on the
                                                  dynamic AND the static graph

The static graph (`make_dim_param_fixed` two_b=2, seq=512, then ORT BASIC: 3,633 → 1,502 nodes,
zero Shape ops) was built for capture and to test the CPU-shape-ops theory; it disproves the
theory and does not unlock capture. Something ORT still places on the CPU; not chased further,
because with dispatch count (Run 7), readback and shape ops all measured small, the non-kernel
~0.25 s per forward is ORT-web's own per-kernel JS/Dawn bookkeeping, and capture is the only
thing that attacks it.

## Run 10 — 2026-09-05 ~14:50 — where this leaves the browser, and the one lever left

Per forward at S≈350 on this GPU: ~400 ms is `MatMulNBits`, ~250 ms is everything else. The
fast `MatMulNBits` paths in ORT-web (DP4A, subgroup-matrix) and the fp16 build all require
`shader-f16`, which Chrome on this Linux/Vulkan adapter does not expose under any flag tried
(`--enable-dawn-features=allow_unsafe_apis`, `WebGPUDeveloperFeatures`, `--use-angle=gl`). A
Mac's Metal adapter, and most Windows/D3D12 ones, do expose it. So the fp16 build is the
experiment — it just has to be run on a machine that can run it, and judged by ear there,
because the TF32 finding says the diffusion loop may not tolerate 10-bit mantissas.

Ceiling estimate for Chrome WebGPU with fp32 activations on this class of GPU: ~0.3× real time
(fusion gets 0.25 → 0.28). With fp16 on an adapter that has it: unknown, plausibly 2× on the
matmul share → ~0.5×. 1× needs the matmul kernel AND the per-kernel overhead to both halve, or a
model with fewer steps — the distillation item in the shelved plan. 2.5× is the native-CUDA
number and is not available through WebGPU as shipped.
