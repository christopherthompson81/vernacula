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
per-op dispatch through Dawn, and 64 forwards (2 CFG passes × 32 steps) each re-uploading the
conditioning. Per forward: 34.8 s / 64 = 544 ms at S≈430 on WebGPU vs ~40 ms on CUDA. Run 8's
177 ms at S=100 → 544 ms at S=430 says the WebGPU path scales roughly linearly with sequence
length, i.e. attention and the dequant-matmuls dominate, not fixed overhead.
