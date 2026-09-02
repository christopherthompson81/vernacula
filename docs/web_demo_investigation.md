# Browser demo — investigation log

Can `vernacula-tts` run entirely client-side (React + `onnxruntime-web`, static-hosted on Netlify)?
The desktop model is 2.45 GB of fp32 ONNX, so the whole question is how far it compresses before the
diffusion loop stops producing speech.

Related: `docs/omnivoice_onnx_investigation.md` (the precision-sensitivity findings this builds on),
`docs/vernacula_tts_investigation.md` (the CLI), `web-demo/README.md` (architecture).

Reference implementation for the browser side: `parakeet_csharp/demo/web-demo` — React 18 + Vite +
`onnxruntime-web`, ~500 MB INT8 models, chunked/resumable Cache-API model store, worker-hosted ORT
sessions, multi-file external data. Proven pattern; reuse rather than reinvent.

---

## Run 1 — 2026-09-01, the size problem, and what the ladder looks like

**Question:** what has to be true for this to work at all?

**Constraints, established up front:**

- Netlify is static hosting. Its functions cap at ~10 s and 50 MB, so nothing model-shaped runs
  server-side. Either it runs in the browser or there is a backend somewhere else.
- The transformer is 2451.8 MB fp32. The Higgs encoder is another 654 MB, the decoder 86 MB.
- ⚠ **The diffusion loop is precision-sensitive**, which is the crux and was already measured:
  CUDA TF32 gave incoherent noise, fp16 a different-but-valid rendering, only fp32 was faithful
  (`omnivoice_onnx_investigation.md`). Iterative unmasking amplifies matmul error. Whatever is true
  of Qwen3 in isolation — and Qwen3 quantizes well — this is Qwen3 inside 32 iterations of that loop.

**Two decisions that fall straight out of existing findings, before any code:**

1. **The 654 MB encoder never ships.** It exists only to turn a reference WAV into codec codes.
   Those codes are a few KB, so they are precomputed offline into `voices.json`.
2. **Generation is always voice-cloned.** Not a stylistic choice: `vernacula_tts_investigation.md`
   Run 7 established that with no reference voice, input under ~5 s is out of distribution (corpus
   median 12 s, 0.21% under 3 s) and can emit noise rather than degrade. A demo receives short
   phrases almost exclusively. Shipping precomputed reference codes fixes the failure *and* drops
   654 MB — the same decision serves both.

---

## Run 2 — 2026-09-01, INT8 dynamic quantization — FAILED

**Command:** `quantize_dynamic(src, dst, weight_type=QInt8, per_channel=True, reduce_range=False)`
on the merged v6 IPA transformer.

**Result:** 2451.8 MB → **616.8 MB** in 34 s, a clean 4x, single file (under the 2 GB protobuf
limit, so no sidecar needed). Generation was also *faster* than fp32 on CPU (29.5 s vs 36.2 s for
~5 s of audio).

**Verdict — NOT USABLE. "Not recognizable as speech — too degraded."** (User listening test, INT8
vs fp32 on identical text with the same reference voice, so the short-input effect could not
confound it.)

⚠ **The screening metric called it speech, and was wrong again.** The 2-8 Hz syllable-band envelope
measure scored INT8 at 0.351 against fp32's 0.358, with earlier confirmed noise at ≤0.293 — i.e.
comfortably on the speech side of the boundary it was calibrated on. That is its **second** false
positive (the first: three of ten cloned clips in `vernacula_tts_investigation.md` Run 7). Two
misses in two novel regimes is enough. **The metric is demoted to a triage hint and is not a gate
for anything.** Quantization verdicts are listening tests.

**Diagnosis — probably the wrong KIND of INT8.** `quantize_dynamic` quantizes **activations** to
int8 at runtime as well as weights. In a loop that runs 32 times over its own output, activation
error is exactly what compounds. The finding that TF32 — a far milder perturbation than int8
activations — already produced noise supports this reading directly.

**Next:** weight-only block-wise quantization (`MatMulNBitsQuantizer`), which compresses only the
weights, in small blocks each carrying its own scale, and leaves **every activation in fp32**. It is
a categorically gentler operation and the one transformers.js uses to run LLMs in browsers. Testing
int8 and int4 at block 32.

**The ladder if weight-only also fails:**

| rung | size | quality evidence |
|---|---|---|
| INT8 dynamic | 617 MB | ✗ FAILED — not speech |
| weight-only int4 / int8 | ~450-750 MB | testing |
| fp16 | ~1.2 GB | listen-confirmed good on CUDA — but see the WASM caveat |
| fp32 | 2.45 GB | reference |

⚠ **fp16 is a WebGPU rung, not a WASM one.** ORT's WASM kernels are fp32/int8; an fp16 graph there
buys download size but not speed, and casts back up internally. On WebGPU (`shader-f16`) it is
native. So "fp16 works" and "fp16 works in this demo" are different claims.

**Sidecar splitting is available and not a blocker** (user's suggestion, confirmed): anything above
the 2 GB single-file protobuf limit can be split, because `onnxruntime-web` accepts
`sessionOptions.externalData` as an **array** of entries. The Parakeet demo already carries that
worker plumbing (`ExternalDataEntry[]`).

---

## Run 3 — 2026-09-01, weight-only quantization, and where the size actually is

**Two changes from Run 2**, one methodological and one from the user:

1. **Weight-only, not dynamic.** `MatMulNBitsQuantizer` compresses weights in blocks of 32, each
   with its own scale, and leaves every activation in fp32 — unlike `quantize_dynamic`, which also
   quantizes activations at runtime. In a loop that runs 32 times over its own output, activation
   error is the term that compounds.
2. **Hold the audio heads at fp32** (user's hypothesis). Their logits over the 1025-token audio
   vocabulary drive the top-k unmask decision on every step, so error there flips token choices
   outright instead of being smoothed downstream. There is exactly ONE such node,
   `/model/audio_heads/MatMul`, and at ~8.4M params it is 34 MB fp32 — exempting it is nearly free.
   ⚠ The first weight-only pass quantized it (`/model/audio_heads/MatMul_Q4`), so the hypothesis was
   untested until `nodes_to_exclude` was set.

**Sizes** (quantization itself takes ~3 s):

| variant | size | vs fp32 |
|---|---|---|
| fp32 | 2451.8 MB | — |
| w8 block32, heads quantized | 1161.5 MB | 2.1x |
| w8 block32, **heads fp32** | 1185.7 MB | 2.1x |
| w4 block32, heads quantized | 937.1 MB | 2.6x |
| w4 block32, **heads fp32** | 965.5 MB | 2.5x |

Exempting the heads costs 24-28 MB. Whatever it buys in quality, it is cheap.

**⚠ THE EMBEDDING IS TWO-THIRDS OF THE COMPRESSED MODEL, AND NOTHING TOUCHED IT.**
`model.llm.embed_tokens.weight` is 151676 x 1024 fp32 = **621 MB**, and it is consumed by a
`Gather`, not a `MatMul` — so `MatMulNBits` skips it by construction. That is why int4 only reaches
937 MB rather than the ~350 MB the linears alone would suggest: 621 MB of the total is an untouched
embedding table. The linears compress from 1.72 GB to ~240 MB; the embedding does not move at all.

So the size ladder has a second dimension nobody has tested yet. If weight-only passes the listening
test, the next lever is the embedding — fp16 would take it to 310 MB (w4 total ~650 MB), int8 to
155 MB (~500 MB). ⚠ Worth noting the IPA fine-tune specifically retrained 5,572 embedding rows, so
this table is not obviously the robust, quantization-tolerant component it usually is.

**Pending:** listening test over the 2x2 (w4/w8 x heads-fp32/heads-quantized) plus an fp32 control,
all generated with the same text and reference voice so neither the short-input effect nor speaker
variation can confound. ⚠ The envelope metric is deliberately NOT reported for these — it has been
wrong in both novel regimes it was applied to and is no longer used as a gate.

---

## Run 4 — 2026-09-01, weight-only PASSES, and the embedding comes down

**Listening verdict on Run 3's 2x2: "All the outputs are fine."** Both widths, both head
treatments. Two conclusions:

1. **The Run 2 diagnosis was correct.** Weight-only survives all the way to **int4**, while INT8
   *dynamic* was unlistenable. The variable that mattered was never the weight width — it was
   whether ACTIVATIONS were quantized. In a 32-iteration loop over its own output, activation error
   compounds and weight error does not. This is consistent with the earlier TF32 result: TF32
   perturbs activations too.
2. **The audio-heads exemption is NOT needed** — `w4_headsquant` was fine. The hypothesis was
   reasonable and cheap to test, and it came back negative; keeping the exemption would have been
   28 MB of cargo cult. Dropped.

So the working configuration is **w4 block-32, weight-only, nothing exempted**, at 937 MB.

**Then the embedding.** 621 MB of that 937 MB is `model.llm.embed_tokens.weight`, untouched because
it feeds a `Gather`. Both variants rewrite the graph so dequantization happens on the GATHERED SLICE
(a few rows) rather than the whole table:

| variant | graph | total | vs fp32 |
|---|---|---|---|
| emb fp32 (baseline) | — | 937.1 MB | 2.6x |
| emb **fp16** | `Gather(fp16) -> Cast(fp32)` | **626.5 MB** | 3.9x |
| emb **int8** | `Gather(int8)`, `Gather(scale)`, `Cast`, `Mul` | **471.8 MB** | 5.2x |

int8 uses **per-row** scales, not per-tensor, deliberately: the IPA fine-tune retrained 5,572 rows
of this table, so it is not the uniformly-distributed tensor an embedding usually is and a single
global scale would be the wrong bet.

Both generate. Pending a listening test.

**Projected bundle** at w4 + emb-int8: 472 MB transformer + 86 MB Higgs decoder + a few KB of
precomputed voice codes = **~558 MB**, against the Parakeet demo's ~500 MB — a size already shown to
work in a browser. The 654 MB Higgs encoder never ships (Run 1).

**⚠ Not yet measured: SPEED, which is a separate risk from size.** Native CPU is ~36 s for 5 s of
audio at 32 steps. WASM will be slower, and the diffusion loop runs the transformer 32 times per
generation — this is the one workload shape that cannot be streamed or partially rendered. Levers:
`NUM_STEPS` 32 -> 16 (the Phase-1 work used 16), and the WebGPU EP. Needs a browser harness to
measure; a demo that takes three minutes per phrase is a different product from one that takes
twenty seconds.

---

## Run 5 — 2026-09-01, speed: WASM is usable, and WebGPU is (so far) SLOWER

**Listening verdict on Run 4: both embedding variants fine.** So the model config is locked:
**w4 block-32 weight-only + int8 per-row embedding = 471.8 MB**, a 5.2x reduction with no audible
cost. Projected bundle ~558 MB with the Higgs decoder.

Size was never the only risk, though. The diffusion loop runs the transformer 32 times per
generation and cannot be streamed or partially rendered — the user watches a spinner for the whole
thing.

**WASM (`onnxruntime-web` in Node, same wasm build the browser loads):**

| threads | S | ms / forward | 16 steps | 32 steps |
|---|---|---|---|---|
| 4 | 200 | 3798 | 60.8 s | 121.5 s |
| 8 | 200 | 2370 | 37.9 s | 75.8 s |
| 4 | 100 | 1981 | 31.7 s | 63.4 s |
| **8** | **100** | **1295** | **20.7 s** | 41.4 s |

S=100 is a short demo phrase; S=200 a long sentence. So on 8 threads with 16 steps a typical phrase
is ~20 s — slow but a usable demo. Thread count needs cross-origin isolation, which `netlify.toml`
already sets; without those headers the browser withholds SharedArrayBuffer and this silently
becomes the 1-thread column.

**WebGPU — measured, and it is worse here.**

Getting a measurement at all took three corrections, all self-inflicted:
- First attempt loaded the 472 MB model straight into headless Chrome and was SIGKILLed after 15
  minutes. ⚠ **Test the cheap failure first.** A 76 KB model carrying only the ops in question
  (`MatMulNBits`, `Gather`, `Cast`, `Mul`) answers the support question in seconds.
- Chrome headless is unusable for this on this box: `navigator.gpu` is present but
  `requestAdapter()` returns NULL and the GPU process segfaults (exit 139); with
  `--enable-features=Vulkan` it sometimes yields a **SwiftShader** (software) adapter, whose timings
  would be meaningless.
- **Firefox headless works** (prefs `dom.webgpu.enabled`, `gfx.webgpu.force-enabled` in a fresh
  profile — there is no command-line switch). ⚠ Use `mkdtemp` for the profile: Firefox leaves
  IndexedDB directories that `rmSync` trips over with ENOTEMPTY on the next run.

Op support and limits are fine:

    adapter limits: maxBufferSize=2147 MB   maxStorageBufferBindingSize=2147 MB
    shader-f16: supported
    tiny model on webgpu EP: MatMulNBits + the embed Gather/Cast/Mul path both run, finite outputs

And the real model runs — VRAM rose 1841 -> 2290 MiB (+449 MB, matching the model), so the weights
genuinely are resident on the GPU, not silently on CPU. But:

    WebGPU (Firefox), S=100:  3011 ms / forward  =>  48.2 s per generation @ 16 steps
    WASM (8 threads), S=100:  1295 ms / forward  =>  20.7 s per generation @ 16 steps

**⚠ THAT COMPARISON WAS A SINGLE POINT, AND IT WAS MISLEADING — see Run 6.** At S=100 WebGPU is
2.3x slower, but the number does not generalise: WebGPU's time turns out to be FLAT in sequence
length, so the ranking inverts as the utterance grows.

**Harness** (kept, both are the record of the above): `web-demo/tools/bench-wasm.mjs`,
`web-demo/tools/bench-webgpu.mjs` (`BROWSER=firefox|chrome`), `web-demo/tools/probe-webgpu.mjs`
(fail-fast adapter + op probe; takes an optional tiny model).


---

## Run 6 — 2026-09-01, the flat curve: WebGPU is overhead-bound, not slow

**Prompted by the user:** "initializing CUDA has some overhead — try running a longer inference."
Correct instinct, and it overturned Run 5's conclusion.

**First, ruling out warm-up contamination.** With 2 warm-ups and 6 timed runs the series is flat:
warm-up 3437, 3006; steady min 3004, mean 3014, spread 3004-3032 ms. Initialization costs ~430 ms,
on the first run only. So the 3 s is not init overhead.

**Then the scaling curve, which is the real finding:**

| S | WASM (8 threads) | WebGPU (Firefox) |
|---|---|---|
| 100 | 1295 ms | 3014 ms |
| 200 | 2370 ms | 3036 ms |
| 400 | **4794 ms** | **3008 ms** |

WASM scales with work. **WebGPU does not move at all** — 3008-3036 ms across a 4x range of sequence
length. Crossover is around S=300; by S=400 WebGPU is 1.6x FASTER. So "WebGPU is slower" was an
artifact of measuring one short sequence, and Run 5's headline is corrected accordingly.

**What the flat time is.** Not a universal stall: the 76 KB tiny model runs in **200 ms** on the
same path, so there is a ~200 ms per-run floor but the remaining 2.8 s is model-dependent. Constant
in S but scaling with model size is the signature of **per-run weight traffic** — roughly
(3010-200)/472 MB ≈ 6 ms/MB, or ~170 MB/s, which is upload-shaped rather than compute-shaped.

**The third data point could not be taken, and that is its own finding.** Re-running at 626 MB
(`emb-fp16`) and 1161 MB (`w8`) both die:

    GPU DEVICE LOST: reason=unknown msg=Out of memory

on a card with ~23 GB free, and after the adapter reports `maxBufferSize = 2147 MB`. **Firefox's
WebGPU has an effective ceiling far below what it advertises** — 472 MB loads, 626 MB does not. Two
consequences: the size-proportionality hypothesis stays unconfirmed, and the 472 MB w4+emb-int8
build is not merely the preferred variant on Firefox, it is the ONLY one that loads.

**⚠ A self-inflicted detour, recorded so it is not repeated.** Several runs in this sequence failed
with "Not enough memory left" at session creation, and the user noticed VRAM was not rising. Cause:
the harness called `process.exit()` without reaping the spawned browser, so **every run leaked a
GPU-holding headless process** — ten had accumulated. Fixed (`CHILD` + `done()` + exit handlers in
both tools). ⚠ Also: `pkill -f <pattern>` matched the cleanup shell's OWN command line and killed
it (exit 144); patterns for self-cleanup must be assembled at runtime.

**Where this leaves the decision.** For a demo, typical S is 100-200 (a short phrase), which is the
region where WASM wins — so **WASM with cross-origin isolation remains the default**. But WebGPU is
overhead-bound rather than compute-bound, which means the headroom is real and mostly a matter of
whether the implementation avoids per-run weight traffic. Chrome + Dawn, which is what ORT's WebGPU
work targets and where that traffic is most likely already avoided, is still unmeasured here
(Chrome headless yields no adapter on this box). That is the one measurement that could still
change the architecture, and it needs a real browser session.

---

## Run 7 — 2026-09-01, the per-buffer cap: real, raisable, and not the problem

**Prompted by the user:** "probably some cap that can be altered." Half right, and worth the check —
the cap is real and it is alterable, but it turns out not to be what was binding.

**The cap is real.** WebGPU's `requestDevice()` grants DEFAULT limits unless you ask for more,
whatever the adapter advertises:

    default device: maxBufferSize=268 MB   maxStorageBufferBindingSize=134 MB
    adapter max:    maxBufferSize=2147 MB  maxStorageBufferBindingSize=2147 MB
    RAISED device:  maxBufferSize=2147 MB  maxStorageBufferBindingSize=2147 MB   <- granted

And the correlation with the OOMs was exact — every failing variant has ONE tensor over the default:

| variant | largest tensor | loads on Firefox WebGPU? |
|---|---|---|
| w4 + emb-int8 | **155.3 MB** `embed_tokens` | ✅ |
| w4 + emb-fp16 | 310.6 MB | ❌ device lost, OOM |
| w4 / w8 + emb-fp32 | 621.3 MB | ❌ device lost, OOM |

ORT exposes `env.webgpu.device`, so a device built with the adapter's maximum can be handed to it
rather than letting ORT take defaults.

**But raising it fixed nothing.** With `maxBufferSize=2147 MB` in force:

- the 626 MB emb-fp16 build **still** dies with `GPU DEVICE LOST: Out of memory`, so Firefox has a
  separate practical ceiling somewhere between 472 and 626 MB that is not the per-buffer limit;
- timing is unchanged — S=100: 3010 ms, S=400: 3039 ms, still flat.

**Net.** The per-buffer default is a genuine trap worth knowing about and worth setting correctly in
the demo regardless (it costs nothing and removes a whole class of failure on other machines). It is
not the explanation for either symptom here. The ~3.0 s constant — against a 200 ms floor for a
76 KB model — remains an unexplained characteristic of Firefox's WebGPU + ORT rather than anything
about our graph, and the size ceiling is Firefox's own.

**Decision, on the evidence available.** Ship **WASM with cross-origin isolation** as the default:
it scales with work, and at the S=100-200 a demo actually sees it is the faster path (1295 ms vs
3010 ms per forward). Keep the w4+emb-int8 build at 472 MB, which is both the only variant Firefox's
WebGPU will load and the only one under the default per-buffer cap — so it is the most portable
choice even setting size aside. Treat WebGPU as opt-in, with limits raised, and measure it on
Chrome + Dawn before promoting it: that configuration is where ORT's WebGPU work is targeted, it
could not be measured on this box, and a flat-in-work profile means the headroom there is real.

---

## Run 8 — 2026-09-01, Chrome/Dawn on the real GPU: 2.8 s per generation

**The configuration that mattered, finally measured.** Runs 5-7 concluded WASM should be the default
because the only WebGPU implementation reachable here was Firefox's. That conclusion is now
**superseded**: on Chrome/Dawn with the actual NVIDIA GPU, WebGPU is not marginally better, it is
7x faster than WASM.

| path | ms/forward @S=100 | 16 steps | @S=400 |
|---|---|---|---|
| WASM, 8 threads | 1295 | 20.7 s | 4794 ms |
| WebGPU, Firefox | 3014 | 48.2 s | 3008 ms (flat) |
| **WebGPU, Chrome/Dawn, NVIDIA** | **177** | **2.8 s** | **622 ms** |

And it **scales with work** — 177 -> 622 ms for a 4x sequence — so it is compute-bound. Firefox's
perfectly flat 3008 ms across S=100..400 was an implementation artifact of Firefox's WebGPU, not a
property of the model or of WebGPU. Chrome also grants a far larger buffer limit on request
(4295 MB vs Firefox's 2147 MB).

**Getting there took two environment fixes, neither of them about WebGPU.**

1. **Chrome was pinned at 131 (Nov 2024) because the noble dist-upgrade disabled Google's apt repo**
   (`google-chrome.sources` had `Enabled: no`; `apt policy` listed only `/var/lib/dpkg/status`).
   Re-enabling it then failed with `NO_PUBKEY FD533C07C264648F` — the key block embedded inline in
   that file is a 2016 snapshot of the SAME primary key (7721F63BD38B4796) that predates the signing
   subkey the repo now uses. Fix: install the current key from
   `https://dl.google.com/linux/linux_signing_key.pub` into `/etc/apt/keyrings` and point the source
   at it with `Signed-By:` instead of an inline block. -> 152.0.7977.64.
2. **Version was not actually the blocker.** Chrome 152 headless still returned SwiftShader. The
   working combination is **HEADED + `--ozone-platform=x11` + `--enable-features=Vulkan
   --use-angle=vulkan`** on `DISPLAY=:0`. Headless Chrome (131 and 152 both) returns SwiftShader or
   NULL; with `--use-angle=vulkan` headless the GPU process segfaults (exit 139). ⚠ A headed run
   also needs `--no-first-run --no-default-browser-check --disable-search-engine-choice-screen`, or
   the setup dialog blocks navigation and the run just times out — which it did, and cost a cycle.

Also worth recording: **snap chromium can never work for this.** Its GPU access is
`content[gpu-2404] -> mesa-2404`, i.e. bound to the mesa driver content snap, with no route to the
NVIDIA userspace driver. It reports SwiftShader regardless of version or flags.

**Revised decision.** Ship **WebGPU as the primary path** with WASM as the fallback — the reverse of
Run 7. 2.8 s for a short phrase is a good demo; 20.7 s is a tolerated one. Request raised device
limits explicitly (Run 7: the 268 MB default is what kills large-tensor models) and keep the
w4+emb-int8 build, whose 155 MB largest tensor clears even the default cap.

⚠ **Caveat on generality.** These numbers are one machine, one GPU, one browser. A visitor on
Firefox gets the 48 s path, and one without WebGPU at all gets 20.7 s on WASM — so the fallback is
not decorative, and the UI needs to say which path it is on rather than silently being 17x slower.

---

## Run 9 — 2026-09-01, the phonemizer runs in the browser, and its IPA is byte-identical

Upstream implemented #1245 (and #1247, publishing the asset tree as `vernacula-phonemizer-data`).
Submodule moved 2855070f -> 34541a5b. C# port unaffected: solution builds, parity gate still
**4 languages byte-identical, 800 rows**.

**The seams, as shipped:** `setDataSource({read(key): Uint8Array})` — synchronous, as the issue
insisted — plus `setOrtLoader(() => import("onnxruntime-web"))`, both re-exported from a new
`src/browser.ts` whose `loadEngine()` hides the engine behind a dynamic import. That indirection is
load-bearing: `registry.ts` reads 182 manifests at MODULE SCOPE, so a static import would run them
before the consumer's `setDataSource()` call. Upstream also built the recording tool
(`tools/browser-prefetch.mts`) rather than a declared manifest, for the reason the issue gave — a
missing optional table is not an error to the engine, it is an empty Map and a plausible wrong
reading.

**⚠ The one constraint that shapes the build: keys come from `import.meta.url`.** `dataPath.ts`
slices after the last `/src/`, so a bundler rewriting module URLs to chunk names erases the only
thing naming the data (it throws rather than guessing). That rules out letting Vite bundle the
engine. Solution: transpile per-file with esbuild (`bundle: false`) into `public/vphon/src/`,
preserving the tree, and load it with a dynamic import of an ABSOLUTE URL, which Vite leaves alone.
`import.meta.url` is then `/vphon/src/languages/welsh/welsh.js` -> key `languages/welsh`. 715
modules, 4.7 MB of JS; esbuild leaves specifiers verbatim so the explicit `.ts` extensions are
rewritten to `.js` afterwards. Verified: no `.ts` specifiers and **no `node:` specifiers** anywhere
in the output.

**Payload, recorded not guessed:**

| phase | files | size |
|---|---|---|
| engine (needed for any language) | 182 | 4.46 MB |
| + `es` | 0 | 0 |
| + `cy` | 1 | 0.04 MB |
| + `en` | 7 | 14.59 MB |

English is the outlier because of the BiLSTM and its lexicons. `es`+`cy`+`en` together stage 190
files / 19.1 MB — against 151 MB for the whole tree.

**Cross-engine verification, which is the point of the run.** Running in headless Firefox through
both seams, against the same sentences the C# CLI rendered earlier:

    [es] bwˈenos ðˈias . el tjˈempo estˈa mˈuᶦ aɣɾaðˈaβle ˈoᶦ .        (8 ms)
    [cy] bˈɔrɛ dˈaː . krˈɔᶤsɔ ˈiː ɡˈəmrɨ .                              (12 ms)
    [en] həlˈoᶷ wˈɝɫd . ðɪs ɪz ðə vɚnˈækjələ tʰˈɛkst tʰuː spˈiːt͡ʃ pʰˈaᶦplaᶦn .  (501 ms)

**Byte-identical to the C# CLI output on all three.** So TypeScript-in-browser, the C# port, and the
engine that phonemized the training corpus all agree. Prefetch 152 ms, `loadEngine()` 517 ms; the
501 ms for English is the first call pulling the BiLSTM through ORT, not per-phrase cost.

Kept: `tools/build-phonemizer.mjs`, `tools/stage-phonemizer-data.mjs` (wraps the upstream recorder,
one child process per language), `tools/smoke-phonemizer.mjs` (the cross-engine check above).
`public/vphon/` and `public/vphon-data/` are generated, and gitignored.

**Remaining for the demo:** the diffusion loop in TypeScript (a port of
`Chatterbox.Base.OmniVoiceTts.RunDiffusion`) and the React UI. The graphs themselves are just
`session.run`.

---

## Run 10 — 2026-09-01, the pipeline ported, and cross-checked against C# by number

The diffusion loop and everything around it are now TypeScript: `qwen3Tokenizer.ts` (byte-level
BPE), `textPrep.ts`, `duration.ts`, `diffusion.ts`, `audioPost.ts`, `omnivoice.ts`. Ports of
`Chatterbox.Base.{Tokenization.Qwen3Tokenizer, OmniVoiceTextPrep, OmniVoiceDuration, OmniVoiceTts,
OmniVoiceAudioPost, OmniVoice}`.

**Two ports needed care rather than transcription:**

- The Qwen split regex opens `(?i:'s|'t|…)` and **JavaScript has no inline group modifiers**, so
  the contractions are spelled out case-by-case. Everything else is identical under `u`.
- `OmniVoiceDuration` switches on .NET `UnicodeCategory`; the JS equivalent is property escapes
  (`\p{Mn}\p{Mc}\p{Me}` -> mark, `\p{Pc}…\p{So}` -> punctuation, `\p{Zs}\p{Zl}\p{Zp}` -> space,
  `\p{Nd}\p{Nl}\p{No}` -> digit), which map one-to-one.

The C# parallelises CFG scoring across (codebook, position) — three 1025-way softmaxes per slot,
the host hot path. JS is single-threaded, so it runs sequentially over reused buffers. Measured
host cost 0.9 s of an 11.2 s generation, so it is not currently worth a worker pool.

**Voice codes are precomputed** (`tools/make-voices.mjs`), mirroring `EncodeReference` exactly —
RMS-boost, remove silence at the REFERENCE parameters (mid 200 / lead 100 / trail 200, not the
output chain's 500/100/100), clip to a hop multiple, encode. Output: **2.8 KB of JSON in place of
the 654 MB encoder**, and it is what lets every generation be voice-cloned, which is what keeps
short input stable.

**Cross-check against the C# CLI, same text, same int4 model, same reference clip:**

| | TypeScript (browser, WebGPU) | C# (CLI, CPU) |
|---|---|---|
| IPA | `ðə kwˈɪk bɹˈaᶷn fˈɑːks…` | identical |
| reference codes | 82 | 82 |
| **target tokens** | **134** | **134** |
| audio length | 5.38 s | 5.3 s |
| wall clock | 11.2 s (transformer 9.8, host 0.9) | 18.7 s |

`targetTokens` depends only on the tokenizer, text prep and duration estimator, so agreement there
means those three ports are exact — a stronger statement than "it produced audio". The diffusion
field itself will differ slightly between execution providers, which is why the audio is compared
by ear rather than diffed.

**Production build verified, not just dev.** Serving `dist/` in a browser:

    phonemizer from built site: həlˈoᶷ wˈɝɫd .     (matches C#)
    crossOriginIsolated: true                      (COOP/COEP -> SharedArrayBuffer -> WASM threads)
    HF ranged fetch: HTTP 206, 1024 bytes          (cross-origin + Range under COEP)

All three were real risks: Vite could have broken the phonemizer's `import.meta.url` keys, missing
isolation would have silently dropped WASM to one thread, and a CORS failure would have made the
model unfetchable from the site's origin.

⚠ **ORT's wasm sidecars are served from jsDelivr, not bundled.** Shipping all four variants put
~100 MB in `dist/` (131 MB -> 52 MB after the change) for files the CDN serves with the CORS headers
COEP needs. This also keeps ORT's dynamic import of its threaded `.mjs` sidecar out of Vite's module
pipeline, which cannot resolve a dynamic import of a file in `public/`.

`tokenizer.json` (11 MB) was missing from the HF repo — a browser has nowhere else to get it — so it
is published there now alongside the model.

---

## Run 11 — 2026-09-01, int4 has quirks, and the EP moves the output too

**⚠ CORRECTION TO RUN 4.** Run 4 recorded "all the outputs are fine" for the 2x2 and for both
embedding variants, and locked w4+emb-int8 on that basis. On further listening the user reports the
int4 build "has quirks — it might be quantized too much". The earlier verdict came from **one
sentence per variant**; quantization damage that a single short clip hides is exactly the failure
mode a one-shot listening test has. Treat Run 4's approval as provisional and superseded here.

**Two independent effects were being conflated, and they are now separated:**

1. **The execution provider changes the output.** Identical input, identical model, same port:

       WebGPU (Chrome/Dawn)  peak 0.3799  rms 0.0488   generate 10.0 s
       WASM   (fp32)         peak 0.4840  rms 0.0617   generate 83.7 s

   Not a subtle difference — the diffusion loop is precision-sensitive and lands on a different
   token field. This is the same phenomenon as the earlier TF32/fp16 results, now reproduced in the
   browser. Whatever model is chosen, WASM and WebGPU need separate listening.

2. **A comparator bug in the port, found and fixed, that turned out NOT to be the cause.**
   `sort((a,b) => score[b] - score[a])` returns **NaN** when both slots carry `-Infinity`
   (already-committed slots), and a NaN comparator leaves the sort order undefined — so the top-k
   could commit arbitrary slots. The C# uses `CompareTo`, which orders infinities correctly. Fixed
   to an explicit comparison. ⚠ Output was **identical to four decimals** afterwards (peak 0.3799,
   rms 0.0488), so this was a latent bug rather than the audible defect. Worth keeping fixed; not
   worth crediting with the symptom.

**The ladder now under test**, all generated through the C# CLI on CPU — the reference path — with
the same longer sentence and reference clip, so quantization is the only variable:

| clip | weights | embedding | size |
|---|---|---|---|
| a | fp32 | fp32 | 2452 MB (ground truth) |
| b | **int8** block-32 | int8 per-row | 696 MB |
| c | int4 block-32 | **fp16** | 626 MB |
| d | int4 block-32 | int8 per-row | 472 MB |

Three-way discrimination by design: b clean + d not -> weight width; c clean + d not -> the int8
embedding; both clean -> both contribute. Pending listening.

⚠ Note the method change: one sentence per variant is what produced the wrong answer in Run 4, so
the sentence here is longer and the comparison is against an fp32 control in the same batch rather
than against memory of an earlier clip.

---

## Run 12 — 2026-09-01, TWO defects: 16 steps, and the DECODER on WebGPU

The browser output was degraded for two unrelated reasons, and separating them took eliminating
most of a page of plausible causes. Recording the dead ends, because several were expensive.

### Defect 1 — 16 diffusion steps

`NUM_STEPS` was set to 16 to halve browser latency. Every clip the user called fine had been
generated at **32** (the CLI default, never overridden); every clip called quirky was at 16 — the
ladder in Run 11 used `--num-step 16`, and so did the browser. Same text, voice, model and provider;
step count was the only variable. **Fixed: 32.** ⚠ This also invalidates Run 11's ladder, which was
comparing quantization levels while unknowingly varying step count.

### Defect 2 — the Higgs decoder on WebGPU

With steps fixed, the browser was still wrong on WebGPU and perfect on WASM. The decisive
measurement was NOT a listening test:

    one forward pass, identical inputs, WASM vs WebGPU:
      max |Δlogit| = 0.0002   mean = 0.00002   argmax agreement = 100.000%

The transformer kernels agree. So the transformer could not be producing a different token field —
which pointed at the only other graph in the pipeline. Splitting the providers settled it:

| transformer | decoder | peak / rms | verdict |
|---|---|---|---|
| WebGPU | WebGPU | 0.394 / 0.055 | bad |
| WebGPU | **WASM** | **0.530 / 0.066** | **perfect** (user-confirmed) |
| WASM | WASM | 0.530 / 0.066 | perfect (user-confirmed) |

The split run reproduces the all-WASM figures exactly, so the token field was always fine: **ORT's
WebGPU convolution kernels garble the codec decoder.** The transformer is attention over a quantized
MatMulNBits graph; the decoder is a convolutional DAC.

**Fixed, and nearly free:** the decoder defaults to WASM even when the transformer is on WebGPU. The
transformer runs 64 times per generation (32 steps x 2 CFG passes), the decoder once — 20.6 s vs
19.5 s, about 1 s for correctness.

### Dead ends, in the order they were eliminated

- **Quantization.** The user reported the same quirks on the UNQUANTIZED fp32 model, which
  exonerated int4 outright and is what redirected the search. Run 11's "int4 is quantized too far"
  framing is **withdrawn**.
- **Symbol coverage.** Every symbol in the target AND reference IPA is in the en_us training corpus
  at healthy counts (`ᵻ` 2098, `ɫ` 4181, `æ` 5561).
- **Phonemizer drift.** **58/60 identical** to the actual training text. ⚠ A first attempt said
  0/60 — it compared against `work/phonemized_vernacula/byid/en_us.tsv`, an older artifact with
  punctuation STRIPPED, while the training shards keep it. Wrong reference, not a finding.
- **Sync vs async phonemization.** Identical for this sentence; every word is in the dictionary and
  the BiLSTM only fires on the OOV tail. Still a real hazard elsewhere — upstream's neural tier
  degrades to the sync reading SILENTLY when a model is missing.
- **A NaN comparator in the port.** `sort((a,b) => score[b] - score[a])` is NaN when both slots are
  `-Infinity`, leaving the order undefined. Genuine bug, fixed — but output was identical to four
  decimals, so it was latent, not the symptom.
- **Input-buffer staleness.** The loop mutates `input_ids` in place, which is exactly the CUDA
  IO-binding bug from `omnivoice_onnx_investigation.md`. Copying per call changed nothing;
  ORT-web copies anyway. Kept, because the class of bug is real and the copy is cheap.
- **Graph optimization level.** `disabled` and `basic` made no difference.
- **`enableFp16Precision`.** Looked like the knob; it is a **QNN** option, not WebGPU. ORT 1.29's
  WebGPU EP exposes only `preferredLayout` and `enableGraphCapture`, and Chrome's NVIDIA adapter
  reports `shader-f16: false` anyway.

### Two facts worth keeping

- **ORT's WebGPU `MatMulNBits` supports only 2-bit and 4-bit.** An int8-weight model fails at
  session creation there ("Only 2b and 4b quantization is supported"), so int4 is not merely the
  smallest option on WebGPU — it is the only quantized one.
- **An fp32 model cannot load in a browser at all**: its 2.45 GB sidecar exceeds the ~2 GB
  ArrayBuffer limit, and the fetch fails outright.

Also applied while here: the app now hands ORT a device built with the adapter's MAXIMUM limits
(Run 7's finding, which had been recorded but never wired into the app). This build's largest tensor
is 155 MB and clears the 268 MB default, but that was luck rather than design.

---

## Run 13 — 2026-09-01, the engine cannot live in public/ under Vite dev

**Found by clicking Generate in a real browser — not by any smoke test.** Both existing smokes serve
`public/` and `build-smoke/` from a plain Node server, so Vite never participates, and the entire
class of "Vite intercepts this request" bug was invisible to them. First real click:

    Failed to load url /vphon/src/browser.js ... This file is in /public and will be copied as-is
    during build without going through the plugin transforms, and therefore should not be imported
    from source code.

Two rounds to fix, because the first fix was incomplete:

1. Vite constant-folds a literal specifier, resolves it into `public/`, and refuses.
   `/* @vite-ignore */` does not help — the path is still statically analyzable. Building the
   specifier from parts at runtime (`["", "vphon", "src", "browser.js"].join("/")`) makes it opaque.
2. That got past the overlay but not the dev server, which still intercepted the request, appended
   `?import`, and 500'd. **Files in `public/` cannot be dynamically imported in dev at all.**

Fix: a small Vite plugin that claims `/vphon/` ahead of Vite's own middleware, strips the query, and
streams the file verbatim. Production is unaffected — `public/` is copied as-is — but `preview` now
gets the same COOP/COEP headers as `server`, so the production build can be checked under the
headers it will actually run with.

**New tool: `tools/smoke-ui.mjs`**, which drives the running server with puppeteer — loads the page,
clicks Generate, reports each progress transition with elapsed time, and fails on any `.error` in
the DOM. This is the only harness that exercises Vite, the app's own module graph, and the UI state
machine together.

**And the load is fast, contrary to appearances.** Measured against the production preview:

    data prefetch     0.3 s  (190 files)
    import browser.js 0.0 s
    loadEngine        0.8 s
    phonemize         0.32 s
    /vphon/src/ module requests: 715

⚠ A 20-minute "stuck on loading phonemizer" the user hit was **my fault, not the app's**: I killed
and restarted the dev server twice mid-session while they were clicking, and fetches against a dead
server hang rather than fail. Recorded because the symptom looked exactly like an application defect
and cost real time to disbelieve.

⚠ Still open: **715 module requests** to load the engine. Fine locally, and Netlify serves HTTP/2 so
they multiplex, but it is a lot of round trips on a cold cache over a real network. The engine
cannot simply be bundled — its data keys come from `import.meta.url` — so if this becomes a problem
the answer is a service worker or a precompiled key map, not a bundler flag.

---

## Run 14 — 2026-09-01, the silent hang: ORT's worker cannot live in the app chunk

**Symptom:** clicking Generate froze at "phonemizing — loading phonemizer (190 files)" forever, with
negligible network traffic and **no error in the UI**. The user killed it after 20 minutes.

⚠ **The first thing to establish was that it was real.** A measurement said the phonemizer loads in
1.1 s (0.3 s prefetch, 0.8 s `loadEngine`), which looked like it exonerated the app — but that
measurement ran the steps by hand inside `page.evaluate`, NOT through the app's own bundled code,
so it exercised a different path than the one failing. A "fast" number from the wrong path is worse
than no number.

**Tooling first, because the loop was too slow to think in.** `tools/browser-repl.mjs` keeps ONE
Chrome alive with remote debugging: `start` / `goto` / `eval` / `logs` / `stop`, each about a
second, with page state surviving between probes. ⚠ Its first version registered the console
listeners in the `start` process and exited, taking them with it — an instrumented build then
appeared to log nothing at all, which reads exactly like "the code never ran". Fixed with a
detached collector process that stays connected.

**With logs actually arriving, the cause was immediate:**

    [phonemizer] phonemizeAsync start
    [phonemizer] ortLoader called
    [phonemizer] ort imported
    [error] worker sent an error! ... Uncaught ReferenceError: document is not defined

`onnxruntime-web` spawns a **Web Worker** for its WASM backend, and the worker loads whatever module
ORT itself came from. Vite had bundled ORT into the app chunk — which also contains React and DOM
code — so the worker hit `document` and died. ORT's promise then never settles: no rejection, no
error, no traffic. A hang rather than a failure is why this looked like an application stall.

⚠ It only fired through `phonemizeAsync`, because English's neural tier is the first thing to ask
for ORT. Upstream's note that a neural path "degrades to the sync engine when its model is absent
rather than throwing" is what made the manual test look healthy: with no ORT loader installed it
quietly took the rule-based path and returned in 7 ms.

**Fix:** `ortInit.ts` now loads ORT from a CDN at runtime and hands out the single module instance
(`getOrt()`), and both the TTS path and the phonemizer's `setOrtLoader` use it. ORT stays in its own
module, whose worker has no DOM references. The specifier is assembled at runtime for the same
reason as the phonemizer's — a literal would be statically analyzed and re-bundled.

Side effects, all good: app chunk **585 kB -> 168 kB**, `dist` **52 MB -> 25 MB**, and ORT's dynamic
import of its threaded `.mjs` sidecar stays out of Vite's pipeline.

**Verified through the real UI on the production build:** 470 MB transformer + 87 MB decoder with
progress, WebGPU transformer + WASM decoder, `2.4s audio in 13.8s · 60 tokens · 32 steps · webgpu`,
no errors from the current bundle.

---

## Run 15 — 2026-09-01, all 30 languages, fetched per language

The picker offered 30 languages while only `en`, `es`, `cy` had staged data — selecting German
would have thrown `phonemizer data not prefetched`. Fixed by staging every offered language and
splitting the fetch in two:

- **engine** — 182 manifests, **4.5 MB**, read by importing the engine whatever language you pick;
- **per-language** — fetched only when that language is chosen.

⚠ **Most languages' lists include English's tables**, which is why nearly every per-language figure
lands around 12.2 MB. `phonemizeAsync` prewarms the English tagger for mixed-Latin text, and a run
in a script the host language does not own is delegated through `core/foreign.ts`. Upstream's
browser notes say to expect this; because the lists are RECORDED from the engine rather than
declared, it is captured rather than guessed at.

⚠ **And it is why the first total was wrong.** Summing per-language sizes gave "202 MB", but nearly
all of it is the same shared files counted once per language. Deduplicated on disk the answer is
**66 MB** — a third of the figure. The staging script now counts each file once and says the
per-language numbers overlap.

Verified through the UI: selecting German and generating downloads only German's tables (the model
came from the Cache API, no re-download) and produces
`dˈas vˈɛtɐ ɪst hˈɔʏ̯tə zeːɐ̯ ʃøːn an deːɐ̯ kˈʏstə` — 2.3 s audio in 12.5 s at 32 steps on WebGPU.

**Status: the demo works.** Text → IPA → speech, client-side, listen-confirmed good by the user.
`dist` is 71 MB (25 MB app + 66 MB phonemizer data); the 558 MB of models come from HuggingFace and
are cached in the browser after first load.

Remaining: karaoke highlighting (deferred by request; `Token` already carries `start`/`end`, and the
honest first version is proportional attribution from audio-token positions at 25/s), and the
Netlify deploy itself. If 66 MB of data on the site is unwelcome, it can move to the HF repo
alongside the models — HF already serves them with the CORS headers COEP needs.

---

## Run 16 — 2026-09-01, per-language voices, and the loudness they drag with them

**User report:** switching languages "seemed to be carrying the English voice with it". Correct, and
by construction: generation is always voice-cloned and `voices.json` held exactly ONE voice, the
English reference. Cloning is ACOUSTIC — the reference carries the speaker's accent as well as their
timbre — so every language was read by an English speaker.

**Fix: one native reference per language, taken from the corpus.** No audio and no encoder needed —
`tokens/codes_<lang>.npz` holds encoded codes and `tokens/manifest_<lang>.jsonl` the exact IPA they
were trained with. ⚠ The reference IPA must be that STORED text, not a fresh phonemization: it is
what the model saw alongside those codes. 28 languages, **0.18 MB total**. `is` and `it` are not in
the corpus and fall back by phonetic proximity (is -> sv, it -> es) rather than to English.

**"Could we just not clone?"** Tempting — it would fix loudness for free and remove the accent
question entirely. But Run 7 measured `es/fr/ca/tr` producing NOISE at 2-3 s in no-reference mode
(at 32 steps), recovering either by lengthening to 7-10 s or by adding a reference. The demo's
sample sentences are exactly that length, so no-reference reintroduces a failure we had already
diagnosed. Cloning stays; the loudness is fixed directly instead.

**Three loudness bugs, all from cloning copying the reference's level.** The corpus references span
rms **0.0017-0.099**, a 58x spread:

1. **Python's un-boost was being applied to references that were never boosted.** It exists to undo
   a gain applied when ENCODING a quiet user clip; corpus codes never went through that. German
   (`rms 0.0167`) came out at 17% of level — the user's "German was too quiet".
2. **Oromo produced 0.0 s of audio.** Its reference is `rms 0.0017`, so the generated audio sat
   entirely below the -50 dBFS silence threshold and `removeSilence` deleted the whole utterance.
   Normalising has to happen BEFORE silence detection.
3. **And again AFTER the fade.** The fine-tune emits a leading transient inside the first 0.1 s;
   normalising before the fade lets it take the headroom and the fade then removes it. This is the
   Run 5 finding, now load-bearing rather than cosmetic.

Final order, a deliberate demo-only deviation from Python parity (the desktop CLI keeps Python's
behaviour): **normalize → removeSilence → fadeAndPad → normalize.** Measured after: German
peak 0.5000, Oromo 0.5000 and 3.6 s of audio (was 0.0 s), Welsh 0.5000.

⚠ **A false trail worth recording.** Three languages measured peak 0.038 / 0.000 AFTER the fix, and
the numbers were self-consistent with the old code — which read as "the fix did not apply". The
cause was outside the app: the user switched browser tabs, and the REPL selects `pages()[0]`, so
every probe was reading a different page. `PAGE_MATCH` now picks the page by URL. A measurement tool
that silently changes what it measures is worse than no tool.

---

## Run 17 — 2026-09-01, an exemplar voice for all 102 FLEURS languages

Extended from 30 to **all 102**, with the mapping DERIVED from the corpus rather than hand-listed:
the demo code is the PHONEMIZER's code, and 100 of 102 are simply the FLEURS prefix. The two that
are not are `fil_ph -> tl` (Filipino is standardized Tagalog) and `ny_mw -> nya` (Chichewa). Every
language yielded a verified 4-8 s exemplar; nothing was skipped.

    voices.jsonc      67 KB   metadata only, hand-editable
    voice-codes.json 626 KB   (251 KB gzipped) the code arrays, keyed by voice id
    speakers          55 FEMALE / 47 MALE

⚠ **The corpus ingested all 102 even though the fine-tune trained on 28** — which is what makes this
possible, and is also why `is` and `it` did not need the phonetic-proximity stand-ins they had in
Run 16. A language outside the coverage set still gets a native voice; what it does not get is a
model that trained on its phones.

**Quietest references, most likely to sound thin or noisy** — these are the ones worth judging
first, and `--alt <lang>=<n>` replaces any of them:

    mt 0.0006 · mn 0.0006 · luo 0.0006 · ps 0.0008 · lt 0.0013
    wo 0.0015 · ckb 0.0016 · sl 0.0017 · az 0.0017 · om 0.0017

against a median around 0.02 and `pa` at 0.389. ⚠ Low reference RMS no longer affects OUTPUT level —
the demo peak-normalizes (Run 16) — but it is a decent proxy for a thin or distant recording, which
cloning will reproduce as timbre even after normalization.

**Not expanded: the picker.** It still offers 30 languages, because offering a language needs its
phonemizer TABLES staged as well as a voice, and recording those is one child process per language
(~10 s each) plus data. Voices are cheap and complete; picker entries are not. Expanding it is a
separate, measurable step.

---

## Run 18 — 2026-09-01, Japanese: "recorded" is not the same as "complete"

**Symptom:** selecting Japanese failed with `phonemizer data not prefetched:
languages/japanese/pitch-accent.tsv`.

**Cause, and it was written down in advance.** `tools/browser-prefetch.mts` says in its header:

> ⚠ AND THE PER-LANGUAGE LIST IS ONLY AS COMPLETE AS THE TEXT YOU FEED IT. Some tables load lazily
> on first USE, not at construction — Japanese's kanji readings and Zhuang's Sawndip dictionary are
> both behind a `??=` that a probe in the wrong script never reaches.

Staging ran the default probe (a Latin word and a number) for every language, so Japanese's
script-gated tables were never touched and were recorded as absent. Recording from the engine is far
better than a hand-kept list — but a recording captures only what the probe reached, and reading the
warning is not the same as acting on it.

**Two fixes, and the second is the one that matters:**

1. Record with the sentences the demo actually ships. They are already in `config.ts`, so the
   staging script reads them from there rather than duplicating them, and warns when a language has
   no sample and falls back to the default probe.
2. **A replay gate.** `tools/verify-phonemizer-data.mjs` installs the staged files as the ONLY data
   source — a frozen Map, nothing else reachable — and phonemizes every shipped sample through the
   real engine, reporting each missing key. Run against the broken staging it reproduced the bug
   exactly (`FAIL ja … missing languages/japanese/pitch-accent.tsv`, 29/30 passing) before any fix
   was applied, which is the property a gate needs.

After re-staging: **30/30 languages phonemize from the staged data alone**, 230 unique files,
75.4 MB. Verified in the browser:

    ja   kʲo̞ꜜːhä käiɡänno̞ te̞ꜜŋkiɡä to̞te̞mo̞ käite̞kide̞sɯᵝ .   5.2 s audio in 23.5 s
    cmn  t͡ɕin˥˥ tʰiɛn˥˥ xaⁱ˨˩˦ piɛn˥˥ tɤ tʰiɛn˥˥ t͡ɕʰi˥˩ …      5.1 s audio in 27.2 s

Japanese pitch accent (`ꜜ`) and Mandarin tone letters both present, which is the point of having
those tables at all.

⚠ The gate runs in `npm run build`. A staged set that cannot phonemize the shipped samples is a
broken site, and the failure is invisible until a visitor picks that language — exactly how this one
was found.

## Run 19 — 2026-09-01 18:10, karaoke highlighting without an aligner

**Question:** can word-level highlighting be made to track playback without shipping a forced
aligner, given that OmniVoice emits no alignment at all (the diffusion loop unmasks every target
position at once — there is nothing to read out of it)?

**Approach:** weight-proportional attribution. The raw decoded audio is exactly `targetTokens/25`
seconds, and `targetTokens` was chosen by `duration.ts` as a sum of per-character script weights
over the IPA. Giving each whitespace-separated IPA token the same fraction of the raw duration as it
contributed to that weight sum is self-consistent with the length the model was asked for — the
words divide time the way the estimator divided tokens. Punctuation tokens (`,` `.`) are timed too;
they are the pauses, and keeping them means consecutive words are gap-free.

**What had to be threaded through, or the highlight leads the audio:**

1. Silence removal deletes spans non-uniformly (every mid-gap over 500 ms, both edges).
   `removeSilenceMapped` now returns the surviving `KeptRun[]` in ORIGINAL sample coordinates, and
   `alignment.ts` maps each raw sample through them, snapping anything inside a removed span to the
   seam. Synthetic check: 1 s tone / 2 s silence / 1 s tone → runs `[0,36000) [84000,120000)`,
   word 1 ends at 1.6 s, `,` collapses to 1.6–1.6, word 2 runs 1.6–3.1 s in a 3.2 s output. The
   last word ends exactly `PAD_SEC` before the end, which is the right answer.
2. `fadeAndPad`'s leading 0.1 s zero-pad, added afterwards.

**In the browser** (preview on 4188, REPL, English default sentence, 5.7 s audio):

    0.45 The · 0.70–0.95 quick · 1.20–1.71 brown · 1.96–2.47 fox · 2.72–3.73 jumps ·
    3.98–4.48 over · 4.73 the · 4.98–5.24 lazy · 5.49 dog.

Monotone, ends aligned, and — by ear against the highlight — within a word of the truth
throughout; "jumps" holds longest because `d͡ʒˈʌmps` is the heaviest token, which is roughly true
of the audio too. Both display paths verified: paired (counts match) and IPA-only fallback
(`20°C` → `twˈɛnti dᵻɡɹˈiːz sˈɛɫsiʲəs`, 9 orthographic vs 12 IPA words) highlight and are
click-to-seek.

**Two things found on the way:**

- First probe showed NO highlight at all during playback, though seeking highlighted correctly.
  Cause: `requestAnimationFrame` is suspended in an occluded window, which the REPL's Chrome is.
  Added a `timeupdate` listener alongside rAF — coarse (~4 Hz) but alive in a background tab, and
  what made the verification possible.
- The first generate failed with `Cannot read properties of undefined (reading 'reduce')`. A
  COMPLETED cache meta written before `sizes` existed returns from `ensureDownloaded` before the
  legacy-discard check, and `fetchModel` then summed its absent `sizes`. A fully cached, valid model
  was unloadable until the cache was cleared. Fixed by allocating `meta.total` (equal to the sum
  for any completed meta). This affected every returning visitor from before the resumable-cache
  change.

**Limits, stated:** this is an estimate, not an alignment. Drift within a sentence of a few hundred
ms is expected and the CSS says so (a soft tint, not a hard edge). A real alignment would mean
shipping an aligner model — `Vernacula.Base.Alignment.NemoNfaAligner` exists on the desktop side —
and is the upgrade path if the estimate proves not good enough.

## Run 20 — 2026-09-01 18:40, karaoke: place words on the speech envelope, not on the clock

**User report on Run 19:** highlight "kicks in too early and then finishes too late".

**Cause:** proportional attribution over the raw generation assumes speech fills the whole span
uniformly. It does not — the model leaves onset silence, and a low-level breathy tail that sits above
the −50 dBFS silence threshold and so survives `removeSilence`. Every word inherited an early shift
from the onset, and the last word was stretched across the tail.

**Fix:** `alignment.ts` now measures a 10 ms RMS envelope over the FINAL audio and places each
word's weight share on the cumulative count of SPEECH frames (within −35 dB of the loudest frame)
rather than on seconds. Pauses and the tail consume no word weight; punctuation tokens carry zero
weight for alignment (their pause is what the envelope already skips); in a gap nothing is lit.
The `KeptRun` map from Run 19 is no longer needed for timing — the envelope is measured after all
post-processing — though `removeSilenceMapped` stays as the implementation of `removeSilence`.

**Browser** (same English sentence, 5.68 s, 50 ms sampling of the lit word):

    0.13 The · 1.24 quick · 1.64 brown · 2.14 fox · 2.59 jumps · 3.25 over · 3.60 the ·
    3.75 lazy · 4.15 dog. · 5.51 end

Onset and end now sit on the speech. "The" holding 0.13→1.24 is the fine-tune's known leading
transient: it is loud, so it counts as speech, and the gap after it is skipped — the first word's
*share* is right but its *start* is the transient. User: "It does seem better after your fix."

**Forced alignment, since the desktop has one** (`Vernacula.Base.Alignment.NemoNfaAligner`):
the Viterbi core (`CtcForcedAlignment.cs`, 189 lines) ports to TypeScript in an afternoon, but the
acoustic model is `stt_en_fastconformer_hybrid_large_pc` — 458 MB fp32, ENGLISH ONLY, sentencepiece
over English orthography. It cannot align Welsh, Japanese or the other 28 languages the demo
offers, and it takes text, not IPA. A browser aligner for this demo would need a multilingual
PHONEME-level CTC model (e.g. wav2vec2-xlsr-53-espeak-cv-ft, ~300 MB int8, emits espeak-style
IPA) plus a symbol fold from vernacula-phonemizer's IPA onto its inventory. That is a separate
spike: another download of the transformer's order of size, for a highlight bar.
