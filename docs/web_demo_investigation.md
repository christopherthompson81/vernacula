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
