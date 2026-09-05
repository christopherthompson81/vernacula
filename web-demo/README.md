# vernacula-tts — browser demo

Text → canonical IPA → speech, entirely client-side. React + Vite + `onnxruntime-web`, deployed to
Netlify as a static site. Modelled on the [Parakeet browser
demo](https://github.com/christopherthompson81/parakeet_csharp) (`demo/web-demo`), which already
solved chunked/resumable model caching, worker-hosted ORT sessions, and multi-file external data.

## Status: scaffold + one open gate

The project skeleton, language list and model manifest are in place. Two pieces of real work
remain, and one measurement gates the whole approach.

### ⚠ The gate: does the diffusion loop survive INT8?

OmniVoice's diffusion loop is **precision-sensitive** in a way most transformers are not — iterative
unmasking amplifies matmul error. Measured previously (`docs/omnivoice_onnx_investigation.md`):

| precision | result |
|---|---|
| CUDA TF32 | incoherent **noise** |
| CUDA fp16 | a *different but valid*, good-sounding rendering (listen-confirmed) |
| CUDA fp32 | faithful to CPU |

Qwen3 itself quantizes well, but this is not Qwen alone — it is Qwen inside 32 iterations of a
diffusion loop, and the loop is where the error compounds. So INT8 is a **test, not a plan**.

Built and pending a listening test:

    omnivoice_transformer_ipa_v6.int8.onnx   617 MB   (from 2452 MB fp32, per-channel dynamic)

An automated envelope check puts it on the speech side (syllable-band modulation 0.351 vs fp32's
0.358; noise scored ≤0.293 in earlier tests), and it generates *faster* than fp32. But that metric
has produced false positives before, so it decides nothing. **Ears decide.**

If INT8 fails, the ladder is:

| rung | size | quality |
|---|---|---|
| INT8 | 617 MB | unknown — pending listen |
| fp16 | ~1.2 GB | previously listen-confirmed good (on CUDA) |
| fp32 | 2.45 GB | reference |

fp16 and fp32 both exceed the 2 GB single-file protobuf limit once the sidecar is counted, which is
fine: `onnxruntime-web` accepts `sessionOptions.externalData` as an **array**, so the weight sidecar
can be split across several files and streamed. The Parakeet demo already has that worker plumbing
(`ExternalDataEntry[]`) to copy. ⚠ Note fp16 is a poor fit for the WASM backend specifically — ORT's
WASM kernels are fp32/int8, so fp16 buys download size but not speed there, and may cast back up
internally. fp16 is really a WebGPU rung.

### Work item 1 — the phonemizer in the browser

`vernacula-phonemizer` is TypeScript, which is the good news. The bad news is that it is
**synchronous** and reads its data through `node:fs`.

- Only **15 modules** touch `fs`, and `src/core/dataPath.ts` is a single choke point, so the surface
  is small.
- The fix is *not* an async refactor. Prefetch the selected language's data files, then serve them
  from an in-memory map through a `readFileSync` shim aliased in `vite.config.ts`. The engine stays
  synchronous and unmodified.
- Neural languages need `onnxruntime-node` → `onnxruntime-web`. `src/neuralRegistry.ts` routes the
  Arabic dialects, `pa`, `sd`, `ur`; English's BiLSTM arrives by a different path. Rule-only
  languages are the cheap first cut.

Per-language data is mostly tiny — Welsh 56 KB, Zulu 28 KB, Turkish 40 KB, Spanish 20 KB — with a
few heavy outliers: Arabic 35 MB, English 14 MB, Japanese 8.9 MB, Russian 8.4 MB. Lazy-load per
language rather than shipping the 151 MB tree.

### Work item 2 — the diffusion loop in TypeScript

`Vernacula.Tts.Base.OmniVoiceTts.RunDiffusion` has to be ported: CFG batch, shifted-timestep schedule,
guidance log-prob mix, layer penalty, top-k unmask, scatter. It is the one genuinely novel port; the
graphs themselves are just `session.run`.

## Design decisions already made

**Voice cloning is always on, and the encoder never ships.** The 654 MB Higgs *encoder* exists only
to turn a reference WAV into codec codes. Those codes are a few KB, so they are precomputed offline
and shipped as `voices.json`. This drops 654 MB *and* fixes a real failure: without a reference
voice, input under ~5 s is out of the fine-tune's distribution (corpus median 12 s, only 0.21% under
3 s) and can emit noise rather than degrade. Always-clone makes short input — which is exactly what
a demo gets — stable.

**The picker lists the 28 trained languages, not all 192.** The phonemizer will produce IPA for any
of them and the model will render it from phones it already holds; that is the premise. But the
result is extrapolated, so the other 164 belong behind an "experimental" affordance.

## Karaoke highlighting

Word highlighting follows playback, and is **estimated, not aligned**. OmniVoice emits no
alignment — the diffusion loop unmasks every target position at once — so `src/inference/alignment.ts`
attributes the raw duration (`targetTokens / 25` s, exactly) to each IPA token in proportion to the
same per-character script weights `duration.ts` used to pick `targetTokens`. That makes it
self-consistent with the length the model was asked for, monotone, and correct at both ends, with
drift of a few hundred ms within a sentence. Silence removal and the leading pad are mapped through
(`removeSilenceMapped` returns the surviving runs), because without that the highlight leads the
audio by everything that was cut.

If the estimate proves not good enough, the upgrade is a real forced alignment of the generated audio
against the IPA — `Vernacula.Base.Alignment.NemoNfaAligner` on the desktop side — at the cost of
shipping another model.

## Layout

    src/types.ts              Token / Utterance / Progress
    src/inference/config.ts   language list, model manifest, step count
    netlify.toml              COOP/COEP headers (SharedArrayBuffer) + immutable model caching

`vite.config.ts` and `netlify.toml` both set `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`; without them the browser withholds
`SharedArrayBuffer` and ORT silently drops to single-threaded WASM.
