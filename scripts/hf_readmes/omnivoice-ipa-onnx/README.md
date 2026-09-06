---
license: cc-by-nc-4.0
base_model: k2-fsa/OmniVoice
tags:
  - text-to-speech
  - onnx
  - ipa
  - phonemes
library_name: onnx
---

# OmniVoice ONNX + IPA fine-tune diff

An **ONNX conversion** of [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) (a
non-autoregressive diffusion-LM TTS), plus a small **IPA fine-tune diff** that teaches the model to
accept IPA phoneme strings (from [vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer))
as text input — so the phonemizer, not the model, owns the linguistic G2P: dictionary + neural G2P,
stress, pitch accent, and text normalization (numbers, %, currency, units spoken in-language) all
happen before the model sees a token.

## Try it

**[vernacula.netlify.app](https://vernacula.netlify.app/)** runs this model entirely in your
browser — 193 languages, no upload, no server inference. It downloads the quantized build below
(472 MB, cached after the first load) and does the phonemization locally too.

The same pipeline also ships as a desktop app and a command-line renderer in
[vernacula](https://github.com/christopherthompson81/vernacula).

## Files

| File | Size | What |
|---|---|---|
| `omnivoice_transformer.onnx` (+`.onnx.data`) | 2.45 GB | **base** transformer (embeds + Qwen3-0.6B + audio heads), fp32 |
| `higgs_encoder.onnx` | 654 MB | Higgs codec encoder (24 kHz audio → codes) |
| `higgs_decoder.onnx` | 86 MB | Higgs codec decoder (codes → 24 kHz audio) |
| `omnivoice_transformer_ipa.int4.onnx` (+`.onnx.data`) | 472 MB | the fine-tuned transformer, **merged and quantized** for browsers — no base or diff needed |
| `tokenizer.json` | 11 MB | the Qwen3 tokenizer, for consumers that synthesise text |
| `ipa_diff.onnx` | 31 MB | the IPA fine-tune, as a reconstruction diff over the base transformer (currently the **v7** extraction, sha256 `31d743997ccf5b2173785ac998f4bb3c579dbc98ce5e01f7d92e5335cb0a4e28`) |
| `voices/voices.jsonc` + `voices/voice-codes.json` | 3 MB | the **stored-voice library**: 530 reference voices covering the phonemizer's languages — per voice the codec codes of a short clip, the IPA it was transcribed with, and its source — so cloning needs neither the 654 MB encoder nor any audio |
| `manifest.json` | | per-file MD5s, for the desktop app's update check |

### The voice library

Every language gets a native voice where a usable clip exists, else a donor from a related
language; `default` marks the one a consumer should start from, and `source` names the exact
clip so a poor exemplar can be traced and replaced. The web demo and the desktop app read the
same two files, so all three renderers share one voice per language. How the clips were chosen,
and what still lacks one, is in
[`docs/voice_sourcing.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/voice_sourcing.md).

The transformer is the base (un-fine-tuned) graph; the encoder/decoder are the codec, unchanged by
the fine-tune. The IPA fine-tune is distributed as a **31 MB diff** rather than a second 2.45 GB
merged graph.

### ⚠ Verify the base before applying the diff

The diff holds a LoRA plus **absolute replacement** embedding rows, so it is only meaningful
against the exact base it was extracted from. Applied to any other weights it produces a
plausible-looking model that is quietly wrong — the perturbation is the same order as the fine-tune
itself — and **nothing in the diff can detect this**: the per-row delta distributions for a right
and a wrong base overlap. Check the fingerprint:

```
sha256sum omnivoice_transformer.onnx.data
# 2ea0980e184bbf8457048fbb3ed2a01f8f8c3816a8ee9fbff3ce0886c1aeeb4a
```

## Applying the diff

The diff carries, per Linear (q/k/v/o/gate/up/down x 28 layers = 196 modules), the LoRA factors
`A (r=16, in)` and `B (out, r)` with `lora_scale = alpha/r = 2.0` in the model metadata, plus the
changed `embed_tokens` rows as `embed_rows` (fp16) and `embed_idx` (int32).

**Recommended — rewrite the graph, keeping the LoRA factored.** Leave the base weights alone and
add nodes:

```
MatMul(x, W) -> Y      becomes
MatMul(x, W) -> Y_base ;  MatMul(x, A^T) -> t ;  MatMul(t, scale*B^T) -> d ;  Add(Y_base, d) -> Y
```

with a sparse additive correction for the embedding (a compact `(changed+1, hidden)` delta table
with a zero row 0 and a `(vocab,)` int32 map, as two Gathers and an Add). Insert each chain at the
original node's position, not at the end of the graph, or the `Add` lands after its own consumers.
Because no weight is replaced, the runtime loads the base from the untouched `.onnx.data` and this
works on **every execution provider**. Measured against a full PyTorch merge: argmax agreement
100.000%, max|dlogit| 1.1e-4 (CPU) / 6.1e-5 (CUDA).

**Alternative — fold into the weights** (`W += ((B@A)*scale)^T`, overwrite the changed embed rows).
Also exact, but note that ⚠ **if you feed the folded weights back through
`SessionOptions.AddInitializer`, ONNX Runtime will silently ignore them on any non-CPU provider** —
a user-supplied initializer must be a CPU tensor, so when the session is planned on CUDA every one
is rejected ("Cannot use user supplied initializer ... planned memory location device is
different") and the base graph is served instead. The output is then bit-identical to using no diff
at all, which for IPA input is noise. If you fold, write the merged model out and load it from
disk.

Reference implementations: `apply_diff.py` (Python, fold) and `OmniVoiceDiff` (C#, graph rewrite)
in [vernacula](https://github.com/christopherthompson81/vernacula).

## The quantized build (`omnivoice_transformer_ipa.int4.onnx`)

For running in a browser via `onnxruntime-web`. The IPA fine-tune is already merged in, so this file
stands alone: no base transformer, no diff, no fold. 2452 MB -> **472 MB** (5.2x).

**How, and why it matters which way.** ⚠ Naive `quantize_dynamic` at INT8 produces output **not
recognizable as speech**. It also quantizes ACTIVATIONS, and the diffusion loop runs 32 times over
its own output, so activation error compounds — the same reason TF32 (a milder perturbation) already
produced noise on CUDA. What works is **weight-only** quantization, which leaves every activation in
fp32:

- Linear layers: `MatMulNBits`, 4-bit, block size 32, symmetric (`MatMulNBitsQuantizer`). Listen-
  tested at both 4 and 8 bits, and with the audio heads held at fp32 as a control — all four
  indistinguishable, so 4-bit with nothing exempted is what ships.
- `embed_tokens` (621 MB, and a `Gather` so the weight-only quantizer cannot reach it): quantized to
  int8 with **per-row** scales, dequantized on the gathered slice via
  `Gather(int8) + Gather(scale) + Cast + Mul`. Per-row rather than per-tensor because the fine-tune
  retrained 5,572 rows of this table.

Every tensor is kept under 256 MB (largest: 155 MB). ⚠ That is deliberate: WebGPU's `requestDevice()`
grants a **default** `maxBufferSize` of 256 MB whatever the adapter advertises, and a model with a
larger single tensor kills the device with "Out of memory" on a GPU with tens of GB free. Raising the
limit explicitly is still worth doing, but this build does not require it.

Measured, RTX 3090, 16 diffusion steps, ~4 s of audio:

| runtime | ms / forward | per generation |
|---|---|---|
| `onnxruntime-web` WASM, 8 threads | 1295 | 20.7 s |
| **`onnxruntime-web` WebGPU (Chrome/Dawn)** | **177** | **2.8 s** |

## Using it

Text -> IPA -> audio, with the phonemizer owning all G2P. The model is conditioned
**language-agnostically** (`<|lang_start|>None<|lang_end|>`); the IPA stream carries everything.

⚠ **Short input needs a reference voice.** The fine-tune corpus is FLEURS read prose — median 12 s,
only 0.21% under 3 s — so under ~5 s the model is out of distribution and can emit noise rather
than degrade gracefully. Supplying a reference clip for voice cloning fixes it, as does lengthening
the text. Generation is greedy (temperature 0) and therefore deterministic: re-running reproduces a
failure byte-for-byte.

## Attribution & license

⚠ **This repo is licensed PER FILE, and the HuggingFace tag names only the strictest part.** The
files come from two different upstreams with different terms, and treating them alike would either
over-restrict the codec or under-restrict the transformer.

| file | origin | licence |
|---|---|---|
| `omnivoice_transformer.onnx(.data)`, `ipa_diff.onnx`, `omnivoice_transformer_ipa.int4.onnx(.data)` | k2-fsa/OmniVoice pre-trained weights | **CC-BY-NC-4.0** |
| `higgs_encoder.onnx`, `higgs_decoder.onnx` | [`bosonai/higgs-audio-v2-tokenizer`](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) | **Boson Higgs Audio 2 Community License** (commercial use permitted below 100k annual active users) |
| `tokenizer.json` | the Qwen3 tokenizer | Apache-2.0 |
| `voices/voices.jsonc`, `voices/voice-codes.json` | codec codes + IPA of short reference clips from public speech corpora | **per clip**, as the `source` field says: FLEURS, Omnilingual ASR, Vaani and most others CC-BY-4.0; Common Voice CC0; OpenSLR 83/158/44 CC-BY-SA-4.0; LibriVox public domain — see [`docs/voice_sourcing.md`](https://github.com/christopherthompson81/vernacula/blob/main/docs/voice_sourcing.md) for the full table |

**Why the codec is not NonCommercial.** Upstream's restriction is on what IT trained:

> "Our code is released under the Apache 2.0 License. The pre-trained model is licensed under the
> CC-BY-NC due to constraints from its training data (e.g., Emilia)."
> — [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice)

The codec is not among those parts. `audio_tokenizer/model.safetensors` inside the OmniVoice release
is **byte-identical** to Boson's published tokenizer — sha256
`fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d` on both — and OmniVoice ships
Boson's LICENSE file beside it unchanged. Emilia never touched it, so it carries Boson's terms, not
OmniVoice's.

**What that means for codec OUTPUT.** Reference-voice codes (the few-KB arrays a cloning consumer
ships instead of the 654 MB encoder) are produced by the Boson codec, not by the OmniVoice
transformer. They are not encumbered by the NonCommercial term, and they do not need regenerating
under a different model to be usable commercially. Whatever licence the source AUDIO carries still
applies to them.

An earlier version of this card said Apache-2.0 for everything. That was wrong twice over: it
applied upstream's code licence to its weights, and it flattened two different upstreams into one.

- The fine-tune was trained on codec tokens derived from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0),
  transcribed to IPA with [vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer)
  (see the [token corpus dataset](https://huggingface.co/datasets/christopherthompson81/omnivoice-ipa-corpus)).
