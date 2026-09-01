# vernacula-tts

IPA-native text-to-speech. `vernacula-tts` joins
[vernacula-phonemizer](https://github.com/christopherthompson81/vernacula-phonemizer) — canonical
IPA for 192 routed language codes — to an IPA fine-tune of
[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), so any language the phonemizer covers becomes
an inference-time application rather than a training target.

The fine-tuned model is conditioned **language-agnostically**: the style prefix is
`<|lang_start|>None<|lang_end|>` and the IPA stream carries everything. `--lang` therefore selects
the *phonemizer*, never the model.

## Build

```bash
git submodule update --init external/vernacula-phonemizer
dotnet build src/Vernacula.Tts.CLI -c Release          # -p:EP=Cuda (default) | DirectML | Cpu
```

The binary lands at `src/Vernacula.Tts.CLI/bin/Release/net10.0/vernacula-tts`.

## What it needs

| Thing | Flag | Notes |
|---|---|---|
| The three ONNX graphs | `--onnx-dir` (`OMNIVOICE_ONNX_DIR`) | `omnivoice_transformer.onnx` + `.onnx.data`, `higgs_encoder.onnx`, `higgs_decoder.onnx` |
| Qwen3 `tokenizer.json` | `--model-dir` (`OMNIVOICE_MODEL_DIR`) | the `k2-fsa-OmniVoice` snapshot; or `--tokenizer-json` directly |
| The IPA fine-tune | `--diff` | defaults to `ipa_diff_v6.onnx` inside `--onnx-dir` |
| The genuine base | `--transformer-file` | required with `--diff` if `--onnx-dir`'s transformer is not the upstream checkpoint — see the warning below |
| Phonemizer data | `--data-dir` | defaults to the `external/vernacula-phonemizer/data` submodule |

Model export lives in [`scripts/omnivoice_export`](../scripts/omnivoice_export); the fine-tune
corpus and training in [`scripts/omnivoice_ipa`](../scripts/omnivoice_ipa).

## Usage

```bash
# Plain synthesis
vernacula-tts --lang en --text "Hello world." --out hello.wav

# Voice cloning — the reference transcript is phonemized too (--ref-lang defaults to --lang)
vernacula-tts --lang cy --text "Bore da." \
              --voice ref.wav --ref-text "This is a reference sample." --ref-lang en \
              --out out.wav

# A document; .md/.markdown has its markup stripped first
vernacula-tts --lang en --text-file page.md --out page.wav

# Supply IPA directly and skip the phonemizer
vernacula-tts --ipa --text "hɛlˈoʊ wˈɜːld." --out out.wav
```

`--print-ipa` shows exactly what reaches the model — the first thing to check when output is wrong.

## Execution provider

The IPA diff is applied as a **graph rewrite**, not a weight fold: the LoRA becomes MatMul/Add
nodes and the embedding correction a pair of Gathers, so the base weights are never replaced and
ORT loads them from the untouched `.onnx.data` on any provider. `--ep cuda` works, and needs no
pre-merged model.

Measured on an RTX 3090, 32 steps, ~4.4 s of audio:

| path | time | vs real-time |
|---|---|---|
| CPU + diff | ~16 s | 0.3× |
| CUDA + diff | ~1.7 s | ~2.5× |

The rewrite reproduces a Python-merged transformer exactly: argmax agreement 100.000%,
max|Δlogit| 1.1e-4 on CPU and 6.1e-5 on CUDA, and end-to-end audio identical to the merged model
(max sample difference 0.00000 on CUDA, 0.00001 CPU-vs-CUDA).

CUDA runs full fp32 with TF32 disabled: the diffusion loop is precision-sensitive and degrades into
noise under TF32. See [`omnivoice_onnx_investigation.md`](omnivoice_onnx_investigation.md).

### ⚠ The diff must be applied to the genuine base

The diff is a LoRA plus **absolute replacement** embedding rows. Applied to weights that are not
the base OmniVoice checkpoint it yields a plausible-looking model that is quietly wrong — the
perturbation is the same order as the fine-tune delta itself. There is no self-check: the per-row
delta distributions for a right and a wrong base overlap (median 0.0086 vs 0.0124), so nothing in
the diff's contents can detect the mistake.

Point `--transformer-file` at the transformer whose weights match the `k2-fsa/OmniVoice`
checkpoint. To verify a candidate, compare its `model.llm.embed_tokens.weight` against
`model.safetensors` in the HF snapshot; the genuine base matches exactly.

## Languages

`--lang` accepts any of the phonemizer's 192 routed codes. The v6 fine-tune trained on 28 of them,
chosen as a greedy cover over IPA primitives rather than by popularity — English for the generalist
Latin base, Zulu for clicks and breathy voice, Hausa for ejectives, Fula for prenasals:

```
am ar ca cmn cs cy de en es ff fr ga ha hi ja kk ko om pt ru sd sv ta th tr vi xh zu
```

Anything outside that list still renders — an IPA-conditioned model draws on phones it already
holds, which is the entire premise — but the CLI prints a note that the result is extrapolated, and
prosody most of all.

## Known limitations

- **Short input needs a reference voice.** The fine-tune's corpus is FLEURS read sentences —
  median 12 s, and only 0.21% under 3 s — so under ~5 s the model is out of distribution and can
  emit noise rather than degrade gracefully. Measured: `es`/`fr`/`ca`/`tr` all produced noise at
  2-3 s in auto mode, and all were clean at the same length once given `--voice`, or when the text
  was lengthened to 7-10 s. Generation is greedy and deterministic, so re-running reproduces the
  failure byte-for-byte — it is not a bad roll. The CLI warns when both conditions hold.
- **Single utterance.** Long-form chunking is not implemented for this backend; past ~1500 tokens
  (~60 s) the CLI warns and quality degrades. Split the input.
- **Output level.** The fine-tune emits a leading transient that absorbs peak normalization before
  the fade removes it, leaving the speech peaking around 0.098 rather than the nominal 0.5. Faithful
  to the Python post-processing order, and clips still sound fine; see
  [the investigation log](vernacula_tts_investigation.md) if you want the headroom back.
- **`--ref-text` is required with `--voice`.** ASR auto-transcription of the reference is not wired.

## See also

- [vernacula-tts investigation log](vernacula_tts_investigation.md) — how this was validated, and what is open
- [OmniVoice ONNX export](omnivoice_onnx_investigation.md) — the graphs, parity, CUDA correctness
- [IPA corpus investigation](omnivoice_ipa_corpus_investigation.md) — the fine-tune corpus
