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

## Execution provider — read this before using `--ep cuda`

**The IPA diff's load-time fold only works on CPU.** `OmniVoiceDiff` supplies the folded weights
from CPU memory via `SessionOptions.AddInitializer`; when ORT plans the session on CUDA it rejects
all 197 of them and falls back to the base graph. Output is then *bit-identical* to running with no
fine-tune at all — stock orthographic OmniVoice fed IPA, which is inaudible as an error and simply
sounds wrong. The CLI refuses that combination rather than let it happen, and `--ep auto` takes CPU.

**The recommended path is CUDA with a pre-merged IPA transformer** — listen-confirmed equal in
quality to the CPU fold and roughly 10x faster. It needs no fold, so nothing gets rejected:

```bash
vernacula-tts --lang en --text "..." --ep cuda --no-diff \
  --transformer-file /path/to/omnivoice_transformer_ipa_v6.onnx --out out.wav
```

Measured on an RTX 3090, 32 steps, ~4.4 s of audio:

| path | time | vs real-time | fine-tune active |
|---|---|---|---|
| CPU + folded diff | ~15–18 s | 0.2–0.3× | yes |
| CUDA + pre-merged transformer | ~1.5 s | ~3× | yes |
| CUDA + folded diff | ~1.4 s | ~3× | **no — silently the base model, which outputs noise** (refused) |

CUDA runs full fp32 with TF32 disabled: the diffusion loop is precision-sensitive and degrades into
noise under TF32. See [`omnivoice_onnx_investigation.md`](omnivoice_onnx_investigation.md).

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
