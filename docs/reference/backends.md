# Pipeline backends and language support

Vernacula's pipeline is modular: you pick an ASR backend, a diarization backend, and an execution provider (EP). This page is the reference for what each choice does, which languages it covers, and how to tune it.

## ASR backends

| Backend | Latency | Languages | Notes |
|---|---|---|---|
| **Parakeet TDT v3 (default)** | Fast | 25 European | Streaming-friendly TDT. Supports beam search + KenLM fusion. |
| **Cohere Transcribe (03-2026)** | Medium | 14 + auto-detect | Transformer encoder-decoder. |
| **Qwen3-ASR 1.7B** | Medium | 29 | Multilingual. Can force language via prompt. |
| **VibeVoice-ASR** | Slow (CUDA only) | ~51, long-tail | All-in-one transcription with built-in diarization. English dominates training (~67%); the remaining ~50 languages have sparse coverage (most ≤ 1% of training data) so practical accuracy scales with representation. |

## Language support matrix

Transcription language support across the ASR backends, plus LM-fusion
coverage for Parakeet (Parakeet-only) and LID coverage (VoxLingua107
*identifies* 107 languages — it doesn't transcribe).

Legend: ● = supported · ○ = not supported · 🅛 = KenLM available

| Code | Language | Parakeet v3 | Cohere | Qwen3-ASR | VibeVoice | Parakeet KenLM |
|------|----------|:--:|:--:|:--:|:--:|:--:|
| aa | Afar | ○ | ○ | ○ | ● | ○ |
| af | Afrikaans | ○ | ○ | ○ | ● | ○ |
| ar | Arabic | ○ | ● | ● | ● | ○ |
| bg | Bulgarian | ● | ○ | ○ | ● | ○ |
| ca | Catalan | ○ | ○ | ○ | ● | ○ |
| cs | Czech | ● | ○ | ● | ● | ○ |
| da | Danish | ● | ○ | ● | ● | ○ |
| de | German | ● | ● | ● | ● | ○ |
| el | Greek | ● | ● | ● | ● | ○ |
| en | English | ● | ● | ● | ● | 🅛 general + medical |
| es | Spanish | ● | ● | ● | ● | ○ |
| et | Estonian | ● | ○ | ○ | ● | ○ |
| fa | Persian | ○ | ○ | ● | ● | ○ |
| fi | Finnish | ● | ○ | ● | ● | ○ |
| fr | French | ● | ● | ● | ● | ○ |
| he | Hebrew | ○ | ○ | ○ | ● | ○ |
| hi | Hindi | ○ | ○ | ● | ● | ○ |
| hr | Croatian | ● | ○ | ○ | ● | ○ |
| hu | Hungarian | ● | ○ | ● | ● | ○ |
| hy | Armenian | ○ | ○ | ○ | ● | ○ |
| id | Indonesian | ○ | ○ | ● | ● | ○ |
| is | Icelandic | ○ | ○ | ○ | ● | ○ |
| it | Italian | ● | ● | ● | ● | ○ |
| ja | Japanese | ○ | ● | ● | ● | ○ |
| jv | Javanese | ○ | ○ | ○ | ● | ○ |
| kl | Kalaallisut (Greenlandic) | ○ | ○ | ○ | ● | ○ |
| ko | Korean | ○ | ● | ● | ● | ○ |
| lt | Lithuanian | ● | ○ | ○ | ● | ○ |
| lv | Latvian | ● | ○ | ○ | ● | ○ |
| mk | Macedonian | ○ | ○ | ● | ○ | ○ |
| mn | Mongolian | ○ | ○ | ○ | ● | ○ |
| ms | Malay | ○ | ○ | ● | ● | ○ |
| mt | Maltese | ● | ○ | ○ | ○ | ○ |
| ne | Nepali | ○ | ○ | ○ | ● | ○ |
| nl | Dutch | ● | ● | ● | ● | ○ |
| no | Norwegian | ○ | ○ | ○ | ● | ○ |
| pl | Polish | ● | ● | ● | ● | ○ |
| pt | Portuguese | ● | ● | ● | ● | ○ |
| ro | Romanian | ● | ○ | ● | ● | ○ |
| ru | Russian | ● | ○ | ● | ● | ○ |
| sk | Slovak | ● | ○ | ○ | ● | ○ |
| sl | Slovenian | ● | ○ | ○ | ● | ○ |
| sr | Serbian | ○ | ○ | ○ | ● | ○ |
| sv | Swedish | ● | ○ | ● | ● | ○ |
| sw | Swahili | ○ | ○ | ○ | ● | ○ |
| th | Thai | ○ | ○ | ● | ● | ○ |
| tl | Filipino (Tagalog) | ○ | ○ | ● | ● | ○ |
| tr | Turkish | ○ | ○ | ● | ● | ○ |
| uk | Ukrainian | ● | ○ | ○ | ● | ○ |
| ur | Urdu | ○ | ○ | ○ | ● | ○ |
| vi | Vietnamese | ○ | ● | ● | ● | ○ |
| yi | Yiddish | ○ | ○ | ○ | ● | ○ |
| zh | Chinese | ○ | ● | ● | ● | ○ |
| **Total** | **52 unique** | **25** | **14** | **29** | **51** | **1 (en)** |

### Reading the matrix

- If you know your audio's language, pick an ASR backend whose row for that language is `●` (prefer Parakeet when possible — it's faster and supports KenLM fusion).
- If your audio might be in any language, leave Cohere or Qwen3 in auto-detect mode, or use `--lid` (VoxLingua107) for explicit identification across 107 languages before routing to a backend.
- **VibeVoice has the broadest nominal coverage** but the distribution is heavily English-skewed (~67% of training data). Non-English languages have sparse exposure (most < 1%) so practical accuracy tracks representation — it's reasonable for well-represented languages and a coin-flip for the long tail.
- English is the only language with KenLM fusion coverage today. More domains / more languages are tractable follow-ups — the [`scripts/kenlm_build/`](../../scripts/kenlm_build/) pipeline is language-agnostic; the current corpora are just English.

## Diarization

| Backend | Speed | Accuracy | Overlap detection |
|---|---|---|---|
| Silero VAD | Fastest | No speaker identity | No |
| Sortformer v2-stream | Fast | Good | Yes (4-speaker max per chunk) |
| DiariZen | Slower | Better | Yes (powerset, 4-speaker max) |
| VibeVoice built-in | — | Bundled with VibeVoice-ASR | Yes |

### DiariZen tuning

DiariZen's segmentation and embedding pipeline can be tuned via environment variables for your hardware:

| Variable | Description |
|---|---|
| `VERNACULA_DIARIZEN_SEG_THREADS` | Segmentation intra-op thread count |
| `VERNACULA_DIARIZEN_SEG_MAX_WORKERS` | Max parallel segmentation workers |
| `VERNACULA_DIARIZEN_SEG_BATCH_SIZE` | Segmentation batch size |
| `VERNACULA_DIARIZEN_EMBED_THREADS` | Embedding intra-op thread count |
| `VERNACULA_DIARIZEN_EMBED_MAX_WORKERS` | Max parallel embedding workers |
| `VERNACULA_DIARIZEN_EMBED_GPU_SAFETY_MB` | GPU memory safety margin (MB) |
| `VERNACULA_DIARIZEN_EMBED_GPU_MAX_BATCH_SIZE` | Max embedding batch size |
| `VERNACULA_DIARIZEN_EMBED_GPU_MAX_BATCH_FRAMES` | Max frames per embedding batch |

## Execution providers

| EP | Platform | Notes |
|---|---|---|
| CUDA | Linux, Windows | Best performance on NVIDIA GPUs |
| CPU | All | Works everywhere; slower |
| DirectML | Windows only | AMD/Intel/NVIDIA via DirectX 12 |

See [Building from source](../building.md) for how to select an EP at build time.
