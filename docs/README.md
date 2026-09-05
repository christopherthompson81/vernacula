# Vernacula documentation

User and reference documentation for Vernacula. Start with [Installation](installation.md) if you're setting up for the first time, or jump straight to the [Desktop app](desktop-app.md) or [CLI reference](cli-reference.md) if you already have it built.

## Getting started

- [Installation](installation.md) — .NET 10, FFmpeg, GPU prerequisites, Linux installer
- [Desktop app](desktop-app.md) — features, screenshots, walkthrough
- [CLI reference](cli-reference.md) — invocation, arguments, examples
- [vernacula-tts](tts-cli.md) — IPA-native text-to-speech (phonemizer → OmniVoice IPA fine-tune)
- [Models](models.md) — required and optional model downloads
- [Building from source](building.md) — execution providers and publish guidance

## Reference

- [Pipeline backends and language support](reference/backends.md) — ASR backends, 52-language matrix, diarization, execution providers, DiariZen tuning
- [Language model fusion (KenLM)](reference/language-model-fusion.md) — shallow fusion with Parakeet
- [Benchmarks](reference/benchmarks.md) — throughput and DER numbers

## Project

- [Licensing](licensing.md) — per-component license attribution
- [A model of our own](own_model_plan.md) — **shelved** plan for unencumbered weights and precomputed voices, kept for its rejected alternatives
- [Developer notes](dev/) — internal investigations and design notes
