# Licensing

Vernacula ships as three projects with two distinct licenses, plus third-party model weights under their own terms.

## Component licenses

| Project | Description | License |
|---|---|---|
| `Vernacula.Base` | Core inference library — ASR, diarization, VAD, audio utilities, KenLM scorer | [MIT](../src/Vernacula.Base/LICENSE) |
| `Vernacula.CLI` | Command-line transcription tool | [MIT](../src/Vernacula.Base/LICENSE) |
| `Vernacula.Avalonia` | Desktop GUI app (Vernacula-Desktop) | [PolyForm Shield 1.0.0](../src/Vernacula.Avalonia/LICENSE) |

### PolyForm Shield summary

PolyForm Shield 1.0.0 allows free use, modification, and distribution of `Vernacula.Avalonia` — with one restriction: you may not use it to build a product that competes with Vernacula-Desktop itself. The full license text is canonical; this summary is informational only.

## Model weights

Model weights are distributed separately through HuggingFace and carry their own licenses. See each repository linked from [Models](models.md) for terms.
