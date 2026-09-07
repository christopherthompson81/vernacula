# Vernacula.CLI

`vernacula-cli` — the command-line transcription tool. It runs the same pipeline as the
desktop app (diarization → ASR → optional language identification and LM fusion) and writes
transcripts as Markdown, plain text, JSON or SRT.

```bash
dotnet run --project src/Vernacula.CLI -p:EP=Cuda -- --audio meeting.wav
```

With no `--models-dir`, models are read from the desktop app's own models root, so anything
downloaded in the app is found without a flag.

**Documentation**

- [Arguments and examples](../../docs/cli-reference.md) — the full reference; `--help` prints the same list
- [Building from source](../../docs/building.md) — the `EP` build property (CUDA / CPU / DirectML)
- [Models](../../docs/models.md) — what each backend needs and where it goes

Licensed under [MIT](LICENSE).
