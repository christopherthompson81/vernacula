# Clicking a word while the document is still rendering

Reported: during bulk generation, click-to-seek does not work; it only works once the whole
render has finished. Hypothesis offered with the report was that the reader needs transactional
state management — a Redux-shaped store — so the TTS backend can mutate state without interfering
with the frontend and playback.

This log is the diagnosis, which landed somewhere else, and the fix.

## Run 1 — 2026-09-20 — is it a state problem?

Question: does the reader's state get corrupted or lost while a job is running?

No, and the paradigm the report asks for is already there.
`src/Vernacula.Avalonia/Services/Tts/TtsJobUiState.cs` is a reducer store, and says so:

```
/// Reducer-style state for one active text-to-speech job ... Accumulates every chunk the
/// backend has produced (audio + word timings) so the reader panel can attach to a job
/// mid-render, play what exists, and keep receiving chunks live. All mutations arrive via
/// Dispatch on the worker thread; the single subscriber marshals to the UI thread itself.
```

Actions (`TtsChunkProducedAction`, `TtsProgressAction`), `Dispatch`, `Subscribe` returning a
snapshot, one subscriber, mutation under a lock. The reader replays the snapshot on attach and
appends live after it.

And it demonstrably works: `AcceptChunk` sets `StartSeconds`/`EndSeconds` on the word view models
as each paragraph lands, which is why the highlight follows the voice during a render. **The words
being clicked already have correct timings.** Nothing about the symptom points at state.

## Run 2 — 2026-09-20 — what the click actually does

```csharp
// TtsReaderViewModel.SeekToWord
if (_audioPath is not null && File.Exists(_audioPath))
    _playback.SeekIntoFile(_audioPath, _audioDuration, word.StartSeconds);   // real seek
else
    _playback.SeekTo(word.StartSeconds);                                      // highlight only
```

and the branch it takes, from `PlaybackService.SeekTo`'s own comment:

```
// For MVP, we just re-anchor the wall-clock and let WaveOut continue playing what's in the
// buffer — meaning seek only affects the HIGHLIGHT, not the audio. Acceptable for
// click-to-highlight UX; full audio-seek is follow-up.
```

**Finding: it is a deferred capability, not a defect, and the TODO is in the code.** `_audioPath`
is the merged WAV, and `SegmentedSynthesis` wrote it in one call *after* the segment loop
(`WriteWavFromChunks(request.OutWavPath, audios, sampleRate)`). So while a job runs there is no
seekable file, the else-branch is taken, and the playhead does not move. Once the job ends the file
appears and the same click works — exactly the reported shape.

**The hypothesis was reasonable and the cause was elsewhere.** Worth recording because the fix it
implied — a state-management rewrite — would have been a large change to the one part of this path
that was already correct, and would not have moved the playhead.

## Run 3 — 2026-09-20 — what is already on disk while a render runs

Question: is there anything seekable mid-render, or does something have to be created?

There is, and more than expected:

| | |
|---|---|
| `seg_NNNN.wav` per paragraph | written by `SegmentedSynthesis` **before** the chunk event is raised |
| absolute start/end per paragraph | in the store's `ChunkRecord` / `ChunkProducedEvent` |
| every rendered sample | in memory twice — `TtsJobUiState._chunks` and the reader's `_receivedAudio` |
| the merged WAV | **only at the end** |

So the audio existed in three forms and none of them was the one the seek path knew how to use.
Of the ways to close that, writing the merged WAV as the run goes is the one that changes a single
place and needs no new seek machinery: `_audioPath` then exists from the first paragraph and
`SeekIntoFile` works unchanged.

## Run 4 — 2026-09-20 — does a half-written WAV read back correctly?

The whole approach rests on one mechanical question: if `WaveFileWriter` is left open and
`Flush()`ed per paragraph, does another component opening the file see the right duration, or a
stale header? Measured rather than assumed:

```
  after segment 1: reader sees 1.00s  (expected 1.00s)  file=96058 bytes
  after segment 2: reader sees 2.00s  (expected 2.00s)  file=192058 bytes
  after segment 3: reader sees 3.00s  (expected 3.00s)  file=288058 bytes
```

`Flush()` patches the RIFF sizes. No hand-rolled WAV writer needed.

Two other preconditions checked while there:

- **Paragraphs are emitted in order**, batched or not — `SegmentedSynthesis` says so and the loop
  is a straight `for idx`. Appending sequentially is therefore safe.
- **A re-render cannot be corrupted by this.** `TtsJobRunner.ReRenderAsync` renders into a staging
  directory and swaps at the end, so the file growing during a re-render is never the one the
  reader is reading.

## Run 5 — 2026-09-20 — the change

**`SegmentedSynthesis`** opens the merged `WaveFileWriter` before the loop, writes and flushes each
paragraph as it completes, and no longer buffers the document's audio to write once. That last part
is a side benefit worth naming: a ten-minute render was holding ~58 MB of float32 alive for no
reason other than the final write.

**`TtsReaderViewModel`** learns the growing file (`job.OutputAudioPath`, which is computed and valid
before the job finishes) and tracks the render frontier from the chunks it already receives. It is
held apart from `_audioPath` deliberately — `_audioPath` means "the finished render", and letting a
half-written file answer to that name would make `PlayPause` abandon the live stream for a file
that stops at the frontier. Only the seek uses it.

**`PlaybackService.SeekIntoFile`** gains `follow`. A player opens a file and plays to the EOF it
found, which for a live render is the frontier at the moment of the click; with `follow` set,
reaching the end re-opens the file if it has grown and carries on from the same position. It stops
when the file stops growing, so nothing has to tell it the render ended.

⚠ **AND THE GROWTH PROBE CANNOT USE `AudioFileReader`.** It opens through `File.OpenRead`, i.e.
`FileShare.Read`, which denies the writer that already holds the file — advisory on Linux and an
`IOException` on Windows every time, so the feature would have worked on the development machine
and silently not on Windows. The probe opens its own `FileStream` with `FileShare.ReadWrite`.

## Run 6 — 2026-09-20 — tests

`SegmentedSynthesisGrowingWavTests` pins the property the reader depends on, reading the file from
the chunk callback — the reader's own vantage point, and after the write, so it tests the contract
rather than a race:

```
TheFileIsReadableMidRunAndDescribesOnlyWhatIsRendered   1.0, 2.0, 3.0, 4.0 seconds
TheFinishedFileIsStillTheWholeRender                    unchanged artifact, 24 kHz mono
ACancelledRunLeavesThePartialAudioReadable              2.0s of a 4-paragraph document
```

The cancelled case matters on its own: a cancelled render is exactly when a half-written file is
being looked at, and `using` on the writer is what leaves it valid.

**Not verified in the running app.** These cover the mechanism — the file grows, reads back
correctly, and survives cancellation — but nobody has clicked a word in a live render on this
build. That is the one check still outstanding.
