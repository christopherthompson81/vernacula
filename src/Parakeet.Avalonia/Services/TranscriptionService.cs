using System.Text.Json;
using Parakeet.Base;
using Parakeet.Base.Models;
using ParakeetCSharp.Models;
using ParakeetAsr = Parakeet.Base.Parakeet;

namespace ParakeetCSharp.Services;

internal class TranscriptionService
{
    private readonly SettingsService _settings;

    public TranscriptionService(SettingsService settings) => _settings = settings;

    /// <summary>
    /// Runs the full diarization + ASR pipeline on a dedicated worker thread.
    /// <para>
    /// Uses <see cref="TaskCreationOptions.LongRunning"/> to obtain a dedicated
    /// OS thread (not a ThreadPool thread) and lowers its priority to
    /// <see cref="ThreadPriority.BelowNormal"/> so that CPU-bound ONNX inference
    /// does not starve the UI thread when falling back to CPU execution.
    /// </para>
    /// <para>
    /// Callbacks are invoked on the worker thread — callers must marshal
    /// to the UI dispatcher before touching UI elements.
    /// </para>
    /// </summary>
    public Task RunAsync(
        string audioPath,
        int    streamIndex,
        string resultsDbPath,
        IProgress<TranscriptionProgress> progress,
        Action<SegmentRow>   onSegmentAdded,
        Action<int, string>  onSegmentText,
        CancellationToken    ct)
    {
        // LongRunning spins up a dedicated OS thread. The async lambda allows
        // internal await points (for pipelined preprocessing) without blocking that
        // thread across the full duration; continuations resume on thread-pool threads.
        return Task.Factory.StartNew(
            async () =>
            {
                Thread.CurrentThread.Priority = ThreadPriority.BelowNormal;
                await RunPipelineAsync(audioPath, streamIndex, resultsDbPath,
                    progress, onSegmentAdded, onSegmentText, ct);
            },
            ct, TaskCreationOptions.LongRunning, TaskScheduler.Default).Unwrap();
    }

    private async Task RunPipelineAsync(
        string audioPath,
        int    streamIndex,
        string resultsDbPath,
        IProgress<TranscriptionProgress> progress,
        Action<SegmentRow>  onSegmentAdded,
        Action<int, string> onSegmentText,
        CancellationToken   ct)
    {
        string modelsDir = _settings.GetModelsDir();

        // ── Phase 1: Load audio ───────────────────────────────────────────────
        progress.Report(new TranscriptionProgress(
            TranscriptionPhase.LoadingAudio, 0, 1, Loc.Instance["progress_loading_audio"]));

        ct.ThrowIfCancellationRequested();
        var (rawSamples, sampleRate, channels) = AudioUtils.ReadAudio(audioPath, streamIndex);
        float[] audio = AudioUtils.AudioTo16000Mono(rawSamples, sampleRate, channels);

        // ── Phase 2: Open results DB ──────────────────────────────────────────
        using var db = new TranscriptionDb(resultsDbPath);
        if (!db.CheckMetadata(audioPath))
            db.PopulateMetadata(audioPath);

        // ── Phase 3: Segmentation (Diarization or VAD) ───────────────────────
        List<(double start, double end, string spkId)> segs;
        bool inlineAsrDone = false;

        bool useDiarization =
            _settings.Current.Segmentation == SegmentationMode.Diarization;

        if (!db.CheckDiarization() && !useDiarization)
        {
            // ── VAD path ──────────────────────────────────────────────────────
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 0, 1, Loc.Instance["progress_detecting_speech"]));

            ct.ThrowIfCancellationRequested();
            List<(double, double)> vadSegs = await Task.Run(
                () =>
                {
                    using var vad = new VadSegmenter(modelsDir);
                    return vad.GetSegments(audio);
                }, ct);

            segs = new List<(double, double, string)>(vadSegs.Count);
            db.InsertSpeaker("speaker_1");
            for (int i = 0; i < vadSegs.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                var (start, end) = vadSegs[i];
                db.InsertResult(
                    diarizationSpeakerId: 1,
                    speakerId:            1,
                    startTime:            start,
                    endTime:              end,
                    startTimeF:           AudioUtils.SecondsToHhMmSs(Math.Round(start)),
                    endTimeF:             AudioUtils.SecondsToHhMmSs(Math.Round(end)),
                    asrContent: null, content: null, tokens: null, timestamps: null, logprobs: null);

                onSegmentAdded(new SegmentRow
                {
                    SegmentId          = i,
                    SpeakerTag         = "speaker_1",
                    SpeakerDisplayName = "speaker_1",
                    StartTime          = start,
                    EndTime            = end,
                });
                segs.Add((start, end, "speaker_1"));
            }

            db.MarkDiarizationComplete();

            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 1, 1,
                Loc.Instance.T("progress_vad_complete", new() { ["count"] = segs.Count.ToString() })));
        }
        else if (!db.CheckDiarization())
        {
            segs = new List<(double, double, string)>();
            var seenSpeakers = new HashSet<string>();

            var (encoderFile, decoderJointFile) =
                Config.GetAsrFiles(_settings.Current.Precision);
            using var parakeet = new ParakeetAsr(modelsDir, encoderFile, decoderJointFile);

            using var streamer = new SortformerStreamer(modelsDir);
            float[,,] melSpec = AudioUtils.LogMelSpectrogram(audio);
            var (totalFrames, chunkStride, numChunks) = streamer.GetPredParams(melSpec);

            int chunkIdx     = 0;
            int asrSegOffset = 0;
            int completedAsr = 0;

            // ── Pipelined inline ASR ──────────────────────────────────────────
            // After Sortformer yields a chunk we immediately:
            //   (a) kick off the next Sortformer chunk on a thread-pool thread (GPU)
            //   (b) run CPU preprocessing for the current batch on another pool thread
            // (a) and (b) run concurrently, hiding mel-feature extraction time inside
            // Sortformer's GPU inference window.

            using var enumerator =
                streamer.GetIncrementalSegments(melSpec, totalFrames, chunkStride, numChunks)
                        .GetEnumerator();

            // Start the very first Sortformer chunk on a pool thread right away.
            Task<bool> sortformerTask = Task.Run(() => enumerator.MoveNext(), ct);

            while (true)
            {
                bool hasChunk = await sortformerTask;
                if (!hasChunk) break;

                ct.ThrowIfCancellationRequested();
                var  newSegs     = enumerator.Current;
                bool isLastChunk = chunkIdx == numChunks - 1;

                // Report diarization progress scaled to 0..(numChunks/(numChunks+1))
                string diarText = Loc.Instance.T("progress_identifying_speakers", new() {
                    ["i"]     = (chunkIdx + 1).ToString(),
                    ["count"] = numChunks.ToString() });
                progress.Report(new TranscriptionProgress(
                    TranscriptionPhase.Diarizing, chunkIdx + 1, numChunks + 1, diarText));
                chunkIdx++;

                // Insert newly-committed segments into DB and notify UI
                foreach (var (start, end, spkId) in newSegs)
                {
                    int diarSpkId = int.Parse(spkId.Replace("speaker_", "")) + 1;
                    db.InsertResult(
                        diarizationSpeakerId: diarSpkId,
                        speakerId:            diarSpkId,
                        startTime:            start,
                        endTime:              end,
                        startTimeF:           AudioUtils.SecondsToHhMmSs(Math.Round(start)),
                        endTimeF:             AudioUtils.SecondsToHhMmSs(Math.Round(end)),
                        asrContent: null, content: null, tokens: null, timestamps: null, logprobs: null);

                    if (seenSpeakers.Add(spkId))
                        db.InsertSpeaker(spkId);

                    onSegmentAdded(new SegmentRow
                    {
                        SegmentId          = segs.Count,
                        SpeakerTag         = spkId,
                        SpeakerDisplayName = spkId,
                        StartTime          = start,
                        EndTime            = end,
                    });
                    segs.Add((start, end, spkId));
                }

                // (a) Next Sortformer chunk — starts immediately on a pool thread (GPU).
                sortformerTask = isLastChunk
                    ? Task.FromResult(false)
                    : Task.Run(() => enumerator.MoveNext(), ct);

                // (b) CPU preprocessing for current segments — runs concurrently with (a).
                if (newSegs.Count > 0)
                {
                    int batchSize = newSegs.Count;
                    var preprocessTask = Task.Run(() => parakeet.PrepareBatch(newSegs, audio), ct);

                    // Await CPU preprocessing (Sortformer GPU is running in parallel).
                    var preparedBatch = await preprocessTask;

                    // GPU encode + decode (Sortformer may still be running; ONNX Runtime
                    // serialises sessions on the same CUDA device via stream ordering).
                    if (preparedBatch != null)
                    {
                        int batchCompleted = 0;
                        foreach (var (segId, text, tokens, timestamps, logprobs) in
                            parakeet.RecognizePrepared(preparedBatch))
                        {
                            ct.ThrowIfCancellationRequested();
                            int absId = asrSegOffset + segId;
                            completedAsr++;
                            batchCompleted++;

                            db.UpdateResult(
                                resultId:   absId + 1,
                                asrContent: text,
                                content:    text,
                                tokens:     JsonSerializer.Serialize(tokens),
                                timestamps: JsonSerializer.Serialize(timestamps),
                                logprobs:   JsonSerializer.Serialize(logprobs));

                            onSegmentText(absId, text);

                            if (isLastChunk)
                            {
                                string asrText = Loc.Instance.T("progress_recognizing_segment", new() {
                                    ["i"]     = completedAsr.ToString(),
                                    ["count"] = segs.Count.ToString() });
                                progress.Report(new TranscriptionProgress(
                                    TranscriptionPhase.Recognizing,
                                    numChunks * batchSize + batchCompleted,
                                    (numChunks + 1) * batchSize,
                                    asrText, absId, text));
                            }
                        }
                        asrSegOffset += batchSize;
                    }
                }
            }

            db.MarkDiarizationComplete();
            inlineAsrDone = true;
        }
        else
        {
            segs = db.GetSegments();

            // Notify UI of all diarized segments (resume path only)
            int segIdx = 0;
            foreach (var seg in segs)
            {
                onSegmentAdded(new SegmentRow
                {
                    SegmentId          = segIdx++,
                    SpeakerTag         = seg.spkId,
                    SpeakerDisplayName = seg.spkId,
                    StartTime          = seg.start,
                    EndTime            = seg.end,
                });
            }
        }

        ct.ThrowIfCancellationRequested();

        // ── Phase 4: ASR (resume or non-incremental path only) ───────────────
        var (asrDone, startSeg) = db.CheckAsr();

        // Populate text for segments already transcribed in a prior run
        if (startSeg > 0)
        {
            var existingTexts = db.GetResultContents();
            for (int i = 0; i < startSeg && i < existingTexts.Count; i++)
                onSegmentText(i, existingTexts[i]);
        }

        if (!asrDone && !inlineAsrDone)
        {
            var (encoderFile, decoderJointFile) =
                Config.GetAsrFiles(_settings.Current.Precision);

            using var parakeet = new ParakeetAsr(modelsDir, encoderFile, decoderJointFile);
            var segsSubset = segs.GetRange(startSeg, segs.Count - startSeg);
            int totalSegs  = segs.Count;
            int completed  = startSeg;

            foreach (var (segId, text, tokens, timestamps, logprobs) in
                parakeet.Recognize(segsSubset, audio))
            {
                ct.ThrowIfCancellationRequested();
                int absId = startSeg + segId;
                completed++;

                db.UpdateResult(
                    resultId:   absId + 1,
                    asrContent: text,
                    content:    text,
                    tokens:     JsonSerializer.Serialize(tokens),
                    timestamps: JsonSerializer.Serialize(timestamps),
                    logprobs:   JsonSerializer.Serialize(logprobs));

                onSegmentText(absId, text);

                string asrText = Loc.Instance.T("progress_recognizing_segment", new() {
                    ["i"]     = completed.ToString(),
                    ["count"] = totalSegs.ToString() });
                progress.Report(new TranscriptionProgress(
                    TranscriptionPhase.Recognizing,
                    completed, totalSegs, asrText, absId, text));
            }
        }

        progress.Report(new TranscriptionProgress(
            TranscriptionPhase.Done, 1, 1, Loc.Instance["transcription_complete"]));
    }
}
