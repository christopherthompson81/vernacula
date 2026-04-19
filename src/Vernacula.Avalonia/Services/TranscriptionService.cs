using System.Text.Json;
using Vernacula.Base;
using Vernacula.Base.Models;
using Vernacula.App.Models;
using ParakeetAsr = Vernacula.Base.Parakeet;
using DFN3Denoiser = Vernacula.Base.DeepFilterNet3Denoiser;

namespace Vernacula.App.Services;

internal class TranscriptionService
{
    private const double DiariZenDiarizationPercentWeight = 40.0;
    private const string CohereSyntheticTimestampMode = "uniform_segment_frames_v1";

    private readonly SettingsService _settings;
    private readonly LangIdService _langId;

    /// <summary>
    /// Optional callback invoked when LID detects a language the current
    /// ASR backend can't handle. Receives the LID result, the current
    /// backend, and the best alternative backend (or null if none). The
    /// callback returns an <see cref="Views.Dialogs.AsrMismatchResult"/>
    /// carrying the user's choice plus, for
    /// <see cref="Views.Dialogs.AsrMismatchChoice.ForceLanguage"/>, the
    /// ISO code they asked the current backend to transcribe as. When
    /// unset (CLI / tests), the pipeline just logs the mismatch and
    /// continues with the current backend — matching what happened before
    /// the popup was added.
    /// </summary>
    public Func<Vernacula.Base.Models.LidResult, AsrBackend, AsrBackend?, Task<Views.Dialogs.AsrMismatchResult?>>? OnAsrLanguageMismatch { get; set; }

    public TranscriptionService(SettingsService settings, LangIdService langId)
    {
        _settings = settings;
        _langId   = langId;
    }

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
        string               asrModelName,
        string               asrLanguageCode,
        CancellationToken    ct,
        Action<string, string>? onAsrConfigEffective = null)
    {
        // LongRunning spins up a dedicated OS thread. The async lambda allows
        // internal await points (for pipelined preprocessing) without blocking that
        // thread across the full duration; continuations resume on thread-pool threads.
        return Task.Factory.StartNew(
            async () =>
            {
                Thread.CurrentThread.Priority = ThreadPriority.BelowNormal;
                try
                {
                    await RunPipelineAsync(audioPath, streamIndex, resultsDbPath,
                        progress, onSegmentAdded, onSegmentText, asrModelName, asrLanguageCode,
                        ct, onAsrConfigEffective);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Transcription] RunPipelineAsync EXCEPTION: {ex}");
                    throw;
                }
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
        string              asrModelName,
        string              asrLanguageCode,
        CancellationToken   ct,
        Action<string, string>? onAsrConfigEffective)
    {
        Console.WriteLine($"[Transcription] RunPipelineAsync starting for '{audioPath}'");
        string parakeetModelsDir = _settings.GetParakeetModelsDir();
        string qwen3AsrModelsDir = _settings.GetQwen3AsrModelsDir();
        string sortformerModelsDir = _settings.GetSortformerModelsDir();
        string sileroModelsDir = _settings.GetSileroModelsDir();

        // ── Phase 1: Load audio ───────────────────────────────────────────────
        progress.Report(new TranscriptionProgress(
            TranscriptionPhase.LoadingAudio, 0, 1, Loc.Instance["progress_loading_audio"]));

        ct.ThrowIfCancellationRequested();
        var (rawSamples, sampleRate, channels) = AudioUtils.ReadAudio(audioPath, streamIndex);

        // ── ASR backend / segmentation flags ─────────────────────────────────
        bool useVibeVoiceAsr  = string.Equals(asrModelName, "vibevoice/vibevoice-asr", StringComparison.Ordinal);
        bool useCohereAsr     = string.Equals(asrModelName, "CohereLabs/cohere-transcribe-03-2026", StringComparison.Ordinal);
        bool useQwen3Asr      = string.Equals(asrModelName, "Qwen/Qwen3-ASR-1.7B", StringComparison.Ordinal);
        var  segmentationMode = _settings.Current.Segmentation;
        bool runVibeVoice     = useVibeVoiceAsr || segmentationMode == SegmentationMode.VibeVoiceBuiltin;

        // ── Phase 1b: Denoising (optional) ────────────────────────────────────
        // For VibeVoice, also track the audio to pass to Transcribe() — it accepts any sample rate.
        float[] audio;
        float[] vibeVoiceAudio;
        int     vibeVoiceSampleRate;
        int     vibeVoiceChannels;
        var denoiserMode = _settings.Current.Denoiser;
        if (denoiserMode == DenoiserMode.DeepFilterNet3)
        {
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Denoising, 0, 4, Loc.Instance["progress_denoising"]));

            ct.ThrowIfCancellationRequested();

            string denoiserModelsDir = _settings.GetDenoiserModelsDir();
            audio = await Task.Run(() =>
            {
                // Downmix to mono at native sample rate, resample to 48 kHz
                float[] mono48k = DFN3Denoiser.ResampleTo48k(
                    Vernacula.Base.AudioUtils.DownmixToMono(rawSamples, channels), sampleRate);

                // Pad to multiple of HopSize (480 samples) before denoising
                const int hopSize = 480;
                int padded = ((mono48k.Length + hopSize - 1) / hopSize) * hopSize;
                if (padded != mono48k.Length)
                    Array.Resize(ref mono48k, padded);

                var denoiseProgress = new Progress<(int current, int total)>(p =>
                    progress.Report(new TranscriptionProgress(
                        TranscriptionPhase.Denoising, p.current, p.total,
                        Loc.Instance["progress_denoising"])));

                using var denoiser = new DFN3Denoiser(denoiserModelsDir, Vernacula.Base.Models.ExecutionProvider.Auto);
                float[] enhanced48k = denoiser.Denoise(mono48k, denoiseProgress);

                // Resample to 16 kHz for ASR
                return DFN3Denoiser.ResampleFrom48k(enhanced48k, Vernacula.Base.AudioUtils.AsrSampleRate);
            }, ct);
            // VibeVoice uses the denoised 16 kHz mono audio directly.
            vibeVoiceAudio      = audio;
            vibeVoiceSampleRate = Vernacula.Base.AudioUtils.AsrSampleRate;
            vibeVoiceChannels   = 1;
        }
        else
        {
            // VibeVoice accepts raw audio at any sample rate; skip the 16 kHz conversion.
            vibeVoiceAudio      = rawSamples;
            vibeVoiceSampleRate = sampleRate;
            vibeVoiceChannels   = channels;
            audio = runVibeVoice
                ? Array.Empty<float>()   // not used on the VibeVoice path
                : AudioUtils.AudioTo16000Mono(rawSamples, sampleRate, channels);
        }

        // ── Phase 2: Open results DB ──────────────────────────────────────────
        using var db = new TranscriptionDb(resultsDbPath);
        if (!db.CheckMetadata(audioPath))
            db.PopulateMetadata(audioPath);

        string effectiveAsrModelName = runVibeVoice ? "vibevoice/vibevoice-asr" : asrModelName;
        db.UpdateMetadata("asr_model", effectiveAsrModelName);
        string effectiveAsrLanguageCode =
            string.IsNullOrWhiteSpace(asrLanguageCode) ? "auto" : asrLanguageCode;
        db.UpdateMetadata("asr_language_code", effectiveAsrLanguageCode);

        // For the VibeVoice / built-in-diarization path the config can't
        // change downstream — fire the callback now so the caller (e.g.
        // JobQueueService) can mirror it into the jobs table without
        // reopening the results DB.
        if (runVibeVoice)
            onAsrConfigEffective?.Invoke(effectiveAsrModelName, effectiveAsrLanguageCode);

        // ── VibeVoice: combined diarization + ASR in a single model pass ────────
        if (runVibeVoice)
        {
            if (!db.CheckDiarization())
            {
                progress.Report(new TranscriptionProgress(
                    TranscriptionPhase.Recognizing, 0, 100, "Running VibeVoice-ASR…"));

                ct.ThrowIfCancellationRequested();
                string vibeVoiceDir  = _settings.GetVibeVoiceModelsDir();
                double vibeDuration  = (double)vibeVoiceAudio.Length / vibeVoiceSampleRate / Math.Max(1, vibeVoiceChannels);
                int    streamSegIdx  = 0;

                IReadOnlyList<VibeVoiceSegment> vibeSegs = await Task.Run(() =>
                {
                    using var vibe = new VibeVoiceAsr(
                        vibeVoiceDir,
                        persistEncoder: false,
                        allowStaticKvCache: false);
                    return vibe.Transcribe(
                        vibeVoiceAudio, vibeVoiceSampleRate, vibeVoiceChannels,
                        onSegment: seg =>
                        {
                            int    i     = streamSegIdx++;
                            string spkId = $"speaker_{seg.Speaker}";
                            onSegmentAdded(new SegmentRow
                            {
                                SegmentId          = i,
                                SpeakerTag         = spkId,
                                SpeakerDisplayName = spkId,
                                StartTime          = seg.Start,
                                EndTime            = seg.End,
                            });
                            onSegmentText(i, seg.Content);
                            double pct = vibeDuration > 0 ? seg.End / vibeDuration * 100.0 : 0;
                            progress.Report(new TranscriptionProgress(
                                TranscriptionPhase.Recognizing, 0, 100,
                                $"{seg.End:F1}s / {vibeDuration:F1}s",
                                i, seg.Content, OverridePercent: pct));
                        },
                        ct: ct);
                }, ct).ConfigureAwait(false);

                db.BeginBulkInsert();
                var seenVibeSpeakers = new HashSet<int>();
                foreach (var seg in vibeSegs)
                {
                    string spkId    = $"speaker_{seg.Speaker}";
                    int diarSpkId   = seg.Speaker + 1;

                    if (seenVibeSpeakers.Add(seg.Speaker))
                        db.InsertSpeaker(spkId);

                    // VibeVoice does not emit timestamps, so we synthesize them uniformly over
                    // the segment while preserving the real decoder token ids and logprobs.
                    int tokenCount = Math.Max(seg.TokenIds.Count, seg.TokenLogprobs.Count);
                    const double frameSeconds = Config.HopLength * 8.0 / Config.SampleRate;
                    double segDuration = seg.End - seg.Start;
                    var persistedTokens     = seg.TokenIds.Count > 0 ? seg.TokenIds.ToArray() : Array.Empty<int>();
                    var syntheticTimestamps = new int[tokenCount];
                    for (int wi = 0; wi < tokenCount; wi++)
                    {
                        syntheticTimestamps[wi] = (int)((double)wi / tokenCount * segDuration / frameSeconds);
                    }

                    db.InsertResult(
                        diarizationSpeakerId: diarSpkId,
                        speakerId:            diarSpkId,
                        startTime:            seg.Start,
                        endTime:              seg.End,
                        startTimeF:           AudioUtils.SecondsToHhMmSs(Math.Round(seg.Start)),
                        endTimeF:             AudioUtils.SecondsToHhMmSs(Math.Round(seg.End)),
                        asrContent:           seg.Content,
                        content:              seg.Content,
                        tokens:               JsonSerializer.Serialize(persistedTokens),
                        timestamps:           JsonSerializer.Serialize(syntheticTimestamps),
                        logprobs:             JsonSerializer.Serialize(
                                                  seg.TokenLogprobs.Count > 0 ? seg.TokenLogprobs : seg.WordLogprobs));
                }
                db.CommitBulkInsert();
                db.MarkDiarizationComplete();
            }
            else
            {
                // Resume path: diarization + ASR already complete — just notify the UI.
                var existingSegs  = db.GetSegments();
                var existingTexts = db.GetResultContents();
                for (int i = 0; i < existingSegs.Count; i++)
                {
                    var (start, end, spkId) = existingSegs[i];
                    onSegmentAdded(new SegmentRow
                    {
                        SegmentId          = i,
                        SpeakerTag         = spkId,
                        SpeakerDisplayName = spkId,
                        StartTime          = start,
                        EndTime            = end,
                    });
                    if (i < existingTexts.Count)
                        onSegmentText(i, existingTexts[i]);
                }
            }

            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Done, 1, 1,
                Loc.Instance["transcription_complete"], OverridePercent: 100));
            return;
        }

        // ── Phase 3: Segmentation (Diarization or VAD) ───────────────────────
        List<(double start, double end, string spkId)> segs;

        if (!db.CheckDiarization() && segmentationMode == SegmentationMode.SileroVad)
        {
            // ── Silero VAD path ──────────────────────────────────────────────
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 0, 1, Loc.Instance["progress_detecting_speech"]));

            ct.ThrowIfCancellationRequested();
            List<(double, double)> vadSegs = await Task.Run(
                () =>
                {
                    using var vad = new VadSegmenter(sileroModelsDir);
                    return vad.GetSegments(audio);
                }, ct).ConfigureAwait(false);

            segs = new List<(double, double, string)>(vadSegs.Count);
            db.InsertSpeaker("speaker_1");
            db.BeginBulkInsert();
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
            db.CommitBulkInsert();
            db.MarkDiarizationComplete();

            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 1, 1,
                Loc.Instance.T("progress_vad_complete", new() { ["count"] = segs.Count.ToString() })));
        }
        else if (!db.CheckDiarization() && segmentationMode == SegmentationMode.Sortformer)
        {
            // ── Sortformer path (batch diarization — discrete step) ───────────
            // Runs Sortformer to completion, then ASR runs separately in Phase 4.
            // This matches the CLI approach and simplifies the pipeline.
            segs = new List<(double, double, string)>();
            var seenSpeakers = new HashSet<string>();

            ct.ThrowIfCancellationRequested();

            // ONNX Runtime InferenceSession construction is expensive (graph
            // optimization, memory allocation, CUDA/DML provider setup).  Report
            // progress before it so the UI doesn't appear frozen.
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 0, 1,
                Loc.Instance["progress_loading_sortformer_model"]));

            var (streamer, melSpec, totalFrames, chunkStride, numChunks) =
                await Task.Run(() =>
                {
                    var s = new SortformerStreamer(sortformerModelsDir);
                    var m = AudioUtils.LogMelSpectrogram(audio);
                    var p = s.GetPredParams(m);
                    return (s, m, p.totalFrames, p.chunkStride, p.numChunks);
                }, ct).ConfigureAwait(false);

            // Process all chunks and collect predictions (mimics Diarize() with progress).
            var allPreds = new List<float[,]>(numChunks);
            int chunkIdx = 0;

            await Task.Run(() =>
            {
                foreach (var (nc, idx, chunkPreds) in
                    streamer.GetPreds(melSpec, totalFrames, chunkStride, numChunks))
                {
                    ct.ThrowIfCancellationRequested();
                    allPreds.Add(chunkPreds);
                    chunkIdx = idx + 1;

                    // Marshal progress callback to UI thread via the progress reporter.
                    string diarText = Loc.Instance.T("progress_identifying_speakers", new() {
                        ["i"]     = chunkIdx.ToString(),
                        ["count"] = numChunks.ToString() });
                    progress.Report(new TranscriptionProgress(
                        TranscriptionPhase.Diarizing, chunkIdx, numChunks, diarText));
                }
            }, ct).ConfigureAwait(false);

            // Filter and binarize predictions into speaker segments.
            var (numPredFrames, medFiltered) = streamer.FilterPreds(allPreds, totalFrames);
            segs = streamer.BinarizePredToSegments(numPredFrames, medFiltered);

            // Insert all segments into DB and notify UI.
            db.BeginBulkInsert();
            int segIdx = 0;
            foreach (var (start, end, spkId) in segs)
            {
                ct.ThrowIfCancellationRequested();
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
                    SegmentId          = segIdx,
                    SpeakerTag         = spkId,
                    SpeakerDisplayName = spkId,
                    StartTime          = start,
                    EndTime            = end,
                });
                segIdx++;
            }
            db.CommitBulkInsert();

            db.MarkDiarizationComplete();
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 1, 1,
                Loc.Instance.T("progress_vad_complete", new() { ["count"] = segs.Count.ToString() })));
        }
        else if (!db.CheckDiarization() && segmentationMode == SegmentationMode.DiariZen)
        {
            // ── DiariZen path (batch diarization) ────────────────────────────
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing,
                0,
                100,
                Loc.Instance["progress_diarizing_diarizen"],
                OverridePercent: 0));

            ct.ThrowIfCancellationRequested();

            string diarizenModelsDir = _settings.GetDiariZenModelsDir();
            string diarizenModel = Path.Combine(diarizenModelsDir, Config.DiariZenFile);
            if (!File.Exists(diarizenModel))
            {
                throw new FileNotFoundException(
                    $"DiariZen model not found: {diarizenModel}. " +
                    "Open Settings to review the external weights notice and configure or download the DiariZen weights.");
            }

            string diarizenEmbedderModel = Path.Combine(diarizenModelsDir, Config.DiariZenEmbedderFile);
            if (!File.Exists(diarizenEmbedderModel))
            {
                throw new FileNotFoundException(
                    $"DiariZen embedder model not found: {diarizenEmbedderModel}. " +
                    "Open Settings to review the external weights notice and configure or download the DiariZen weights.");
            }

            List<DiarizationSegment> diarSegments = [];
            double diarizationFraction = 0;
            await Task.Run(() =>
            {
                using var diarizer = new DiariZenDiarizer(diarizenModel, diarizenEmbedderModel);
                diarSegments = diarizer.Diarize(
                    audio,
                    minSpeakers: 1,
                    maxSpeakers: 8,
                    progress: message =>
                    {
                        double estimatedFraction = EstimateDiariZenDiarizationFraction(message);
                        diarizationFraction = Math.Max(diarizationFraction, estimatedFraction);
                        progress.Report(new TranscriptionProgress(
                            TranscriptionPhase.Diarizing,
                            (int)Math.Round(diarizationFraction * 100),
                            100,
                            message,
                            OverridePercent: ScaleOverallProgress(
                                0,
                                DiariZenDiarizationPercentWeight,
                                diarizationFraction)));
                    });
            }, ct).ConfigureAwait(false);

            segs = new List<(double, double, string)>(diarSegments.Count);
            var seenSpeakers = new HashSet<string>();

            db.BeginBulkInsert();
            foreach (var seg in diarSegments)
            {
                ct.ThrowIfCancellationRequested();
                string spkId = seg.Speaker.StartsWith("speaker_", StringComparison.Ordinal)
                    ? seg.Speaker
                    : $"speaker_{seg.Speaker}";

                string speakerSuffix = spkId["speaker_".Length..];
                int diarSpkId = int.TryParse(speakerSuffix, out int speakerIndex)
                    ? speakerIndex + 1
                    : 1;
                db.InsertResult(
                    diarizationSpeakerId: diarSpkId,
                    speakerId:            diarSpkId,
                    startTime:            seg.Start,
                    endTime:              seg.End,
                    startTimeF:           AudioUtils.SecondsToHhMmSs(Math.Round(seg.Start, 3)),
                    endTimeF:             AudioUtils.SecondsToHhMmSs(Math.Round(seg.End, 3)),
                    asrContent: null, content: null, tokens: null, timestamps: null, logprobs: null);

                if (seenSpeakers.Add(spkId))
                    db.InsertSpeaker(spkId);

                onSegmentAdded(new SegmentRow
                {
                    SegmentId          = segs.Count,
                    SpeakerTag         = spkId,
                    SpeakerDisplayName = spkId,
                    StartTime          = seg.Start,
                    EndTime            = seg.End,
                });
                segs.Add((seg.Start, seg.End, spkId));
            }
            db.CommitBulkInsert();
            db.MarkDiarizationComplete();

            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing,
                100,
                100,
                Loc.Instance.T("progress_diarizen_complete", new() { ["count"] = segs.Count.ToString() }),
                OverridePercent: DiariZenDiarizationPercentWeight));
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

        // ── Phase 3b: Language identification (optional) ─────────────────────
        // Runs after segmentation so we have VAD/diarizer segments to choose
        // from, and before ASR so the detected language can inform / pre-fill
        // force-language choices downstream. LangIdService gates on
        // settings + model-present, so this no-ops when LID is off or the
        // VoxLingua assets haven't been downloaded.
        if (_langId.IsAvailable && db.GetMetadata("detected_language") is null)
        {
            progress.Report(new TranscriptionProgress(
                TranscriptionPhase.Diarizing, 100, 100, "Detecting language…"));

            ct.ThrowIfCancellationRequested();
            var vadForLid = segs.Select(s => (s.start, s.end)).ToList();
            var lidResult = await Task.Run(
                () => _langId.DetectLanguage(audio, vadForLid), ct).ConfigureAwait(false);

            if (lidResult is not null)
            {
                db.InsertMetadata("detected_language",             lidResult.Iso);
                db.InsertMetadata("detected_language_name",        lidResult.Top.Name);
                db.InsertMetadata("detected_language_probability", lidResult.TopProbability.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                db.InsertMetadata("detected_language_ambiguous",   lidResult.IsAmbiguous ? "1" : "0");
                // Top-K as compact JSON so downstream UI can render the chip /
                // the ambiguity dropdown without recomputing softmax.
                string topKJson = JsonSerializer.Serialize(
                    lidResult.TopK.Select(c => new { iso = c.Iso, name = c.Name, p = c.Probability }));
                db.InsertMetadata("detected_language_topk", topKJson);
                Console.WriteLine($"[Transcription] LID → {lidResult.FormatSummary()}");

                // Override: if LID returned an unambiguous result and the
                // current ASR backend supports it, substitute it for the
                // user's configured force-language for this run. Leaves the
                // settings value untouched — this is a per-job override that
                // the Results / Editor view will see via the
                // asr_language_code metadata we record below.
                //
                // The ambiguity gate (VoxLinguaAmbiguityThreshold, 0.60) is
                // the only confidence threshold. Opting into LID is the
                // user's consent to overrides; if they want the settings
                // force-language respected verbatim, they disable LID.
                if (!lidResult.IsAmbiguous)
                {
                    var backend = AsrLanguageSupport.BackendOf(asrModelName);
                    if (backend is not null && AsrLanguageSupport.Supports(backend.Value, lidResult.Iso))
                    {
                        Console.WriteLine(
                            $"[Transcription] LID override: " +
                            $"asr_language_code '{asrLanguageCode}' → '{lidResult.Iso}'");
                        asrLanguageCode = lidResult.Iso;
                    }
                    else if (backend is not null)
                    {
                        var alt = AsrLanguageSupport.PickBestBackend(lidResult.Iso);

                        // If the host wired up the interactive callback (Avalonia
                        // does; CLI / tests don't), ask the user what to do.
                        // Otherwise fall through to the log-and-continue fallback.
                        Views.Dialogs.AsrMismatchResult? result = null;
                        if (OnAsrLanguageMismatch is not null)
                        {
                            try
                            {
                                result = await OnAsrLanguageMismatch(lidResult, backend.Value, alt).ConfigureAwait(false);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[Transcription] OnAsrLanguageMismatch callback failed: {ex.Message}");
                                // fall through to log-and-continue with current backend
                            }
                        }

                        switch (result?.Choice)
                        {
                            case Views.Dialogs.AsrMismatchChoice.SwitchBackend when alt is not null:
                                string newModelName = AsrLanguageSupport.ModelName(alt.Value);
                                Console.WriteLine(
                                    $"[Transcription] LID mismatch: user switched ASR " +
                                    $"{backend.Value} → {alt.Value}. " +
                                    $"asrModelName '{asrModelName}' → '{newModelName}'");
                                asrModelName = newModelName;
                                asrLanguageCode = lidResult.Iso;
                                // Recompute the per-backend flags so the Phase-4 ASR
                                // branch picks the right code path. (VibeVoice can't
                                // be switched TO here because its diarization path
                                // runs earlier; AsrLanguageSupport.PickBestBackend
                                // prefers Qwen3-ASR/Cohere/Parakeet in that order.)
                                useVibeVoiceAsr = string.Equals(asrModelName, "vibevoice/vibevoice-asr", StringComparison.Ordinal);
                                useCohereAsr    = string.Equals(asrModelName, "CohereLabs/cohere-transcribe-03-2026", StringComparison.Ordinal);
                                useQwen3Asr     = string.Equals(asrModelName, "Qwen/Qwen3-ASR-1.7B", StringComparison.Ordinal);
                                // Reflect the switch in the results DB so the
                                // Results view reads the *effective* backend
                                // (and so the mismatch banner doesn't appear
                                // for a mismatch the user already resolved).
                                db.UpdateMetadata("asr_model", newModelName);
                                db.InsertMetadata("asr_model_overridden_from", AsrLanguageSupport.BackendOf(AsrLanguageSupport.ModelName(backend.Value))?.ToString() ?? backend.Value.ToString());
                                break;

                            case Views.Dialogs.AsrMismatchChoice.ForceLanguage
                                when result is { ForcedIso: { Length: > 0 } pickedIso }
                                  && AsrLanguageSupport.Supports(backend.Value, pickedIso):
                                string forcedIso = AsrLanguageSupport.NormalizeIso(pickedIso);
                                Console.WriteLine(
                                    $"[Transcription] LID mismatch: user forced language " +
                                    $"'{forcedIso}' on {backend.Value} despite detected '{lidResult.Iso}'.");
                                asrLanguageCode = forcedIso;
                                // Mark the override so the Results view shows a
                                // "user-forced language" badge instead of the
                                // amber mismatch banner. Also remember what was
                                // detected so the badge can explain the gap.
                                db.InsertMetadata("asr_language_user_forced", "1");
                                db.InsertMetadata("asr_language_user_forced_from", lidResult.Iso);
                                break;

                            case Views.Dialogs.AsrMismatchChoice.ForceLanguage:
                                // Picker emitted a code the current backend
                                // doesn't actually support, or no code at all.
                                // The dialog's own filter should prevent this,
                                // so log loudly and treat as KeepCurrent.
                                Console.WriteLine(
                                    $"[Transcription] LID mismatch: ForceLanguage rejected — " +
                                    $"forcedIso='{result?.ForcedIso ?? "<null>"}' is not supported " +
                                    $"by {backend.Value}. Falling through to KeepCurrent.");
                                goto default;

                            case Views.Dialogs.AsrMismatchChoice.CancelJob:
                                Console.WriteLine("[Transcription] LID mismatch: user cancelled the job.");
                                throw new OperationCanceledException(
                                    "Job cancelled by user after LID detected an unsupported language.");

                            case Views.Dialogs.AsrMismatchChoice.KeepCurrent:
                            case null:
                            default:
                                // Mismatch is *unresolved* — record so the
                                // Results banner can offer the reprocess remedy.
                                // Set inside this branch (not before the
                                // popup) so a SwitchBackend choice doesn't
                                // leave a stale mismatch flag behind.
                                db.InsertMetadata("detected_language_backend_mismatch", "1");
                                if (alt is not null)
                                    db.InsertMetadata("detected_language_suggested_backend", alt.Value.ToString());
                                Console.WriteLine(
                                    $"[Transcription] LID detected '{lidResult.Iso}' ({lidResult.Top.Name}) " +
                                    $"but {backend.Value} does not support it. " +
                                    (alt is not null
                                        ? $"User kept {backend.Value} anyway (alt would be {alt.Value})."
                                        : $"No installed backend supports {lidResult.Iso}.") +
                                    " Continuing with the configured backend.");
                                break;
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine("[Transcription] LID skipped (no VAD segment long enough).");
            }
        }

        // Persist the effective asr_language_code that the ASR step will use,
        // so the Results / Editor views can read it (possibly differing from
        // the job's ControlDb row when LID overrode it).
        db.InsertMetadata("asr_language_code", asrLanguageCode);

        // Notify the caller of the effective ASR config (may differ from
        // what was passed in if SwitchBackend ran). Fired once here, after
        // the LID phase has settled but before Phase 4 ASR; JobQueueService
        // uses this to mirror the values into the jobs table without
        // reopening the results DB at end-of-run.
        onAsrConfigEffective?.Invoke(asrModelName, asrLanguageCode);

        ct.ThrowIfCancellationRequested();

        // ── Phase 3c: Per-segment LID (optional) ─────────────────────────────
        // Independent of the file-level LID step above. Writes lid_language
        // per row so the editor can surface code-switched / mixed-language
        // segments. Short segments (< 2 s) inherit the file-level language
        // when one was detected — VoxLingua needs a few seconds of audio to
        // reliably distinguish close cousins (ru/uk/be, mk/bg/sr). Skipped
        // entirely when the user hasn't opted in or no segments exist yet.
        // Reads result_ids from the DB so writes target the right row even
        // if positional alignment with `segs` ever drifts.
        if (_settings.Current.LidPerSegment && _langId.IsAvailable
            && db.GetMetadata("lid_per_segment_complete") != "1")
        {
            var resultRows = db.GetAllResultSegments();
            if (resultRows.Count > 0)
            {
                string? fileLevelIso = db.GetMetadata("detected_language");
                progress.Report(new TranscriptionProgress(
                    TranscriptionPhase.Diarizing, 0, resultRows.Count, "Per-segment LID…"));

                var segPairs = resultRows.Select(r => (r.start, r.end)).ToList();
                var perSegLid = await Task.Run(
                    () => _langId.ClassifyEachSegment(
                        audio, segPairs,
                        minSegmentSeconds: 2.0,
                        onProgress: (done, total) =>
                            progress.Report(new TranscriptionProgress(
                                TranscriptionPhase.Diarizing, done, total,
                                $"Per-segment LID ({done}/{total})…")),
                        ct: ct),
                    ct).ConfigureAwait(false);

                for (int i = 0; i < perSegLid.Count; i++)
                {
                    string? iso = perSegLid[i]?.Top.Iso ?? fileLevelIso;
                    if (iso is not null)
                        db.UpdateResultLidLanguage(resultId: resultRows[i].resultId, lidLanguage: iso);
                }
                db.InsertMetadata("lid_per_segment_complete", "1");
            }
        }

        ct.ThrowIfCancellationRequested();

        // ── Phase 4: ASR (resume or non-incremental path only) ───────────────
        var (asrDone, unfilledResultIds) = db.CheckAsr();

        // Populate text for any rows already filled (resume from prior run).
        // With scattered-hole resume, "already filled" is no longer a prefix, so
        // notify for every position — empty strings for unfilled rows are
        // overwritten when those rows are processed below.
        if (unfilledResultIds.Count < segs.Count)
        {
            var existingTexts = db.GetResultContents();
            for (int i = 0; i < segs.Count && i < existingTexts.Count; i++)
                onSegmentText(i, existingTexts[i]);
        }

        if (!asrDone)
        {
            // Build the subset to process from the unfilled result_ids by
            // querying `results` directly. Earlier this used segs[rid - 1]
            // — implicitly assuming `segs[i].result_id == i + 1`, which
            // holds for fresh diarization but breaks once segment_cards has
            // been split / merged by the user. Pulling the rows by id keeps
            // the data path correct regardless.
            var segsByRid = db.GetResultSegmentsByIds(unfilledResultIds);
            var segsSubset = new List<(double start, double end, string spkId)>(unfilledResultIds.Count);
            foreach (int rid in unfilledResultIds)
            {
                if (!segsByRid.TryGetValue(rid, out var entry))
                    throw new InvalidOperationException(
                        $"Unfilled result_id {rid} not present in results table.");
                segsSubset.Add(entry);
            }
            int totalSegs  = segs.Count;
            int completed  = totalSegs - segsSubset.Count;

            // Per-segment LID (issue #10): when phase 3c populated lid_language,
            // group the unfilled rows by LID-detected language and call the ASR
            // backend once per language group with the matching forceLanguage.
            // This preserves batched throughput within a group while still
            // routing each segment of a code-switched recording to its actual
            // language. Backends without forceLanguage (Parakeet) still see a
            // single all-segments group below — they ignore the per-group label.
            bool perSegmentLidAvailable =
                db.GetMetadata("lid_per_segment_complete") == "1";
            Dictionary<int, string?>? lidByRid = perSegmentLidAvailable
                ? db.GetResultLidLanguagesByIds(unfilledResultIds)
                : null;

            if (useCohereAsr)
            {
                string cohereModelsDir = _settings.GetCohereModelsDir();
                string? fileLevelForceLanguage =
                    string.Equals(asrLanguageCode, "auto", StringComparison.OrdinalIgnoreCase) ||
                    string.IsNullOrWhiteSpace(asrLanguageCode)
                        ? null
                        : asrLanguageCode;
                using var cohere = new CohereTranscribe(cohereModelsDir);

                foreach (var group in GroupSegmentsByLidLanguage(
                    unfilledResultIds, lidByRid, fileLevelForceLanguage))
                {
                    var groupSegs = group.LocalIndices
                        .Select(i => segsSubset[i]).ToList();
                    foreach (var result in cohere.RecognizeDetailed(
                        groupSegs, audio, forceLanguage: group.ForceLanguage))
                    {
                        ct.ThrowIfCancellationRequested();
                        int localIdx = group.LocalIndices[result.SegmentId];
                        int rid   = unfilledResultIds[localIdx];
                        int absId = rid - 1;
                        completed++;
                        var syntheticTimestamps = BuildSyntheticTokenTimestamps(
                            groupSegs[result.SegmentId].end - groupSegs[result.SegmentId].start,
                            result.TextTokens.Count);

                        db.UpdateResult(
                            resultId:   rid,
                            asrContent: result.Text,
                            content:    result.Text,
                            tokens:     JsonSerializer.Serialize(result.TextTokens),
                            timestamps: JsonSerializer.Serialize(syntheticTimestamps),
                            logprobs:   JsonSerializer.Serialize(result.TextLogprobs),
                            language:   result.Meta.Language,
                            emotion:    result.Meta.Emotion,
                            asrMeta:    result.Meta.ToJson(
                                rawDecoderTokens: result.RawTokens,
                                storedTextTokens: result.TextTokens,
                                syntheticTimestamps: true,
                                timestampMode: CohereSyntheticTimestampMode));

                        onSegmentText(absId, result.Text);

                        string asrText = Loc.Instance.T("progress_recognizing_segment", new() {
                            ["i"]     = completed.ToString(),
                            ["count"] = totalSegs.ToString() });
                        double? overridePercent = segmentationMode == SegmentationMode.DiariZen
                            ? ScaleOverallProgress(
                                DiariZenDiarizationPercentWeight,
                                100,
                                totalSegs > 0 ? completed / (double)totalSegs : 1)
                            : null;
                        progress.Report(new TranscriptionProgress(
                            TranscriptionPhase.Recognizing,
                            completed,
                            totalSegs,
                            asrText,
                            absId,
                            result.Text,
                            overridePercent));
                    }
                }
            }
            else if (useQwen3Asr)
            {
                string? fileLevelQwenForceLanguage =
                    string.Equals(asrLanguageCode, "auto", StringComparison.OrdinalIgnoreCase) ||
                    string.IsNullOrWhiteSpace(asrLanguageCode)
                        ? null
                        : asrLanguageCode;
                bool hasBatchedFiles = File.Exists(Path.Combine(qwen3AsrModelsDir, Qwen3Asr.EncoderBatchedFile)) &&
                                       (File.Exists(Path.Combine(qwen3AsrModelsDir, Qwen3Asr.DecoderFile)) ||
                                        File.Exists(Path.Combine(qwen3AsrModelsDir, Qwen3Asr.DecoderInitBatchedFile)));
                using var qwen3Asr = new Qwen3Asr(qwen3AsrModelsDir, preferBatched: hasBatchedFiles);

                foreach (var group in GroupSegmentsByLidLanguage(
                    unfilledResultIds, lidByRid, fileLevelQwenForceLanguage))
                {
                    var groupSegs = group.LocalIndices
                        .Select(i => segsSubset[i]).ToList();
                    var recognitionResults = hasBatchedFiles
                        ? qwen3Asr.RecognizeBatchedDetailed(groupSegs, audio, forceLanguage: group.ForceLanguage)
                        : qwen3Asr.RecognizeDetailed(groupSegs, audio, forceLanguage: group.ForceLanguage);
                    foreach (var result in recognitionResults)
                    {
                        ct.ThrowIfCancellationRequested();
                        int localIdx = group.LocalIndices[result.SegmentId];
                        int rid   = unfilledResultIds[localIdx];
                        int absId = rid - 1;
                        completed++;
                        var syntheticTimestamps = BuildSyntheticTokenTimestamps(
                            groupSegs[result.SegmentId].end - groupSegs[result.SegmentId].start,
                            result.TextTokens.Count);

                        db.UpdateResult(
                            resultId:   rid,
                            asrContent: result.Text,
                            content:    result.Text,
                            tokens:     JsonSerializer.Serialize(result.TextTokens),
                            timestamps: JsonSerializer.Serialize(syntheticTimestamps),
                            logprobs:   JsonSerializer.Serialize(result.TextLogprobs),
                            language:   result.Language,
                            asrMeta:    JsonSerializer.Serialize(new
                            {
                                raw_decoder_text = result.RawText,
                                parsed_language = result.Language,
                            }));

                        onSegmentText(absId, result.Text);

                        string asrText = Loc.Instance.T("progress_recognizing_segment", new() {
                            ["i"]     = completed.ToString(),
                            ["count"] = totalSegs.ToString() });
                        double? overridePercent = segmentationMode == SegmentationMode.DiariZen
                            ? ScaleOverallProgress(
                                DiariZenDiarizationPercentWeight,
                                100,
                                totalSegs > 0 ? completed / (double)totalSegs : 1)
                            : null;
                        progress.Report(new TranscriptionProgress(
                            TranscriptionPhase.Recognizing,
                            completed,
                            totalSegs,
                            asrText,
                            absId,
                            result.Text,
                            overridePercent));
                    }
                }
            }
            else
            {
                var (encoderFile, decoderJointFile) =
                    Config.GetAsrFiles(ModelPrecision.Fp32);

                // LM fusion requires beam search. If the user has an LM set but beam is still 1,
                // silently bump to 4 so the feature works without requiring them to also
                // remember to raise the beam.
                string lmPath       = _settings.Current.ParakeetLmPath ?? "";
                bool   lmActive     = !string.IsNullOrWhiteSpace(lmPath) && File.Exists(lmPath);
                int    effectiveBeam = lmActive && _settings.Current.ParakeetBeamWidth < 2
                    ? 4 : _settings.Current.ParakeetBeamWidth;

                using var parakeet = new ParakeetAsr(parakeetModelsDir, encoderFile, decoderJointFile,
                    beamWidth: effectiveBeam);
                if (lmActive)
                {
                    parakeet.LmScorer        = KenLmScorer.LoadArpa(lmPath);
                    parakeet.LmWeight        = _settings.Current.ParakeetLmWeight;
                    parakeet.LmLengthPenalty = _settings.Current.ParakeetLmLengthPenalty;
                }
                foreach (var (segId, text, tokens, timestamps, durations, logprobs) in
                    parakeet.Recognize(segsSubset, audio))
                {
                    ct.ThrowIfCancellationRequested();
                    int rid   = unfilledResultIds[segId];
                    int absId = rid - 1;
                    completed++;

                    db.UpdateResult(
                        resultId:   rid,
                        asrContent: text,
                        content:    text,
                        tokens:     JsonSerializer.Serialize(tokens),
                        timestamps: JsonSerializer.Serialize(timestamps),
                        logprobs:   JsonSerializer.Serialize(logprobs),
                        durations:  durations.Count > 0 ? JsonSerializer.Serialize(durations) : null);

                    onSegmentText(absId, text);

                    string asrText = Loc.Instance.T("progress_recognizing_segment", new() {
                        ["i"]     = completed.ToString(),
                        ["count"] = totalSegs.ToString() });
                    double? overridePercent = segmentationMode == SegmentationMode.DiariZen
                        ? ScaleOverallProgress(
                            DiariZenDiarizationPercentWeight,
                            100,
                            totalSegs > 0 ? completed / (double)totalSegs : 1)
                        : null;
                    progress.Report(new TranscriptionProgress(
                        TranscriptionPhase.Recognizing,
                        completed,
                        totalSegs,
                        asrText,
                        absId,
                        text,
                        overridePercent));
                }
            }
        }

        progress.Report(new TranscriptionProgress(
            TranscriptionPhase.Done,
            1,
            1,
            Loc.Instance["transcription_complete"],
            OverridePercent: 100));
    }


    private static double EstimateDiariZenDiarizationFraction(string message)
    {
        if (TryParseProgressFraction(message, "reconstructed chunk ", out double reconstructedFraction))
            return 0.94 + 0.05 * reconstructedFraction;
        if (TryParseProgressFraction(message, "segmented chunk ", out double segmentedFraction))
            return 0.15 + 0.60 * segmentedFraction;
        if (TryParseProgressFraction(message, "processed embeddings for chunk ", out double embeddingFraction))
            return 0.12 + 0.56 * embeddingFraction;
        if (TryParseProgressFraction(message, "decoded chunk ", out double decodedFraction))
            return 0.10 + 0.50 * decodedFraction;

        if (message.StartsWith("chunked audio into ", StringComparison.Ordinal))
            return 0.02;
        if (message.StartsWith("running segmentation model", StringComparison.Ordinal))
            return 0.05;
        if (message.StartsWith("extracting speaker embeddings", StringComparison.Ordinal))
            return 0.08;
        if (message.StartsWith("collected ", StringComparison.Ordinal))
            return 0.78;
        if (message.StartsWith("clustering ", StringComparison.Ordinal))
            return 0.82;
        if (message.StartsWith("AHC produced ", StringComparison.Ordinal))
            return 0.86;
        if (message.StartsWith("VBx refined clustering to ", StringComparison.Ordinal)
            || message.StartsWith("small-cluster merge produced ", StringComparison.Ordinal))
            return 0.90;
        if (message.StartsWith("clustered to ", StringComparison.Ordinal))
            return 0.92;
        if (message.StartsWith("reconstructing global speaker timeline", StringComparison.Ordinal))
            return 0.94;
        if (message.StartsWith("extracted ", StringComparison.Ordinal))
            return 1.0;

        return 0;
    }

    private static bool TryParseProgressFraction(string message, string prefix, out double fraction)
    {
        fraction = 0;
        if (!message.StartsWith(prefix, StringComparison.Ordinal))
            return false;

        ReadOnlySpan<char> remainder = message.AsSpan(prefix.Length);
        int slashIndex = remainder.IndexOf('/');
        if (slashIndex <= 0)
            return false;

        ReadOnlySpan<char> currentSpan = remainder[..slashIndex];
        ReadOnlySpan<char> totalSpan = remainder[(slashIndex + 1)..];
        int totalEnd = totalSpan.IndexOfAny(' ', ')');
        if (totalEnd >= 0)
            totalSpan = totalSpan[..totalEnd];

        if (!int.TryParse(currentSpan, out int current) || !int.TryParse(totalSpan, out int total) || total <= 0)
            return false;

        fraction = Math.Clamp(current / (double)total, 0, 1);
        return true;
    }

    private static double ScaleOverallProgress(double startPercent, double endPercent, double fraction) =>
        startPercent + (endPercent - startPercent) * Math.Clamp(fraction, 0, 1);

    /// <summary>
    /// One ASR call's worth of segments after grouping by per-segment LID.
    /// <see cref="LocalIndices"/> are positions into the unfilled-rids list
    /// (and parallel <c>segsSubset</c>) — caller maps them back to result_ids
    /// and absolute UI slots.
    /// </summary>
    private readonly record struct LidLanguageGroup(string? ForceLanguage, List<int> LocalIndices);

    /// <summary>
    /// Partition <paramref name="unfilledResultIds"/> into runs that share a
    /// forceLanguage value. When <paramref name="lidByRid"/> is null (per-segment
    /// LID didn't run), returns a single group with <paramref name="fallback"/>
    /// — preserving today's "one ASR call" behaviour. When LID is available,
    /// each rid's lid_language wins; rows without a LID value fall back to the
    /// file-level language.
    /// </summary>
    private static List<LidLanguageGroup> GroupSegmentsByLidLanguage(
        IReadOnlyList<int>          unfilledResultIds,
        Dictionary<int, string?>?   lidByRid,
        string?                     fallback)
    {
        var groups = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);
        var nullGroup = new List<int>();
        var orderedKeys = new List<string?>();

        for (int i = 0; i < unfilledResultIds.Count; i++)
        {
            int rid = unfilledResultIds[i];
            string? raw = (lidByRid is not null && lidByRid.TryGetValue(rid, out var v)) ? v : null;
            string? effective = string.IsNullOrWhiteSpace(raw) ? fallback : raw.Trim();

            if (effective is null)
            {
                if (nullGroup.Count == 0) orderedKeys.Add(null);
                nullGroup.Add(i);
            }
            else
            {
                if (!groups.TryGetValue(effective, out var bucket))
                {
                    bucket = new List<int>();
                    groups[effective] = bucket;
                    orderedKeys.Add(effective);
                }
                bucket.Add(i);
            }
        }

        var result = new List<LidLanguageGroup>(orderedKeys.Count);
        foreach (var key in orderedKeys)
            result.Add(new LidLanguageGroup(key, key is null ? nullGroup : groups[key]));
        return result;
    }

    private static List<int> BuildSyntheticTokenTimestamps(double segmentDurationSeconds, int tokenCount)
    {
        if (tokenCount <= 0)
            return [];

        const double frameSeconds = Config.HopLength * 8.0 / Config.SampleRate;
        int maxFrame = Math.Max((int)Math.Round(segmentDurationSeconds / frameSeconds), 0);
        if (tokenCount == 1)
            return [0];

        var timestamps = new List<int>(tokenCount);
        for (int i = 0; i < tokenCount; i++)
        {
            int frame = (int)Math.Round(i * maxFrame / (double)(tokenCount - 1));
            if (timestamps.Count > 0 && frame < timestamps[^1])
                frame = timestamps[^1];
            timestamps.Add(frame);
        }

        return timestamps;
    }
}
