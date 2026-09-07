using Vernacula.App.Models;
using Vernacula.App.Services.Tts;

namespace Vernacula.App.Services;

/// <summary>
/// One queued job, whatever its kind. <paramref name="AudioPath"/> is the input file (media for
/// ASR, the document for TTS) and <paramref name="DbPath"/> the results file (results DB for ASR,
/// alignment sidecar for TTS) — the names follow the jobs table's columns.
/// </summary>
internal record QueueEntry(
    int             JobId,
    string          AudioPath,
    string          DbPath,
    int             StreamIndex     = -1,
    string          AsrLanguageCode = "auto",
    string          AsrModelName    = "nvidia/parakeet-tdt-0.6b-v3",
    JobKind         Kind            = JobKind.Asr,
    TtsJobSettings? Tts             = null);

/// <summary>
/// The live per-job state a running job publishes for the panels to attach to. The queue keeps
/// one per active job and needs only the percentage from it; each kind's own panel casts back to
/// the concrete type (<see cref="JobUiState"/>, <see cref="TtsJobUiState"/>) for the rest.
/// </summary>
internal interface IJobUiState
{
    double Percent { get; }
}

/// <summary>
/// How <see cref="JobQueueService"/> runs one kind of job.
/// <para>
/// The queue owns everything that is the same for every kind — the slot, the cancellation token,
/// the running/complete/cancelled/failed ladder, the run-time stopwatch, and the live UI state
/// dictionary. A runner owns only what is specific to its kind: the worker it drives, the shape
/// of its UI state, the extra columns it writes on success, and how a stored row becomes a queue
/// entry again.
/// </para>
/// <para>
/// The point of the split is that a third job kind should be one new implementation plus one
/// registration, rather than another copy of the ladder and another branch in each of the
/// queue's accessors. See issue #129.
/// </para>
/// </summary>
internal interface IJobRunner
{
    JobKind Kind { get; }

    /// <summary>The live state object for a starting job; the queue holds it until the job ends.</summary>
    IJobUiState CreateUiState(QueueEntry entry);

    /// <summary>Rebuilds a queue entry from a stored job row, for requeue.</summary>
    QueueEntry EntryFor(JobRecord job);

    /// <summary>
    /// Runs the job to completion. Throwing <see cref="OperationCanceledException"/> cancels it;
    /// any other exception fails it. Anything the kind must persist beyond status and run time —
    /// an output duration, an effective ASR model — is written here before returning, so it lands
    /// before the queue marks the job complete.
    /// <para>
    /// <paramref name="onPercentChanged"/> asks the queue to re-read
    /// <see cref="IJobUiState.Percent"/> and publish it, so the clamping each state applies is
    /// what reaches the job list rather than the raw number the worker reported.
    /// </para>
    /// </summary>
    Task RunAsync(QueueEntry entry, IJobUiState state, Action onPercentChanged, CancellationToken ct);
}

/// <summary>Runs speech-to-text jobs through <see cref="TranscriptionService"/>.</summary>
internal sealed class AsrQueueRunner : IJobRunner
{
    private readonly TranscriptionService _transcription;
    private readonly ControlDb            _controlDb;

    public AsrQueueRunner(TranscriptionService transcription, ControlDb controlDb)
    {
        _transcription = transcription;
        _controlDb     = controlDb;
    }

    public JobKind Kind => JobKind.Asr;

    /// <summary>Full pipeline progress for the progress panel; the queue forwards it onwards.</summary>
    public event Action<int, TranscriptionProgress>? ProgressInfo;

    public IJobUiState CreateUiState(QueueEntry entry) => new JobUiState();

    public QueueEntry EntryFor(JobRecord job) =>
        new(job.JobId, job.AudioFilePath, job.ResultsFile, job.AudioStreamIndex,
            job.AsrLanguageCode, job.AsrModelName);

    public async Task RunAsync(
        QueueEntry entry, IJobUiState uiState, Action onPercentChanged, CancellationToken ct)
    {
        // Safe by construction: the queue only ever hands a runner the state that same runner
        // returned from CreateUiState.
        var state = (JobUiState)uiState;

        var progress = new Progress<TranscriptionProgress>(p =>
        {
            // Dispatch first so the state's monotonic clamp runs, then publish the clamped
            // value — otherwise home-screen job rows would still see raw percents and rewind.
            state.Dispatch(new ProgressUpdatedAction(p));
            onPercentChanged();
            ProgressInfo?.Invoke(entry.JobId, p);
        });

        void OnSegmentAdded(SegmentRow seg) =>
            state.Dispatch(new SegmentAddedAction(
                seg.SegmentId, seg.SpeakerTag, seg.SpeakerDisplayName,
                seg.StartTime, seg.EndTime));

        void OnSegmentText(int segId, string text) =>
            state.Dispatch(new SegmentTextUpdatedAction(segId, text));

        // Captures the effective ASR config (post-LID / SwitchBackend) via the
        // TranscriptionService callback so we can mirror it into the jobs table after
        // RunAsync completes — without reopening the results DB.
        string? effectiveModel = null;
        string? effectiveLang  = null;
        void OnAsrConfigEffective(string model, string lang)
        {
            effectiveModel = model;
            effectiveLang  = lang;
        }

        await _transcription.RunAsync(
            entry.AudioPath, entry.StreamIndex, entry.DbPath,
            progress,
            OnSegmentAdded,
            OnSegmentText,
            entry.AsrModelName,
            entry.AsrLanguageCode,
            ct,
            OnAsrConfigEffective);

        if (effectiveModel is not null &&
            (!string.Equals(effectiveModel, entry.AsrModelName, StringComparison.Ordinal) ||
             !string.Equals(effectiveLang ?? "auto", entry.AsrLanguageCode, StringComparison.Ordinal)))
        {
            _controlDb.UpdateJobAsr(entry.JobId, effectiveModel, effectiveLang ?? "auto");
        }
    }
}

/// <summary>Runs text-to-speech jobs through <see cref="TtsJobRunner"/>.</summary>
internal sealed class TtsQueueRunner : IJobRunner
{
    private readonly TtsJobRunner _tts;
    private readonly ControlDb    _controlDb;

    public TtsQueueRunner(TtsJobRunner tts, ControlDb controlDb)
    {
        _tts       = tts;
        _controlDb = controlDb;
    }

    public JobKind Kind => JobKind.Tts;

    /// <summary>Phase message ("synthesizing (3/12)"); the queue forwards it onwards.</summary>
    public event Action<int, ProgressEvent>? Progress;

    public IJobUiState CreateUiState(QueueEntry entry) =>
        new TtsJobUiState(TtsJobRunner.SampleRateFor(entry.Tts!));

    public QueueEntry EntryFor(JobRecord job) =>
        new(job.JobId, job.AudioFilePath, job.ResultsFile, Kind: JobKind.Tts,
            Tts: job.TtsSettings ?? new TtsJobSettings(job.TtsBackend, job.TtsLanguage, job.TtsVoice,
                                                       job.TtsSpeed, job.TtsNumStep));

    public async Task RunAsync(
        QueueEntry entry, IJobUiState uiState, Action onPercentChanged, CancellationToken ct)
    {
        // Safe by construction — see AsrQueueRunner.RunAsync.
        var state = (TtsJobUiState)uiState;

        void OnProgress(ProgressEvent p)
        {
            state.Dispatch(new TtsProgressAction(p));
            Progress?.Invoke(entry.JobId, p);
            onPercentChanged();
        }

        void OnChunk(ChunkProducedEvent ev)
        {
            state.Dispatch(new TtsChunkProducedAction(ev));
            onPercentChanged();
        }

        var sidecar = await _tts.RunAsync(entry.AudioPath, entry.DbPath, entry.Tts!, OnChunk, OnProgress, ct);
        _controlDb.UpdateJobOutputDuration(entry.JobId, sidecar.AudioDurationSeconds);
    }
}
