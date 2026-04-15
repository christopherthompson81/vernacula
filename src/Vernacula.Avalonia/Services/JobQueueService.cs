using Vernacula.App.Models;

namespace Vernacula.App.Services;

internal sealed class JobQueueService
{
    private readonly TranscriptionService _transcription;
    private readonly ControlDb            _controlDb;
    private readonly SettingsService      _settings;

    private readonly SemaphoreSlim _slots;
    private readonly object        _lock        = new();
    private readonly Queue<QueueEntry>                        _pendingQueue   = new();
    private readonly Dictionary<int, CancellationTokenSource> _activeCts      = new();
    private readonly Dictionary<int, JobUiState>              _jobUiStates    = new();

    /// <summary>Number of jobs that may run concurrently.</summary>
    public int SlotCount { get; }

    /// <summary>
    /// Fired on any thread when a job's status changes.
    /// Subscribers must marshal to the UI thread themselves.
    /// </summary>
    public event Action<int, JobStatus, string?, int?>? JobStatusChanged;

    /// <summary>
    /// Fired on any thread when a running job's progress percentage changes.
    /// Subscribers must marshal to the UI thread themselves.
    /// </summary>
    public event Action<int, double>? JobProgressUpdated;

    /// <summary>
    /// Fired on any thread with full pipeline progress details (for the progress panel).
    /// Subscribers must marshal to the UI thread themselves.
    /// </summary>
    public event Action<int, TranscriptionProgress>? JobProgressInfoUpdated;

    public JobQueueService(
        TranscriptionService transcription,
        ControlDb            controlDb,
        SettingsService      settings,
        int                  slotCount = 1)
    {
        _transcription = transcription;
        _controlDb     = controlDb;
        _settings      = settings;
        SlotCount      = slotCount;
        _slots         = new SemaphoreSlim(slotCount, slotCount);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Smart entry point for any media file.  For video containers with multiple
    /// audio streams, automatically creates one job per stream.  For all other
    /// files (audio-only, or single-stream video), creates exactly one job.
    /// Returns the list of job IDs that were created.
    /// </summary>
    public async Task<List<int>> EnqueueFileAsync(string filePath, string title)
    {
        if (FFmpegDecoder.VideoExtensions.Contains(Path.GetExtension(filePath)))
        {
            var streams = await Task.Run(() => FFmpegDecoder.ProbeAudioStreams(filePath));

            if (streams.Count == 0)
                return []; // video with no audio — nothing to transcribe

            var ids = new List<int>(streams.Count);
            foreach (var stream in streams)
            {
                string label    = stream.Language ?? stream.Title ?? $"Stream {stream.StreamIndex}";
                string jobTitle = streams.Count == 1 ? title : $"{title} ({label})";
                ids.Add(await EnqueueNewJobAsync(filePath, jobTitle, stream.StreamIndex));
            }
            return ids;
        }

        return [await EnqueueNewJobAsync(filePath, title)];
    }

    /// <summary>
    /// Computes the audio hash, inserts a new job record with 'queued' status,
    /// adds it to the run queue, and returns the new job ID.
    /// </summary>
    public async Task<int> EnqueueNewJobAsync(
        string audioPath, string jobTitle, int streamIndex = -1)
    {
        Console.WriteLine($"[Queue] EnqueueNewJobAsync starting for '{audioPath}'");
        string sha256 = await Task.Run(() => AudioUtils.Sha256Checksum(audioPath));
        Console.WriteLine($"[Queue] SHA256 computed: {sha256[..8]}...");

        // Each stream from the same file gets its own results database
        string dbName = streamIndex >= 0
            ? $"{sha256[..16]}_s{streamIndex}_results.sqlite3"
            : $"{sha256[..16]}_results.sqlite3";
        string dbPath = Path.Combine(_settings.GetJobsDir(), dbName);
        Console.WriteLine($"[Queue] DB path: {dbPath}");

        string fileDateStamp = File.GetLastWriteTime(audioPath)
            .ToString("yyyy-MM-dd HH:mm:ss");
        string asrModelName = GetJobAsrModelName();
        string asrLanguageCode = GetJobAsrLanguageCode();

        Console.WriteLine("[Queue] Inserting job into database...");
        int jobId = _controlDb.InsertNewJob(
            jobTitle, dbPath, audioPath, sha256, fileDateStamp, streamIndex, asrLanguageCode, asrModelName);
        Console.WriteLine($"[Queue] Job inserted with ID: {jobId}");

        Enqueue(new QueueEntry(jobId, audioPath, dbPath, streamIndex, asrLanguageCode, asrModelName));
        JobStatusChanged?.Invoke(jobId, JobStatus.Queued, null, null);
        Console.WriteLine("[Queue] Firing JobStatusChanged (Queued) and TryStartNextAsync");
        _ = TryStartNextAsync();

        Console.WriteLine("[Queue] EnqueueNewJobAsync complete");
        return jobId;
    }

    /// <summary>
    /// Re-adds an existing (failed / cancelled) job back into the run queue.
    /// </summary>
    public void RequeueJob(int jobId, string dbPath, string audioPath, int streamIndex = -1,
        string asrLanguageCode = "auto", string asrModelName = "nvidia/parakeet-tdt-0.6b-v3")
    {
        _controlDb.UpdateJobStatus(jobId, JobStatus.Queued);
        Enqueue(new QueueEntry(jobId, audioPath, dbPath, streamIndex, asrLanguageCode, asrModelName));
        JobStatusChanged?.Invoke(jobId, JobStatus.Queued, null, null);
        _ = TryStartNextAsync();
    }

    /// <summary>Requests cancellation of a running or queued job.</summary>
    public void CancelJob(int jobId)
    {
        // Remove from pending queue if not yet started
        lock (_lock)
        {
            var remaining = _pendingQueue
                .Where(e => e.JobId != jobId).ToArray();
            _pendingQueue.Clear();
            foreach (var e in remaining) _pendingQueue.Enqueue(e);
        }

        // Cancel if actively running
        CancellationTokenSource? cts;
        lock (_lock) _activeCts.TryGetValue(jobId, out cts);
        cts?.Cancel();

        // If it was only queued (not yet running), update status immediately
        bool wasActive;
        lock (_lock) wasActive = _activeCts.ContainsKey(jobId);
        if (!wasActive)
        {
            _controlDb.UpdateJobStatus(jobId, JobStatus.Cancelled);
            JobStatusChanged?.Invoke(jobId, JobStatus.Cancelled, null, null);
        }
    }

    /// <summary>Cancels all active and pending jobs.</summary>
    public void CancelAllJobs()
    {
        lock (_lock)
        {
            _pendingQueue.Clear();
            foreach (var cts in _activeCts.Values)
                cts.Cancel();
        }
    }

    public bool IsAnyJobRunning
    {
        get { lock (_lock) return _activeCts.Count > 0; }
    }

    public bool IsJobActivelyRunning(int jobId)
    {
        lock (_lock) return _activeCts.ContainsKey(jobId);
    }

    public double GetJobProgress(int jobId)
    {
        lock (_lock) return _jobUiStates.TryGetValue(jobId, out var s) ? s.Percent : 0;
    }

    public TranscriptionProgress? GetJobLastProgress(int jobId)
    {
        lock (_lock) return _jobUiStates.TryGetValue(jobId, out var s) ? s.LastProgress : null;
    }

    /// <summary>
    /// Returns the live UI state for an actively running job, or null if the
    /// job is not currently running.  The caller can subscribe to receive
    /// incremental actions after atomically snapshotting current state.
    /// </summary>
    public JobUiState? GetJobUiState(int jobId)
    {
        lock (_lock) return _jobUiStates.TryGetValue(jobId, out var s) ? s : null;
    }

    // ── Internal queue mechanics ──────────────────────────────────────────────

    private void Enqueue(QueueEntry entry)
    {
        lock (_lock) _pendingQueue.Enqueue(entry);
    }

    private async Task TryStartNextAsync()
    {
        Console.WriteLine("[Queue] TryStartNextAsync called");
        if (!_slots.Wait(0)) { Console.WriteLine("[Queue] TryStartNextAsync: no slot available"); return; }
        Console.WriteLine("[Queue] TryStartNextAsync: acquired slot");

        QueueEntry? entry;
        lock (_lock)
        {
            if (!_pendingQueue.TryDequeue(out entry))
            {
                Console.WriteLine("[Queue] TryStartNextAsync: queue empty");
                _slots.Release();
                return;
            }
            Console.WriteLine($"[Queue] TryStartNextAsync: dequeued job {entry.JobId}");
        }

        try
        {
            await RunJobAsync(entry);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Queue] RunJobAsync EXCEPTION: {ex}");
        }
        finally
        {
            _slots.Release();
            Console.WriteLine("[Queue] TryStartNextAsync: slot released, recursing");
            _ = TryStartNextAsync();
        }
    }

    private async Task RunJobAsync(QueueEntry entry)
    {
        Console.WriteLine($"[Queue] RunJobAsync starting for job {entry.JobId}");
        var cts   = new CancellationTokenSource();
        var state = new JobUiState();
        lock (_lock) { _activeCts[entry.JobId] = cts; _jobUiStates[entry.JobId] = state; }

        string runStamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        Console.WriteLine($"[Queue] Calling SetJobRunning for job {entry.JobId}");
        _controlDb.SetJobRunning(entry.JobId, runStamp);
        Console.WriteLine($"[Queue] Firing JobStatusChanged (Running) for job {entry.JobId}");
        JobStatusChanged?.Invoke(entry.JobId, JobStatus.Running, null, null);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        var progress = new Progress<TranscriptionProgress>(p =>
        {
            state.Dispatch(new ProgressUpdatedAction(p));
            JobProgressUpdated?.Invoke(entry.JobId, p.Percent);
            JobProgressInfoUpdated?.Invoke(entry.JobId, p);
        });

        void OnSegmentAdded(SegmentRow seg) =>
            state.Dispatch(new SegmentAddedAction(
                seg.SegmentId, seg.SpeakerTag, seg.SpeakerDisplayName,
                seg.StartTime, seg.EndTime));

        void OnSegmentText(int segId, string text) =>
            state.Dispatch(new SegmentTextUpdatedAction(segId, text));

        try
        {
            await _transcription.RunAsync(
                entry.AudioPath, entry.StreamIndex, entry.DbPath,
                progress,
                OnSegmentAdded,
                OnSegmentText,
                entry.AsrModelName,
                entry.AsrLanguageCode,
                cts.Token);

            sw.Stop();
            int elapsed = (int)sw.Elapsed.TotalSeconds;
            _controlDb.UpdateJobStatus(entry.JobId, JobStatus.Complete, runTimeSeconds: elapsed);
            lock (_lock) { _jobUiStates.Remove(entry.JobId); }
            JobStatusChanged?.Invoke(entry.JobId, JobStatus.Complete, null, elapsed);
        }
        catch (OperationCanceledException)
        {
            sw.Stop();
            _controlDb.UpdateJobStatus(entry.JobId, JobStatus.Cancelled,
                runTimeSeconds: (int)sw.Elapsed.TotalSeconds);
            lock (_lock) { _jobUiStates.Remove(entry.JobId); }
            JobStatusChanged?.Invoke(entry.JobId, JobStatus.Cancelled, null, (int)sw.Elapsed.TotalSeconds);
        }
        catch (Exception ex)
        {
            sw.Stop();
            string error = ex.ToString();
            _controlDb.UpdateJobStatus(entry.JobId, JobStatus.Failed, error,
                runTimeSeconds: (int)sw.Elapsed.TotalSeconds);
            lock (_lock) { _jobUiStates.Remove(entry.JobId); }
            JobStatusChanged?.Invoke(entry.JobId, JobStatus.Failed, error, (int)sw.Elapsed.TotalSeconds);
        }
        finally
        {
            lock (_lock) _activeCts.Remove(entry.JobId);
        }
    }

    private string GetJobAsrModelName() => _settings.Current.AsrBackend switch
    {
        AsrBackend.Cohere     => "CohereLabs/cohere-transcribe-03-2026",
        AsrBackend.VibeVoice  => "vibevoice/vibevoice-asr",
        _                     => "nvidia/parakeet-tdt-0.6b-v3",
    };

    private string GetJobAsrLanguageCode()
    {
        if (_settings.Current.AsrBackend != AsrBackend.Cohere)
            return "auto";

        return string.IsNullOrWhiteSpace(_settings.Current.CohereLanguage)
            ? "auto"
            : _settings.Current.CohereLanguage;
    }

    private record QueueEntry(
        int JobId,
        string AudioPath,
        string DbPath,
        int StreamIndex = -1,
        string AsrLanguageCode = "auto",
        string AsrModelName = "nvidia/parakeet-tdt-0.6b-v3");
}
