using System.Security.Cryptography;
using System.Text;
using Vernacula.App.Models;
using Vernacula.App.Services.Tts;

namespace Vernacula.App.Services;

/// <summary>
/// The one queue behind the Home screen's job list, for jobs of every kind.
/// <para>
/// The queue owns the slots, the cancellation tokens, the running → complete/cancelled/failed
/// ladder, the run-time bookkeeping, the status/progress events and the per-job live UI state
/// the panels attach to. What differs per kind — the worker, the shape of that live state, the
/// extra columns written on success — belongs to an <see cref="IJobRunner"/>, one per
/// <see cref="JobKind"/>, so nothing here branches on kind.
/// </para>
/// </summary>
internal sealed class JobQueueService
{
    private readonly ControlDb       _controlDb;
    private readonly SettingsService _settings;

    /// <summary>One runner per job kind; the queue never branches on kind itself.</summary>
    private readonly Dictionary<JobKind, IJobRunner> _runners;

    private readonly SemaphoreSlim _slots;
    private readonly object        _lock        = new();
    private readonly Queue<QueueEntry>                        _pendingQueue = new();
    private readonly Dictionary<int, CancellationTokenSource> _activeCts    = new();
    private readonly Dictionary<int, IJobUiState>             _uiStates     = new();

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

    /// <summary>
    /// Fired on any thread with a TTS job's phase message ("synthesizing (3/12)"). The Home
    /// grid's Progress column shows it; the reader panel attaches to the job's UI state instead.
    /// </summary>
    public event Action<int, ProgressEvent>? JobTtsProgressUpdated;

    public JobQueueService(
        TranscriptionService transcription,
        TtsJobRunner         tts,
        ControlDb            controlDb,
        SettingsService      settings,
        int                  slotCount = 1)
    {
        _controlDb = controlDb;
        _settings  = settings;
        SlotCount  = slotCount;
        _slots     = new SemaphoreSlim(slotCount, slotCount);

        // Each runner's own typed progress event is forwarded to the queue's, so that
        // subscribers keep one place to attach to while the run ladder stays kind-agnostic.
        var asr = new AsrQueueRunner(transcription, controlDb);
        asr.ProgressInfo += (id, p) => JobProgressInfoUpdated?.Invoke(id, p);
        var ttsRunner = new TtsQueueRunner(tts, controlDb);
        ttsRunner.Progress += (id, p) => JobTtsProgressUpdated?.Invoke(id, p);
        _runners = new Dictionary<JobKind, IJobRunner>
        {
            [asr.Kind]       = asr,
            [ttsRunner.Kind] = ttsRunner,
        };

        // A kind with no runner would otherwise surface as a KeyNotFoundException on a
        // background thread the first time someone enqueues one — long after the mistake, and
        // nowhere near it. Fail at startup instead.
        foreach (var kind in Enum.GetValues<JobKind>())
            if (!_runners.ContainsKey(kind))
                throw new InvalidOperationException(
                    $"No IJobRunner is registered for JobKind.{kind}. Add one in JobRunner.cs "
                  + "and register it here.");
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
        string asrModelName = AsrLanguageSupport.ModelName(_settings.Current.AsrBackend);
        string asrLanguageCode = AsrLanguageSupport.LanguageCode(_settings.Current);

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
    /// Creates a text-to-speech job for <paramref name="documentPath"/> with 'queued' status and
    /// adds it to the run queue. Returns the new job ID. The output lands in the jobs directory
    /// as <c>{sha16}_{settings8}_tts.json</c> (alignment sidecar) + <c>.wav</c>, keyed on both the
    /// document and the rendering choices so the same text in two voices makes two jobs.
    /// </summary>
    public async Task<int> EnqueueNewTtsJobAsync(string documentPath, string jobTitle, TtsJobSettings tts)
    {
        Console.WriteLine($"[Queue] EnqueueNewTtsJobAsync starting for '{documentPath}'");
        string sha256 = await Task.Run(() => AudioUtils.Sha256Checksum(documentPath));
        string sidecarPath = Path.Combine(_settings.GetJobsDir(), TtsResultsFileName(sha256, tts));
        string fileDateStamp = File.GetLastWriteTime(documentPath).ToString("yyyy-MM-dd HH:mm:ss");

        int jobId = _controlDb.InsertNewTtsJob(jobTitle, sidecarPath, documentPath, sha256, fileDateStamp, tts);
        Console.WriteLine($"[Queue] TTS job inserted with ID: {jobId}");

        Enqueue(new QueueEntry(jobId, documentPath, sidecarPath, Kind: JobKind.Tts, Tts: tts));
        JobStatusChanged?.Invoke(jobId, JobStatus.Queued, null, null);
        _ = TryStartNextAsync();
        return jobId;
    }

    /// <summary>Sidecar file name for a TTS job — see <see cref="EnqueueNewTtsJobAsync"/>.</summary>
    internal static string TtsResultsFileName(string documentSha256, TtsJobSettings tts)
    {
        // Invariant formatting: the key names a file that must stay the same under any UI culture.
        string settingsKey = string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{tts.Backend}|{tts.Language}|{tts.Voice}|{tts.Speed:F2}|{tts.NumStep}");
        string settingsHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(settingsKey)))[..8].ToLowerInvariant();
        return $"{documentSha256[..16]}_{settingsHash}_tts.json";
    }

    /// <summary>Re-adds an existing (failed / cancelled) job of any kind back into the run queue.</summary>
    public void RequeueJob(JobRecord job)
    {
        _controlDb.UpdateJobStatus(job.JobId, JobStatus.Queued);
        Enqueue(_runners[job.Kind].EntryFor(job));
        JobStatusChanged?.Invoke(job.JobId, JobStatus.Queued, null, null);
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
        lock (_lock) return _uiStates.TryGetValue(jobId, out var s) ? s.Percent : 0;
    }

    public ProgressEvent? GetTtsJobLastProgress(int jobId) => GetTtsJobUiState(jobId)?.LastProgress;

    /// <summary>
    /// Returns the live UI state for an actively running TTS job (chunks so far + progress),
    /// or null if the job is not currently running. See <see cref="GetJobUiState"/>.
    /// </summary>
    public TtsJobUiState? GetTtsJobUiState(int jobId)
    {
        lock (_lock) return _uiStates.GetValueOrDefault(jobId) as TtsJobUiState;
    }

    public TranscriptionProgress? GetJobLastProgress(int jobId) => GetJobUiState(jobId)?.LastProgress;

    /// <summary>
    /// Returns the live UI state for an actively running job, or null if the
    /// job is not currently running.  The caller can subscribe to receive
    /// incremental actions after atomically snapshotting current state.
    /// </summary>
    public JobUiState? GetJobUiState(int jobId)
    {
        lock (_lock) return _uiStates.GetValueOrDefault(jobId) as JobUiState;
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

    /// <summary>
    /// The one run ladder, shared by every job kind: take a cancellation token and a live UI
    /// state, mark the job running, time it, and land it on complete / cancelled / failed. The
    /// kind-specific work is entirely inside <see cref="IJobRunner.RunAsync"/>.
    /// </summary>
    private async Task RunJobAsync(QueueEntry entry)
    {
        Console.WriteLine($"[Queue] RunJobAsync starting for job {entry.JobId} ({entry.Kind})");
        var runner = _runners[entry.Kind];
        var cts    = new CancellationTokenSource();
        var state  = runner.CreateUiState(entry);
        lock (_lock) { _activeCts[entry.JobId] = cts; _uiStates[entry.JobId] = state; }

        string runStamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        _controlDb.SetJobRunning(entry.JobId, runStamp);
        JobStatusChanged?.Invoke(entry.JobId, JobStatus.Running, null, null);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        void OnPercentChanged() => JobProgressUpdated?.Invoke(entry.JobId, state.Percent);

        try
        {
            JobStatus status;
            string?   error = null;
            try
            {
                await runner.RunAsync(entry, state, OnPercentChanged, cts.Token);
                status = JobStatus.Complete;
            }
            catch (OperationCanceledException)
            {
                status = JobStatus.Cancelled;
            }
            catch (Exception ex)
            {
                status = JobStatus.Failed;
                error  = ex.ToString();
            }

            sw.Stop();
            int elapsed = (int)sw.Elapsed.TotalSeconds;
            _controlDb.UpdateJobStatus(entry.JobId, status, error, runTimeSeconds: elapsed);
            lock (_lock) _uiStates.Remove(entry.JobId);
            JobStatusChanged?.Invoke(entry.JobId, status, error, elapsed);
        }
        finally
        {
            // After the status event, as it always has been: a subscriber that asks
            // IsJobActivelyRunning while handling the completion still sees the job as active.
            lock (_lock) _activeCts.Remove(entry.JobId);
        }
    }
}
