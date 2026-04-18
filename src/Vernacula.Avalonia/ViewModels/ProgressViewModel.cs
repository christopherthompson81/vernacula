using System.Collections.ObjectModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.App.Models;
using Vernacula.App.Services;

namespace Vernacula.App.ViewModels;

internal partial class ProgressViewModel : ObservableObject
{
    private readonly TranscriptionService _svc;
    private readonly ControlDb            _controlDb;
    private readonly SettingsService      _settings;
    private readonly JobQueueService      _queue;

    // Used by the legacy direct-run path (StartNew / LoadJob)
    private CancellationTokenSource? _cts;
    private Task?                    _runningTask;
    private int?                     _activeJobId;

    // Used by the queue-watch path (WatchJob)
    private int? _watchedJobId;

    [ObservableProperty]
    private double _progressPercent = 0;

    [ObservableProperty]
    private string _statusMessage = "";

    [ObservableProperty]
    private bool _isRunning = false;

    [ObservableProperty]
    private bool _isIndeterminate = false;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    private string? _errorMessage;

    public bool HasError => ErrorMessage != null;

    public ObservableCollection<SegmentRow> Segments { get; } = new();

    public string? CompletedResultsDbPath { get; private set; }
    public int?    CompletedJobId          { get; private set; }
    public string  CompletedAudioBaseName  { get; private set; } = "";

    public Action? NavigateToResults { get; set; }
    public Action? NavigateBack      { get; set; }

    public ProgressViewModel(
        TranscriptionService svc,
        ControlDb            controlDb,
        SettingsService      settings,
        JobQueueService      queue)
    {
        _svc       = svc;
        _controlDb = controlDb;
        _settings  = settings;
        _queue     = queue;
    }

    // ── Queue-watch path ──────────────────────────────────────────────────────

    /// <summary>
    /// Attaches the progress panel to a job that is already managed by the
    /// <see cref="JobQueueService"/> (Running or Queued).  Any previously
    /// watched job is detached first.
    /// </summary>
    public void WatchJob(int jobId, string dbPath, string audioPath, string audioBaseName)
    {
        Detach();

        CompletedJobId         = jobId;
        CompletedResultsDbPath = dbPath;
        CompletedAudioBaseName = audioBaseName;
        _watchedJobId          = jobId;

        Segments.Clear();
        IsRunning = true; // always show Cancel while watching

        // Subscribe to job lifecycle events first so we don't miss status transitions.
        _queue.JobStatusChanged += OnWatchedJobStatusChanged;

        var state = _queue.GetJobUiState(jobId);
        if (state != null)
        {
            // Atomically snapshot accumulated state and register for live actions.
            // No action can be lost in the gap between reading the snapshot and going live.
            var snapshot = state.Subscribe(action =>
                Dispatcher.UIThread.InvokeAsync(() => HandleAction(action)));

            foreach (var seg in snapshot.Segments)
                Segments.Add(seg);

            ApplyProgress(snapshot.LastProgress, snapshot.Percent);
        }
        else
        {
            // Job is queued but not yet running — no state available yet.
            IsIndeterminate = true;
            StatusMessage   = Loc.Instance["progress_queued_waiting"];
        }
    }

    private void HandleAction(JobUiAction action)
    {
        switch (action)
        {
            case SegmentAddedAction a:
                // Guard: snapshot already contains segments up to SegmentId — skip duplicates.
                if (a.SegmentId >= Segments.Count)
                    Segments.Add(new SegmentRow
                    {
                        SegmentId          = a.SegmentId,
                        SpeakerTag         = a.SpeakerTag,
                        SpeakerDisplayName = a.SpeakerDisplayName,
                        StartTime          = a.StartTime,
                        EndTime            = a.EndTime,
                    });
                break;

            case SegmentTextUpdatedAction a:
                if (a.SegmentId < Segments.Count)
                    Segments[a.SegmentId].Text = a.Text;
                break;

            case ProgressUpdatedAction a:
                ApplyProgress(a.Progress, a.Progress.Percent);
                break;
        }
    }

    private void ApplyProgress(TranscriptionProgress? p, double percent)
    {
        if (p is null)
        {
            IsIndeterminate = true;
            StatusMessage   = Loc.Instance["progress_resuming"];
            return;
        }
        ProgressPercent = percent;
        StatusMessage   = p.StatusMessage;
        IsIndeterminate = p.TotalSteps == 0 || p.Phase == TranscriptionPhase.LoadingAudio;
    }

    private void OnWatchedJobStatusChanged(int jobId, JobStatus status, string? error, int? runTimeSeconds)
    {
        if (jobId != _watchedJobId) return;
        Dispatcher.UIThread.InvokeAsync(() =>
        {
            switch (status)
            {
                case JobStatus.Running:
                    // Job just started running after being queued — subscribe to its state now.
                    var state = _queue.GetJobUiState(jobId);
                    if (state != null)
                    {
                        var snapshot = state.Subscribe(action =>
                            Dispatcher.UIThread.InvokeAsync(() => HandleAction(action)));
                        foreach (var seg in snapshot.Segments)
                            Segments.Add(seg);
                        ApplyProgress(snapshot.LastProgress, snapshot.Percent);
                    }
                    else
                    {
                        IsIndeterminate = true;
                        StatusMessage   = Loc.Instance["progress_resuming"];
                    }
                    break;

                case JobStatus.Complete:
                    IsRunning       = false;
                    IsIndeterminate = false;
                    Detach();
                    NavigateToResults?.Invoke();
                    break;

                case JobStatus.Cancelled:
                    IsRunning       = false;
                    IsIndeterminate = false;
                    StatusMessage   = Loc.Instance["progress_cancelled"];
                    Detach();
                    break;

                case JobStatus.Failed:
                    IsRunning       = false;
                    IsIndeterminate = false;
                    ErrorMessage    = error;
                    StatusMessage   = Loc.Instance.T("error_generic",
                        new() { ["error"] = error ?? "Job failed" });
                    Detach();
                    break;
            }
        });
    }

    private void Detach()
    {
        if (_watchedJobId == null) return;
        _queue.GetJobUiState(_watchedJobId.Value)?.Unsubscribe();
        _queue.JobStatusChanged -= OnWatchedJobStatusChanged;
        _watchedJobId = null;
    }

    // ── Legacy direct-run path (kept for backward compat) ────────────────────

    public async Task StartNew(string audioPath, string jobTitle)
    {
        Segments.Clear();
        IsRunning       = true;
        IsIndeterminate = true;
        StatusMessage   = Loc.Instance["progress_preparing"];
        ProgressPercent = 0;
        _cts            = new CancellationTokenSource();

        CompletedAudioBaseName = Path.GetFileNameWithoutExtension(audioPath);

        string sha256   = await Task.Run(() => AudioUtils.Sha256Checksum(audioPath));
        string dbPath   = Path.Combine(
            _settings.GetJobsDir(), $"{sha256[..16]}_results.sqlite3");
        string runStamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        string fileDateStamp = File.GetLastWriteTime(audioPath)
            .ToString("yyyy-MM-dd HH:mm:ss");
        string asrModelName = GetJobAsrModelName();
        string asrLanguageCode = GetJobAsrLanguageCode();

        int jobId = _controlDb.UpsertJob(
            jobTitle, dbPath, audioPath, sha256, fileDateStamp, runStamp, asrLanguageCode, asrModelName);

        CompletedJobId         = jobId;
        CompletedResultsDbPath = dbPath;

        await RunTranscription(audioPath, dbPath, jobId, asrModelName: asrModelName,
            asrLanguageCode: asrLanguageCode);
    }

    public async Task LoadJob(int jobId, string dbPath, string audioPath, string audioBaseName)
    {
        CompletedJobId         = jobId;
        CompletedResultsDbPath = dbPath;
        CompletedAudioBaseName = audioBaseName;

        if (_activeJobId == jobId && _runningTask is { IsCompleted: false })
        {
            IsRunning = true;
            return;
        }

        Segments.Clear();
        IsRunning       = false;
        ProgressPercent = 0;

        using var db = new TranscriptionDb(dbPath);
        var (done, _) = db.CheckAsr();

        if (done)
        {
            PopulateSegmentsFromDb(db);
            NavigateToResults?.Invoke();
            return;
        }

        IsRunning       = true;
        IsIndeterminate = true;
        StatusMessage   = Loc.Instance["progress_resuming"];
        _cts            = new CancellationTokenSource();
        string asrModelName = _controlDb.GetJobAsrModelName(jobId);
        string asrLanguageCode = _controlDb.GetJobAsrLanguageCode(jobId);
        await RunTranscription(audioPath, dbPath, jobId, asrModelName: asrModelName,
            asrLanguageCode: asrLanguageCode);
    }

    private async Task RunTranscription(
        string audioPath,
        string dbPath,
        int jobId,
        int streamIndex = -1,
        string asrModelName = "nvidia/parakeet-tdt-0.6b-v3",
        string asrLanguageCode = "auto")
    {
        _activeJobId = jobId;
        _controlDb.UpdateJobStatus(jobId, JobStatus.Running);

        var progress = new Progress<TranscriptionProgress>(p =>
        {
            ProgressPercent = p.Percent;
            StatusMessage   = p.StatusMessage;
            IsIndeterminate = p.TotalSteps == 0 || p.Phase == TranscriptionPhase.LoadingAudio;

            if (p.Phase == TranscriptionPhase.Done)
            {
                IsRunning       = false;
                IsIndeterminate = false;
                _controlDb.UpdateJobStatus(jobId, JobStatus.Complete);
                NavigateToResults?.Invoke();
            }
        });

        void OnSegmentAdded(SegmentRow seg)
        {
            Dispatcher.UIThread.InvokeAsync(() => Segments.Add(seg));
        }

        void OnSegmentText(int segId, string text)
        {
            Dispatcher.UIThread.InvokeAsync(() =>
            {
                if (segId < Segments.Count)
                    Segments[segId].Text = text;
            });
        }

        _runningTask = _svc.RunAsync(
            audioPath, streamIndex, dbPath, progress, OnSegmentAdded, OnSegmentText,
            asrModelName, asrLanguageCode, _cts!.Token);

        try
        {
            await _runningTask;
        }
        catch (OperationCanceledException)
        {
            StatusMessage   = Loc.Instance["progress_cancelled"];
            IsRunning       = false;
            IsIndeterminate = false;
            _controlDb.UpdateJobStatus(jobId, JobStatus.Cancelled);
        }
        catch (Exception ex)
        {
            ErrorMessage    = ex.ToString();
            StatusMessage   = Loc.Instance.T("error_generic", new() { ["error"] = ex.Message });
            IsRunning       = false;
            IsIndeterminate = false;
            _controlDb.UpdateJobStatus(jobId, JobStatus.Failed, ex.ToString());
        }
        finally
        {
            _activeJobId = null;
            _runningTask = null;
        }
    }

    private string GetJobAsrModelName() => _settings.Current.AsrBackend switch
    {
        AsrBackend.Cohere => "CohereLabs/cohere-transcribe-03-2026",
        AsrBackend.Qwen3Asr => "Qwen/Qwen3-ASR-1.7B",
        _                 => "nvidia/parakeet-tdt-0.6b-v3",
    };

    private string GetJobAsrLanguageCode()
    {
        if (_settings.Current.AsrBackend != AsrBackend.Cohere)
            return "auto";

        return string.IsNullOrWhiteSpace(_settings.Current.CohereLanguage)
            ? "auto"
            : _settings.Current.CohereLanguage;
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    private void PopulateSegmentsFromDb(TranscriptionDb db)
    {
        var transcript = db.GetTranscript();
        var segs       = db.GetSegments();
        for (int i = 0; i < segs.Count; i++)
        {
            var seg = segs[i];
            string text = i < transcript.Count
                ? transcript[i].GetValueOrDefault("content", "")
                : "";
            Segments.Add(new SegmentRow
            {
                SegmentId          = i,
                SpeakerTag         = seg.spkId,
                SpeakerDisplayName = i < transcript.Count
                    ? transcript[i].GetValueOrDefault("speaker_name", seg.spkId)
                    : seg.spkId,
                StartTime          = seg.start,
                EndTime            = seg.end,
                Text               = text,
            });
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        if (_watchedJobId.HasValue)
            _queue.CancelJob(_watchedJobId.Value);
        else
            _cts?.Cancel();
    }

    [RelayCommand]
    private void GoBack()
    {
        IsRunning = false;
        Detach();
        NavigateBack?.Invoke();
    }
}
