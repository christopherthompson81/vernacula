using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using Parakeet.Base;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;

namespace ParakeetCSharp.ViewModels;

internal enum AppPanel { Home, Progress, Results }

internal partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    private AppPanel _currentPanel = AppPanel.Home;

    public HomeViewModel     Home     { get; }
    public ConfigViewModel   Config   { get; }
    public ProgressViewModel Progress { get; }
    public ResultsViewModel  Results  { get; }
    public SettingsViewModel Settings { get; }

    public object? CurrentPanelViewModel => CurrentPanel switch
    {
        AppPanel.Home     => Home,
        AppPanel.Progress => Progress,
        AppPanel.Results  => Results,
        _                 => null,
    };

    partial void OnCurrentPanelChanged(AppPanel value)
    {
        OnPropertyChanged(nameof(CurrentPanelViewModel));
    }

    private readonly JobQueueService _queue;
    private Window? _mainWindow;

    public bool IsAnyJobRunning => _queue.IsAnyJobRunning || Progress.IsRunning;

    public void CancelAllJobs()
    {
        _queue.CancelAllJobs();
        Progress.CancelCommand.Execute(null);
    }

    public void SetMainWindow(Window window) => _mainWindow = window;

    public MainViewModel(
        SettingsService      settings,
        ControlDb            controlDb,
        ModelManagerService  modelManager,
        TranscriptionService transcription,
        JobQueueService      queue,
        ExportService        export)
    {
        _queue   = queue;
        Settings = new SettingsViewModel(settings, modelManager);
        Home     = new HomeViewModel(modelManager, controlDb);
        Config   = new ConfigViewModel();
        Progress = new ProgressViewModel(transcription, controlDb, settings, queue);
        Results  = new ResultsViewModel(export);

        // ── Navigation wiring ─────────────────────────────────────────────────

        // Home → Config (new transcription dialog)
        Home.NavigateToConfig = () =>
        {
            var win = new Views.Dialogs.NewTranscriptionWindow
            {
                DataContext = Config,
            };
            Config.NavigateBack = win.Close;
            win.ShowDialog(_mainWindow!);
        };

        // Home → Requeue (resume a failed / cancelled job)
        Home.RequeueJob = job =>
        {
            queue.RequeueJob(job.JobId, job.ResultsFile, job.AudioFilePath, job.AudioStreamIndex);
            RefreshJobsAndSync();
        };

        // Home → Cancel a queued or running job
        Home.CancelJob = jobId =>
        {
            queue.CancelJob(jobId);
            RefreshJobsAndSync();
        };

        // Home → Results (load a completed job)
        Home.LoadJobToResults = job =>
        {
            Results.Load(job.ResultsFile, job.AudioBaseName, _mainWindow);
            CurrentPanel = AppPanel.Results;
        };

        // Home → Progress (monitor a running / queued job)
        Home.MonitorJob = job =>
        {
            Progress.WatchJob(job.JobId, job.ResultsFile, job.AudioFilePath, job.AudioBaseName);
            CurrentPanel = AppPanel.Progress;
        };

        // Home → Bulk enqueue multiple files with auto-generated names
        Home.BulkEnqueueFiles = async (paths) =>
        {
            foreach (var path in paths)
                await queue.EnqueueFileAsync(path, Path.GetFileNameWithoutExtension(path));
            await Dispatcher.UIThread.InvokeAsync(RefreshJobsAndSync);
        };

        // Home → Bulk file picker (multi-select)
        Home.PickMultipleAudioFiles = async () =>
        {
            var files = await _mainWindow!.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title         = Loc.Instance["dlg_select_audio"],
                AllowMultiple = true,
                FileTypeFilter =
                [
                    new FilePickerFileType("Media files")
                    {
                        Patterns = ["*.wav","*.mp3","*.flac","*.m4a","*.ogg","*.aac","*.webm",
                                    "*.opus","*.wma","*.aiff","*.mp4","*.mov","*.mkv","*.avi",
                                    "*.wmv","*.flv","*.ts","*.mts","*.m2ts","*.3gp"],
                    },
                    new FilePickerFileType("All files") { Patterns = ["*"] },
                ],
            });
            return files.Count > 0 ? [.. files.Select(f => f.Path.LocalPath)] : null;
        };

        // Config → audio file picker (single-select)
        Config.PickAudioFile = async () =>
        {
            var files = await _mainWindow!.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title         = Loc.Instance["dlg_select_audio"],
                AllowMultiple = false,
                FileTypeFilter =
                [
                    new FilePickerFileType("Media files")
                    {
                        Patterns = ["*.wav","*.mp3","*.flac","*.m4a","*.ogg","*.aac","*.webm",
                                    "*.opus","*.wma","*.aiff","*.mp4","*.mov","*.mkv","*.avi",
                                    "*.wmv","*.flv","*.ts","*.mts","*.m2ts","*.3gp"],
                    },
                    new FilePickerFileType("All files") { Patterns = ["*"] },
                ],
            });
            return files.Count > 0 ? files[0].Path.LocalPath : null;
        };

        // Config → Enqueue (window closes via NavigateBack after enqueue)
        Config.EnqueueJob = async (audioPath, jobTitle) =>
        {
            Console.WriteLine($"[MainVM] EnqueueJob called with audioPath='{audioPath}', jobTitle='{jobTitle}'");
            try
            {
                await queue.EnqueueFileAsync(audioPath, jobTitle);
                await Dispatcher.UIThread.InvokeAsync(RefreshJobsAndSync);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MainVM] EnqueueJob EXCEPTION: {ex}");
            }
        };

        // Progress → Results (completed — kept for any direct Progress usage)
        Progress.NavigateToResults = () =>
        {
            if (Progress.CompletedResultsDbPath is { } dbPath)
                Results.Load(dbPath, Progress.CompletedAudioBaseName, _mainWindow);
            CurrentPanel = AppPanel.Results;
        };

        // Progress → Home (back / error)
        Progress.NavigateBack = () =>
        {
            RefreshJobsAndSync();
            CurrentPanel = AppPanel.Home;
        };

        // Results → Home (back)
        Results.NavigateBack = () =>
        {
            RefreshJobsAndSync();
            CurrentPanel = AppPanel.Home;
        };

        // Re-check model presence when precision changes or after a download
        Settings.OnPrecisionChanged = () => _ = Home.CheckModelsAsync();
        Settings.AfterDownload      = () => _ = Home.CheckModelsAsync();

        // Propagate update-available signal to the Home banner
        Settings.OnUpdateAvailable  = () =>
        {
            Home.UpdateAvailable  = true;
            Home.HasOutdatedFiles = true;
        };
        
        // Reset Home banner when update check completes with no outdated files
        Settings.OnUpdateCheckComplete = () =>
        {
            Home.UpdateAvailable  = false;
            Home.HasOutdatedFiles = false;
        };

     // Wire Home's "Open Settings" button to open the Settings window
        Home.OpenSettings = () =>
        {
            if (_mainWindow is Views.MainWindow mw)
                mw.OpenSettingsWindow();
        };

        // Startup will be triggered when MainWindow is loaded

        // ── Queue event wiring ────────────────────────────────────────────────

        queue.JobStatusChanged += (jobId, status, error, runTimeSecs) =>
        {
            Console.WriteLine($"[MainVM] JobStatusChanged event: jobId={jobId} status={status}");
            Dispatcher.UIThread.InvokeAsync(() =>
            {
                Console.WriteLine($"[MainVM] ApplyJobStatus on UI thread: jobId={jobId} status={status}");
                ApplyJobStatus(jobId, status, error, runTimeSecs);
            });
        };

        queue.JobProgressUpdated += (jobId, percent) =>
            Dispatcher.UIThread.InvokeAsync(() =>
                ApplyJobProgress(jobId, percent));

        queue.JobProgressInfoUpdated += (jobId, progress) =>
            Dispatcher.UIThread.InvokeAsync(() =>
                ApplyJobPhase(jobId, progress));

         // Startup will be triggered when MainWindow is loaded via StartAsync()
    }

    // ── Startup ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Call this after the MainWindow has been loaded to ensure UI bindings are active.
    /// </summary>
    public async Task StartAsync()
    {
        await Home.InitializeAsync();
        if (Home.ModelsReady)
            _ = Settings.CheckForUpdatesAsync();   // non-blocking; skipped if offline
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Reloads the Jobs list from the database and then synchronises all
    /// transient runtime state (progress, phase, indeterminate flag) that
    /// only the queue service knows about.
    /// </summary>
    private void RefreshJobsAndSync()
    {
        Console.WriteLine("[MainVM] RefreshJobsAndSync called");
        Home.RefreshJobs();
        Console.WriteLine($"[MainVM] Home.Jobs count after RefreshJobs: {Home.Jobs.Count}");
        foreach (var job in Home.Jobs)
        {
            if (_queue.IsJobActivelyRunning(job.JobId))
            {
                job.IsActivelyRunning = true;
                job.ProgressPercent   = _queue.GetJobProgress(job.JobId);

                if (_queue.GetJobLastProgress(job.JobId) is { } p)
                    ApplyJobPhase(job.JobId, p);
                else
                    job.IsIndeterminate = true; // running but no progress event yet
            }
        }
    }

    private void ApplyJobStatus(int jobId, JobStatus status, string? error = null,
                                int? runTimeSeconds = null)
    {
        Console.WriteLine($"[MainVM] ApplyJobStatus jobId={jobId} status={status}");
        var job = Home.Jobs.FirstOrDefault(j => j.JobId == jobId);
        if (job == null) { Console.WriteLine($"[MainVM] ApplyJobStatus: job not found in Home.Jobs (count={Home.Jobs.Count})"); return; } // will be picked up on next RefreshJobsAndSync

        job.Status            = status;
        job.IsActivelyRunning = status == JobStatus.Running;
        if (error != null)          job.ErrorMessage   = error;
        if (runTimeSeconds != null) job.RunTimeSeconds = runTimeSeconds;

        if (status == JobStatus.Running)
        {
            job.IsIndeterminate = true; // refined when first progress event fires
        }
        else if (status is JobStatus.Complete or JobStatus.Failed or JobStatus.Cancelled)
        {
            job.ProgressPercent = status == JobStatus.Complete ? 100 : job.ProgressPercent;
            job.PhaseLabel      = "";
            job.IsIndeterminate = false;
        }
    }

    private void ApplyJobProgress(int jobId, double percent)
    {
        var job = Home.Jobs.FirstOrDefault(j => j.JobId == jobId);
        if (job != null)
            job.ProgressPercent = percent;
    }

    private void ApplyJobPhase(int jobId, TranscriptionProgress progress)
    {
        var job = Home.Jobs.FirstOrDefault(j => j.JobId == jobId);
        if (job == null) return;

        job.PhaseLabel = progress.Phase switch
        {
            TranscriptionPhase.LoadingAudio => Loc.Instance["phase_audio_analysis"],
            TranscriptionPhase.Diarizing    => Loc.Instance["phase_audio_analysis"],
            TranscriptionPhase.Recognizing  => Loc.Instance["phase_speech_recognition"],
            _                               => ""
        };
        job.IsIndeterminate = progress.TotalSteps == 0 || progress.Phase == TranscriptionPhase.LoadingAudio;
    }
}
