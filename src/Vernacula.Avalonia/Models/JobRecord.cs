using CommunityToolkit.Mvvm.ComponentModel;
using Avalonia;
using Avalonia.Media;
using Vernacula.App;

namespace Vernacula.App.Models;

public enum JobStatus { Pending, Queued, Running, Complete, Failed, Cancelled }

public class JobRecord : ObservableObject
{
    public JobRecord()
    {
    }

    public int     JobId                     { get; set; }

    private string _jobTitle = "";
    public string JobTitle
    {
        get => _jobTitle;
        set => SetProperty(ref _jobTitle, value);
    }
    public string  ResultsFile               { get; set; } = "";
    public string  AudioFilePath             { get; set; } = "";
    public string  AudioFileSha256Sum        { get; set; } = "";
    public string  AsrModelName              { get; set; } = "nvidia/parakeet-tdt-0.6b-v3";
    public string  AsrLanguageCode           { get; set; } = "auto";
    public string? AudioFileDatestamp        { get; set; }
    public string? TranscriptionRunDatestamp { get; set; }
    public DateTime? TranscriptionRunStartedAt { get; set; }
    public string  CreatedAt                 { get; set; } = "";

    private string? _errorMessage;
    public string? ErrorMessage
    {
        get => _errorMessage;
        set
        {
            if (SetProperty(ref _errorMessage, value))
                OnPropertyChanged(nameof(HasError));
        }
    }
    public bool HasError => ErrorMessage != null;
    /// <summary>
    /// Index of the audio stream within the source file decoded by FFmpeg.
    /// -1 means a single-stream audio file handled by NAudio (legacy / default).
    /// </summary>
    public int     AudioStreamIndex          { get; set; } = -1;

    private JobStatus _status = JobStatus.Pending;
    public JobStatus Status
    {
        get => _status;
        set
        {
            if (SetProperty(ref _status, value))
            {
                OnPropertyChanged(nameof(IsResumable));
                OnPropertyChanged(nameof(IsLoadable));
                OnPropertyChanged(nameof(IsCancellable));
                OnPropertyChanged(nameof(IsDeletable));
                OnPropertyChanged(nameof(ShowProgress));
                OnPropertyChanged(nameof(StatusLabel));
                OnPropertyChanged(nameof(StatusBrush));
            }
        }
    }

    private double _progressPercent;
    public double ProgressPercent
    {
        get => _progressPercent;
        set => SetProperty(ref _progressPercent, Math.Clamp(value, 0, 100));
    }

    private bool _isActivelyRunning;
    public bool IsActivelyRunning
    {
        get => _isActivelyRunning;
        set => SetProperty(ref _isActivelyRunning, value);
    }

    private string _phaseLabel = "";
    /// <summary>Human-readable phase name shown while the job is actively running.</summary>
    public string PhaseLabel
    {
        get => _phaseLabel;
        set => SetProperty(ref _phaseLabel, value);
    }

    private bool _isIndeterminate;
    public bool IsIndeterminate
    {
        get => _isIndeterminate;
        set => SetProperty(ref _isIndeterminate, value);
    }

    public string StatusLabel => Status switch
    {
        JobStatus.Complete  => Loc.Instance["status_complete"],
        JobStatus.Running   => Loc.Instance["status_running"],
        JobStatus.Failed    => Loc.Instance["status_failed"],
        JobStatus.Cancelled => Loc.Instance["status_cancelled"],
        JobStatus.Pending   => Loc.Instance["status_pending"],
        JobStatus.Queued    => Loc.Instance["status_queued"],
        _                   => Status.ToString().ToLowerInvariant(),
    };

    public IBrush StatusBrush
    {
        get
        {
            string key = Status switch
            {
                JobStatus.Complete  => "GreenBrush",
                JobStatus.Failed    => "RedBrush",
                JobStatus.Cancelled => "YellowBrush",
                JobStatus.Running   => "AccentBrush",
                JobStatus.Queued    => "AccentBrush",
                _                   => "SubtextBrush",
            };

            var app = Application.Current;
            if (app?.Resources.TryGetResource(key, null, out var resourceValue) == true)
            {
                if (resourceValue is IBrush brush)
                    return brush;

                if (resourceValue is Color color)
                    return new SolidColorBrush(color);
            }

            return Brushes.Transparent;
        }
    }

    public string ResumeLabel => Loc.Instance["btn_resume"];
    public string MonitorLabel => Loc.Instance["btn_monitor"];
    public string PauseLabel => Loc.Instance["btn_pause"];
    public string LoadLabel => Loc.Instance["btn_load"];
    public string RemoveLabel => Loc.Instance["btn_remove"];

    public string AudioBaseName =>
        Path.GetFileName(AudioFilePath);

    public string DisplayDate =>
        TranscriptionRunDatestamp ?? CreatedAt;

    public bool IsResumable =>
        Status is JobStatus.Failed or JobStatus.Cancelled;

    public bool IsCancellable =>
        Status is JobStatus.Running or JobStatus.Queued;

    public bool IsDeletable =>
        Status is not (JobStatus.Running or JobStatus.Queued);

    public bool IsLoadable =>
        Status == JobStatus.Complete;

    public bool ShowProgress =>
        Status is JobStatus.Running or JobStatus.Queued or JobStatus.Failed or JobStatus.Cancelled;

    private int? _runTimeSeconds;
    public int? RunTimeSeconds
    {
        get => _runTimeSeconds;
        set
        {
            if (SetProperty(ref _runTimeSeconds, value))
                OnPropertyChanged(nameof(RunTimeLabel));
        }
    }

    public string RunTimeLabel
    {
        get
        {
            if (_runTimeSeconds is not { } secs) return "";
            if (secs < 60)   return $"{secs}s";
            if (secs < 3600) return $"{secs / 60}m {secs % 60:D2}s";
            return $"{secs / 3600}h {secs % 3600 / 60}m";
        }
    }

    public void RefreshThemeBindings()
    {
        OnPropertyChanged(nameof(StatusBrush));
    }

    public void RefreshLocalizedText()
    {
        OnPropertyChanged(nameof(StatusLabel));
        OnPropertyChanged(nameof(ResumeLabel));
        OnPropertyChanged(nameof(MonitorLabel));
        OnPropertyChanged(nameof(PauseLabel));
        OnPropertyChanged(nameof(LoadLabel));
        OnPropertyChanged(nameof(RemoveLabel));
    }
}
