using CommunityToolkit.Mvvm.ComponentModel;
using Avalonia;
using Avalonia.Media;
using Vernacula.App;

namespace Vernacula.App.Models;

public enum JobStatus { Pending, Queued, Running, Complete, Failed, Cancelled }

/// <summary>
/// What a job does. Persisted as its lowercase name in the jobs table (job_kind), so new
/// values must not rename existing ones.
/// </summary>
public enum JobKind
{
    /// <summary>Speech → text: an audio/video file transcribed into a results database.</summary>
    Asr,
    /// <summary>Text → speech: a text/markdown document synthesized into a WAV + alignment sidecar.</summary>
    Tts,
}

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
    public JobKind Kind                      { get; set; } = JobKind.Asr;

    /// <summary>
    /// ASR: the results SQLite database. TTS: the alignment sidecar JSON; the rendered WAV sits
    /// beside it (see <see cref="OutputAudioPath"/>).
    /// </summary>
    public string  ResultsFile               { get; set; } = "";
    /// <summary>
    /// The job's input file: the media file for ASR, the text/markdown document for TTS. The
    /// column keeps its original name (audio_file_path) so older databases open unchanged.
    /// </summary>
    public string  AudioFilePath             { get; set; } = "";
    public string  AudioFileSha256Sum        { get; set; } = "";

    // ── TTS-only settings, snapshotted per job so a requeue renders the same way ──
    /// <summary>TtsBackendKind name ("Chatterbox" / "Kokoro" / "OmniVoice"); "" for ASR jobs.</summary>
    public string  TtsBackend                { get; set; } = "";
    /// <summary>vernacula-phonemizer language code (OmniVoice); "" where the backend has no choice.</summary>
    public string  TtsLanguage               { get; set; } = "";
    /// <summary>Backend-specific voice: a WAV path (Chatterbox), a voice name (Kokoro), a library id (OmniVoice).</summary>
    public string  TtsVoice                  { get; set; } = "";
    public float   TtsSpeed                  { get; set; } = 1.0f;
    public int     TtsNumStep                { get; set; } = 32;
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
                OnPropertyChanged(nameof(RunTimeLabel));
                OnPropertyChanged(nameof(ShowAsrProgress));
                OnPropertyChanged(nameof(ShowTtsProgress));
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
    /// <summary>"Load" opens an ASR job's results; "Open" a TTS job's reader.</summary>
    public string LoadLabel => Loc.Instance[IsTts ? "btn_open" : "btn_load"];
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

    private double? _outputDurationSeconds;
    /// <summary>TTS: length of the rendered audio once the job completes. Null for ASR jobs.</summary>
    public double? OutputDurationSeconds
    {
        get => _outputDurationSeconds;
        set
        {
            if (SetProperty(ref _outputDurationSeconds, value))
                OnPropertyChanged(nameof(RunTimeLabel));
        }
    }

    /// <summary>
    /// The Time column. ASR jobs, and TTS jobs still rendering, show wall-clock run time; a
    /// finished TTS job shows the length of the audio it produced, which is what the user
    /// wants to know about a synthesis — how long it took to render matters less than how
    /// long it plays.
    /// </summary>
    public string RunTimeLabel
    {
        get
        {
            if (Kind == JobKind.Tts && Status == JobStatus.Complete && _outputDurationSeconds is { } dur)
                return FormatSeconds((int)Math.Round(dur, MidpointRounding.AwayFromZero));
            if (_runTimeSeconds is not { } secs) return "";
            return FormatSeconds(secs);
        }
    }

    private static string FormatSeconds(int secs)
    {
        if (secs < 60)   return $"{secs}s";
        if (secs < 3600) return $"{secs / 60}m {secs % 60:D2}s";
        return $"{secs / 3600}h {secs % 3600 / 60}m";
    }

    /// <summary>TTS: the rendered WAV, beside the sidecar in <see cref="ResultsFile"/>.</summary>
    public string OutputAudioPath =>
        Kind == JobKind.Tts ? Path.ChangeExtension(ResultsFile, ".wav") : "";

    /// <summary>TTS: the folder holding one WAV per paragraph, beside the sidecar.</summary>
    public string SegmentsDir =>
        Kind == JobKind.Tts && ResultsFile.Length > 0
            ? Path.Combine(Path.GetDirectoryName(ResultsFile) ?? "", Path.GetFileNameWithoutExtension(ResultsFile) + "_segments")
            : "";

    public bool IsTts => Kind == JobKind.Tts;
    public bool IsAsr => Kind == JobKind.Asr;

    /// <summary>The Kind column's short badge text.</summary>
    public string KindLabel => Kind == JobKind.Tts ? Loc.Instance["kind_tts"] : Loc.Instance["kind_asr"];

    /// <summary>
    /// The Progress column's free-text line. ASR jobs use the phase label + percent bar; TTS
    /// jobs render a phase message ("chunk 3/12") since a chunk count is the honest unit.
    /// </summary>
    private string _progressText = "";
    public string ProgressText
    {
        get => _progressText;
        set => SetProperty(ref _progressText, value);
    }

    /// <summary>ASR rows draw the phase + percent bar; TTS rows draw <see cref="ProgressText"/>.</summary>
    public bool ShowAsrProgress => ShowProgress && Kind == JobKind.Asr;
    public bool ShowTtsProgress => ShowProgress && Kind == JobKind.Tts;

    public void RefreshThemeBindings()
    {
        OnPropertyChanged(nameof(StatusBrush));
    }

    public void RefreshLocalizedText()
    {
        OnPropertyChanged(nameof(StatusLabel));
        OnPropertyChanged(nameof(KindLabel));
        OnPropertyChanged(nameof(ResumeLabel));
        OnPropertyChanged(nameof(MonitorLabel));
        OnPropertyChanged(nameof(PauseLabel));
        OnPropertyChanged(nameof(LoadLabel));
        OnPropertyChanged(nameof(RemoveLabel));
    }
}
