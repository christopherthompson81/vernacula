using System.Collections.ObjectModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.App.Models;
using Vernacula.App.Services;

namespace Vernacula.App.ViewModels;

internal partial class ResultsViewModel : ObservableObject
{
    private readonly ExportService _export;

    [ObservableProperty]
    private string _audioBaseName = "";

    // ── LID banner state ──────────────────────────────────────────────────

    /// <summary>
    /// True when LID ran on this job and produced a result (ambiguous or
    /// not). Drives the always-visible info chip under the header.
    /// </summary>
    [ObservableProperty] private bool _showDetectedLanguageInfo;
    [ObservableProperty] private string _detectedLanguageInfoText = "";

    /// <summary>
    /// True when the job's metadata indicates LID detected a language the
    /// ASR backend couldn't handle. Drives the amber warning banner.
    /// </summary>
    [ObservableProperty] private bool _showMismatchBanner;
    [ObservableProperty] private string _mismatchBannerText = "";
    [ObservableProperty] private string _reprocessButtonText = "Reprocess";
    [ObservableProperty] private bool _canReprocess;

    private int?       _jobId;
    private string?    _mismatchDetectedIso;
    private AsrBackend? _mismatchSuggestedBackend;

    /// <summary>
    /// Injected by MainViewModel — the actions Reprocess needs to actually
    /// do its work (update ControlDb + requeue). Null in contexts that
    /// don't have a job-queue (rare / test).
    /// </summary>
    public Action<int, AsrBackend, string>? ReprocessJob { get; set; }

    private string? _dbPath;
    private Window? _ownerWindow;
    private Views.TranscriptEditorWindow? _editorWindow;

    public ObservableCollection<SegmentRow> Segments { get; } = new();

    public Action? NavigateBack { get; set; }

    public ResultsViewModel(ExportService export) => _export = export;

    public void Load(string dbPath, string audioBaseName, Window? owner = null, int? jobId = null)
    {
        _dbPath       = dbPath;
        AudioBaseName = audioBaseName;
        _ownerWindow = owner;
        _jobId        = jobId;
        Segments.Clear();
        using var db  = new TranscriptionDb(dbPath);
        PopulateSegments(db);
        LoadLidBanners(db);
    }

    private void LoadLidBanners(TranscriptionDb db)
    {
        ShowDetectedLanguageInfo = false;
        DetectedLanguageInfoText = "";
        ShowMismatchBanner       = false;
        CanReprocess             = false;
        _mismatchDetectedIso     = null;
        _mismatchSuggestedBackend = null;

        string? detectedIso  = db.GetMetadata("detected_language");
        string? detectedName = db.GetMetadata("detected_language_name") ?? detectedIso;
        string? probStr      = db.GetMetadata("detected_language_probability");
        string? ambStr       = db.GetMetadata("detected_language_ambiguous");
        string? effectiveIso = db.GetMetadata("asr_language_code");
        bool    mismatch     = db.GetMetadata("detected_language_backend_mismatch") == "1";
        string? suggestedStr = db.GetMetadata("detected_language_suggested_backend");
        string? asrModel     = db.GetMetadata("asr_model");

        if (string.IsNullOrWhiteSpace(detectedIso)) return;

        // Info chip: shown whenever LID ran, regardless of outcome.
        bool ambiguous = ambStr == "1";
        string probSuffix = "";
        if (float.TryParse(probStr, System.Globalization.NumberStyles.Float,
                           System.Globalization.CultureInfo.InvariantCulture, out float p0))
            probSuffix = $", {p0:P0} confidence";

        string effectiveSuffix = "";
        if (!string.IsNullOrWhiteSpace(effectiveIso)
            && !string.Equals(effectiveIso, "auto", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(effectiveIso, detectedIso, StringComparison.OrdinalIgnoreCase))
        {
            // Rare: ASR ran with a different language than LID detected
            // (e.g. user kept the current backend on the mismatch popup).
            effectiveSuffix = $"; ASR ran as {effectiveIso}";
        }

        DetectedLanguageInfoText =
            (ambiguous ? "Language detection (ambiguous): " : "Language detected: ")
            + $"{detectedName} ({detectedIso}{probSuffix}){effectiveSuffix}";
        ShowDetectedLanguageInfo = true;

        if (!mismatch) return;

        string currentBackendLabel = asrModel switch
        {
            "CohereLabs/cohere-transcribe-03-2026" => "Cohere Transcribe",
            "Qwen/Qwen3-ASR-1.7B"                  => "Qwen3-ASR",
            "vibevoice/vibevoice-asr"              => "VibeVoice-ASR",
            "nvidia/parakeet-tdt-0.6b-v3"          => "Parakeet",
            _                                       => asrModel ?? "the current ASR backend",
        };

        string confidenceSuffix = "";
        if (float.TryParse(probStr, System.Globalization.NumberStyles.Float,
                           System.Globalization.CultureInfo.InvariantCulture, out float p))
        {
            confidenceSuffix = $" (confidence {p:P0})";
        }

        AsrBackend? suggested = null;
        if (!string.IsNullOrWhiteSpace(suggestedStr) &&
            Enum.TryParse<AsrBackend>(suggestedStr, out var parsed))
        {
            suggested = parsed;
        }

        _mismatchDetectedIso     = detectedIso;
        _mismatchSuggestedBackend = suggested;

        if (suggested is not null && ReprocessJob is not null && _jobId is not null)
        {
            ReprocessButtonText = $"Reprocess with {HumanBackend(suggested.Value)}";
            CanReprocess        = true;
        }
        else
        {
            ReprocessButtonText = "Reprocess";
            CanReprocess        = false;
        }

        MismatchBannerText =
            $"Language identification detected {detectedName} ({detectedIso}){confidenceSuffix}, " +
            $"but this job was transcribed with {currentBackendLabel}, which doesn't support {detectedName}. " +
            (suggested is not null
                ? $"{HumanBackend(suggested.Value)} supports {detectedName}; use the button to re-queue this job with {HumanBackend(suggested.Value)} and force-language {detectedIso}."
                : "No installed ASR backend supports this language; you'll need to install a different backend before reprocessing.");

        ShowMismatchBanner = true;
    }

    private static string HumanBackend(AsrBackend b) => b switch
    {
        AsrBackend.Parakeet  => "Parakeet",
        AsrBackend.Cohere    => "Cohere Transcribe",
        AsrBackend.Qwen3Asr  => "Qwen3-ASR",
        AsrBackend.VibeVoice => "VibeVoice-ASR",
        _                    => b.ToString(),
    };

    [RelayCommand(CanExecute = nameof(CanReprocess))]
    private void ReprocessWithSuggestedBackend()
    {
        if (_jobId is null || _mismatchSuggestedBackend is null
            || _mismatchDetectedIso is null || ReprocessJob is null)
            return;
        ReprocessJob(_jobId.Value, _mismatchSuggestedBackend.Value, _mismatchDetectedIso);
        // Banner goes away once the job restarts; the caller (MainViewModel) will
        // navigate back to Home / Progress as appropriate.
        ShowMismatchBanner = false;
    }

    private void PopulateSegments(TranscriptionDb db)
    {
        var transcript = db.GetTranscript();
        var segs       = db.GetSegments();
        for (int i = 0; i < segs.Count; i++)
        {
            var seg = segs[i];
            Segments.Add(new SegmentRow
            {
                SegmentId          = i,
                SpeakerTag         = seg.spkId,
                SpeakerDisplayName = i < transcript.Count
                    ? transcript[i].GetValueOrDefault("speaker_name", seg.spkId)
                    : seg.spkId,
                StartTime          = seg.start,
                EndTime            = seg.end,
                Text               = i < transcript.Count
                    ? transcript[i].GetValueOrDefault("content", "")
                    : "",
            });
        }
    }

    [RelayCommand]
    private void EditTranscript()
    {
        if (_dbPath is null || _ownerWindow is null) return;
        if (_editorWindow is { IsVisible: true })
        {
            _editorWindow.Activate();
            if (_editorWindow.WindowState == WindowState.Minimized)
                _editorWindow.WindowState = WindowState.Normal;
            return;
        }

        _editorWindow = new Views.TranscriptEditorWindow(_dbPath, AudioBaseName);
        _editorWindow.DataChanged += ReloadSegments;
        _editorWindow.Closed += (_, _) => _editorWindow = null;
        _editorWindow.Show(_ownerWindow);
    }

    private void ReloadSegments()
    {
        if (_dbPath is null) return;
        Segments.Clear();
        using var db = new TranscriptionDb(_dbPath);
        PopulateSegments(db);
    }

    [RelayCommand(CanExecute = nameof(CanEditSpeakerNames))]
    private async Task EditSpeakerNamesAsync()
    {
        if (_dbPath is null) return;
        var dlg = new Views.Dialogs.SpeakerNamesDialog(_dbPath);
        await dlg.ShowDialog(_ownerWindow!);
        if (!dlg.DialogResult) return;
        // Reload segments to reflect updated speaker names
        Segments.Clear();
        using var db = new TranscriptionDb(_dbPath);
        PopulateSegments(db);
    }

    private bool CanEditSpeakerNames() => _dbPath is not null;

    [RelayCommand(CanExecute = nameof(CanExport))]
    private async Task ExportAsync()
    {
        if (_dbPath is null || _ownerWindow is null) return;

        var fmtDlg = new Views.Dialogs.ExportDialog();
        await fmtDlg.ShowDialog(_ownerWindow);
        if (!fmtDlg.DialogResult) return;

        var fmt = fmtDlg.SelectedFormat;

        (string ext, string typeLabel) = fmt switch
        {
            Views.Dialogs.ExportFormat.Xlsx => (".xlsx", "Excel Workbook"),
            Views.Dialogs.ExportFormat.Csv  => (".csv",  "CSV File"),
            Views.Dialogs.ExportFormat.Json => (".json", "JSON File"),
            Views.Dialogs.ExportFormat.Srt  => (".srt",  "SubRip Subtitle"),
            Views.Dialogs.ExportFormat.Md   => (".md",   "Markdown File"),
            Views.Dialogs.ExportFormat.Docx => (".docx", "Word Document"),
            Views.Dialogs.ExportFormat.Db   => (".db",   "SQLite Database"),
            _                               => (".xlsx", "Excel Workbook"),
        };

        bool isDb      = fmt == Views.Dialogs.ExportFormat.Db;
        string sugName = isDb
            ? Path.GetFileNameWithoutExtension(_dbPath)
            : $"{Path.GetFileNameWithoutExtension(AudioBaseName)}_transcript";

        var file = await _ownerWindow.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            SuggestedFileName = sugName,
            DefaultExtension  = ext.TrimStart('.'),
            FileTypeChoices   = [new FilePickerFileType(typeLabel) { Patterns = [$"*{ext}"] }],
        });
        if (file is null) return;

        var savePath = file.Path.LocalPath;

        if (isDb)
        {
            _export.ExportDbCopy(_dbPath, savePath);
            return;
        }

        using var db = new TranscriptionDb(_dbPath);
        switch (fmt)
        {
            case Views.Dialogs.ExportFormat.Xlsx: _export.ExportXlsx(db, savePath); break;
            case Views.Dialogs.ExportFormat.Csv:  _export.ExportCsv(db, savePath);  break;
            case Views.Dialogs.ExportFormat.Json: _export.ExportJson(db, savePath); break;
            case Views.Dialogs.ExportFormat.Srt:  _export.ExportSrt(db, savePath);  break;
            case Views.Dialogs.ExportFormat.Md:   _export.ExportMd(db, savePath);   break;
            case Views.Dialogs.ExportFormat.Docx: _export.ExportDocx(db, savePath); break;
        }
    }

    private bool CanExport() => _dbPath is not null;

    [RelayCommand]
    private void Back() => NavigateBack?.Invoke();
}
