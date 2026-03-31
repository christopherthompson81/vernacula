using System.Collections.ObjectModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ParakeetCSharp.Models;
using ParakeetCSharp.Services;

namespace ParakeetCSharp.ViewModels;

internal partial class ResultsViewModel : ObservableObject
{
    private readonly ExportService _export;

    [ObservableProperty]
    private string _audioBaseName = "";

    private string? _dbPath;
    private Window? _ownerWindow;

    public ObservableCollection<SegmentRow> Segments { get; } = new();

    public Action? NavigateBack { get; set; }

    public ResultsViewModel(ExportService export) => _export = export;

    public void Load(string dbPath, string audioBaseName, Window? owner = null)
    {
        _dbPath       = dbPath;
        AudioBaseName = audioBaseName;
        _ownerWindow = owner;
        Segments.Clear();
        using var db  = new TranscriptionDb(dbPath);
        PopulateSegments(db);
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
        // TODO: Enable when TranscriptEditorWindow is ported
        System.Diagnostics.Debug.WriteLine("Transcript editing not yet available");
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
        if (_dbPath is null) return;
        var fmtDlg = new Views.Dialogs.ExportDialog();
        await fmtDlg.ShowDialog(_ownerWindow!);
        if (!fmtDlg.DialogResult) return;

        var fmt = fmtDlg.SelectedFormat;

        string defaultExt = fmt switch
        {
            Views.Dialogs.ExportFormat.Xlsx => ".xlsx",
            Views.Dialogs.ExportFormat.Csv  => ".csv",
            Views.Dialogs.ExportFormat.Json => ".json",
            Views.Dialogs.ExportFormat.Srt  => ".srt",
            Views.Dialogs.ExportFormat.Md   => ".md",
            Views.Dialogs.ExportFormat.Docx => ".docx",
            Views.Dialogs.ExportFormat.Db   => ".db",
            _                               => ".xlsx",
        };

        // TODO: Implement file save dialog using _ownerWindow
        // For now, just log a message
        System.Diagnostics.Debug.WriteLine($"Export requested: {fmt}");
    }

    private bool CanExport() => _dbPath is not null;

    [RelayCommand]
    private void Back() => NavigateBack?.Invoke();
}
