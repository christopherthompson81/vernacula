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
    private Views.TranscriptEditorWindow? _editorWindow;

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
