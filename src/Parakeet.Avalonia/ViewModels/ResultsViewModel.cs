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
    private Views.TranscriptEditorWindow? _editorWindow;
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
        if (_dbPath is null) return;
        if (_editorWindow is { IsLoaded: true })
        {
            _editorWindow.Activate();
            if (_editorWindow.WindowState == WindowState.Minimized)
                _editorWindow.WindowState = WindowState.Normal;
            return;
        }
        _editorWindow = new Views.TranscriptEditorWindow(_dbPath, AudioBaseName);
        _editorWindow.Show(_ownerWindow);
        _editorWindow.DataChanged += ReloadSegments;
    }

    private void ReloadSegments()
    {
        if (_dbPath is null) return;
        Segments.Clear();
        using var db = new TranscriptionDb(_dbPath);
        PopulateSegments(db);
    }

    [RelayCommand]
    private void EditSpeakerNames()
    {
        if (_dbPath is null) return;
        var dlg = new Views.Dialogs.SpeakerNamesDialog(_dbPath);
        if (dlg.ShowDialog(_ownerWindow) != true) return;
        // Reload segments to reflect updated speaker names
        Segments.Clear();
        using var db = new TranscriptionDb(_dbPath);
        PopulateSegments(db);
    }

    [RelayCommand]
    private async Task ExportAsync()
    {
        if (_dbPath is null) return;
        var fmtDlg = new Views.Dialogs.ExportDialog();
        if (fmtDlg.ShowDialog(_ownerWindow) != true) return;

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

        string filter = fmt switch
        {
            Views.Dialogs.ExportFormat.Xlsx => Loc.Instance["dlg_filter_xlsx"],
            Views.Dialogs.ExportFormat.Csv  => Loc.Instance["dlg_filter_csv"],
            Views.Dialogs.ExportFormat.Json => Loc.Instance["dlg_filter_json"],
            Views.Dialogs.ExportFormat.Srt  => Loc.Instance["dlg_filter_srt"],
            Views.Dialogs.ExportFormat.Md   => Loc.Instance["dlg_filter_md"],
            Views.Dialogs.ExportFormat.Docx => Loc.Instance["dlg_filter_docx"],
            Views.Dialogs.ExportFormat.Db   => Loc.Instance["dlg_filter_db"],
            _                               => Loc.Instance["dlg_filter_xlsx"],
        };

        bool isDb = fmt == Views.Dialogs.ExportFormat.Db;
        string baseName = isDb
            ? Path.GetFileNameWithoutExtension(_dbPath)
            : $"{Path.GetFileNameWithoutExtension(AudioBaseName)}_transcript";

        // Use Avalonia file picker for save dialog
        var topLevel = _ownerWindow ?? TopLevel.GetTopLevel((Visual?)this);
        if (topLevel == null) return;

        var suggestedName = baseName + defaultExt;
        var file = await topLevel.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save transcript",
            SuggestedFileName = suggestedName,
            FileTypeChoices = new[]
            {
                new FilePickerFileType { Name = filter, Extensions = [defaultExt.TrimStart('.')] }
            }
        });

        if (file == null) return;

        var filePath = file.Path.LocalPath;

        if (isDb)
        {
            _export.ExportDbCopy(_dbPath, filePath);
            return;
        }

        using var db = new TranscriptionDb(_dbPath);
        switch (fmt)
        {
            case Views.Dialogs.ExportFormat.Xlsx: _export.ExportXlsx(db, filePath); break;
            case Views.Dialogs.ExportFormat.Csv:  _export.ExportCsv(db, filePath);  break;
            case Views.Dialogs.ExportFormat.Json: _export.ExportJson(db, filePath); break;
            case Views.Dialogs.ExportFormat.Srt:  _export.ExportSrt(db, filePath);  break;
            case Views.Dialogs.ExportFormat.Md:   _export.ExportMd(db, filePath);   break;
            case Views.Dialogs.ExportFormat.Docx: _export.ExportDocx(db, filePath); break;
        }
    }

    [RelayCommand]
    private void Back() => NavigateBack?.Invoke();
}
