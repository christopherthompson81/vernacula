using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Vernacula.App.Models;
using System.ComponentModel;

namespace Vernacula.App.Views.Dialogs;

public partial class SpeakerNamesDialog : Window
{
    private string? _dbPath;
    private readonly List<SpeakerEntry> _entries = new();
    private bool _isEditing;

    public bool DialogResult { get; private set; }

    public SpeakerNamesDialog()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
    }

    public SpeakerNamesDialog(string dbPath) : this()
    {
        _dbPath = dbPath;
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);

        using var db = new TranscriptionDb(dbPath);
        foreach (var (speakerId, name) in db.GetSpeakers())
        {
            _entries.Add(new SpeakerEntry
            {
                SpeakerTag = $"speaker_{speakerId - 1}",
                Name       = name,
            });
        }

        SpeakersGrid.ItemsSource = _entries;
    }

    private void Save_Click(object sender, RoutedEventArgs e)
    {
        if (_dbPath is null) return;

        using var db = new TranscriptionDb(_dbPath);
        foreach (var entry in _entries)
        {
            // speaker_id = index in 1-based: parse from SpeakerTag
            int speakerId = int.Parse(entry.SpeakerTag.Replace("speaker_", "")) + 1;
            db.UpdateSpeaker(speakerId, entry.Name);
        }
        DialogResult = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;

        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        Title = Loc.Instance["modal_speakers_heading"];
        SpeakerNamesHeadingText.Text = Loc.Instance["modal_speakers_heading"];
        if (SpeakersGrid.Columns.Count > 0)
        {
            SpeakersGrid.Columns[0].Header = Loc.Instance["modal_speakers_col_id"];
            if (SpeakersGrid.Columns.Count > 1)
                SpeakersGrid.Columns[1].Header = Loc.Instance["modal_speakers_col_name"];
        }
        SaveButton.Content = Loc.Instance["btn_save"];
        CancelButton.Content = Loc.Instance["btn_cancel"];
    }

    private void SpeakersGrid_BeginningEdit(object? sender, DataGridBeginningEditEventArgs e)
        => _isEditing = true;

    private void SpeakersGrid_CellEditEnding(object? sender, DataGridCellEditEndingEventArgs e)
        => _isEditing = false;

    private void SpeakersGrid_PreviewKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Tab || !_isEditing)
            return;

        int currentRow = _entries.IndexOf((SpeakerEntry?)SpeakersGrid.SelectedItem!);
        int nextRow = currentRow + 1;

        if (nextRow < 0 || nextRow >= _entries.Count)
            return;

        e.Handled = true;
        SpeakersGrid.CommitEdit(DataGridEditingUnit.Row, true);

        var nextItem = _entries[nextRow];
        SpeakersGrid.SelectedItem = nextItem;
        if (SpeakersGrid.Columns.Count > 1)
            SpeakersGrid.CurrentColumn = SpeakersGrid.Columns[1];
        SpeakersGrid.ScrollIntoView(nextItem, SpeakersGrid.CurrentColumn);
        SpeakersGrid.BeginEdit();
    }

    protected override void OnClosed(EventArgs e)
    {
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        base.OnClosed(e);
    }
}
