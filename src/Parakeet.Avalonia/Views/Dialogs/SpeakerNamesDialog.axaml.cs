using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public partial class SpeakerNamesDialog : Window
{
    private readonly string _dbPath;
    private readonly List<SpeakerEntry> _entries = new();
    private bool _isEditing;

    public bool DialogResult { get; private set; }

    public SpeakerNamesDialog(string dbPath)
    {
        _dbPath = dbPath;
        InitializeComponent();
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

    private void SpeakersGrid_BeginningEdit(object sender, System.EventArgs e)
        => _isEditing = true;

    private void SpeakersGrid_CellEditEnding(object sender, System.EventArgs e)
        => _isEditing = false;

    private void SpeakersGrid_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key != Key.Tab || !_isEditing)
            return;

        // Simplified tab handling - Avalonia DataGrid handles this differently
        e.Handled = true;
    }
}
