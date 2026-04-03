using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;
using System.ComponentModel;

namespace ParakeetCSharp.Views;

public partial class HomeView : UserControl
{
    public HomeView()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loaded += (_, _) => Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        Unloaded += (_, _) => Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;

        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        OpenSettingsButton.Content = Loc.Instance["btn_open_settings"];
        UpdateAvailableText.Text = Loc.Instance["home_update_available"];
        BannerOpenSettingsButton.Content = Loc.Instance["btn_open_settings"];
        DismissBannerButton.Content = Loc.Instance["btn_dismiss"];
        NewTranscriptionButton.Content = Loc.Instance["btn_new_transcription"];
        BulkAddButton.Content = Loc.Instance["btn_bulk_add"];
        HistoryHeadingText.Text = Loc.Instance["history_heading"];
        HistoryEmptyText.Text = Loc.Instance["history_empty"];
        JobsGrid.Columns[0].Header = Loc.Instance["history_col_title"];
        JobsGrid.Columns[1].Header = Loc.Instance["history_col_audio"];
        JobsGrid.Columns[2].Header = Loc.Instance["history_col_date"];
        JobsGrid.Columns[3].Header = Loc.Instance["history_col_time"];
        JobsGrid.Columns[4].Header = Loc.Instance["history_col_status"];
        JobsGrid.Columns[5].Header = Loc.Instance["history_col_progress"];
        JobsGrid.Columns[6].Header = Loc.Instance["history_col_actions"];
    }

    private void HomeView_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Escape)
            return;

        ClearJobsGridSelection();
        e.Handled = true;
    }

    private void HomeView_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (e.Source is not Control source || source == JobsGrid || source.GetVisualAncestors().Contains(JobsGrid))
            return;

        ClearJobsGridSelection();
    }

    private void ClearJobsGridSelection()
    {
        JobsGrid.SelectedItem = null;
        JobsGrid.SelectedIndex = -1;
        Focus();
    }
}
