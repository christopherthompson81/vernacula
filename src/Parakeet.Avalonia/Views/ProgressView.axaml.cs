using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;
using System.ComponentModel;

namespace ParakeetCSharp.Views;

public partial class ProgressView : UserControl
{
    public ProgressView()
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
        ProgressHeadingText.Text = Loc.Instance["progress_heading"];
        SegmentsGrid.Columns[0].Header = Loc.Instance["segments_col_speaker"];
        SegmentsGrid.Columns[1].Header = Loc.Instance["segments_col_start"];
        SegmentsGrid.Columns[2].Header = Loc.Instance["segments_col_end"];
        SegmentsGrid.Columns[3].Header = Loc.Instance["segments_col_content"];
        PauseButton.Content = Loc.Instance["btn_pause"];
        BackHomeButton.Content = Loc.Instance["btn_back_home"];
    }

    private void ProgressView_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Escape)
            return;

        ClearSegmentsGridSelection();
        e.Handled = true;
    }

    private void ProgressView_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (e.Source is not Control source || source == SegmentsGrid || source.GetVisualAncestors().Contains(SegmentsGrid))
            return;

        ClearSegmentsGridSelection();
    }

    private void ClearSegmentsGridSelection()
    {
        SegmentsGrid.SelectedItem = null;
        SegmentsGrid.SelectedIndex = -1;
        Focus();
    }
}
