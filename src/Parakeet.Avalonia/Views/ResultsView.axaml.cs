using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.VisualTree;
using ParakeetCSharp.Extensions;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views;

public partial class ResultsView : UserControl
{
    public ResultsView()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loaded   += OnLoaded;
        Unloaded += OnUnloaded;
    }

    private void OnLoaded(object? sender, RoutedEventArgs e)
    {
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
        ApplyLocalizedText();
        var s = App.Current.Settings.Current;
        if (s.ResultsColSpeakerWidth > 0)
            SegmentsGrid.Columns[0].Width = new DataGridLength(s.ResultsColSpeakerWidth);
        if (s.ResultsColStartWidth > 0)
            SegmentsGrid.Columns[1].Width = new DataGridLength(s.ResultsColStartWidth);
        if (s.ResultsColEndWidth > 0)
            SegmentsGrid.Columns[2].Width = new DataGridLength(s.ResultsColEndWidth);

        // TODO: Port GetVisualParent to Avalonia
        // var window = this.GetVisualParent<Window>();
        // if (window != null)
        // {
        //     window.Closing -= OnWindowClosing;
        //     window.Closing += OnWindowClosing;
        // }
    }

    private void OnWindowClosing(object? sender, System.ComponentModel.CancelEventArgs e)
        => SaveColumnWidths();

    private void OnUnloaded(object? sender, RoutedEventArgs e)
    {
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        SaveColumnWidths();
    }

    private void SaveColumnWidths()
    {
        var s  = App.Current.Settings.Current;
        var w0 = SegmentsGrid.Columns[0].Width.DisplayValue;
        var w1 = SegmentsGrid.Columns[1].Width.DisplayValue;
        var w2 = SegmentsGrid.Columns[2].Width.DisplayValue;
        if (!double.IsNaN(w0) && w0 > 0) s.ResultsColSpeakerWidth = w0;
        if (!double.IsNaN(w1) && w1 > 0) s.ResultsColStartWidth   = w1;
        if (!double.IsNaN(w2) && w2 > 0) s.ResultsColEndWidth     = w2;
        App.Current.Settings.Save();
    }

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;

        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        SegmentsLabelRun.Text = Loc.Instance["results_segments_label"];
        SegmentsGrid.Columns[0].Header = Loc.Instance["segments_col_speaker"];
        SegmentsGrid.Columns[1].Header = Loc.Instance["segments_col_start"];
        SegmentsGrid.Columns[2].Header = Loc.Instance["segments_col_end"];
        SegmentsGrid.Columns[3].Header = Loc.Instance["segments_col_content"];
        EditTranscriptButton.Content = Loc.Instance["btn_edit_transcript"];
        EditSpeakersButton.Content = Loc.Instance["btn_edit_speakers"];
        ExportButton.Content = Loc.Instance["btn_export"];
        BackHistoryButton.Content = Loc.Instance["btn_back_history"];
    }

    private void ResultsView_KeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Escape)
            return;

        ClearSegmentsGridSelection();
        e.Handled = true;
    }

    private void ResultsView_PointerPressed(object? sender, PointerPressedEventArgs e)
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
