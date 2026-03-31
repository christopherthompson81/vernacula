using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Extensions;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views;

public partial class ResultsView : UserControl
{
    public ResultsView()
    {
        InitializeComponent();
        Loaded   += OnLoaded;
        Unloaded += OnUnloaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
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

    private void OnUnloaded(object sender, RoutedEventArgs e)
        => SaveColumnWidths();

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
}
