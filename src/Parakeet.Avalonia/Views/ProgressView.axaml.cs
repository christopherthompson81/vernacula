using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;

namespace ParakeetCSharp.Views;

public partial class ProgressView : UserControl
{
    public ProgressView() => InitializeComponent();

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
