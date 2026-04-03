using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;

namespace ParakeetCSharp.Views;

public partial class HomeView : UserControl
{
    public HomeView()
    {
        InitializeComponent();
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
