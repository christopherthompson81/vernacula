using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;

namespace ParakeetCSharp.Behaviors;

public static class DataGridBehaviors
{
    public static readonly AttachedProperty<bool> AutoScrollProperty =
        AvaloniaProperty.RegisterAttached<DataGrid, bool>(
            "AutoScroll",
            typeof(DataGrid),
            defaultValue: false);

    public static void SetAutoScroll(DataGrid grid, bool value) =>
        grid.SetValue(AutoScrollProperty, value);

    public static bool GetAutoScroll(DataGrid grid) =>
        grid.GetValue<bool>(AutoScrollProperty);

    static DataGridBehaviors()
    {
        AutoScrollProperty.Changed.AddClassHandler<DataGrid>(OnAutoScrollChanged);
    }

    private static void OnAutoScrollChanged(DataGrid grid, AvaloniaPropertyChangedEventArgs e)
    {
        if (e.NewValue is not bool enabled) return;

        if (enabled)
        {
            grid.AddHandler(Control.LoadedEvent, OnGridLoaded);
        }
        else
        {
            grid.RemoveHandler(Control.LoadedEvent, OnGridLoaded);
        }
    }

    private static void OnGridLoaded(object? sender, System.EventArgs e)
    {
        if (sender is not DataGrid grid) return;

        // Subscribe to collection changes for auto-scroll
        if (grid.ItemsSource is INotifyCollectionChanged ncc)
        {
            ncc.CollectionChanged += (_, _) =>
            {
                // Auto-scroll to last item when collection changes
                // Note: Avalonia DataGrid may need different approach
                // This is a placeholder - actual implementation may vary
            };
        }
    }
}
