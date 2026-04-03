using System.Collections.Specialized;
using System.Runtime.CompilerServices;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Threading;

namespace ParakeetCSharp.Behaviors;

public static class DataGridBehaviors
{
    private sealed class AutoScrollSubscription
    {
        public INotifyCollectionChanged? Collection;
        public NotifyCollectionChangedEventHandler? Handler;
    }

    private static readonly ConditionalWeakTable<DataGrid, AutoScrollSubscription> Subscriptions = new();

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
            grid.AddHandler(Control.UnloadedEvent, OnGridUnloaded);
            AttachToItemsSource(grid);
        }
        else
        {
            grid.RemoveHandler(Control.LoadedEvent, OnGridLoaded);
            grid.RemoveHandler(Control.UnloadedEvent, OnGridUnloaded);
            DetachFromItemsSource(grid);
        }
    }

    private static void OnGridLoaded(object? sender, System.EventArgs e)
    {
        if (sender is not DataGrid grid) return;
        AttachToItemsSource(grid);
        ScrollToLastItem(grid);
    }

    private static void OnGridUnloaded(object? sender, System.EventArgs e)
    {
        if (sender is DataGrid grid)
        {
            DetachFromItemsSource(grid);
        }
    }

    private static void AttachToItemsSource(DataGrid grid)
    {
        DetachFromItemsSource(grid);

        if (grid.ItemsSource is not INotifyCollectionChanged collection)
        {
            return;
        }

        NotifyCollectionChangedEventHandler handler = (_, e) =>
        {
            if (e.Action is NotifyCollectionChangedAction.Add
                or NotifyCollectionChangedAction.Move
                or NotifyCollectionChangedAction.Replace
                or NotifyCollectionChangedAction.Reset)
            {
                ScrollToLastItem(grid);
            }
        };

        collection.CollectionChanged += handler;
        var subscription = Subscriptions.GetOrCreateValue(grid);
        subscription.Collection = collection;
        subscription.Handler = handler;
    }

    private static void DetachFromItemsSource(DataGrid grid)
    {
        if (!Subscriptions.TryGetValue(grid, out var subscription))
        {
            return;
        }

        if (subscription.Collection is not null && subscription.Handler is not null)
        {
            subscription.Collection.CollectionChanged -= subscription.Handler;
        }

        subscription.Collection = null;
        subscription.Handler = null;
    }

    private static void ScrollToLastItem(DataGrid grid)
    {
        if (grid.ItemsSource is null)
        {
            return;
        }

        Dispatcher.UIThread.Post(() =>
        {
            if (!GetAutoScroll(grid) || grid.ItemsSource is null)
            {
                return;
            }

            object? lastItem = null;
            foreach (var item in grid.ItemsSource)
            {
                lastItem = item;
            }

            if (lastItem is not null)
            {
                grid.ScrollIntoView(lastItem, null);
            }
        }, DispatcherPriority.Background);
    }
}
