using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using ParakeetCSharp.Models;
using ParakeetCSharp.ViewModels;

namespace ParakeetCSharp.Views;

public partial class HomeView : UserControl
{
    public HomeView()
    {
        InitializeComponent();
        Loaded   += OnLoaded;
        Unloaded += OnUnloaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        var s = App.Current.Settings.Current;
        if (s.HomeColAudioWidth > 0)
            JobsGrid.Columns[1].Width = new DataGridLength(s.HomeColAudioWidth);
        if (s.HomeColTitleWidth > 0)
            JobsGrid.Columns[0].Width = new DataGridLength(s.HomeColTitleWidth);

        // Subscribe to window close so we can save before the visual tree tears down
        var window = Window.GetWindow(this);
        if (window != null)
        {
            window.Closing -= OnWindowClosing;
            window.Closing += OnWindowClosing;
        }
    }

    private void OnWindowClosing(object? sender, System.ComponentModel.CancelEventArgs e)
        => SaveColumnWidths();

    // Fired during navigation (Home → Results). Visual still intact here.
    private void OnUnloaded(object sender, RoutedEventArgs e)
        => SaveColumnWidths();

    private void SaveColumnWidths()
    {
        var s  = App.Current.Settings.Current;
        var w0 = JobsGrid.Columns[0].Width.DisplayValue;
        var w1 = JobsGrid.Columns[1].Width.DisplayValue;
        if (!double.IsNaN(w0) && w0 > 0) s.HomeColTitleWidth = w0;
        if (!double.IsNaN(w1) && w1 > 0) s.HomeColAudioWidth = w1;
        App.Current.Settings.Save();
    }

    private string? _originalTitle;
    private bool _shouldSaveTitle;

    private void OnViewPreviewMouseDown(object sender, object e)
    {
        if (Keyboard.FocusedElement is not TextBox tb) return;
        if (tb.DataContext is not JobRecord) return;
        if (e.Source is AvaloniaObject src && IsDescendantOrSelf(tb, src)) return;

        // Click is outside the focused title TextBox — cancel by moving focus away
        JobsGrid.Focus();
    }

    private static bool IsDescendantOrSelf(AvaloniaObject ancestor, AvaloniaObject element)
    {
        var current = element;
        while (current != null)
        {
            if (current == ancestor) return true;
            current = VisualTreeHelper.GetParent(current);
        }
        return false;
    }

    private void OnTitleTextBoxGotFocus(object sender, RoutedEventArgs e)
    {
        if (sender is not TextBox tb) return;
        if (tb.DataContext is not JobRecord job) return;
        _originalTitle = job.JobTitle;
        _shouldSaveTitle = false;
    }

    private void OnTitleTextBoxKeyDown(object sender, KeyEventArgs e)
    {
        if (sender is not TextBox tb) return;
        if (e.Key == Key.Enter)
        {
            _shouldSaveTitle = true;
            JobsGrid.Focus();
            e.Handled = true;
        }
        else if (e.Key == Key.Escape)
        {
            tb.Text = _originalTitle ?? tb.Text;
            JobsGrid.Focus();
            e.Handled = true;
        }
    }

    private void OnStatusBadgeClicked(object sender, object e)
    {
        if (sender is FrameworkElement { DataContext: JobRecord { HasError: true, ErrorMessage: { } error } })
            Clipboard.SetText(error);
    }

    private void OnTitleTextBoxLostFocus(object sender, RoutedEventArgs e)
    {
        if (sender is not TextBox tb) return;
        if (tb.DataContext is not JobRecord job) return;

        if (!_shouldSaveTitle)
        {
            tb.Text = _originalTitle ?? job.JobTitle; // revert
            return;
        }

        _shouldSaveTitle = false;

        var newTitle = tb.Text.Trim();
        if (string.IsNullOrEmpty(newTitle))
        {
            tb.Text = _originalTitle ?? job.JobTitle; // revert
            return;
        }

        if (newTitle == job.JobTitle) return; // nothing changed

        job.JobTitle = newTitle;
        if (DataContext is HomeViewModel vm)
            vm.RenameJobCommand.Execute(job);
    }
}
