using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using ParakeetCSharp.Models;
using ParakeetCSharp.ViewModels;
using ParakeetCSharp.Views;

namespace ParakeetCSharp.Views;

public partial class MainWindow : Window
{
    private PixelPoint? _lastNormalPosition;
    private Size? _lastNormalSize;

    public MainWindow()
    {
        InitializeComponent();
        Console.WriteLine("[MainWindow] Constructor called");
        RestoreInitialWindowSettings();
        Loaded += MainWindow_Loaded;
        Opened += MainWindow_Opened;
        Closing += Window_Closing;
        PositionChanged += (_, _) => CaptureNormalBounds();
        Resized += (_, _) => CaptureNormalBounds();
        Closed += (_, _) => Console.WriteLine("[MainWindow] Closed event fired!");
    }

    private void MainWindow_Loaded(object? sender, RoutedEventArgs e)
    {
        Console.WriteLine("[MainWindow] MainWindow_Loaded");
        // Set the main window reference in the view model
        if (DataContext is MainViewModel vm)
        {
            vm.SetMainWindow(this);
            // Start initialization now that the window is loaded and bindings are active
            _ = vm.StartAsync();
        }

        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        CaptureNormalBounds();
    }

    private void MainWindow_Opened(object? sender, EventArgs e)
    {
        var s = App.Current.Settings.Current;
        if (s.WindowLeft.HasValue && s.WindowTop.HasValue)
        {
            Position = WindowHelper.ClampToVisibleArea(
                this,
                new PixelPoint((int)Math.Round(s.WindowLeft.Value), (int)Math.Round(s.WindowTop.Value)),
                Width,
                Height);
        }

        if (s.WindowMaximized)
        {
            WindowState = WindowState.Maximized;
        }

        CaptureNormalBounds();
    }

    private void RestoreInitialWindowSettings()
    {
        var s = App.Current.Settings.Current;
        Width = Math.Max(MinWidth, s.WindowWidth);
        Height = Math.Max(MinHeight, s.WindowHeight);

        if (s.WindowLeft.HasValue && s.WindowTop.HasValue)
        {
            Position = new PixelPoint(
                (int)Math.Round(s.WindowLeft.Value),
                (int)Math.Round(s.WindowTop.Value));
        }
    }

    private void CaptureNormalBounds()
    {
        if (WindowState != WindowState.Normal)
        {
            return;
        }

        _lastNormalPosition = Position;
        _lastNormalSize = new Size(Width, Height);
    }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        bool isAnyJobRunning = DataContext is MainViewModel vm && vm.IsAnyJobRunning;
        Console.WriteLine($"[MainWindow] Closing event fired! IsAnyJobRunning={isAnyJobRunning}");
        // If any job is running or queued, prompt the user before closing
        if (isAnyJobRunning)
        {
            // Simple confirmation - for now just cancel if jobs are running
            // TODO: Replace with proper Avalonia dialog
            e.Cancel = true;
            return;
        }

        var s = App.Current.Settings.Current;

        var normalPosition = _lastNormalPosition ?? Position;
        var normalSize = _lastNormalSize ?? new Size(Width, Height);

        s.WindowLeft      = normalPosition.X;
        s.WindowTop       = normalPosition.Y;
        s.WindowWidth     = normalSize.Width;
        s.WindowHeight    = normalSize.Height;
        s.WindowMaximized = WindowState == WindowState.Maximized;
        App.Current.Settings.Save();
    }

    private SettingsWindow? _settingsWindow;

    private void Settings_Click(object? sender, RoutedEventArgs e) => OpenSettingsWindow();

    internal void OpenSettingsWindow()
    {
        if (_settingsWindow is { IsVisible: true })
        {
            _settingsWindow.Activate();
            if (_settingsWindow.WindowState == WindowState.Minimized)
                _settingsWindow.WindowState = WindowState.Normal;
            return;
        }

        if (DataContext is not MainViewModel vm) return;
        _settingsWindow = new SettingsWindow
        {
            DataContext = vm.Settings,
        };
        _settingsWindow.Show(this);
    }

    private void Exit_Click(object? sender, RoutedEventArgs e) => Close();

    private HelpWindow? _helpWindow;

    private void HelpContents_Click(object? sender, RoutedEventArgs e) => OpenHelp(null);

    internal void OpenHelp(string? topicId)
    {
        if (_helpWindow is { IsVisible: true })
        {
            if (topicId is not null)
                _helpWindow.DisplayTopic(topicId);
            _helpWindow.Activate();
            if (_helpWindow.WindowState == WindowState.Minimized)
                _helpWindow.WindowState = WindowState.Normal;
            return;
        }
        _helpWindow = new HelpWindow(topicId);
        _helpWindow.Show(this);
    }

    private void About_Click(object? sender, RoutedEventArgs e) =>
        new Dialogs.AboutDialog().ShowDialog(this);
}
