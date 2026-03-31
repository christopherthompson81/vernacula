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
    public MainWindow()
    {
        InitializeComponent();
        
        // Set up window state restoration in Loaded event
        Loaded += MainWindow_Loaded;
    }

    private void MainWindow_Loaded(object? sender, RoutedEventArgs e)
    {
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);

        var s = App.Current.Settings.Current;
        Width  = s.WindowWidth;
        Height = s.WindowHeight;

        var virtualBounds = WindowHelper.GetVirtualScreenBounds();
        if (s.WindowLeft.HasValue && s.WindowTop.HasValue)
        {
            // Ensure the saved position is on a visible part of the virtual screen
            double l = s.WindowLeft.Value, t = s.WindowTop.Value;
            if (l >= virtualBounds.X &&
                l <  virtualBounds.Right  - 100 &&
                t >= virtualBounds.Y &&
                t <  virtualBounds.Bottom - 100)
            {
                Position = new PixelPoint((int)l, (int)t);
            }
        }

        if (s.WindowMaximized)
        {
            BeginMaximize();
        }
    }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        // If any job is running or queued, prompt the user before closing
        if (DataContext is MainViewModel vm && vm.IsAnyJobRunning)
        {
            var result = MessageBox.Show(
                Loc.Instance["confirm_exit_message"],
                Loc.Instance["confirm_exit_title"],
                MessageBoxButtons.YesNo,
                MessageBoxImage.Warning);

            if (result == MessageBoxResult.No)
            {
                e.Cancel = true;
                return;
            }

            vm.CancelAllJobs();
        }

        var s = App.Current.Settings.Current;
        
        // Get current window bounds
        double left = Position.X;
        double top = Position.Y;
        double width = Width;
        double height = Height;
        
        // If maximized, use restore bounds
        if (WindowState == WindowState.Maximized)
        {
            var restoreBounds = GetRestoreBounds();
            left = restoreBounds.X;
            top = restoreBounds.Y;
            width = restoreBounds.Width;
            height = restoreBounds.Height;
        }
        
        s.WindowLeft      = left;
        s.WindowTop       = top;
        s.WindowWidth     = width;
        s.WindowHeight    = height;
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
            Owner       = this,
            DataContext = vm.Settings,
        };
        _settingsWindow.Show();
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
        _helpWindow = new HelpWindow(topicId) { Owner = this };
        _helpWindow.Show();
    }

    private void About_Click(object? sender, RoutedEventArgs e) =>
        new Dialogs.AboutDialog { Owner = this }.ShowDialog();
}
