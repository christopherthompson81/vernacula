using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;
using ParakeetCSharp.ViewModels;

namespace ParakeetCSharp.Views;

public partial class SettingsWindow : Window
{
    public SettingsWindow()
    {
        InitializeComponent();
    }

    private void Window_SourceInitialized(object sender, EventArgs e)
    {
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        ThemeManager.ThemeChanged += OnThemeChanged;
    }

    private void OnThemeChanged(AppTheme theme) =>
        WindowHelper.SetDarkMode(this, theme == AppTheme.Dark);

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        if (DataContext is SettingsViewModel vm)
            await vm.InitializeAsync();
    }

    private void Close_Click(object sender, RoutedEventArgs e) => Close();

    protected override void OnClosed(EventArgs e)
    {
        ThemeManager.ThemeChanged -= OnThemeChanged;
        base.OnClosed(e);
    }
}
