using Avalonia.Controls;
using Avalonia.Interactivity;
using Vernacula.App.Models;

namespace Vernacula.App.Views.Dialogs;

public partial class CloseWhileRunningDialog : Window
{
    public CloseWhileRunningDialog()
    {
        InitializeComponent();
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    private void KeepWorking_Click(object? sender, RoutedEventArgs e) => Close(false);

    private void CloseAnyway_Click(object? sender, RoutedEventArgs e) => Close(true);
}
