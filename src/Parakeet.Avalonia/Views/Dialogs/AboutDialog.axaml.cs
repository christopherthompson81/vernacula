using Avalonia.Controls;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public partial class AboutDialog : Window
{
    public AboutDialog()
    {
        InitializeComponent();
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    private void OK_Click(object sender, RoutedEventArgs e) => Close();
}
