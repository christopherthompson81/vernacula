using Avalonia.Controls;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public partial class NewTranscriptionWindow : Window
{
    public NewTranscriptionWindow()
    {
        InitializeComponent();
        Loaded += (_, _) =>
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }
}
