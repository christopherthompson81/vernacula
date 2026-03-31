using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using ParakeetCSharp.Models;

namespace ParakeetCSharp.Views.Dialogs;

public partial class AddSpeakerDialog : Window
{
    public string SpeakerName { get; private set; } = "";

    public AddSpeakerDialog()
    {
        InitializeComponent();
        Loaded += (_, _) =>
        {
            WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
            NameBox.Focus();
        };
    }

    private void Add_Click(object sender, RoutedEventArgs e) => TryConfirm();

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        Close();
    }

    private void NameBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
            TryConfirm();
        else if (e.Key == Key.Escape)
        {
            Close();
        }
    }

    private void TryConfirm()
    {
        string name = NameBox.Text.Trim();
        if (string.IsNullOrEmpty(name)) return;
        SpeakerName  = name;
        Close();
    }
}
