using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Vernacula.App.Models;

namespace Vernacula.App.Views.Dialogs;

public partial class DiariZenNoticeDialog : Window
{
    public DiariZenNoticeDialog()
    {
        InitializeComponent();
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
    }

    private void AgreementCheckBox_Changed(object? sender, RoutedEventArgs e) =>
        AgreeButton.IsEnabled = AgreementCheckBox.IsChecked == true;

    private void OpenWeightsRepo_Click(object? sender, RoutedEventArgs e) =>
        Process.Start(new ProcessStartInfo(
            "https://huggingface.co/christopherthompson81/diarizen_onnx") { UseShellExecute = true });

    private void Cancel_Click(object? sender, RoutedEventArgs e) => Close(false);

    private void Agree_Click(object? sender, RoutedEventArgs e) => Close(true);
}
