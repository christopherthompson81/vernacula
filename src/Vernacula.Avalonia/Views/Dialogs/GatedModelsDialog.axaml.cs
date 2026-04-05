using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Vernacula.App.Models;
using Vernacula.App.ViewModels;

namespace Vernacula.App.Views.Dialogs;

internal partial class GatedModelsDialog : Window
{
    private const string ReferenceRepoUrl = "https://huggingface.co/christopherthompson81/diarizen_onnx";
    private const string UpstreamRepoUrl = "https://github.com/BUTSpeechFIT/DiariZen";
    private const string UpstreamModelCardUrl = "https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md";
    private const string LicenseUrl = "https://creativecommons.org/licenses/by-nc/4.0/";
    private const string ModelLicenseUrl = "https://github.com/BUTSpeechFIT/DiariZen/blob/main/MODEL_LICENSE";

    private readonly SettingsViewModel _settings;

    public GatedModelsDialog(SettingsViewModel settings)
    {
        _settings = settings;
        InitializeComponent();
        DataContext = settings;
        WindowHelper.SetDarkMode(this, App.Current.Settings.Current.Theme == AppTheme.Dark);
        RefreshDiariZenState();
    }

    private void RefreshDiariZenState()
    {
        bool accepted = _settings.HasAcceptedDiariZenNotice;
        DiariZenStatusText.Text = accepted ? "Unlocked" : "Locked";
        DiariZenAgreementCheckBox.IsVisible = !accepted;
        DiariZenAcceptedText.IsVisible = accepted;
        DiariZenManagementPanel.IsVisible = accepted;
        AcceptDiariZenButton.IsVisible = !accepted;
        AcceptDiariZenButton.IsEnabled = !accepted && DiariZenAgreementCheckBox.IsChecked == true;
        DownloadDiariZenButton.IsEnabled = accepted && !_settings.IsDownloadingDiariZen;
    }

    private void DiariZenAgreementCheckBox_Changed(object? sender, RoutedEventArgs e) =>
        AcceptDiariZenButton.IsEnabled = DiariZenAgreementCheckBox.IsChecked == true;

    private void OpenWeightsRepo_Click(object? sender, RoutedEventArgs e) => OpenUrl(ReferenceRepoUrl);

    private void OpenLicense_Click(object? sender, RoutedEventArgs e) => OpenUrl(LicenseUrl);

    private void OpenReferenceRepo_Click(object? sender, RoutedEventArgs e) => OpenUrl(ReferenceRepoUrl);

    private void OpenUpstreamRepo_Click(object? sender, RoutedEventArgs e) => OpenUrl(UpstreamRepoUrl);

    private void OpenUpstreamModelCard_Click(object? sender, RoutedEventArgs e) => OpenUrl(UpstreamModelCardUrl);

    private void OpenModelLicense_Click(object? sender, RoutedEventArgs e) => OpenUrl(ModelLicenseUrl);

    private void AcceptDiariZen_Click(object? sender, RoutedEventArgs e)
    {
        _settings.MarkDiariZenNoticeAccepted();
        RefreshDiariZenState();
    }

    private async void ChooseDiariZenFolder_Click(object? sender, RoutedEventArgs e)
    {
        if (!_settings.HasAcceptedDiariZenNotice)
            return;

        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Choose a folder containing external DiariZen weights",
            AllowMultiple = false,
        });

        if (folders.Count == 0)
            return;

        await _settings.SetDiariZenModelsDirAsync(folders[0].Path.LocalPath);
        RefreshDiariZenState();
    }

    private async void DownloadDiariZenWeights_Click(object? sender, RoutedEventArgs e)
    {
        if (!_settings.HasAcceptedDiariZenNotice)
            return;

        RefreshDiariZenState();
        await _settings.DownloadDiariZenModelsCommand.ExecuteAsync(null);
        RefreshDiariZenState();
    }

    private void Close_Click(object? sender, RoutedEventArgs e) => Close();

    private static void OpenUrl(string url) =>
        Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
}
