using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using ParakeetCSharp.Models;
using ParakeetCSharp.ViewModels;
using ParakeetCSharp.Views.Dialogs;
using System.Diagnostics;
using System.ComponentModel;

namespace ParakeetCSharp.Views;

public partial class SettingsWindow : Window
{
    public SettingsWindow()
    {
        InitializeComponent();
        ApplyLocalizedText();
        Loc.Instance.PropertyChanged += OnLocalePropertyChanged;
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
        ApplyLocalizedText();

        if (DataContext is SettingsViewModel vm)
            await vm.InitializeAsync();
    }

    private void Close_Click(object sender, RoutedEventArgs e) => Close();

    private void OnLocalePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(Loc.CurrentLanguage) && e.PropertyName != "Item[]")
            return;

        ApplyLocalizedText();
    }

    private void ApplyLocalizedText()
    {
        Title = Loc.Instance["settings_window_title"];
        HardwareSectionHeader.Text = Loc.Instance["settings_section_hardware"];
        HardwareCheckingText.Text = Loc.Instance["settings_hw_checking"];
        CudaToolkitLabel.Text = Loc.Instance["settings_hw_cuda_toolkit"];
        DownloadCudaButton.Content = Loc.Instance["settings_hw_download_cuda"];
        CudnnLabel.Text = Loc.Instance["settings_hw_cudnn"];
        DownloadCudnnButton.Content = Loc.Instance["settings_hw_download_cudnn"];
        CudaEpLabel.Text = Loc.Instance["settings_hw_cuda_ep"];
        NoGpuText.Text = Loc.Instance["settings_hw_no_gpu"];
        RecheckHardwareButton.Content = Loc.Instance["settings_hw_recheck"];

        ModelsSectionHeader.Text = Loc.Instance["settings_section_models"];
        UpdateModelsButton.Content = Loc.Instance["btn_update_models"];
        DismissUpdateBannerButton.Content = Loc.Instance["btn_dismiss"];
        DownloadModelsButton.Content = Loc.Instance["btn_download_models"];
        CancelDownloadButton.Content = Loc.Instance["btn_cancel"];
        CheckUpdatesButton.Content = Loc.Instance["btn_check_updates"];
        ModelPrecisionLabel.Text = Loc.Instance["menu_model_precision"];
        PrecisionFp32Radio.Content = Loc.Instance["menu_precision_fp16"];
        PrecisionInt8Radio.Content = Loc.Instance["menu_precision_int8"];

        SegmentationSectionHeader.Text = Loc.Instance["settings_section_segmentation"];
        SegmentationSileroLabel.Text = Loc.Instance["settings_segmentation_vad"];
        SegmentationSileroDescription.Text = Loc.Instance["settings_segmentation_vad_desc"];
        SegmentationSortformerLabel.Text = Loc.Instance["settings_segmentation_diarization"];
        SegmentationSortformerDescription.Text = Loc.Instance["settings_segmentation_diarization_desc"];
        SegmentationDiariZenLabel.Text = Loc.Instance["settings_segmentation_diarizen"];
        SegmentationDiariZenDescription.Text = Loc.Instance["settings_segmentation_diarizen_desc"];
        ReviewDiariZenNoticeButton.Content = "Review External Weights Notice";
        ChooseDiariZenFolderButton.Content = "Choose Weights Folder";
        DownloadDiariZenButton.Content = "Download External Weights";
        OpenDiariZenRepoButton.Content = "Open External Weights Repo";

        EditorSectionHeader.Text = Loc.Instance["settings_section_editor"];
        EditorPlaybackModeLabel.Text = Loc.Instance["settings_editor_playback_mode"];
        EditorContinuousRadio.Content = Loc.Instance["editor_mode_continuous"];
        EditorSingleRadio.Content = Loc.Instance["editor_mode_single"];
        EditorAutoAdvanceRadio.Content = Loc.Instance["editor_auto_advance"];

        AppearanceSectionHeader.Text = Loc.Instance["settings_section_appearance"];
        ThemeLabel.Text = Loc.Instance["menu_theme"];
        ThemeDarkRadio.Content = Loc.Instance["menu_theme_dark"];
        ThemeLightRadio.Content = Loc.Instance["menu_theme_light"];

        LanguageSectionHeader.Text = Loc.Instance["settings_section_language"];
        OkButton.Content = Loc.Instance["btn_ok"];
    }

    protected override void OnClosed(EventArgs e)
    {
        Loc.Instance.PropertyChanged -= OnLocalePropertyChanged;
        ThemeManager.ThemeChanged -= OnThemeChanged;
        base.OnClosed(e);
    }

    private async Task<bool> EnsureDiariZenNoticeAcceptedAsync()
    {
        if (DataContext is not SettingsViewModel vm)
            return false;

        if (vm.HasAcceptedDiariZenNotice)
            return true;

        var dialog = new DiariZenNoticeDialog();
        bool accepted = await dialog.ShowDialog<bool>(this);
        if (accepted)
            vm.MarkDiariZenNoticeAccepted();
        return accepted;
    }

    private async void ReviewDiariZenNotice_Click(object? sender, RoutedEventArgs e)
    {
        _ = await EnsureDiariZenNoticeAcceptedAsync();
    }

    private async void ChooseDiariZenFolder_Click(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not SettingsViewModel vm)
            return;

        if (!await EnsureDiariZenNoticeAcceptedAsync())
            return;

        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Choose a folder containing external DiariZen weights",
            AllowMultiple = false,
        });

        if (folders.Count == 0)
            return;

        await vm.SetDiariZenModelsDirAsync(folders[0].Path.LocalPath);
    }

    private async void DownloadDiariZenWeights_Click(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not SettingsViewModel vm)
            return;

        if (!await EnsureDiariZenNoticeAcceptedAsync())
            return;

        await vm.DownloadDiariZenModelsCommand.ExecuteAsync(null);
    }

    private void OpenDiariZenRepo_Click(object? sender, RoutedEventArgs e) =>
        Process.Start(new ProcessStartInfo(
            "https://huggingface.co/christopherthompson81/diarizen_onnx") { UseShellExecute = true });
}
