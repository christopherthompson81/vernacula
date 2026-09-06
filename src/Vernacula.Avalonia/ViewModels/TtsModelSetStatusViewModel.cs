using Avalonia;
using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.App.Models;
using Vernacula.App.Services;

namespace Vernacula.App.ViewModels;

/// <summary>
/// One row of the Settings → TTS tab's "Models &amp; data" list: where a model set lives,
/// whether it is complete, and the buttons to point it elsewhere or fetch it. Mirrors the
/// VoxLingua/DiariZen status blocks in <see cref="SettingsViewModel"/>, factored into a row so
/// five sets do not mean five copies of the same twelve properties.
/// </summary>
internal sealed partial class TtsModelSetStatusViewModel : ObservableObject
{
    private readonly ModelManagerService _modelMgr;
    private readonly SettingsService     _svc;
    private readonly Action              _onChanged;
    private CancellationTokenSource?     _downloadCts;

    public ModelManagerService.TtsModelSet Set { get; }
    public string Name        { get; }
    public string Description { get; }

    [ObservableProperty] private string _locationText       = "";
    [ObservableProperty] private string _statusText         = "";
    [ObservableProperty] private IBrush _statusBrush        = Brushes.Gray;
    [ObservableProperty] private bool   _ready              = false;
    [ObservableProperty] private bool   _hasCustomLocation  = false;
    [ObservableProperty] private bool   _isDownloading      = false;
    [ObservableProperty] private double _downloadPercent    = 0;
    [ObservableProperty] private string _downloadStatusText = "";

    /// <summary>Hidden until the set is published somewhere fetchable (see ModelManagerService).</summary>
    public bool CanDownload => ModelManagerService.CanDownloadTtsModelSet(Set);

    public TtsModelSetStatusViewModel(
        ModelManagerService.TtsModelSet set, string name, string description,
        ModelManagerService modelMgr, SettingsService svc, Action onChanged)
    {
        Set          = set;
        Name         = name;
        Description  = description;
        _modelMgr    = modelMgr;
        _svc         = svc;
        _onChanged   = onChanged;
        StatusText   = $"Checking {name}…";
        LocationText = modelMgr.GetTtsModelSetDir(set);
    }

    // ── Where the set's directory is stored in AppSettings ───────────────────

    private string CustomLocation
    {
        get => Set switch
        {
            ModelManagerService.TtsModelSet.Chatterbox      => _svc.Current.ChatterboxBundleDir,
            ModelManagerService.TtsModelSet.Kokoro          => _svc.Current.KokoroModelDir,
            ModelManagerService.TtsModelSet.OmniVoice       => _svc.Current.OmniVoiceOnnxDir,
            ModelManagerService.TtsModelSet.OmniVoiceVoices => _svc.Current.OmniVoiceVoiceLib,
            ModelManagerService.TtsModelSet.PhonemizerData  => _svc.Current.PhonemizerDataDir,
            _                                               => "",
        };
        set
        {
            switch (Set)
            {
                case ModelManagerService.TtsModelSet.Chatterbox:      _svc.Current.ChatterboxBundleDir = value; break;
                case ModelManagerService.TtsModelSet.Kokoro:          _svc.Current.KokoroModelDir      = value; break;
                case ModelManagerService.TtsModelSet.OmniVoice:       _svc.Current.OmniVoiceOnnxDir    = value; break;
                case ModelManagerService.TtsModelSet.OmniVoiceVoices: _svc.Current.OmniVoiceVoiceLib   = value; break;
                case ModelManagerService.TtsModelSet.PhonemizerData:  _svc.Current.PhonemizerDataDir   = value; break;
            }
            _svc.Save();
        }
    }

    // ── Status ───────────────────────────────────────────────────────────────

    public async Task CheckAsync()
    {
        LocationText      = _modelMgr.GetTtsModelSetDir(Set);
        HasCustomLocation = !string.IsNullOrWhiteSpace(CustomLocation);

        IReadOnlyList<string> missing = [];
        await Task.Run(() => missing = _modelMgr.GetMissingTtsFiles(Set));

        Ready = missing.Count == 0;
        StatusText = Ready
            ? $"{Name} found and ready."
            : CanDownload
                ? $"{Name} incomplete ({missing.Count} missing): {string.Join(", ", missing)}"
                : $"{Name} incomplete ({missing.Count} missing): {string.Join(", ", missing)}. "
                  + "Not published for download yet — place the files in the folder below, or choose the folder that has them.";
        StatusBrush = Application.Current!.Resources[Ready ? "GreenBrush" : "YellowBrush"] as IBrush
                      ?? (Ready ? Brushes.LimeGreen : Brushes.Goldenrod);
    }

    // ── Commands ─────────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task PickLocation()
    {
        var path = await StoragePickers.PickFolderAsync($"Choose the {Name} folder");
        if (path is null) return;
        CustomLocation = path;
        await CheckAsync();
        _onChanged();
    }

    [RelayCommand]
    private async Task UseDefaultLocation()
    {
        CustomLocation = "";
        await CheckAsync();
        _onChanged();
    }

    [RelayCommand]
    private async Task Download()
    {
        IsDownloading      = true;
        DownloadPercent    = 0;
        DownloadStatusText = $"Starting {Name} download…";
        _downloadCts       = new CancellationTokenSource();

        var progress = new Progress<DownloadProgress>(p =>
        {
            DownloadPercent    = p.OverallPercent;
            DownloadStatusText = $"[{p.FileIndex + 1}/{p.TotalFiles}] {p.FileName} — " +
                (string.IsNullOrEmpty(p.OverallSizeText)
                    ? p.SizeText
                    : $"{p.SizeText}  |  {p.OverallSizeText} total  ({p.OverallPercent:F1}%)");
        });

        try
        {
            await _modelMgr.DownloadMissingTtsModelsAsync(Set, progress, _downloadCts.Token);
            DownloadStatusText = $"{Name} download complete.";
            await CheckAsync();
            _onChanged();
        }
        catch (OperationCanceledException)
        {
            DownloadStatusText = $"{Name} download cancelled.";
        }
        catch (Exception ex)
        {
            DownloadStatusText = $"{Name} download failed: {ex.Message}";
        }
        finally
        {
            IsDownloading = false;
        }
    }

    [RelayCommand] private void CancelDownload() => _downloadCts?.Cancel();
}
