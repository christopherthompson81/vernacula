using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.Base;
using Vernacula.Base.Models;
using Vernacula.App.Models;
using Vernacula.App.Services;

namespace Vernacula.App.ViewModels;

internal partial class SettingsViewModel : ObservableObject
{
    private const string DiariZenGatedModelId = SettingsService.DiariZenGatedModelId;

    private readonly SettingsService     _svc;
    private readonly ModelManagerService _modelMgr;

    // ── Theme / Language ─────────────────────────────────────────────────────

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDark), nameof(IsLight))]
    private AppTheme _selectedTheme;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSileroVad), nameof(IsSortformer), nameof(IsDiariZen), nameof(IsVibeVoiceBuiltin))]
    [NotifyPropertyChangedFor(nameof(ShowStandardSegmentationOptions), nameof(ShowVibeVoiceBuiltinSegmentation), nameof(ShowDiariZenInSegmentation), nameof(ShowGatedSegmentationHint))]
    private SegmentationMode _selectedSegmentation;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsAsrParakeet), nameof(IsAsrCohere), nameof(IsAsrQwen3Asr), nameof(IsAsrVibeVoice), nameof(ShowStandardSegmentationOptions), nameof(ShowVibeVoiceBuiltinSegmentation), nameof(ShowDiariZenInSegmentation), nameof(ShowGatedSegmentationHint), nameof(CanUseVibeVoiceAsr), nameof(VibeVoiceAsrLabel), nameof(VibeVoiceAsrDescription))]
    private AsrBackend _selectedAsrBackend;

    [ObservableProperty]
    private CohereLanguageOption? _selectedCohereLanguage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowCohereLanguagePicker))]
    [NotifyPropertyChangedFor(nameof(IsDenoiserNone), nameof(IsDenoiserDfn3))]
    private DenoiserMode _selectedDenoiser;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsEditorSingle), nameof(IsEditorAutoAdvance), nameof(IsEditorContinuous))]
    private PlaybackMode _selectedEditorPlaybackMode;

    [ObservableProperty] private string             _selectedLanguage;
    [ObservableProperty] private Loc.LanguageInfo?  _selectedLanguageInfo;

    /// <summary>Called after a successful download so Home can refresh its model status.</summary>
    public Action? AfterDownload       { get; set; }
    /// <summary>Called after segmentation mode or DiariZen status changes.</summary>
    public Action? OnSegmentationChanged { get; set; }
    /// <summary>Called when the update check finds outdated files.</summary>
    public Action? OnUpdateAvailable   { get; set; }
    /// <summary>Called when the update check completes and no outdated files are found.</summary>
    public Action? OnUpdateCheckComplete { get; set; }

    public bool IsDark              => SelectedTheme == AppTheme.Dark;
    public bool IsLight             => SelectedTheme == AppTheme.Light;
    public bool IsSileroVad         => SelectedSegmentation == SegmentationMode.SileroVad;
    public bool IsSortformer        => SelectedSegmentation == SegmentationMode.Sortformer;
    public bool IsDiariZen          => SelectedSegmentation == SegmentationMode.DiariZen;
    public bool IsVibeVoiceBuiltin  => SelectedSegmentation == SegmentationMode.VibeVoiceBuiltin;
    public bool IsAsrParakeet       => SelectedAsrBackend == AsrBackend.Parakeet;
    public bool IsAsrCohere         => SelectedAsrBackend == AsrBackend.Cohere;
    public bool IsAsrQwen3Asr       => SelectedAsrBackend == AsrBackend.Qwen3Asr;
    public bool IsAsrVibeVoice      => SelectedAsrBackend == AsrBackend.VibeVoice;
    public bool CanUseVibeVoiceAsr  => CudaEpWorking;
    public string VibeVoiceAsrLabel => CanUseVibeVoiceAsr ? "VibeVoice-ASR" : "VibeVoice-ASR (Unavailable - CUDA Missing)";
    public string VibeVoiceAsrDescription => CanUseVibeVoiceAsr
        ? "Whole-recording ASR with built-in diarization. Downloads into the vibevoice_asr models folder."
        : "Unavailable because the CUDA execution provider check did not pass.";
    public bool ShowCohereLanguagePicker => SelectedAsrBackend == AsrBackend.Cohere;
    public bool IsDenoiserNone      => SelectedDenoiser == DenoiserMode.None;
    public bool IsDenoiserDfn3      => SelectedDenoiser == DenoiserMode.DeepFilterNet3;
    public bool ShowStandardSegmentationOptions => SelectedAsrBackend != AsrBackend.VibeVoice;
    public bool ShowVibeVoiceBuiltinSegmentation => SelectedAsrBackend == AsrBackend.VibeVoice;
    public bool ShowDiariZenInSegmentation => HasAcceptedDiariZenNotice && ShowStandardSegmentationOptions;
    public bool ShowGatedSegmentationHint => !HasAcceptedDiariZenNotice && ShowStandardSegmentationOptions;
    public bool IsEditorSingle      => SelectedEditorPlaybackMode == PlaybackMode.Single;
    public bool IsEditorAutoAdvance => SelectedEditorPlaybackMode == PlaybackMode.AutoAdvance;
    public bool IsEditorContinuous  => SelectedEditorPlaybackMode == PlaybackMode.Continuous;
    public bool HasUnlockedGatedModels => HasAcceptedDiariZenNotice;
    public string GatedModelsStatusText => HasUnlockedGatedModels
        ? "Unlocked gated models: DiariZen"
        : "Some optional models require accepting their own license terms before they appear in settings.";

    public record CohereLanguageOption(string Code, string DisplayName);
    public static readonly IReadOnlyList<CohereLanguageOption> CohereLanguages =
    [
        new("",   "Auto-detect"),
        new("ar", "Arabic"),
        new("de", "German"),
        new("el", "Greek"),
        new("en", "English"),
        new("es", "Spanish"),
        new("fr", "French"),
        new("it", "Italian"),
        new("ja", "Japanese"),
        new("ko", "Korean"),
        new("nl", "Dutch"),
        new("pl", "Polish"),
        new("pt", "Portuguese"),
        new("vi", "Vietnamese"),
        new("zh", "Chinese"),
    ];

    // ── Hardware check state ─────────────────────────────────────────────────

    [ObservableProperty] private bool   _gpuDetected          = false;
    [ObservableProperty] private string _gpuVramText          = "";
    [ObservableProperty] private bool   _cudaToolkitInstalled = false;
    [ObservableProperty] private bool   _cudnnInstalled       = false;
    [ObservableProperty] private bool   _cudaEpWorking        = false;
    [ObservableProperty] private bool   _isCheckingHardware   = false;
    [ObservableProperty] private string _batchCeilingText     = "";

    // ── Model management state ───────────────────────────────────────────────

    [ObservableProperty] private string _modelStatusText     = "";
    [ObservableProperty] private IBrush _modelStatusBrush    = Brushes.Gray;
    [ObservableProperty] private bool   _modelsReady         = false;
    [ObservableProperty] private bool   _downloadVisible     = false;
    [ObservableProperty] private bool   _isDownloading       = false;
    [ObservableProperty] private double _downloadPercent     = 0;
    [ObservableProperty] private string _downloadStatusText  = "";
    [ObservableProperty] private bool   _updateBannerVisible = false;
    [ObservableProperty] private string _updateBannerText    = "";
    [ObservableProperty] private bool   _isCheckingUpdates   = false;
    [ObservableProperty] private bool   _hasOutdatedFiles    = false;
    [ObservableProperty] private string _diariZenStatusText  = "";
    [ObservableProperty] private IBrush _diariZenStatusBrush = Brushes.Gray;
    [ObservableProperty] private string _diariZenModelsLocationText = "";
    [ObservableProperty] private bool   _diariZenReady       = false;
    [ObservableProperty] private bool   _isDownloadingDiariZen = false;
    [ObservableProperty] private double _diariZenDownloadPercent = 0;
    [ObservableProperty] private string _diariZenDownloadStatusText = "";

    private CancellationTokenSource?  _downloadCts;
    private CancellationTokenSource?  _diariZenDownloadCts;
    private IReadOnlyList<string>     _lastMissing    = [];
    private IReadOnlyList<string>     _lastPresent    = [];
    private IReadOnlyList<string>     _lastDiariZenMissing = [];
    private IReadOnlyList<string>     _outdatedFiles  = [];
    private bool                      _modelCheckDone = false;
    private double                    _batchSecs      = 0;
    private bool                      _batchIsFallback = true;
    private SegmentationMode          _lastNonVibeSegmentation = SegmentationMode.Sortformer;

    // ── Construction ─────────────────────────────────────────────────────────

    public SettingsViewModel(SettingsService svc, ModelManagerService modelMgr)
    {
        _svc                   = svc;
        _modelMgr              = modelMgr;
        _selectedTheme                = svc.Current.Theme;
        _selectedAsrBackend           = svc.Current.AsrBackend;
        _selectedSegmentation         = NormalizeSegmentationForBackend(
            svc.Current.Segmentation == SegmentationMode.DiariZen && !svc.IsGatedModelAccepted(DiariZenGatedModelId)
                ? SegmentationMode.Sortformer
                : svc.Current.Segmentation,
            _selectedAsrBackend);
        if (_selectedSegmentation != SegmentationMode.VibeVoiceBuiltin)
            _lastNonVibeSegmentation = _selectedSegmentation;
        _selectedCohereLanguage       = CohereLanguages.FirstOrDefault(l => l.Code == svc.Current.CohereLanguage)
                                        ?? CohereLanguages[0];
        _selectedDenoiser             = svc.Current.Denoiser;
        _selectedEditorPlaybackMode   = svc.Current.EditorPlaybackMode;
        _selectedLanguage             = svc.Current.Language;
        _selectedLanguageInfo         = Loc.Languages.FirstOrDefault(l => l.Code == svc.Current.Language) 
                                        ?? Loc.Languages.FirstOrDefault(l => l.Code == "en")!;
        if (svc.Current.Segmentation != _selectedSegmentation)
        {
            svc.Current.Segmentation = _selectedSegmentation;
            svc.Save();
        }
        ModelStatusText    = Loc.Instance["model_status_checking"];
        DiariZenStatusText = "Checking external DiariZen weights…";
        DiariZenModelsLocationText = _svc.GetDiariZenModelsDir();

        Loc.Instance.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName != "Item[]") return;
            if (_modelCheckDone)
            {
                ApplyModelStatusText();
                ApplyUpdateBannerText();
            }
            if (_batchSecs > 0)
                ApplyBatchCeilingText();
        };
    }

    // ── Settings change handlers ─────────────────────────────────────────────

    partial void OnSelectedThemeChanged(AppTheme value)
    {
        _svc.Current.Theme = value;
        _svc.Save();
        ThemeManager.Apply(value);
    }

    partial void OnSelectedSegmentationChanged(SegmentationMode value)
    {
        SegmentationMode normalized = NormalizeSegmentationForBackend(value, SelectedAsrBackend);
        if (normalized != value)
        {
            SelectedSegmentation = normalized;
            return;
        }

        if (value == SegmentationMode.DiariZen && !HasAcceptedDiariZenNotice)
            return;

        if (value != SegmentationMode.VibeVoiceBuiltin)
            _lastNonVibeSegmentation = value;

        _svc.Current.Segmentation = value;
        _svc.Save();
        OnSegmentationChanged?.Invoke();
    }

    partial void OnSelectedAsrBackendChanged(AsrBackend value)
    {
        if (value == AsrBackend.VibeVoice && !CanUseVibeVoiceAsr)
        {
            SelectedAsrBackend = AsrBackend.Parakeet;
            return;
        }

        if (value == AsrBackend.VibeVoice && SelectedSegmentation != SegmentationMode.VibeVoiceBuiltin)
            _lastNonVibeSegmentation = NormalizeSegmentationForBackend(SelectedSegmentation, AsrBackend.Parakeet);

        _svc.Current.AsrBackend = value;
        SegmentationMode normalized = NormalizeSegmentationForBackend(SelectedSegmentation, value);
        if (normalized != SelectedSegmentation)
            SelectedSegmentation = normalized;
        else
        {
            _svc.Current.Segmentation = normalized;
            _svc.Save();
        }

        _svc.Save();
        OnPropertyChanged(nameof(ShowCohereLanguagePicker));
        OnSegmentationChanged?.Invoke();
        _ = CheckModelsAsync();
    }

    partial void OnSelectedCohereLanguageChanged(CohereLanguageOption? value)
    {
        _svc.Current.CohereLanguage = value?.Code ?? "";
        _svc.Save();
    }

    partial void OnSelectedDenoiserChanged(DenoiserMode value)
    {
        _svc.Current.Denoiser = value;
        _svc.Save();
    }

    partial void OnSelectedEditorPlaybackModeChanged(PlaybackMode value)
    {
        _svc.Current.EditorPlaybackMode = value;
        _svc.Save();
    }

    partial void OnSelectedLanguageChanged(string value)
    {
        try
        {
            _svc.Current.Language = value;
            _svc.Save();
            Loc.Instance.SetLanguage(value);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to set language {value}: {ex}");
        }
    }

    partial void OnSelectedLanguageInfoChanged(Loc.LanguageInfo? value)
    {
        if (value is not null)
        {
            try
            {
                _svc.Current.Language = value.Code;
                _svc.Save();
                SelectedLanguage = value.Code;
                Loc.Instance.SetLanguage(value.Code);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Failed to set language {value.Code}: {ex}");
            }
        }
    }

    [RelayCommand] private void SetTheme(string n)              { if (Enum.TryParse<AppTheme>(n,         out var t)) SelectedTheme              = t; }
    [RelayCommand] private void SetSegmentation(string n)       { if (Enum.TryParse<SegmentationMode>(n, out var s)) SelectedSegmentation       = s; }
    [RelayCommand] private void SetAsrBackend(string n)         { if (Enum.TryParse<AsrBackend>(n,      out var a)) SelectedAsrBackend         = a; }
    [RelayCommand] private void SetDenoiser(string n)           { if (Enum.TryParse<DenoiserMode>(n,     out var d)) SelectedDenoiser           = d; }
    [RelayCommand] private void SetEditorPlaybackMode(string n) { if (Enum.TryParse<PlaybackMode>(n,     out var m)) SelectedEditorPlaybackMode = m; }
    [RelayCommand] private void SetLanguage(string l)           => SelectedLanguage = l;

    // ── Hardware checks ──────────────────────────────────────────────────────

    [RelayCommand]
    internal async Task RecheckHardwareAsync()
    {
        IsCheckingHardware = true;
        bool cudaOk = false;

        await Task.Run(() =>
        {
            var (totalMb, _)   = HardwareInfo.GetGpuMemoryMb();
            GpuDetected        = totalMb > 0;
            GpuVramText        = totalMb > 0
                ? Loc.Instance.T("settings_hw_vram", new() { ["vram"] = $"{totalMb / 1024.0:F1}" })
                : "";

            CudaToolkitInstalled = HardwareInfo.IsCudaToolkitInstalled();
            CudnnInstalled       = HardwareInfo.IsCudnnInstalled();
            (cudaOk, _)          = _modelMgr.CheckCuda();
        });
        CudaEpWorking = cudaOk;
        OnPropertyChanged(nameof(CanUseVibeVoiceAsr));
        OnPropertyChanged(nameof(VibeVoiceAsrLabel));
        OnPropertyChanged(nameof(VibeVoiceAsrDescription));

        if (!CudaEpWorking && SelectedAsrBackend == AsrBackend.VibeVoice)
            SelectedAsrBackend = AsrBackend.Parakeet;

        // Batch ceiling — query free VRAM (accurate post-load figure)
        var (_, freeMb) = HardwareInfo.GetGpuMemoryMb();
        long batchFrames;
        bool isFallback;
        if (freeMb > 0)
        {
            double avail = freeMb - Config.VramSafetyMb;
            long   frames = avail > 0
                ? (long)((avail - Config.VramInterceptMb) / Config.VramSlopePerSample)
                : Config.FallbackMaxFrames;
            batchFrames = frames > 0 ? frames : Config.FallbackMaxFrames;
            isFallback  = false;
        }
        else
        {
            batchFrames = Config.FallbackMaxFrames;
            isFallback  = true;
        }

        _batchSecs       = batchFrames / (double)Config.SampleRate;
        _batchIsFallback = isFallback;
        ApplyBatchCeilingText();

        IsCheckingHardware = false;
    }

    [RelayCommand]
    private void OpenCudaDownloadPage() =>
        Process.Start(new ProcessStartInfo(
            "https://developer.nvidia.com/cuda-downloads") { UseShellExecute = true });

    [RelayCommand]
    private void OpenCudnnDownloadPage() =>
        Process.Start(new ProcessStartInfo(
            "https://developer.nvidia.com/cudnn-downloads") { UseShellExecute = true });

    private void ApplyBatchCeilingText()
    {
        string key   = _batchIsFallback ? "settings_hw_batch_fallback" : "settings_hw_batch_ceiling";
        BatchCeilingText = Loc.Instance.T(key, new() { ["secs"] = $"{_batchSecs:F0}" });
    }

    // ── Model management ─────────────────────────────────────────────────────

    private void ApplyUpdateBannerText()
    {
        if (_outdatedFiles.Count > 0)
        {
            UpdateBannerText = Loc.Instance.T("update_banner_text",
                new() { ["count"] = _outdatedFiles.Count.ToString() });
            return;
        }

        if (UpdateBannerVisible && !IsCheckingUpdates)
            UpdateBannerText = Loc.Instance["settings_models_up_to_date"];
    }

    private void ApplyModelStatusText(string? sortformerWarning = null)
    {
        if (_lastMissing.Count == 0 && string.IsNullOrEmpty(sortformerWarning))
        {
            ModelStatusText = Loc.Instance.T("model_status_ok", new() { ["count"] = _lastPresent.Count.ToString() });
            return;
        }

        if (_lastMissing.Count > 0)
        {
            if (SelectedAsrBackend == AsrBackend.Cohere)
            {
                ModelStatusText = $"Missing {_lastMissing.Count} required model file(s): {string.Join(", ", _lastMissing)}. " +
                                  $"Place Cohere weights under {_svc.GetCohereModelsDir()}.";
                return;
            }

            if (SelectedAsrBackend == AsrBackend.Qwen3Asr)
            {
                ModelStatusText = $"Missing {_lastMissing.Count} required model file(s): {string.Join(", ", _lastMissing)}. " +
                                  $"Place Qwen3-ASR weights under {_svc.GetQwen3AsrModelsDir()}.";
                return;
            }

            if (SelectedAsrBackend == AsrBackend.VibeVoice)
            {
                ModelStatusText = $"Missing {_lastMissing.Count} required model file(s): {string.Join(", ", _lastMissing)}. " +
                                  $"Use Download Missing Models, or place VibeVoice-ASR weights under {_svc.GetVibeVoiceModelsDir()}.";
                return;
            }

            ModelStatusText = Loc.Instance.T("model_status_missing",
                new() { ["count"] = _lastMissing.Count.ToString(),
                        ["files"] = string.Join(", ", _lastMissing) });
            if (!string.IsNullOrEmpty(sortformerWarning))
                ModelStatusText += $"  ({sortformerWarning})";
            return;
        }

        // All files present but Sortformer model is outdated
        if (!string.IsNullOrEmpty(sortformerWarning))
        {
            ModelStatusText = $"{Loc.Instance.T("model_status_ok", new() { ["count"] = _lastPresent.Count.ToString() })}  {sortformerWarning}";
            return;
        }

        ModelStatusText = Loc.Instance.T("model_status_ok", new() { ["count"] = _lastPresent.Count.ToString() });
    }

    private void ApplyDiariZenStatusText()
    {
        DiariZenModelsLocationText = _svc.GetDiariZenModelsDir();
        DiariZenStatusText = _lastDiariZenMissing.Count == 0
            ? "External DiariZen weights found and ready."
            : $"External DiariZen weights incomplete ({_lastDiariZenMissing.Count} file(s) missing): {string.Join(", ", _lastDiariZenMissing)}";

        DiariZenStatusBrush = Application.Current!.Resources[_lastDiariZenMissing.Count == 0 ? "GreenBrush" : "YellowBrush"] as IBrush
                              ?? (_lastDiariZenMissing.Count == 0 ? Brushes.LimeGreen : Brushes.Goldenrod);
        DiariZenReady = _lastDiariZenMissing.Count == 0;
    }

    [RelayCommand]
    internal async Task CheckModelsAsync()
    {
        ModelStatusText  = Loc.Instance["model_status_checking"];
        ModelStatusBrush = Application.Current!.Resources["SubtextBrush"] as IBrush ?? Brushes.Gray;

        IReadOnlyList<string> missing = [], present = [];
        string? sortformerWarning = null;

        await Task.Run(() =>
        {
            missing         = _modelMgr.GetMissingFiles();
            present         = _modelMgr.GetPresentFiles();
            ModelsReady     = missing.Count == 0;
            DownloadVisible = _modelMgr.GetMissingDownloadableFiles().Count > 0;
        });

        // Check Sortformer model version (non-blocking — does not block on network)
        try
        {
            var result = await _modelMgr.CheckSortformerModelAsync();
            if (result.HasValue && result.Value.isCorrect == false)
                sortformerWarning = result.Value.message;
        }
        catch { /* Network or other error — silently skip */ }

        _lastMissing    = missing;
        _lastPresent    = present;
        _modelCheckDone = true;
        ApplyModelStatusText(sortformerWarning);

        ModelStatusBrush = Application.Current.Resources[missing.Count == 0 && sortformerWarning == null ? "GreenBrush" : "YellowBrush"] as IBrush
                           ?? (missing.Count == 0 && sortformerWarning == null ? Brushes.LimeGreen : Brushes.Goldenrod);
    }

    internal async Task CheckDiariZenModelsAsync()
    {
        DiariZenStatusText = "Checking external DiariZen weights…";
        DiariZenModelsLocationText = _svc.GetDiariZenModelsDir();

        IReadOnlyList<string> missing = [];
        await Task.Run(() => missing = _modelMgr.GetMissingDiariZenFiles(_svc.GetDiariZenModelsDir()));

        _lastDiariZenMissing = missing;
        ApplyDiariZenStatusText();
        OnSegmentationChanged?.Invoke();
    }

    public bool HasAcceptedDiariZenNotice => _svc.IsGatedModelAccepted(DiariZenGatedModelId);

    internal void MarkDiariZenNoticeAccepted()
    {
        if (!_svc.AcceptGatedModel(DiariZenGatedModelId))
            return;

        OnPropertyChanged(nameof(HasAcceptedDiariZenNotice));
        OnPropertyChanged(nameof(ShowDiariZenInSegmentation));
        OnPropertyChanged(nameof(ShowGatedSegmentationHint));
        OnPropertyChanged(nameof(HasUnlockedGatedModels));
        OnPropertyChanged(nameof(GatedModelsStatusText));
    }

    private SegmentationMode NormalizeSegmentationForBackend(SegmentationMode requested, AsrBackend backend)
    {
        if (backend == AsrBackend.VibeVoice)
            return SegmentationMode.VibeVoiceBuiltin;

        if (requested == SegmentationMode.VibeVoiceBuiltin)
            return NormalizeStandardSegmentation(_lastNonVibeSegmentation);

        return NormalizeStandardSegmentation(requested);
    }

    private SegmentationMode NormalizeStandardSegmentation(SegmentationMode requested)
    {
        if (requested == SegmentationMode.DiariZen && !HasAcceptedDiariZenNotice)
            return SegmentationMode.Sortformer;

        return requested == SegmentationMode.VibeVoiceBuiltin
            ? SegmentationMode.Sortformer
            : requested;
    }

    internal async Task SetDiariZenModelsDirAsync(string path)
    {
        _svc.Current.DiariZenModelsDir = path;
        _svc.Save();
        await CheckDiariZenModelsAsync();
    }

    [RelayCommand]
    private async Task DownloadDiariZenModels()
    {
        IsDownloadingDiariZen = true;
        DiariZenDownloadStatusText = "Starting external DiariZen weights download…";
        DiariZenDownloadPercent = 0;
        _diariZenDownloadCts = new CancellationTokenSource();

        var progress = new Progress<DownloadProgress>(p =>
        {
            DiariZenDownloadPercent = p.OverallPercent;
            DiariZenDownloadStatusText = $"[{p.FileIndex + 1}/{p.TotalFiles}] {p.FileName} — " +
                (string.IsNullOrEmpty(p.OverallSizeText)
                    ? p.SizeText
                    : $"{p.SizeText}  |  {p.OverallSizeText} total  ({p.OverallPercent:F1}%)");
        });

        try
        {
            await _modelMgr.DownloadMissingDiariZenModelsAsync(progress, _svc.GetDiariZenModelsDir(), _diariZenDownloadCts.Token);
            DiariZenDownloadStatusText = "External DiariZen weights download complete.";
            await CheckDiariZenModelsAsync();
            AfterDownload?.Invoke();
        }
        catch (OperationCanceledException)
        {
            DiariZenDownloadStatusText = "External DiariZen weights download cancelled.";
        }
        catch (Exception ex)
        {
            DiariZenDownloadStatusText = $"External DiariZen weights download failed: {ex.Message}";
        }
        finally
        {
            IsDownloadingDiariZen = false;
        }
    }

    [RelayCommand] private void CancelDiariZenDownload() => _diariZenDownloadCts?.Cancel();

    [RelayCommand]
    private async Task DownloadModels()
    {
        IsDownloading      = true;
        DownloadStatusText = Loc.Instance["download_starting"];
        DownloadPercent    = 0;
        _downloadCts       = new CancellationTokenSource();

        var progress = new Progress<DownloadProgress>(p =>
        {
            DownloadPercent    = p.OverallPercent;
            DownloadStatusText = Loc.Instance.T("download_progress", new()
            {
                ["current"] = (p.FileIndex + 1).ToString(),
                ["total"]   = p.TotalFiles.ToString(),
                ["file"]    = p.FileName,
                ["size"]    = string.IsNullOrEmpty(p.OverallSizeText)
                                  ? p.SizeText
                                  : $"{p.SizeText}  |  {p.OverallSizeText} total  ({p.OverallPercent:F1}%)",
            });
        });

        try
        {
            await _modelMgr.DownloadMissingModelsAsync(progress, _downloadCts.Token);
            DownloadStatusText = Loc.Instance["download_complete"];
            await CheckModelsAsync();
            AfterDownload?.Invoke();
        }
        catch (OperationCanceledException)
        {
            DownloadStatusText = Loc.Instance["download_cancelled"];
        }
        catch (Exception ex)
        {
            DownloadStatusText = Loc.Instance.T("download_failed", new() { ["error"] = ex.Message });
        }
        finally
        {
            IsDownloading = false;
        }
    }

    [RelayCommand] private void CancelDownload() => _downloadCts?.Cancel();

    [RelayCommand]
    internal async Task CheckForUpdatesAsync()
    {
        IsCheckingUpdates   = true;
        UpdateBannerVisible = false;

        var hashProgress = new Progress<(string fileName, int index, int total)>(p =>
        {
            UpdateBannerText    = Loc.Instance.T("settings_checking_hash",
                new() { ["file"] = p.fileName, ["i"] = (p.index + 1).ToString(), ["n"] = p.total.ToString() });
            UpdateBannerVisible = true;
        });

        var outdated = await _modelMgr.GetOutdatedFilesAsync(hashProgress);

        if (outdated == null)
        {
            // Network unavailable — hide any in-progress message
            UpdateBannerVisible = false;
        }
        else if (outdated.Count > 0)
        {
            _outdatedFiles    = outdated;
            HasOutdatedFiles  = true;
            ApplyUpdateBannerText();
            UpdateBannerVisible = true;
            OnUpdateAvailable?.Invoke();
        }
        else
        {
            _outdatedFiles      = [];
            HasOutdatedFiles    = false;
            UpdateBannerText    = Loc.Instance["settings_models_up_to_date"];
            UpdateBannerVisible = true;
            OnUpdateCheckComplete?.Invoke();
        }

        IsCheckingUpdates = false;
    }

    [RelayCommand]
    private void DismissUpdateBanner()
    {
        UpdateBannerVisible = false;
        HasOutdatedFiles    = false;
        _outdatedFiles      = [];
    }

    [RelayCommand]
    private async Task UpdateModels()
    {
        UpdateBannerVisible = false;
        await Task.Run(() => _modelMgr.PrepareRedownload(_outdatedFiles));
        _outdatedFiles = [];
        await CheckModelsAsync();
        await DownloadModels();
    }

    // ── Window initialisation ────────────────────────────────────────────────

    /// <summary>
    /// Run when the Settings window is first shown: check model files and hardware
    /// in parallel, then trigger an update check if models are present.
    /// </summary>
    public async Task InitializeAsync()
    {
        var modelTask    = CheckModelsAsync();
        var diarizenTask = CheckDiariZenModelsAsync();
        var hardwareTask = RecheckHardwareAsync();
        await Task.WhenAll(modelTask, diarizenTask, hardwareTask);

        if (ModelsReady)
            _ = CheckForUpdatesAsync();   // non-blocking; skipped if offline
    }
}
