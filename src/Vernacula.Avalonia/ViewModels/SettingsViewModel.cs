using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Platform.Storage;
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
    [NotifyPropertyChangedFor(nameof(IsAsrParakeet), nameof(IsAsrCohere), nameof(IsAsrQwen3Asr), nameof(IsAsrVibeVoice), nameof(ShowStandardSegmentationOptions), nameof(ShowVibeVoiceBuiltinSegmentation), nameof(ShowDiariZenInSegmentation), nameof(ShowGatedSegmentationHint), nameof(CanUseVibeVoiceAsr), nameof(VibeVoiceAsrLabel), nameof(VibeVoiceAsrDescription), nameof(ShowCohereLanguagePicker), nameof(ShowQwen3AsrLanguagePicker))]
    private AsrBackend _selectedAsrBackend;

    [ObservableProperty]
    private int _parakeetBeamWidth;

    // Selected KenLM option (from KenLmCatalog.All). Changing this triggers
    // a lazy download when a built-in option isn't yet on disk.
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowCustomLmPath), nameof(IsLmDownloading), nameof(LmStatusText))]
    private KenLmOption? _selectedLmOption;

    // Free-form path used only when SelectedLmOption is the "custom" entry.
    [ObservableProperty]
    private string _parakeetLmPath = "";

    [ObservableProperty]
    private float _parakeetLmWeight;

    [ObservableProperty]
    private float _parakeetLmLengthPenalty;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLmDownloading), nameof(LmStatusText))]
    private bool _isLmDownloadingInternal;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LmStatusText))]
    private double _lmDownloadPercent;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LmStatusText))]
    private string _lmDownloadStatusText = "";

    private CancellationTokenSource? _lmDownloadCts;

    public IReadOnlyList<KenLmOption> AvailableLmOptions => KenLmCatalog.All;

    public bool ShowCustomLmPath  => SelectedLmOption?.Key == KenLmCatalog.KeyCustom;
    public bool IsLmDownloading   => IsLmDownloadingInternal;

    public string LmStatusText
    {
        get
        {
            if (IsLmDownloading) return LmDownloadStatusText;
            var opt = SelectedLmOption;
            if (opt is null) return "";
            if (opt.Key == KenLmCatalog.KeyNone) return "Fusion disabled — decoder uses greedy or beam only.";
            if (opt.Key == KenLmCatalog.KeyCustom)
                return string.IsNullOrWhiteSpace(ParakeetLmPath)
                    ? "Custom — pick an ARPA file to enable fusion."
                    : File.Exists(ParakeetLmPath)
                        ? "Custom ARPA ready."
                        : "Custom ARPA path doesn't exist.";
            return _modelMgr.IsKenLmReady(opt)
                ? "Downloaded and ready."
                : $"Not yet downloaded (~{(opt.ExpectedSizeBytes ?? 0) / 1_000_000} MB). Will fetch on save.";
        }
    }

    [ObservableProperty]
    private AsrLanguageOption? _selectedCohereLanguage;

    [ObservableProperty]
    private AsrLanguageOption? _selectedQwen3AsrLanguage;

    [ObservableProperty]
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
    public bool ShowCohereLanguagePicker   => SelectedAsrBackend == AsrBackend.Cohere;
    public bool ShowQwen3AsrLanguagePicker => SelectedAsrBackend == AsrBackend.Qwen3Asr;
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

    public static readonly IReadOnlyList<AsrLanguageOption> CohereLanguages =
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

    public static readonly IReadOnlyList<AsrLanguageOption> Qwen3AsrLanguages =
    [
        new("",   "Auto-detect"),
        new("ar", "Arabic"),
        new("zh", "Chinese"),
        new("cs", "Czech"),
        new("da", "Danish"),
        new("nl", "Dutch"),
        new("en", "English"),
        new("tl", "Filipino"),
        new("fi", "Finnish"),
        new("fr", "French"),
        new("de", "German"),
        new("el", "Greek"),
        new("hi", "Hindi"),
        new("hu", "Hungarian"),
        new("id", "Indonesian"),
        new("it", "Italian"),
        new("ja", "Japanese"),
        new("ko", "Korean"),
        new("mk", "Macedonian"),
        new("ms", "Malay"),
        new("fa", "Persian"),
        new("pl", "Polish"),
        new("pt", "Portuguese"),
        new("ro", "Romanian"),
        new("ru", "Russian"),
        new("es", "Spanish"),
        new("sv", "Swedish"),
        new("th", "Thai"),
        new("tr", "Turkish"),
        new("vi", "Vietnamese"),
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

    // ── Language identification (VoxLingua107) ───────────────────────────
    [ObservableProperty] private bool   _lidEnabled;
    [ObservableProperty] private bool   _lidPerSegment;
    [ObservableProperty] private string _voxLinguaStatusText = "";
    [ObservableProperty] private IBrush _voxLinguaStatusBrush = Brushes.Gray;
    [ObservableProperty] private string _voxLinguaModelsLocationText = "";
    [ObservableProperty] private bool   _voxLinguaReady = false;
    [ObservableProperty] private bool   _isDownloadingVoxLingua = false;
    [ObservableProperty] private double _voxLinguaDownloadPercent = 0;
    [ObservableProperty] private string _voxLinguaDownloadStatusText = "";
    private CancellationTokenSource?    _voxLinguaDownloadCts;
    private IReadOnlyList<string>       _lastVoxLinguaMissing = [];

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
        _selectedQwen3AsrLanguage     = Qwen3AsrLanguages.FirstOrDefault(l => l.Code == svc.Current.Qwen3AsrLanguage)
                                        ?? Qwen3AsrLanguages[0];
        _parakeetBeamWidth            = Math.Max(1, svc.Current.ParakeetBeamWidth);
        _selectedLmOption             = KenLmCatalog.Find(svc.Current.ParakeetLmSelection) ?? KenLmCatalog.All[0];
        _parakeetLmPath               = svc.Current.ParakeetLmPath ?? "";
        _parakeetLmWeight             = svc.Current.ParakeetLmWeight;
        _parakeetLmLengthPenalty      = svc.Current.ParakeetLmLengthPenalty;
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

        _lidEnabled        = svc.Current.LidEnabled;
        _lidPerSegment     = svc.Current.LidPerSegment;
        VoxLinguaStatusText = "Checking VoxLingua107 weights…";
        VoxLinguaModelsLocationText = _svc.GetVoxLinguaModelsDir();

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

    partial void OnSelectedCohereLanguageChanged(AsrLanguageOption? value)
    {
        _svc.Current.CohereLanguage = value?.Code ?? "";
        _svc.Save();
    }

    partial void OnSelectedQwen3AsrLanguageChanged(AsrLanguageOption? value)
    {
        _svc.Current.Qwen3AsrLanguage = value?.Code ?? "";
        _svc.Save();
    }

    partial void OnSelectedDenoiserChanged(DenoiserMode value)
    {
        _svc.Current.Denoiser = value;
        _svc.Save();
    }

    partial void OnParakeetBeamWidthChanged(int value)
    {
        int clamped = Math.Clamp(value, 1, 16);
        if (clamped != value)
        {
            ParakeetBeamWidth = clamped;
            return;
        }
        _svc.Current.ParakeetBeamWidth = clamped;
        _svc.Save();
    }

    partial void OnParakeetLmPathChanged(string value)
    {
        _svc.Current.ParakeetLmPath = value ?? "";
        _svc.Save();
        OnPropertyChanged(nameof(LmStatusText));
    }

    partial void OnSelectedLmOptionChanged(KenLmOption? value)
    {
        if (value is null) return;
        _svc.Current.ParakeetLmSelection = value.Key;
        _svc.Save();
        OnPropertyChanged(nameof(LmStatusText));

        // Auto-download built-in options on selection, unless already present.
        if (value.RemoteFileName is not null && !_modelMgr.IsKenLmReady(value))
            _ = DownloadSelectedLmAsync();
    }

    [RelayCommand]
    private async Task DownloadSelectedLmAsync()
    {
        var opt = SelectedLmOption;
        if (opt is null || opt.RemoteFileName is null) return;
        if (_modelMgr.IsKenLmReady(opt))
        {
            LmDownloadStatusText = "Already downloaded.";
            return;
        }

        IsLmDownloadingInternal = true;
        LmDownloadPercent       = 0;
        LmDownloadStatusText    = $"Downloading {opt.DisplayName}…";
        _lmDownloadCts          = new CancellationTokenSource();

        var progress = new Progress<DownloadProgress>(p =>
        {
            LmDownloadPercent    = p.OverallPercent;
            LmDownloadStatusText =
                $"{opt.DisplayName} — {p.SizeText} ({p.OverallPercent:F1}%)";
        });

        try
        {
            await _modelMgr.DownloadKenLmAsync(opt, progress, _lmDownloadCts.Token);
            LmDownloadStatusText = $"{opt.DisplayName} ready.";
        }
        catch (OperationCanceledException)
        {
            LmDownloadStatusText = $"{opt.DisplayName} download cancelled.";
        }
        catch (Exception ex)
        {
            LmDownloadStatusText = $"{opt.DisplayName} download failed: {ex.Message}";
        }
        finally
        {
            IsLmDownloadingInternal = false;
            OnPropertyChanged(nameof(LmStatusText));
        }
    }

    [RelayCommand]
    private void CancelLmDownload() => _lmDownloadCts?.Cancel();

    partial void OnParakeetLmWeightChanged(float value)
    {
        float clamped = Math.Clamp(value, 0f, 2f);
        if (Math.Abs(clamped - value) > 1e-6f)
        {
            ParakeetLmWeight = clamped;
            return;
        }
        _svc.Current.ParakeetLmWeight = clamped;
        _svc.Save();
    }

    partial void OnParakeetLmLengthPenaltyChanged(float value)
    {
        float clamped = Math.Clamp(value, 0f, 2f);
        if (Math.Abs(clamped - value) > 1e-6f)
        {
            ParakeetLmLengthPenalty = clamped;
            return;
        }
        _svc.Current.ParakeetLmLengthPenalty = clamped;
        _svc.Save();
    }

    [RelayCommand]
    private async Task BrowseParakeetLm()
    {
        if (Application.Current?.ApplicationLifetime is not
            Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime desktop)
            return;
        var window = desktop.MainWindow;
        if (window is null) return;

        var picker = await window.StorageProvider.OpenFilePickerAsync(new()
        {
            Title = "Select KenLM ARPA file",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("KenLM ARPA (.arpa, .arpa.gz)")
                {
                    Patterns = new[] { "*.arpa", "*.arpa.gz" },
                },
                new FilePickerFileType("All files") { Patterns = new[] { "*.*" } },
            },
        });
        if (picker.Count == 0) return;
        var p = picker[0].TryGetLocalPath();
        if (!string.IsNullOrEmpty(p))
            ParakeetLmPath = p;
    }

    [RelayCommand]
    private void ClearParakeetLm() => ParakeetLmPath = "";

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

    // ── VoxLingua107 (language ID) ───────────────────────────────────────

    partial void OnLidEnabledChanged(bool value)
    {
        _svc.Current.LidEnabled = value;
        _svc.Save();
    }

    partial void OnLidPerSegmentChanged(bool value)
    {
        _svc.Current.LidPerSegment = value;
        _svc.Save();
    }

    private void ApplyVoxLinguaStatusText()
    {
        VoxLinguaModelsLocationText = _svc.GetVoxLinguaModelsDir();
        VoxLinguaStatusText = _lastVoxLinguaMissing.Count == 0
            ? "VoxLingua107 language-ID weights found and ready."
            : $"VoxLingua107 weights incomplete ({_lastVoxLinguaMissing.Count} file(s) missing): "
              + string.Join(", ", _lastVoxLinguaMissing);

        VoxLinguaStatusBrush = Application.Current!.Resources[
            _lastVoxLinguaMissing.Count == 0 ? "GreenBrush" : "YellowBrush"] as IBrush
            ?? (_lastVoxLinguaMissing.Count == 0 ? Brushes.LimeGreen : Brushes.Goldenrod);
        VoxLinguaReady = _lastVoxLinguaMissing.Count == 0;
    }

    internal async Task CheckVoxLinguaModelsAsync()
    {
        VoxLinguaStatusText = "Checking VoxLingua107 weights…";
        VoxLinguaModelsLocationText = _svc.GetVoxLinguaModelsDir();

        IReadOnlyList<string> missing = [];
        await Task.Run(() => missing = _modelMgr.GetMissingVoxLinguaFiles(_svc.GetVoxLinguaModelsDir()));

        _lastVoxLinguaMissing = missing;
        ApplyVoxLinguaStatusText();
    }

    [RelayCommand]
    private async Task DownloadVoxLinguaModels()
    {
        IsDownloadingVoxLingua = true;
        VoxLinguaDownloadStatusText = "Starting VoxLingua107 weights download…";
        VoxLinguaDownloadPercent = 0;
        _voxLinguaDownloadCts = new CancellationTokenSource();

        var progress = new Progress<DownloadProgress>(p =>
        {
            VoxLinguaDownloadPercent = p.OverallPercent;
            VoxLinguaDownloadStatusText = $"[{p.FileIndex + 1}/{p.TotalFiles}] {p.FileName} — " +
                (string.IsNullOrEmpty(p.OverallSizeText)
                    ? p.SizeText
                    : $"{p.SizeText}  |  {p.OverallSizeText} total  ({p.OverallPercent:F1}%)");
        });

        try
        {
            await _modelMgr.DownloadMissingVoxLinguaModelsAsync(
                progress, _svc.GetVoxLinguaModelsDir(), _voxLinguaDownloadCts.Token);
            VoxLinguaDownloadStatusText = "VoxLingua107 weights download complete.";
            await CheckVoxLinguaModelsAsync();
        }
        catch (OperationCanceledException)
        {
            VoxLinguaDownloadStatusText = "VoxLingua107 download cancelled.";
        }
        catch (Exception ex)
        {
            VoxLinguaDownloadStatusText = $"VoxLingua107 download failed: {ex.Message}";
        }
        finally
        {
            IsDownloadingVoxLingua = false;
        }
    }

    [RelayCommand] private void CancelVoxLinguaDownload() => _voxLinguaDownloadCts?.Cancel();

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
        var voxLinguaTask = CheckVoxLinguaModelsAsync();
        var hardwareTask = RecheckHardwareAsync();
        await Task.WhenAll(modelTask, diarizenTask, voxLinguaTask, hardwareTask);

        if (ModelsReady)
            _ = CheckForUpdatesAsync();   // non-blocking; skipped if offline
    }
}
