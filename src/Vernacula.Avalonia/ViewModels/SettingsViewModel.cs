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
    [NotifyPropertyChangedFor(nameof(IsAsrParakeet), nameof(IsAsrCohere), nameof(IsAsrQwen3Asr), nameof(IsAsrVibeVoice), nameof(IsAsrVibeVoiceStreaming), nameof(IsAsrIndicConformer), nameof(IsAsrWhisperTurbo), nameof(IsAsrGraniteSpeech), nameof(ShowStandardSegmentationOptions), nameof(ShowVibeVoiceBuiltinSegmentation), nameof(ShowDiariZenInSegmentation), nameof(ShowGatedSegmentationHint), nameof(CanUseVibeVoiceAsr), nameof(VibeVoiceAsrLabel), nameof(VibeVoiceAsrDescription), nameof(CanUseVibeVoiceStreamingAsr), nameof(VibeVoiceStreamingOnCpu), nameof(VibeVoiceStreamingAsrLabel), nameof(VibeVoiceStreamingAsrDescription), nameof(ShowVibeVoiceStreamingSizePicker), nameof(ShowCohereLanguagePicker), nameof(ShowQwen3AsrLanguagePicker), nameof(ShowIndicConformerLanguagePicker), nameof(ShowWhisperTurboLanguagePicker))]
    private AsrBackend _selectedAsrBackend;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsVibeVoiceStreamingSmall), nameof(IsVibeVoiceStreamingLarge),
                              nameof(VibeVoiceStreamingSizeWarning), nameof(ShowVibeVoiceStreamingSizeWarning))]
    private VibeVoiceStreamingSize _selectedVibeVoiceStreamingSize;

    [ObservableProperty]
    private string _vibeVoiceStreamingHotwords = "";

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
    private AsrLanguageOption? _selectedIndicConformerLanguage;

    [ObservableProperty]
    private AsrLanguageOption? _selectedWhisperTurboLanguage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsEditorSingle), nameof(IsEditorAutoAdvance), nameof(IsEditorContinuous))]
    private PlaybackMode _selectedEditorPlaybackMode;

    [ObservableProperty] private string             _selectedLanguage;
    [ObservableProperty] private Loc.LanguageInfo?  _selectedLanguageInfo;

    /// <summary>Called after a successful download so Home can refresh its model status.</summary>
    public Action? AfterDownload       { get; set; }
    /// <summary>Called after segmentation mode or DiariZen status changes.</summary>
    public Action? OnSegmentationChanged { get; set; }
    /// <summary>
    /// Called after the user changes the active ASR backend so other panels
    /// (Home's missing-models banner, Results' backend-drift banner) can
    /// refresh live rather than waiting for the next app launch.
    /// </summary>
    public Action? OnAsrBackendChanged { get; set; }
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
    public bool IsAsrVibeVoiceStreaming => SelectedAsrBackend == AsrBackend.VibeVoiceStreaming;
    public bool IsAsrIndicConformer => SelectedAsrBackend == AsrBackend.IndicConformer;
    public bool IsAsrWhisperTurbo   => SelectedAsrBackend == AsrBackend.WhisperTurbo;
    public bool IsAsrGraniteSpeech  => SelectedAsrBackend == AsrBackend.GraniteSpeech;
    public bool CanUseVibeVoiceAsr  => CudaEpWorking;
    /// <summary>
    /// Always selectable. The backend takes its KV cache from host memory when the session is
    /// not on CUDA, so a machine without a working CUDA provider transcribes correctly — it is
    /// simply slow. Measured at RTF 12.2 on a 16-thread CPU against 0.20 on an RTX 3090, with
    /// byte-identical text and speaker labels (Run 35). Gating the option off meant refusing a
    /// working configuration; the description says what it costs instead.
    /// </summary>
    public bool CanUseVibeVoiceStreamingAsr => true;

    /// <summary>Whether this machine will run the streaming backend on the GPU or the CPU.</summary>
    public bool VibeVoiceStreamingOnCpu => !CudaEpWorking;

    public bool IsVibeVoiceStreamingSmall => SelectedVibeVoiceStreamingSize == VibeVoiceStreamingSize.Small1_5B;
    public bool IsVibeVoiceStreamingLarge => SelectedVibeVoiceStreamingSize == VibeVoiceStreamingSize.Large7B;

    /// <summary>
    /// The shape of each published package, as the settings window has to know it before
    /// anything is downloaded: bytes of weights on disk, the decoder's hidden size, the KV
    /// geometry, and the export's context ceiling in positions. The memory arithmetic itself
    /// lives in <see cref="VibeVoiceStreamingBudget"/>, which is what the backend uses on the
    /// package it has actually loaded — one model, two callers.
    /// </summary>
    internal static (long WeightBytes, int HiddenSize, int Layers, int KvHeads, int HeadDim, int Ceiling)
        VibeVoiceStreamingShape(VibeVoiceStreamingSize size) =>
            size == VibeVoiceStreamingSize.Large7B
                ? (9_700_788_736L, 3584, 28, 4, 128, 131072)
                : (3_421_310_464L, 1536, 28, 2, 128,  65536);

    /// <summary>
    /// Says what this card can actually do with the selected checkpoint. The cache is sized to
    /// the recording rather than to the export ceiling (issue #150), so the honest answer is
    /// not "fits / does not fit" but the length that fits: a 16 GB card runs the 7B for about
    /// an hour of audio and cannot run it for the full two.
    ///
    /// Reads FREE memory, not total. The runtime decides on what is free when the job starts,
    /// and a card is never entirely yours — the desktop alone is most of a gigabyte. Quoting
    /// the total overstated exactly the small-card case this warning exists for.
    /// </summary>
    public string VibeVoiceStreamingSizeWarning
    {
        get
        {
            var (totalMb, freeMb) = HardwareInfo.GetGpuMemoryMb();
            if (totalMb <= 0 || freeMb <= 0) return "";
            return VibeVoiceStreamingWarningFor(SelectedVibeVoiceStreamingSize, freeMb * 1024L * 1024L);
        }
    }

    /// <summary>
    /// The warning itself, given a checkpoint and the free memory to spend on it. Separated from
    /// the NVML read so it can be tested: the bug this replaced was in the threshold, not the
    /// arithmetic, and a threshold is only reachable with a particular card in front of you.
    /// </summary>
    internal static string VibeVoiceStreamingWarningFor(VibeVoiceStreamingSize size, long freeBytes)
    {
        var shape = VibeVoiceStreamingShape(size);
        double fits = VibeVoiceStreamingBudget.MinutesThatFit(
            freeBytes, shape.WeightBytes, shape.HiddenSize,
            VibeVoiceStreamingBudget.KvBytesPerPosition(shape.Layers, shape.KvHeads, shape.HeadDim),
            shape.Ceiling);

        double freeGiB = freeBytes / (double)(1L << 30);
        if (fits <= 0)
        {
            double fixedGiB = (shape.WeightBytes + VibeVoiceStreamingBudget.WorkingSetBytes(shape.HiddenSize))
                            / (double)(1L << 30);
            return $"This card has {freeGiB:F1} GB of VRAM free; this checkpoint needs about "
                 + $"{fixedGiB:F1} GB before any audio is cached.";
        }

        // The ceiling the package itself carries, not a rounded figure: quoting "2 hours" here
        // would call a card short at 130 minutes and stay silent at 125.
        double capMinutes = VibeVoiceStreamingBudget.CeilingMinutes(shape.Ceiling);
        return fits >= capMinutes
            ? ""
            : $"This card has {freeGiB:F1} GB of VRAM free, which fits about {fits:F0} minutes "
            + $"of audio with this checkpoint rather than the full {capMinutes:F0}.";
    }

    public bool ShowVibeVoiceStreamingSizeWarning => VibeVoiceStreamingSizeWarning.Length > 0;
    public string VibeVoiceStreamingAsrLabel => VibeVoiceStreamingOnCpu
        ? "VibeVoice-ASR Streaming (CPU - very slow)"
        : "VibeVoice-ASR Streaming";
    public string VibeVoiceStreamingAsrDescription => VibeVoiceStreamingOnCpu
        ? "Chunked ASR with built-in speaker attribution, in 10 languages. No CUDA provider was found, so this runs on the CPU: the transcript is the same but decoding takes roughly 12x the length of the recording, so a 10-minute file is about two hours. Use the 1.5B, and prefer another backend for anything long."
        : "Chunked ASR with built-in speaker attribution, in 10 languages. Text appears as the recording is decoded rather than after it finishes. Recording length is capped by the checkpoint's context: about 68 minutes (1.5B) or 2 hours (7B).";
    public string VibeVoiceAsrLabel => CanUseVibeVoiceAsr ? "VibeVoice-ASR" : "VibeVoice-ASR (Unavailable - CUDA Missing)";
    public string VibeVoiceAsrDescription => CanUseVibeVoiceAsr
        ? "Whole-recording ASR with built-in diarization. Downloads into the vibevoice_asr models folder."
        : "Unavailable because the CUDA execution provider check did not pass.";
    public bool ShowVibeVoiceStreamingSizePicker => SelectedAsrBackend == AsrBackend.VibeVoiceStreaming;
    public bool ShowCohereLanguagePicker         => SelectedAsrBackend == AsrBackend.Cohere;
    public bool ShowQwen3AsrLanguagePicker       => SelectedAsrBackend == AsrBackend.Qwen3Asr;
    public bool ShowIndicConformerLanguagePicker => SelectedAsrBackend == AsrBackend.IndicConformer;
    public bool ShowWhisperTurboLanguagePicker   => SelectedAsrBackend == AsrBackend.WhisperTurbo;
    // Both VibeVoice checkpoints segment and attribute speakers themselves, so neither
    // offers the standard segmentation choices.
    public bool ShowStandardSegmentationOptions =>
        SelectedAsrBackend is not (AsrBackend.VibeVoice or AsrBackend.VibeVoiceStreaming);
    public bool ShowVibeVoiceBuiltinSegmentation =>
        SelectedAsrBackend is AsrBackend.VibeVoice or AsrBackend.VibeVoiceStreaming;
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

    // Whisper large-v3-turbo — 99 languages. Built from
    // AsrLanguageSupport.WhisperTurboLangs so any future change to the
    // language set flows through one source of truth. Includes "Auto-detect"
    // as the first entry — Whisper's decoder emits the detected <|lang|>
    // token as its first output when no language prefix is forced.
    public static readonly IReadOnlyList<AsrLanguageOption> WhisperTurboLanguages =
        BuildWhisperTurboLanguages();

    private static IReadOnlyList<AsrLanguageOption> BuildWhisperTurboLanguages()
    {
        var list = new List<AsrLanguageOption> { new("", "Auto-detect") };
        list.AddRange(AsrLanguageSupport.LanguageOptions(AsrBackend.WhisperTurbo));
        return list;
    }

    // AI4Bharat IndicConformer 600M — the 22 official Indian languages.
    // Unlike Cohere/Qwen3 there is no "auto-detect" option: the model has
    // 22 distinct per-language CTC heads and picking one is required at
    // inference. Five codes use ISO 639-3 because they have no 639-1
    // assignment (brx, doi, kok, mai, mni, sat); seventeen use 639-1.
    public static readonly IReadOnlyList<AsrLanguageOption> IndicConformerLanguages =
    [
        new("as",  "Assamese"),
        new("bn",  "Bengali"),
        new("brx", "Bodo"),
        new("doi", "Dogri"),
        new("gu",  "Gujarati"),
        new("hi",  "Hindi"),
        new("kn",  "Kannada"),
        new("kok", "Konkani"),
        new("ks",  "Kashmiri"),
        new("mai", "Maithili"),
        new("ml",  "Malayalam"),
        new("mni", "Manipuri"),
        new("mr",  "Marathi"),
        new("ne",  "Nepali"),
        new("or",  "Odia"),
        new("pa",  "Punjabi"),
        new("sa",  "Sanskrit"),
        new("sat", "Santali"),
        new("sd",  "Sindhi"),
        new("ta",  "Tamil"),
        new("te",  "Telugu"),
        new("ur",  "Urdu"),
    ];

    // ── Hardware check state ─────────────────────────────────────────────────

    [ObservableProperty] private bool   _gpuDetected          = false;
    [ObservableProperty] private string _gpuVramText          = "";
    /// <summary>
    /// Names the card in the hardware panel. Every CUDA path here is pinned to device 0, so on a
    /// machine with two GPUs "NVIDIA GPU ✓" does not answer the question the user is asking
    /// (issue #149) -- which of them is doing the work.
    /// </summary>
    [ObservableProperty] private string _gpuNameText          = "NVIDIA GPU";
    [ObservableProperty] private bool   _cudaToolkitInstalled = false;
    [ObservableProperty] private bool   _cudnnInstalled       = false;
    [ObservableProperty] private bool   _cudaEpWorking        = false;

    /// <summary>
    /// Why CUDA is unavailable, when the probe worked it out — a CUDA of the wrong major, a library
    /// present but off the loader path, no cuDNN at all.
    ///
    /// ⚠ DELIBERATELY NOT LOCALISED. It names file paths and shell commands (`ldconfig`,
    /// `/etc/ld.so.conf.d`), which do not translate, and it is the only thing standing between a
    /// user whose GPU went quiet and a working install.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasCudaProbeNote))]
    private string _cudaProbeNote = "";

    public bool HasCudaProbeNote => !string.IsNullOrEmpty(CudaProbeNote);

    /// <summary>
    /// Whether to offer the download links. ⚠ NOT SIMPLY THE NEGATION OF "installed". A CUDA that
    /// is present but unreachable — the wrong major, or outside the loader path — needs an ldconfig
    /// line or a different version, not another download; offering one beside a note saying the
    /// library was found contradicts the note.
    /// </summary>
    [ObservableProperty] private bool _offerCudaDownload = true;
    [ObservableProperty] private bool _offerCudnnDownload = true;
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
        _selectedVibeVoiceStreamingSize = svc.Current.VibeVoiceStreamingSize;
        _vibeVoiceStreamingHotwords     = svc.Current.VibeVoiceStreamingHotwords;
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
        _selectedIndicConformerLanguage = IndicConformerLanguages.FirstOrDefault(l => l.Code == svc.Current.IndicConformerLanguage)
                                        ?? IndicConformerLanguages.FirstOrDefault(l => l.Code == "hi")
                                        ?? IndicConformerLanguages[0];
        _selectedWhisperTurboLanguage = WhisperTurboLanguages.FirstOrDefault(l => l.Code == svc.Current.WhisperTurboLanguage)
                                        ?? WhisperTurboLanguages[0];
        _parakeetBeamWidth            = Math.Max(1, svc.Current.ParakeetBeamWidth);
        _selectedLmOption             = KenLmCatalog.Find(svc.Current.ParakeetLmSelection) ?? KenLmCatalog.All[0];
        // If the user's saved selection is a built-in option whose file isn't
        // on disk (fresh install, interrupted download, etc.) kick off the
        // download now. Without this the dropdown shows the right label but
        // the decoder silently runs without fusion.
        if (_selectedLmOption.RemoteFileName is not null && !modelMgr.IsKenLmReady(_selectedLmOption))
            _ = DownloadSelectedLmAsync();
        _parakeetLmPath               = svc.Current.ParakeetLmPath ?? "";
        _parakeetLmWeight             = svc.Current.ParakeetLmWeight;
        _parakeetLmLengthPenalty      = svc.Current.ParakeetLmLengthPenalty;
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

        InitTtsSettings();

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

    partial void OnVibeVoiceStreamingHotwordsChanged(string value)
    {
        _svc.Current.VibeVoiceStreamingHotwords = value ?? "";
        _svc.Save();
    }

    partial void OnSelectedVibeVoiceStreamingSizeChanged(VibeVoiceStreamingSize value)
    {
        _svc.Current.VibeVoiceStreamingSize = value;
        _svc.Save();
        // The two sizes live in different folders, so the missing-file set changes with them.
        _ = CheckModelsAsync();
    }

    partial void OnSelectedAsrBackendChanged(AsrBackend value)
    {
        if ((value == AsrBackend.VibeVoice && !CanUseVibeVoiceAsr)
            || (value == AsrBackend.VibeVoiceStreaming && !CanUseVibeVoiceStreamingAsr))
        {
            SelectedAsrBackend = AsrBackend.Parakeet;
            return;
        }

        if (value is AsrBackend.VibeVoice or AsrBackend.VibeVoiceStreaming
            && SelectedSegmentation != SegmentationMode.VibeVoiceBuiltin)
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
        OnAsrBackendChanged?.Invoke();
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

    partial void OnSelectedWhisperTurboLanguageChanged(AsrLanguageOption? value)
    {
        _svc.Current.WhisperTurboLanguage = value?.Code ?? "";
        _svc.Save();
    }

    partial void OnSelectedIndicConformerLanguageChanged(AsrLanguageOption? value)
    {
        // No "auto" entry for IndicConformer — if someone binds null through a
        // cleared combo, fall back to the last good code rather than persist "".
        _svc.Current.IndicConformerLanguage = value?.Code ?? _svc.Current.IndicConformerLanguage;
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
    [RelayCommand] private void SetVibeVoiceStreamingSize(string n) { if (Enum.TryParse<VibeVoiceStreamingSize>(n, out var v)) SelectedVibeVoiceStreamingSize = v; }
    [RelayCommand] private void SetEditorPlaybackMode(string n) { if (Enum.TryParse<PlaybackMode>(n,     out var m)) SelectedEditorPlaybackMode = m; }
    [RelayCommand] private void SetLanguage(string l)           => SelectedLanguage = l;

    // ── Hardware checks ──────────────────────────────────────────────────────

    private static string? FirstLine(string? text) =>
        text is null ? null : text.Split('\n', 2)[0].Trim();

    /// <summary>
    /// Called when a hardware re-check changes which model bundle is active, so other views can
    /// refresh what they latched. Set by MainViewModel, like the rest of the cross-view wiring.
    /// </summary>
    internal Func<Task>? ModelSelectionChanged { get; set; }

    /// <summary>The Re-check button: discards the cached probe and looks again.</summary>
    [RelayCommand]
    internal Task RecheckHardwareAsync() => RefreshHardwareAsync(force: true);

    /// <param name="force">True only when the user asked. Opening the window reads what is already
    /// known; throwing the probe away there would make every model init afterwards repeat the
    /// scan.</param>
    private async Task RefreshHardwareAsync(bool force)
    {
        IsCheckingHardware = true;
        try
        {
            bool cudaOk = false;
            string? cudaMessage = null;
            bool bf16Before = false, bf16After = false;

            // Collected off the UI thread, applied on it: these are bound properties, and raising
            // PropertyChanged from a thread-pool thread updates Avalonia bindings off the UI thread.
            long totalMb = 0;
            string gpuName = "";
            int gpuCount = 0;
            bool cudaToolkit = false, cudnnFound = false, cudaPresent = false, cudnnPresent = false;

            await Task.Run(() =>
            {
                // ⚠ THE HARDWARE WORK BELONGS HERE, AND IN THIS ORDER. After an invalidation the
                // caches are cold by construction, so every read is a full probe -- on Windows a
                // recursive walk of the CUDA, cuDNN and PATH roots, plus an NVML cycle. Reading
                // before invalidating would report the answers this refresh was meant to replace.
                //
                // ⚠ SAMPLED, NOT SKIPPED. An earlier version only compared when something had
                // already asked for this, to avoid forcing a cold probe -- but the model verdicts on
                // this window come from ActiveRepos, which can be computed without ever forcing it
                // (a non-Granite backend, or a BF16 directory that is not downloaded). Skipping the
                // comparison then left those verdicts stale after a re-check that changed the
                // answer. One probe is cheaper than that.
                bf16Before = HardwareInfo.SupportsBf16Acceleration();
                if (force) HardwareInfo.InvalidateCudaProbes();

                (totalMb, _) = HardwareInfo.GetGpuMemoryMb();
                gpuName = HardwareInfo.GetGpuName();
                gpuCount = HardwareInfo.GetGpuCount();
                cudaToolkit = HardwareInfo.IsCudaToolkitInstalled();
                cudnnFound = HardwareInfo.IsCudnnInstalled();
                cudaPresent = HardwareInfo.IsCudaRuntimePresent;
                cudnnPresent = HardwareInfo.IsCudnnPresent;

                // Re-register from the refreshed probe even when the check then fails: CheckCuda
                // only reaches AddCudaToSearchPath on its success path, so an unavailable result
                // would otherwise leave the loader holding directories from a discarded answer.
                if (force && OperatingSystem.IsWindows()) ModelManagerService.AddCudaToSearchPath();

                var cudaCheck = _modelMgr.CheckCuda();
                cudaOk = cudaCheck.Available;
                // Only when the check actually ran: otherwise the message is about something else --
                // on first launch, a model file that has not been downloaded yet -- and showing it
                // here would blame CUDA for it.
                cudaMessage = cudaCheck.Ran ? cudaCheck.Message : null;
                bf16After = HardwareInfo.SupportsBf16Acceleration();
            });

            GpuDetected          = totalMb > 0;
            GpuNameText          = gpuName.Length == 0 ? "NVIDIA GPU"
                                 : gpuCount > 1        ? $"{gpuName} (GPU 0 of {gpuCount})"
                                 : gpuName;
            GpuVramText          = totalMb > 0
                ? Loc.Instance.T("settings_hw_vram", new() { ["vram"] = $"{totalMb / 1024.0:F1}" })
                : "";
            CudaToolkitInstalled = cudaToolkit;
            CudnnInstalled       = cudnnFound;
            OfferCudaDownload    = !cudaToolkit && !cudaPresent;
            OfferCudnnDownload   = !cudnnFound && !cudnnPresent;
            CudaEpWorking = cudaOk;
            // The probe's note when it has one; otherwise whatever CheckCuda said. The provider can
            // fail AFTER the probe passes -- a swallowed load failure is exactly the case worth
            // explaining -- and that message was being thrown away.
            // ⚠ FIRST LINE ONLY. CheckCuda's failure message is a full exception dump -- message,
            // inner exception and stack trace -- written to cuda_debug.txt on purpose. Rendering all of
            // it into a 12px label is not a diagnostic, it is a wall.
            CudaProbeNote = cudaOk
                ? ""
                : (HardwareInfo.CudaProbeNote ?? FirstLine(cudaMessage) ?? "");
            OnPropertyChanged(nameof(CanUseVibeVoiceAsr));
            OnPropertyChanged(nameof(CanUseVibeVoiceStreamingAsr));
            OnPropertyChanged(nameof(VibeVoiceStreamingOnCpu));
            OnPropertyChanged(nameof(VibeVoiceStreamingAsrLabel));
            OnPropertyChanged(nameof(VibeVoiceStreamingAsrDescription));
            OnPropertyChanged(nameof(VibeVoiceAsrLabel));
            OnPropertyChanged(nameof(VibeVoiceAsrDescription));

            // ⚠ AND WHAT DEPENDS ON IT. A forced refresh can flip SupportsBf16Acceleration, which
            // decides which Granite bundle is active, so the model verdicts on this window were
            // computed against an answer that may no longer hold. Re-checking hardware without
            // re-checking models leaves the two disagreeing.
            if (force && bf16Before != bf16After)
            {
                await CheckModelsAsync();
                await CheckDiariZenModelsAsync();
                await CheckVoxLinguaModelsAsync();
                // And the views that latched the same answer: the home screen's model status is
                // computed from the active bundle, and would otherwise disagree with this window
                // until the app restarted.
                if (ModelSelectionChanged is not null) await ModelSelectionChanged();
            }

            if (!CudaEpWorking && SelectedAsrBackend is AsrBackend.VibeVoice or AsrBackend.VibeVoiceStreaming)
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

        }
        finally
        {
            // ⚠ finally: the model checks below can throw on IO, and a stuck flag hides
            // the hardware rows behind "Checking…" and disables Re-check for the rest of
            // the session.
            IsCheckingHardware = false;
        }
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

            if (SelectedAsrBackend == AsrBackend.VibeVoiceStreaming)
            {
                ModelStatusText = $"Missing {_lastMissing.Count} required model file(s): {string.Join(", ", _lastMissing)}. " +
                                  $"Use Download Missing Models, or place VibeVoice-ASR Streaming weights under {_svc.GetVibeVoiceStreamingModelsDir()}.";
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
        if (backend is AsrBackend.VibeVoice or AsrBackend.VibeVoiceStreaming)
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
        var ttsTask      = CheckTtsModelsAsync();
        var hardwareTask = RefreshHardwareAsync(force: false);
        await Task.WhenAll(modelTask, diarizenTask, voxLinguaTask, ttsTask, hardwareTask);

        if (ModelsReady)
            _ = CheckForUpdatesAsync();   // non-blocking; skipped if offline
    }
}
