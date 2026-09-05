using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Vernacula.Tts.App.Models;
using Vernacula.Tts.App.Services;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Vernacula.Tts.App.ViewModels;

/// <summary>
/// Reader VM. Holds picker state, persists it across runs via
/// <see cref="SettingsService"/>, drives streaming synthesis +
/// playback (chunks start playing as soon as the first one is ready),
/// and toggles between the flat word-by-word view and a Markdown
/// rendering of the source.
/// </summary>
public sealed partial class MainViewModel : ObservableObject, IDisposable
{
    // Pickers. NotifyCanExecuteChangedFor is what re-queries the
    // SynthesizeCommand's CanExecute when the picker fills.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _voicePath = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _onnxBundleDir = "";

    // ── TTS backend selection ────────────────────────────────────────
    // Which engine synthesizes. Switching invalidates the cached backend
    // and flips which config controls are relevant (IsChatterbox/IsKokoro).
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    [NotifyPropertyChangedFor(nameof(IsChatterbox))]
    [NotifyPropertyChangedFor(nameof(IsKokoro))]
    [NotifyPropertyChangedFor(nameof(IsOmniVoice))]
    [NotifyPropertyChangedFor(nameof(NeedsPhonemizer))]
    private TtsBackendKind _selectedBackend;

    public bool IsChatterbox => SelectedBackend == TtsBackendKind.Chatterbox;
    public bool IsKokoro => SelectedBackend == TtsBackendKind.Kokoro;
    public bool IsOmniVoice => SelectedBackend == TtsBackendKind.OmniVoice;
    /// <summary>Kokoro and OmniVoice both phonemize through vernacula-phonemizer.</summary>
    public bool NeedsPhonemizer => IsKokoro || IsOmniVoice;
    public IReadOnlyList<TtsBackendKind> Backends { get; } =
        new[] { TtsBackendKind.Chatterbox, TtsBackendKind.Kokoro, TtsBackendKind.OmniVoice };

    // vernacula-phonemizer data/ root — shared by the two phonemizing backends.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _phonemizerDataDir = "";

    // Kokoro config: model dir (kokoro.onnx + voices/), named voice, and speed.
    // Voices are discovered from the model dir.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _kokoroModelDir = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _kokoroVoice = "";

    [ObservableProperty] private float _kokoroSpeed = 1.0f;

    public ObservableCollection<string> KokoroVoices { get; } = new();

    // OmniVoice-IPA config: the ONNX dir (base transformer + codec + versioned diff), an
    // optional explicit tokenizer.json (auto-located otherwise), the stored-voice library,
    // the phonemizer language, the voice, and the diffusion step count.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _omniVoiceOnnxDir = "";

    // ⚠ This one gates the button too: the tokenizer is auto-located from the ONNX dir, so when it
    // is NOT found there it is the LAST thing the user picks -- and without this attribute the
    // button's CanExecute was never re-queried after that pick, so it stayed greyed out.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _omniVoiceTokenizerJson = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _omniVoiceVoiceLib = "";

    // The language: the persisted truth is the CODE (OmniVoiceLang). The picker is a type-ahead
    // over LanguageCatalog -- its text (OmniVoiceLangQuery) drives LanguageMatches, and choosing a
    // row sets OmniVoiceLanguage, which sets the code. Typing a code exactly ("cy") sets it too,
    // without a row having to be chosen.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _omniVoiceLang = "en";

    [ObservableProperty] private string _omniVoiceLangQuery = "";
    [ObservableProperty] private LanguageOption? _omniVoiceLanguage;

    // ⚠ WHOLE-LIST REPLACEMENT, NOT AN ObservableCollection THAT IS CLEARED IN PLACE, and the
    // replacement is POSTED rather than applied inline. Both halves are required, and the crash
    // that taught us is worth stating: choosing a row makes AutoCompleteBox call CloseDropDown(),
    // which sets SelectedItem on its inner list, which fires the query/selection hooks below --
    // and clearing the bound collection there left the selector enumerating a list it had already
    // emptied: "Index was out of range" out of SelectingItemsControl.OnSelectionModelSelectionChanged,
    // killing the app on every language or voice pick.
    [ObservableProperty] private IReadOnlyList<LanguageOption> _languageMatches = LanguageCatalog.All;

    // The voice: the library's voices for the language (else its donor's, else all of them --
    // any voice can read any language's IPA), narrowed by the picker's text.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private StoredVoice.Info? _omniVoiceVoice;

    [ObservableProperty] private string _omniVoiceVoiceQuery = "";
    /// <summary>The voice picker's rows. Replaced wholesale and off the event — see LanguageMatches.</summary>
    [ObservableProperty] private IReadOnlyList<StoredVoice.Info> _omniVoiceVoices = Array.Empty<StoredVoice.Info>();
    private IReadOnlyList<StoredVoice.Info> _allOmniVoiceVoices = Array.Empty<StoredVoice.Info>();

    [ObservableProperty] private int _omniVoiceNumStep = 32;

    // Text input — either typed/pasted in the UI or loaded from a .md file.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    private string _text = "";

    // Markdown vs plain word-grid rendering of the source pane. Persisted
    // across runs.
    [ObservableProperty] private bool _renderMarkdown;

    // Status line below the synthesize button.
    [ObservableProperty] private string _statusMessage = "Ready.";

    // Active while synthesis is running. Gates SynthesizeCommand
    // (don't allow re-start mid-run) and CancelSynthesisCommand (only
    // enabled while there's a synth to cancel). PlayPause and Stop are
    // intentionally independent — mid-synth replay is allowed.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SynthesizeCommand))]
    [NotifyCanExecuteChangedFor(nameof(CancelSynthesisCommand))]
    private bool _isBusy;

    // True once at least one chunk has been produced. Enables Play
    // (including mid-synth replay-from-beginning of the partial result).
    // Set on the first chunk-produced event; reset when a new
    // Synthesize starts.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand))]
    private bool _hasSynthesizedAudio;

    // Accumulators that persist across Play/Stop clicks during a synth
    // run. _writtenChunks tracks how many chunks were included in the
    // last WAV write — Play uses it to skip the rewrite when nothing
    // new has arrived since the last replay.
    private readonly List<float[]> _receivedAudio = new();
    private readonly List<AlignedWord> _receivedWords = new();
    private readonly object _receivedLock = new();
    private int _writtenChunks;

    // Mirrors PlaybackService.IsPlaying ("audio currently coming out").
    // False both when stopped and when paused. PlayPauseLabel /
    // CanStop derive from this + IsPausedBack.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyPropertyChangedFor(nameof(PlayPauseLabel))]
    private bool _isPlayingBack;

    // Mirrors PlaybackService.IsPaused. Distinct from !IsPlayingBack:
    // a paused session keeps its backend alive (so Resume is cheap and
    // doesn't restart from the beginning) whereas a stopped one is
    // fully torn down.
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayPauseCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyPropertyChangedFor(nameof(PlayPauseLabel))]
    private bool _isPausedBack;

    public string PlayPauseLabel => IsPausedBack ? "▶ Resume"
        : IsPlayingBack ? "⏸ Pause"
        : "▶ Play";

    // Chunk progress drives the determinate progress bar. ChunksDone is
    // bound to ProgressBar.Value; TotalChunks to .Maximum. Both zero
    // when idle (the bar's IsVisible binding to IsBusy hides it then).
    [ObservableProperty] private int _chunksDone;
    [ObservableProperty] private int _totalChunks;

    // Flat, in-order word list — the same WordItemViewModel objects that live in
    // DisplayBlocks. Built up front from the markdown so every word renders
    // immediately; timing is attached as alignment streams in. Used by the
    // highlight machinery (FindWordAt indexes this by StartSeconds).
    public ObservableCollection<WordItemViewModel> Words { get; } = new();

    // The structured (markdown-styled) karaoke view: blocks of words.
    public ObservableCollection<BlockItemViewModel> DisplayBlocks { get; } = new();

    // How many streamed alignment words have had their timing attached so far.
    private int _streamWordCursor;

    // -1 = nothing currently highlighted.
    private int _currentWordIndex = -1;

    [ObservableProperty] private string _positionLabel = "0.00 / 0.00 s";

    private AlignmentSidecar? _lastAlignment;
    private string? _lastAudioPath;
    private ITtsBackend? _synthService;
    private readonly PlaybackService _playback = new();
    private readonly SettingsService _settings = new();
    private CancellationTokenSource? _synthCts;

    // Gates the autosave during the initial settings load so we don't
    // write N times as each property hydrates.
    private bool _loadingSettings;

    public MainViewModel()
    {
        _playback.PositionChanged += OnPlaybackPositionChanged;
        _playback.PlaybackStopped += _ =>
        {
            if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
                Words[_currentWordIndex].IsCurrent = false;
            _currentWordIndex = -1;
            // Don't overwrite synthesis-in-progress status with "stopped".
            if (!IsBusy) StatusMessage = "Playback stopped.";
        };
        _playback.IsPlayingChanged += playing => Dispatcher.UIThread.Post(() => IsPlayingBack = playing);
        _playback.IsPausedChanged += paused => Dispatcher.UIThread.Post(() => IsPausedBack = paused);
        // Refresh the position label whenever the total grows — without
        // this it would only update on tick-timer ticks, so after Stop
        // (which kills the timer) any chunks still landing would update
        // the total internally but the label would stay frozen.
        _playback.TotalChanged += total => Dispatcher.UIThread.Post(() =>
            PositionLabel = $"{_playback.PositionSeconds:F2} / {total:F2} s");

        LoadSettings();

        if (!_playback.CanPlayOnThisPlatform)
            StatusMessage = _playback.UnavailableReason!;
    }

    // ── Settings persistence ─────────────────────────────────────────

    private void LoadSettings()
    {
        _loadingSettings = true;
        try
        {
            _settings.Load();
            VoicePath = _settings.Current.VoicePath;
            OnnxBundleDir = _settings.Current.OnnxBundleDir;
            RenderMarkdown = _settings.Current.RenderMarkdown;
            SelectedBackend = Enum.TryParse<TtsBackendKind>(_settings.Current.TtsBackend, out var b)
                ? b : TtsBackendKind.Chatterbox;
            KokoroModelDir = _settings.Current.KokoroModelDir;
            // The saved root, else the pre-rename Kokoro one, else the submodule. A saved dir
            // that is not a data root is a path from before the frontend moved to
            // vernacula-phonemizer; re-resolve rather than fail at load time.
            PhonemizerDataDir = PhonemizerData.IsDataRoot(_settings.Current.PhonemizerDataDir)
                ? _settings.Current.PhonemizerDataDir
                : PhonemizerData.IsDataRoot(_settings.Current.KokoroDataDir)
                ? _settings.Current.KokoroDataDir
                : (PhonemizerData.Resolve(null) ?? "");
            KokoroVoice = _settings.Current.KokoroVoice;
            KokoroSpeed = _settings.Current.KokoroSpeed > 0 ? _settings.Current.KokoroSpeed : 1.0f;
            RefreshKokoroVoices();
            OmniVoiceOnnxDir = string.IsNullOrWhiteSpace(_settings.Current.OmniVoiceOnnxDir)
                ? (Environment.GetEnvironmentVariable("OMNIVOICE_ONNX_DIR") ?? "") : _settings.Current.OmniVoiceOnnxDir;
            OmniVoiceTokenizerJson = _settings.Current.OmniVoiceTokenizerJson ?? "";
            OmniVoiceVoiceLib = StoredVoice.IsLibrary(_settings.Current.OmniVoiceVoiceLib)
                ? _settings.Current.OmniVoiceVoiceLib : (StoredVoice.ResolveDefaultLibrary() ?? "");
            OmniVoiceLang = string.IsNullOrWhiteSpace(_settings.Current.OmniVoiceLang) ? "en" : _settings.Current.OmniVoiceLang;
            OmniVoiceLanguage = LanguageCatalog.ByCode(OmniVoiceLang);
            OmniVoiceLangQuery = OmniVoiceLanguage?.Name ?? OmniVoiceLang;
            OmniVoiceNumStep = _settings.Current.OmniVoiceNumStep is > 0 and <= 64 ? _settings.Current.OmniVoiceNumStep : 32;
            RefreshOmniVoiceVoices(_settings.Current.OmniVoiceVoice);
            OmniVoiceVoiceQuery = OmniVoiceVoice?.ToString() ?? "";
        }
        finally { _loadingSettings = false; }
        UpdatePrerequisiteStatus();
    }

    /// <summary>
    /// Explains a disabled Synthesize button. The path checks in
    /// <see cref="CanSynthesize"/> keep the app from failing deep inside
    /// model loading, but a greyed-out button with no reason is its own
    /// dead end — so name the missing path here. Silent when the config is
    /// fine, so it never overwrites a synthesis/playback status.
    /// </summary>
    /// <summary>The last message <see cref="UpdatePrerequisiteStatus"/> put on screen, so it can
    /// take it back down without clobbering a status written by anything else.</summary>
    private string? _prerequisiteStatus;

    private void UpdatePrerequisiteStatus()
    {
        if (IsBusy) return;
        string? missing = SelectedBackend switch
        {
            TtsBackendKind.Kokoro =>
                !DirExists(KokoroModelDir) ? $"Kokoro model dir not found: {Describe(KokoroModelDir)}"
                : !PhonemizerData.IsDataRoot(PhonemizerDataDir) ? $"vernacula-phonemizer data dir not found: {Describe(PhonemizerDataDir)}"
                : string.IsNullOrWhiteSpace(KokoroVoice) ? "No Kokoro voice selected."
                : null,
            TtsBackendKind.OmniVoice =>
                !DirExists(OmniVoiceOnnxDir) ? $"OmniVoice ONNX dir not found: {Describe(OmniVoiceOnnxDir)}"
                : !FileExists(Path.Combine(OmniVoiceOnnxDir, IpaFineTune.DefaultDiffFile))
                    ? $"IPA fine-tune diff not found: {Path.Combine(OmniVoiceOnnxDir, IpaFineTune.DefaultDiffFile)}"
                : OmniVoiceTokenizerPath() is null
                    ? "Qwen3 tokenizer.json not found (put it beside the graphs, set OMNIVOICE_MODEL_DIR, or pick it)."
                : !PhonemizerData.IsDataRoot(PhonemizerDataDir) ? $"vernacula-phonemizer data dir not found: {Describe(PhonemizerDataDir)}"
                : !StoredVoice.IsLibrary(OmniVoiceVoiceLib) ? $"Voice library not found: {Describe(OmniVoiceVoiceLib)}"
                : OmniVoiceVoice is null ? "No OmniVoice voice selected."
                : LanguageCatalog.ByCode(OmniVoiceLang) is null ? $"Unknown language \"{OmniVoiceLang}\" — pick one from the list."
                : null,
            _ =>
                !DirExists(OnnxBundleDir) ? $"ONNX bundle dir not found: {Describe(OnnxBundleDir)}"
                : !FileExists(VoicePath) ? $"Reference voice clip not found: {Describe(VoicePath)}"
                : null,
        };
        // ⚠ CLEAR OUR OWN MESSAGE WHEN IT IS RESOLVED. Only setting it left the old
        // "not found" text on screen after the user fixed the path -- the button went
        // live while the status still said it could not run, which reads as a bug in
        // the button. Only a message this method wrote is cleared, so a synthesis or
        // playback status set elsewhere is never stepped on.
        if (missing is not null) { StatusMessage = missing; _prerequisiteStatus = missing; }
        else if (_prerequisiteStatus is not null && StatusMessage == _prerequisiteStatus)
        {
            StatusMessage = string.Empty;
            _prerequisiteStatus = null;
        }

        static string Describe(string? path) =>
            string.IsNullOrWhiteSpace(path) ? "(not set)" : path;
    }

    private void PersistSettings()
    {
        if (_loadingSettings) return;
        _settings.Current.VoicePath = VoicePath;
        _settings.Current.OnnxBundleDir = OnnxBundleDir;
        _settings.Current.RenderMarkdown = RenderMarkdown;
        _settings.Current.TtsBackend = SelectedBackend.ToString();
        _settings.Current.PhonemizerDataDir = PhonemizerDataDir;
        _settings.Current.KokoroModelDir = KokoroModelDir;
        _settings.Current.KokoroVoice = KokoroVoice;
        _settings.Current.KokoroSpeed = KokoroSpeed;
        _settings.Current.OmniVoiceOnnxDir = OmniVoiceOnnxDir;
        _settings.Current.OmniVoiceTokenizerJson = OmniVoiceTokenizerJson;
        _settings.Current.OmniVoiceVoiceLib = OmniVoiceVoiceLib;
        _settings.Current.OmniVoiceLang = OmniVoiceLang;
        _settings.Current.OmniVoiceVoice = OmniVoiceVoice?.Id ?? "";
        _settings.Current.OmniVoiceNumStep = OmniVoiceNumStep;
        _settings.Save();
    }

    // Generated [ObservableProperty] partials let us hook each setter
    // for the autosave + (for model dirs) cached-service invalidation.
    partial void OnVoicePathChanged(string value)
    {
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOnnxBundleDirChanged(string value)
    {
        InvalidateSynthService();
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnRenderMarkdownChanged(bool value) => PersistSettings();
    // Live structured preview: rebuild the karaoke blocks as the source text changes
    // (skip during synthesis — the stream is filling the words and the box is disabled).
    partial void OnTextChanged(string value)
    {
        if (!IsBusy) { BuildDisplayStructure(value); _streamWordCursor = 0; }
    }
    partial void OnSelectedBackendChanged(TtsBackendKind value)
    {
        InvalidateSynthService();
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnKokoroModelDirChanged(string value)
    {
        InvalidateSynthService();
        RefreshKokoroVoices();
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnPhonemizerDataDirChanged(string value)
    {
        InvalidateSynthService();
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOmniVoiceOnnxDirChanged(string value)
    {
        InvalidateSynthService();
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOmniVoiceTokenizerJsonChanged(string value)
    {
        InvalidateSynthService();
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOmniVoiceVoiceLibChanged(string value)
    {
        InvalidateSynthService();
        RefreshOmniVoiceVoices(OmniVoiceVoice?.Id);
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOmniVoiceLangChanged(string value)
    {
        // A new language gets its default voice, not whichever voice the last language left behind.
        RefreshOmniVoiceVoices(keepId: null);
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOmniVoiceLanguageChanged(LanguageOption? value)
    {
        if (value is not null && value.Code != OmniVoiceLang) OmniVoiceLang = value.Code;
    }
    partial void OnOmniVoiceLangQueryChanged(string value)
    {
        // The chosen language's own name means "nothing typed yet": offer the whole list, so
        // focusing the box and opening the drop-down shows everything, not the one current row.
        var q = value?.Trim() ?? "";
        var matches = q.Length == 0 || q == OmniVoiceLanguage?.Name ? LanguageCatalog.All : LanguageCatalog.Search(q);
        AfterCurrentEvent(() => LanguageMatches = matches);
        // A code typed exactly counts without a row being chosen.
        if (LanguageCatalog.ByCode(q) is { } byCode && byCode.Code != OmniVoiceLang
            && string.Equals(byCode.Code, q, StringComparison.OrdinalIgnoreCase))
            OmniVoiceLang = byCode.Code;
    }
    partial void OnOmniVoiceVoiceChanged(StoredVoice.Info? value)
    {
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnOmniVoiceVoiceQueryChanged(string value) => NarrowOmniVoiceVoices();
    partial void OnOmniVoiceNumStepChanged(int value) => PersistSettings();

    /// <summary>
    /// Run <paramref name="action"/> once the control that raised the current change has finished
    /// with it. Replacing a picker's ItemsSource from inside its own selection commit is what
    /// crashed the app; a posted replacement lands after the commit unwinds. Outside a dispatcher
    /// (unit tests, the settings load) it runs inline.
    /// </summary>
    private static void AfterCurrentEvent(Action action)
    {
        if (Dispatcher.UIThread.CheckAccess()) Dispatcher.UIThread.Post(action, DispatcherPriority.Background);
        else action();
    }

    /// <summary>The tokenizer the OmniVoice backend will use: the picked file if set, else
    /// what <see cref="OmniVoiceIpaTts.LocateTokenizerJson"/> finds. Null when neither exists.</summary>
    private string? OmniVoiceTokenizerPath() =>
        FileExists(OmniVoiceTokenizerJson) ? OmniVoiceTokenizerJson
        : DirExists(OmniVoiceOnnxDir) ? OmniVoiceIpaTts.LocateTokenizerJson(OmniVoiceOnnxDir) : null;

    // Reload the library and re-pick: keep the selection when it is still a candidate for the
    // language (and a keepId was given); otherwise the language's `default` entry, its donor's,
    // any default, else the first candidate.
    private void RefreshOmniVoiceVoices(string? keepId)
    {
        _allOmniVoiceVoices = StoredVoice.IsLibrary(OmniVoiceVoiceLib)
            ? SafeListVoices(OmniVoiceVoiceLib) : Array.Empty<StoredVoice.Info>();
        var candidates = VoiceCandidates();
        var lang = (OmniVoiceLang ?? "").Trim();
        var donor = LanguageCatalog.VoiceLangOf(lang);
        bool IsDefaultFor(StoredVoice.Info v, string code) =>
            v.IsDefault && string.Equals(v.Lang, code, StringComparison.OrdinalIgnoreCase);
        OmniVoiceVoice = (keepId is null ? null : candidates.FirstOrDefault(v => v.Id == keepId))
            ?? candidates.FirstOrDefault(v => IsDefaultFor(v, lang))
            ?? candidates.FirstOrDefault(v => IsDefaultFor(v, donor))
            ?? candidates.FirstOrDefault(v => v.IsDefault)
            ?? candidates.FirstOrDefault();
        OmniVoiceVoiceQuery = OmniVoiceVoice?.ToString() ?? "";
        NarrowOmniVoiceVoices();
        return;

        static IReadOnlyList<StoredVoice.Info> SafeListVoices(string dir)
        {
            try { return StoredVoice.ListVoices(dir); }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[OmniVoice] voice library unreadable: {ex.Message}");
                return Array.Empty<StoredVoice.Info>();
            }
        }
    }

    // The voices that can read the current language: its own, else its donor's (the web demo's
    // voiceLangOf -- Faroese read by the Icelandic voice), else the whole library.
    private List<StoredVoice.Info> VoiceCandidates()
    {
        var lang = (OmniVoiceLang ?? "").Trim();
        List<StoredVoice.Info> For(string code) =>
            _allOmniVoiceVoices.Where(v => string.Equals(v.Lang, code, StringComparison.OrdinalIgnoreCase)).ToList();
        var own = For(lang);
        if (own.Count > 0) return own;
        var donor = For(LanguageCatalog.VoiceLangOf(lang));
        return donor.Count > 0 ? donor : _allOmniVoiceVoices.ToList();
    }

    // The voice picker's rows: the candidates, narrowed by its text. The chosen voice's own label
    // means "nothing typed yet" and shows every candidate, as the language picker does.
    private void NarrowOmniVoiceVoices()
    {
        var candidates = VoiceCandidates();
        var q = (OmniVoiceVoiceQuery ?? "").Trim();
        IReadOnlyList<StoredVoice.Info> shown = q.Length == 0 || q == OmniVoiceVoice?.ToString()
            ? candidates
            : candidates.Select(v => (score: VoiceScore(v, q.ToLowerInvariant()), v))
                .Where(t => t.score >= 0).OrderBy(t => t.score).ThenBy(t => t.v.Id, StringComparer.Ordinal)
                .Select(t => t.v).ToList();
        AfterCurrentEvent(() => OmniVoiceVoices = shown);
        return;

        static int VoiceScore(StoredVoice.Info v, string q)
        {
            var id = v.Id.ToLowerInvariant(); var label = v.Label.ToLowerInvariant();
            if (id == q) return 0;
            if (id.StartsWith(q, StringComparison.Ordinal)) return 1;
            if (label.StartsWith(q, StringComparison.Ordinal)) return 2;
            if (id.Contains(q, StringComparison.Ordinal) || label.Contains(q, StringComparison.Ordinal)
                || v.Lang.Equals(q, StringComparison.OrdinalIgnoreCase)) return 3;
            return -1;
        }
    }
    partial void OnKokoroVoiceChanged(string value)
    {
        PersistSettings();
        UpdatePrerequisiteStatus();
    }
    partial void OnKokoroSpeedChanged(float value) => PersistSettings();

    // Discover Kokoro voices by scanning <modelDir>/voices/*.bin (produced by
    // scripts/kokoro_export/export_voices.py). Keeps the current selection if
    // it's still present; otherwise defaults to the first voice found.
    private void RefreshKokoroVoices()
    {
        KokoroVoices.Clear();
        var dir = Path.Combine(KokoroModelDir ?? "", "voices");
        if (!Directory.Exists(dir)) return;
        foreach (var f in Directory.EnumerateFiles(dir, "*.bin").OrderBy(p => p))
            KokoroVoices.Add(Path.GetFileNameWithoutExtension(f));
        if (KokoroVoices.Count > 0 && !KokoroVoices.Contains(KokoroVoice))
            KokoroVoice = KokoroVoices[0];
    }

    private void InvalidateSynthService()
    {
        var stale = _synthService;
        _synthService = null;
        stale?.Dispose();
    }

    // ── Pickers ──────────────────────────────────────────────────────

    [RelayCommand]
    private async Task PickVoiceAsync()
    {
        var path = await PickFileAsync("Pick a voice reference WAV",
            new FilePickerFileType("Audio") { Patterns = new[] { "*.wav", "*.flac", "*.mp3" } });
        if (path is not null) VoicePath = path;
    }

    [RelayCommand]
    private async Task PickOnnxBundleAsync()
    {
        var path = await PickFolderAsync("Pick the Chatterbox ONNX bundle directory");
        if (path is not null) OnnxBundleDir = path;
    }

    [RelayCommand]
    private async Task PickKokoroModelDirAsync()
    {
        var path = await PickFolderAsync("Pick the Kokoro model directory (kokoro.onnx + voices/)");
        if (path is not null) KokoroModelDir = path;
    }

    [RelayCommand]
    private async Task PickPhonemizerDataDirAsync()
    {
        var path = await PickFolderAsync("Pick the vernacula-phonemizer data/ directory");
        if (path is not null) PhonemizerDataDir = path;
    }

    [RelayCommand]
    private async Task PickOmniVoiceOnnxDirAsync()
    {
        var path = await PickFolderAsync("Pick the OmniVoice ONNX directory (transformer + Higgs codec + ipa_diff)");
        if (path is not null) OmniVoiceOnnxDir = path;
    }

    [RelayCommand]
    private async Task PickOmniVoiceTokenizerAsync()
    {
        var path = await PickFileAsync("Pick the Qwen3 tokenizer.json",
            new FilePickerFileType("tokenizer.json") { Patterns = new[] { "tokenizer.json", "*.json" } });
        if (path is not null) OmniVoiceTokenizerJson = path;
    }

    [RelayCommand]
    private async Task PickOmniVoiceVoiceLibAsync()
    {
        var path = await PickFolderAsync("Pick the voice library (voices.jsonc + voice-codes.json)");
        if (path is not null) OmniVoiceVoiceLib = path;
    }

    [RelayCommand]
    private async Task PickTextFileAsync()
    {
        var path = await PickFileAsync("Pick a markdown or text file",
            new FilePickerFileType("Text") { Patterns = new[] { "*.md", "*.markdown", "*.txt" } });
        if (path is null) return;
        try { Text = await File.ReadAllTextAsync(path); }
        catch (Exception ex) { StatusMessage = $"Read failed: {ex.Message}"; }
    }

    // ── Synthesis (streaming) ────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanSynthesize))]
    private async Task SynthesizeAsync()
    {
        IsBusy = true;
        StatusMessage = _synthService is null ? "Loading models (one-time)..." : "Synthesizing...";
        // Stop any in-progress playback from a prior run; clear the
        // highlight + word list, then rebuild the full structured display from
        // the markdown so every word renders immediately (timing fills in as
        // chunks stream).
        _playback.Stop();
        BuildDisplayStructure(Text);
        _streamWordCursor = 0;
        _currentWordIndex = -1;
        HasSynthesizedAudio = false;
        _lastAlignment = null;
        _lastAudioPath = null;
        ChunksDone = 0;
        TotalChunks = 0;
        lock (_receivedLock)
        {
            _receivedAudio.Clear();
            _receivedWords.Clear();
        }
        _writtenChunks = 0;
        _synthCts?.Dispose();
        _synthCts = new CancellationTokenSource();
        var token = _synthCts.Token;

        try
        {
            _synthService ??= SelectedBackend switch
            {
                TtsBackendKind.Kokoro => new KokoroSynthesisService(KokoroModelDir, PhonemizerDataDir),
                TtsBackendKind.OmniVoice => new OmniVoiceSynthesisService(OmniVoiceOnnxDir,
                    FileExists(OmniVoiceTokenizerJson) ? OmniVoiceTokenizerJson : null, PhonemizerDataDir, OmniVoiceVoiceLib),
                _ => new SynthesisService(OnnxBundleDir),
            };

            // ms precision avoids collisions when the user re-Synthesizes
            // (or clicks Play, below) twice in the same second.
            string outWav = Path.Combine(Path.GetTempPath(),
                $"vernacula_tts_{DateTime.UtcNow:yyyyMMddHHmmss_fff}.wav");
            bool streamingStarted = false;

            var request = SelectedBackend switch
            {
                TtsBackendKind.Kokoro => new TtsRequest(Text, outWav, KokoroVoice, KokoroSpeed),
                TtsBackendKind.OmniVoice => new TtsRequest(Text, outWav, OmniVoiceVoice!.Id,
                    Lang: OmniVoiceLang.Trim(), NumStep: OmniVoiceNumStep),
                _ => new TtsRequest(Text, outWav, VoicePath),
            };

            var result = await _synthService.SynthesizeStreamingAsync(
                request,
                onChunkProduced: ev =>
                {
                    // Audio appending runs HERE — on the alignment-chain
                    // thread (background), NOT on the dispatcher. On the
                    // ffplay backend, AppendSamples blocks under pipe
                    // backpressure (ffplay only drains as fast as it
                    // plays). If we did this on the UI thread, the
                    // progress bar would freeze, queued Word adds would
                    // stack up, and the user would see chunks arrive in
                    // bursts instead of streaming live.
                    try
                    {
                        if (!streamingStarted)
                        {
                            _playback.StartStreaming(_synthService!.SampleRate, channels: 1);
                            streamingStarted = true;
                        }
                        _playback.AppendSamples(ev.Audio24k);
                    }
                    catch (Exception ex)
                    {
                        Dispatcher.UIThread.Post(() =>
                            StatusMessage = $"Playback failed: {ex.Message}");
                    }

                    // Accumulate for replay-from-beginning. Play() reads
                    // these to write a partial WAV on demand.
                    bool firstChunk;
                    lock (_receivedLock)
                    {
                        firstChunk = _receivedAudio.Count == 0;
                        _receivedAudio.Add(ev.Audio24k);
                        if (ev.Words.Count > 0) _receivedWords.AddRange(ev.Words);
                    }
                    if (firstChunk)
                    {
                        Dispatcher.UIThread.Post(() => HasSynthesizedAudio = true);
                    }

                    // Attach timing to the pre-built words by running index (the
                    // streamed words are 1:1 in order with the display words).
                    // Touches WordItemViewModel state → dispatcher. Capture ev by
                    // local so the closure doesn't race a subsequent chunk's event.
                    var wordsForChunk = ev.Words;
                    int chunksDone = ev.ChunkIndex + 1;
                    int totalChunks = ev.TotalChunks;
                    Dispatcher.UIThread.Post(() =>
                    {
                        foreach (var w in wordsForChunk)
                        {
                            if (_streamWordCursor < Words.Count)
                            {
                                var vm = Words[_streamWordCursor];
                                vm.StartSeconds = w.StartSeconds;
                                vm.EndSeconds = w.EndSeconds;
                            }
                            _streamWordCursor++;
                        }
                        ChunksDone = chunksDone;
                        TotalChunks = totalChunks;
                    });
                },
                onProgress: p => Dispatcher.UIThread.Post(() =>
                {
                    StatusMessage = p.ChunkIndex is int idx && p.TotalChunks is int total
                        ? $"{p.Phase} ({idx}/{total})"
                        : p.Phase;
                    // Pick up the total at the "chunked into N paragraphs"
                    // phase so the progress bar can render proportionally
                    // before any chunk has landed.
                    if (p.TotalChunks is int t && t > TotalChunks)
                        TotalChunks = t;
                }),
                cancellationToken: token);

            _lastAlignment = result.Alignment;
            _lastAudioPath = result.AudioPath;
            // The service already wrote the WAV with ALL received chunks,
            // so cache the count to skip the rewrite if Play is clicked
            // without any new chunks since.
            lock (_receivedLock) _writtenChunks = _receivedAudio.Count;
            HasSynthesizedAudio = true;  // idempotent — set on first chunk too
            // Tell the playback service no more samples are coming so it
            // can fire PlaybackStopped naturally when the buffer drains.
            if (streamingStarted) _playback.EndOfStream();
            StatusMessage = $"Done. {result.Alignment.AudioDurationSeconds:F2}s audio, "
                + $"{result.Alignment.Words.Count} words.";
        }
        catch (OperationCanceledException)
        {
            // Play() will write a partial WAV on demand from the
            // accumulated chunks — no need to write here. Just leave
            // _lastAudioPath null so Play knows to (re)write.
            int chunksReceived;
            lock (_receivedLock) chunksReceived = _receivedAudio.Count;
            StatusMessage = chunksReceived > 0
                ? $"Cancelled — {chunksReceived} chunk(s) ready to play."
                : "Synthesis cancelled.";
        }
        catch (Exception ex)
        {
            // The status label is a single line and easy to miss; the full
            // exception goes to stderr so a failure is diagnosable from the
            // terminal without reproducing it under a debugger.
            Console.Error.WriteLine($"[Synthesize] failed: {ex}");
            StatusMessage = $"Synthesis failed: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanSynthesize()
    {
        if (IsBusy || string.IsNullOrWhiteSpace(Text)) return false;
        return SelectedBackend switch
        {
            TtsBackendKind.Kokoro =>
                DirExists(KokoroModelDir)
                && DirExists(PhonemizerDataDir)
                && !string.IsNullOrWhiteSpace(KokoroVoice),
            TtsBackendKind.OmniVoice =>
                DirExists(OmniVoiceOnnxDir)
                && FileExists(Path.Combine(OmniVoiceOnnxDir, IpaFineTune.DefaultDiffFile))
                && OmniVoiceTokenizerPath() is not null
                && DirExists(PhonemizerDataDir)
                && StoredVoice.IsLibrary(OmniVoiceVoiceLib)
                && OmniVoiceVoice is not null
                && !string.IsNullOrWhiteSpace(OmniVoiceLang),
            _ =>
                FileExists(VoicePath)
                && DirExists(OnnxBundleDir),
        };
    }

    // Existence checks, not just non-empty checks. settings.json holds
    // absolute paths from whenever the user last picked them, so a moved
    // repo or an unmounted drive leaves a plausible-looking string behind.
    // Without these, Synthesize stays enabled and fails deep inside model
    // loading, which reads to the user as "the button did nothing".
    private static bool DirExists(string? path) =>
        !string.IsNullOrWhiteSpace(path) && Directory.Exists(path);

    private static bool FileExists(string? path) =>
        !string.IsNullOrWhiteSpace(path) && File.Exists(path);

    /// <summary>Cancels in-flight synthesis. Cancellation is checked
    /// at chunk boundaries — the LM rollout for the chunk that's
    /// currently being generated can't be interrupted, so a click here
    /// takes effect after the current chunk finishes (a few seconds at
    /// worst). Any chunks that completed before cancel are written to a
    /// partial WAV so Play still works.</summary>
    [RelayCommand(CanExecute = nameof(CanCancelSynthesis))]
    private void CancelSynthesis()
    {
        _synthCts?.Cancel();
        StatusMessage = "Cancelling — will stop after the current chunk.";
    }

    private bool CanCancelSynthesis() => IsBusy;

    // ── Playback ─────────────────────────────────────────────────────

    /// <summary>Single play / pause / resume toggle. Behavior depends
    /// on the current state:
    /// - Paused  → Resume (cheap, audio backend kept alive)
    /// - Playing → Pause (suspends WaveOut or sends SIGSTOP to ffplay)
    /// - Idle    → Play from the beginning of the current WAV (writing
    ///   a partial WAV from received chunks first if needed)
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanPlayPause))]
    private void PlayPause()
    {
        if (IsPausedBack)
        {
            _playback.Resume();
            StatusMessage = "Resumed.";
            return;
        }
        if (IsPlayingBack)
        {
            _playback.Pause();
            StatusMessage = "Paused.";
            return;
        }

        // Idle — start from beginning. Snapshot the current
        // received-chunks state; if new chunks landed since the last
        // WAV write (or no WAV exists yet), write a fresh one.
        List<float[]>? snapshot = null;
        List<AlignedWord>? wordsSnap = null;
        lock (_receivedLock)
        {
            if (_receivedAudio.Count == 0) return;
            bool wavStale = _lastAudioPath is null
                || !File.Exists(_lastAudioPath)
                || _writtenChunks != _receivedAudio.Count;
            if (wavStale)
            {
                snapshot = new List<float[]>(_receivedAudio);
                wordsSnap = new List<AlignedWord>(_receivedWords);
            }
        }

        // Stop any in-progress streaming playback so SeekIntoFile gets
        // a clean backend. If we're mid-synth and the user clicked
        // Play, subsequent chunks won't auto-play (streamingStarted
        // stays latched in the synth lambda) — that's intentional, the
        // user explicitly asked for replay.
        _playback.Stop();

        if (snapshot is not null)
        {
            var outWav = Path.Combine(Path.GetTempPath(),
                $"chatterbox_app_{DateTime.UtcNow:yyyyMMddHHmmss_fff}_play.wav");
            try
            {
                SynthesisService.WriteWavFromChunks(outWav, snapshot, ChatterboxConstants.S3GenSr);
                int totalSamples = 0;
                foreach (var c in snapshot) totalSamples += c.Length;
                double duration = totalSamples / (double)ChatterboxConstants.S3GenSr;
                _lastAudioPath = outWav;
                _lastAlignment = new AlignmentSidecar
                {
                    AudioPath = outWav,
                    SampleRate = ChatterboxConstants.S3GenSr,
                    AudioDurationSeconds = duration,
                    Aligner = "chatterbox_attention",
                    Words = wordsSnap!,
                };
                _writtenChunks = snapshot.Count;
            }
            catch (Exception ex)
            {
                StatusMessage = $"Save for playback failed: {ex.Message}";
                return;
            }
        }

        if (_lastAudioPath is null || _lastAlignment is null) return;
        try
        {
            _playback.SeekIntoFile(_lastAudioPath, _lastAlignment.AudioDurationSeconds, 0);
            StatusMessage = "Playing.";
        }
        catch (Exception ex) { StatusMessage = $"Play failed: {ex.Message}"; }
    }

    // PlayPause is enabled whenever we have at least one chunk. The
    // command body branches on play/pause/idle state internally.
    private bool CanPlayPause() => HasSynthesizedAudio
        && _playback.CanPlayOnThisPlatform;

    /// <summary>Stops audio playback. Immediate — does NOT cancel any
    /// in-flight synthesis (use the Cancel button for that). The two
    /// were combined in an earlier iteration; the user fed back that
    /// the controls felt muddled, so they're separate now.</summary>
    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _playback.Stop();

    // Stop is enabled when audio is playing OR paused — both states
    // have a live backend that needs tearing down.
    private bool CanStop() => IsPlayingBack || IsPausedBack;

    // ── Structured display ───────────────────────────────────────────

    /// <summary>
    /// Rebuild <see cref="DisplayBlocks"/> + the flat <see cref="Words"/> from the markdown
    /// so the whole document renders immediately (structured, every word visible) before any
    /// audio. Words start un-timed (StartSeconds = +∞ so they're never the highlight target);
    /// the streaming handler attaches real timing by index. The word sequence here matches the
    /// backend's aligned-word sequence 1:1 (both come from MarkdownTextExtractor.Extract(Text)
    /// split on whitespace), which is what lets timing attach by running index.
    /// </summary>
    private void BuildDisplayStructure(string text)
    {
        DisplayBlocks.Clear();
        Words.Clear();
        var extract = MarkdownTextExtractor.Extract(text ?? "");
        var et = extract.Text;
        var blocks = extract.Blocks;
        var ranges = extract.Ranges;

        BlockItemViewModel? current = null;
        int currentBlockIdx = -2;
        int i = 0;
        while (i < et.Length)
        {
            while (i < et.Length && char.IsWhiteSpace(et[i])) i++;
            if (i >= et.Length) break;
            int start = i;
            while (i < et.Length && !char.IsWhiteSpace(et[i])) i++;

            int bi = FindBlockIndex(blocks, start);
            var kind = bi >= 0 ? blocks[bi].Kind : BlockKind.Paragraph;
            var level = bi >= 0 ? blocks[bi].Level : 0;
            var style = FindStyle(ranges, start);

            if (current is null || bi != currentBlockIdx)
            {
                current = new BlockItemViewModel(kind, level);
                DisplayBlocks.Add(current);
                currentBlockIdx = bi;
            }

            var w = new WordItemViewModel(et.Substring(start, i - start), Words.Count, kind, level, style, SeekToWord)
            {
                StartSeconds = double.MaxValue,
            };
            current.Words.Add(w);
            Words.Add(w);
        }
    }

    /// <summary>Index of the block whose output span contains <paramref name="offset"/>, or -1.</summary>
    private static int FindBlockIndex(IReadOnlyList<BlockSpan> blocks, int offset)
    {
        int lo = 0, hi = blocks.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (blocks[mid].OutputStart <= offset) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return best >= 0 && offset < blocks[best].OutputStart + blocks[best].OutputLength ? best : -1;
    }

    /// <summary>Inline style of the range containing <paramref name="offset"/>, or None.</summary>
    private static InlineStyle FindStyle(IReadOnlyList<TextRange> ranges, int offset)
    {
        int lo = 0, hi = ranges.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (ranges[mid].OutputStart <= offset) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return best >= 0 && offset < ranges[best].OutputStart + ranges[best].OutputLength
            ? ranges[best].Style : InlineStyle.None;
    }

    // Click-to-seek wiring. Each WordItemViewModel calls this back via a
    // delegate set at construction (avoids per-word command boilerplate +
    // keeps the VM list as a pure data collection).
    private void SeekToWord(WordItemViewModel word)
    {
        // Un-timed word (alignment hasn't reached it yet) — nothing to seek to.
        if (word.StartSeconds is double.MaxValue or <= 0 && word.EndSeconds == 0) return;

        // After synthesis completes we have an on-disk WAV and can do a
        // real audio seek. Mid-synthesis we can only re-anchor the
        // highlight clock (audio is being buffered as we go and seeking
        // into a streaming buffer would be a much bigger surgery).
        if (HasSynthesizedAudio && _lastAudioPath is not null && _lastAlignment is not null)
        {
            try
            {
                _playback.SeekIntoFile(_lastAudioPath, _lastAlignment.AudioDurationSeconds,
                    word.StartSeconds);
                StatusMessage = $"Seek → {word.StartSeconds:F2}s.";
            }
            catch (Exception ex) { StatusMessage = $"Seek failed: {ex.Message}"; }
        }
        else
        {
            _playback.SeekTo(word.StartSeconds);
        }
    }

    private void OnPlaybackPositionChanged(double posSec)
    {
        int idx = FindWordAt(posSec);
        if (idx != _currentWordIndex)
        {
            if (_currentWordIndex >= 0 && _currentWordIndex < Words.Count)
                Words[_currentWordIndex].IsCurrent = false;
            _currentWordIndex = idx;
            if (idx >= 0 && idx < Words.Count) Words[idx].IsCurrent = true;
        }
        // Use the live playback total — it grows as chunks append during
        // streaming, and matches _lastAlignment.AudioDurationSeconds once
        // synthesis finishes. (Reading _lastAlignment alone would show
        // "/ 0.00 s" for the entire stream.)
        PositionLabel = $"{posSec:F2} / {_playback.TotalEstimatedSeconds:F2} s";
    }

    private int FindWordAt(double posSec)
    {
        if (Words.Count == 0) return -1;
        int lo = 0, hi = Words.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (Words[mid].StartSeconds <= posSec) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return best;
    }

    // ── Picker plumbing ──────────────────────────────────────────────

    private static async Task<string?> PickFileAsync(string title, FilePickerFileType filter)
    {
        var top = TopLevel();
        if (top is null) return null;
        var picks = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = new[] { filter },
        });
        return picks.Count > 0 ? picks[0].TryGetLocalPath() : null;
    }

    private static async Task<string?> PickFolderAsync(string title)
    {
        var top = TopLevel();
        if (top is null) return null;
        var picks = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
        });
        return picks.Count > 0 ? picks[0].TryGetLocalPath() : null;
    }

    private static TopLevel? TopLevel()
    {
        if (Avalonia.Application.Current?.ApplicationLifetime
            is IClassicDesktopStyleApplicationLifetime desktop
            && desktop.MainWindow is not null)
        {
            return Avalonia.Controls.TopLevel.GetTopLevel(desktop.MainWindow);
        }
        return null;
    }

    public void Dispose()
    {
        _synthCts?.Cancel();
        _synthCts?.Dispose();
        _playback.Dispose();
        _synthService?.Dispose();
    }
}
