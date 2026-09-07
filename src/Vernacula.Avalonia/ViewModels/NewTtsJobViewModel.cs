using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base;

namespace Vernacula.App.ViewModels;

/// <summary>
/// The New Text-to-Speech Job dialog: the document to read and the per-job rendering choices
/// (engine, language, voice, speed / steps). Opens seeded from the Settings → TTS defaults and
/// the last choices used; Start writes those back so the next job starts where this one left
/// off. Model locations are Settings' business — the dialog only reports what is missing.
/// </summary>
internal partial class NewTtsJobViewModel : ObservableObject
{
    private readonly SettingsService _settings;

    // ── Document ─────────────────────────────────────────────────────────────

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _documentPath = "";

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _jobName = "";

    // ── Engine ───────────────────────────────────────────────────────────────

    public IReadOnlyList<TtsBackendKind> Backends { get; } =
        [TtsBackendKind.Kokoro, TtsBackendKind.OmniVoice, TtsBackendKind.Chatterbox];

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    [NotifyPropertyChangedFor(nameof(IsChatterbox), nameof(IsKokoro), nameof(IsOmniVoice))]
    private TtsBackendKind _selectedBackend;

    public bool IsChatterbox => SelectedBackend == TtsBackendKind.Chatterbox;
    public bool IsKokoro     => SelectedBackend == TtsBackendKind.Kokoro;
    public bool IsOmniVoice  => SelectedBackend == TtsBackendKind.OmniVoice;

    // ── Chatterbox: reference clip ───────────────────────────────────────────

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _chatterboxVoicePath = "";

    // ── Kokoro: named voice + speed ──────────────────────────────────────────

    public ObservableCollection<string> KokoroVoices { get; } = new();

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _kokoroVoice = "";

    [ObservableProperty] private float _kokoroSpeed = 1.0f;

    // ── OmniVoice: language (type-ahead), voice (type-ahead), steps ──────────
    // The persisted truth is the CODE (OmniVoiceLang). The picker's text drives LanguageMatches;
    // choosing a row sets OmniVoiceLanguage → the code. Typing a code exactly ("cy") also works.

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private string _omniVoiceLang = "en";

    [ObservableProperty] private string _omniVoiceLangQuery = "";
    [ObservableProperty] private LanguageOption? _omniVoiceLanguage;

    // ⚠ WHOLE-LIST REPLACEMENT, POSTED, NOT AN ObservableCollection CLEARED IN PLACE. Choosing a
    // row makes AutoCompleteBox call CloseDropDown(), which fires the selection hooks below; a
    // list cleared inside that left the selector enumerating an emptied collection ("Index was
    // out of range") and killed the app on every pick. See AfterCurrentEvent.
    [ObservableProperty] private IReadOnlyList<LanguageOption> _languageMatches = LanguageCatalog.All;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartCommand))]
    private StoredVoice.Info? _omniVoiceVoice;

    [ObservableProperty] private string _omniVoiceVoiceQuery = "";
    [ObservableProperty] private IReadOnlyList<StoredVoice.Info> _omniVoiceVoices = Array.Empty<StoredVoice.Info>();
    private IReadOnlyList<StoredVoice.Info> _allOmniVoiceVoices = Array.Empty<StoredVoice.Info>();

    [ObservableProperty] private int _omniVoiceNumStep = 32;

    // ── Prerequisites ────────────────────────────────────────────────────────

    /// <summary>Why Start is disabled, in terms of a path to fix; empty when the job can run.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPrerequisiteMessage))]
    private string _prerequisiteMessage = "";

    public bool HasPrerequisiteMessage => !string.IsNullOrEmpty(PrerequisiteMessage);

    // ── Hooks (wired by MainViewModel) ───────────────────────────────────────

    /// <summary>Called when the user confirms: (documentPath, jobTitle, settings).</summary>
    public Func<string, string, TtsJobSettings, Task>? EnqueueJob { get; set; }
    public Action? NavigateBack { get; set; }

    private bool _loading;

    public NewTtsJobViewModel(SettingsService settings)
    {
        _settings = settings;
        Reset();
    }

    /// <summary>Re-seeds every field from Settings + last-used choices. Call before showing the dialog.</summary>
    public void Reset()
    {
        _loading = true;
        try
        {
            DocumentPath = "";
            JobName = "";
            var s = _settings.Current;
            SelectedBackend     = TtsJobRunner.ParseBackend(s.TtsBackend);
            ChatterboxVoicePath = s.ChatterboxVoicePath ?? "";
            KokoroSpeed         = s.KokoroSpeed > 0 ? s.KokoroSpeed : 1.0f;
            KokoroVoice         = s.KokoroVoice ?? "";
            RefreshKokoroVoices();
            OmniVoiceLang       = string.IsNullOrWhiteSpace(s.OmniVoiceLang) ? "en" : s.OmniVoiceLang;
            OmniVoiceLanguage   = LanguageCatalog.ByCode(OmniVoiceLang);
            OmniVoiceLangQuery  = OmniVoiceLanguage?.Name ?? OmniVoiceLang;
            OmniVoiceNumStep    = s.OmniVoiceNumStep is > 0 and <= 64 ? s.OmniVoiceNumStep : 32;
            RefreshOmniVoiceVoices(s.OmniVoiceVoice);
            OmniVoiceVoiceQuery = OmniVoiceVoice?.ToString() ?? "";
        }
        finally { _loading = false; }
        UpdatePrerequisites();
    }

    /// <summary>The choices as they would be stored on the job.</summary>
    public TtsJobSettings CurrentSettings() => SelectedBackend switch
    {
        TtsBackendKind.Kokoro    => new TtsJobSettings("Kokoro", "", KokoroVoice, KokoroSpeed),
        TtsBackendKind.OmniVoice => new TtsJobSettings("OmniVoice", OmniVoiceLang.Trim(), OmniVoiceVoice?.Id ?? "", NumStep: OmniVoiceNumStep),
        _                        => new TtsJobSettings("Chatterbox", "", ChatterboxVoicePath),
    };

    private void UpdatePrerequisites()
    {
        if (_loading) return;
        PrerequisiteMessage = TtsPrerequisites.Describe(SelectedBackend, _settings, CurrentSettings()) ?? "";
        StartCommand.NotifyCanExecuteChanged();
    }

    // ── Change hooks ─────────────────────────────────────────────────────────

    partial void OnSelectedBackendChanged(TtsBackendKind value) => UpdatePrerequisites();
    partial void OnChatterboxVoicePathChanged(string value)     => UpdatePrerequisites();
    partial void OnKokoroVoiceChanged(string value)             => UpdatePrerequisites();
    partial void OnOmniVoiceVoiceChanged(StoredVoice.Info? value) => UpdatePrerequisites();

    partial void OnOmniVoiceLangChanged(string value)
    {
        if (_loading) return;
        // A new language gets its default voice, not whichever voice the last language left behind.
        RefreshOmniVoiceVoices(keepId: null);
        UpdatePrerequisites();
    }

    partial void OnOmniVoiceLanguageChanged(LanguageOption? value)
    {
        if (value is not null && value.Code != OmniVoiceLang) OmniVoiceLang = value.Code;
    }

    partial void OnOmniVoiceLangQueryChanged(string value)
    {
        // The chosen language's own name means "nothing typed yet": offer the whole list.
        var q = value?.Trim() ?? "";
        var matches = q.Length == 0 || q == OmniVoiceLanguage?.Name ? LanguageCatalog.All : LanguageCatalog.Search(q);
        AfterCurrentEvent(() => LanguageMatches = matches);
        // A code typed exactly counts without a row being chosen.
        if (LanguageCatalog.ByCode(q) is { } byCode && byCode.Code != OmniVoiceLang
            && string.Equals(byCode.Code, q, StringComparison.OrdinalIgnoreCase))
            OmniVoiceLang = byCode.Code;
    }

    partial void OnOmniVoiceVoiceQueryChanged(string value) => NarrowOmniVoiceVoices();

    /// <summary>
    /// Run <paramref name="action"/> once the control that raised the current change has
    /// finished with it — replacing a picker's ItemsSource from inside its own selection commit
    /// is what crashed the standalone reader. Outside a dispatcher it runs inline.
    /// </summary>
    private static void AfterCurrentEvent(Action action)
    {
        if (Dispatcher.UIThread.CheckAccess()) Dispatcher.UIThread.Post(action, DispatcherPriority.Background);
        else action();
    }

    // ── Voice lists ──────────────────────────────────────────────────────────

    private void RefreshKokoroVoices()
    {
        KokoroVoices.Clear();
        var dir = Path.Combine(_settings.GetKokoroModelsDir(), "voices");
        if (!Directory.Exists(dir)) return;
        foreach (var f in Directory.EnumerateFiles(dir, "*.bin").OrderBy(p => p))
            KokoroVoices.Add(Path.GetFileNameWithoutExtension(f));
        if (KokoroVoices.Count > 0 && !KokoroVoices.Contains(KokoroVoice))
            KokoroVoice = KokoroVoices[0];
    }

    // Reload the library and re-pick: keep keepId when it is still a candidate for the
    // language; otherwise the candidates' default entry, else the first candidate.
    private void RefreshOmniVoiceVoices(string? keepId)
    {
        string lib = _settings.GetOmniVoiceVoiceLibDir();
        _allOmniVoiceVoices = StoredVoice.IsLibrary(lib) ? SafeListVoices(lib) : Array.Empty<StoredVoice.Info>();
        var candidates = VoiceCandidates();
        OmniVoiceVoice = (keepId is null ? null : candidates.FirstOrDefault(v => v.Id == keepId))
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

    // The voices that can read the current language: its own, else its donor's (Faroese read
    // by the Icelandic voice), else the whole library — any voice can read any language's IPA.
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

    // ── Commands ─────────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task SelectDocumentAsync()
    {
        var path = await StoragePickers.PickFileAsync(Loc.Instance["label_document"],
            StoragePickers.TextDocuments, StoragePickers.AllFiles);
        if (path is null) return;
        DocumentPath = path;
        if (string.IsNullOrWhiteSpace(JobName))
            JobName = Path.GetFileNameWithoutExtension(path);
    }

    [RelayCommand]
    private async Task PickChatterboxVoiceAsync()
    {
        var path = await StoragePickers.PickFileAsync(Loc.Instance["label_reference_clip"],
            StoragePickers.AudioClips, StoragePickers.AllFiles);
        if (path is not null) ChatterboxVoicePath = path;
    }

    private bool CanStart() =>
        !string.IsNullOrWhiteSpace(DocumentPath) &&
        File.Exists(DocumentPath) &&
        !string.IsNullOrWhiteSpace(JobName) &&
        !HasPrerequisiteMessage;

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task Start()
    {
        var tts = CurrentSettings();
        RememberChoices(tts);
        try
        {
            if (EnqueueJob != null)
                await EnqueueJob(DocumentPath, JobName, tts);
            NavigateBack?.Invoke();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[NewTtsJobVM] Start() EXCEPTION: {ex}");
            PrerequisiteMessage = ex.Message;
        }
    }

    /// <summary>The next dialog opens on these choices.</summary>
    private void RememberChoices(TtsJobSettings tts)
    {
        var s = _settings.Current;
        s.TtsBackend = tts.Backend;
        switch (SelectedBackend)
        {
            case TtsBackendKind.Kokoro:
                s.KokoroVoice = KokoroVoice;
                s.KokoroSpeed = KokoroSpeed;
                break;
            case TtsBackendKind.OmniVoice:
                s.OmniVoiceLang    = OmniVoiceLang;
                s.OmniVoiceVoice   = OmniVoiceVoice?.Id ?? "";
                s.OmniVoiceNumStep = OmniVoiceNumStep;
                break;
            default:
                s.ChatterboxVoicePath = ChatterboxVoicePath;
                break;
        }
        _settings.Save();
    }

    [RelayCommand]
    private void Back() => NavigateBack?.Invoke();
}
