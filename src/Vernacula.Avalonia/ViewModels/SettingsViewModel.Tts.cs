using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;

namespace Vernacula.App.ViewModels;

/// <summary>
/// The Settings → Text-to-Speech tab: default engine, where each model set lives (with
/// status + download), and the engine defaults new TTS jobs start from. Per-job choices
/// (language, voice) are made in the New TTS Job dialog, not here.
/// </summary>
internal partial class SettingsViewModel
{
    // ── Default engine ───────────────────────────────────────────────────────

    /// <summary>One row per engine for the picker — the list is TtsEngines.All, not a hand-written set.</summary>
    public ObservableCollection<TtsEngineOptionViewModel> TtsEngineOptions { get; } = new();

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowTtsSpeedDefault), nameof(ShowTtsStepsDefault),
                              nameof(ShowTtsReferenceClipDefault), nameof(ShowTtsTokenizerDefault))]
    private TtsEngine _selectedTtsEngine = TtsEngines.All[0];

    // The defaults section shows a control when the engine has that knob, not when it has that name.
    public bool ShowTtsSpeedDefault         => SelectedTtsEngine.UsesSpeed;
    public bool ShowTtsStepsDefault         => SelectedTtsEngine.UsesDiffusionSteps;
    public bool ShowTtsReferenceClipDefault => SelectedTtsEngine.UsesReferenceClip;
    /// <summary>Only the OmniVoice path needs a Qwen3 tokenizer pick; it is the one with a language picker.</summary>
    public bool ShowTtsTokenizerDefault     => SelectedTtsEngine.UsesLanguage;

    [RelayCommand] private void SetTtsBackend(string n) => SelectedTtsEngine = TtsEngines.For(n);

    partial void OnSelectedTtsEngineChanged(TtsEngine value)
    {
        foreach (var option in TtsEngineOptions) option.Refresh(value);
        _svc.Current.TtsBackend = value.Kind.ToString();
        _svc.Save();
        OnTtsModelsChanged?.Invoke();
    }

    // ── Model sets ───────────────────────────────────────────────────────────

    public ObservableCollection<TtsModelSetStatusViewModel> TtsModelSets { get; } = new();

    /// <summary>
    /// Called when a model location or the default engine changes, so the job runner drops its
    /// cached backend and the New TTS Job dialog re-evaluates what can run.
    /// </summary>
    public Action? OnTtsModelsChanged { get; set; }

    // ── Engine defaults ──────────────────────────────────────────────────────

    [ObservableProperty] private string _chatterboxVoicePath   = "";
    [ObservableProperty] private string _omniVoiceTokenizerJson = "";
    [ObservableProperty] private float  _kokoroSpeed            = 1.0f;
    [ObservableProperty] private int    _omniVoiceNumStep       = 32;
    /// <summary>Seconds of idleness before the loaded TTS model is released. See
    /// <see cref="AppSettings.TtsModelIdleReleaseSeconds"/>; the slider's 0 means "as soon as the
    /// work stops" and its maximum means "keep it for the session".</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(TtsModelIdleReleaseLabel))]
    private int _ttsModelIdleReleaseSeconds = 60;

    partial void OnChatterboxVoicePathChanged(string value)
    {
        _svc.Current.ChatterboxVoicePath = value;
        _svc.Save();
    }

    partial void OnOmniVoiceTokenizerJsonChanged(string value)
    {
        _svc.Current.OmniVoiceTokenizerJson = value;
        _svc.Save();
        _ = CheckTtsModelsAsync();
        OnTtsModelsChanged?.Invoke();
    }

    partial void OnKokoroSpeedChanged(float value)
    {
        _svc.Current.KokoroSpeed = value;
        _svc.Save();
    }

    partial void OnOmniVoiceNumStepChanged(int value)
    {
        _svc.Current.OmniVoiceNumStep = value;
        _svc.Save();
    }

    partial void OnTtsModelIdleReleaseSecondsChanged(int value)
    {
        _svc.Current.TtsModelIdleReleaseSeconds = StoredIdleRelease(value);
        _svc.Save();
    }

    /// <summary>The slider position that means "keep the model loaded for the session".</summary>
    public const int KeepModelLoaded = 600;

    /// <summary>
    /// What the runner stores for a slider position.
    ///
    /// <para>⚠ THE TOP STOP IS "NEVER", NOT 600 SECONDS. A slider cannot express "no timeout", and a
    /// ten-minute hold is indistinguishable from forever within a session — so the end of the track
    /// stores the NEGATIVE the runner reads as "keep it", which is what makes the label true.</para>
    /// </summary>
    public static int StoredIdleRelease(int slider) => slider >= KeepModelLoaded ? -1 : Math.Max(0, slider);

    /// <summary>What a slider position means, in words.</summary>
    public static string IdleReleaseLabel(int slider) => slider switch
    {
        >= KeepModelLoaded => "Release idle TTS model: never — keep it for the session",
        0                  => "Release idle TTS model: as soon as synthesis stops",
        var s              => $"Release idle TTS model: after {s} s idle",
    };

    public string TtsModelIdleReleaseLabel => IdleReleaseLabel(TtsModelIdleReleaseSeconds);

    [RelayCommand]
    private async Task PickChatterboxVoice()
    {
        var path = await StoragePickers.PickFileAsync("Pick the default reference voice clip",
            StoragePickers.AudioClips, StoragePickers.AllFiles);
        if (path is not null) ChatterboxVoicePath = path;
    }

    [RelayCommand]
    private async Task PickOmniVoiceTokenizer()
    {
        var path = await StoragePickers.PickFileAsync("Pick the Qwen3 tokenizer.json",
            new Avalonia.Platform.Storage.FilePickerFileType("tokenizer.json") { Patterns = ["tokenizer.json", "*.json"] },
            StoragePickers.AllFiles);
        if (path is not null) OmniVoiceTokenizerJson = path;
    }

    [RelayCommand]
    private void ClearOmniVoiceTokenizer() => OmniVoiceTokenizerJson = "";

    // ── Construction / checks ────────────────────────────────────────────────

    /// <summary>Called from the constructor: seeds the TTS fields from settings and builds the model-set rows.</summary>
    private void InitTtsSettings()
    {
        // Seeding the backing fields, as the constructor does for the ASR settings: going through
        // the properties would fire the change hooks (a settings save, OnTtsModelsChanged) before
        // anything is wired. The analyzer only waives that for code textually inside a constructor.
#pragma warning disable MVVMTK0034
        _selectedTtsEngine      = TtsEngines.For(_svc.Current.TtsBackend);
        _chatterboxVoicePath    = _svc.Current.ChatterboxVoicePath ?? "";
        _omniVoiceTokenizerJson = _svc.Current.OmniVoiceTokenizerJson ?? "";
        _kokoroSpeed            = _svc.Current.KokoroSpeed > 0 ? _svc.Current.KokoroSpeed : 1.0f;
        _omniVoiceNumStep       = _svc.Current.OmniVoiceNumStep is > 0 and <= 64 ? _svc.Current.OmniVoiceNumStep : 32;
        // A negative stored value is "never", which the slider shows at its top stop.
        _ttsModelIdleReleaseSeconds = _svc.Current.TtsModelIdleReleaseSeconds < 0
            ? KeepModelLoaded : Math.Min(_svc.Current.TtsModelIdleReleaseSeconds, KeepModelLoaded);
#pragma warning restore MVVMTK0034

        foreach (var engine in TtsEngines.All)
            TtsEngineOptions.Add(new TtsEngineOptionViewModel(engine, SelectedTtsEngine));

        void Changed() => OnTtsModelsChanged?.Invoke();
        // One row per set — the list is the table, not a hand-written copy of it. (Fully
        // qualified: this view model's own TtsModelSets property, the rows, shadows the class.)
        foreach (var set in Services.Tts.TtsModelSets.All)
            TtsModelSets.Add(new(set, _modelMgr, _svc, Changed));
    }

    // Presence only. The manifest compare hashes the whole set (9 GB for Chatterbox) and is
    // behind each row's Check for Updates button rather than run on every Settings open.
    internal async Task CheckTtsModelsAsync()
    {
        foreach (var set in TtsModelSets)
            await set.CheckAsync();
    }
}
