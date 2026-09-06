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

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsTtsChatterbox), nameof(IsTtsKokoro), nameof(IsTtsOmniVoice))]
    private TtsBackendKind _selectedTtsBackend;

    public bool IsTtsChatterbox => SelectedTtsBackend == TtsBackendKind.Chatterbox;
    public bool IsTtsKokoro     => SelectedTtsBackend == TtsBackendKind.Kokoro;
    public bool IsTtsOmniVoice  => SelectedTtsBackend == TtsBackendKind.OmniVoice;

    [RelayCommand] private void SetTtsBackend(string n)
    {
        if (Enum.TryParse<TtsBackendKind>(n, out var k)) SelectedTtsBackend = k;
    }

    partial void OnSelectedTtsBackendChanged(TtsBackendKind value)
    {
        _svc.Current.TtsBackend = value.ToString();
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
        _selectedTtsBackend     = TtsJobRunner.ParseBackend(_svc.Current.TtsBackend);
        _chatterboxVoicePath    = _svc.Current.ChatterboxVoicePath ?? "";
        _omniVoiceTokenizerJson = _svc.Current.OmniVoiceTokenizerJson ?? "";
        _kokoroSpeed            = _svc.Current.KokoroSpeed > 0 ? _svc.Current.KokoroSpeed : 1.0f;
        _omniVoiceNumStep       = _svc.Current.OmniVoiceNumStep is > 0 and <= 64 ? _svc.Current.OmniVoiceNumStep : 32;

        void Changed() => OnTtsModelsChanged?.Invoke();
        TtsModelSets.Add(new(ModelManagerService.TtsModelSet.Kokoro, "Kokoro-82M",
            "kokoro.onnx + voices/*.bin from scripts/kokoro_export. English voices; fast, light.",
            _modelMgr, _svc, Changed));
        TtsModelSets.Add(new(ModelManagerService.TtsModelSet.OmniVoice, "OmniVoice-IPA",
            "The OmniVoice base transformer, Higgs codec graphs and the IPA fine-tune diff (scripts/omnivoice_export). Any language the phonemizer covers.",
            _modelMgr, _svc, Changed));
        TtsModelSets.Add(new(ModelManagerService.TtsModelSet.OmniVoiceVoices, "OmniVoice voice library",
            "voices.jsonc + voice-codes.json — 530 stored reference voices OmniVoice reads in, one or more per language (shared with the web demo).",
            _modelMgr, _svc, Changed));
        TtsModelSets.Add(new(ModelManagerService.TtsModelSet.Chatterbox, "Chatterbox",
            "The Chatterbox ONNX bundle (scripts/chatterbox_export) + tokenizer.json. English; clones a reference clip.",
            _modelMgr, _svc, Changed));
        TtsModelSets.Add(new(ModelManagerService.TtsModelSet.PhonemizerData, "Phonemizer data",
            "The vernacula-phonemizer data/ tree (text → IPA) that Kokoro and OmniVoice need. Found automatically beside a source checkout.",
            _modelMgr, _svc, Changed));
    }

    internal async Task CheckTtsModelsAsync()
    {
        foreach (var set in TtsModelSets)
            await set.CheckAsync();
        // Manifest compares hash gigabytes; they trail the presence checks and never hold
        // the window (the ASR update check is fire-and-forget for the same reason).
        _ = Task.Run(async () =>
        {
            foreach (var set in TtsModelSets)
                await Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(set.CheckForUpdatesAsync);
        });
    }
}
