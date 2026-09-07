using CommunityToolkit.Mvvm.ComponentModel;
using Vernacula.App.Services.Tts;

namespace Vernacula.App.ViewModels;

/// <summary>
/// One engine in the Settings → Text-to-Speech picker: what to show, and whether it is the
/// current choice. A row per <see cref="TtsEngines.All"/> entry, so the picker gains an engine
/// when the registry does rather than when someone remembers to add a RadioButton.
/// </summary>
internal sealed partial class TtsEngineOptionViewModel : ObservableObject
{
    public TtsEngine Engine { get; }
    public string Name => Engine.DisplayName;
    public string Description => Engine.Description;
    /// <summary>Command parameter for SetTtsBackend.</summary>
    public string KindName => Engine.Kind.ToString();

    [ObservableProperty] private bool _isSelected;

    public TtsEngineOptionViewModel(TtsEngine engine, TtsEngine selected)
    {
        Engine = engine;
        _isSelected = ReferenceEquals(engine, selected);
    }

    public void Refresh(TtsEngine selected) => IsSelected = ReferenceEquals(Engine, selected);
}
