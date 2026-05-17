using System;
using Chatterbox.App.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Chatterbox.App.ViewModels;

/// <summary>
/// Per-word presentation wrapper. The aligner emits flat
/// <see cref="AlignedWord"/> records; the UI needs an extra
/// <see cref="IsCurrent"/> bool that the AXAML can class-trigger on
/// to apply the highlight style. MainViewModel toggles it as
/// playback advances. The <see cref="ClickCommand"/> calls back to
/// MainViewModel for click-to-seek.
/// </summary>
public sealed partial class WordItemViewModel : ObservableObject
{
    private readonly Action<WordItemViewModel>? _onClicked;

    public WordItemViewModel(AlignedWord word, int index, Action<WordItemViewModel>? onClicked = null)
    {
        Index = index;
        Text = word.Text;
        StartSeconds = word.StartSeconds;
        EndSeconds = word.EndSeconds;
        _onClicked = onClicked;
    }

    public int Index { get; }
    public string Text { get; }
    public double StartSeconds { get; }
    public double EndSeconds { get; }

    [ObservableProperty] private bool _isCurrent;

    [RelayCommand]
    private void Click() => _onClicked?.Invoke(this);
}
