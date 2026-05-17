using Chatterbox.App.Models;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Chatterbox.App.ViewModels;

/// <summary>
/// Per-word presentation wrapper. The aligner emits flat
/// <see cref="AlignedWord"/> records; the UI needs an extra
/// <see cref="IsCurrent"/> bool that the AXAML can class-trigger on
/// to apply the highlight style. MainViewModel toggles this on the
/// current word as playback advances.
/// </summary>
public sealed partial class WordItemViewModel : ObservableObject
{
    public WordItemViewModel(AlignedWord word, int index)
    {
        Index = index;
        Text = word.Text;
        StartSeconds = word.StartSeconds;
        EndSeconds = word.EndSeconds;
    }

    public int Index { get; }
    public string Text { get; }
    public double StartSeconds { get; }
    public double EndSeconds { get; }

    [ObservableProperty] private bool _isCurrent;
}
