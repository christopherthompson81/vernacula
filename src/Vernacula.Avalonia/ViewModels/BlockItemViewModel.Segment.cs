using CommunityToolkit.Mvvm.ComponentModel;

namespace Vernacula.App.ViewModels;

/// <summary>
/// The block as a rendered segment: its index, the span of audio it became, and whether the
/// playhead is inside it. Timing arrives when the segment's chunk streams in (or from the
/// sidecar for a finished job); until then the box shows no duration.
/// </summary>
public sealed partial class BlockItemViewModel
{
    /// <summary>0-based segment index in document order — the same index as the sidecar chunk and seg_NNNN.wav.</summary>
    public int Index { get; init; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasTiming), nameof(DurationLabel), nameof(HeaderLabel))]
    private double _startSeconds = double.MaxValue;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasTiming), nameof(DurationLabel), nameof(HeaderLabel))]
    private double _endSeconds;

    /// <summary>True while the audio is somewhere in this segment.</summary>
    [ObservableProperty] private bool _isCurrent;

    public bool HasTiming => StartSeconds != double.MaxValue && EndSeconds > StartSeconds;

    public string DurationLabel => HasTiming ? $"{EndSeconds - StartSeconds:F1}s" : "";

    /// <summary>"¶ 3 · 4.2s" — the box's caption.</summary>
    public string HeaderLabel => HasTiming ? $"¶ {Index + 1}  ·  {DurationLabel}" : $"¶ {Index + 1}";

    public void SetTiming(double start, double end)
    {
        StartSeconds = start;
        EndSeconds = end;
    }
}
