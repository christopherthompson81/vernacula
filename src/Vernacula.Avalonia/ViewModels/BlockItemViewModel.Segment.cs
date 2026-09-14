using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

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

    // ── Editing this card in place ───────────────────────────────────────────

    /// <summary>Editing mode is on, so this card can be opened for editing. ⚠ It is ALSO what
    /// suppresses word-click-to-seek: the overlay that takes the click covers the words.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowEditOverlay))]
    private bool _cardEditingEnabled;

    /// <summary>This card is the one open for editing.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowEditOverlay))]
    private bool _isEditingBlock;

    /// <summary>⚠ The overlay must GO AWAY once this card is open, or it sits on top of the card's
    /// own text box and swallows every click and keystroke meant for it.</summary>
    public bool ShowEditOverlay => CardEditingEnabled && !IsEditingBlock;

    /// <summary>The card's own markdown while it is open — seeded from, and spliced back into, the
    /// document by <see cref="Vernacula.App.Services.Tts.MarkdownSegmentSpans"/>.</summary>
    [ObservableProperty] private string _editText = "";

    /// <summary>Wired by the reader view model, like WordItemViewModel's click.</summary>
    public Action<BlockItemViewModel>? EditRequested  { get; set; }
    public Action<BlockItemViewModel>? EditCommitted  { get; set; }
    public Action<BlockItemViewModel>? EditCancelled  { get; set; }

    [RelayCommand] private void RequestEdit() => EditRequested?.Invoke(this);
    [RelayCommand] private void CommitEdit()  => EditCommitted?.Invoke(this);
    [RelayCommand] private void CancelEdit()  => EditCancelled?.Invoke(this);

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
