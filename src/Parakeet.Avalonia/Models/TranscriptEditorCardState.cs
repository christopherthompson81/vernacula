using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ParakeetCSharp.Models;

internal sealed partial class TranscriptEditorCardState : ObservableObject
{
    public sealed record SpeakerChoice(int SpeakerId, string Name)
    {
        public override string ToString() => Name;
    }

    public TranscriptEditorCardState(EditorSegment segment, int index)
    {
        Segment = segment;
        Index = index;
        _draftSpeakerName = segment.SpeakerDisplayName;
        _draftContent = segment.Content;
        _timeRangeText = FormatTimeRange(segment.PlayStart, segment.PlayEnd);
        _suppressButtonText = "Suppress";
    }

    internal EditorSegment Segment { get; }

    public int Index { get; set; }

    public ObservableCollection<SpeakerChoice> SpeakerChoices { get; } = [];

    [ObservableProperty] private bool _isFocused;
    [ObservableProperty] private string _draftSpeakerName;
    [ObservableProperty] private string _draftContent;
    [ObservableProperty] private SpeakerChoice? _selectedSpeaker;
    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private string _timeRangeText;
    [ObservableProperty] private string _suppressButtonText;
    [ObservableProperty] private string _suppressGlyph = "⊘";
    [ObservableProperty] private bool _canMergePrev;
    [ObservableProperty] private bool _canMergeNext;
    [ObservableProperty] private bool _canSplit;
    [ObservableProperty] private bool _canRedoAsr;
    [ObservableProperty] private bool _canAdjustTimes = true;

    public bool IsVerified => Segment.Verified;
    public bool IsSuppressed => Segment.IsSuppressed;
    public bool HasUserEdits => DraftContent != Segment.AsrContent;
    public string AsrContent => Segment.AsrContent;

    public void SyncDraftsFromSegment()
    {
        DraftSpeakerName = Segment.SpeakerDisplayName;
        DraftContent = Segment.Content;
    }

    public void SetSpeakerChoices(IEnumerable<SpeakerChoice> choices)
    {
        SpeakerChoices.Clear();
        foreach (var choice in choices)
            SpeakerChoices.Add(choice);

        SelectedSpeaker = SpeakerChoices.FirstOrDefault(c => c.SpeakerId == Segment.SpeakerId);
    }

    public void ApplyActionAvailability(
        bool canMergePrev,
        bool canMergeNext,
        bool canSplit,
        bool canRedoAsr,
        bool canAdjustTimes)
    {
        CanMergePrev = canMergePrev;
        CanMergeNext = canMergeNext;
        CanSplit = canSplit;
        CanRedoAsr = canRedoAsr;
        CanAdjustTimes = canAdjustTimes;
    }

    public void RefreshFromSegment(bool preserveDrafts, IEnumerable<SpeakerChoice> choices,
        string suppressText, string unsuppressText, bool canMergePrev, bool canMergeNext,
        bool canSplit, bool canRedoAsr, bool canAdjustTimes)
    {
        if (!preserveDrafts)
            SyncDraftsFromSegment();

        RefreshDerived();
        SetSpeakerChoices(choices);
        SuppressButtonText = Segment.IsSuppressed ? unsuppressText : suppressText;
        SuppressGlyph = Segment.IsSuppressed ? "↺" : "⊘";
        ApplyActionAvailability(canMergePrev, canMergeNext, canSplit, canRedoAsr, canAdjustTimes);
    }

    public void RefreshDerived()
    {
        TimeRangeText = FormatTimeRange(Segment.PlayStart, Segment.PlayEnd);
        OnPropertyChanged(nameof(IsVerified));
        OnPropertyChanged(nameof(IsSuppressed));
        OnPropertyChanged(nameof(HasUserEdits));
        OnPropertyChanged(nameof(AsrContent));
    }

    partial void OnDraftContentChanged(string value) => OnPropertyChanged(nameof(HasUserEdits));

    private static string FormatTimeRange(double start, double end)
        => FormattableString.Invariant($"{start:F3}s - {end:F3}s");
}
