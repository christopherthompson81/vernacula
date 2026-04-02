using System.Collections.ObjectModel;
using System.Linq;
using Avalonia.Media;
using Avalonia.Media.Imaging;
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
    private readonly List<Color> _asrBaseColors = [];
    private readonly List<Color> _adjacentBaseColors = [];
    private int _highlightedAsrToken = -1;

    public int Index { get; set; }

    public ObservableCollection<SpeakerChoice> SpeakerChoices { get; } = [];
    public ObservableCollection<TranscriptEditorTextRunState> AsrRuns { get; } = [];
    public ObservableCollection<TranscriptEditorTextRunState> AdjacentRuns { get; } = [];

    [ObservableProperty] private bool _isFocused;
    [ObservableProperty] private string _draftSpeakerName;
    [ObservableProperty] private string _draftContent;
    [ObservableProperty] private SpeakerChoice? _selectedSpeaker;
    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private string _timeRangeText;
    [ObservableProperty] private string _suppressButtonText;
    [ObservableProperty] private bool _canMergePrev;
    [ObservableProperty] private bool _canMergeNext;
    [ObservableProperty] private bool _canSplit;
    [ObservableProperty] private bool _canRedoAsr;
    [ObservableProperty] private bool _canAdjustTimes = true;
    [ObservableProperty] private bool _isRedoAsrSpinning;
    [ObservableProperty] private Bitmap? _suppressIconImage;
    [ObservableProperty] private Bitmap? _suppressAlertIconImage;
    [ObservableProperty] private Bitmap? _adjustTimesIconImage;
    [ObservableProperty] private Bitmap? _mergePrevIconImage;
    [ObservableProperty] private Bitmap? _mergeNextIconImage;
    [ObservableProperty] private Bitmap? _splitIconImage;
    [ObservableProperty] private Bitmap? _redoAsrIconImage;
    [ObservableProperty] private Bitmap? _redoAsrSpinnerImage;
    [ObservableProperty] private bool _showAdjacentRuns;
    [ObservableProperty] private string _adjacentPlainText = "";
    [ObservableProperty] private IBrush _adjacentBackground = Brushes.Transparent;
    [ObservableProperty] private IBrush _adjacentForeground = Brushes.Black;
    [ObservableProperty] private TextDecorationCollection? _adjacentTextDecorations;
    [ObservableProperty] private IBrush _focusedBackground = Brushes.Transparent;
    [ObservableProperty] private IBrush _focusedBorderBrush = Brushes.Transparent;
    [ObservableProperty] private IBrush _suppressIconBrush = Brushes.Black;

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

    public void RebuildAsrRuns(IReadOnlyList<(string text, Color background)> runs)
    {
        AsrRuns.Clear();
        _asrBaseColors.Clear();
        _highlightedAsrToken = -1;

        foreach (var (text, background) in runs)
        {
            _asrBaseColors.Add(background);
            AsrRuns.Add(new TranscriptEditorTextRunState(text, new SolidColorBrush(background)));
        }
    }

    public void RebuildAdjacentRuns(IReadOnlyList<(string text, Color background)> runs, IBrush foreground)
    {
        AdjacentRuns.Clear();
        _adjacentBaseColors.Clear();
        foreach (var (text, background) in runs)
        {
            _adjacentBaseColors.Add(background);
            AdjacentRuns.Add(new TranscriptEditorTextRunState(text, new SolidColorBrush(background)));
        }

        ShowAdjacentRuns = true;
        AdjacentForeground = foreground;
        AdjacentTextDecorations = null;
    }

    public void SetAdjacentPlainText(string text, IBrush background, IBrush foreground,
        TextDecorationCollection? decorations = null)
    {
        AdjacentPlainText = text;
        AdjacentBackground = background;
        AdjacentForeground = foreground;
        AdjacentTextDecorations = decorations;
        ShowAdjacentRuns = false;
        AdjacentRuns.Clear();
        _adjacentBaseColors.Clear();
    }

    public void SetFocusedAppearance(IBrush background, IBrush borderBrush, IBrush suppressIconBrush)
    {
        FocusedBackground = background;
        FocusedBorderBrush = borderBrush;
        SuppressIconBrush = suppressIconBrush;
    }

    public void ApplyHighlightedAsrToken(int tokenIndex, Color accentColor)
    {
        if (_highlightedAsrToken == tokenIndex)
            return;

        if (_highlightedAsrToken >= 0 && _highlightedAsrToken < AsrRuns.Count && _highlightedAsrToken < _asrBaseColors.Count)
            AsrRuns[_highlightedAsrToken].Background = new SolidColorBrush(_asrBaseColors[_highlightedAsrToken]);

        if (tokenIndex >= 0 && tokenIndex < AsrRuns.Count)
            AsrRuns[tokenIndex].Background = new SolidColorBrush(Color.FromArgb(120, accentColor.R, accentColor.G, accentColor.B));

        _highlightedAsrToken = tokenIndex;
    }

    partial void OnDraftContentChanged(string value) => OnPropertyChanged(nameof(HasUserEdits));

    private static string FormatTimeRange(double start, double end)
        => FormattableString.Invariant($"{start:F3}s - {end:F3}s");
}
