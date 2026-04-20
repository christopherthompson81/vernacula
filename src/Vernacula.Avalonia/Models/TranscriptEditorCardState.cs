using System.Collections.ObjectModel;
using System.Linq;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using Vernacula.App.ViewModels;

namespace Vernacula.App.Models;

internal sealed partial class TranscriptEditorCardState : ObservableObject
{
    public sealed record SpeakerChoice(int SpeakerId, string Name)
    {
        public override string ToString() => Name;
    }

    public static IReadOnlyList<AsrLanguageOption> Qwen3AsrLanguages => SettingsViewModel.Qwen3AsrLanguages;

    public TranscriptEditorCardState(EditorSegment segment, int index)
    {
        Segment = segment;
        Index = index;
        _draftSpeakerName = segment.SpeakerDisplayName;
        _draftContent = segment.Content;
        _timeRangeText = FormatTimeRange(segment.PlayStart, segment.PlayEnd);
        _suppressButtonText = "Suppress";
        _selectedRedoLanguage = MatchLanguageOption(segment.Language);
    }

    internal static AsrLanguageOption MatchLanguageOption(string? language)
    {
        if (string.IsNullOrEmpty(language)) return Qwen3AsrLanguages[0];
        return Qwen3AsrLanguages.FirstOrDefault(l =>
                   string.Equals(l.DisplayName, language, StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(l.Code,        language, StringComparison.OrdinalIgnoreCase))
               ?? Qwen3AsrLanguages[0];
    }

    private static bool LanguagesMatch(string? a, string? b)
    {
        if (string.Equals(a?.Trim(), b?.Trim(), StringComparison.OrdinalIgnoreCase))
            return true;
        // Map both through the AsrLanguageOption table so "English" and "en" compare equal.
        var optA = MatchLanguageOption(a);
        var optB = MatchLanguageOption(b);
        if (optA == Qwen3AsrLanguages[0] || optB == Qwen3AsrLanguages[0])
            return false; // unknown on either side — trust raw inequality above
        return string.Equals(optA.Code, optB.Code, StringComparison.OrdinalIgnoreCase);
    }

    internal EditorSegment Segment { get; }
    private readonly List<Color> _asrBaseColors = [];
    private readonly List<Color> _adjacentBaseColors = [];
    private int _highlightedAsrToken = -1;

    public int Index { get; set; }

    public ObservableCollection<SpeakerChoice> SpeakerChoices { get; } = [];
    public ObservableCollection<TranscriptEditorTextRunState> AsrRuns { get; } = [];
    public ObservableCollection<TranscriptEditorTextRunState> AdjacentRuns { get; } = [];

    /// <summary>Flow direction for the ASR/Edit columns in the focused card.
    /// Set to RightToLeft for Urdu/Kashmiri/Sindhi (and other RTL codes from
    /// non-Indic backends). Driven by the segment's language at build time,
    /// so mixed-direction files where LID per-segment split languages across
    /// scripts get each card rendered in the correct direction.</summary>
    [ObservableProperty] private FlowDirection _asrFlowDirection = FlowDirection.LeftToRight;

    [ObservableProperty] private bool _isFocused;
    [ObservableProperty] private AsrLanguageOption? _selectedRedoLanguage;
    public bool ShowLanguageChip { get; set; }
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
    [ObservableProperty] private Bitmap? _reassignSpeakerIconImage;
    [ObservableProperty] private Bitmap? _addSpeakerIconImage;
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
    [ObservableProperty] private IBrush _cardBackground = Brushes.Transparent;
    [ObservableProperty] private IBrush _cardBorderBrush = Brushes.Transparent;
    [ObservableProperty] private IBrush _focusedBackground = Brushes.Transparent;
    [ObservableProperty] private IBrush _focusedBorderBrush = Brushes.Transparent;
    [ObservableProperty] private IBrush _suppressIconBrush = Brushes.Black;
    private IBrush _unfocusedBackground = Brushes.Transparent;
    private IBrush _unfocusedBorderBrush = Brushes.Transparent;

    public bool IsVerified => Segment.Verified;
    public bool IsSuppressed => Segment.IsSuppressed;
    public bool HasUserEdits => DraftContent != Segment.AsrContent;
    public string AsrContent => Segment.AsrContent;
    public bool IsLanguageDivergent =>
        !string.IsNullOrWhiteSpace(Segment.Language) &&
        !string.IsNullOrWhiteSpace(Segment.LidLanguage) &&
        !LanguagesMatch(Segment.Language, Segment.LidLanguage);
    public string LidLanguageDisplay =>
        string.IsNullOrWhiteSpace(Segment.LidLanguage)
            ? ""
            : $"LID: {Segment.LidLanguage!.Trim().ToLowerInvariant()}";
    public string LidLanguageTooltip =>
        IsLanguageDivergent
            ? string.Format(Loc.Instance["editor_lid_mismatch_tooltip"],
                Segment.Language, Segment.LidLanguage)
            : "";
    public string VerifiedLabel => Loc.Instance["editor_verified"];
    public string AddSpeakerTooltip => Loc.Instance["editor_add_speaker"];
    public string AdjustTimesTooltip => Loc.Instance["editor_adjust_times"];
    public string MergePrevTooltip => Loc.Instance["editor_merge_prev"];
    public string MergeNextTooltip => Loc.Instance["editor_merge_next"];
    public string SplitTooltip => Loc.Instance["editor_split"];
    public string RedoAsrTooltip => Loc.Instance["editor_redo_asr"];

    public void RefreshLocalizedText()
    {
        OnPropertyChanged(nameof(VerifiedLabel));
        OnPropertyChanged(nameof(AddSpeakerTooltip));
        OnPropertyChanged(nameof(AdjustTimesTooltip));
        OnPropertyChanged(nameof(MergePrevTooltip));
        OnPropertyChanged(nameof(MergeNextTooltip));
        OnPropertyChanged(nameof(SplitTooltip));
        OnPropertyChanged(nameof(RedoAsrTooltip));
        OnPropertyChanged(nameof(LidLanguageTooltip));
    }

    public void SyncDraftsFromSegment()
    {
        DraftSpeakerName = Segment.SpeakerDisplayName;
        DraftContent = Segment.Content;
    }

    public void SyncSpeakerDraftFromSegment()
    {
        DraftSpeakerName = Segment.SpeakerDisplayName;
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
        else
            SyncSpeakerDraftFromSegment();

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
        OnPropertyChanged(nameof(IsLanguageDivergent));
        OnPropertyChanged(nameof(LidLanguageDisplay));
        OnPropertyChanged(nameof(LidLanguageTooltip));
    }

    public void RebuildAsrRuns(IReadOnlyList<(string text, Color background)> runs, IBrush foreground)
    {
        AsrRuns.Clear();
        _asrBaseColors.Clear();
        _highlightedAsrToken = -1;

        foreach (var (text, background) in runs)
        {
            _asrBaseColors.Add(background);
            AsrRuns.Add(new TranscriptEditorTextRunState(text, new SolidColorBrush(background), foreground));
        }
    }

    public void RebuildAdjacentRuns(IReadOnlyList<(string text, Color background)> runs, IBrush foreground)
    {
        AdjacentRuns.Clear();
        _adjacentBaseColors.Clear();
        foreach (var (text, background) in runs)
        {
            _adjacentBaseColors.Add(background);
            AdjacentRuns.Add(new TranscriptEditorTextRunState(text, new SolidColorBrush(background), foreground));
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
        RefreshCardChrome();
    }

    public void SetUnfocusedAppearance(IBrush background, IBrush borderBrush)
    {
        _unfocusedBackground = background;
        _unfocusedBorderBrush = borderBrush;
        RefreshCardChrome();
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

    partial void OnIsFocusedChanged(bool value) => RefreshCardChrome();

    private void RefreshCardChrome()
    {
        CardBackground = IsFocused ? FocusedBackground : _unfocusedBackground;
        CardBorderBrush = IsFocused ? FocusedBorderBrush : _unfocusedBorderBrush;
    }

    private static string FormatTimeRange(double start, double end)
        => FormattableString.Invariant($"{start:F3}s - {end:F3}s");
}
