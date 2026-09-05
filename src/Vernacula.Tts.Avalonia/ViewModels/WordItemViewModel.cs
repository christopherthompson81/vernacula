using System;
using Avalonia.Media;
using Vernacula.Tts.Base.Markdown;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Vernacula.Tts.App.ViewModels;

/// <summary>
/// Per-word presentation wrapper for the structured (markdown-styled) karaoke view.
/// Carries the word's text, its block context (kind + heading level) and inline style
/// (bold/italic/code/link) so the AXAML can render it like rendered markdown, plus an
/// <see cref="IsCurrent"/> highlight flag the playback loop toggles. Timing
/// (<see cref="StartSeconds"/>/<see cref="EndSeconds"/>) is attached after construction as
/// alignment streams in — the display is built from the full document up front, then lit up.
/// </summary>
public sealed partial class WordItemViewModel : ObservableObject
{
    private readonly Action<WordItemViewModel>? _onClicked;

    public WordItemViewModel(string text, int index, BlockKind blockKind, int headingLevel,
        InlineStyle style, Action<WordItemViewModel>? onClicked = null)
    {
        Index = index;
        Text = text;
        BlockKind = blockKind;
        HeadingLevel = headingLevel;
        Style = style;
        _onClicked = onClicked;
    }

    public int Index { get; }
    public string Text { get; }
    public BlockKind BlockKind { get; }
    public int HeadingLevel { get; }
    public InlineStyle Style { get; }

    // Timing is mutable: set when the matching AlignedWord streams in.
    public double StartSeconds { get; set; }
    public double EndSeconds { get; set; }

    [ObservableProperty] private bool _isCurrent;

    /// <summary>
    /// The word's reading, shown small above it (furigana style) when IPA annotation is on.
    /// Null or empty means nothing is drawn — punctuation has no reading, and the annotation is
    /// cleared wholesale when the option is off or the text changes under it.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasIpa))]
    private string? _ipa;

    public bool HasIpa => !string.IsNullOrEmpty(Ipa);

    // ── Self-describing display properties (bound directly by the word button) ──
    public double FontSize => BlockKind == BlockKind.Heading
        ? HeadingLevel switch { 1 => 30, 2 => 24, 3 => 20, 4 => 18, _ => 16 }
        : 18;

    public FontWeight FontWeight =>
        BlockKind == BlockKind.Heading || Style.HasFlag(InlineStyle.Bold)
            ? FontWeight.Bold : FontWeight.Normal;

    /// <summary>Ruby size: small enough to read as an annotation, never so small it is unreadable
    /// under a heading's larger body text.</summary>
    public double IpaFontSize => Math.Max(10.0, FontSize * 0.6);

    public FontStyle FontStyle => Style.HasFlag(InlineStyle.Italic) ? FontStyle.Italic : FontStyle.Normal;

    public bool IsCode => Style.HasFlag(InlineStyle.Code);
    public bool IsLink => Style.HasFlag(InlineStyle.Link);

    [RelayCommand]
    private void Click() => _onClicked?.Invoke(this);
}
