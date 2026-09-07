using System;
using System.Collections.Generic;
using System.Linq;
using Avalonia.Media;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Vernacula.App.ViewModels;

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
    [ObservableProperty] private string? _ipa;

    /// <summary>
    /// The word's sub-parts, when it is written without spaces between its words (Japanese,
    /// Chinese) and the phonemizer could say where the boundaries are. Empty for ordinary words,
    /// which render as one piece of text with one reading above it.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPieces))]
    private IReadOnlyList<RubyPieceViewModel> _pieces = Array.Empty<RubyPieceViewModel>();

    public bool HasPieces => Pieces.Count > 0;

    /// <summary>Whether this word reserves the ruby line. True once an annotation has been attached
    /// — with the annotation off there is no row at all, and no space held for one.</summary>
    [ObservableProperty] private bool _hasRubyRow;

    /// <summary>
    /// Which way this word's own pieces are laid out. The block's direction mirrors everything
    /// inside it, which is right for the words but not for the pieces of a word from the other
    /// direction: a segmented Latin or Japanese word inside an Arabic line would have its pieces
    /// rendered back to front.
    /// </summary>
    public FlowDirection PieceFlowDirection =>
        TextDirection.StrongDirectionOf(Text) == true ? FlowDirection.RightToLeft : FlowDirection.LeftToRight;

    /// <summary>Attach (or with null, clear) the annotation for this word.</summary>
    public void SetRuby(WordRuby? ruby)
    {
        // The row is reserved for every annotated word, including one with no reading of its own
        // (punctuation): a word that skipped the row would sit higher than the rest of its line.
        HasRubyRow = ruby is not null;
        Ipa = ruby?.Ipa;
        Pieces = ruby is null || ruby.Pieces.Count == 0
            ? Array.Empty<RubyPieceViewModel>()
            : ruby.Pieces.Select(p => new RubyPieceViewModel(this, p.Text, p.Ipa, p.Weight)).ToList();
    }

    /// <summary>
    /// Light the piece being spoken at <paramref name="posSec"/>. The aligner times this word as a
    /// whole -- a Japanese sentence with no spaces is one aligned word -- so its span is divided
    /// among the pieces in proportion to how much speech each one's reading is worth, the same
    /// weighting the duration model uses. That is an estimate within the word, not a measurement.
    /// </summary>
    public void HighlightPieceAt(double posSec)
    {
        if (!HasPieces) return;
        var total = Pieces.Sum(p => p.Weight);
        var span = EndSeconds - StartSeconds;
        var current = -1;
        if (total > 0 && span > 0)
        {
            var at = StartSeconds;
            for (var i = 0; i < Pieces.Count; i++)
            {
                var end = at + span * Pieces[i].Weight / total;
                if (posSec < end) { current = i; break; }
                at = end;
            }
            if (current < 0 && posSec >= StartSeconds && posSec <= EndSeconds) current = Pieces.Count - 1;
        }
        for (var i = 0; i < Pieces.Count; i++) Pieces[i].IsCurrent = i == current;
    }

    /// <summary>Drop the piece highlight (the word is no longer the one being spoken).</summary>
    public void ClearPieceHighlight()
    {
        foreach (var p in Pieces) p.IsCurrent = false;
    }

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

    /// <summary>
    /// How tall a line box is, as a multiple of its font size.
    ///
    /// ⚠ GENEROUS ON PURPOSE, AND IT HAS TO BE. Avalonia only centres a run in the box while the
    /// box is taller than the run's natural height; ask for less and it clamps the box and pins the
    /// baseline to the ascent, leaving the descent to spill out below — over the row beneath, since
    /// a TextBlock does not clip. Devanagari and Thai fallbacks run to about 1.5 em before their
    /// line gap, and a mixed run takes the tallest ascent and deepest descent of any font in it, so
    /// the factor has to clear that. IPA needs the room anyway: it stacks tie bars and tone letters
    /// above the letters and marks below them.
    /// </summary>
    private double LineBox => ScriptFonts.LineBoxFor(Text);

    /// <summary>
    /// A fixed line box for the word, so every word in a line sits on the same baseline. Its size
    /// depends on the script (see <see cref="ScriptFonts.LineBoxFor"/>).
    ///
    /// ⚠ EACH WORD IS ITS OWN CONTROL, so nothing aligns them for us. Left to size themselves they
    /// do not agree: IPA and many scripts fall back font by font, and a fallback's ascent and
    /// descent are its own, so "d͡ʒˈʌmps" measures taller than "ðə" and the word beneath it rides
    /// down. A fixed box takes that out of the LAYOUT — the box no longer grows, so no word moves
    /// its neighbours. It does not make every font sit identically inside the box: Avalonia centres
    /// the run there, so fonts of different ascent/descent asymmetry still place their glyphs a
    /// little differently. That is why the ruby line also names a font that covers IPA outright,
    /// rather than relying on the box alone.
    /// </summary>
    public double LineHeight => Math.Ceiling(FontSize * LineBox);

    /// <summary>
    /// The font this word is set in: monospace for inline code, else the UI font unless the word's
    /// script needs a specific face (see <see cref="ScriptFonts"/>).
    ///
    /// ⚠ CODE IS DECIDED HERE, NOT IN A STYLE. A font set on the control beats one set by a style,
    /// so a `.code` style setter would be ignored the moment this property exists; keeping both
    /// decisions in one place is what stops them from silently disagreeing.
    /// </summary>
    public FontFamily FontFamily => IsCode ? ScriptFonts.Mono : ScriptFonts.For(Text);

    /// <summary>The ruby line's fixed height, on the same reasoning.</summary>
    public double IpaLineHeight => Math.Ceiling(IpaFontSize * LineBox);

    public FontStyle FontStyle => Style.HasFlag(InlineStyle.Italic) ? FontStyle.Italic : FontStyle.Normal;

    public bool IsCode => Style.HasFlag(InlineStyle.Code);
    public bool IsLink => Style.HasFlag(InlineStyle.Link);

    [RelayCommand]
    private void Click() => _onClicked?.Invoke(this);
}

/// <summary>
/// One sub-part of a split word: the characters and their reading, plus whether it is the piece
/// being spoken. Font sizes come from the owning word, so a piece inside a heading scales with it.
/// </summary>
public sealed partial class RubyPieceViewModel : ObservableObject
{
    public RubyPieceViewModel(WordItemViewModel word, string text, string ipa, double weight)
    {
        Word = word;
        Text = text;
        Ipa = ipa;
        Weight = weight;
    }

    public WordItemViewModel Word { get; }
    public string Text { get; }
    public string Ipa { get; }
    public double Weight { get; }

    [ObservableProperty] private bool _isCurrent;
}
