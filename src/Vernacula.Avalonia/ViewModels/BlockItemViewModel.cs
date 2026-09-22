using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Avalonia.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.ViewModels;

/// <summary>
/// One markdown block (heading / paragraph / list item / quote / table) in the structured karaoke
/// view, holding the word view models it contains. The AXAML switches layout on the
/// <see cref="IsHeading"/>/<see cref="IsListItem"/>/<see cref="IsQuote"/>/<see cref="IsTable"/>
/// helpers (bullet, indent, spacing, grid); per-word font/size/style live on
/// <see cref="WordItemViewModel"/>.
/// </summary>
public sealed partial class BlockItemViewModel : ObservableObject
{
    public BlockItemViewModel(BlockKind kind, int level)
    {
        Kind = kind;
        Level = level;
    }

    public BlockKind Kind { get; }
    public int Level { get; }
    /// <summary>The block's words in LOGICAL order — the order they are spoken, which is what
    /// alignment attaches timing to.</summary>
    public ObservableCollection<WordItemViewModel> Words { get; } = new();

    /// <summary>
    /// The same words in the order the panel must place them to read correctly. For a block with
    /// one direction throughout this is just <see cref="Words"/>; where the two directions mix, the
    /// embedded run is reversed so that the panel's mirroring puts it back the right way round.
    /// </summary>
    public ObservableCollection<WordItemViewModel> Display { get; } = new();

    public bool IsHeading => Kind == BlockKind.Heading;
    public bool IsParagraph => Kind == BlockKind.Paragraph;
    public bool IsListItem => Kind == BlockKind.ListItem;
    public bool IsQuote => Kind == BlockKind.Quote;
    public bool IsTable => Kind == BlockKind.Table;

    // ── A table card ─────────────────────────────────────────────────────────

    /// <summary>
    /// The grid, for a <see cref="BlockKind.Table"/> block; empty for every other kind. The rows
    /// hold the same word view models as <see cref="Words"/>, arranged by grid position, so the
    /// card draws a table while the alignment still walks one flat list of words.
    /// </summary>
    public ObservableCollection<TableRowViewModel> Rows { get; } = new();

    // Cell spans in output order (the extractor's reading order), and the cell at each position,
    // for placing words as they are built.
    private IReadOnlyList<TableCellSpan> _cellSpans = Array.Empty<TableCellSpan>();
    private readonly Dictionary<(int Row, int Column), TableCellViewModel> _cellAt = new();

    /// <summary>
    /// Build the empty grid this block's words will be placed into. The shape comes from the cells
    /// that produced text, so a table whose last column is blank throughout is a narrower table —
    /// which is what it looks like, and what it reads as.
    /// </summary>
    public void InitTable(IReadOnlyList<TableCellSpan> cells)
    {
        Rows.Clear();
        _cellAt.Clear();
        _cellSpans = cells;
        if (cells.Count == 0) return;

        int rowCount = 0, columnCount = 0;
        foreach (var c in cells)
        {
            if (c.Row + 1 > rowCount) rowCount = c.Row + 1;
            if (c.Column + 1 > columnCount) columnCount = c.Column + 1;
        }
        var headerRows = new HashSet<int>();
        foreach (var c in cells)
            if (c.IsHeader) headerRows.Add(c.Row);

        for (int r = 0; r < rowCount; r++)
        {
            var row = new TableRowViewModel(r, headerRows.Contains(r), columnCount);
            for (int c = 0; c < columnCount; c++)
            {
                var cell = new TableCellViewModel(r, c, row.IsHeader);
                row.Cells.Add(cell);
                _cellAt[(r, c)] = cell;
            }
            Rows.Add(row);
        }
    }

    /// <summary>
    /// Place a word in the cell whose span contains <paramref name="outputOffset"/>. A word that
    /// falls between cells — the extractor's row separators are the only text there — belongs to no
    /// cell and is simply not drawn in the grid; it is still in <see cref="Words"/>, so it is still
    /// spoken and still timed.
    /// </summary>
    public void PlaceWordInCell(int outputOffset, WordItemViewModel word)
    {
        int lo = 0, hi = _cellSpans.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (_cellSpans[mid].OutputStart <= outputOffset) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (best < 0) return;
        var span = _cellSpans[best];
        if (outputOffset >= span.OutputStart + span.OutputLength) return;
        if (!_cellAt.TryGetValue((span.Row, span.Column), out var cell)) return;
        cell.Words.Add(word);
        // The cell span stops before the separator the extractor writes between cells, so what
        // falls past its end is punctuation this machinery added rather than text the author wrote.
        word.TrimDisplayTo(span.OutputStart + span.OutputLength - outputOffset);
    }

    /// <summary>
    /// One card, built from one segment: its words in spoken order, and — for a table — the same
    /// words arranged into the grid. <paramref name="firstWordIndex"/> is where this card's words
    /// start in the document's flat word list, which is the index alignment attaches timing by.
    /// </summary>
    public static BlockItemViewModel FromSegment(TextSegment segment, string extractedText,
        IReadOnlyList<TextRange> ranges, int firstWordIndex, Action<WordItemViewModel>? onWordClicked,
        string? lang = null)
    {
        var block = new BlockItemViewModel(segment.Kind, segment.Level) { Index = segment.Index };
        // A table card holds its words twice over: once flat, for speech and timing, and once by
        // grid position, for drawing. The grid has to exist before the words are built so each one
        // can be dropped into its cell as it is made.
        if (segment.Cells is { Count: > 0 }) block.InitTable(segment.Cells);

        // ⚠ THE WORD UNIT COMES FROM THE LANGUAGE, NOT FROM WHITESPACE. This scanned for spaces
        // itself, which is right for most languages and gives exactly one clickable word per
        // sentence in Japanese or Chinese — the whole paragraph lighting up at once, and a click
        // anywhere in it seeking to its start. WordSegmentation returns the same whitespace split
        // for everything else, so nothing changes where nothing needed to.
        int end = segment.OutputStart + segment.OutputLength;
        foreach (var span in WordSegmentation.Segment(extractedText, segment.OutputStart, end, lang))
        {
            var word = new WordItemViewModel(extractedText[span.Start..span.End],
                firstWordIndex + block.Words.Count,
                segment.Kind, segment.Level, ParagraphSegmenter.StyleAt(ranges, span.Start), onWordClicked)
            {
                StartSeconds = double.MaxValue,
            };
            block.Words.Add(word);
            if (block.IsTable) block.PlaceWordInCell(span.Start, word);
        }
        return block;
    }

    /// <summary>
    /// Which way this block's words are laid out. A browser reorders inline elements by the
    /// bidirectional algorithm; a layout panel does not, so an Arabic or Persian line would run
    /// left to right with its first word on the left -- backwards. Set once the block's words are
    /// in, from the text itself rather than the picked language, so a quoted RTL passage inside an
    /// English document is still laid out correctly.
    /// </summary>
    [ObservableProperty] private FlowDirection _flowDirection = FlowDirection.LeftToRight;

    /// <summary>
    /// Decide the block's direction from the words it now holds, and lay them out for it.
    ///
    /// ⚠ A PANEL MIRRORS EVERY CHILD, WHICH IS NOT WHAT BIDIRECTIONAL TEXT DOES. Mirroring an
    /// Arabic line is right; mirroring an English phrase embedded in it is not — "Text To Speech"
    /// inside a Persian sentence would read "Speech To Text". The bidirectional algorithm reverses
    /// runs, not words, so each embedded run is reversed here and the panel's mirroring undoes it.
    /// Words with no strong direction of their own (digits, punctuation) stay with the run they
    /// are in, exactly as they take their direction from their surroundings in text.
    /// </summary>
    /// <param name="languageIsRtl">Whether the language being read is right-to-left, which is what
    /// settles a block that contains both directions.</param>
    public void UpdateFlowDirection(bool? languageIsRtl = null)
    {
        FlowDirection = LayOut(Words, languageIsRtl, Display);
        // A table's cells are laid out one at a time: a cell is its own run of text, so an English
        // column beside an Arabic one is not one mixed line but two blocks of prose side by side.
        foreach (var row in Rows)
            foreach (var cell in row.Cells)
                cell.FlowDirection = LayOut(cell.Words, languageIsRtl, cell.Display);
    }

    /// <summary>
    /// Fill <paramref name="display"/> with <paramref name="words"/> in the order a panel must
    /// place them, and answer the direction that panel must be given.
    /// </summary>
    private static FlowDirection LayOut(IReadOnlyList<WordItemViewModel> words, bool? languageIsRtl,
        ObservableCollection<WordItemViewModel> display)
    {
        var rtl = TextDirection.Resolve(string.Join(' ', words.Select(w => w.Text)), languageIsRtl);

        display.Clear();
        var run = new List<WordItemViewModel>();      // an embedded run, awaiting its reversal
        var pending = new List<WordItemViewModel>();  // neutrals whose side is not settled yet
        foreach (var w in words)
        {
            var strong = TextDirection.StrongDirectionOf(w.Text);
            if (strong is null)
            {
                if (run.Count == 0) display.Add(w);
                else if (TextDirection.IsNumberWord(w.Text))
                {
                    // A number keeps company with the word before it -- "iPhone 15" stays "iPhone
                    // 15" -- so it joins the run rather than waiting to see what follows.
                    run.AddRange(pending);
                    pending.Clear();
                    run.Add(w);
                }
                // Any other neutral takes its direction from what surrounds it, so it cannot be
                // placed until the next strong word says which side of the boundary it fell on.
                else pending.Add(w);
            }
            else if (strong == rtl)
            {
                // Back to the block's own direction: the run ends, and any neutrals waiting inside
                // it were trailing it -- a full stop after an English phrase still ends the
                // Persian line -- so they belong to the block.
                FlushRun();
                foreach (var n in pending) display.Add(n);
                pending.Clear();
                display.Add(w);
            }
            else
            {
                // The run continues, and it swallows the neutrals in the middle of it: "Text &
                // Speech" is one embedded phrase, not two with an ampersand between them.
                run.AddRange(pending);
                pending.Clear();
                run.Add(w);
            }
        }
        FlushRun();
        foreach (var n in pending) display.Add(n);
        return rtl ? FlowDirection.RightToLeft : FlowDirection.LeftToRight;

        // Reversed, so that the panel mirroring the block puts the run back the right way round.
        void FlushRun()
        {
            for (var i = run.Count - 1; i >= 0; i--) display.Add(run[i]);
            run.Clear();
        }
    }
}
