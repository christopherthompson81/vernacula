using System.Linq;
using Vernacula.App.ViewModels;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace AsrBackendCoverage;

/// <summary>
/// A markdown table drawn as one card: the grid the reader lays out, and the words that go in it.
///
/// <para>Tables were dropped by the extractor until documents needed reviewing rather than just
/// reading aloud — a dropped table is missing content, and it went missing silently. These pin
/// both halves of what replaced that: the table is spoken row by row, and it is drawn as a grid
/// of the same word view models the rest of the reader uses.</para>
/// </summary>
public class TableCardTests
{
    private static BlockItemViewModel CardFor(string markdown, int index = 0)
    {
        var extract = MarkdownTextExtractor.Extract(markdown);
        var segment = ParagraphSegmenter.Segment(extract)[index];
        return BlockItemViewModel.FromSegment(segment, extract.Text, extract.Ranges, 0, null);
    }

    private const string Grid = """
        | Item | Status | Notes |
        |------|--------|-------|
        | Ledger | Open | Two entries |
        | Appendix C |  | Awaiting the vendor |
        """;

    [Fact]
    public void ATableCardIsAGridOfTheRightShape()
    {
        var card = CardFor(Grid);
        Assert.True(card.IsTable);
        Assert.True(card.ShowTableGrid);
        Assert.False(card.ShowWordFlow);
        Assert.Equal(3, card.Rows.Count);
        Assert.All(card.Rows, r => Assert.Equal(3, r.Cells.Count));
        Assert.Equal(new[] { true, false, false }, card.Rows.Select(r => r.IsHeader));
    }

    [Fact]
    public void EveryWordLandsInItsOwnCell()
    {
        var card = CardFor(Grid);
        string Cell(int r, int c) =>
            string.Join(' ', card.Rows[r].Cells[c].Words.Select(w => w.DisplayText));

        Assert.Equal("Item", Cell(0, 0));
        Assert.Equal("Status", Cell(0, 1));
        Assert.Equal("Notes", Cell(0, 2));
        Assert.Equal("Two entries", Cell(1, 2));
        Assert.Equal("Appendix C", Cell(2, 0));
        Assert.Equal("Awaiting the vendor", Cell(2, 2));
    }

    /// <summary>⚠ A blank cell must leave its column EMPTY rather than closing up, or every cell
    /// after it in the row slides under the wrong heading.</summary>
    [Fact]
    public void ABlankCellLeavesItsColumnEmpty()
    {
        var card = CardFor(Grid);
        Assert.Empty(card.Rows[2].Cells[1].Words);
    }

    /// <summary>The cells hold the SAME word objects as the flat list, so highlighting a word as
    /// the audio reaches it lights it up in the grid.</summary>
    [Fact]
    public void TheGridHoldsTheSameWordsAsTheFlatList()
    {
        var card = CardFor(Grid);
        var inGrid = card.Rows.SelectMany(r => r.Cells).SelectMany(c => c.Words).ToList();
        Assert.Equal(card.Words.Count, inGrid.Count);
        foreach (var w in inGrid) Assert.Contains(w, card.Words);
    }

    /// <summary>⚠ The comma and the period are the EXTRACTOR'S, for prosody. They are spoken and
    /// they are not drawn — a header cell reading "Status," is the machinery showing through.</summary>
    [Fact]
    public void TheSpokenSeparatorsAreNotDrawn()
    {
        var card = CardFor(Grid);
        var header = card.Rows[0].Cells[1].Words.Single();
        Assert.Equal("Status,", header.Text);
        Assert.Equal("Status", header.DisplayText);

        var lastInRow = card.Rows[1].Cells[2].Words.Last();
        Assert.Equal("entries.", lastInRow.Text);
        Assert.Equal("entries", lastInRow.DisplayText);
    }

    /// <summary>Everything that is not a table is untouched: one flow of words, no grid.</summary>
    [Fact]
    public void AnOrdinaryParagraphIsStillAFlowOfWords()
    {
        var card = CardFor("Just a sentence with words.");
        Assert.False(card.IsTable);
        Assert.True(card.ShowWordFlow);
        Assert.Empty(card.Rows);
        Assert.Equal(5, card.Words.Count);
        Assert.Equal("words.", card.Words.Last().DisplayText);
    }
}
