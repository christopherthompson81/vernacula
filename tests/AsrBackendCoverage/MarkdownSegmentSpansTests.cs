using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace AsrBackendCoverage;

/// <summary>
/// Mapping a rendered card back to its markdown, which is what lets a card be edited in place
/// without rewriting the document around it.
/// </summary>
public class MarkdownSegmentSpansTests
{
    private static string TextOfCard(string md, int index)
    {
        var span = MarkdownSegmentSpans.For(md).Single(s => s.Index == index);
        return MarkdownSegmentSpans.TextOf(md, span);
    }

    private static string EditCard(string md, int index, string replacement)
    {
        var span = MarkdownSegmentSpans.For(md).Single(s => s.Index == index);
        return MarkdownSegmentSpans.Splice(md, span, replacement);
    }

    [Fact]
    public void ACardsSpanIsItsOwnWords()
    {
        const string md = "First para.\n\nSecond para.\n\nThird para.";
        Assert.Equal("First para.", TextOfCard(md, 0));
        Assert.Equal("Second para.", TextOfCard(md, 1));
        Assert.Equal("Third para.", TextOfCard(md, 2));
    }

    [Fact]
    public void EditingOneCardLeavesEveryOtherByteAlone()
    {
        const string md = "First para.\n\nSecond para.\n\nThird para.";
        Assert.Equal("First para.\n\nRewritten entirely.\n\nThird para.",
            EditCard(md, 1, "Rewritten entirely."));
    }

    /// <summary>⚠ THE MARKUP IS OUTSIDE THE SPAN. Editing a heading's words cannot demote it, because
    /// the `##` was never in the extracted text and so is never in the extent.</summary>
    [Fact]
    public void AHeadingKeepsItsHashes()
    {
        const string md = "## Some heading\n\nA paragraph.";
        Assert.Equal("Some heading", TextOfCard(md, 0));
        Assert.Equal("## New words\n\nA paragraph.", EditCard(md, 0, "New words"));
    }

    [Fact]
    public void AListItemKeepsItsBulletAndAQuoteItsMarker()
    {
        Assert.Equal("- changed\n- second", EditCard("- first\n- second", 0, "changed"));
        Assert.Equal("> changed", EditCard("> quoted text", 0, "changed"));
    }

    /// <summary>Inline markup lies BETWEEN two text runs of one card, so it is inside the extent and
    /// is edited as literal markdown. The intended bargain, pinned so it is not a surprise.</summary>
    [Fact]
    public void InlineMarkupIsInsideTheSpanAndEditedRaw()
    {
        const string md = "Some **bold** text.";
        Assert.Equal("Some **bold** text.", TextOfCard(md, 0));
    }

    /// <summary>⚠ The whole reason not to rebuild by joining blocks: unusual whitespace and
    /// indentation outside the edited extent survive exactly.</summary>
    [Fact]
    public void FormattingOutsideTheEditedExtentSurvivesExactly()
    {
        const string md = "First.\n\n\n\nSecond.\n\n   \n\nThird.";
        var result = EditCard(md, 1, "Changed.");
        Assert.Equal("First.\n\n\n\nChanged.\n\n   \n\nThird.", result);
    }

    [Fact]
    public void SpansAreInDocumentOrderAndDoNotOverlap()
    {
        var spans = MarkdownSegmentSpans.For("# Title\n\nOne.\n\n- a\n- b\n\n> q");
        for (int i = 1; i < spans.Count; i++)
        {
            Assert.True(spans[i].Index > spans[i - 1].Index);
            Assert.True(spans[i].SourceStart >= spans[i - 1].SourceEnd,
                $"span {i} starts at {spans[i].SourceStart}, before {spans[i - 1].SourceEnd}");
        }
    }

    /// <summary>Every card the reader draws should be editable; a gap would be a read-only card.</summary>
    [Fact]
    public void EveryRenderedCardHasASpan()
    {
        const string md = "# Title\n\nA paragraph with **bold**.\n\n- item one\n- item two\n\n> a quote";
        var segments = ParagraphSegmenter.Segment(md);
        var spans = MarkdownSegmentSpans.For(md);
        Assert.Equal(segments.Count, spans.Count);
    }

    [Fact]
    public void AnEmptyDocumentHasNoSpans() => Assert.Empty(MarkdownSegmentSpans.For(""));
}
