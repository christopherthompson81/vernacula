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

    /// <summary>⚠ THE BLOCK MARKER IS INSIDE THE SPAN, so the card's own kind is editable — the
    /// reported miss was that a title could not be changed from a title, nor a bullet from a bullet.</summary>
    [Fact]
    public void AHeadingsHashesAreInTheEditableText()
    {
        const string md = "## Some heading\n\nA paragraph.";
        Assert.Equal("## Some heading", TextOfCard(md, 0));
        Assert.Equal("### New words\n\nA paragraph.", EditCard(md, 0, "### New words"));
    }

    /// <summary>The whole point: the marker can be changed to a different one, or dropped.</summary>
    [Fact]
    public void ACardsKindCanBeChangedFromItsOwnEditor()
    {
        Assert.Equal("Just a paragraph now.\n\nA paragraph.",
            EditCard("## Some heading\n\nA paragraph.", 0, "Just a paragraph now."));
        Assert.Equal("## Promoted\n- second", EditCard("- first\n- second", 0, "## Promoted"));
    }

    [Fact]
    public void AListItemAndAQuoteCarryTheirMarkersToo()
    {
        Assert.Equal("- first", TextOfCard("- first\n- second", 0));
        Assert.Equal("- second", TextOfCard("- first\n- second", 1));
        Assert.Equal("> quoted text", TextOfCard("> quoted text", 0));
        Assert.Equal("* changed\n- second", EditCard("- first\n- second", 0, "* changed"));
    }

    /// <summary>An ordered list's number is markup like any other marker.</summary>
    [Fact]
    public void AnOrderedListItemCarriesItsNumber()
    {
        const string md = "1. first\n2. second";
        Assert.Equal("1. first", TextOfCard(md, 0));
        Assert.Equal("2. second", TextOfCard(md, 1));
    }

    /// <summary>⚠ INDENTATION IS PART OF THE MARKER, because it is what makes the item nested —
    /// leaving it outside would show an editor that cannot express the nesting it is displaying.</summary>
    [Fact]
    public void ANestedListItemCarriesItsIndentation()
    {
        const string md = "- top\n  - nested";
        Assert.Equal("  - nested", TextOfCard(md, 1));
        Assert.Equal("- top\n    - deeper", EditCard(md, 1, "    - deeper"));
    }

    /// <summary>⚠ A PLAIN PARAGRAPH HAS NO MARKER TO WIDEN OVER, and must not pick up anything —
    /// this is the case where a sloppy "extend to the start of the line" would eat real text.</summary>
    [Fact]
    public void APlainParagraphSpanIsStillJustItsWords()
    {
        const string md = "First para.\n\nSecond para.";
        Assert.Equal("First para.", TextOfCard(md, 0));
        Assert.Equal("Second para.", TextOfCard(md, 1));
    }

    /// <summary>⚠ A SETEXT HEADING'S UNDERLINE IS TRAILING, not leading, so it stays outside the
    /// extent and the heading survives an edit of its words. A known limit of the marker rule: this
    /// is the one heading shape whose level the card editor cannot change.</summary>
    [Fact]
    public void ASetextUnderlineStaysOutsideTheExtent()
    {
        const string md = "Some heading\n===\n\nA paragraph.";
        Assert.Equal("Some heading", TextOfCard(md, 0));
        Assert.Equal("Reworded\n===\n\nA paragraph.", EditCard(md, 0, "Reworded"));
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

    /// <summary>
    /// ⚠ SEQUENTIAL EDITS MUST COMPOSE, because that is how the reader uses this: every card commit
    /// recomputes the spans from the CURRENT document, so an edit that changes a paragraph's length
    /// has to leave the later spans correct. Editing card 0 to something much longer and then editing
    /// card 2 is the case that breaks if spans were cached from the original text.
    /// </summary>
    [Fact]
    public void EditsComposeWhenAnEarlierCardChangesLength()
    {
        string md = "Short.\n\nMiddle para.\n\nLast para.";
        md = EditCard(md, 0, "A very much longer opening paragraph than before.");
        md = EditCard(md, 2, "Rewritten last.");
        Assert.Equal("A very much longer opening paragraph than before.\n\nMiddle para.\n\nRewritten last.", md);
    }

    /// <summary>…and shortening works the same way.</summary>
    [Fact]
    public void EditsComposeWhenAnEarlierCardShrinks()
    {
        string md = "A fairly long first paragraph here.\n\nMiddle.\n\nLast.";
        md = EditCard(md, 0, "Tiny.");
        md = EditCard(md, 1, "Changed middle.");
        Assert.Equal("Tiny.\n\nChanged middle.\n\nLast.", md);
    }

    /// <summary>A card whose text is unchanged splices to a byte-identical document — the reader
    /// relies on this to skip work when a card is opened and closed without a change.</summary>
    [Fact]
    public void ReplacingACardWithItsOwnTextIsAnIdentity()
    {
        const string md = "# Title\n\nOne.\n\n- a\n- b";
        foreach (var span in MarkdownSegmentSpans.For(md))
            Assert.Equal(md, MarkdownSegmentSpans.Splice(md, span, MarkdownSegmentSpans.TextOf(md, span)));
    }
}
