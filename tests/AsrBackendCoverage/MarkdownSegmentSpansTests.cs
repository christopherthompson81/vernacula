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

    /// <summary>
    /// ⚠ THE REGRESSION THAT MADE THE FEATURE LOOK DEAD. A heading written `# **Title**` puts inline
    /// markup between the block marker and the first word the extractor emits, so a pattern that
    /// stopped at the block marker rejected the prefix and widened nothing — the card opened with no
    /// `# ` in it, exactly as reported. Bare headings are the shape this was first tested with and
    /// the shape real documents are least likely to use.
    /// </summary>
    [Fact]
    public void AHeadingWhoseTitleIsBoldStillCarriesItsHashes()
    {
        const string md = "# **A bold title**\n\nBody text.";
        Assert.Equal("# **A bold title**", TextOfCard(md, 0));
        Assert.Equal("### **A bold title**\n\nBody text.", EditCard(md, 0, "### **A bold title**"));
    }

    /// <summary>…and the same shape in a list, which is how a lead-in is usually written.</summary>
    [Fact]
    public void AListItemWithABoldLeadInCarriesItsMarker()
    {
        const string md = "1. **Lead-in.** The rest of the item.\n2. **Another.** More.";
        Assert.Equal("1. **Lead-in.** The rest of the item.", TextOfCard(md, 0));
        Assert.Equal("2. **Another.** More.", TextOfCard(md, 1));
    }

    /// <summary>
    /// ⚠ AND A PARAGRAPH WITH A BOLD LEAD-IN, which has no block marker at all. Its opening `**`
    /// used to sit outside the extent while the CLOSING one fell inside, so the editor opened on
    /// text carrying a `**` that closed nothing — an invitation to corrupt the document by editing
    /// around it. Widening over the opener keeps the pair together.
    /// </summary>
    [Fact]
    public void AParagraphWithABoldLeadInKeepsThePairTogether()
    {
        const string md = "**Lead-in.** The rest of the paragraph.";
        string editor = TextOfCard(md, 0);
        Assert.Equal(md, editor);
        Assert.Equal(2, System.Text.RegularExpressions.Regex.Matches(editor, @"\*\*").Count);
    }

    /// <summary>Backticks and links open the same way and are widened over for the same reason.</summary>
    [Theory]
    [InlineData("- `code` then words", "- `code` then words")]
    [InlineData("## [A link](/x) heading", "## [A link](/x) heading")]
    [InlineData("> _quiet_ start", "> _quiet_ start")]
    public void OtherInlineOpenersAfterAMarkerAreWidenedOverToo(string md, string expected) =>
        Assert.Equal(expected, TextOfCard(md, 0));

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

    /// <summary>
    /// ⚠ THE PREMISE THE READER'S IMMEDIATE REFRESH RESTS ON: editing the marker through the card's
    /// own extent changes what the segmenter says the block IS. The cards are rebuilt on a commit
    /// only when this comes out different, so if it ever stopped being true the marker would become
    /// editable again in name only.
    /// </summary>
    [Fact]
    public void EditingTheMarkerChangesTheSegmentsKindAndLevel()
    {
        const string md = "## A heading\n\nA paragraph.";
        Assert.Equal(BlockKind.Heading, ParagraphSegmenter.Segment(md)[0].Kind);
        Assert.Equal(2, ParagraphSegmenter.Segment(md)[0].Level);

        Assert.Equal(BlockKind.Paragraph,
            ParagraphSegmenter.Segment(EditCard(md, 0, "Just words now."))[0].Kind);
        Assert.Equal(4,
            ParagraphSegmenter.Segment(EditCard(md, 0, "#### A heading"))[0].Level);
        Assert.Equal(BlockKind.ListItem,
            ParagraphSegmenter.Segment(EditCard(md, 0, "- A heading"))[0].Kind);
    }

    /// <summary>
    /// ⚠ A TABLE IS ONE CARD, AND ITS EXTENT MUST COVER THE WHOLE GRID — the leading `|` of the
    /// first row, the alignment row, and the trailing `|` of the last. An extent that started at
    /// the first word would open an editor on a table with its outer pipes shorn off, and saving
    /// that would stop the block being a table at all.
    /// </summary>
    [Fact]
    public void ATablesExtentIsTheWholeTable()
    {
        const string md = "Intro.\n\n| Item | Status |\n|------|--------|\n| Ledger | Open |\n\nOutro.";
        Assert.Equal("| Item | Status |\n|------|--------|\n| Ledger | Open |", TextOfCard(md, 1));
    }

    [Fact]
    public void ATableCanBeEditedWithoutTouchingWhatSurroundsIt()
    {
        const string md = "Intro.\n\n| Item | Status |\n|---|---|\n| Ledger | Open |\n\nOutro.";
        Assert.Equal("Intro.\n\n| Item | Status | Owner |\n|---|---|---|\n| Ledger | Open | AP |\n\nOutro.",
            EditCard(md, 1, "| Item | Status | Owner |\n|---|---|---|\n| Ledger | Open | AP |"));
    }

    /// <summary>And the card's kind follows the edit, as it does for every other marker.</summary>
    [Fact]
    public void UnmakingATableTurnsTheCardBackIntoProse()
    {
        const string md = "| Item | Status |\n|---|---|\n| Ledger | Open |";
        Assert.Equal(BlockKind.Table, ParagraphSegmenter.Segment(md)[0].Kind);
        Assert.Equal(BlockKind.Paragraph,
            ParagraphSegmenter.Segment(EditCard(md, 0, "Just a sentence."))[0].Kind);
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
