using System;
using System.Linq;
using Vernacula.Tts.Base.Markdown;
using Xunit;

namespace Vernacula.Tts.Tests;

public class ParagraphSegmenterTests
{
    [Fact]
    public void OneSegmentPerBlockInOrder()
    {
        var segs = ParagraphSegmenter.Segment("# Title\n\nFirst para, two sentences. Really.\n\n- item one\n- item two\n\n> quoted\n");
        Assert.Equal(new[] { BlockKind.Heading, BlockKind.Paragraph, BlockKind.ListItem, BlockKind.ListItem, BlockKind.Quote },
            segs.Select(s => s.Kind));
        Assert.Equal(Enumerable.Range(0, 5), segs.Select(s => s.Index));
        Assert.Equal("First para, two sentences. Really.", segs[1].Text);
    }

    [Fact]
    public void SegmentWordsConcatenateToTheWholeExtraction()
    {
        // The alignment attaches timing by running word index across the whole document while
        // the engines render a segment at a time; both must count the same words.
        string md = "# H\n\nSome *styled* text with `code`.\n\n1. a\n2. b c\n\n> q1\n> q2 continues\n";
        var extract = MarkdownTextExtractor.Extract(md);
        var whole = extract.Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var perSegment = ParagraphSegmenter.Segment(extract)
            .SelectMany(s => s.Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToArray();
        Assert.Equal(whole, perSegment);
    }

    [Fact]
    public void EmptyInputHasNoSegments()
    {
        Assert.Empty(ParagraphSegmenter.Segment("   \n\n"));
    }
}
