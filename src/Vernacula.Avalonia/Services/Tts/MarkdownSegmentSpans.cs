using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Where each rendered paragraph came from IN THE MARKDOWN, so a card edited in place can be spliced
/// back into the document without rewriting the rest of it.
///
/// <para>The segmenter gives every card a span in the EXTRACTED text; the extractor's
/// <see cref="TextRange"/> index maps extracted text back to source offsets. Composing the two gives
/// the source extent of a card's words.</para>
///
/// <para>⚠ THE EXTENT COVERS THE TEXT, NOT THE BLOCK — and that is what makes splicing safe. A
/// heading's `## `, a list item's `- `, a quote's `> ` are markup the extractor never emitted, so
/// they fall OUTSIDE the span and survive an edit untouched: editing the words of a heading cannot
/// accidentally demote it to a paragraph. Inline markup is different — `**bold**` lies BETWEEN two
/// text runs of the same card, so it falls inside the span and is edited as the literal `**bold**`.
/// That is the intended bargain: structure is rendered and preserved, inline markup is raw.</para>
///
/// <para>⚠ AND THE DOCUMENT IS NEVER REBUILT BY JOINING BLOCKS. Splicing one extent leaves every
/// byte outside it exactly as the author wrote it — blank lines, trailing spaces, setext underlines,
/// indentation. A join would silently reformat a file this app writes back over.</para>
/// </summary>
internal static class MarkdownSegmentSpans
{
    /// <summary>A card's words as they appear in the markdown.</summary>
    public sealed record Span(int Index, int SourceStart, int SourceLength)
    {
        public int SourceEnd => SourceStart + SourceLength;
    }

    /// <summary>
    /// One entry per segment, in document order, for the segments whose source extent is known. A
    /// segment with no mapped range is omitted rather than guessed at — the caller then leaves that
    /// card read-only, which is always a safe answer.
    /// </summary>
    public static IReadOnlyList<Span> For(string markdown)
    {
        var extract = MarkdownTextExtractor.Extract(markdown ?? "");
        var segments = ParagraphSegmenter.Segment(extract);
        var spans = new List<Span>(segments.Count);

        foreach (var seg in segments)
        {
            int outEnd = seg.OutputStart + seg.OutputLength;
            int srcStart = int.MaxValue, srcEnd = -1;
            foreach (var r in extract.Ranges)
            {
                if (r.OutputStart >= outEnd) break;          // Ranges are sorted by OutputStart.
                if (r.OutputStart < seg.OutputStart) continue;
                if (r.SourceLength <= 0) continue;           // a degenerate span indexes nothing
                srcStart = Math.Min(srcStart, r.SourceStart);
                srcEnd   = Math.Max(srcEnd, r.SourceStart + r.SourceLength);
            }
            if (srcEnd > srcStart && srcStart != int.MaxValue)
                spans.Add(new Span(seg.Index, srcStart, srcEnd - srcStart));
        }
        return spans;
    }

    /// <summary>The markdown behind one card — what its editor is seeded with.</summary>
    public static string TextOf(string markdown, Span span) =>
        markdown.Substring(span.SourceStart, span.SourceLength);

    /// <summary>
    /// <paramref name="markdown"/> with one card's extent replaced. Everything outside the extent —
    /// including the block's own markup — is preserved byte for byte.
    /// </summary>
    public static string Splice(string markdown, Span span, string replacement) =>
        string.Concat(markdown.AsSpan(0, span.SourceStart), replacement,
                      markdown.AsSpan(span.SourceEnd));
}
