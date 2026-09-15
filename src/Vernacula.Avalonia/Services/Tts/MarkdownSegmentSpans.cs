using System.Text.RegularExpressions;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Where each rendered paragraph came from IN THE MARKDOWN, so a card edited in place can be spliced
/// back into the document without rewriting the rest of it.
///
/// <para>The segmenter gives every card a span in the EXTRACTED text; the extractor's
/// <see cref="TextRange"/> index maps extracted text back to source offsets. Composing the two gives
/// the source extent of a card's words, which is then WIDENED LEFT over the block's own marker.</para>
///
/// <para>⚠ THE EXTENT DELIBERATELY INCLUDES THE BLOCK MARKER — the `## ` of a heading, the `- ` of a
/// list item, the `&gt; ` of a quote. It did not, once, on the theory that keeping markup out of the
/// box made editing a heading's words safe from demoting it. What that actually bought was a card
/// whose kind could never be changed: reported as "the per-card editing does not permit editing of
/// the primary render marker (title level can't be changed from title level, bullet can't be changed
/// from bullet)". Promoting a heading or unmaking a bullet is a normal edit, and the card editor is
/// where the user is already standing.</para>
///
/// <para>The widening is conservative in two ways, because a wrong extent corrupts the document
/// rather than merely annoying: the text between the start of the line and the card's first word
/// must be NOTHING BUT marker characters (<see cref="BlockMarker"/>), and the widened start must not
/// reach back into the previous card's extent. Either test failing leaves the span on the words
/// alone, which is always safe.</para>
///
/// <para>Inline markup was always inside: `**bold**` lies BETWEEN two text runs of the same card, so
/// it is edited as the literal `**bold**`. A setext underline (the `===` on the NEXT line) stays
/// outside — it is trailing, not leading — so a setext heading keeps its level.</para>
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
    /// The leading markup of a block, in full: indentation, then any stack of ATX hashes, bullets,
    /// ordered-list numbers and quote arrows. Anchored at both ends, so it only ever matches when
    /// the WHOLE run from the line start to the card's first word is marker — a line that begins
    /// with a word the extractor did not emit (which would mean the index is out of step) fails the
    /// test and the span is left alone.
    /// </summary>
    private static readonly Regex BlockMarker = new(
        @"^[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|\d{1,9}[.)][ \t]+|>[ \t]*)+$", RegexOptions.Compiled);

    /// <summary>
    /// One entry per segment, in document order, for the segments whose source extent is known. A
    /// segment with no mapped range is omitted rather than guessed at — the caller then leaves that
    /// card read-only, which is always a safe answer.
    /// </summary>
    public static IReadOnlyList<Span> For(string markdown)
    {
        markdown ??= "";
        var extract = MarkdownTextExtractor.Extract(markdown);
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
            if (srcEnd <= srcStart || srcStart == int.MaxValue) continue;

            int floor = spans.Count > 0 ? spans[^1].SourceEnd : 0;
            srcStart = WidenOverMarker(markdown, srcStart, floor);
            spans.Add(new Span(seg.Index, srcStart, srcEnd - srcStart));
        }
        return spans;
    }

    /// <summary>
    /// <paramref name="start"/> moved back to the start of its line when everything in between is
    /// block marker and the move does not cross <paramref name="floor"/> (the previous card's end).
    /// </summary>
    private static int WidenOverMarker(string markdown, int start, int floor)
    {
        if (start <= 0) return start;
        int lineStart = markdown.LastIndexOf('\n', start - 1) + 1;
        if (start <= lineStart || lineStart < floor) return start;
        return BlockMarker.IsMatch(markdown[lineStart..start]) ? lineStart : start;
    }

    /// <summary>The markdown behind one card — what its editor is seeded with.</summary>
    public static string TextOf(string markdown, Span span) =>
        markdown.Substring(span.SourceStart, span.SourceLength);

    /// <summary>
    /// <paramref name="markdown"/> with one card's extent replaced. Everything outside the extent is
    /// preserved byte for byte.
    /// </summary>
    public static string Splice(string markdown, Span span, string replacement) =>
        string.Concat(markdown.AsSpan(0, span.SourceStart), replacement,
                      markdown.AsSpan(span.SourceEnd));
}
