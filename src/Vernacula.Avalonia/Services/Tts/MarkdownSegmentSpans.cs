using System.Text.RegularExpressions;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Where each rendered paragraph came from IN THE MARKDOWN, so a card edited in place can be spliced
/// back into the document without rewriting the rest of it.
///
/// <para>The segmenter gives every card a span in the EXTRACTED text; the extractor's
/// <see cref="TextRange"/> index maps extracted text back to source offsets. Composing the two gives
/// the source extent of a card's WORDS, which is then widened at BOTH ENDS over the markup that
/// belongs to those words — see <see cref="BlockMarker"/> and <see cref="TrailingInline"/>.</para>
///
/// <para>⚠ THE EXTENT DELIBERATELY INCLUDES THE BLOCK MARKER — the `## ` of a heading, the `- ` of a
/// list item, the `&gt; ` of a quote, the opening and closing `|` of a table's first and last rows.
/// It did not, once, on the theory that keeping markup out of the
/// box made editing a heading's words safe from demoting it. What that actually bought was a card
/// whose kind could never be changed: reported as "the per-card editing does not permit editing of
/// the primary render marker (title level can't be changed from title level, bullet can't be changed
/// from bullet)". Promoting a heading or unmaking a bullet is a normal edit, and the card editor is
/// where the user is already standing.</para>
///
/// <para>The widening is conservative in three ways, because a wrong extent corrupts the document
/// rather than merely annoying: the run being widened over must be NOTHING BUT markup, it must
/// contain at least one markup character rather than being blank (two trailing spaces are a hard
/// line break, and indentation is the author's layout), and it must not reach into the neighbouring
/// card. Any of those failing leaves that end of the span on the words, which is always safe.</para>
///
/// <para>Inline markup is edited raw either way: `**bold**` in the middle of a card was always
/// inside the extent, and an opener or closer at the very edge of one is now inside it too, so the
/// pair stays together. A setext underline (the `===` on the NEXT line) is still outside — the
/// widening never leaves the card's own line — so that heading shape keeps its level.</para>
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
    /// ordered-list numbers, quote arrows and table pipes, and then ANY INLINE OPENERS that follow
    /// it. Anchored
    /// at both ends, so it only ever matches when the WHOLE run from the line start to the card's
    /// first word is markup — a line that begins with a word the extractor did not emit (which
    /// would mean the index is out of step) fails the test and the span is left alone.
    ///
    /// ⚠ THE INLINE OPENERS ARE NOT OPTIONAL POLISH, they are most of real documents. A heading
    /// written `# **Title**`, or a list written `1. **Lead-in** the rest`, puts `**` between the
    /// block marker and the first word the extractor emits — so a pattern that stopped at the block
    /// marker rejected the prefix and widened nothing. That is the whole of the report that "cards
    /// can change their own kind" did not work: it worked on every document I had written to test
    /// it with and on none of the user's, because mine had bare headings.
    ///
    /// The block marker is optional for the same reason. A paragraph starting `**Bold lead-in** …`
    /// has its opening `**` outside the extent while the CLOSING one falls inside, so its editor
    /// opened on text carrying a stray `**` that closed nothing — an invitation to corrupt the
    /// document by editing around it. Widening over the opener keeps the pair together.
    /// </summary>
    private static readonly Regex BlockMarker = new(
        @"^[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|\d{1,9}[.)][ \t]+|>[ \t]*|\|[ \t]*)*[*_`~\[!]*$", RegexOptions.Compiled);

    /// <summary>
    /// The mirror of <see cref="BlockMarker"/> on the other end: inline CLOSERS, an optional link
    /// target, a table row's closing pipe, and the closing hashes of an ATX heading, up to the end
    /// of the line.
    ///
    /// ⚠ WIDENING ONLY LEFT LEAVES THE PAIR BROKEN THE OTHER WAY. A heading written `# **Title**` is
    /// bold to its last word, so the closing `**` lies past the last text the extractor emitted and
    /// the editor opened on `# **Title` — an opener with nothing to close it, which is worse than
    /// the missing marker it was meant to fix, since saving it changes what the document means.
    ///
    /// ⚠ THE INNER CLASS TAKES NO `+`, and that is not style. Written `(?:[*_`~]+|…)*` it is a
    /// quantifier inside a quantifier over the same characters, and a run of those characters ending
    /// in one that cannot match costs exponential time to REJECT — 24 of them measured at a second,
    /// 34 at hours, and this runs on the UI thread on every commit. No input I could construct
    /// actually reaches it (a long delimiter run is consumed into the extractor's ranges rather than
    /// left trailing), so this is a latent hazard rather than a live one; the shape is still wrong
    /// and costs nothing to get right.
    /// </summary>
    private static readonly Regex TrailingInline = new(
        @"^(?:[*_`~]|\]\([^()\s]*\))*[ \t]*\|?[ \t]*#*[ \t]*$", RegexOptions.Compiled);

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
        // Raw extents first — the words only — because widening one card's end needs to know where
        // the NEXT card's words begin, and widening its start needs the previous card's widened end.
        var raw = new List<Span>(segments.Count);
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
            raw.Add(new Span(seg.Index, srcStart, srcEnd - srcStart));
        }

        var spans = new List<Span>(raw.Count);
        for (int i = 0; i < raw.Count; i++)
        {
            // The floor is the PREVIOUS card's widened end and the ceiling the NEXT card's raw
            // start; together they are what keeps two cards on one line from overlapping.
            int floor   = spans.Count > 0 ? spans[^1].SourceEnd : 0;
            int ceiling = i + 1 < raw.Count ? raw[i + 1].SourceStart : markdown.Length;
            int start = WidenOverMarker(markdown, raw[i].SourceStart, floor);
            int end   = WidenOverTrailingInline(markdown, raw[i].SourceEnd, ceiling);
            spans.Add(new Span(raw[i].Index, start, end - start));
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
        var prefix = markdown[lineStart..start];
        // ⚠ Whitespace alone is not markup to widen over. Indentation that introduces nothing is the
        // author's layout, and swallowing it into the editor invites deleting it without seeing it.
        return HasMarkup(prefix) && BlockMarker.IsMatch(prefix) ? lineStart : start;
    }

    /// <summary>
    /// <paramref name="end"/> moved forward to the end of its line when everything in between is
    /// inline closing markup and the move does not cross <paramref name="ceiling"/> (the next card's
    /// first word).
    /// </summary>
    private static int WidenOverTrailingInline(string markdown, int end, int ceiling)
    {
        int lineEnd = markdown.IndexOf('\n', end);
        if (lineEnd < 0) lineEnd = markdown.Length;
        // A trailing \r belongs to the line ending, not to the card.
        if (lineEnd > end && markdown[lineEnd - 1] == '\r') lineEnd--;
        if (lineEnd <= end || lineEnd > ceiling) return end;
        var trailing = markdown[end..lineEnd];
        // ⚠ Same rule the other way, and here it is load-bearing: two trailing spaces are a markdown
        // HARD LINE BREAK. Widening over them would put an invisible, load-bearing pair of spaces at
        // the end of the editor for the user to delete by accident.
        return HasMarkup(trailing) && TrailingInline.IsMatch(trailing) ? lineEnd : end;
    }

    /// <summary>Whether a run is more than blank space — the thing that makes it worth widening over.</summary>
    private static bool HasMarkup(ReadOnlySpan<char> run) => run.IndexOfAnyExcept(' ', '\t') >= 0;

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
