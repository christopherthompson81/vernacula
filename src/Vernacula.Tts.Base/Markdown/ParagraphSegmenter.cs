namespace Vernacula.Tts.Base.Markdown;

/// <summary>
/// One unit of synthesis: a markdown block (heading, paragraph, list item, quote) as speakable
/// text. <see cref="Text"/> is the exact slice of the extracted text from the segment's first
/// word to its last, so splitting it on whitespace yields the same words, in the same order, as
/// splitting the whole extraction — the invariant the reader's word-by-word alignment rests on.
/// </summary>
public sealed record TextSegment(int Index, BlockKind Kind, int Level, string Text, int OutputStart, int OutputLength)
{
    /// <summary>The segment's words, as the aligner counts them.</summary>
    public int WordCount => Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
}

/// <summary>
/// Cuts an extraction into the segments the engines render one at a time and the reader draws
/// one box each — the same cut on both sides, so a segment's audio is exactly the audio of the
/// box it is shown in and can be re-rendered on its own later.
///
/// A segment is a run of consecutive words in one block. Words the extractor placed outside any
/// block (there should be none, but the index is best-effort) form a paragraph segment of their
/// own. Blocks that contain no words produce no segment.
/// </summary>
public static class ParagraphSegmenter
{
    public static IReadOnlyList<TextSegment> Segment(MarkdownExtractionResult extract)
    {
        var et = extract.Text;
        var blocks = extract.Blocks;
        var segments = new List<TextSegment>();

        int i = 0, curBlock = int.MinValue, segStart = -1, segEnd = -1;
        BlockKind kind = BlockKind.Paragraph;
        int level = 0;
        while (i < et.Length)
        {
            while (i < et.Length && char.IsWhiteSpace(et[i])) i++;
            if (i >= et.Length) break;
            int start = i;
            while (i < et.Length && !char.IsWhiteSpace(et[i])) i++;

            int bi = BlockIndexAt(blocks, start);
            if (segStart < 0 || bi != curBlock)
            {
                Flush();
                segStart = start;
                curBlock = bi;
                kind  = bi >= 0 ? blocks[bi].Kind  : BlockKind.Paragraph;
                level = bi >= 0 ? blocks[bi].Level : 0;
            }
            segEnd = i;
        }
        Flush();
        return segments;

        void Flush()
        {
            if (segStart < 0) return;
            segments.Add(new TextSegment(segments.Count, kind, level,
                et.Substring(segStart, segEnd - segStart), segStart, segEnd - segStart));
            segStart = -1;
        }
    }

    /// <summary>Convenience: extract then segment.</summary>
    public static IReadOnlyList<TextSegment> Segment(string markdown) =>
        Segment(MarkdownTextExtractor.Extract(markdown ?? ""));

    /// <summary>Index of the block whose output span contains <paramref name="offset"/>, or -1.</summary>
    public static int BlockIndexAt(IReadOnlyList<BlockSpan> blocks, int offset)
    {
        int lo = 0, hi = blocks.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (blocks[mid].OutputStart <= offset) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return best >= 0 && offset < blocks[best].OutputStart + blocks[best].OutputLength ? best : -1;
    }

    /// <summary>Inline style of the range containing <paramref name="offset"/>, or None.</summary>
    public static InlineStyle StyleAt(IReadOnlyList<TextRange> ranges, int offset)
    {
        int lo = 0, hi = ranges.Count - 1, best = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (ranges[mid].OutputStart <= offset) { best = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        return best >= 0 && offset < ranges[best].OutputStart + ranges[best].OutputLength
            ? ranges[best].Style : InlineStyle.None;
    }
}
