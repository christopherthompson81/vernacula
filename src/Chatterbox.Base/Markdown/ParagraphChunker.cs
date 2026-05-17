using System.Text.RegularExpressions;

namespace Chatterbox.Base.Markdown;

/// <summary>
/// Splits speakable text into chunks at paragraph boundaries for the
/// long-form synthesis path (Stage 1 #8).
///
/// Pairs with <see cref="MarkdownTextExtractor"/> — the extractor emits
/// <c>\n\n</c> between paragraphs precisely so this chunker has a clean
/// boundary. List items separated by single <c>\n</c> stay grouped with
/// their surrounding paragraph (a bulleted list is one chunk, not N).
///
/// The chunker doesn't know about tokens — it works on character count
/// against a coarse heuristic upper bound. The real per-chunk limit
/// comes from <see cref="AcousticLM"/>'s <c>maxSteps</c> (256 default,
/// ~10 s of audio). A typical paragraph fits comfortably; a very long
/// single paragraph that exceeds <see cref="MaxCharsPerChunk"/> gets
/// split on sentence boundaries as a fallback.
/// </summary>
public static class ParagraphChunker
{
    /// <summary>
    /// Default upper bound on chunk size in characters. Sized so a
    /// typical paragraph at this length tokenizes well below the LM's
    /// 256-step rollout cap (rough rule: 1 LM step ≈ 4-6 source chars
    /// of English prose at the default exaggeration). 600 chars is
    /// well inside that with margin.
    /// </summary>
    public const int MaxCharsPerChunk = 600;

    /// <summary>
    /// Don't bother chunking text shorter than this — single-shot
    /// synthesis is cheaper than the pipeline orchestration overhead
    /// for trivial inputs.
    /// </summary>
    public const int MinCharsForChunking = 200;

    // \r?\n followed by optional whitespace + \r?\n, repeated. Handles
    // both Unix (\n\n) and Windows (\r\n\r\n) line endings.
    // MarkdownTextExtractor only emits \n\n, but --text and other input
    // sources can carry CRLF — the chunker shouldn't silently fail on
    // Windows-line-ended input.
    private static readonly Regex ParagraphSplit = new(@"\r?\n[ \t\r]*\n+", RegexOptions.Compiled);

    // Sentence-end split for the over-long-paragraph fallback. Looks for
    // .!? followed by whitespace; conservative (won't split on
    // abbreviations like "Dr." since we'd need full sentence tokenization
    // for that — acceptable for the fallback case).
    private static readonly Regex SentenceSplit = new(@"(?<=[.!?])\s+", RegexOptions.Compiled);

    /// <summary>
    /// Split <paramref name="text"/> into chunks at paragraph boundaries.
    /// Returns the input as a single chunk if it has no paragraph breaks
    /// or is shorter than <see cref="MinCharsForChunking"/>.
    /// </summary>
    /// <param name="maxCharsPerChunk">Per-chunk cap. Paragraphs longer than
    /// this are sub-split on sentence boundaries.</param>
    public static List<string> Chunk(string text, int maxCharsPerChunk = MaxCharsPerChunk)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new List<string>();

        var trimmed = text.Trim();
        if (trimmed.Length < MinCharsForChunking)
            return new List<string> { trimmed };

        var paragraphs = ParagraphSplit.Split(trimmed)
            .Select(p => p.Trim())
            .Where(p => p.Length > 0)
            .ToList();
        if (paragraphs.Count == 0)
            return new List<string>();

        var chunks = new List<string>();
        foreach (var p in paragraphs)
        {
            if (p.Length <= maxCharsPerChunk)
            {
                chunks.Add(p);
            }
            else
            {
                // Over-long paragraph: split on sentence boundaries, then
                // greedily pack sentences into ≤maxCharsPerChunk chunks.
                // A single sentence longer than the cap stays as its own
                // chunk (we don't split mid-sentence; the LM's maxSteps
                // will truncate the audio if it really is too long).
                foreach (var sub in PackSentences(p, maxCharsPerChunk))
                    chunks.Add(sub);
            }
        }
        return chunks;
    }

    private static IEnumerable<string> PackSentences(string paragraph, int maxChars)
    {
        var sentences = SentenceSplit.Split(paragraph);
        var buffer = new System.Text.StringBuilder();
        foreach (var s in sentences)
        {
            var sent = s.Trim();
            if (sent.Length == 0) continue;
            // +1 for the space we'll insert between sentences in the buffer.
            int projected = buffer.Length == 0 ? sent.Length : buffer.Length + 1 + sent.Length;
            if (projected > maxChars && buffer.Length > 0)
            {
                yield return buffer.ToString();
                buffer.Clear();
            }
            if (buffer.Length > 0) buffer.Append(' ');
            buffer.Append(sent);
        }
        if (buffer.Length > 0) yield return buffer.ToString();
    }
}
