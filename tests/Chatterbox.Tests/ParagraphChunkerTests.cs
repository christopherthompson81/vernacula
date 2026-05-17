using Chatterbox.Base.Markdown;
using Xunit;

namespace Chatterbox.Tests;

public class ParagraphChunkerTests
{
    [Fact]
    public void Short_text_returns_single_chunk()
    {
        var chunks = ParagraphChunker.Chunk("Hello world.");
        Assert.Single(chunks);
        Assert.Equal("Hello world.", chunks[0]);
    }

    [Fact]
    public void Empty_input_returns_empty_list()
    {
        Assert.Empty(ParagraphChunker.Chunk(""));
        Assert.Empty(ParagraphChunker.Chunk("   \n\n   "));
    }

    [Fact]
    public void Short_multi_paragraph_text_stays_single_chunk()
    {
        // Under MinCharsForChunking (200) → don't bother chunking even if
        // there are paragraph breaks. Lets the existing one-shot path
        // handle short multi-paragraph inputs without orchestration overhead.
        var input = "Para one.\n\nPara two.";
        var chunks = ParagraphChunker.Chunk(input);
        Assert.Single(chunks);
        Assert.Equal(input, chunks[0]);
    }

    [Fact]
    public void Long_multi_paragraph_text_splits_per_paragraph()
    {
        // Two paragraphs, total > MinCharsForChunking, each ≤ MaxCharsPerChunk.
        var p1 = new string('a', 150) + ".";  // 151 chars
        var p2 = new string('b', 150) + ".";
        var input = $"{p1}\n\n{p2}";
        var chunks = ParagraphChunker.Chunk(input);
        Assert.Equal(2, chunks.Count);
        Assert.Equal(p1, chunks[0]);
        Assert.Equal(p2, chunks[1]);
    }

    [Fact]
    public void Multiple_blank_lines_collapse_to_one_break()
    {
        var p1 = new string('a', 150) + ".";
        var p2 = new string('b', 150) + ".";
        var input = $"{p1}\n\n\n\n{p2}";
        var chunks = ParagraphChunker.Chunk(input);
        Assert.Equal(2, chunks.Count);
    }

    [Fact]
    public void Single_paragraph_with_internal_newlines_stays_one_chunk()
    {
        // List items joined by single \n (extractor's list emission) — should
        // NOT be split. Only \n\n is a chunk boundary.
        var input = "Long paragraph that exceeds the threshold. " + new string('x', 200);
        var chunks = ParagraphChunker.Chunk(input);
        Assert.Single(chunks);
    }

    [Fact]
    public void Over_long_paragraph_splits_on_sentence_boundaries()
    {
        // Paragraph > MaxCharsPerChunk (600) — should sub-split.
        var sentences = string.Join(" ", Enumerable.Range(1, 20).Select(i =>
            $"This is sentence number {i} with some filler text to bulk it out a bit."));
        // sentences is ~1500 chars in one paragraph, no \n\n inside.
        Assert.True(sentences.Length > 600);
        var chunks = ParagraphChunker.Chunk(sentences);
        Assert.True(chunks.Count > 1, $"expected multiple chunks for {sentences.Length} chars, got {chunks.Count}");
        foreach (var c in chunks)
            Assert.True(c.Length <= 600 || !c.Contains(". "),
                $"chunk over 600 chars and containing splittable sentence boundary: '{c[..Math.Min(80, c.Length)]}...'");
        // Loss-free invariant: joining the chunks with " " reconstructs the
        // input (PackSentences only does .Trim() per sentence and rejoins
        // with " ", so an input that was itself " "-joined round-trips).
        Assert.Equal(sentences, string.Join(" ", chunks));
    }

    // ── Line-ending handling ──────────────────────────────────────────

    [Fact]
    public void CRLF_paragraph_breaks_are_recognized()
    {
        // Windows-line-ended input via --text or a CRLF-saved file should
        // chunk the same as Unix line endings. Regression test for the
        // ParagraphSplit regex.
        var p1 = "First paragraph. " + new string('a', 200);
        var p2 = "Second paragraph. " + new string('b', 200);
        var unix = $"{p1}\n\n{p2}";
        var crlf = $"{p1}\r\n\r\n{p2}";
        var unixChunks = ParagraphChunker.Chunk(unix);
        var crlfChunks = ParagraphChunker.Chunk(crlf);
        Assert.Equal(2, unixChunks.Count);
        Assert.Equal(2, crlfChunks.Count);
        Assert.Equal(unixChunks, crlfChunks);
    }

    [Fact]
    public void Chunks_preserve_content_when_concatenated()
    {
        // For a paragraph-split case (no sub-sentence splits triggered), the
        // chunks joined back with "\n\n" should reconstruct the input modulo
        // trim. This is a loose round-trip invariant.
        var p1 = "First paragraph. " + new string('a', 100);
        var p2 = "Second paragraph. " + new string('b', 100);
        var p3 = "Third paragraph. " + new string('c', 100);
        var input = $"{p1}\n\n{p2}\n\n{p3}";
        var chunks = ParagraphChunker.Chunk(input);
        Assert.Equal(3, chunks.Count);
        Assert.Equal(input, string.Join("\n\n", chunks));
    }

    [Fact]
    public void Single_sentence_longer_than_cap_stays_one_chunk()
    {
        // No sentence terminator to split on; chunker doesn't mid-sentence-cut.
        // LM's maxSteps will truncate the audio if this really is too long.
        var oneSentence = "This is a single sentence with no terminator " + new string('x', 800);
        var chunks = ParagraphChunker.Chunk(oneSentence);
        Assert.Single(chunks);
    }

    [Fact]
    public void Trims_whitespace_around_chunks()
    {
        var p1 = "First paragraph. " + new string('a', 200);
        var p2 = "Second paragraph. " + new string('b', 200);
        var input = $"   {p1}   \n\n   {p2}   ";
        var chunks = ParagraphChunker.Chunk(input);
        Assert.Equal(2, chunks.Count);
        Assert.Equal(p1, chunks[0]);
        Assert.Equal(p2, chunks[1]);
    }
}
