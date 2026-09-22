using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The chunker two engines now share, and the vocabulary filter that keeps them saying the same
/// thing. Both moved out of the ONNX path when audio.cpp's Kokoro began taking a supplied phoneme
/// stream: the ONNX graph needs the cut because its position embedding fails above the window,
/// audio.cpp needs it because a caller's phonemes arrive as a list of chunks, and if the two ever
/// cut differently the same document reads differently on the two backends.
/// </summary>
public class KokoroChunkerTests
{
    private static KokoroChunker Chunker()
    {
        if (PhonemizerData.Resolve(null) is null)
            Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        return new KokoroChunker(new KokoroPhonemizer());
    }

    [Fact]
    public void ChunkingSplitsAtWhitespaceAndLosesNoWord()
    {
        // The property every caller of this leans on: the chunks' words, concatenated, ARE the
        // source words in order. Word-level alignment indexes into the source split, so a chunker
        // that dropped or merged a word would misalign every word after it and report nothing.
        var text = string.Join(' ', Enumerable.Repeat(
            "The harbour was quiet this morning, and the boats had not yet returned.", 20));

        var chunks = Chunker().ChunkForSynthesis(text);

        Assert.True(chunks.Count > 1, "a 20-sentence paragraph did not split");
        Assert.Equal(
            text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries),
            chunks.SelectMany(c => c.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)));
    }

    [Fact]
    public void EveryChunkFitsKokorosContextWindow()
    {
        var chunker = Chunker();
        var text = string.Join(' ', Enumerable.Repeat("antidisestablishmentarianism", 400));

        foreach (var chunk in chunker.ChunkForSynthesis(text))
            Assert.InRange(chunker.CountTokens(chunk), 1, 510);
    }

    [Fact]
    public void AShortParagraphIsOneChunk()
    {
        Assert.Single(Chunker().ChunkForSynthesis("The harbour was quiet this morning."));
    }

    [Fact]
    public void ALongJapaneseParagraphIsSplitWithinTheEngineBudget()
    {
        // ⚠ THE BUDGET IS IN PHONEMES AND THE LANGUAGE DECIDES HOW MANY. Measuring a Japanese
        // paragraph with the English G2P returns a number unrelated to the ja render that is
        // actually sent, so a chunk could be handed over far past the engine's 510-symbol entry
        // cap — and the engine refuses the whole request rather than degrading. Worse, the
        // sentence splitter only knew ASCII terminators, so a 。-terminated paragraph never split
        // at all and a script without spaces has one "word" to fall back to.
        var chunker = Chunker();
        var paragraph = string.Concat(Enumerable.Repeat("今日はいい天気ですね。科学者たちが発表しました。", 30));

        var chunks = chunker.ChunkForSynthesis(paragraph, "ja");

        Assert.True(chunks.Count > 1, "a 30-sentence Japanese paragraph came back as one chunk");
        foreach (var chunk in chunks)
            Assert.InRange(chunker.CountTokens(chunk, "ja"), 1, 510);
    }

    [Fact]
    public void AMandarinParagraphWithNoSpacesAtAllStillSplits()
    {
        // The same, with no sentence marks either: the last resort is characters, which is the
        // only cut a script without spaces leaves. Entries are merged back into one buffer, so a
        // seam costs prosody rather than correctness.
        var chunker = Chunker();
        var paragraph = string.Concat(Enumerable.Repeat("今天天气很好我们去公园散步", 60));

        var chunks = chunker.ChunkForSynthesis(paragraph, "cmn");

        Assert.True(chunks.Count > 1, "a paragraph with no spaces and no sentence marks did not split");
        foreach (var chunk in chunks)
            Assert.InRange(chunker.CountTokens(chunk, "cmn"), 1, 510);
    }

    [Fact]
    public void KeepKnownDropsOnlyWhatKokoroHasNoTokenFor()
    {
        // ⚠ THE TWO ENGINES DISAGREE ABOUT AN UNKNOWN SYMBOL AND BOTH ARE RIGHT. The ONNX path
        // never sees one (Encode skips it, as misaki/KModel does); audio.cpp REFUSES a supplied
        // stream carrying one, because a caller with its own G2P can correct what it sent. This
        // filter is what lets one stream satisfy both.
        var kept = KokoroVocab.KeepKnown("hɛlˈOʘ wˈɜɹld", out var dropped);

        Assert.Equal("hɛlˈO wˈɜɹld", kept);
        Assert.Equal(['ʘ'], dropped);
        foreach (var ch in kept) Assert.True(KokoroVocab.Contains(ch));
    }

    [Fact]
    public void KeepKnownLeavesACleanStreamUntouched()
    {
        const string clean = "hɛlˈO wˈɜɹld.";
        Assert.Same(clean, KokoroVocab.KeepKnown(clean, out var dropped));
        Assert.Empty(dropped);
    }
}
