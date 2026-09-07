using System.Text;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Per-token character offsets, which decide which tokens a speaker turn claims when a chunk
/// is split. The tokenizer is byte-level, so a token boundary lands inside a multi-byte
/// character routinely for the CJK languages this model covers — byte offsets would silently
/// attribute the wrong tokens, and the confidence colours in the editor would follow.
/// </summary>
public class VibeVoiceStreamingOffsetTests
{
    private static (byte[] utf8, int[] byteEnds) Split(string text, params int[] byteCuts)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        var ends = new int[byteCuts.Length + 1];
        for (int i = 0; i < byteCuts.Length; i++) ends[i] = byteCuts[i];
        ends[^1] = utf8.Length;
        return (utf8, ends);
    }

    [Fact]
    public void AsciiOffsetsMatchByteOffsets()
    {
        var (utf8, ends) = Split("abcd", 1, 3);
        Assert.Equal([1, 3, 4], VibeVoiceStreamingAsr.CharEndsFromByteEnds(utf8, ends));
    }

    [Fact]
    public void MultiByteCharactersCountAsOne()
    {
        // "发布" is 2 characters, 6 bytes. Cutting cleanly between them must give 1 then 2.
        var (utf8, ends) = Split("发布", 3);
        Assert.Equal(6, utf8.Length);
        Assert.Equal([1, 2], VibeVoiceStreamingAsr.CharEndsFromByteEnds(utf8, ends));
    }

    [Fact]
    public void ATokenBoundaryInsideACharacterDoesNotDoubleCountIt()
    {
        // The byte-level tokenizer can end a token mid-character. The character must be
        // counted once, when it completes — not once per token that touched its bytes.
        var (utf8, ends) = Split("发布", 1, 3, 4);   // cuts inside both characters
        int[] chars = VibeVoiceStreamingAsr.CharEndsFromByteEnds(utf8, ends);
        Assert.Equal([0, 1, 1, 2], chars);
        // Offsets never go backwards, and end at the true character count.
        Assert.Equal("发布".Length, chars[^1]);
        for (int i = 1; i < chars.Length; i++) Assert.True(chars[i] >= chars[i - 1]);
    }

    [Fact]
    public void MixedScriptsEndAtTheRealCharacterCount()
    {
        const string text = "ok 发布 done";
        var (utf8, ends) = Split(text, 2, 5, 9, 12);
        int[] chars = VibeVoiceStreamingAsr.CharEndsFromByteEnds(utf8, ends);
        Assert.Equal(text.Length, chars[^1]);
        Assert.True(chars[^1] < utf8.Length, "characters should be fewer than bytes here");
    }
}
