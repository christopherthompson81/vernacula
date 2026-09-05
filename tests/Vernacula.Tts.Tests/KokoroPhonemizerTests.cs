using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The text → Kokoro-phonemes path and its phoneme-group → source-word map, through the real
/// phonemizer. Needs the vernacula-phonemizer submodule's data/ tree; skips without it.
/// </summary>
public class KokoroPhonemizerTests
{
    private static KokoroPhonemizer? TryCreate()
        => PhonemizerData.Resolve(null) is null ? null : new KokoroPhonemizer();

    [Fact]
    public void OneGroupPerSpokenWord_PunctuationAttached()
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        var r = g2p.Phonemize("Hello world. This is a test, isn't it?");
        // 8 source words → 8 groups; the three marks ride on their words, not as groups of their own.
        var groups = r.Phonemes.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(8, groups.Length);
        Assert.EndsWith(".", groups[1]);
        Assert.EndsWith(",", groups[5]);
        Assert.EndsWith("?", groups[7]);
        Assert.NotNull(r.GroupSourceWords);
        Assert.Equal(Enumerable.Range(0, 8), r.GroupSourceWords);
        foreach (var ch in r.Phonemes)
            Assert.True(ch == ' ' || KokoroVocab.Contains(ch), $"'{ch}' is not a Kokoro token");
    }

    [Fact]
    public void ExpandedNumbersCollapseOntoTheirWrittenWord()
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        // "$3.14" reads as several words; every one of them maps back to source word 4.
        var r = g2p.Phonemize("I paid about it $3.14 yesterday.");
        Assert.NotNull(r.GroupSourceWords);
        var groups = r.Phonemes.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(groups.Length, r.GroupSourceWords!.Count);
        var fromPrice = r.GroupSourceWords.Count(w => w == 4);
        Assert.True(fromPrice >= 2, $"expected the price to read as several groups, map: {string.Join(",", r.GroupSourceWords)}");
        Assert.Equal(5, r.GroupSourceWords[^1]);   // "yesterday." is the last word
        Assert.True(r.GroupSourceWords.Zip(r.GroupSourceWords.Skip(1)).All(p => p.First <= p.Second), "map is monotone");
    }

    [Fact]
    public void BritishUsesTheGbReading()
    {
        var g2p = TryCreate();
        if (g2p is null) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        Assert.Equal("ɡˈO hˈOm", g2p.ToPhonemes("go home"));
        Assert.Equal("ɡˈQ hˈQm", g2p.ToPhonemes("go home", british: true));
    }
}
