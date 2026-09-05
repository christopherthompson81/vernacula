using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The reader's estimated word map for OmniVoice: one entry per whitespace-delimited source word,
/// contiguous, covering the chunk, weighted by the IPA each word became (from the phonemizer's
/// trace spans). Needs the vernacula-phonemizer submodule's data/ tree; skips without it.
/// </summary>
public class OmniVoiceIpaAlignmentTests
{
    private static bool HavePhonemizer()
    {
        if (PhonemizerData.Resolve(null) is null) return false;
        Vernacula.Phonemizer.Registry.EnsureLanguages();
        return true;
    }

    [Fact]
    public void OneWordPerSourceWord_ContiguousAndCoveringTheChunk()
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        const string text = "The quick brown fox jumps over the lazy dog, doesn't it?";
        var trace = Vernacula.Phonemizer.Phonemizer.PhonemizeTrace(text, "en");
        var words = OmniVoiceIpaAlignment.Proportional(text, trace, totalSec: 4.0);

        Assert.Equal(11, words.Count);
        Assert.Equal(0.0, words[0].StartSec, 9);
        Assert.Equal(4.0, words[^1].EndSec, 6);
        for (var i = 1; i < words.Count; i++)
            Assert.Equal(words[i - 1].EndSec, words[i].StartSec, 9);
        // "jumps" (d͡ʒˈʌmps) should get more time than "the" (ðə); weights are IPA, not spelling.
        double Dur(string w) => words.First(x => x.Text == w).EndSec - words.First(x => x.Text == w).StartSec;
        Assert.True(Dur("jumps") > Dur("the"), $"jumps={Dur("jumps"):F3} the={Dur("the"):F3}");
    }

    [Fact]
    public void ExpandedNumberWeightsItsWrittenWord()
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");

        const string text = "It cost $3.14 today.";
        var trace = Vernacula.Phonemizer.Phonemizer.PhonemizeTrace(text, "en");
        var words = OmniVoiceIpaAlignment.Proportional(text, trace, totalSec: 3.0);

        Assert.Equal(4, words.Count);
        // "$3.14" reads as three spoken words; it should own the largest share of the chunk.
        var price = words[2];
        Assert.Equal("$3.14", price.Text);
        var priceDur = price.EndSec - price.StartSec;
        foreach (var w in words.Where(w => w != price))
            Assert.True(priceDur > w.EndSec - w.StartSec, $"{w.Text} outweighs the price");
    }

    [Fact]
    public void UntracedFallsBackToCharacterLengths()
    {
        var trace = new Vernacula.Phonemizer.PhonemeTrace { Ipa = "", Traced = false };
        var words = OmniVoiceIpaAlignment.Proportional("ab abcdef", trace, totalSec: 8.0);
        Assert.Equal(2, words.Count);
        Assert.Equal(2.0, words[0].EndSec, 6);
        Assert.Equal(8.0, words[1].EndSec, 6);
    }
}
