using Xunit;
using Vernacula.Tts.Base;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The word-level half of bidirectional layout. A panel mirrors every child, which is right for
/// the block's own script and wrong for a phrase embedded from the other direction, so embedded
/// runs are reversed before the panel mirrors them. This tests the classification the reader's
/// block view model builds that ordering from.
/// </summary>
public class BidiRunTests
{
    [Theory]
    [InlineData("سلام", true)]
    [InlineData("שלום", true)]
    [InlineData("Speech", false)]
    [InlineData("Привет", false)]
    public void StrongWordsReportTheirOwnDirection(string word, bool rtl)
        => Assert.Equal(rtl, TextDirection.StrongDirectionOf(word));

    [Theory]
    [InlineData("15")]
    [InlineData("۱۲۳")]
    [InlineData("3.14")]
    public void NumbersAreNumbers(string word)
    {
        // A number after a left-to-right word stays with it ("iPhone 15", not "15 iPhone"), which
        // is why it is classed apart from punctuation even though neither is strong.
        Assert.True(TextDirection.IsNumberWord(word));
        Assert.Null(TextDirection.StrongDirectionOf(word));
    }

    [Theory]
    [InlineData("—")]
    [InlineData("&")]
    [InlineData("iPhone")]
    [InlineData("سلام")]
    [InlineData("")]
    [InlineData(null)]
    public void EverythingElseIsNotANumber(string? word)
        => Assert.False(TextDirection.IsNumberWord(word));

    [Theory]
    [InlineData("123")]
    [InlineData("—")]
    [InlineData("...")]
    [InlineData("۱۲۳")]   // Persian digits are weak, not strong RTL
    [InlineData("")]
    [InlineData(null)]
    public void NeutralWordsForceNothing(string? word)
        => Assert.Null(TextDirection.StrongDirectionOf(word));

    [Theory]
    [InlineData("،")]      // Arabic comma
    [InlineData("؟")]      // Arabic question mark
    [InlineData("؛")]      // Arabic semicolon
    public void ArabicPunctuationIsPunctuation(string word)
    {
        // These live in the Arabic block but are neutrals. Counting them as strong made an English
        // block containing one look "mixed", so it stopped answering for itself.
        Assert.Null(TextDirection.StrongDirectionOf(word));
        var census = TextDirection.Census("Hello world " + word);
        Assert.False(census.HasRtl);
    }

    [Fact]
    public void HebrewPointingIsNotAWordOfItsOwn()
    {
        // Combining marks are not letters; the letters they sit on are what count.
        Assert.Null(TextDirection.StrongDirectionOf("\u05B7\u05B8"));
        Assert.True(TextDirection.StrongDirectionOf("שָׁלוֹם"));
    }

    [Fact]
    public void ADirectionIsTakenFromTheFirstStrongCharacterOfTheWord()
    {
        // Mixed within one word: "1990م" is Arabic, "v2سلام" leads Latin. The word is one item in
        // the layout either way, so its own first strong character decides where it belongs.
        Assert.True(TextDirection.StrongDirectionOf("1990م"));
        Assert.False(TextDirection.StrongDirectionOf("v2سلام"));
    }
}
