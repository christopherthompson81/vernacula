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
    [InlineData("123")]
    [InlineData("—")]
    [InlineData("...")]
    [InlineData("۱۲۳")]   // Persian digits are weak, not strong RTL
    [InlineData("")]
    [InlineData(null)]
    public void NeutralWordsForceNothing(string? word)
        => Assert.Null(TextDirection.StrongDirectionOf(word));

    [Fact]
    public void ADirectionIsTakenFromTheFirstStrongCharacterOfTheWord()
    {
        // Mixed within one word: "1990م" is Arabic, "v2سلام" leads Latin. The word is one item in
        // the layout either way, so its own first strong character decides where it belongs.
        Assert.True(TextDirection.StrongDirectionOf("1990م"));
        Assert.False(TextDirection.StrongDirectionOf("v2سلام"));
    }
}
