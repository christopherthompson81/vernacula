using Xunit;
using Vernacula.Tts.Base;

namespace Vernacula.Tts.Tests;

public class TextDirectionTests
{
    [Theory]
    [InlineData("سلام دنیا")]                       // Persian
    [InlineData("مرحبا بالعالم")]                   // Arabic
    [InlineData("שלום עולם")]                       // Hebrew
    [InlineData("السلام علیکم")]                    // Urdu
    [InlineData("ދިވެހި")]                            // Dhivehi (Thaana)
    public void RightToLeftScriptsReadRightToLeft(string text)
        => Assert.True(TextDirection.IsRightToLeft(text));

    [Theory]
    [InlineData("Hello world")]
    [InlineData("Bonjour le monde")]
    [InlineData("日本語を勉強しています")]
    [InlineData("Привет мир")]
    [InlineData("")]
    [InlineData(null)]
    public void EverythingElseReadsLeftToRight(string? text)
        => Assert.False(TextDirection.IsRightToLeft(text));

    [Fact]
    public void DigitsAndPunctuationDoNotDecide()
    {
        // Weak and neutral characters take direction from their surroundings; a phone number or a
        // pile of punctuation must not flip a paragraph either way.
        Assert.False(TextDirection.IsRightToLeft("+1 (555) 010-9999 -- ..."));
        Assert.True(TextDirection.IsRightToLeft("شماره ۰۲۱ ۱۲۳۴۵۶۷ است"));
    }

    [Fact]
    public void TheReadingLanguageSettlesAMixedBlock()
    {
        // Counting characters cannot settle these: the English half of the first has more letters
        // than the Persian half, and the second opens with a Latin brand name. The language the
        // document is being READ in is the one thing that knows, so it breaks the tie.
        const string embeddedEnglish = "برنامه Text To Speech است";
        const string leadingLatin = "Vernacula یک برنامه متن به گفتار است";

        Assert.True(TextDirection.Resolve(embeddedEnglish, languageIsRtl: true));
        Assert.True(TextDirection.Resolve(leadingLatin, languageIsRtl: true));
        Assert.False(TextDirection.Resolve(embeddedEnglish, languageIsRtl: false));
    }

    [Fact]
    public void TextOfOneDirectionAnswersForItselfWhateverTheLanguageSays()
    {
        // A quoted English paragraph inside a Persian document is still English, and laying it out
        // right-to-left because the reader picked Persian would be wrong.
        Assert.False(TextDirection.Resolve("The sign above the door", languageIsRtl: true));
        Assert.True(TextDirection.Resolve("مرحبا بالعالم", languageIsRtl: false));
    }

    [Fact]
    public void WithoutALanguageTheFirstStrongCharacterDecidesAMixedBlock()
    {
        // The dir="auto" rule, used only as a fallback -- the reader always has a language.
        Assert.True(TextDirection.IsRightToLeft("سلام Text To Speech"));
        Assert.False(TextDirection.IsRightToLeft("Hello سلام دنیا"));
    }

    [Theory]
    [InlineData("fa", true)]
    [InlineData("ar", true)]
    [InlineData("he", true)]
    [InlineData("ur", true)]
    [InlineData("ps", true)]
    [InlineData("ckb", true)]
    [InlineData("en", false)]
    [InlineData("ja", false)]
    [InlineData("ru", false)]
    [InlineData("not-a-language", false)]
    public void ALanguagesDirectionComesFromItsOwnEndonym(string code, bool rtl)
        => Assert.Equal(rtl, LanguageCatalog.IsRightToLeft(code));

    [Fact]
    public void AnRtlQuotationInsideEnglishStaysWithItsOwnBlock()
    {
        // Direction is decided per block, so the English narration and the quoted line differ.
        Assert.False(TextDirection.IsRightToLeft("The sign above the door read as follows"));
        Assert.True(TextDirection.IsRightToLeft("مرحبا بالعالم"));
    }
}
