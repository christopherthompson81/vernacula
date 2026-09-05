using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// <see cref="KokoroFormat"/> over the IPA vernacula-phonemizer emits. The expected strings are what
/// misaki's espeak post-processing gives the same words (docs/kokoro_vphon_investigation.md, Run 2).
/// </summary>
public class KokoroFormatTests
{
    [Theory]
    // diphthongs, dark l, NURSE, detached punctuation
    [InlineData("həlˈoᶷ wˈɝɫd . ðɪs ɪz ə tʰˈɛst .", "həlˈO wˈɜɹld. ðɪs ɪz ə tˈɛst.")]
    // affricates with tie bars, flaps, aspiration
    [InlineData("t͡ʃˈɝt͡ʃ d͡ʒˈʌd͡ʒ bˈʌt̬ən hˈɪd̬ən", "ʧˈɜɹʧ ʤˈʌʤ bˈʌTən hˈɪdən")]
    // the five superscript offglides and the rhotic schwa
    [InlineData("lˈeᶦzi bɹˈaᶷn aᶦ pʰˈɔᶦnt oᶷvɚ", "lˈAzi bɹˈWn I pˈYnt Ovəɹ")]
    // en-us drops length marks; a palatal glide is dropped, not turned into j
    [InlineData("fˈɑːks θˈɔːt jɚˈeᶦniʲəm", "fˈɑks θˈɔt jəɹˈAniəm")]
    // ᵻ is a Kokoro token and survives; secondary stress survives
    [InlineData("ɹᵻmˈɛmbɚ jˈɛstɚd̬ˌeᶦz", "ɹᵻmˈɛmbəɹ jˈɛstəɹdˌAz")]
    // every clause mark attaches to the word before it
    [InlineData("wˈeᶦt , hiː sˈɛd . ˈɪzənt ɪt ?", "wˈAt, hi sˈɛd. ˈɪzənt ɪt?")]
    public void RendersAmerican(string ipa, string expected)
        => Assert.Equal(expected, KokoroFormat.Render(ipa));

    [Theory]
    // GOAT is Q, length marks stay, SQUARE is ɛː, NEAR stays ɪə
    [InlineData("həlˈəᶷ wˈɜːɫd . ðˈɛə hˈɪə", "həlˈQ wˈɜːld. ðˈɛː hˈɪə")]
    [InlineData("ɡˈəᶷ hˈəᶷm nˈaᶷ !", "ɡˈQ hˈQm nˈW!")]
    [InlineData("fˈaᶦə , ʃˈɛə , kjˈʊə .", "fˈIə, ʃˈɛː, kjˈʊə.")]
    public void RendersBritish(string ipa, string expected)
        => Assert.Equal(expected, KokoroFormat.Render(ipa, british: true));

    [Theory]
    [InlineData("mˈɪstɚ smˈɪθ ɚˈaᶦvd æt tʰˈɛn θˈɝd̬iː ˈeᶦ ˈɛm ˈɑːn tʰˈuːzdi , mˈɑːɹt͡ʃ θˈɝd , twˈɛnti twˈɛnti fˈɔːɹ .", false)]
    [InlineData("jˈɛstədˌeᶦz wˈɛðə wˈɒz bˈɛtə ðæn tədˈeᶦz , wˈɒzənt ɪt ?", true)]
    [InlineData("sˈɪŋɪŋ , θˈɪŋkɪŋ , lˈɛŋkθ , ðə kʰˈɪŋz ɹˈɪŋ .", false)]
    public void EveryOutputCodepointIsInTheVocab(string ipa, bool british)
    {
        var ps = KokoroFormat.Render(ipa, british);
        foreach (var ch in ps)
            Assert.True(KokoroVocab.Contains(ch), $"'{ch}' (U+{(int)ch:X4}) is not a Kokoro token, in: {ps}");
        Assert.Equal(ps.Length, KokoroVocab.Encode(ps).Length - 2);   // nothing was dropped
    }

    [Fact]
    public void EmptyIsEmpty()
    {
        Assert.Equal("", KokoroFormat.Render(""));
        Assert.Equal("", KokoroFormat.Render(null!));
    }
}
