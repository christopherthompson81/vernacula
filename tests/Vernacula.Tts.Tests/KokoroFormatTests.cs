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
    // A word-FINAL unstressed vowel after a flap takes the tap ɾ, not T: Kokoro's duration predictor
    // over-allocates T there and the vowel detaches from the word.
    [InlineData("ðə dˈeᶦt̬ə ɹᵻkwˈaᶦɚd", "ðə dˈAɾə ɹᵻkwˈIəɹd")]
    // …including before a clause mark, which the punctuation pass has already attached by then.
    [InlineData("ðə bˈeᶦt̬ə , ðə d̬ˈeᶦt̬ə .", "ðə bˈAɾə, ðə dˈAɾə.")]
    // the -ity/-y family is the bulk of the rule's scope
    [InlineData("ðə sˈɪt̬i hæz kwˈɑːlᵻt̬i", "ðə sˈɪɾi hæz kwˈɑlᵻɾi")]
    // ⚠ word-INTERNAL flaps keep T — it is what preserves the underlying /t/, so writer stays
    // distinct from rider. Mapping these to d would merge the pair at the token level.
    [InlineData("ɹˈaᶦt̬ɚ ɹˈaᶦd̬ɚ lˈæt̬ɚ lˈæd̬ɚ", "ɹˈITəɹ ɹˈIdəɹ lˈæTəɹ lˈædəɹ")]
    [InlineData("mˈiːt̬ɪŋ bˈɛt̬ɚ ɹᵻlˈeᶦt̬ᵻd lˈɪmᵻt̬ᵻd", "mˈiTɪŋ bˈɛTəɹ ɹᵻlˈATᵻd lˈɪmᵻTᵻd")]
    public void WordFinalFlapBecomesATap(string ipa, string expected)
        => Assert.Equal(expected, KokoroFormat.Render(ipa));

    [Fact]
    public void TheTapIsAKokoroToken()
    {
        var ps = KokoroFormat.Render("ðə dˈeᶦt̬ə");
        Assert.Contains('ɾ', ps);
        foreach (var ch in ps) Assert.True(KokoroVocab.Contains(ch), $"'{ch}' is not a Kokoro token");
    }

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

    // ── Languages other than English ─────────────────────────────────────────

    /// <summary>
    /// The English path must not move. It is correct and byte-for-byte verified against the
    /// goldens, so the language-aware overload has to route to exactly it — any drift here is a
    /// regression in the only language that currently ships.
    /// </summary>
    [Theory]
    [InlineData("ðə dˈeᶦt̬ə", "en")]
    [InlineData("hɛlˈoᶷ wˈɜɹld", "en")]
    [InlineData("ðə dˈeᶦt̬ə", "en-GB")]
    public void TheLanguageAwareOverloadRoutesEnglishToTheEnglishPath(string ipa, string lang)
        => Assert.Equal(KokoroFormat.Render(ipa, british: lang == "en-GB"), KokoroFormat.Render(ipa, lang));

    /// <summary>
    /// ⚠ THESE ARE CONTRASTS, NOT PREFERENCES. Each assertion below is a pair of sounds the
    /// language distinguishes and English does not, which the English rules used to collapse —
    /// and every symbol involved is one Kokoro's vocabulary already carries, so the collapse
    /// bought nothing. Restoring "consistency" with the English table would silently merge real
    /// words again.
    /// </summary>
    [Theory]
    // Spanish: the tap/trill contrast (pero "but" vs perro "dog"), and the jota.
    [InlineData("es", "pˈeɾo", "pˈeɾo")]
    [InlineData("es", "pˈero", "pˈero")]
    [InlineData("es", "xamˈon", "xamˈon")]
    // Italian: the trill, which r→ɹ used to flatten into an English approximant.
    [InlineData("it", "kˈorre", "kˈorre")]
    // French: nasal vowels, which the tilde strip used to delete outright.
    [InlineData("fr", "bɔ̃", "bɔ̃")]
    // Portuguese: avô /o/ against avó /ɔ/, which the blanket o→ɔ used to merge.
    [InlineData("pt-BR", "avˈo", "avˈo")]
    [InlineData("pt-BR", "avˈɔ", "avˈɔ")]
    // Hindi: aspiration is phonemic here, unlike English, so it must survive; vowel length too.
    [InlineData("hi", "kʰaː", "kʰaː")]
    public void ANonEnglishContrastSurvivesRendering(string lang, string ipa, string expected)
        => Assert.Equal(expected, KokoroFormat.Render(ipa, lang));

    /// <summary>
    /// Hindi writes ह voiced and marks breathy voice and dental place. Kokoro's alphabet holds
    /// none of the three, so these collapse to the nearest it does hold — the one case where
    /// losing a distinction is the vocabulary's limit rather than an English habit.
    /// </summary>
    [Theory]
    [InlineData("ɦɛ", "hɛ")]
    [InlineData("ɡʱoʃ", "ɡʰoʃ")]
    [InlineData("sˈəkt̪a", "sˈəkta")]
    public void HindiCollapsesWhatKokorosAlphabetCannotCarry(string ipa, string expected)
        => Assert.Equal(expected, KokoroFormat.Render(ipa, "hi"));

    /// <summary>
    /// Portuguese nasal vowels are precomposed in our IPA and decomposed in Kokoro's vocabulary,
    /// which carries the base vowel plus U+0303. Same sound, two spellings — so it decomposes
    /// rather than dropping the character, which would delete either the nasality or the syllable.
    /// </summary>
    [Theory]
    [InlineData("õ")]
    [InlineData("ĩ")]
    [InlineData("ũ")]
    [InlineData("ẽ")]
    public void APrecomposedNasalVowelDecomposesIntoSymbolsKokoroHas(string ipa)
    {
        var rendered = KokoroFormat.Render(ipa, "pt-BR");
        Assert.All(rendered, c => Assert.True(KokoroVocab.Contains(c), $"U+{(int)c:X4} is not in the vocabulary"));
        Assert.Contains('\u0303', rendered);   // the nasality survived rather than being dropped
    }
}
