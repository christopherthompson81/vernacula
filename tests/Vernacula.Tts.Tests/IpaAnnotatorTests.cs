using Xunit;
using Vernacula.Tts.Base;

namespace Vernacula.Tts.Tests;

public class IpaAnnotatorTests
{
    private static bool HavePhonemizer() => PhonemizerData.Resolve(null) is not null;

    private static IReadOnlyList<string> Annotate(params string[] words)
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        var got = IpaAnnotator.Annotate(words, "en");
        Assert.NotNull(got);
        return got!.Select(w => w.Ipa).ToList();
    }

    private static IReadOnlyList<WordRuby> Ruby(string word, string lang)
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        var got = IpaAnnotator.Annotate(new[] { word }, lang);
        Assert.NotNull(got);
        return got!;
    }

    [Fact]
    public void OneEntryPerWord()
    {
        var ipa = Annotate("hello", "world");
        Assert.Equal(2, ipa.Count);
        Assert.All(ipa, s => Assert.NotEqual("", s));
    }

    [Fact]
    public void AnnotationIsThatWordsReading()
    {
        // Each word's annotation is its own reading, not a slice of the neighbour's.
        var ipa = Annotate("the", "cat", "sat");
        Assert.Contains("æ", ipa[1]);
        Assert.DoesNotContain("æ", ipa[0]);
    }

    [Fact]
    public void ExpandedNumberStaysOverItsOwnWord()
    {
        // "$3.14" reads as several words but is written as one: all of it annotates that word,
        // and the words around it keep their own readings.
        var ipa = Annotate("costs", "$3.14", "today");
        Assert.Equal(3, ipa.Count);
        Assert.Contains(" ", ipa[1].Trim());          // several groups stacked over the one word
        Assert.NotEqual("", ipa[2]);
    }

    [Fact]
    public void SpanCoveringSeveralWordsIsNotDistributedUnlessItLinesUp()
    {
        // "$ 3.14" is one rewritten span whose three spoken tokens do not correspond one for one
        // with its two written words. Handing them out in order put "three" over "$" and
        // "dollars fourteen" over "3.14" -- a reading over the wrong word. The group stacks over
        // the span's first word instead.
        var ipa = Annotate("costs", "$", "3.14");
        Assert.Equal(3, ipa.Count);
        Assert.Contains("dˈɑːlɚz", ipa[1]);
        Assert.Equal("", ipa[2]);
    }

    [Fact]
    public void OneTokenPerWordInASpanStillFollowsTheWords()
    {
        // The shape that does line up: two tokens, two written words.
        var ipa = Annotate("Mr.", "Smith");
        Assert.Contains("mˈɪst", ipa[0]);
        Assert.Contains("smˈɪθ", ipa[1]);
    }

    [Theory]
    [InlineData("fr-CA")]   // applies an accent after assembly, so IpaSpan is withheld
    [InlineData("as")]      // collapses a doubled aspirate, likewise
    [InlineData("en")]
    public void NeverShowsAReadingTheSynthesizerWillNotSay(string lang)
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        var words = new[] { "Bonjour", "le", "monde", "aujourd'hui" };
        var ipa = IpaAnnotator.Annotate(words, lang);
        if (ipa is null) return;   // withheld, which is the other allowed answer
        var reading = global::Vernacula.Phonemizer.Phonemizer.Phonemize(string.Join(' ', words), lang);
        foreach (var s in ipa.Select(w => w.Ipa).Where(s => s.Length > 0))
            foreach (var group in s.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                Assert.Contains(group, reading);
    }

    // ── Scriptio continua: the render has to be segmented before ruby means anything ──

    [Fact]
    public void JapaneseSentenceSplitsIntoItsOwnUnits()
    {
        // One written "word" (no spaces) becomes the phonemizer's bunsetsu-like units, each with
        // its own reading over its own characters.
        var pieces = Ruby("東京都に住んでいます。", "ja")[0].Pieces;
        Assert.Equal(new[] { "東京都に", "住んで", "います", "。" }, pieces.Select(p => p.Text));
        Assert.Contains("sɯᵝ", pieces[1].Ipa);
        Assert.All(pieces.Take(3), p => Assert.True(p.Weight > 0, "a spoken piece carries duration weight"));
    }

    [Fact]
    public void JapaneseSplitsEvenWhenTheRewriteDropsProvenance()
    {
        // 私は -> 私わ loses every input span; the surfaces still account for the input character
        // for character, so the boundaries are recoverable -- and the text shown stays the text
        // the reader typed (私は), never the normalizer's rewrite (私わ).
        var pieces = Ruby("私は日本語を勉強しています。", "ja")[0].Pieces;
        Assert.Equal(new[] { "私は", "日本語を", "勉強して", "います", "。" }, pieces.Select(p => p.Text));
    }

    [Fact]
    public void ChineseSplitsPerCharacter()
    {
        var pieces = Ruby("我喜欢学习中文", "cmn")[0].Pieces;
        Assert.Equal(7, pieces.Count);
        Assert.Equal("我", pieces[0].Text);
        Assert.DoesNotContain(" ", pieces[0].Ipa.Trim());   // one syllable per character
    }

    [Fact]
    public void UnsegmentableScriptStaysWhole()
    {
        // Thai has no token boundaries here: three groups for fifteen characters. Rather than
        // invent a split, the whole run keeps one reading.
        var ruby = Ruby("ฉันเรียนภาษาไทย", "th")[0];
        Assert.Empty(ruby.Pieces);
        Assert.NotEqual("", ruby.Ipa);
    }

    [Fact]
    public void StackedReadingDoesNotBecomePieces()
    {
        // "$3.14" is three readings of ONE range, not three ranges: it must not be chopped up.
        var ruby = Ruby("$3.14", "en")[0];
        Assert.Empty(ruby.Pieces);
        Assert.Contains("dˈɑːlɚz", ruby.Ipa);
    }

    [Fact]
    public void PiecesSpellTheWordBackExactly()
    {
        foreach (var (word, lang) in new[] { ("東京都に住んでいます。", "ja"), ("我喜欢学习中文", "cmn"),
                                             ("私は日本語を勉強しています。", "ja") })
        {
            var pieces = Ruby(word, lang)[0].Pieces;
            Assert.Equal(word, string.Concat(pieces.Select(p => p.Text)));
        }
    }

    [Fact]
    public void EmptyInputIsEmptyNotNull()
        => Assert.Empty(IpaAnnotator.Annotate(Array.Empty<string>(), "en")!);

    [Fact]
    public void UnknownLanguageDeclinesRatherThanThrowing()
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        Assert.Null(IpaAnnotator.Annotate(new[] { "hello" }, "zz-not-a-language"));
    }
}
