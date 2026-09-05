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
    public void EmptyInputIsEmptyNotNull()
        => Assert.Empty(IpaAnnotator.Annotate(Array.Empty<string>(), "en")!);

    [Fact]
    public void UnknownLanguageDeclinesRatherThanThrowing()
    {
        if (!HavePhonemizer()) Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        Assert.Null(IpaAnnotator.Annotate(new[] { "hello" }, "zz-not-a-language"));
    }
}
