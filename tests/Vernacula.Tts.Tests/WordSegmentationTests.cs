using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// What counts as a word when the language does not put spaces between them.
///
/// <para>
/// ⚠ THE UNIT HERE IS SHARED BY THE READER AND THE ALIGNER, which is the reason it exists at all.
/// Both used to split on whitespace independently — agreeing by construction in English, and
/// producing one word per sentence in Japanese, so a whole paragraph lit up at once.
/// </para>
/// </summary>
public class WordSegmentationTests
{
    private static string Slice(string text, WordSpan w) => text[w.Start..w.End];

    private static IReadOnlyList<WordSpan> Segment(string text, string lang)
    {
        if (PhonemizerData.Resolve(null) is null && WordSegmentation.NeedsTrace(lang))
            Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        return WordSegmentation.Segment(text, 0, text.Length, lang);
    }

    [Fact]
    public void EnglishIsTheWhitespaceSplitItAlwaysWas()
    {
        const string text = "The harbour was quiet.";
        var words = WordSegmentation.Segment(text, 0, text.Length, "en");
        Assert.Equal(["The", "harbour", "was", "quiet."], words.Select(w => Slice(text, w)));
    }

    [Fact]
    public void ASubRangeSegmentsWithoutSeeingTheRestOfTheDocument()
    {
        // The reader segments one block at a time out of a larger extracted string, so the spans
        // have to be offsets into that string rather than into the slice.
        const string text = "skip this. The harbour was quiet.";
        var words = WordSegmentation.Segment(text, 11, text.Length, "en");
        Assert.Equal(["The", "harbour", "was", "quiet."], words.Select(w => Slice(text, w)));
        Assert.Equal(11, words[0].Start);
    }

    [Fact]
    public void JapaneseSplitsIntoPhrasesWhereWhitespaceSeesOneWord()
    {
        const string text = "科学者たちが発表しました。";
        var words = Segment(text, "ja");

        // ⚠ THIS CANNOT CATCH A COLD-INIT REGRESSION, and used to hide one. vernacula-phonemizer
        // #1408 made the FIRST ja trace in a process return every InputSpan null — but the defect
        // is once per process, so by the time any test runs, another has already warmed the
        // language and this passes either way. The gate has to spawn its own process, which is
        // what upstream's csharp/tools/trace-cold does. What is pinned here is the segmentation.

        // ⚠ PHRASES, NOT WORDS, and that is what the trace offers. Two units against the one
        // whitespace would give is the whole improvement; this is not morphological segmentation.
        Assert.True(words.Count >= 2, $"expected the trace to split the sentence, got {words.Count}");
        Assert.Equal(1, WordSegmentation.Whitespace(text, 0, text.Length).Count);
        // The trailing 。produces no spoken group and is not a clickable word.
        Assert.DoesNotContain("。", words.Select(w => Slice(text, w)));
        // Every span stays inside the text and in order.
        var previous = 0;
        foreach (var w in words)
        {
            Assert.InRange(w.Start, previous, text.Length);
            Assert.InRange(w.End, w.Start + 1, text.Length);
            previous = w.End;
        }
    }

    [Fact]
    public void PureHanziMandarinSplitsPerCharacter()
    {
        // One hanzi is one syllable and the IPA carries one group per syllable, so the groups walk
        // onto the characters. Measured over the phonemizer's 200 Mandarin goldens: all 126
        // pure-hanzi rows match exactly.
        const string text = "今天天气很好。";
        var words = Segment(text, "cmn");

        Assert.Equal(["今", "天", "天", "气", "很", "好"], words.Select(w => Slice(text, w)));
    }

    [Fact]
    public void MandarinWithDigitsKeepsTheTokenWholeRatherThanShifting()
    {
        // ⚠ THE CASE THE COUNT CHECK EXISTS FOR. `11` becomes two syllables, so there are more
        // groups than hanzi and a positional walk would put every later character on the wrong
        // syllable. Every one of the 74 goldens that fails the count has digits or latin in it.
        // Keeping the token whole is the old behaviour, which is worse granularity and not wrong.
        const string text = "11点20分警察要求。";
        var words = Segment(text, "cmn");

        Assert.True(words.Count < text.Count(c => c is >= '一' and <= '鿿'),
                    "a span whose group count disagrees with its hanzi count must not be split per character");
        foreach (var w in words) Assert.InRange(w.End, w.Start + 1, text.Length);
    }

    [Fact]
    public void MixedScriptJapaneseDoesNotOfferTwoWordsCoveringTheSameCharacters()
    {
        // ⚠ THE CASE THE MERGE EXISTS FOR, and ordinary Japanese: PDF, Wi-Fi, CD and latin proper
        // nouns are everywhere. `PDF` normalizes to ピーディーエフ, and the expansion's provenance
        // cannot reach back into the original, so all three tokens report the WHOLE sentence as
        // their input span. One unit per token then gives the reader three clickable words that
        // light identical text, of which only the first gets any time -- the rest are zero-length
        // markers. Measured before the merge: 14 of 123 golden rows, every one of them
        // mixed-script, worst case eight units each covering 30 of 34 characters.
        const string text = "PDFファイルを開いてください。";
        var words = Segment(text, "ja");

        // Distinctness is vacuously true of an empty list, so pin that there is something to be
        // distinct ABOUT. Deliberately not pinning the count: if upstream ever maps the expansion's
        // sub-spans properly this yields three distinct units and should still pass.
        Assert.NotEmpty(words);
        Assert.Equal(words.Count, words.Distinct().Count());
        // Merging costs granularity and must not cost correctness: what is left still spans text.
        foreach (var w in words) Assert.InRange(w.End, w.Start + 1, text.Length);
    }

    [Fact]
    public void TracedWordsAreOrderedAndNeverOverlap()
    {
        // The reader pairs displayed words to sidecar words BY INDEX, and both come from here, so
        // a unit list that overlaps or runs backwards misattributes audio rather than merely
        // looking odd. Cheap to assert, and the property the merge is really maintaining.
        foreach (var text in new[]
                 {
                     "科学者たちが発表しました。",
                     "彼女は新しい本を読んでいます。",
                     "PDFファイルを開いてください。",
                     "今天天气很好。",
                 })
        {
            var lang = text.Any(c => c is >= '\u3040' and <= '\u30ff') ? "ja" : "cmn";
            var words = Segment(text, lang);
            Assert.NotEmpty(words);   // the loop below is vacuous on an empty or single-unit list
            for (var i = 1; i < words.Count; i++)
                Assert.True(words[i].Start >= words[i - 1].End,
                            $"{lang} \"{text}\": unit {i} starts at {words[i].Start}, inside the previous unit ending at {words[i - 1].End}");
        }
    }

    [Fact]
    public void AnUnknownLanguageFallsBackToWhitespaceRatherThanThrowing()
    {
        // The reader builds words when a document is OPENED, which never needed a phonemizer, so
        // anything that goes wrong here has to cost granularity and not correctness.
        const string text = "one two three";
        Assert.Equal(3, WordSegmentation.Segment(text, 0, text.Length, "zzz-not-a-language").Count);
        Assert.Equal(3, WordSegmentation.Segment(text, 0, text.Length, null).Count);
    }
}
