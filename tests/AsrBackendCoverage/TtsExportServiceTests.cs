using Vernacula.Tts.Base.Alignment;
using System;
using System.IO;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The reader's CSV export cuts the extracted text into sentences and times each from the
/// aligned words. The cut must fall on whitespace so every sentence is a whole number of the
/// words the alignment is keyed on — otherwise every later sentence's timing is off by the
/// mis-cut word.
/// </summary>
public class TtsExportServiceTests
{
    private static AlignedWord W(string t, double s, double e) => new() { Text = t, StartSeconds = s, EndSeconds = e };

    [Fact]
    public void SentencesTakeTimingFromTheirWords()
    {
        string md = "# Title\n\nOne two. Three four! Five…  Six\n";
        // Extracted (the heading gains a full stop for prosody), whitespace-split: Title. One two. Three four! Five… Six
        var words = new[]
        {
            W("Title.", 0.0, 0.5), W("One", 1.0, 1.2), W("two.", 1.2, 1.5), W("Three", 2.0, 2.3),
            W("four!", 2.3, 2.8), W("Five…", 3.0, 3.4), W("Six", 4.0, 4.4),
        };
        var s = TtsExportService.SplitSentences(md, words);
        Assert.Equal(new[] { "Title.", "One two.", "Three four!", "Five…", "Six" }, s.Select(x => x.Text));
        Assert.Equal((1.0, 1.5), (s[1].Start, s[1].End));
        Assert.Equal((2.0, 2.8), (s[2].Start, s[2].End));
        Assert.Equal((4.0, 4.4), (s[4].Start, s[4].End));
    }

    [Fact]
    public void NonAsciiWhitespaceCountsAsAWordBoundary()
    {
        // The aligners split on Unicode whitespace, so the export must too: a no-break space
        // (ordinary in pasted text) counted as one word here would shift every later row's
        // timing by one word per occurrence.
        string md = "Bonjour\u00A0! Deuxième phrase.";
        var words = new[] { W("Bonjour", 0.0, 0.4), W("!", 0.4, 0.6), W("Deuxième", 1.0, 1.4), W("phrase.", 1.4, 1.9) };
        var s = TtsExportService.SplitSentences(md, words);
        Assert.Equal(new[] { "Bonjour !", "Deuxième phrase." }, s.Select(x => x.Text));
        Assert.Equal((1.0, 1.9), (s[1].Start, s[1].End));
    }

    [Fact]
    public void CsvQuotesCommasAndQuotes()
    {
        string path = Path.Combine(Path.GetTempPath(), $"vernacula-export-{Guid.NewGuid():N}.csv");
        try
        {
            TtsExportService.WriteCsv(path, [new TtsExportService.SentenceRow(1, 0.5, 1.25, "Hi, \"there\".", "haɪ ðɛɹ")], "ipa");
            var lines = File.ReadAllLines(path);
            Assert.Equal("index,start_seconds,end_seconds,text,phonemes,phoneme_scheme", lines[0]);
            Assert.Equal("1,0.500,1.250,\"Hi, \"\"there\"\".\",haɪ ðɛɹ,ipa", lines[1]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void KokoroRowsCarryKokoroPhonemes()
    {
        var settings = new SettingsService(); settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");
        var rows = TtsExportService.BuildRows([("Hello world.", 0, 1)], settings,
            new TtsJobSettings("Kokoro", "", "af_heart"));
        Assert.Single(rows);
        Assert.False(string.IsNullOrWhiteSpace(rows[0].Phonemes));
        Assert.DoesNotContain("<error", rows[0].Phonemes);
    }
}
