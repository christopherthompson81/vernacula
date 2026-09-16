using Vernacula.Tts.Base.Alignment;
using System;
using System.IO;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Vernacula.Tts.Base;
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
            TtsExportService.WriteCsv(path,
                [new TtsExportService.SentenceRow(1, 0.5, 1.25, "Hi, \"there\".", "hˈaᶦ ðɛɹ", "haɪ ðɛɹ")], "kokoro");
            var lines = File.ReadAllLines(path);
            Assert.Equal("index,start_seconds,end_seconds,text,ipa,engine_phonemes,phoneme_scheme", lines[0]);
            Assert.Equal("1,0.500,1.250,\"Hi, \"\"there\"\".\",hˈaᶦ ðɛɹ,haɪ ðɛɹ,kokoro", lines[1]);
        }
        finally { File.Delete(path); }
    }

    /// <summary>
    /// ⚠ BOTH READINGS, AND THEY MUST DIFFER. The export used to carry only the engine's own stream
    /// under the name `phonemes`, and it was read as "the IPA" — reasonably, since the reader draws
    /// canonical IPA above every word and that is not what came out of the file. Kokoro's scheme has
    /// no aspiration and no length marks, so `pʰˈɜɹ ˈɑː` reaches it as `pˈɜɹ ˈɑ`: asserting the two
    /// columns are not equal is what pins that they are actually two different readings.
    /// </summary>
    [Fact]
    public void RowsCarryCanonicalIpaAndTheEnginesOwnStream()
    {
        var settings = new SettingsService(); settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");
        var rows = TtsExportService.BuildRows([("Percy waited.", 0, 1)], settings,
            new TtsJobSettings("Kokoro", "", "af_heart"), "en");
        Assert.Single(rows);

        Assert.False(string.IsNullOrWhiteSpace(rows[0].EnginePhonemes));
        Assert.DoesNotContain("<error", rows[0].EnginePhonemes);
        Assert.False(string.IsNullOrWhiteSpace(rows[0].Ipa));
        Assert.DoesNotContain("<error", rows[0].Ipa);
        Assert.Contains("ʰ", rows[0].Ipa);                       // canonical marks aspiration
        Assert.NotEqual(rows[0].Ipa, rows[0].EnginePhonemes);
    }

    /// <summary>
    /// ⚠ THE IPA COLUMN MUST COME FROM THE NEURAL ENTRY, and nothing else in the suite pins which
    /// entry produced it — the sync and async paths return the same type and agree on every word the
    /// dictionary carries, so a regression to <c>Phonemize</c> would pass every other test here.
    ///
    /// <para>"auditable" is the probe because it is out-of-dictionary and the two paths disagree
    /// legibly: the n-gram letter-to-sound engine reads the suffix as FACE
    /// (<c>ˈɔːd̬it̬ˌeᶦbəɫ</c> — "audit-AY-bul"), the BiLSTM tagger reduces it
    /// (<c>ˈɔːd̬ət̬əbəɫ</c>). It stands for a class: 28.9% of <c>-able</c>/<c>-ible</c> words the
    /// dictionary misses against 0.2%, measured over misaki's lexicon in
    /// docs/investigations/kokoro_vphon_investigation.md Run 13.</para>
    ///
    /// <para>⚠ SKIPPED, NOT FAILED, WHEN THE TAGGER IS ABSENT. The phonemizer degrades to the sync
    /// reading when ONNX Runtime or the model is missing (EnglishTagger.cs returns null and
    /// EnglishNeural.cs falls through), and it does so SILENTLY — there is no API to ask. So the
    /// state is probed the only way available: if the two entries agree on the probe word, the
    /// tagger did not load and there is nothing here to test. That probe is itself the reason this
    /// investigation went wrong once — a harness with no ONNX Runtime measured the fallback for a
    /// full sweep and nothing said so.</para>
    /// </summary>
    [Fact]
    public async System.Threading.Tasks.Task TheIpaColumnIsTheNeuralReadingNotTheNGramFallback()
    {
        var settings = new SettingsService(); settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");

        const string probe = "auditable";
        // Resolve the data tree before touching the phonemizer, exactly as CanonicalReader does —
        // Registry.EnsureLanguages() reads it eagerly and throws from a static initializer otherwise.
        if (PhonemizerData.Resolve(settings.GetPhonemizerDataDir()) is null)
            Assert.Skip("no vernacula-phonemizer data tree here");
        Vernacula.Phonemizer.Registry.EnsureLanguages();
        var neural = (await Vernacula.Phonemizer.Phonemizer.PhonemizeAsync(probe, "en")).Trim();
        var ngram  = Vernacula.Phonemizer.Phonemizer.PhonemizeTrace(probe, "en").Ipa.Trim();
        if (neural == ngram)
            Assert.Skip("the English neural tagger did not load here, so both entries return the n-gram reading");

        var rows = TtsExportService.BuildRows([($"The report is {probe}.", 0, 1)], settings,
            new TtsJobSettings("Kokoro", "", "af_heart"), "en");
        Assert.Single(rows);
        Assert.DoesNotContain("<error", rows[0].Ipa);

        // Escapes, not literals: an earlier version of this comparison was typo'd one codepoint away
        // in a lookalike superscript and would have passed while asserting nothing.
        Assert.Contains("\u0259b\u0259\u026b", rows[0].Ipa);        // əbəɫ — reduced, the neural reading
        Assert.DoesNotContain("e\u1DA6b\u0259\u026b", rows[0].Ipa); // eᶦbəɫ — FACE, the n-gram reading
        Assert.Contains(neural, rows[0].Ipa);
    }

    /// <summary>A language the phonemizer does not carry leaves the IPA column EMPTY rather than an
    /// error string in every row — the reader draws nothing in that case either.</summary>
    [Fact]
    public void AnUnknownAnnotationLanguageLeavesTheIpaColumnEmpty()
    {
        var settings = new SettingsService(); settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");
        var rows = TtsExportService.BuildRows([("Hello world.", 0, 1)], settings,
            new TtsJobSettings("Kokoro", "", "af_heart"), "zzz-not-a-language");
        Assert.Single(rows);
        Assert.DoesNotContain("<error", rows[0].Ipa);
        Assert.False(string.IsNullOrWhiteSpace(rows[0].EnginePhonemes));
    }

    /// <summary>
    /// ⚠ THE AUDIO HALF OF THE EXPORT HAD NO TEST AT ALL, because it was a bare File.Copy in the
    /// view model rather than part of this service. That is exactly the half a user reported missing.
    /// </summary>
    [Fact]
    public void WriteAudioCopiesTheRenderedFile()
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            string rendered = Path.Combine(dir.FullName, "job.wav");
            File.WriteAllBytes(rendered, [1, 2, 3, 4]);

            string audio = TtsExportService.WriteAudio(Path.Combine(dir.FullName, "out.wav"), rendered);

            Assert.Equal(Path.Combine(dir.FullName, "out.wav"), audio);
            Assert.Equal(new byte[] { 1, 2, 3, 4 }, File.ReadAllBytes(audio));
        }
        finally { dir.Delete(recursive: true); }
    }

    /// <summary>
    /// ⚠ ONE EXPORT WRITES ONE FILE. Asking for the audio must not also drop a CSV beside it — that
    /// was the reported behaviour ("export types should be done one-at-a-time"), and a stray sibling
    /// is exactly what it looked like from the file manager.
    /// </summary>
    [Fact]
    public void ExportingOneKindWritesNothingElse()
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            string rendered = Path.Combine(dir.FullName, "job.wav");
            File.WriteAllBytes(rendered, [9]);

            TtsExportService.WriteAudio(Path.Combine(dir.FullName, "out.wav"), rendered);
            Assert.False(File.Exists(Path.Combine(dir.FullName, "out.csv")));
            Assert.False(File.Exists(Path.Combine(dir.FullName, "out.md")));

            TtsExportService.WriteTranscript(Path.Combine(dir.FullName, "t.csv"), [], "kokoro");
            Assert.False(File.Exists(Path.Combine(dir.FullName, "t.wav")));

            TtsExportService.WriteMarkdown(Path.Combine(dir.FullName, "d.md"), "# Doc");
            Assert.False(File.Exists(Path.Combine(dir.FullName, "d.wav")));
            Assert.False(File.Exists(Path.Combine(dir.FullName, "d.csv")));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void WriteTranscriptWritesTheCsv()
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            var rows = new[] { new TtsExportService.SentenceRow(1, 0, 1.5, "Hello.", "h\u025bl\u02c8o\u1d76", "hɛlˈoʊ") };
            string csv = TtsExportService.WriteTranscript(Path.Combine(dir.FullName, "out.csv"), rows, "kokoro");
            Assert.Equal(Path.Combine(dir.FullName, "out.csv"), csv);
            Assert.Contains("Hello.", File.ReadAllText(csv));
        }
        finally { dir.Delete(recursive: true); }
    }

    /// <summary>The markdown export is of the document as it stands — the edited text, verbatim.</summary>
    [Fact]
    public void WriteMarkdownWritesTheDocumentVerbatim()
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            const string doc = "# Title\n\nA paragraph with **bold**.\n\n- item\n";
            string md = TtsExportService.WriteMarkdown(Path.Combine(dir.FullName, "out.md"), doc);
            Assert.Equal(doc, File.ReadAllText(md));
        }
        finally { dir.Delete(recursive: true); }
    }

    /// <summary>A picker can hand back a name with no extension, or one left over from a type the
    /// user then changed; the file is named for what is actually written into it.</summary>
    [Theory]
    [InlineData("out", ".wav")]
    [InlineData("out.csv", ".wav")]
    public void TheAudioIsNamedForItsContent(string chosen, string expected)
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            string rendered = Path.Combine(dir.FullName, "job.wav");
            File.WriteAllBytes(rendered, [5]);
            string audio = TtsExportService.WriteAudio(Path.Combine(dir.FullName, chosen), rendered);
            Assert.Equal(expected, Path.GetExtension(audio));
        }
        finally { dir.Delete(recursive: true); }
    }

    /// <summary>⚠ .markdown and .txt are kept as chosen rather than forced to .md — the user picked
    /// the name, and all three are the same bytes.</summary>
    [Theory]
    [InlineData("out.md", ".md")]
    [InlineData("out.markdown", ".markdown")]
    [InlineData("out.txt", ".txt")]
    [InlineData("out", ".md")]
    public void TheMarkdownKeepsAnyMarkdownExtensionItWasGiven(string chosen, string expected)
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            string md = TtsExportService.WriteMarkdown(Path.Combine(dir.FullName, chosen), "x");
            Assert.Equal(expected, Path.GetExtension(md));
        }
        finally { dir.Delete(recursive: true); }
    }

    /// <summary>The picker hands back a path, not the file type that produced it, so the extension
    /// is what chooses the export. Anything else is refused rather than guessed at.</summary>
    [Theory]
    [InlineData("a.wav",      TtsExportService.ExportKind.Audio)]
    [InlineData("a.WAV",      TtsExportService.ExportKind.Audio)]
    [InlineData("a.csv",      TtsExportService.ExportKind.Csv)]
    [InlineData("a.md",       TtsExportService.ExportKind.Markdown)]
    [InlineData("a.markdown", TtsExportService.ExportKind.Markdown)]
    [InlineData("a.txt",      TtsExportService.ExportKind.Markdown)]
    internal void TheExtensionChoosesTheExport(string path, TtsExportService.ExportKind expected) =>
        Assert.Equal(expected, TtsExportService.KindOf(path));

    [Theory]
    [InlineData("a.mp3")]
    [InlineData("a.pdf")]
    [InlineData("a")]
    public void AnUnrecognizedExtensionExportsNothing(string path) =>
        Assert.Null(TtsExportService.KindOf(path));

    /// <summary>⚠ Exporting on top of the job's own file is a no-op — File.Copy throws on same-path.</summary>
    [Fact]
    public void ExportingOntoTheRenderedFileItselfDoesNotThrow()
    {
        var dir = Directory.CreateTempSubdirectory("tts-export-test");
        try
        {
            string rendered = Path.Combine(dir.FullName, "same.wav");
            File.WriteAllBytes(rendered, [7]);
            string audio = TtsExportService.WriteAudio(rendered, rendered);
            Assert.Equal(new byte[] { 7 }, File.ReadAllBytes(audio));
        }
        finally { dir.Delete(recursive: true); }
    }
}
