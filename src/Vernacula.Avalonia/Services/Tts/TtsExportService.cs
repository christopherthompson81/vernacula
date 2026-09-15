using Vernacula.Tts.Base.Alignment;
using System.Text;
using System.Text.RegularExpressions;
using Vernacula.App.Models;
using Vernacula.Phonemizer;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Export of a finished TTS job, ONE FILE PER INVOCATION — the rendered audio, a
/// sentence-by-sentence CSV, or the source markdown. The CSV says what was read (orthography),
/// what the engine was given for it (phonemes), and where in the audio it lies; sentences are cut
/// from the same extracted text the words came from, at whitespace after terminal punctuation, so
/// each maps onto a run of the aligned words and takes its timing from them.
/// </summary>
internal static class TtsExportService
{
    /// <summary>One exported row.</summary>
    public sealed record SentenceRow(int Index, double StartSeconds, double EndSeconds, string Text, string Phonemes);

    // Terminal punctuation (Latin, ellipsis, CJK) followed by whitespace. Only whitespace
    // boundaries are cut so every sentence is a whole number of whitespace-split words —
    // the unit the alignment is keyed on. Paragraph breaks are whitespace too.
    private static readonly Regex SentenceEnd = new(@"(?<=[.!?…。！？])\s+", RegexOptions.Compiled);

    /// <summary>
    /// The sentences of <paramref name="sourceText"/> (markdown or plain text) with timing from
    /// <paramref name="words"/> — the sidecar's aligned words, 1:1 with the whitespace-split
    /// extracted text. A sentence whose words carry no timing gets 0/0.
    /// </summary>
    public static List<(string Text, double Start, double End)> SplitSentences(string sourceText, IReadOnlyList<AlignedWord> words)
    {
        string extracted = MarkdownTextExtractor.Extract(sourceText ?? "").Text;
        var result = new List<(string, double, double)>();
        int wordCursor = 0;
        foreach (var raw in SentenceEnd.Split(extracted))
        {
            // Split((char[]?)null): Unicode whitespace, the same tokenizer the aligners and the
            // segmenter use — an ASCII-only set miscounted words around a no-break space and
            // shifted every later row's timing.
            var tokens = raw.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0) continue;
            int first = wordCursor, last = wordCursor + tokens.Length - 1;
            wordCursor += tokens.Length;
            double start = 0, end = 0;
            if (first < words.Count)
            {
                start = words[first].StartSeconds;
                end   = words[Math.Min(last, words.Count - 1)].EndSeconds;
            }
            result.Add((string.Join(' ', tokens), start, end));
        }
        return result;
    }

    /// <summary>Phonemizes each sentence the way the job's engine would read it. Blocking; call off the UI thread.</summary>
    public static List<SentenceRow> BuildRows(
        IReadOnlyList<(string Text, double Start, double End)> sentences,
        SettingsService settings, TtsJobSettings job)
    {
        // The engine owns its own text → phonemes path, so the CSV shows what that engine reads.
        var phonemize = TtsEngines.For(job.Backend).CreatePhonemizer(settings, job);

        var rows = new List<SentenceRow>(sentences.Count);
        for (int i = 0; i < sentences.Count; i++)
        {
            var (text, start, end) = sentences[i];
            string phonemes;
            try { phonemes = phonemize(text); }
            catch (Exception ex) { phonemes = $"<error: {ex.Message}>"; }
            rows.Add(new SentenceRow(i + 1, start, end, text, phonemes));
        }
        return rows;
    }

    public static void WriteCsv(string path, IEnumerable<SentenceRow> rows, string scheme)
    {
        using var writer = new StreamWriter(path, append: false, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        writer.WriteLine("index,start_seconds,end_seconds,text,phonemes,phoneme_scheme");
        foreach (var r in rows)
            writer.WriteLine(string.Join(',',
                r.Index.ToString(),
                r.StartSeconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture),
                r.EndSeconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture),
                CsvEscape(r.Text), CsvEscape(r.Phonemes), scheme));
    }

    /// <summary>What a single export invocation produces.</summary>
    public enum ExportKind
    {
        /// <summary>The rendered audio, copied out of the job's own folder.</summary>
        Audio,
        /// <summary>The sentence-by-sentence transcript with timing and phonemes.</summary>
        Csv,
        /// <summary>The source document as it now stands, including any edits made in the reader.</summary>
        Markdown,
    }

    /// <summary>
    /// Which export the chosen path asks for, or null when the extension names none of them.
    ///
    /// ⚠ THE FILE TYPE IS THE CHOICE. The export used to write the audio AND the transcript on every
    /// invocation regardless of which type the picker was on — reported as "export types should be
    /// done one-at-a-time instead of all types exporting at once". The picker's own type dropdown is
    /// the selector; this is where its answer is read back, since a save picker hands back a path and
    /// not the type that produced it.
    /// </summary>
    public static ExportKind? KindOf(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".wav"                      => ExportKind.Audio,
        ".csv"                      => ExportKind.Csv,
        ".md" or ".markdown" or ".txt" => ExportKind.Markdown,
        _                           => null,
    };

    /// <summary>
    /// The rendered audio, copied to <paramref name="chosenPath"/>; the written path is returned.
    ///
    /// ⚠ THE AUDIO COPY LIVES HERE, not at the call site, and that is the point of this method. It
    /// used to be a bare <c>File.Copy</c> in the view model, which is why nothing covered it: every
    /// other part of the export had a test and the half the user actually wanted had none.
    /// </summary>
    public static string WriteAudio(string chosenPath, string renderedAudioPath)
    {
        string audioPath = WithExtension(chosenPath, ".wav");
        // Exporting on top of the job's own rendered file is a no-op, not an error: File.Copy throws
        // when source and destination are the same path.
        if (!PathsEqual(audioPath, renderedAudioPath))
            File.Copy(renderedAudioPath, audioPath, overwrite: true);
        return audioPath;
    }

    /// <summary>The transcript CSV; the written path is returned.</summary>
    public static string WriteTranscript(string chosenPath, IEnumerable<SentenceRow> rows, string scheme)
    {
        string csvPath = WithExtension(chosenPath, ".csv");
        WriteCsv(csvPath, rows, scheme);
        return csvPath;
    }

    /// <summary>
    /// The document itself, written out verbatim. In editing mode this is the EDITED markdown, which
    /// is the point of offering it: the reader is where the revision happened.
    /// </summary>
    public static string WriteMarkdown(string chosenPath, string markdown)
    {
        string mdPath = KindOf(chosenPath) == ExportKind.Markdown ? chosenPath : WithExtension(chosenPath, ".md");
        File.WriteAllText(mdPath, markdown ?? "", new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        return mdPath;
    }

    /// <summary>
    /// The path with <paramref name="extension"/> forced. A picker can hand back a name the user
    /// typed with no extension at all, or with the one from a type they then changed; the file that
    /// gets written is named for what is actually in it.
    /// </summary>
    private static string WithExtension(string path, string extension) =>
        string.Equals(Path.GetExtension(path), extension, StringComparison.OrdinalIgnoreCase)
            ? path : Path.ChangeExtension(path, extension);

    private static bool PathsEqual(string a, string b) =>
        string.Equals(Path.GetFullPath(a), Path.GetFullPath(b),
            OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);

    private static string CsvEscape(string s)
    {
        if (s.IndexOfAny([',', '"', '\n', '\r']) < 0) return s;
        return "\"" + s.Replace("\"", "\"\"") + "\"";
    }
}
