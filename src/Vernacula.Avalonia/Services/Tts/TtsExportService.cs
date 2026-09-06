using System.Text;
using System.Text.RegularExpressions;
using Vernacula.App.Models;
using Vernacula.Phonemizer;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Export of a finished TTS job: the rendered audio, and a sentence-by-sentence CSV of what
/// was read (orthography), what the engine was given for it (phonemes), and where in the
/// audio it lies. Sentences are cut from the same extracted text the words came from, at
/// whitespace after terminal punctuation, so each maps onto a run of the aligned words and
/// takes its timing from them.
/// </summary>
internal static class TtsExportService
{
    /// <summary>One exported row.</summary>
    public sealed record SentenceRow(int Index, double StartSeconds, double EndSeconds, string Text, string Phonemes);

    /// <summary>How the phoneme column was produced — named in the CSV so it is never mistaken for another scheme.</summary>
    public static string PhonemeScheme(TtsBackendKind kind) => kind switch
    {
        TtsBackendKind.Kokoro => "kokoro",              // Kokoro's own vocabulary, exactly what the model consumed
        _                     => "ipa",                 // vernacula-phonemizer canonical IPA
    };

    // Terminal punctuation (Latin, ellipsis, CJK) followed by whitespace. Only whitespace
    // boundaries are cut so every sentence is a whole number of whitespace-split words —
    // the unit the alignment is keyed on. Paragraph breaks are whitespace too.
    private static readonly Regex SentenceEnd = new(@"(?<=[.!?…。！？])\s+", RegexOptions.Compiled);
    private static readonly char[] Whitespace = [' ', '\t', '\n', '\r'];

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
            var tokens = raw.Split(Whitespace, StringSplitOptions.RemoveEmptyEntries);
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
        TtsBackendKind kind, string lang, string voice, string? phonemizerDataDir)
    {
        Func<string, string> phonemize;
        switch (kind)
        {
            case TtsBackendKind.Kokoro:
            {
                bool british = voice.StartsWith("bf_", StringComparison.Ordinal) || voice.StartsWith("bm_", StringComparison.Ordinal);
                var g2p = new KokoroPhonemizer(phonemizerDataDir);
                phonemize = s => g2p.ToPhonemes(s, british);
                break;
            }
            default:
            {
                if (PhonemizerData.Resolve(phonemizerDataDir) is null)
                    throw new DirectoryNotFoundException(PhonemizerData.NotFoundMessage());
                Registry.EnsureLanguages();
                string code = string.IsNullOrWhiteSpace(lang) ? "en" : lang.Trim();
                phonemize = s => OmniVoiceIpaTts.Phonemize(s, code);
                break;
            }
        }

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

    private static string CsvEscape(string s)
    {
        if (s.IndexOfAny([',', '"', '\n', '\r']) < 0) return s;
        return "\"" + s.Replace("\"", "\"\"") + "\"";
    }
}
