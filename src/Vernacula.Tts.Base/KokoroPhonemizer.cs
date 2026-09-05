using Vernacula.Phonemizer;

namespace Vernacula.Tts.Base;

/// <summary>
/// One phonemized text: the Kokoro-alphabet string, and for each phoneme group in it (a run of
/// tokens between space tokens — one per spoken word) the index of the whitespace-delimited source
/// word it came from. <see cref="GroupSourceWords"/> is null when the phonemizer could not account
/// for every group; callers fall back to an even split.
/// </summary>
public sealed record KokoroPhonemization(string Phonemes, IReadOnlyList<int>? GroupSourceWords);

/// <summary>
/// Kokoro's G2P frontend: text → canonical IPA (vernacula-phonemizer, <c>en</c> / <c>en-GB</c>) →
/// Kokoro's alphabet (<see cref="KokoroFormat"/>). Needs only the phonemizer's data tree, not the
/// model, so it is usable — and testable — without an ONNX session.
///
/// ⚠ TWO PHONEMIZER ENTRIES ARE USED PER CALL, ON PURPOSE. <c>PhonemizeAsync</c> is the best
/// reading (English routes out-of-vocabulary words through its BiLSTM), but only the synchronous
/// <c>PhonemizeTrace</c> reports which characters of the input each IPA span came from — and that
/// map is what word-level alignment needs. The neural path changes how an OOV word is READ, not
/// how many words there are, so the trace's per-word group counts describe the async output too;
/// this is checked, and when the two disagree the traced reading is used so the map is never wrong.
/// </summary>
public sealed class KokoroPhonemizer
{
    /// <param name="dataDir">The vernacula-phonemizer <c>data/</c> root. Null resolves it the way
    /// <see cref="PhonemizerData.Resolve"/> does (VERNACULA_DATA_DIR, then the submodule).</param>
    public KokoroPhonemizer(string? dataDir = null)
    {
        if (PhonemizerData.Resolve(dataDir) is null)
            throw new DirectoryNotFoundException(PhonemizerData.NotFoundMessage());
        Registry.EnsureLanguages();
    }

    private static string Lang(bool british) => british ? "en-GB" : "en";

    /// <summary>Text → Kokoro-alphabet phoneme string.</summary>
    public string ToPhonemes(string text, bool british = false) => Phonemize(text, british).Phonemes;

    /// <summary>Inner phoneme-token count (excludes the 2 pad tokens) for <paramref name="text"/>.</summary>
    public int CountTokens(string text, bool british = false)
        => Math.Max(0, KokoroVocab.Encode(ToPhonemes(text, british)).Length - 2);

    /// <summary>Text → Kokoro phonemes plus the phoneme-group → source-word map.</summary>
    public KokoroPhonemization Phonemize(string text, bool british = false)
    {
        var lang = Lang(british);
        var trace = global::Vernacula.Phonemizer.Phonemizer.PhonemizeTrace(text, lang);
        var map = GroupSourceWords(trace, text);

        string ipa;
        try
        {
            // Every caller is already off the UI thread (Task.Run in the reader, top-level in the
            // CLIs), and the phonemizer awaits with ConfigureAwait(false), so blocking here is safe.
            ipa = global::Vernacula.Phonemizer.Phonemizer.PhonemizeAsync(text, lang).GetAwaiter().GetResult();
        }
        catch (Exception)
        {
            ipa = trace.Ipa;   // a missing OOV model must not take the utterance down
        }
        // The map was built from the traced reading; use it only for a reading with the same shape.
        if (map is not null && CountWordGroups(ipa) != map.Count)
            ipa = trace.Ipa;

        return new KokoroPhonemization(KokoroFormat.Render(ipa, british), map);
    }

    /// <summary>
    /// One source-word index per spoken IPA group, from the trace's spans. A token's input span
    /// says which source word it was; its IPA span says how many groups it became (a number reads
    /// as several words). Null when any token is missing a span, since a partial map would assign
    /// the wrong words to every group after the gap.
    /// </summary>
    private static List<int>? GroupSourceWords(PhonemeTrace trace, string text)
    {
        if (!trace.Traced) return null;

        // Character offset → index of the whitespace-delimited word containing it.
        var wordAt = new int[text.Length];
        var w = -1; var inWord = false;
        for (var i = 0; i < text.Length; i++)
        {
            if (char.IsWhiteSpace(text[i])) { inWord = false; wordAt[i] = w + 1; continue; }
            if (!inWord) { w++; inWord = true; }
            wordAt[i] = w;
        }
        var wordCount = w + 1;

        var map = new List<int>();
        (int Start, int End)? lastSpan = null;
        var lastWord = -1;
        foreach (var tok in trace.Tokens)
        {
            if (tok.InputSpan is not { } input || input.Start < 0 || input.Start >= text.Length) return null;
            int groups;
            if (tok.IpaSpan is { } span)
                groups = CountWordGroups(trace.Ipa[span.Start..span.End]);
            else if (tok.Emitted.Count > 0)
                groups = tok.Emitted.Count;
            else
                return null;
            // Tokens that share one input span came from one normalizer rewrite. When the span is
            // one written word ("$3.14" → three, dollars, fourteen) they all belong to it; when it
            // covers several ("Mr. Smith" → mister, Smith) each successive token takes the next
            // word in the span, so the highlight moves with the speech instead of sticking.
            var lastInSpan = wordAt[Math.Min(input.End, text.Length) - 1];
            var word = lastSpan == input ? Math.Min(lastWord + 1, lastInSpan) : wordAt[input.Start];
            if (word >= wordCount) return null;
            for (var g = 0; g < groups; g++) map.Add(word);
            lastSpan = input; lastWord = word;
        }
        return map;
    }

    /// <summary>Space-delimited groups that contain a letter — i.e. not the phonemizer's
    /// stand-alone punctuation tokens, which <see cref="KokoroFormat"/> folds into the word before.</summary>
    private static int CountWordGroups(string ipa)
    {
        var n = 0;
        foreach (var g in ipa.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            if (g.Any(char.IsLetter)) n++;
        return n;
    }
}
