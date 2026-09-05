using Vernacula.Phonemizer;

namespace Vernacula.Tts.Base;

/// <summary>
/// Per-word IPA for display above the written text, furigana style.
///
/// The reading itself is the phonemizer's; what this adds is the attribution — which slice of the
/// IPA belongs over which written word. That comes from the trace's two-way index (#1150): a
/// token's <c>InputSpan</c> says which characters the reader typed, its <c>IpaSpan</c> which
/// characters of the reading they became.
///
/// ⚠ THE WORDS ARE ANNOTATED AS ONE UTTERANCE, NOT ONE AT A TIME. Phonemizing each word alone
/// would lose the context the normalizer needs — "Mr. Smith" and "$3.14" are read across word
/// boundaries — so the words are rejoined with single spaces and traced in one pass, which also
/// keeps the annotation consistent with what the synthesizer will say.
/// </summary>
public static class IpaAnnotator
{
    /// <summary>
    /// One IPA string per entry of <paramref name="words"/>, or null when the reading cannot be
    /// attributed. Entries may be empty (a word that produced no phonemes of its own, such as a
    /// second written word folded into a preceding rewrite).
    ///
    /// Null is returned rather than a partial map: an annotation that is silently off by one word
    /// is worse than none, and the trace withholds spans precisely when it cannot vouch for them.
    /// </summary>
    /// <param name="dataDir">The vernacula-phonemizer <c>data/</c> root, resolved as
    /// <see cref="PhonemizerData.Resolve"/> does when null.</param>
    public static IReadOnlyList<string>? Annotate(IReadOnlyList<string> words, string lang,
        string? dataDir = null)
    {
        if (words.Count == 0) return Array.Empty<string>();
        if (PhonemizerData.Resolve(dataDir) is null) return null;

        var text = string.Join(' ', words);
        PhonemeTrace trace;
        try
        {
            // Registering the languages reads the data tree, so it can fail for the same reasons
            // the reading can; both are "no annotation", never a crash in a display path.
            Registry.EnsureLanguages();
            trace = global::Vernacula.Phonemizer.Phonemizer.PhonemizeTrace(text, lang);
        }
        catch (Exception) { return null; }   // unknown language, unreadable data: no annotation
        if (!trace.Traced) return null;

        // Character offset -> index of the word containing it. The words were joined with single
        // spaces, so this is exact rather than a re-split of the caller's text.
        var wordAt = new int[text.Length];
        var pos = 0;
        for (var i = 0; i < words.Count; i++)
        {
            for (var k = 0; k < words[i].Length; k++) wordAt[pos + k] = i;
            pos += words[i].Length;
            if (pos < text.Length) wordAt[pos++] = i;   // the joining space belongs to the word before
        }

        var parts = new List<string>[words.Count];
        (int Start, int End)? lastSpan = null;
        var lastWord = -1;
        foreach (var tok in trace.Tokens)
        {
            if (tok.InputSpan is not { } input || input.Start < 0 || input.Start >= text.Length) return null;
            var ipa = tok.IpaSpan is { } span && span.Start >= 0 && span.End <= trace.Ipa.Length
                ? trace.Ipa[span.Start..span.End]
                : string.Concat(tok.Emitted);
            // Tokens sharing one input span came from one normalizer rewrite. When the span covers
            // several written words ("Mr. Smith" -> mister, Smith) each successive token takes the
            // next word in it, so the annotation lands over the word it is reading; when it is one
            // word ("$3.14" -> three, dollars, fourteen) they all stack over that word.
            var lastInSpan = wordAt[Math.Min(input.End, text.Length) - 1];
            var word = lastSpan == input ? Math.Min(lastWord + 1, lastInSpan) : wordAt[input.Start];
            if (word < 0 || word >= words.Count) return null;
            (parts[word] ??= new List<string>()).Add(ipa.Trim());
            lastSpan = input; lastWord = word;
        }

        var result = new string[words.Count];
        for (var i = 0; i < words.Count; i++)
            result[i] = parts[i] is { } p ? string.Join(' ', p.Where(s => s.Length > 0)) : "";
        return result;
    }
}
