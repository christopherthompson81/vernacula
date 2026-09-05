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
/// <summary>One displayable sub-part of a word: a slice of the written text and the reading of
/// that slice. <see cref="Weight"/> is its share of the word's spoken duration, so a highlight can
/// move through the pieces of a word the aligner only timed as a whole.</summary>
public sealed record RubyPiece(string Text, string Ipa, double Weight);

/// <summary>A word's annotation: the whole reading, plus the pieces it splits into when the word
/// is written without spaces (Japanese, Chinese) and the trace can say where the boundaries are.
/// <see cref="Pieces"/> is empty for ordinary spaced words, which are their own single piece.</summary>
public sealed record WordRuby(string Ipa, IReadOnlyList<RubyPiece> Pieces);

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
    public static IReadOnlyList<WordRuby>? Annotate(IReadOnlyList<string> words, string lang,
        string? dataDir = null)
    {
        if (words.Count == 0) return Array.Empty<WordRuby>();
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

        // Japanese drops span provenance through some normalizer rewrites (は read as わ), and a
        // word written without spaces is exactly where the boundaries matter most. When every span
        // is missing but the token surfaces still account for the input character for character,
        // the boundaries can be reconstructed from their lengths; when they cannot (a number that
        // expands, say) this returns null and the caller declines rather than guesses.
        var spans = InputSpans(trace, text);
        if (spans is null) return null;

        // Group the tokens by input span first. A span is the unit the normalizer rewrote, and
        // only by looking at a whole group can we tell whether its tokens line up with the written
        // words one for one -- which is what decides whether they may be distributed across them.
        var wordStart = new int[words.Count];
        for (int i = 0, at = 0; i < words.Count; i++) { wordStart[i] = at; at += words[i].Length + 1; }

        var parts = new List<Contribution>[words.Count];
        var i0 = 0;
        while (i0 < trace.Tokens.Count)
        {
            var input = spans[i0];
            var i1 = i0;
            while (i1 + 1 < trace.Tokens.Count && spans[i1 + 1] == input) i1++;

            var first = wordAt[input.Start];
            var last = wordAt[Math.Min(input.End, text.Length) - 1];
            if (first < 0 || last >= words.Count || last < first) return null;

            // One token per written word in the span ("Mr. Smith" -> mister, Smith): each token
            // annotates its own word. Any other shape -- "$3.14" read as three words, or "$ 3.14"
            // whose three tokens do not line up with its two written words -- is stacked over the
            // span's first word instead. Distributing a group that does NOT correspond one for one
            // puts the reading over the wrong word, which is worse than stacking it.
            var oneForOne = i1 - i0 == last - first;
            for (var t = i0; t <= i1; t++)
            {
                var tok = trace.Tokens[t];
                string ipa;
                if (tok.IpaSpan is { } span)
                {
                    if (span.Start < 0 || span.End > trace.Ipa.Length || span.End < span.Start) return null;
                    ipa = trace.Ipa[span.Start..span.End];
                }
                else if (tok.Emitted.Count == 0)
                {
                    // No reading at all (punctuation). It still occupies characters of the word,
                    // so it is recorded with an empty reading -- otherwise the pieces would not
                    // spell the word back and the 。 would vanish from the render.
                    ipa = "";
                }
                else
                {
                    // The span was withheld because a post-assembly rewrite moved the offsets, and
                    // Emitted is what the token PRODUCED, not necessarily what the reading says
                    // (fr-CA's accent, `as`'s aspirate collapse). Showing it would put a reading on
                    // screen that the synthesizer will not speak.
                    return null;
                }
                var word = oneForOne ? first + (t - i0) : first;
                // Which characters of that word this reading covers. Every token in the group
                // shares the one input span, so this is that span clipped to the word.
                var ws = Math.Max(input.Start, wordStart[word]);
                var we = Math.Min(input.End, wordStart[word] + words[word].Length);
                (parts[word] ??= new List<Contribution>()).Add(new Contribution(ws, we, ipa.Trim()));
            }
            i0 = i1 + 1;
        }

        var result = new WordRuby[words.Count];
        for (var i = 0; i < words.Count; i++)
        {
            var p = parts[i];
            var ipa = p is null ? "" : string.Join(' ', p.Select(c => c.Ipa).Where(x => x.Length > 0));
            result[i] = new WordRuby(ipa, p is null ? Array.Empty<RubyPiece>() : Pieces(p, text, wordStart[i], words[i]));
        }
        return result;
    }

    /// <summary>One token's contribution: the characters of the word it covers, and their reading.</summary>
    private readonly record struct Contribution(int Start, int End, string Ipa);

    /// <summary>
    /// The word's sub-pieces, or empty when it does not split. A word splits only when its tokens
    /// cover DISTINCT, in-order ranges of it -- "$3.14" is three readings of one range and stays
    /// stacked, while 東京都に/住んで/います are three ranges and become three pieces.
    ///
    /// ⚠ THE PIECES MUST SPELL THE WORD BACK, CHARACTER FOR CHARACTER. They replace the word in the
    /// render, so a character no piece covers is a character the reader loses: Chinese punctuation
    /// gets no token at all, and 我住在北京。 would have lost its 。 Characters between (or around)
    /// the tokens become pieces with no reading, which is what they are.
    ///
    /// A piece that is one all-Han run whose reading has exactly one group per character splits
    /// again, per character: that is pinyin over hanzi, the same annotation Chinese readers expect.
    /// </summary>
    private static IReadOnlyList<RubyPiece> Pieces(List<Contribution> parts, string text, int wordStart, string word)
    {
        // Only scriptio continua is segmented. A spaced language already has the boundaries its
        // readers use, and splitting "hello," into two pieces would churn every English render.
        if (!word.Any(IsContinua)) return Array.Empty<RubyPiece>();

        var wordEnd = wordStart + word.Length;
        var pieces = new List<RubyPiece>(parts.Count);
        var cursor = wordStart;
        foreach (var c in parts)
        {
            // Overlapping or out-of-order ranges are not a segmentation of anything; decline.
            if (c.Start < cursor || c.End <= c.Start || c.End > wordEnd) return Array.Empty<RubyPiece>();
            if (c.Start > cursor) AddPiece(text[cursor..c.Start], "");   // uncovered: punctuation
            AddPiece(text[c.Start..c.End], c.Ipa);
            cursor = c.End;
        }
        if (cursor < wordEnd) AddPiece(text[cursor..wordEnd], "");
        // One piece is not a segmentation -- that is just the word, and it renders as one.
        return pieces.Count > 1 ? pieces : Array.Empty<RubyPiece>();

        void AddPiece(string slice, string ipa)
        {
            var groups = ipa.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (slice.Length > 1 && groups.Length == slice.Length && slice.All(IsHan))
                for (var k = 0; k < slice.Length; k++)
                    Add(slice[k].ToString(), groups[k]);
            else if (TibetanSyllables(slice, ipa) is { } syllables)
                foreach (var (text, reading) in syllables) Add(text, reading);
            else
                Add(slice, ipa);
        }

        void Add(string text, string ipa) =>
            pieces.Add(new RubyPiece(text, ipa, ipa.Length == 0 ? 0 : OmniVoiceDuration.TotalWeight(ipa)));
    }

    /// <summary>Scripts written without spaces between words, where the render has to be segmented
    /// before a per-word annotation means anything: Han, kana, Thai, Lao, Khmer and Myanmar.</summary>
    private static bool IsContinua(char c) =>
        IsHan(c)
        || c is >= (char)0x3040 and <= (char)0x30FF      // hiragana + katakana
        || c is >= (char)0x0E00 and <= (char)0x0EFF      // Thai, Lao
        || c is >= (char)0x1000 and <= (char)0x109F      // Myanmar
        || c is >= (char)0x0F00 and <= (char)0x0FFF      // Tibetan
        || c is >= (char)0x1780 and <= (char)0x17FF;     // Khmer

    /// <summary>
    /// Tibetan paired syllable by syllable, or null when it does not pair.
    ///
    /// Tibetan writes no spaces between words but does mark every syllable, with a tsheg (་), and
    /// its reading marks every syllable too, with a tone letter. When the two counts agree the
    /// pairing is unambiguous, and a sentence that would otherwise carry its whole reading in one
    /// unreadable run gets a reading over each syllable — which is how Tibetan is annotated on the
    /// page. When they disagree (a number that reads as several syllables, say) this declines and
    /// the reading stays whole rather than sliding out of step.
    /// </summary>
    private static List<(string Text, string Ipa)>? TibetanSyllables(string slice, string ipa)
    {
        if (!slice.Any(IsTibetanLetter)) return null;

        // The tsheg follows its syllable, so it stays with it; so does any closing punctuation
        // (the shad ། ends a sentence, not a syllable of its own).
        var text = new List<string>();
        var start = 0;
        for (var i = 0; i < slice.Length; i++)
            if (slice[i] == '\u0F0B' && i + 1 < slice.Length)   // TIBETAN MARK INTERSYLLABIC TSHEG
            {
                text.Add(slice[start..(i + 1)]);
                start = i + 1;
            }
        if (start < slice.Length) text.Add(slice[start..]);
        if (text.Count < 2) return null;

        // The reading breaks after each run of tone letters.
        var read = new List<string>();
        start = 0;
        for (var i = 0; i < ipa.Length; i++)
            if (IsToneLetter(ipa[i]) && (i + 1 == ipa.Length || !IsToneLetter(ipa[i + 1])))
            {
                read.Add(ipa[start..(i + 1)]);
                start = i + 1;
            }
        if (start < ipa.Length) read.Add(ipa[start..]);

        if (read.Count != text.Count) return null;
        return text.Zip(read).ToList();
    }

    /// <summary>Tibetan consonants and vowel signs — not its punctuation, which is what the tsheg
    /// and the shad are.</summary>
    private static bool IsTibetanLetter(char c) => c is >= (char)0x0F40 and <= (char)0x0FBC;

    /// <summary>The tone letters the phonemizer ends a tonal syllable with (˥˦˧˨˩).</summary>
    private static bool IsToneLetter(char c) => c is >= (char)0x02E5 and <= (char)0x02E9;

    /// <summary>CJK ideographs -- the characters that carry one syllable each in Chinese.</summary>
    private static bool IsHan(char c) =>
        c is >= (char)0x4E00 and <= (char)0x9FFF or >= (char)0x3400 and <= (char)0x4DBF;

    /// <summary>
    /// Each token's span into <paramref name="text"/>, or null when they cannot be established.
    /// Reported spans are used as given. When every token is missing one, the surfaces are laid
    /// back over the input by length -- valid only if they account for every non-space character,
    /// which is what makes the length-preserving rewrites (は -> わ) recoverable and the
    /// length-changing ones (3800 -> its reading) declined.
    /// </summary>
    private static (int Start, int End)[]? InputSpans(PhonemeTrace trace, string text)
    {
        var spans = new (int Start, int End)[trace.Tokens.Count];
        var missing = 0;
        for (var i = 0; i < trace.Tokens.Count; i++)
        {
            if (trace.Tokens[i].InputSpan is { } s
                && s.Start >= 0 && s.End > s.Start && s.Start < text.Length)
                spans[i] = (s.Start, Math.Min(s.End, text.Length));
            else { missing++; spans[i] = (-1, -1); }
        }
        if (missing == 0) return spans;
        if (missing != trace.Tokens.Count) return null;   // a partial map desyncs everything after the gap

        var pos = 0;
        for (var i = 0; i < trace.Tokens.Count; i++)
        {
            while (pos < text.Length && char.IsWhiteSpace(text[pos])) pos++;
            var len = trace.Tokens[i].Surface.Length;
            if (len <= 0 || pos + len > text.Length) return null;
            spans[i] = (pos, pos + len);
            pos += len;
        }
        while (pos < text.Length && char.IsWhiteSpace(text[pos])) pos++;
        return pos == text.Length ? spans : null;   // surfaces must account for the whole input
    }
}
