using Vernacula.Phonemizer;

namespace Vernacula.Tts.Base;

/// <summary>
/// One phonemized text: the Kokoro-alphabet string, and for each phoneme group in it (a run of
/// tokens between space tokens — one per spoken word) the index of the whitespace-delimited source
/// word it came from. <see cref="GroupSourceWords"/> is null when the phonemizer could not account
/// for every group; callers fall back to an even split.
/// </summary>
/// <param name="Words">The word units the map indexes into. Whitespace for most languages, the
/// trace's own segmentation for the ones that do not space — see <see cref="WordSegmentation"/>.
/// Carried on the result so a caller aligns against the SAME units the reader displays, rather
/// than splitting the text a second time and hoping the two agree.</param>
public sealed record KokoroPhonemization(
    string Phonemes, IReadOnlyList<int>? GroupSourceWords, IReadOnlyList<WordSpan> Words);

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

    /// <summary>Text → Kokoro-alphabet phoneme string, in <paramref name="lang"/>.</summary>
    public string ToPhonemes(string text, string lang) => Phonemize(text, lang).Phonemes;

    /// <summary>Inner phoneme-token count (excludes the 2 pad tokens) for <paramref name="text"/>.</summary>
    public int CountTokens(string text, bool british = false)
        => Math.Max(0, KokoroVocab.Encode(ToPhonemes(text, british)).Length - 2);

    /// <summary>The same count, phonemized as <paramref name="lang"/>.</summary>
    /// <remarks>
    /// ⚠ THE LANGUAGE MATTERS HERE AND A COMMENT ONCE CLAIMED IT DID NOT. The budget is in
    /// phonemes, and how many phonemes a paragraph becomes absolutely depends on which G2P read
    /// it: running the English one over kana or hanzi yields a number unrelated to the ja or cmn
    /// render that will actually be sent, so a chunk measured in English can be far past the
    /// engine's 510-symbol entry limit and the engine refuses the whole request.
    /// </remarks>
    public int CountTokens(string text, string lang)
        => Math.Max(0, KokoroVocab.Encode(ToPhonemes(text, lang)).Length - 2);

    /// <summary>Text → Kokoro phonemes plus the phoneme-group → source-word map.</summary>
    public KokoroPhonemization Phonemize(string text, bool british = false)
        => Phonemize(text, Lang(british));

    /// <summary>
    /// Text → Kokoro phonemes plus the group → source-word map, in <paramref name="lang"/>.
    /// </summary>
    /// <remarks>
    /// ⚠ THE RENDER TARGET IS PER LANGUAGE AND IS NOT A COURTESY. <see cref="KokoroFormat"/> has
    /// an arm for each — the English one collapses diphthongs into Kokoro's single-symbol
    /// convention, the others mostly drop notation this repo carries and Kokoro does not (the tie
    /// bar, the superscript off-glides) and decompose what Kokoro spells apart (Portuguese writes
    /// a nasal vowel precomposed; Kokoro carries the base vowel plus the combining tilde). Every
    /// one of the five Kokoro speaks lands entirely inside its 114-symbol vocabulary, measured
    /// over the phonemizer's own goldens.
    /// </remarks>
    public KokoroPhonemization Phonemize(string text, string lang)
    {
        // ⚠ ONE TRACE, HANDED TO BOTH. The map indexes into the words, so the two have to have been
        // read off the same spans — and this used to take a trace here and let Segment take another,
        // asserting they agreed. #1408 was a case of exactly that assumption failing. Also halves
        // the phonemization work per paragraph for the two languages that need a trace at all.
        var trace = WordSegmentation.Trace(text, lang);
        var words = WordSegmentation.Segment(text, 0, text.Length, lang, trace);
        var map = GroupSourceWords(trace, text, words);

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

        return new KokoroPhonemization(KokoroFormat.Render(ipa, lang), map, words);
    }

    /// <summary>
    /// One source-word index per spoken IPA group, from the trace's spans. A token's input span
    /// says which source word it was; its IPA span says how many groups it became (a number reads
    /// as several words). Null when any token is missing a span, since a partial map would assign
    /// the wrong words to every group after the gap.
    /// </summary>
    private static List<int>? GroupSourceWords(PhonemeTrace trace, string text, IReadOnlyList<WordSpan> words)
    {
        if (!trace.Traced || words.Count == 0) return null;

        // ⚠ CHARACTER OFFSET → WORD INDEX, OVER THE SUPPLIED UNITS RATHER THAN WHITESPACE. This
        // used to scan for whitespace itself, which meant a language without spaces had exactly
        // one word and every group mapped to it — the segmentation the trace had just produced was
        // thrown away one line after it arrived.
        //
        // A character between two words (a space, or the 。 that follows a Japanese phrase) takes
        // the index of the word that FOLLOWS it, which is what the old scan did for whitespace and
        // is what keeps a token starting on a separator attached to the right side.
        var wordAt = new int[text.Length + 1];
        var next = 0;
        for (var i = 0; i < text.Length; i++)
        {
            if (next < words.Count && i >= words[next].End) next++;
            wordAt[i] = next < words.Count && i >= words[next].Start ? next : Math.Min(next, words.Count - 1);
        }
        wordAt[text.Length] = words.Count - 1;
        var wordCount = words.Count;

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
                // ⚠ A TOKEN THAT SAYS NOTHING CONTRIBUTES NOTHING, and this used to abandon the
                // whole map. In English punctuation rides on the word before it and never becomes
                // a token of its own, so the case never arose; a Japanese sentence ends with 。as
                // its own token with no IPA at all, which nulled the map for every Japanese
                // paragraph and sent the aligner to an even split.
                continue;
            if (groups == 0) continue;

            var lastInSpan = wordAt[Math.Clamp(input.End - 1, 0, text.Length)];
            var first = wordAt[input.Start];

            // ⚠ ONE TOKEN CAN COVER SEVERAL WORD UNITS, which is how Mandarin arrives: a single
            // token spans the sentence and carries one group per syllable, and WordSegmentation
            // has already cut that span into one unit per hanzi on the same count. Distributing
            // the groups across them is what makes the two agree — without it all six syllables
            // mapped to word 0 and the highlight covered the sentence.
            if (lastSpan != input && groups > 1 && lastInSpan - first + 1 == groups)
            {
                for (var g = 0; g < groups; g++) map.Add(first + g);
                lastSpan = input; lastWord = lastInSpan;
                continue;
            }

            // Tokens that share one input span came from one normalizer rewrite. When the span is
            // one written word ("$3.14" → three, dollars, fourteen) they all belong to it; when it
            // covers several ("Mr. Smith" → mister, Smith) each successive token takes the next
            // word in the span, so the highlight moves with the speech instead of sticking.
            var word = lastSpan == input ? Math.Min(lastWord + 1, lastInSpan) : first;
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
