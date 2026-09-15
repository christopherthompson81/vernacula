using System.Text.RegularExpressions;

namespace Vernacula.Tts.Base;

/// <summary>
/// Renders vernacula-phonemizer's canonical IPA into the phoneme alphabet hexgrad/Kokoro-82M was
/// trained on. Kokoro's alphabet is misaki's: espeak-ng IPA after the deterministic post-processing
/// in misaki's <c>EspeakFallback.__call__</c> (diphthongs collapsed to single letters, the rhotic
/// schwa split, en-us length marks dropped). This is that post-processing, re-keyed on what
/// vernacula-phonemizer actually emits instead of what espeak does — which differs in five places:
///
///   · offglides are SUPERSCRIPT (<c>oᶷ eᶦ aᶦ aᶷ ɔᶦ</c>, en-GB <c>əᶷ</c>) rather than digraphs;
///   · affricates carry a tie bar (<c>d͡ʒ t͡ʃ</c>), stops carry aspiration (<c>tʰ</c>), and l is
///     dark (<c>ɫ</c>) where espeak wrote plain <c>dʒ tʃ t l</c>;
///   · the en-us flap is <c>t̬</c> / <c>d̬</c> (voicing diacritic), not <c>ɾ</c>;
///   · the stressed rhotic vowel is <c>ɝ</c> (espeak: <c>ɜː</c>), the unstressed one <c>ɚ</c>;
///   · clause punctuation SURVIVES, as its own space-delimited token (<c>… dˈɔːɡ , dˈʌzənt</c>),
///     where espeak collapsed it — so the punctuation Kokoro uses for pauses is re-attached here
///     rather than reconstructed from the source text.
///
/// Kokoro tolerates slightly different IPA, so byte-exact misaki parity is not the bar; the bar is
/// that every output codepoint is in <see cref="KokoroVocab"/> and the common words land on the
/// same tokens misaki would give them. docs/kokoro_vphon_investigation.md has the measurements.
/// </summary>
public static class KokoroFormat
{
    // Ordered sequential replacements, like misaki's E2M. Where one key is a prefix of another
    // the longer runs first (a diphthong before its bare offglide).
    private static readonly (string Old, string New)[] Common =
    [
        ("͡", ""),       // tie bar: d͡ʒ → dʒ, t͡ʃ → tʃ, consumed below
        ("ʰ", ""),            // aspiration: tʰ → t
        ("ʲ", ""),            // palatal glide: iʲə → iə (misaki's lexicon: jʊɹˈAniəm)
        ("t̬", "T"),     // flapped t → Kokoro's flap token
        ("d̬", "d"),     // flapped d: misaki's lexicon keeps d (θˈɜɹdi)
        ("ɫ", "l"),
        ("oᶷ", "O"),          // misaki o^ʊ
        ("eᶦ", "A"),          // misaki e^ɪ
        ("aᶦ", "I"),          // misaki a^ɪ
        ("aᶷ", "W"),          // misaki a^ʊ
        ("ɔᶦ", "Y"),          // misaki ɔ^ɪ
        ("ᶦ", "ɪ"),           // any offglide not consumed by a diphthong above
        ("ᶷ", "ʊ"),
        ("dʒ", "ʤ"),
        ("tʃ", "ʧ"),
        ("ɝ", "ɜɹ"),          // NURSE: misaki writes ɜɹ for en-us
        ("ɚ", "əɹ"),          // misaki ɚ → əɹ
        ("ɐ", "ə"),
        ("r", "ɹ"),
        ("x", "k"),
        ("ç", "k"),
        ("ɬ", "l"),
        ("̃", ""),       // nasalisation tilde
        ("ʔ", "t"),
        ("ɾ", "T"),
        // ᵻ (U+1D7B) is deliberately unmapped: it is Kokoro vocab id 177, not out-of-vocab.
    ];

    // A flap whose following vowel is WORD-FINAL and unstressed, realised as the tap ɾ (vocab 125)
    // rather than the flap token T (vocab 36). Kokoro's duration predictor over-allocates T in this
    // position — measured at 75ms against a 50ms vowel, where a word-INTERNAL flap gets 50ms — and a
    // word-final vowel has no following segment to prop it up, so the vowel ends up shorter than the
    // consonant before it and detaches from the word ("data" heard as "date" plus a stray schwa).
    // ⚠ The guard is the LOOKAHEAD, and it is load-bearing: word-internal flaps (writer, meeting,
    // better, related) must keep T, because T is what preserves the underlying /t/ — mapping them to
    // d instead would merge writer/rider, latter/ladder, metal/medal at the token level. A stressed
    // final vowel never matches either, since its stress mark sits between the T and the nucleus.
    // Scope is ~1,356 lexicon words: the -a nouns (data, beta, meta) and, far more numerous, the
    // -ity/-y family (city, quality, activity). docs/investigations/kokoro_word_final_flap_investigation.md.
    private static readonly Regex WordFinalFlapRe =
        new(@"T([əɐaeiouɑɔɛɪʊʌæɜAIOWYᵻ])(?=[ ,.;:!?…—]|$)", RegexOptions.Compiled);

    // A punctuation token the phonemizer emitted on its own, with the space that precedes it.
    // Kokoro's training data attaches punctuation to the word before it (`wˈɜɹld.`), and the
    // word-alignment code counts a run of non-space tokens as one word, so it must not stand alone.
    private static readonly Regex DetachedPunctRe = new(@" +([,.;:!?…—]+)(?= |$)", RegexOptions.Compiled);

    // ── Everything above this point is ENGLISH ───────────────────────────────
    // The tables and regexes above encode English phonology, not Kokoro's alphabet, and applying
    // them to another language is silently destructive rather than merely approximate. Measured
    // over 60 golden sentences per language:
    //
    //   es   x→k ×62 flattens the jota (jamón → kamón); r→ɹ ×59 and ɾ→T ×458 merge the trill
    //        and the tap, so perro and pero stop contrasting
    //   fr   the nasalisation tilde stripped ×400 — phonemic in French
    //   it   r→ɹ ×502
    //   hi   ʰ stripped ×125 — aspiration is PHONEMIC in Hindi (क vs ख), not allophonic as in
    //        English; ɾ→T ×450; the tilde ×242
    //   pt   ɐ→ə ×382 and the tilde ×271, both phonemic
    //
    // ⚠ EVERY ONE OF THOSE REWRITES A SYMBOL KOKORO'S VOCABULARY ALREADY CARRIES — r, ɹ, ɾ, x,
    // ɐ and the combining tilde are all in it. The collapses are English conveniences (English
    // has no trill, and its ɾ really is an allophone of /t/), not limits of the model.
    //
    // So the two paths are kept apart rather than parameterised: the English one is left exactly
    // as it was, because it is correct and byte-for-byte verified against the goldens, and the
    // other languages get the alphabet conventions WITHOUT the allophone collapses.

    /// <summary>The Kokoro alphabet's own conventions, which hold whatever the language is.</summary>
    /// <remarks>
    /// audio.cpp applies the same tie-collapsing table to every eSpeak language it drives
    /// (<c>espeak_text()</c>), so these are properties of the alphabet Kokoro was trained on
    /// rather than of English.
    /// </remarks>
    private static readonly (string Old, string New)[] AlphabetConventions =
    [
        ("\u0361", ""),       // tie bar: d͡ʒ → dʒ, consumed just below
        ("oᶷ", "O"), ("eᶦ", "A"), ("aᶦ", "I"), ("aᶷ", "W"), ("ɔᶦ", "Y"),
        ("ᶦ", "ɪ"), ("ᶷ", "ʊ"),   // an offglide no diphthong claimed
        ("dʒ", "ʤ"), ("tʃ", "ʧ"),
        // The voicing diacritic has no token of its own. English reads t̬ as its flap and maps it
        // to T; everywhere else it simply means "voiced", and the voiced counterpart is the
        // nearest thing the vocabulary holds. Stripping it instead would leave t, which is the
        // opposite sound.
        ("t\u032c", "d"), ("d\u032c", "d"),
    ];

    /// <summary>
    /// Per-language collapses, for distinctions Kokoro's alphabet genuinely cannot carry — as
    /// opposed to ones English happens not to make.
    /// </summary>
    private static readonly Dictionary<string, (string Old, string New)[]> LanguageRules = new()
    {
        // Hindi writes ह as the VOICED glottal fricative and marks breathy voice and dental
        // place; Kokoro's alphabet has none of the three. h and ʰ are the nearest it holds, and
        // the dental bridge has no counterpart at all, so it goes. Aspiration is NOT stripped
        // here the way it is for English: क/ख is a phonemic contrast, not an allophone.
        ["hi"] = [("ɦ", "h"), ("ʱ", "ʰ"), ("\u032a", "")],
    };

    /// <summary>
    /// Render canonical IPA from vernacula-phonemizer into a Kokoro-vocab phoneme string, for the
    /// language it was phonemized as. <paramref name="lang"/> is a phonemizer code
    /// (<c>en</c>, <c>en-GB</c>, <c>es</c>, <c>fr</c>, <c>hi</c>, <c>it</c>, <c>pt-BR</c>).
    /// </summary>
    public static string Render(string ipa, string lang)
    {
        if (string.IsNullOrEmpty(ipa)) return ipa ?? string.Empty;
        return lang is "en" or "en-GB" or "en-US" or null
            ? Render(ipa, british: lang == "en-GB")
            : RenderNonEnglish(ipa, lang);
    }

    private static string RenderNonEnglish(string ipa, string lang)
    {
        var ps = ipa.Trim();
        foreach (var (old, neu) in AlphabetConventions) ps = ps.Replace(old, neu);
        if (LanguageRules.TryGetValue(lang, out var extra))
            foreach (var (old, neu) in extra) ps = ps.Replace(old, neu);

        ps = DecomposeUnknown(ps);
        ps = DetachedPunctRe.Replace(ps, "$1");
        return ps;
    }

    /// <summary>
    /// Last resort for a codepoint the vocabulary has no id for: if Unicode decomposes it into
    /// pieces the vocabulary DOES carry, use those.
    /// </summary>
    /// <remarks>
    /// This is not a nicety — it is the whole of Portuguese. Our IPA writes nasal vowels
    /// precomposed (õ ĩ ũ ẽ) and Kokoro carries the base vowels plus the combining tilde
    /// (U+0303, token 17), so the same sound is spelled one way here and another there.
    /// Decomposing is exact; dropping the character would delete the nasality, and dropping the
    /// vowel would delete the syllable.
    /// </remarks>
    private static string DecomposeUnknown(string ps)
    {
        if (ps.All(KokoroVocab.Contains)) return ps;
        var sb = new System.Text.StringBuilder(ps.Length + 8);
        foreach (var c in ps)
        {
            if (KokoroVocab.Contains(c)) { sb.Append(c); continue; }
            var decomposed = c.ToString().Normalize(System.Text.NormalizationForm.FormD);
            // All-or-nothing: half a decomposition is a different sound, not a closer one.
            if (decomposed.Length > 1 && decomposed.All(KokoroVocab.Contains)) sb.Append(decomposed);
            else sb.Append(c);   // left in, so the engine's own validation names it
        }
        return sb.ToString();
    }

    /// <summary>
    /// Render canonical IPA from vernacula-phonemizer (<c>en</c> or <c>en-GB</c>) into a Kokoro-vocab
    /// phoneme string. Set <paramref name="british"/> for text phonemized as en-GB (lang_code 'b',
    /// the bf_/bm_ voices); default is en-us ('a').
    /// </summary>
    public static string Render(string ipa, bool british = false)
    {
        if (string.IsNullOrEmpty(ipa)) return ipa ?? string.Empty;

        var ps = ipa.Trim();
        if (british)
            ps = ps.Replace("əᶷ", "Q");    // misaki ə^ʊ — before Common turns the ᶷ into ʊ

        foreach (var (old, neu) in Common)
            ps = ps.Replace(old, neu);

        if (british)
        {
            ps = ps.Replace("ɛə", "ɛː");   // misaki e^ə (SQUARE); en-gb keeps its length marks
        }
        else
        {
            ps = ps.Replace("ː", "");      // en-us drops length marks
        }

        ps = ps.Replace("o", "ɔ");         // misaki: espeak < 1.52 compatibility; O is already consumed
        ps = DetachedPunctRe.Replace(ps, "$1");
        // en-us only: flapping is American, and en-GB never produces the T token.
        if (!british) ps = WordFinalFlapRe.Replace(ps, "ɾ$1");   // after punctuation, so the lookahead sees it
        return ps;
    }
}
