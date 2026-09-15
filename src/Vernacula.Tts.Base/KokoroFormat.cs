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
        // English symbols that reach other languages through the phonemizer's foreign-run
        // delegation — an English name inside a Japanese or Chinese sentence is read by English.
        // Rare (one or two per corpus) but they would otherwise be the only thing out of vocab.
        // ⚠ NOT ɚ, which is in the vocabulary and which Mandarin needs: er4 is ɚ↘.
        ("ɫ", "l"), ("ɝ", "ɜɹ"),
    ];

    /// <summary>
    /// Per-language collapses, for distinctions Kokoro's alphabet genuinely cannot carry — as
    /// opposed to ones English happens not to make.
    /// </summary>
    private static readonly Dictionary<string, (string Old, string New)[]> LanguageRules = new()
    {
        // ── Mandarin ─────────────────────────────────────────────────────────
        // Target taken from the engine's own pinyin table (g2p/zh.json), whose inventory is
        //     a e f h i j k l m n o p s t u w x y ŋ ɔ ɕ ə ɚ ɛ ɤ ɥ ɨ ɻ ʂ ʦ ʨ ʰ → ↓ ↗ ↘ ꭧ
        // Tone is handled separately below, because it MOVES rather than maps.
        ["cmn"] =
        [
            ("ʈʂ", "ꭧ"),          // zhong1 = ꭧʊ→ŋ
            ("ts", "ʦ"), ("tɕ", "ʨ"),
            ("ʐ", "ɻ"),           // ri4 = ɻɨ↘
            ("ɹ̩", "ɨ"), ("ɹ", "ɨ"),   // the apical vowel: si1 = sɨ→, shi1 = ʂɨ→
            ("\u0329", ""),        // any syllabic mark the pair above did not consume
            ("ᵘ", "u"), ("ⁱ", "i"),
            ("ɑ", "a"), ("æ", "ɛ"),
        ],

        // ── Japanese ─────────────────────────────────────────────────────────
        // Target taken from the engine's own kana table (g2p/ja.json in a multilingual GGUF),
        // whose entire output inventory is
        //     a b d e h i j k m n o p s t v z ç ɕ ɡ ɨ ɯ ɲ ɸ ɾ ʣ ʥ ʦ ʨ ʲ β ᵝ
        // Our transcription is narrower than that in three ways, and each one has to go
        // somewhere Kokoro actually saw during training.
        ["ja"] =
        [
            // Affricates are single ligatures there (つ = ʦɨ, ち = ʨi, じ = ʥi). The tie bar is
            // already gone by this point, so these are the bare sequences.
            ("ts", "ʦ"), ("tɕ", "ʨ"), ("dʑ", "ʥ"),
            // ⚠ ʑ ALONE IS NOT IN KOKORO'S VOCABULARY, though ɕ is. Their table spells じ as the
            // affricate ʥ, so a bare ʑ goes there rather than to ʒ — matching what the model was
            // trained on beats matching the IPA more closely.
            ("ʑ", "ʥ"),
            // Narrow-transcription detail with no token: centralised ä, the lowering diacritic
            // under e̞ o̞, and the uvular nasal their table never emits.
            ("ä", "a"), ("\u031e", ""), ("ɴ", "n"),
            // ⚠ PITCH ACCENT HAS NOWHERE TO GO. Kokoro's Japanese does not encode it — there is
            // no downstep, and none of → ↓ ↗ ↘ appears anywhere in its kana table, though the
            // vocabulary carries them for Mandarin tone. Dropping the mark is lossy and there is
            // no alternative: inventing a token the model never saw would be worse than losing a
            // distinction it never learned.
            ("ꜜ", ""),
        ],

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

        if (lang == "cmn") ps = PlaceMandarinTone(NormalizeMandarinSyllables(ps));

        ps = DecomposeUnknown(ps);
        ps = DetachedPunctRe.Replace(ps, "$1");
        return ps;
    }

    /// <summary>
    /// Brings a Mandarin syllable into the shape the engine's pinyin table writes, before the
    /// tone is placed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Scored against that table — 993 syllables taken from the golden corpus, their pinyin from
    /// pypinyin and their phonemes from g2p/zh.json — rather than reasoned about. Each rule below
    /// closed a mismatch class the score named; without it these would be plausible guesses that
    /// land in-vocabulary and sound wrong, which is the failure this whole exercise keeps hitting.
    /// </para>
    /// </remarks>
    private static string NormalizeMandarinSyllables(string ps)
    {
        var syllables = ps.Split(' ');
        for (var i = 0; i < syllables.Length; i++)
        {
            var syl = syllables[i];
            if (syl.Length == 0) continue;

            // A zero-initial syllable whose glide merely duplicates its own vowel is written
            // without it: yi is i, wu is u, yu is y. ⚠ ONLY when they match — wang is waŋ and yao
            // is jau, so dropping every initial glide loses the onset outright. Where it IS
            // dropped, the vowel behind it gets re-glided below if something follows, which turns
            // our jiou into their jou and our jyɛn into their ɥɛn.
            if (syl.Length > 1 &&
                ((syl[0] == 'j' && (syl[1] == 'i' || syl[1] == 'y')) || (syl[0] == 'w' && syl[1] == 'u')))
                syl = syl[1..];

            // er is r-coloured, not a schwa with an r after it.
            syl = syl.Replace("ər", "ɚ");
            // -un after a palatal is y there, where we write a rounded vowel plus a schwa.
            // ⚠ Keyed on y, not ɥ: the glide conversion happens in the loop below, so at this
            // point the syllable still spells it y.
            syl = syl.Replace("yə", "y").Replace("yi", "y");

            var sb = new System.Text.StringBuilder(syl.Length + 2);
            for (var k = 0; k < syl.Length; k++)
            {
                var c = syl[k];
                var next = k + 1 < syl.Length ? syl[k + 1] : '\0';
                var prev = sb.Length > 0 ? sb[^1] : '\0';

                // Prenuclear high vowels are glides: ʨia -> ʨja, tuan -> twan, ɕye -> ɕɥe.
                if (IsMandarinNucleus(next))
                {
                    if (c == 'i') { sb.Append('j'); continue; }
                    if (c == 'u') { sb.Append('w'); continue; }
                    if (c == 'y') { sb.Append('ɥ'); continue; }
                }
                // The apical vowel, which only ever follows a retroflex or sibilant initial. Our
                // transcription keeps it retroflex (ʂɻ); theirs writes ɨ (ʂɨ, ɻɨ, sɨ). The
                // aspiration mark sits between the two for chi/ci, so it is looked past.
                var onset = prev == 'ʰ' && sb.Length > 1 ? sb[^2] : prev;
                if (c == 'ɻ' && (onset == 'ʂ' || onset == 'ꭧ' || onset == 'ɻ' || onset == 's'))
                {
                    sb.Append('ɨ');
                    continue;
                }
                // -ong is ʊŋ there, not oŋ.
                if (c == 'o' && next == 'ŋ') { sb.Append('ʊ'); continue; }
                // A labial before o carries a rounded glide: mo is mwo. Only in an open
                // syllable though — mou stays mou.
                if (c == 'o' && !IsMandarinNucleus(next) &&
                    (prev == 'm' || prev == 'p' || prev == 'f' || prev == 'ʰ'))
                {
                    sb.Append('w').Append('o');
                    continue;
                }
                // ⚠ AFTER A PALATAL GLIDE THE MID VOWEL'S HEIGHT FOLLOWS THE CODA, which is why
                // this is not one rule: xie is ɕje with an open syllable, yuan is ɥɛn before the
                // nasal. Getting it backwards merges different finals.
                if (c == 'ɛ' && (prev == 'ɥ' || prev == 'j'))
                {
                    bool nasalCoda = next == 'n' || next == 'ŋ';
                    sb.Append(nasalCoda ? 'ɛ' : 'e');
                    continue;
                }
                sb.Append(c);
            }
            syllables[i] = sb.ToString();
        }
        return string.Join(' ', syllables);
    }

    private static bool IsMandarinNucleus(char c) => MandarinNuclei.IndexOf(c) >= 0;

    // A run of tone letters, always at the end of a syllable in our transcription.
    private static readonly Regex ToneRunRe = new(@"[\u02e5-\u02e9]+", RegexOptions.Compiled);

    private const string MandarinNuclei = "aeiouyɛɤəɨʊɔɚ";

    /// <summary>
    /// Rewrites Mandarin tone from IPA tone letters into the arrows Kokoro carries, and MOVES it
    /// to where Kokoro puts it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ⚠ THE ARROW GOES AFTER THE NUCLEUS, NOT AFTER THE SYLLABLE, which is why this cannot be a
    /// row in the table above. From the engine's own pinyin table: <c>zhong1 = ꭧʊ→ŋ</c> and
    /// <c>yan1 = jɛ→n</c> — the mark sits before the coda. Our transcription writes the tone
    /// letters at the end of the whole syllable, so the contour has to be lifted off and
    /// reinserted after the last vowel.
    /// </para>
    /// <para>
    /// The five contours, read off that table: 55 level → <c>→</c>, 35 rising → <c>↗</c>,
    /// 214 dipping → <c>↓</c>, 51 falling → <c>↘</c>, and the neutral tone carries no mark at all
    /// (<c>shi5 = ʂɨ</c>).
    /// </para>
    /// </remarks>
    private static string PlaceMandarinTone(string ps)
    {
        var syllables = ps.Split(' ');
        for (var i = 0; i < syllables.Length; i++)
        {
            var match = ToneRunRe.Match(syllables[i]);
            if (!match.Success) continue;                       // neutral tone: no mark, correctly
            var arrow = ToneArrow(match.Value);
            var bare = ToneRunRe.Replace(syllables[i], "");
            if (arrow.Length == 0) { syllables[i] = bare; continue; }

            var nucleus = bare.LastIndexOfAny(MandarinNuclei.ToCharArray());
            // A syllable with no vowel we recognise keeps its tone at the end rather than losing
            // it: wrong position beats absent contour.
            syllables[i] = nucleus < 0 ? bare + arrow : bare.Insert(nucleus + 1, arrow);
        }
        return string.Join(' ', syllables);
    }

    private static string ToneArrow(string letters) => letters switch
    {
        "\u02e5\u02e5" => "→",                       // 55 high level
        "\u02e7\u02e5" => "↗",                       // 35 rising
        "\u02e8\u02e9\u02e6" => "↓",                 // 214 dipping
        "\u02e5\u02e9" => "↘",                       // 51 falling
        // Anything else: fall back on the contour's own shape, so a transcription variant still
        // lands on a real tone rather than dropping one.
        _ => letters.Length < 2 ? ""
            : letters[^1] > letters[0] ? "↗"
            : letters[^1] < letters[0] ? "↘"
            : "→",
    };

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
