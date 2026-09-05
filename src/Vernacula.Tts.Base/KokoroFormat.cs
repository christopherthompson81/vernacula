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

    // A punctuation token the phonemizer emitted on its own, with the space that precedes it.
    // Kokoro's training data attaches punctuation to the word before it (`wˈɜɹld.`), and the
    // word-alignment code counts a run of non-space tokens as one word, so it must not stand alone.
    private static readonly Regex DetachedPunctRe = new(@" +([,.;:!?…—]+)(?= |$)", RegexOptions.Compiled);

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
        return ps;
    }
}
