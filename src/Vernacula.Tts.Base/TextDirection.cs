using System.Text;

namespace Vernacula.Tts.Base;

/// <summary>
/// Which way a run of text reads. Needed because the reader lays each word out as its own control:
/// a browser applies the bidirectional algorithm across inline elements and gets Arabic or Hebrew
/// word order right for free, while a layout panel just places children in order, so the direction
/// has to be decided here and handed to the panel.
/// </summary>
public static class TextDirection
{
    /// <summary>What directions a run of text actually contains, and which came first.</summary>
    /// <param name="HasRtl">Any strongly right-to-left character.</param>
    /// <param name="HasLtr">Any strongly left-to-right character.</param>
    /// <param name="FirstStrongIsRtl">The direction of the first strong character; null if none.</param>
    public readonly record struct DirectionCensus(bool HasRtl, bool HasLtr, bool? FirstStrongIsRtl);

    /// <summary>Count the strong characters of each direction, and note the first.</summary>
    public static DirectionCensus Census(string? text)
    {
        var rtl = false;
        var ltr = false;
        bool? first = null;
        if (!string.IsNullOrEmpty(text))
            foreach (var rune in text.EnumerateRunes())
            {
                if (!Rune.IsLetter(rune)) continue;   // digits, punctuation and marks are not strong
                var isRtl = IsRtlScript(rune.Value);
                first ??= isRtl;
                if (isRtl) rtl = true; else ltr = true;
            }
        return new DirectionCensus(rtl, ltr, first);
    }

    /// <summary>
    /// Which way <paramref name="text"/> should be laid out.
    ///
    /// Text of one direction answers for itself. Mixed text does not — and counting characters to
    /// settle it does not work, because scripts differ in how many characters a word takes:
    /// "برنامه Text To Speech است" is a Persian sentence with an English product name in it, and
    /// the English half has more letters. So the tie is broken by <paramref name="languageIsRtl"/>,
    /// the language the document is being read in, which is the one thing that actually knows.
    /// Without it, the first strong character decides, as HTML's dir="auto" does.
    /// </summary>
    public static bool Resolve(string? text, bool? languageIsRtl = null)
    {
        var (hasRtl, hasLtr, firstIsRtl) = Census(text);
        if (hasRtl != hasLtr) return hasRtl;              // all one direction (or none at all)
        if (!hasRtl) return languageIsRtl ?? false;       // neutral text: the language, or default
        return languageIsRtl ?? firstIsRtl ?? false;      // mixed: the language, else first-strong
    }

    /// <summary>Convenience for text with no language context — <see cref="Resolve"/> with none.</summary>
    public static bool IsRightToLeft(string? text) => Resolve(text, null);

    /// <summary>
    /// The direction a single word forces, or null when it forces none — digits, punctuation and
    /// symbols, which read with whatever surrounds them.
    /// </summary>
    public static bool? StrongDirectionOf(string? word)
    {
        if (string.IsNullOrEmpty(word)) return null;
        foreach (var rune in word.EnumerateRunes())
            if (Rune.IsLetter(rune))
                return IsRtlScript(rune.Value);
        return null;
    }

    /// <summary>
    /// A word that is a number rather than a word — no letters, at least one digit.
    ///
    /// ⚠ NOT THE SAME AS NEUTRAL. Punctuation between two scripts belongs to whichever surrounds
    /// it, but a number after a left-to-right word stays with it: "iPhone 15" in a Persian line
    /// reads "iPhone 15", not "15 iPhone". That is the bidirectional algorithm's W7, and treating
    /// digits as neutral put the number on the wrong side of the phrase.
    /// </summary>
    public static bool IsNumberWord(string? word)
    {
        if (string.IsNullOrEmpty(word)) return false;
        var digit = false;
        foreach (var rune in word.EnumerateRunes())
        {
            if (Rune.IsLetter(rune)) return false;
            digit |= Rune.IsDigit(rune);
        }
        return digit;
    }

    /// <summary>
    /// The scripts written right to left, as blocks. Callers ask this only about LETTERS, which is
    /// what keeps the Arabic-Indic digits, the Arabic comma and question mark, and the Hebrew
    /// pointing marks out of the answer: they live in these blocks but are numbers, neutrals and
    /// combining marks, and none of them says which way the text around them runs.
    /// </summary>
    private static bool IsRtlScript(int c) =>
        c is >= 0x0590 and <= 0x05FF        // Hebrew
        or >= 0x0600 and <= 0x06FF          // Arabic
        or >= 0x0700 and <= 0x074F          // Syriac
        or >= 0x0750 and <= 0x077F          // Arabic Supplement
        or >= 0x0780 and <= 0x07BF          // Thaana (Dhivehi)
        or >= 0x07C0 and <= 0x07FF          // NKo
        or >= 0x0800 and <= 0x083F          // Samaritan
        or >= 0x0840 and <= 0x085F          // Mandaic
        or >= 0x0860 and <= 0x08FF          // Syriac Supplement, Arabic Extended-A/B
        or >= 0xFB1D and <= 0xFB4F          // Hebrew presentation forms
        or >= 0xFB50 and <= 0xFDFF          // Arabic presentation forms-A
        or >= 0xFE70 and <= 0xFEFF          // Arabic presentation forms-B
        or >= 0x10800 and <= 0x10FFF        // Cypriot, Phoenician, Kharoshthi, Old Persian, ...
        or >= 0x1E800 and <= 0x1EFFF;       // Mende Kikakui, Adlam, Arabic Mathematical
}
