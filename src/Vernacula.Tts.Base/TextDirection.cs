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
                bool isRtl;
                if (IsStrongRtl(rune.Value)) isRtl = true;
                else if (Rune.IsLetter(rune)) isRtl = false;
                else continue;
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
        {
            if (IsStrongRtl(rune.Value)) return true;
            if (Rune.IsLetter(rune)) return false;
        }
        return null;
    }

    /// <summary>
    /// The scripts written right to left. Digits and punctuation are deliberately absent: they are
    /// weak or neutral, take their direction from the text around them, and counting them would let
    /// a phone number decide a paragraph's direction.
    /// </summary>
    private static bool IsStrongRtl(int c) =>
        // Arabic-Indic and extended Arabic-Indic digits, and the separators that go with them, sit
        // inside the Arabic block but are NUMBERS, not letters: Unicode gives them a weak class
        // and they take direction from their surroundings like any other digit.
        !(c is >= 0x0660 and <= 0x066C or >= 0x06F0 and <= 0x06F9)
        && c is >= 0x0590 and <= 0x05FF     // Hebrew
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
