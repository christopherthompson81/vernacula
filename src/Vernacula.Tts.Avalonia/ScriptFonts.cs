using System.Text;
using Avalonia.Media;

namespace Vernacula.Tts.App;

/// <summary>
/// Which font to set text in, by the script it is written in.
///
/// ⚠ THE SYSTEM FALLBACK IS NOT ALWAYS THE RIGHT ANSWER. Where the UI font has no glyphs, the
/// renderer picks a replacement itself, and for Tibetan it chose a bitmap fallback with no Tibetan
/// shaping: the stacked syllables came out flat-topped and cut, in the reader and in the edit box
/// alike, while a real Tibetan face renders them stacked and whole. Listing the families ahead of
/// the UI font does not help — the fallback ignores them — so the font is chosen here, from the
/// text, and set on the control.
///
/// Only scripts observed to fall back badly are listed. Anything else keeps the default, because
/// the system's own choice is right far more often than a list maintained here would be.
/// </summary>
public static class ScriptFonts
{
    /// <summary>The UI font, for text no entry below claims.</summary>
    public static FontFamily Default { get; } = new("Inter, Noto Sans, DejaVu Sans, sans-serif");

    /// <summary>Faces that shape Tibetan properly, in order of preference; missing ones are
    /// skipped. Dzongkha and Ladakhi are written in the same script and get the same faces.</summary>
    private static readonly FontFamily Tibetan =
        new("Noto Serif Tibetan, Noto Sans Tibetan, Jomolhari, DDC Uchen, Tibetan Machine Uni, Kailasa, Microsoft Himalaya");

    /// <summary>The font for <paramref name="text"/>, by the first script in it that names one.</summary>
    public static FontFamily For(string? text)
    {
        if (string.IsNullOrEmpty(text)) return Default;
        foreach (var rune in text.EnumerateRunes())
            if (rune.Value is >= 0x0F00 and <= 0x0FFF)   // Tibetan, including its marks and digits
                return Tibetan;
        return Default;
    }
}
