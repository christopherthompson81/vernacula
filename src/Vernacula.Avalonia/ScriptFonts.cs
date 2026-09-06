using System.Text;
using Avalonia.Media;

namespace Vernacula.App;

/// <summary>
/// Which font to set text in, by the script it is written in.
///
/// ⚠ THE SYSTEM FALLBACK IS NOT ALWAYS THE RIGHT ANSWER. Where the UI font has no glyphs, the
/// renderer picks a replacement itself, and for Tibetan it picked Unifont — confirmed by forcing
/// Unifont explicitly and getting the same broken line back. Unifont is scalable and covers very
/// nearly every codepoint, which is exactly why a fallback reaches for it, but it carries no
/// OpenType shaping: Tibetan's subjoined letters and vowel signs cannot stack, so the syllables
/// came out flat-topped and cut, in the reader and in the edit box alike. Coverage is not the
/// same thing as being able to render a script. Listing better families ahead of it in one
/// composite family does not help either — the fallback ignores the list — so the font is chosen
/// here, from the text, and set on the control.
///
/// Only scripts observed to fall back badly are listed. Anything else keeps the default, because
/// the system's own choice is right far more often than a list maintained here would be.
/// </summary>
public static class ScriptFonts
{
    /// <summary>
    /// The app's own font (Inter, from <c>WithInterFont</c>) with explicit fallbacks, used for text
    /// no entry below claims — which is nearly all of it.
    /// </summary>
    public static FontFamily Default { get; } = new("Inter, Noto Sans, DejaVu Sans, sans-serif");

    /// <summary>Inline code, which wants a monospace face whatever the script around it.</summary>
    public static FontFamily Mono { get; } = new("Cascadia Code, Consolas, Menlo, DejaVu Sans Mono, monospace");

    /// <summary>Faces that shape Tibetan properly, in order of preference; missing ones are
    /// skipped. Dzongkha and Ladakhi are written in the same script and get the same faces.</summary>
    private static readonly FontFamily Tibetan =
        new("Noto Serif Tibetan, Noto Sans Tibetan, Jomolhari, DDC Uchen, Tibetan Machine Uni, Kailasa, Microsoft Himalaya");

    /// <summary>
    /// The font for <paramref name="text"/>, chosen by the script MOST of it is written in.
    ///
    /// ⚠ NOT FIRST-HIT. This is asked about whole documents as well as single words, and a
    /// Tibetan face has little or no Latin: letting one quoted Tibetan sentence choose the font for
    /// an English document would push its entire body back into per-run fallback.
    /// </summary>
    public static FontFamily For(string? text)
    {
        if (string.IsNullOrEmpty(text)) return Default;
        var tibetan = 0;
        var other = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            if (!Rune.IsLetter(rune)) continue;
            if (rune.Value is >= 0x0F00 and <= 0x0FFF) tibetan++; else other++;
        }
        return tibetan > other ? Tibetan : Default;
    }

    /// <summary>
    /// How tall a line box has to be for <paramref name="text"/>, as a multiple of the font size.
    ///
    /// ⚠ ONE FACTOR DOES NOT FIT EVERY SCRIPT. Avalonia centres a run in its line box only while
    /// the box is the taller of the two; ask for less and it clamps the box and lets the descent
    /// spill over whatever is below. Tibetan stacks a syllable vertically — superscript, root,
    /// subscript, vowel sign — and needs far more room than the factor that suits Latin.
    /// </summary>
    public static double LineBoxFor(string? text) => For(text) == Tibetan ? 2.6 : 1.7;
}
