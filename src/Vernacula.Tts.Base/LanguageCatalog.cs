namespace Vernacula.Tts.Base;

/// <summary>One row of the language picker — the C# twin of the web demo's <c>LanguageOption</c>.</summary>
/// <param name="Code">vernacula-phonemizer language code.</param>
/// <param name="Name">English name.</param>
/// <param name="Native">The language's name as its own speakers write it, in its own script. Null
/// where that is the same as the English name, or where no source names this specific variety.</param>
/// <param name="Trained">In the IPA fine-tune's coverage set. Everything else renders, but extrapolated.</param>
/// <param name="Voice">Donor language for the reference voice — null when the language has a native one.</param>
public sealed record LanguageOption(string Code, string Name, string? Native, bool Trained, string? Voice)
{
    public override string ToString() => Name;
}

/// <summary>
/// The phonemizer's languages as the pickers list them. The rows are generated from the web demo's
/// catalogue (<c>LanguageCatalog.g.cs</c>) so the reader and the browser offer the same list; the
/// lookups and the search live here.
/// </summary>
public static partial class LanguageCatalog
{
    // ⚠ LAZY, NOT A FIELD INITIALIZER. `All` lives in the generated half of this partial class, and
    // static field initializers run in file order -- so an eager map here observed `All` as null
    // and every member of the class then threw TypeInitializationException.
    private static readonly Lazy<Dictionary<string, LanguageOption>> ByCodeMap =
        new(() => All.ToDictionary(l => l.Code, StringComparer.OrdinalIgnoreCase));

    public static LanguageOption? ByCode(string? code) =>
        code is not null && ByCodeMap.Value.TryGetValue(code.Trim(), out var l) ? l : null;

    /// <summary>
    /// Which language's REFERENCE VOICE renders this one — itself, or its donor. Cloning is
    /// acoustic: the reference carries the speaker's accent, so a donor is always a near neighbour
    /// (Faroese read by Icelandic, Sesotho by Sepedi) rather than a default English voice.
    /// </summary>
    /// <summary>
    /// True when <paramref name="code"/> is written right to left. Read from the language's own
    /// endonym, which is written in its own script -- so this stays correct as the catalog is
    /// regenerated, with no list of right-to-left codes to maintain alongside it.
    /// </summary>
    public static bool IsRightToLeft(string? code) =>
        ByCode(code ?? "") is { Native: var native } && TextDirection.IsRightToLeft(native);

    public static string VoiceLangOf(string code) => ByCode(code)?.Voice ?? code;

    /// <summary>
    /// Rank a language against a type-ahead query. Lower is better; -1 means no match. Port of the
    /// web demo's langSearch.ts, and the same argument applies: the endonym is SEARCHABLE, not
    /// decorative — someone looking for their own language types it the way they write it.
    /// </summary>
    public static int Score(LanguageOption l, string q)
    {
        var name = l.Name.ToLowerInvariant();
        var code = l.Code.ToLowerInvariant();
        var nat = l.Native?.ToLowerInvariant() ?? "";
        if (code == q) return 0;
        // An EXACT name match outranks a prefix match, or a language whose endonym merely starts
        // with another's loses to it ("Sesotho" vs "Sesotho sa Leboa").
        if (name == q || nat == q) return 0;
        if (name.StartsWith(q, StringComparison.Ordinal) || nat.StartsWith(q, StringComparison.Ordinal)) return 1;
        if (code.StartsWith(q, StringComparison.Ordinal)) return 2;
        if (WordStart(name, q)) return 3;
        if (nat.Length > 0 && (WordStart(nat, q) || nat.Contains(q, StringComparison.Ordinal))) return 3;
        if (name.Contains(q, StringComparison.Ordinal)) return 4;
        return -1;

        // "\b" + q, without the regex: q at the start of a word. Word starts are found on
        // whitespace and ASCII punctuation, so a non-Latin endonym gets the Contains path above.
        static bool WordStart(string hay, string q)
        {
            var i = 0;
            while ((i = hay.IndexOf(q, i, StringComparison.Ordinal)) >= 0)
            {
                if (i == 0 || !char.IsLetterOrDigit(hay[i - 1])) return true;
                i++;
            }
            return false;
        }
    }

    /// <summary>The picker's match list for a query: scored, filtered, best first, then alphabetical.</summary>
    public static IReadOnlyList<LanguageOption> Search(string? query)
    {
        var q = (query ?? "").Trim().ToLowerInvariant();
        if (q.Length == 0) return All;
        return All.Select(l => (score: Score(l, q), l))
            .Where(t => t.score >= 0)
            .OrderBy(t => t.score).ThenBy(t => t.l.Name, StringComparer.Ordinal)
            .Select(t => t.l).ToList();
    }
}
