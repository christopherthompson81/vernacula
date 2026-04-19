using Vernacula.Base;

namespace Vernacula.App.Models;

/// <summary>
/// One selectable KenLM option for Parakeet shallow fusion. Maps a stable
/// key persisted in <see cref="AppSettings.ParakeetLmSelection"/> to a
/// user-facing label and (for built-in options) the HF file we download
/// on demand. <c>None</c> and <c>Custom</c> have no HF file.
/// </summary>
public sealed record KenLmOption(
    string  Key,
    string  DisplayName,
    string? RemoteFileName,
    long?   ExpectedSizeBytes);

public static class KenLmCatalog
{
    public const string KeyNone   = "none";
    public const string KeyCustom = "custom";

    /// <summary>
    /// Built-in options, ordered as they should appear in the Settings
    /// dropdown. Keep <c>None</c> first and <c>Custom</c> last. New
    /// languages/domains slot between them.
    /// </summary>
    public static readonly IReadOnlyList<KenLmOption> All =
    [
        new(KeyNone,       "None (greedy / beam only)",                    null,                 null),
        new("en-general",  "English — General (conversational)",           "en-general.arpa.gz", 69_980_766L),
        new("en-medical",  "English — Medical (clinical dictation + dialogue)", "en-medical.arpa.gz", 66_389_428L),
        new(KeyCustom,     "Custom ARPA file…",                            null,                 null),
    ];

    public static KenLmOption? Find(string? key) =>
        string.IsNullOrEmpty(key)
            ? null
            : All.FirstOrDefault(o => o.Key == key);

    /// <summary>
    /// Resolves the active LM path the decoder should load, based on the
    /// user's saved selection. Returns null when fusion should be off
    /// (selection is <c>None</c>, a built-in isn't downloaded yet, or a
    /// custom path doesn't exist). Callers should treat null as "decoder
    /// runs without fusion".
    /// </summary>
    public static string? ResolvePath(AppSettings settings, string kenLmDir)
    {
        string sel = settings.ParakeetLmSelection ?? KeyNone;
        if (sel == KeyNone) return null;

        if (sel == KeyCustom)
        {
            string p = settings.ParakeetLmPath ?? "";
            return !string.IsNullOrWhiteSpace(p) && File.Exists(p) ? p : null;
        }

        var opt = Find(sel);
        if (opt?.RemoteFileName is null) return null;

        string localPath = Path.Combine(kenLmDir, opt.RemoteFileName);
        return File.Exists(localPath) ? localPath : null;
    }
}
