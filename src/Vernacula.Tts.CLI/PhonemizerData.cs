namespace Vernacula.Tts.CLI;

/// <summary>
/// Locates the vernacula-phonemizer `data/` tree and hands it to the phonemizer.
///
/// The phonemizer's own DataPath finds the tree by walking up from AppContext.BaseDirectory for an
/// ancestor holding `data/core/phonology.jsonc` — which works when you run from inside the
/// phonemizer repo, and does not work here: our build output is
/// src/Vernacula.Tts.CLI/bin/&lt;cfg&gt;/net10.0, and this repo has no top-level `data/`. The tree
/// lives one level in, under the submodule. So we resolve it ourselves and set
/// VERNACULA_DATA_DIR, which DataPath honours ahead of the walk.
/// </summary>
internal static class PhonemizerData
{
    private const string Sentinel = "core/phonology.jsonc";
    private const string SubmoduleData = "external/vernacula-phonemizer/data";

    /// <summary>Resolve the data root and publish it via VERNACULA_DATA_DIR. An explicit
    /// <paramref name="dataDir"/> wins; then an already-set VERNACULA_DATA_DIR; then the
    /// submodule, found by walking up from the executable. Returns the root, or null when nothing
    /// resolved — the caller reports that only if it actually needs to phonemize.</summary>
    public static string? Resolve(string? dataDir)
    {
        string? root = dataDir ?? Environment.GetEnvironmentVariable("VERNACULA_DATA_DIR");
        if (string.IsNullOrEmpty(root))
        {
            for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            {
                var candidate = Path.Combine(d.FullName, SubmoduleData.Replace('/', Path.DirectorySeparatorChar));
                if (IsDataRoot(candidate)) { root = candidate; break; }
            }
        }
        if (string.IsNullOrEmpty(root) || !IsDataRoot(root)) return null;

        root = Path.GetFullPath(root);
        Environment.SetEnvironmentVariable("VERNACULA_DATA_DIR", root);
        return root;
    }

    /// <summary>The same sentinel check DataPath uses — the file, not just a `data` directory, so
    /// an unrelated `data/` on the walk cannot match.</summary>
    private static bool IsDataRoot(string dir) =>
        File.Exists(Path.Combine(dir, Sentinel.Replace('/', Path.DirectorySeparatorChar)));

    public static string NotFoundMessage() =>
        "Could not locate the vernacula-phonemizer data tree (looked for "
        + $"{SubmoduleData}/{Sentinel} above \"{AppContext.BaseDirectory}\").\n"
        + "  If the submodule isn't checked out:  git submodule update --init external/vernacula-phonemizer\n"
        + "  Otherwise point at it explicitly:    --data-dir <path to vernacula-phonemizer/data>\n"
        + "  Or pass --ipa to skip phonemization and supply IPA directly.";
}
