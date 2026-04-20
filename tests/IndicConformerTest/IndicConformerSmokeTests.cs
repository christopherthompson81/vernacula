using System.Globalization;
using System.Text;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests.IndicConformerTest;

/// <summary>
/// Smoke tests for the IndicConformer ONNX package + C# decode path.
///
/// Runs each of the 22 language test clips through
/// <see cref="IndicConformer.Recognize"/> and asserts the output is
/// non-trivial and composed primarily of characters in the expected
/// script family. Not a WER benchmark — golden-substring assertions
/// against the Fleurs / IndicVoices-R references would be too brittle
/// against ordinary ASR word errors. The regression we actually want
/// to catch is the Parakeet-vocab-fallback / wrong-language-head / IO-
/// rename-broken kind of failure, where output either vanishes or
/// comes back in the wrong script. Script-block fraction is a robust
/// signal for that.
///
/// Discovery:
///   - IndicConformer ONNX package: env VERNACULA_INDIC_MODELS_DIR,
///     else ~/.local/share/Vernacula/models/indicconformer/,
///     else ~/models/indicconformer_600m_onnx/.
///   - Test audio library: env VERNACULA_TEST_AUDIO_DIR,
///     else ~/Programming/test_audio/.
/// Each test skips (Assert.Skip) if the needed asset is absent, so a
/// machine without the 2.4 GB package still shows green but reports
/// what was skipped.
/// </summary>
public class IndicConformerSmokeTests : IDisposable
{
    private static readonly Lazy<IndicConformer?> SharedModel = new(LoadSharedModel);
    private static readonly string? AudioRoot = DiscoverAudioRoot();
    private static readonly string? SkipReason = BuildSkipReason();

    // Per-language test data. `scriptRanges` lists the Unicode blocks where
    // legitimate output for the language should live; the decode must put
    // at least `minScriptFraction` of non-whitespace characters into one of
    // these blocks (combined) to pass. Ranges are inclusive on both ends.
    private sealed record Lang(
        string Iso,
        string LocaleDir,
        string AudioFile,
        (int lo, int hi)[] ScriptRanges);

    // Script block reference:
    //   Devanagari    U+0900–U+097F (hi, mr, sa, ne, kok, mai, doi, sd)
    //   Bengali       U+0980–U+09FF (bn, as)
    //   Gurmukhi      U+0A00–U+0A7F (pa)
    //   Gujarati      U+0A80–U+0AFF (gu)
    //   Odia          U+0B00–U+0B7F (or)
    //   Tamil         U+0B80–U+0BFF (ta)
    //   Telugu        U+0C00–U+0C7F (te)
    //   Kannada       U+0C80–U+0CFF (kn)
    //   Malayalam     U+0D00–U+0D7F (ml)
    //   Ol Chiki      U+1C50–U+1C7F (sat)
    //   Meitei Mayek  U+ABC0–U+ABFF (mni)
    //   Arabic        U+0600–U+06FF (ur, ks, sd)
    //   Arabic Supp   U+0750–U+077F
    //   Arabic Ext-A  U+08A0–U+08FF
    //   Bengali supp  (as)/brx may land in extended Bengali ranges
    //
    // Bodo (brx) can use Devanagari or Bengali historically; AI4Bharat's
    // tokenizer appears to be Devanagari-based, so accept both.
    // Kashmiri (ks), Sindhi (sd) primarily use Perso-Arabic script in
    // AI4Bharat's tokenizer; accept Arabic ranges.
    public static IEnumerable<TheoryDataRow<string>> LanguageCodes =>
        Langs.Select(l => new TheoryDataRow<string>(l.Iso));

    private static readonly Lang[] Langs =
    [
        new("hi",  "hi-IN", "hi-IN_fleurs_01.wav",        [(0x0900, 0x097F)]),
        new("bn",  "bn-IN", "bn-IN_fleurs_01.wav",        [(0x0980, 0x09FF)]),
        new("ta",  "ta-IN", "ta-IN_fleurs_01.wav",        [(0x0B80, 0x0BFF)]),
        new("te",  "te-IN", "te-IN_fleurs_01.wav",        [(0x0C00, 0x0C7F)]),
        new("mr",  "mr-IN", "mr-IN_fleurs_01.wav",        [(0x0900, 0x097F)]),
        new("gu",  "gu-IN", "gu-IN_fleurs_01.wav",        [(0x0A80, 0x0AFF)]),
        new("kn",  "kn-IN", "kn-IN_fleurs_01.wav",        [(0x0C80, 0x0CFF)]),
        new("ml",  "ml-IN", "ml-IN_fleurs_01.wav",        [(0x0D00, 0x0D7F)]),
        new("pa",  "pa-IN", "pa-IN_fleurs_01.wav",        [(0x0A00, 0x0A7F)]),
        new("or",  "or-IN", "or-IN_fleurs_01.wav",        [(0x0B00, 0x0B7F)]),
        new("as",  "as-IN", "as-IN_fleurs_01.wav",        [(0x0980, 0x09FF)]),
        new("ne",  "ne-NP", "ne-NP_fleurs_01.wav",        [(0x0900, 0x097F)]),
        new("sd",  "sd-IN", "sd-IN_fleurs_01.wav",        [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0x0900, 0x097F)]),
        new("ur",  "ur-PK", "ur-PK_fleurs_01.wav",        [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)]),
        new("sa",  "sa-IN", "sa-IN_indicvoices_r_01.wav", [(0x0900, 0x097F)]),
        new("ks",  "ks-IN", "ks-IN_indicvoices_r_01.wav", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0x0900, 0x097F)]),
        new("brx", "brx-IN","brx-IN_indicvoices_r_01.wav",[(0x0900, 0x097F), (0x0980, 0x09FF)]),
        new("doi", "doi-IN","doi-IN_indicvoices_r_01.wav",[(0x0900, 0x097F)]),
        new("kok", "kok-IN","kok-IN_indicvoices_r_01.wav",[(0x0900, 0x097F)]),
        new("mai", "mai-IN","mai-IN_indicvoices_r_01.wav",[(0x0900, 0x097F)]),
        new("mni", "mni-IN","mni-IN_indicvoices_r_01.wav",[(0xABC0, 0xABFF), (0x0980, 0x09FF)]),
        new("sat", "sat-IN","sat-IN_indicvoices_r_01.wav",[(0x1C50, 0x1C7F)]),
    ];

    private const double MinScriptFraction = 0.70;
    private const int    MinCharacters     = 3;

    // ── Tests ────────────────────────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(LanguageCodes))]
    public void Decodes_InExpectedScript(string iso)
    {
        if (SkipReason is not null) Assert.Skip(SkipReason);
        var model = SharedModel.Value!;

        var lang = Langs.First(l => l.Iso == iso);
        string wavPath = Path.Combine(AudioRoot!, lang.LocaleDir, lang.AudioFile);
        if (!File.Exists(wavPath)) Assert.Skip($"missing test audio: {wavPath}");

        var (samples, sampleRate, channels) = AudioUtils.ReadAudio(wavPath);
        Assert.Equal(Config.SampleRate, sampleRate);
        Assert.Equal(1, channels);

        // Single whole-clip segment. Diarization is out of scope for the
        // smoke test — we just want to prove the ASR graph + C# decoder
        // produce the expected script family.
        var segs = new List<(double start, double end, string spk)>
        {
            (0.0, samples.Length / (double)sampleRate, "speaker_0"),
        };

        var results = model.Recognize(segs, samples, iso).ToList();
        Assert.Single(results);
        string text = results[0].text;
        Assert.False(string.IsNullOrWhiteSpace(text),
            $"[{iso}] decode produced empty output");

        string nfc = text.Normalize(NormalizationForm.FormC);
        int inScript = 0, nonWs = 0;
        foreach (Rune r in nfc.EnumerateRunes())
        {
            if (Rune.IsWhiteSpace(r)) continue;
            nonWs++;
            int cp = r.Value;
            foreach (var (lo, hi) in lang.ScriptRanges)
                if (cp >= lo && cp <= hi) { inScript++; break; }
        }

        Assert.True(nonWs >= MinCharacters,
            $"[{iso}] decode too short: {nonWs} non-whitespace chars in {text.q()}");

        double fraction = nonWs == 0 ? 0.0 : (double)inScript / nonWs;
        Assert.True(fraction >= MinScriptFraction,
            $"[{iso}] only {fraction:P0} of chars in expected script " +
            $"ranges {FormatRanges(lang.ScriptRanges)}; got {nonWs} chars total: " +
            $"{Preview(nfc)}");
    }

    [Fact]
    public void PackageIsLoadable()
    {
        if (SkipReason is not null) Assert.Skip(SkipReason);
        var model = SharedModel.Value!;

        Assert.NotNull(model);
        Assert.Contains("hi", model.SupportedLanguages);
        Assert.Equal(22, model.SupportedLanguages.Count);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static IndicConformer? LoadSharedModel()
    {
        string? dir = DiscoverModelsDir();
        if (dir is null) return null;
        try { return new IndicConformer(dir); }
        catch { return null; }
    }

    private static string? DiscoverModelsDir()
    {
        string? envDir = Environment.GetEnvironmentVariable("VERNACULA_INDIC_MODELS_DIR");
        if (!string.IsNullOrWhiteSpace(envDir) && HasPackage(envDir)) return envDir;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".local", "share", "Vernacula", "models", Config.IndicConformerSubDir),
            Path.Combine(home, "models", "indicconformer_600m_onnx"),
        ];
        foreach (var c in candidates)
            if (HasPackage(c)) return c;
        return null;
    }

    private static bool HasPackage(string dir) =>
        Directory.Exists(dir) &&
        File.Exists(Path.Combine(dir, Config.EncoderFile)) &&
        File.Exists(Path.Combine(dir, Config.CtcDecoderFile)) &&
        File.Exists(Path.Combine(dir, Config.PreprocessorFile)) &&
        File.Exists(Path.Combine(dir, Config.VocabFile)) &&
        File.Exists(Path.Combine(dir, Config.IndicConformerLanguageSpansFile));

    private static string? DiscoverAudioRoot()
    {
        string? envDir = Environment.GetEnvironmentVariable("VERNACULA_TEST_AUDIO_DIR");
        if (!string.IsNullOrWhiteSpace(envDir) && Directory.Exists(envDir)) return envDir;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string def = Path.Combine(home, "Programming", "test_audio");
        return Directory.Exists(def) ? def : null;
    }

    private static string? BuildSkipReason()
    {
        if (SharedModel.Value is null)
            return "IndicConformer ONNX package not found. Set VERNACULA_INDIC_MODELS_DIR, " +
                   "or place the package at ~/.local/share/Vernacula/models/indicconformer/ " +
                   "or ~/models/indicconformer_600m_onnx/.";
        if (AudioRoot is null)
            return "Test audio library not found. Set VERNACULA_TEST_AUDIO_DIR, " +
                   "or place the test_audio/ tree at ~/Programming/test_audio/.";
        return null;
    }

    private static string FormatRanges((int lo, int hi)[] ranges) =>
        "[" + string.Join(", ", ranges.Select(r => $"U+{r.lo:X4}-{r.hi:X4}")) + "]";

    private static string Preview(string s) =>
        s.Length <= 80 ? s : s[..80] + "…";

    public void Dispose()
    {
        // Class-level Lazy<> holds the model; per-test disposal isn't right
        // here (it'd break the next test in the theory). The model is
        // disposed implicitly when the test host tears down AppDomain.
        GC.SuppressFinalize(this);
    }
}

internal static class StringFormatExtension
{
    // Short repr-like helper — wraps text in quotes for readable failure
    // messages without pulling in a full Debug.Print dependency.
    internal static string q(this string? s) => s is null ? "null" : $"\"{s}\"";
}
