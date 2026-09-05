using Vernacula.Tts.Base;
using Xunit;

namespace Vernacula.Tts.Tests;

/// <summary>
/// The reader's language search — the port of web-demo/tools/check-language-search.mjs: every
/// language is findable, and ranked first, by its English name, its code and its endonym; an
/// endonym prefix finds it; no two share an endonym.
/// </summary>
public class LanguageCatalogTests
{
    [Fact]
    public void CatalogueIsNonTrivialAndHasEnglish()
    {
        Assert.True(LanguageCatalog.All.Count > 150, $"only {LanguageCatalog.All.Count} languages");
        Assert.Equal("English", LanguageCatalog.ByCode("en")!.Name);
        Assert.True(LanguageCatalog.ByCode("en")!.Trained);
        Assert.Equal("en", LanguageCatalog.VoiceLangOf("en"));
    }

    [Fact]
    public void EveryLanguageIsFoundFirstByNameCodeAndEndonym()
    {
        var failures = new List<string>();
        foreach (var l in LanguageCatalog.All)
        {
            var probes = new List<(string what, string q)> { ("name", l.Name), ("code", l.Code) };
            if (l.Native is not null) probes.Add(("native", l.Native));
            foreach (var (what, q) in probes)
            {
                var hits = LanguageCatalog.Search(q);
                if (hits.Count == 0 || hits.All(h => h.Code != l.Code)) failures.Add($"{l.Code}: not found by {what} \"{q}\"");
                else if (hits[0].Code != l.Code) failures.Add($"{l.Code}: found by {what} \"{q}\" but ranked behind {hits[0].Code}");
            }
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void EndonymPrefixFindsTheLanguage()
    {
        var failures = new List<string>();
        foreach (var l in LanguageCatalog.All)
        {
            if (l.Native is null) continue;
            var runes = l.Native.EnumerateRunes().ToList();
            if (runes.Count < 2) continue;
            var q = string.Concat(runes.Take(3).Select(r => r.ToString()));
            if (LanguageCatalog.Search(q).All(h => h.Code != l.Code))
                failures.Add($"{l.Code}: endonym prefix \"{q}\" does not find {l.Native}");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void NoTwoLanguagesShareAnEndonym()
    {
        var seen = new Dictionary<string, string>();
        foreach (var l in LanguageCatalog.All)
        {
            if (l.Native is null) continue;
            var k = l.Native.ToLowerInvariant();
            Assert.False(seen.ContainsKey(k), $"{l.Code} and {seen.GetValueOrDefault(k)} share the endonym \"{l.Native}\"");
            seen[k] = l.Code;
        }
    }

    [Fact]
    public void DonorVoicesExistInTheCatalogueOrAreKnownDonors()
    {
        // A donor code the picker does not list is allowed (the web demo names four), but it must
        // not be a typo of a listed one: every donor is either a listed language or one of those.
        var known = new HashSet<string> { "zoc", "dag", "ipk", "bgp" };
        foreach (var l in LanguageCatalog.All.Where(l => l.Voice is not null))
            Assert.True(LanguageCatalog.ByCode(l.Voice) is not null || known.Contains(l.Voice!),
                $"{l.Code}'s donor {l.Voice} is neither listed nor a known donor");
    }
}
