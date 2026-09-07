using System;
using System.IO;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// TTS model sets used to be an enum plus seven switch tables, and a set missing from any one
/// of them still compiled — the default arms reported it as ready with nothing downloadable.
/// Now each set is one entry in <see cref="TtsModelSets"/> with every fact required (#133).
/// These pin what the compiler cannot: that the entries are consistent with each other, with
/// the engines that require them, and with the settings fields they read and write.
/// </summary>
public class TtsModelSetTableTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public TtsModelSetTableTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static SettingsService NewSettings()
    {
        var s = new SettingsService();
        s.Current.ModelsDir = Path.Combine(Path.GetTempPath(), "vernacula-tests", "models-root");
        return s;
    }

    [Fact]
    public void EveryEntryIsCompleteAndDistinct()
    {
        Assert.NotEmpty(TtsModelSets.All);
        foreach (var set in TtsModelSets.All)
        {
            Assert.False(string.IsNullOrWhiteSpace(set.Name), "a set needs a name");
            Assert.False(string.IsNullOrWhiteSpace(set.Description), $"{set.Name} needs a description");
            Assert.False(string.IsNullOrWhiteSpace(set.SubDir), $"{set.Name} needs a default subfolder");
            // A hosted set must say where its manifest is, or explicitly have none; an unhosted
            // set cannot have one. Either way the two flags agree with the URLs.
            Assert.Equal(set.RepoBase.Length > 0, set.CanDownload);
            Assert.Equal(set.ManifestUrl.Length > 0, set.HasManifest);
            if (set.HasManifest) Assert.True(set.CanDownload, $"{set.Name} has a manifest but no repo");
            if (set.CanDownload) Assert.NotEmpty(set.Downloadable);
        }
        Assert.Equal(TtsModelSets.All.Count, TtsModelSets.All.Select(s => s.Name).Distinct().Count());
        Assert.Equal(TtsModelSets.All.Count, TtsModelSets.All.Select(s => s.SubDir).Distinct().Count());
        // The named statics are the table, not copies of it.
        foreach (var named in new[] { TtsModelSets.Kokoro, TtsModelSets.OmniVoice, TtsModelSets.OmniVoiceVoices, TtsModelSets.Chatterbox, TtsModelSets.PhonemizerData })
            Assert.Single(TtsModelSets.All, s => ReferenceEquals(s, named));
    }

    [Fact]
    public void EveryEngineRequiresOnlySetsInTheTable()
    {
        foreach (var engine in TtsEngines.All)
        {
            Assert.NotEmpty(engine.RequiredSets);
            foreach (var set in engine.RequiredSets)
                Assert.Contains(TtsModelSets.All, s => ReferenceEquals(s, set));
        }
    }

    /// <summary>
    /// GetOverride and SetOverride are two lambdas per entry that must name the same
    /// AppSettings field; nothing but this test notices when a copy-paste makes them differ.
    /// </summary>
    [Fact]
    public void OverrideReadsBackWhatItWrote_ForEverySet()
    {
        var settings = NewSettings();
        foreach (var set in TtsModelSets.All)
        {
            string pick = Path.Combine(_dir, set.SubDir + "-picked");
            set.SetOverride(settings.Current, pick);
            Assert.Equal(pick, set.GetOverride(settings.Current));
            Assert.Equal(pick, set.Dir(settings));          // the pick wins over every default
            set.SetOverride(settings.Current, "");
            Assert.Equal("", set.GetOverride(settings.Current));
        }
        // And no two sets share a field: setting one must not change another.
        foreach (var a in TtsModelSets.All)
        {
            a.SetOverride(settings.Current, "/only/" + a.SubDir);
            foreach (var b in TtsModelSets.All.Where(b => !ReferenceEquals(a, b)))
                Assert.NotEqual("/only/" + a.SubDir, b.GetOverride(settings.Current));
            a.SetOverride(settings.Current, "");
        }
    }

    [Fact]
    public void DefaultDirectoryIsTheSubfolderOfTheModelsDir()
    {
        var settings = NewSettings();
        // Kokoro and Chatterbox have no resolver, so with no pick the default is fully determined.
        Assert.Equal(Path.Combine(settings.GetModelsDir(), TtsModelSets.Kokoro.SubDir), TtsModelSets.Kokoro.Dir(settings));
        Assert.Equal(Path.Combine(settings.GetModelsDir(), TtsModelSets.Chatterbox.SubDir), TtsModelSets.Chatterbox.Dir(settings));
        // The named SettingsService entry points are the same answer, not a second copy of the rule.
        Assert.Equal(TtsModelSets.Kokoro.Dir(settings), settings.GetKokoroModelsDir());
        Assert.Equal(TtsModelSets.Chatterbox.Dir(settings), settings.GetChatterboxModelsDir());
        Assert.Equal(TtsModelSets.OmniVoice.Dir(settings), settings.GetOmniVoiceModelsDir());
        Assert.Equal(TtsModelSets.OmniVoiceVoices.Dir(settings), settings.GetOmniVoiceVoiceLibDir());
        Assert.Equal(TtsModelSets.PhonemizerData.Dir(settings), settings.GetPhonemizerDataDir());
    }

    [Fact]
    public void KokoroPresenceIsTheGraphPlusAnyVoicePack()
    {
        var settings = NewSettings();
        TtsModelSets.Kokoro.SetOverride(settings.Current, _dir);

        Assert.Equal(["kokoro.onnx", "voices/*.bin"], TtsModelSets.Kokoro.MissingFiles(settings));

        File.WriteAllText(Path.Combine(_dir, "kokoro.onnx"), "");
        Assert.Equal(["voices/*.bin"], TtsModelSets.Kokoro.MissingFiles(settings));

        Directory.CreateDirectory(Path.Combine(_dir, "voices"));
        File.WriteAllText(Path.Combine(_dir, "voices", "af_heart.bin"), "");
        Assert.Empty(TtsModelSets.Kokoro.MissingFiles(settings));   // one voice is enough to run
        // ...but a download would still fetch every voice the repo holds.
        Assert.Equal(TtsModelSets.KokoroVoices.Length + 1, TtsModelSets.Kokoro.Downloadable.Count());
    }

    [Fact]
    public void UnhostedDataTreeIsJudgedByItsSentinel()
    {
        var settings = NewSettings();
        TtsModelSets.PhonemizerData.SetOverride(settings.Current, _dir);
        Assert.False(TtsModelSets.PhonemizerData.CanDownload);
        var missing = Assert.Single(TtsModelSets.PhonemizerData.MissingFiles(settings));
        Assert.Contains("core/phonology.jsonc", missing);
    }

    [Fact]
    public void TokenizerRidesWithTheEnginesThatNeedIt()
    {
        Assert.Contains(TtsModelSets.Chatterbox.Downloadable, a => a.LocalRelativePath == "tokenizer.json");
        Assert.Contains(TtsModelSets.OmniVoice.Downloadable,  a => a.LocalRelativePath == "tokenizer.json");
        Assert.DoesNotContain(TtsModelSets.Kokoro.Downloadable, a => a.LocalRelativePath == "tokenizer.json");
        // The voice library is not judged by the tokenizer either way.
        Assert.Equal(["voices.jsonc", "voice-codes.json"], TtsModelSets.OmniVoiceVoices.Assets.Select(a => a.LocalRelativePath));
    }
}
