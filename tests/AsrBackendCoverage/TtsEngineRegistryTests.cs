using System;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The engine registry is the one place an engine is enumerated (issue: the same three-way
/// switch used to live in about a dozen files, each with a silent default arm). These assert
/// that every TtsBackendKind is actually registered and describes itself completely, so a new
/// enum member fails here rather than being quietly treated as Chatterbox at run time.
/// </summary>
public class TtsEngineRegistryTests
{
    [Fact]
    public void EveryBackendKindHasAnEngine()
    {
        foreach (TtsBackendKind kind in Enum.GetValues<TtsBackendKind>())
            Assert.Equal(kind, TtsEngines.For(kind).Kind);
        Assert.Equal(Enum.GetValues<TtsBackendKind>().Length, TtsEngines.All.Count);
    }

    [Fact]
    public void EveryEngineDescribesItself()
    {
        foreach (var e in TtsEngines.All)
        {
            Assert.False(string.IsNullOrWhiteSpace(e.DisplayName), $"{e.Kind} DisplayName");
            Assert.False(string.IsNullOrWhiteSpace(e.Description), $"{e.Kind} Description");
            Assert.False(string.IsNullOrWhiteSpace(e.PhonemeScheme), $"{e.Kind} PhonemeScheme");
            Assert.True(e.SampleRate > 0, $"{e.Kind} SampleRate");
            Assert.NotEmpty(e.RequiredSets);
            // A job's voice must come from somewhere the dialog can offer.
            Assert.True(e.UsesReferenceClip || e.UsesVoiceList || e.UsesLanguage,
                $"{e.Kind} offers no way to choose a voice");
        }
    }

    [Fact]
    public void EnginesAreLookedUpByPersistedNameAndFallBackSafely()
    {
        Assert.Equal(TtsBackendKind.Kokoro, TtsEngines.For("Kokoro").Kind);
        Assert.Equal(TtsBackendKind.Kokoro, TtsEngines.For("kokoro").Kind);      // case-insensitive
        Assert.Equal(TtsEngines.All[0].Kind, TtsEngines.For("nonsense").Kind);   // never throws on old settings
        Assert.Equal(TtsEngines.All[0].Kind, TtsEngines.For((string?)null).Kind);
    }

    [Fact]
    public void RequestCarriesTheFieldsTheEngineUses()
    {
        var job = new TtsJobSettings("Kokoro", "", "af_heart", Speed: 1.4f, NumStep: 24);
        var req = TtsEngines.For(TtsBackendKind.Kokoro).BuildRequest("hi", "/tmp/o.wav", "/tmp/segs", job);
        Assert.Equal(1.4f, req.Speed);
        Assert.Equal("/tmp/segs", req.SegmentsDir);

        var ovJob = new TtsJobSettings("OmniVoice", "cy", "cy_default", NumStep: 24);
        var ovReq = TtsEngines.For(TtsBackendKind.OmniVoice).BuildRequest("hi", "/tmp/o.wav", "/tmp/segs", ovJob);
        Assert.Equal("cy", ovReq.Lang);
        Assert.Equal(24, ovReq.NumStep);

        // An OmniVoice job with no language still renders: the engine defaults it rather than
        // handing the phonemizer an empty code.
        var blank = TtsEngines.For(TtsBackendKind.OmniVoice)
            .BuildRequest("hi", "/tmp/o.wav", "/tmp/segs", new TtsJobSettings("OmniVoice", "", "v"));
        Assert.Equal("en", blank.Lang);
    }

    [Fact]
    public void AnUnknownEngineNameIsRecognisedAsUnknown()
    {
        // For() falls back so a job can still be described and a settings file still loads;
        // TryFor() reports the truth, which is what callers that only describe a job use so a
        // job saved by another build is not relabelled as the fallback engine.
        Assert.Null(TtsEngines.TryFor("PiperTTS"));
        Assert.Null(TtsEngines.TryFor(""));
        Assert.Null(TtsEngines.TryFor(null));
        Assert.Equal(TtsEngines.Default.Kind, TtsEngines.For("PiperTTS").Kind);
        Assert.Equal(TtsEngines.All[0], TtsEngines.Default);
    }

    [Fact]
    public void JobDescriptionAndAnnotationLanguageComeFromTheEngine()
    {
        var british = new JobRecord { Kind = JobKind.Tts, TtsSettings = new("Kokoro", "", "bf_emma", Speed: 1.0f) };
        Assert.Equal("en-GB", TtsEngines.For(british).AnnotationLanguage(british));
        Assert.Contains("bf_emma", TtsEngines.For(british).DescribeJob(british));

        var american = new JobRecord { Kind = JobKind.Tts, TtsSettings = new("Kokoro", "", "af_heart") };
        Assert.Equal("en", TtsEngines.For(american).AnnotationLanguage(american));

        var omni = new JobRecord { Kind = JobKind.Tts, TtsSettings = new("OmniVoice", "cy", "v1", NumStep: 32) };
        Assert.Equal("cy", TtsEngines.For(omni).AnnotationLanguage(omni));

        // Chatterbox's voice is a path; the reader shows the file name, not the whole path.
        var cb = new JobRecord { Kind = JobKind.Tts, TtsSettings = new("Chatterbox", "", "/home/someone/clips/ref.wav") };
        Assert.Contains("ref.wav", TtsEngines.For(cb).DescribeJob(cb));
        Assert.DoesNotContain(TtsEngines.For(cb).DescribeJob(cb), p => p.Contains('/'));
    }
}
