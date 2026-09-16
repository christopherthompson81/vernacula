#if AUDIOCPP_BACKEND
using System;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services.Tts;
using Vernacula.AudioCpp;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The audio.cpp Kokoro engine's lexical facts — which voices exist and what language each one
/// obliges — and the estimated word map it has to produce because the ABI reports none.
///
/// <para>
/// None of this loads a model. The voice table was measured once against the installed package
/// (docs/investigations/audiocpp_tts_backend_investigation.md Run 3) and SHIPPED as data,
/// precisely because the binding cannot enumerate voices at run time; these guard the shape of
/// that data and the rules read off it.
/// </para>
/// </summary>
public class AudioCppTtsEngineTests
{
    private static TtsEngine Engine => TtsEngines.For(TtsBackendKind.AudioCppKokoro);

    [Fact]
    public void TheEngineIsRegisteredAndOffersAVoiceListAndSpeed()
    {
        Assert.Equal(TtsBackendKind.AudioCppKokoro, Engine.Kind);
        Assert.True(Engine.UsesVoiceList);
        Assert.True(Engine.UsesSpeed);
        // ⚠ NOT a language control. The engine rejects a voice/language mismatch outright, and
        // the language is implied by the voice, so exposing both would be exposing a way to fail.
        Assert.False(Engine.UsesLanguage);
        Assert.False(Engine.UsesReferenceClip);
        Assert.Equal(24_000, Engine.SampleRate);
    }

    [Fact]
    public void EveryShippedVoiceIsDistinctAndHasABakedInLanguage()
    {
        var all = AudioCppKokoroVoices.All;
        Assert.Equal(41, all.Length);
        Assert.Equal(all.Length, all.Distinct().Count());
        foreach (var voice in all)
        {
            // A voice whose prefix fell through to the default would be offered in the picker and
            // then rendered in the wrong language, or rejected by the engine — both silent here.
            Assert.NotEqual("", AudioCppKokoroVoices.EngineLanguage(voice));
            Assert.True(AudioCppKokoroVoices.IsKnown(voice));
        }
    }

    [Theory]
    [InlineData("af_heart",  "en-us", "en")]
    [InlineData("am_adam",   "en-us", "en")]
    [InlineData("bf_alice",  "en-gb", "en-GB")]
    [InlineData("bm_george", "en-gb", "en-GB")]
    [InlineData("ef_dora",   "es",    "es")]
    [InlineData("ff_siwis",  "fr-fr", "fr")]
    [InlineData("hf_alpha",  "hi",    "hi")]
    [InlineData("if_sara",   "it",    "it")]
    [InlineData("pm_alex",   "pt-br", "pt-BR")]
    public void TheVoicePrefixDecidesBothLanguages(string voice, string engineLang, string phonemizerLang)
    {
        // Two vocabularies, not one: the engine wants "en-us" and "pt-br"; the phonemizer has no
        // "en-us" at all and spells Brazilian Portuguese "pt-BR".
        Assert.Equal(engineLang, AudioCppKokoroVoices.EngineLanguage(voice));
        Assert.Equal(phonemizerLang, AudioCppKokoroVoices.PhonemizerLanguage(voice));
    }

    /// <summary>
    /// The thirteen voices the family advertises and the installed package will not render: the
    /// Japanese ones want UniDic resources this GGUF has no copy of, and the Chinese ones hit a
    /// vocab missing a phoneme symbol. They are left out so the failure cannot land after a
    /// document has already been queued.
    /// </summary>
    [Theory]
    [InlineData("jf_alpha")]
    [InlineData("jm_kumo")]
    [InlineData("zf_xiaoxiao")]
    [InlineData("zm_yunyang")]
    public void TheVoicesThisPackageCannotRenderAreNotOffered(string voice)
    {
        Assert.DoesNotContain(voice, AudioCppKokoroVoices.All);
        Assert.False(AudioCppKokoroVoices.IsKnown(voice));
    }

    [Fact]
    public void AVoiceTheTableDoesNotKnowIsRejectedBeforeTheDocumentIsQueued()
    {
        Assert.NotNull(Engine.DescribeJobIssue(null!, Job("jf_alpha")));
        Assert.NotNull(Engine.DescribeJobIssue(null!, Job("")));
        // A voice it does know gets past the job check; whether the PACKAGE is on disk is a
        // different question, asked of RequiredSets rather than here.
        Assert.Null(Engine.DescribeJobIssue(null!, Job("af_heart")));
    }

    [Fact]
    public void TheRequestCarriesTheVoiceAndSpeedAndNoLanguage()
    {
        var req = Engine.BuildRequest("hello", "/tmp/o.wav", "/tmp/segs", Job("bm_george", speed: 1.3f));
        Assert.Equal("bm_george", req.Voice);
        Assert.Equal(1.3f, req.Speed);
        Assert.Equal("/tmp/segs", req.SegmentsDir);
        Assert.Null(req.Lang);
    }

    [Fact]
    public void AFinishedJobIsAnnotatedInTheLanguageItWasSpokenIn()
    {
        Assert.Equal("en-GB", Engine.AnnotationLanguage(Rendered("bf_emma")));
        Assert.Equal("fr",    Engine.AnnotationLanguage(Rendered("ff_siwis")));
        // A job from a build with a wider voice table still describes itself rather than throwing.
        Assert.Equal("en",    Engine.AnnotationLanguage(Rendered("")));
    }

    [Fact]
    public void ItsVoiceIsStoredApartFromTheOnnxKokorosVoice()
    {
        // The two engines' lists do not overlap — one reads voices/*.bin off disk, the other
        // ships the presets baked into a GGUF — so one shared field would keep handing each
        // engine a voice the other picked.
        var s = new AppSettings();
        Engine.WriteStoredVoice(s, "ff_siwis");
        TtsEngines.For(TtsBackendKind.Kokoro).WriteStoredVoice(s, "af_bella");
        Assert.Equal("ff_siwis", Engine.ReadStoredVoice(s));
        Assert.Equal("af_bella", TtsEngines.For(TtsBackendKind.Kokoro).ReadStoredVoice(s));
    }

    // ── The estimated word map ───────────────────────────────────────────────

    [Fact]
    public void EstimatedWordsCoverTheWholeSegmentInOrder()
    {
        const string text = "The harbour was quiet this morning.";
        var words = AudioCppSynthesisService.EstimateWords(text, 4.0);

        Assert.Equal(text.Split(' ').Length, words.Count);
        Assert.Equal("The", words[0].Text);
        Assert.Equal(0.0, words[0].StartSeconds, 6);
        Assert.Equal(4.0, words[^1].EndSeconds, 6);
        for (int i = 1; i < words.Count; i++)
        {
            Assert.True(words[i].StartSeconds >= words[i - 1].StartSeconds, "words run backwards");
            // Contiguous: the reader's highlight walks these, so a gap would show as a word the
            // highlight skips over rather than as anything that looks like a timing bug.
            Assert.Equal(words[i - 1].EndSeconds, words[i].StartSeconds, 6);
        }
    }

    [Fact]
    public void ALongerWordGetsALongerShare()
    {
        // Weighted by length, which is all that can be known: audio.cpp phonemizes inside the
        // session with eSpeak-ng and hands back no trace, no phonemes and no timings.
        var words = AudioCppSynthesisService.EstimateWords("a extraordinarily", 2.0);
        Assert.True(words[1].EndSeconds - words[1].StartSeconds
                  > words[0].EndSeconds - words[0].StartSeconds);
    }

    [Fact]
    public void AnEmptySegmentProducesNoWordsRatherThanThrowing()
    {
        Assert.Empty(AudioCppSynthesisService.EstimateWords("   ", 1.0));
    }

    private static TtsJobSettings Job(string voice, float speed = 1.0f) =>
        new(TtsBackendKind.AudioCppKokoro.ToString(), "", voice, speed);

    private static JobRecord Rendered(string voice) =>
        new() { Kind = JobKind.Tts, TtsSettings = Job(voice) };
}
#endif
