#if AUDIOCPP_BACKEND
using System;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services.Tts;
using Vernacula.AudioCpp;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Alignment;
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
        Assert.Equal(54, all.Length);
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
    /// The thirteen voices this table used to exclude, because the installed package refused
    /// them.
    ///
    /// <para>
    /// ⚠ BOTH REFUSALS WERE ABOUT THE ENGINE'S OWN G2P, WHICH THIS APP NO LONGER USES. The
    /// Japanese ones wanted UniDic resources the GGUF has no copy of; the Chinese ones hit a
    /// vocabulary missing a symbol the engine's own phonemizer produced. Every voice EMBEDDING
    /// was in the package the whole time — 54 of 54 sidecars — so supplying the phonemes removes
    /// the only thing that was failing. Measured: jf_alpha renders 2.38 s of audio on the
    /// supplied path where the text path still refuses outright.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData("jf_alpha", "ja", "ja")]
    [InlineData("jm_kumo", "ja", "ja")]
    [InlineData("zf_xiaoxiao", "zh", "cmn")]
    [InlineData("zm_yunyang", "zh", "cmn")]
    public void TheVoicesTheEnginesOwnG2pRefusedAreOfferedNow(string voice, string engineLang, string phonemizerLang)
    {
        Assert.Contains(voice, AudioCppKokoroVoices.All);
        Assert.True(AudioCppKokoroVoices.IsKnown(voice));
        Assert.Equal(engineLang, AudioCppKokoroVoices.EngineLanguage(voice));
        // ⚠ `cmn`, not `zh`: the engine's language vocabulary and the phonemizer's are different
        // sets that merely overlap, which is the reason the two tables exist separately.
        Assert.Equal(phonemizerLang, AudioCppKokoroVoices.PhonemizerLanguage(voice));
        // And a render target has to exist, or supplying is not possible and the voice would be
        // offered only to fail the way it used to.
        Assert.True(KokoroFormat.CanRender(phonemizerLang));
    }

    [Fact]
    public void AVoiceTheTableDoesNotKnowIsRejectedBeforeTheDocumentIsQueued()
    {
        Assert.NotNull(Engine.DescribeJobIssue(null!, Job("qq_nobody")));
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

    // ── The supplied phoneme stream ──────────────────────────────────────────

    /// <summary>
    /// The phonemizer's data tree, or null — the same skip every other phonemizer-backed test
    /// uses. A checkout without the submodule initialised must still run the suite.
    /// </summary>
    private static (KokoroPhonemizer G2p, KokoroChunker Chunker) Phonemizer()
    {
        if (PhonemizerData.Resolve(null) is null)
            Assert.Skip("vernacula-phonemizer data/ not found (submodule not checked out?).");
        var g2p = new KokoroPhonemizer();
        return (g2p, new KokoroChunker(g2p));
    }

    [Fact]
    public void EveryEntrySentToTheEngineIsNonEmptyAndInKokorosVocabulary()
    {
        var (g2p, chunker) = Phonemizer();

        // The two rules audio.cpp validates a supplied stream against, and it refuses rather
        // than degrades: an empty entry is a caller error, and so is a symbol with no token id.
        // Both are checked here because the engine's refusal lands after a document is queued.
        const string text = "The button was forgotten, and it cost $3.14 on 24 March.";
        var supplied = AudioCppSynthesisService.Supply(g2p, chunker, text, "en", _ => { });

        Assert.NotNull(supplied);
        Assert.NotEmpty(supplied!.Chunks);
        foreach (var entry in supplied.Chunks)
        {
            Assert.False(string.IsNullOrWhiteSpace(entry));
            Assert.InRange(entry.Length, 1, 510);
            foreach (var ch in entry)
                Assert.True(KokoroVocab.Contains(ch), $"U+{(int)ch:X4} has no Kokoro token id");
        }
    }

    [Fact]
    public void ALongParagraphIsCutIntoSeveralEntriesRatherThanOneOverLongOne()
    {
        var (g2p, chunker) = Phonemizer();

        // The engine chunks TEXT on its own; it cannot chunk a caller's phonemes, because only
        // the G2P that produced a stream knows where it may be cut. So a paragraph past the
        // window has to arrive as a list, and that is this repo's chunker doing the cutting.
        var text = string.Join(' ', Enumerable.Repeat(
            "The harbour was quiet this morning, and the boats had not yet returned.", 20));
        var supplied = AudioCppSynthesisService.Supply(g2p, chunker, text, "en", _ => { });

        Assert.NotNull(supplied);
        Assert.True(supplied!.Chunks.Count > 1, "a 20-sentence paragraph came back as one entry");
        Assert.All(supplied.Chunks, entry => Assert.InRange(entry.Length, 1, 510));
    }

    [Fact]
    public void WordsAreWeightedByTheirPhonemesNotTheirLetters()
    {
        var (g2p, chunker) = Phonemizer();

        // "through" is seven letters and three phonemes; "spa" is three letters and three
        // phonemes. Letter-weighted, the first outlasts the second by more than twice; from the
        // phonemes they are close. This is the whole gain of supplying the stream for alignment.
        const string text = "through spa";
        var supplied = AudioCppSynthesisService.Supply(g2p, chunker, text, "en", _ => { });
        Assert.NotNull(supplied?.PhonemesPerWord);

        var words = AudioCppSynthesisService.SpreadByWeight(text, supplied!.PhonemesPerWord!, 2.0);
        var letters = AudioCppSynthesisService.EstimateWords(text, 2.0);

        double Span(IReadOnlyList<AlignedWord> w, int i) => w[i].EndSeconds - w[i].StartSeconds;
        Assert.True(Span(words, 0) / Span(words, 1) < Span(letters, 0) / Span(letters, 1),
                    "phoneme weighting did not narrow the gap that spelling opened");
    }

    [Fact]
    public void AWeightedSpreadCoversTheSegmentInOrderAndInFull()
    {
        const string text = "one two three four";
        var words = AudioCppSynthesisService.SpreadByWeight(text, [1.0, 2.0, 3.0, 4.0], 10.0);

        Assert.Equal(4, words.Count);
        Assert.Equal(0.0, words[0].StartSeconds, 6);
        Assert.Equal(10.0, words[^1].EndSeconds, 6);
        Assert.Equal(1.0, words[0].EndSeconds, 6);          // 1/10 of the ten seconds
        for (var i = 1; i < words.Count; i++)
            Assert.Equal(words[i - 1].EndSeconds, words[i].StartSeconds, 6);
    }

    [Fact]
    public void AWordThatBecameNoPhonemesGetsAZeroLengthMarkerRatherThanVanishing()
    {
        // The reader indexes words by the source-text whitespace split, so an unpronounceable
        // one still has to appear — the ONNX path does the same with measured durations.
        var words = AudioCppSynthesisService.SpreadByWeight("hello 🙂 world", [5.0, 0.0, 5.0], 2.0);

        Assert.Equal(3, words.Count);
        Assert.Equal("🙂", words[1].Text);
        Assert.Equal(words[1].StartSeconds, words[1].EndSeconds, 6);
        Assert.Equal(1.0, words[1].StartSeconds, 6);
        Assert.Equal(2.0, words[^1].EndSeconds, 6);
    }

    [Fact]
    public void AWeightMapOfTheWrongLengthIsRefusedRatherThanMisaligned()
    {
        // A short map would silently shift every word after the gap, which shows up as a
        // highlight on the wrong word and nothing else. Better to produce none.
        Assert.Empty(AudioCppSynthesisService.SpreadByWeight("one two three", [1.0, 1.0], 3.0));
    }

    [Fact]
    public void AllZeroWeightsFallBackToTheLengthEstimateRatherThanDividingByZero()
    {
        var words = AudioCppSynthesisService.SpreadByWeight("one two", [0.0, 0.0], 2.0);
        Assert.Equal(2, words.Count);
        Assert.Equal(2.0, words[^1].EndSeconds, 6);
    }

    // ── The measured word map ────────────────────────────────────────────────

    private static AudioCppSpeech Spoken(params (string Phonemes, double Start, double End)[] groups)
        => new(new float[24_000], [.. groups.Select(g => new AudioCppPhonemeGroup(g.Phonemes, g.Start, g.End))]);

    [Fact]
    public void MeasuredGroupTimingsBeatEveryProportionalGuess()
    {
        // The whole point of the engine reporting timings: "a" is short and "extraordinarily" is
        // long, and here the ENGINE says so rather than the speller or the phoneme counter. The
        // weights deliberately disagree with the measurement, so a pass proves the measurement won.
        var supplied = new AudioCppSynthesisService.SuppliedPhonemes(
            ["ə ɛkstɹˌɔːɹdənˈɛɹəli"], [0, 1], [9.0, 1.0]);
        var spoken = Spoken(("ə", 0.0, 0.2), ("ɛkstɹˌɔːɹdənˈɛɹəli", 0.2, 2.0));

        var words = AudioCppSynthesisService.Align("a extraordinarily", supplied, spoken, 2.0, out _);

        Assert.Equal(2, words.Count);
        Assert.Equal(0.0, words[0].StartSeconds, 6);
        Assert.Equal(0.2, words[0].EndSeconds, 6);
        Assert.Equal(2.0, words[1].EndSeconds, 6);
    }

    [Fact]
    public void SeveralGroupsOnOneWordCollapseOntoIt()
    {
        // "$3.14" is one written word and several spoken ones; the word takes the union of their
        // spans, exactly as the ONNX path does with measured durations.
        var supplied = new AudioCppSynthesisService.SuppliedPhonemes(["x"], [0, 0, 0, 1], [3.0, 1.0]);
        var spoken = Spoken(("θɹˈi", 0.0, 0.5), ("dˈɑləɹz", 0.5, 1.0), ("fˌɔɹtˈin", 1.0, 1.5), ("nˈW", 1.5, 2.0));

        var words = AudioCppSynthesisService.Align("$3.14 now", supplied, spoken, 2.0, out _);

        Assert.Equal(2, words.Count);
        Assert.Equal(0.0, words[0].StartSeconds, 6);
        Assert.Equal(1.5, words[0].EndSeconds, 6);     // the union of its three groups
        Assert.Equal(1.5, words[1].StartSeconds, 6);
    }

    [Fact]
    public void AGroupCountTheMapDisagreesWithFallsBackRatherThanShifting()
    {
        // The engine cuts groups at ITS view of the stream and the map was built from ours. A
        // disagreement must not be absorbed: every word after it would sit on the wrong audio, and
        // a highlight one word out looks like nothing at all.
        var supplied = new AudioCppSynthesisService.SuppliedPhonemes(["x"], [0, 1], [1.0, 1.0]);
        var spoken = Spoken(("ə", 0.0, 0.2), ("b", 0.2, 1.0), ("c", 1.0, 2.0));   // three, not two

        var words = AudioCppSynthesisService.Align("one two", supplied, spoken, 2.0, out _);

        Assert.Equal(2, words.Count);
        Assert.Equal(1.0, words[0].EndSeconds, 6);     // the even phoneme-weighted split, not 0.2
    }

    [Fact]
    public void AnEngineThatReportsNoTimingsStillGetsTheWeightedSpread()
    {
        // An engine built before the kokoro_tts family declared word_timestamps. AUDIOCPP_NATIVE_DIR
        // is read at BUILD time, so this is a real configuration and not a formality.
        var supplied = new AudioCppSynthesisService.SuppliedPhonemes(["x"], [0, 1], [1.0, 3.0]);
        var words = AudioCppSynthesisService.Align("one two", supplied, new AudioCppSpeech(new float[24_000], []), 2.0, out _);

        Assert.Equal(2, words.Count);
        Assert.Equal(0.5, words[0].EndSeconds, 6);     // 1/4 of two seconds
    }

    [Fact]
    public void WithNoSuppliedStreamAtAllItIsTheLengthEstimate()
    {
        // A non-English voice: the engine phonemized internally, so there is no map to join its
        // groups to even if it reported them.
        var words = AudioCppSynthesisService.Align("uno dos", null, Spoken(("ˈuno", 0.0, 1.0)), 2.0, out _);
        Assert.Equal(2, words.Count);
        Assert.Equal(2.0, words[^1].EndSeconds, 6);
    }

    [Fact]
    public void OnlyTheMeasuredPathReportsItselfAsMeasured()
    {
        // ⚠ THIS FLAG IS WHAT THE SIDECAR'S ALIGNER NAME IS BUILT FROM, so it has to be true on
        // exactly the tier that earned it. Nothing can answer "will this job be measured?" before
        // a render — the engine option is opt-in, the family declares no capability for it, and a
        // published package's contract predates the option — so the name describes what happened,
        // and this is the thing it describes.
        var supplied = new AudioCppSynthesisService.SuppliedPhonemes(["x"], [0, 1], [1.0, 1.0]);
        var spoken = Spoken(("ə", 0.0, 1.0), ("b", 1.0, 2.0));

        AudioCppSynthesisService.Align("one two", supplied, spoken, 2.0, out var measured);
        Assert.True(measured, "the engine's groups joined to the word map is the measured tier");

        // No timings from the engine: the phoneme-count spread is not a measurement.
        AudioCppSynthesisService.Align("one two", supplied, new AudioCppSpeech(new float[24_000], []),
                                       2.0, out var noTimings);
        Assert.False(noTimings);

        // Timings, but a group count the map disagrees with: falls back, so not measured.
        var mismatched = Spoken(("ə", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 1.0, 2.0));
        AudioCppSynthesisService.Align("one two", supplied, mismatched, 2.0, out var shifted);
        Assert.False(shifted);

        // A non-English voice: no supplied stream at all, so nothing to join the groups to.
        AudioCppSynthesisService.Align("uno dos", null, spoken, 2.0, out var noStream);
        Assert.False(noStream);
    }

    private static TtsJobSettings Job(string voice, float speed = 1.0f) =>
        new(TtsBackendKind.AudioCppKokoro.ToString(), "", voice, speed);

    private static JobRecord Rendered(string voice) =>
        new() { Kind = JobKind.Tts, TtsSettings = Job(voice) };
}
#endif
