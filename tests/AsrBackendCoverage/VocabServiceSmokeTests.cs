using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.Tests.AsrBackendCoverage.Fixtures;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// Per-<see cref="VocabService.VocabKind"/> decode smoke tests (issue #38).
///
/// <para>
/// The Granite Speech editor bug (PR #35) was a missing <c>VocabKind</c> case: the dispatch
/// fell through to the Parakeet text loader, which read a JSON <c>vocab.json</c> as lines,
/// produced an empty vocab, and left the editor rendering raw GPT-2 ByteLevel characters. The
/// existing coverage test proves each backend's model name is RECOGNISED; nothing proved the
/// branch it selects can actually decode. These tests close that gap: every kind loads a tiny
/// fixture vocab from disk and decodes a known token sequence.
/// </para>
///
/// <para>
/// They also pin the three HF-format kinds APART. VibeVoice, Qwen3/Whisper and Granite share a
/// loader and a byte table but post-process differently — a boundary-quote trim, a
/// leading-space strip, and neither — so "it decoded something" would not catch one being
/// wired to another's decoder. The divergence tests below are the ones that would.
/// </para>
/// </summary>
[Collection(nameof(AsrBackendCoverageTests))]
public class VocabServiceSmokeTests
{
    // Parameterised by BACKEND rather than by kind: VocabKind is internal, so it cannot appear
    // in a public test signature, and the backend is what a caller actually has anyway. Every
    // kind is still exercised — EveryVocabKindIsReachableFromABackend keeps that true.
    public static IEnumerable<object[]> AllBackends =>
        Enum.GetValues<AsrBackend>().Select(b => new object[] { b });

    private static VocabService.VocabKind KindOf(AsrBackend backend) => VocabService.KindOfBackend(backend);

    private static VocabService Load(AsrBackend backend, string dir) =>
        new(dir, AsrLanguageSupport.ModelName(backend));

    private static VocabService Load(VocabService.VocabKind kind, string dir) =>
        new(dir, AsrLanguageSupport.ModelName(
            Enum.GetValues<AsrBackend>().First(b => VocabService.KindOfBackend(b) == kind)));

    [Fact]
    public void EveryVocabKindIsReachableFromABackend()
    {
        // The theories below iterate backends, so a VocabKind no backend maps to would go
        // untested in silence. Fail here instead.
        var covered = Enum.GetValues<AsrBackend>().Select(VocabService.KindOfBackend).ToHashSet();
        Assert.Equal(Enum.GetValues<VocabService.VocabKind>().ToHashSet(), covered);
    }

    // ── Every kind decodes its own fixture ───────────────────────────────────

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void EveryKind_DecodesItsFixtureToTheKnownPhrase(AsrBackend backend)
    {
        using var fixture = VocabFixtures.Write(KindOf(backend));
        var vocab = Load(backend, fixture.Dir);
        Assert.Equal(VocabFixtures.Phrase, vocab.DecodeTokens(fixture.PhraseTokens));
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void EveryKind_ProducesOneRunPerToken_ThatConcatenatesToTheDecodedText(AsrBackend backend)
    {
        using var fixture = VocabFixtures.Write(KindOf(backend));
        var vocab = Load(backend, fixture.Dir);
        {
            var tokens = fixture.PhraseTokens;
            var logprobs = tokens.Select((_, i) => -0.1f * (i + 1)).ToArray();

            var runs = vocab.GetTokenRuns(tokens, logprobs);

            // One run per token: the editor pairs runs with tokens positionally, so a
            // dropped or merged run silently misattributes every later token's confidence.
            Assert.Equal(tokens.Length, runs.Count);
            Assert.Equal(logprobs, runs.Select(r => r.logprob));

            // ⚠ NOT "every run is non-empty". The fixture splits "ö" across two tokens, and a
            // streaming UTF-8 decoder correctly emits nothing for the first half — that is the
            // behaviour being checked, not a defect. What must hold is that the runs together
            // say the same thing as the whole-sequence decode.
            Assert.Equal(vocab.DecodeTokens(tokens).Trim(),
                         string.Concat(runs.Select(r => r.text)).Trim());
            Assert.Contains(runs, r => r.text.Length > 0);
        }
    }

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void EveryKind_ToleratesShortLogprobsAndUnknownTokens(AsrBackend backend)
    {
        using var fixture = VocabFixtures.Write(KindOf(backend));
        var vocab = Load(backend, fixture.Dir);

        // A token id past the end of the vocab reaches the editor whenever a model and its
        // vocab drift apart; it must render as nothing rather than throw.
        int[] tokens = [.. fixture.PhraseTokens, 99_999];

        var runs = vocab.GetTokenRuns(tokens, [-0.1f]);   // deliberately too few logprobs

        Assert.Equal(tokens.Length, runs.Count);
        Assert.Equal(0f, runs[^1].logprob);               // missing logprobs default, not throw
        Assert.Equal("", runs[^1].text);
        Assert.Equal(VocabFixtures.Phrase, vocab.DecodeTokens(tokens));
    }

    // ── The HF-format kinds must not be wired to each other's decoder ────────

    [Fact]
    public void VibeVoice_TrimsBoundaryQuotes_WhereGraniteKeepsThem()
    {
        // VibeVoice is a JSON-mode model whose output arrives wrapped in quotes; Granite is
        // plain ASR, where a quote is content. Same bytes, deliberately different results.
        int[] quoted = [VocabFixtures.Quote, .. HfPhrase, VocabFixtures.Quote];

        string vibe = Decode(VocabService.VocabKind.VibeVoice, quoted);
        string granite = Decode(VocabService.VocabKind.GraniteSpeech, quoted);

        Assert.Equal(VocabFixtures.Phrase, vibe);
        Assert.Equal($"\"{VocabFixtures.Phrase}\"", granite);
    }

    [Fact]
    public void Qwen3AndWhisper_StripOneLeadingSpace_WhereGraniteKeepsIt()
    {
        int[] leadingSpace = [VocabFixtures.SpaceHello, VocabFixtures.SpaceW,
                              VocabFixtures.OSplitHigh, VocabFixtures.OSplitLowRld, VocabFixtures.Dot];

        Assert.Equal(VocabFixtures.Phrase, Decode(VocabService.VocabKind.Qwen3Asr, leadingSpace));
        Assert.Equal(VocabFixtures.Phrase, Decode(VocabService.VocabKind.WhisperTurbo, leadingSpace));
        Assert.Equal($" {VocabFixtures.Phrase}", Decode(VocabService.VocabKind.GraniteSpeech, leadingSpace));
    }

    [Fact]
    public void SpecialTokens_RenderAsContentForVibeVoiceAndGranite_AndAreDroppedByQwen3AndWhisper()
    {
        int[] withSpecial = [.. HfPhrase, VocabFixtures.Special];
        string expectedWithContent = VocabFixtures.Phrase + VocabFixtures.SpecialContent;

        Assert.Equal(expectedWithContent, Decode(VocabService.VocabKind.VibeVoice, withSpecial));
        Assert.Equal(expectedWithContent, Decode(VocabService.VocabKind.GraniteSpeech, withSpecial));
        Assert.Equal(VocabFixtures.Phrase, Decode(VocabService.VocabKind.Qwen3Asr, withSpecial));
        Assert.Equal(VocabFixtures.Phrase, Decode(VocabService.VocabKind.WhisperTurbo, withSpecial));
    }

    /// <summary>The phrase's tokens in the shared HF-format fixture (the divergence tests are HF-only).</summary>
    private static int[] HfPhrase => VocabFixtures.Write(VocabService.VocabKind.GraniteSpeech) is var f
        ? Use(f) : [];

    private static int[] Use(VocabFixtures.Fixture f)
    {
        using (f) return f.PhraseTokens;
    }

    private static string Decode(VocabService.VocabKind kind, IReadOnlyList<int> tokens)
    {
        using var fixture = VocabFixtures.Write(kind);
        return Load(kind, fixture.Dir).DecodeTokens(tokens);
    }

    // ── The loud-fallthrough path ────────────────────────────────────────────

    [Fact]
    public void UnrecognisedModelName_WarnsAndStillLoads()
    {
        // The other half of the Granite bug: a backend whose branch is missing must say so on
        // stderr rather than mis-decode in silence. Console.SetError is process-global, hence
        // this class's [Collection] with parallelization disabled.
        using var fixture = VocabFixtures.Write(VocabService.VocabKind.Parakeet);
        var origErr = Console.Error;
        using var stderr = new StringWriter();
        VocabService vocab;
        Console.SetError(stderr);
        try { vocab = new VocabService(fixture.Dir, "acme/some-unreleased-asr"); }
        finally { Console.SetError(origErr); }

        string err = stderr.ToString();
        Assert.Contains("WARNING", err, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("acme/some-unreleased-asr", err, StringComparison.Ordinal);

        // And it falls back to Parakeet rather than throwing — the editor still opens.
        Assert.Equal(VocabFixtures.Phrase, vocab.DecodeTokens(fixture.PhraseTokens));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void NoModelName_FallsBackToParakeetSilently(string? asrModel)
    {
        // A job whose asr_model row has not been written yet is a legitimate caller, not a
        // dispatch gap: it must not cry wolf on stderr.
        using var fixture = VocabFixtures.Write(VocabService.VocabKind.Parakeet);
        var origErr = Console.Error;
        using var stderr = new StringWriter();
        VocabService vocab;
        Console.SetError(stderr);
        try { vocab = new VocabService(fixture.Dir, asrModel); }
        finally { Console.SetError(origErr); }

        Assert.DoesNotContain("WARNING", stderr.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.Equal(VocabFixtures.Phrase, vocab.DecodeTokens(fixture.PhraseTokens));
    }

    // ── Missing files ────────────────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(AllBackends))]
    public void EveryKind_SurvivesAMissingVocabFile(AsrBackend backend)
    {
        // Every loader returns an empty vocab for a missing file: the editor opens on a job
        // whose model was never downloaded, showing nothing rather than crashing.
        string empty = Path.Combine(Path.GetTempPath(), "vernacula-vocab-fixtures", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(empty);
        try
        {
            var vocab = Load(backend, empty);
            int[] tokens = [VocabFixtures.Hello, VocabFixtures.SpaceW, VocabFixtures.Dot];
            Assert.Equal("", vocab.DecodeTokens(tokens));
            Assert.Equal(tokens.Length, vocab.GetTokenRuns(tokens, []).Count);
        }
        finally { VocabFixtures.Cleanup(empty); }
    }
}
