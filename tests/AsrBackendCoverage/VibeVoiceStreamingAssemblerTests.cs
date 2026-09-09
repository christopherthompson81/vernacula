using System.Collections.Generic;
using System.Linq;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The streaming backend's promise is that a transcript appears while the audio is still
/// arriving. That only holds if the assembler exposes turns as they open and lets the newest
/// one grow, so these pin the incremental behaviour, and pin it equal to the batch fold.
/// </summary>
public class VibeVoiceStreamingAssemblerTests
{
    private static VibeVoiceStreamingChunk C(int i, string text) =>
        new(i, i * 2.933, (i + 1) * 2.933, text, [], [], []);

    /// <summary>A chunk whose tokens are one character each, so attribution is checkable by eye.</summary>
    private static VibeVoiceStreamingChunk PerChar(int i, string text)
    {
        var ids  = new long[text.Length];
        var lp   = new float[text.Length];
        var ends = new int[text.Length];
        for (int k = 0; k < text.Length; k++) { ids[k] = 1000 + k; lp[k] = -0.5f - k; ends[k] = k + 1; }
        return new(i, i * 2.933, (i + 1) * 2.933, text, ids, lp, ends);
    }

    private static readonly List<VibeVoiceStreamingChunk> Conversation =
    [
        C(0, " \n Speaker 0:Hello there. \n Speaker 1:Some"),
        C(1, "thing follows."),
        C(2, " \n Speaker 0:And back again."),
    ];

    [Fact]
    public void TurnsAppearBeforeTheRecordingEnds()
    {
        var asm = new VibeVoiceStreamingAsr.SegmentAssembler();
        asm.Add(Conversation[0]);
        // Two turns are visible after the very first chunk: the first is closed, the second open.
        Assert.Equal(2, asm.Segments.Count);
        Assert.Equal("Hello there.", asm.Segments[0].Content);
        Assert.Equal(0, asm.Segments[0].Speaker);
        Assert.Equal(1, asm.Segments[1].Speaker);
    }

    [Fact]
    public void TheOpenTurnGrowsWithEachChunkInsteadOfBeingDuplicated()
    {
        var asm = new VibeVoiceStreamingAsr.SegmentAssembler();
        asm.Add(Conversation[0]);
        Assert.Equal("Some", asm.Segments[^1].Content);

        asm.Add(Conversation[1]);
        Assert.Equal(2, asm.Segments.Count);                 // still two turns, not three
        Assert.Equal("Something follows.", asm.Segments[^1].Content);
    }

    [Fact]
    public void IncrementalAndBatchAgree()
    {
        var asm = new VibeVoiceStreamingAsr.SegmentAssembler();
        foreach (var c in Conversation) asm.Add(c);
        var incremental = asm.Finish();
        var batch = VibeVoiceStreamingAsr.ToSegments(Conversation);

        Assert.Equal(batch.Count, incremental.Count);
        Assert.Equal(batch.Select(s => (s.Speaker, s.Content)),
                     incremental.Select(s => (s.Speaker, s.Content)));
    }

    [Fact]
    public void TurnStartsNeverGoBackwardsAndEndsCoverTheirText()
    {
        var segs = VibeVoiceStreamingAsr.ToSegments(Conversation);
        Assert.Equal(3, segs.Count);
        for (int i = 0; i < segs.Count; i++)
        {
            Assert.True(segs[i].End >= segs[i].Start, $"segment {i} ends before it starts");
            if (i > 0) Assert.True(segs[i].Start >= segs[i - 1].Start, $"segment {i} starts before its predecessor");
        }
    }

    [Fact]
    public void TokensFollowTheTextIntoTheirSegment_AndMarkersBelongToNeither()
    {
        // A and B go to the first turn, C and D to the second, and the speaker marker's own
        // tokens to neither, since a marker is structure rather than speech.
        const string text = "AB \n Speaker 1:CD";
        var segs = VibeVoiceStreamingAsr.ToSegments([PerChar(0, text)]);
        Assert.Equal(2, segs.Count);
        Assert.Equal("AB", segs[0].Content);
        Assert.Equal("CD", segs[1].Content);

        Assert.Equal(2, segs[0].TokenIds.Count);
        Assert.Equal(2, segs[1].TokenIds.Count);
        Assert.Equal(segs[0].TokenIds.Count, segs[0].TokenLogprobs.Count);
        // The second turn's tokens are the last two of the chunk, not the first two again.
        Assert.Equal([1000 + text.Length - 2, 1000 + text.Length - 1], segs[1].TokenIds);
    }

    [Fact]
    public void TokensAccumulateAcrossChunksWithinOneTurn()
    {
        var segs = VibeVoiceStreamingAsr.ToSegments([PerChar(0, "abc"), PerChar(1, "de")]);
        var one = Assert.Single(segs);
        Assert.Equal("abcde", one.Content);
        Assert.Equal(5, one.TokenIds.Count);
    }

    [Fact]
    public void ConfidencesAreOmittedRatherThanFaked_WhenTheyWereNotComputed()
    {
        // Transcribe() leaves TokenLogprobs empty unless asked; a segment must then report no
        // confidences at all rather than a count that does not line up with its tokens.
        var chunk = PerChar(0, "hi") with { TokenLogprobs = [] };
        var one = Assert.Single(VibeVoiceStreamingAsr.ToSegments([chunk]));
        Assert.Equal(2, one.TokenIds.Count);
        Assert.Empty(one.TokenLogprobs);
    }

    [Fact]
    public void NonAsciiTextAttributesTokensByCharacter_NotByByte()
    {
        // The offsets a chunk carries are character positions. If they were byte positions,
        // a turn split after a multi-byte word would claim the wrong tokens — and this model
        // covers Chinese, Japanese and Korean, so that is the normal case, not a corner one.
        // "发布" is two characters and six UTF-8 bytes.
        const string text = "发布 \n Speaker 1:ok";
        var chunk = PerChar(0, text);
        var segs = VibeVoiceStreamingAsr.ToSegments([chunk]);

        Assert.Equal(2, segs.Count);
        Assert.Equal("发布", segs[0].Content);
        Assert.Equal("ok", segs[1].Content);
        Assert.Equal(2, segs[0].TokenIds.Count);          // 发, 布 — not 6 bytes' worth
        Assert.Equal(2, segs[1].TokenIds.Count);
    }

    private static VibeVoiceStreamingChunk InPass(int pass, int i, string text) =>
        new(i, i * 2.933, (i + 1) * 2.933, text, [], [], [], pass);

    [Fact]
    public void ASecondPassGetsItsOwnSpeakerLabels()
    {
        // A recording too long for the cache is decoded in passes, and the model renumbers from
        // scratch after each reset. Its "Speaker 0" on the far side is not the same person, so
        // the labels must not collide — deciding they are the same person is the user's call.
        var segs = VibeVoiceStreamingAsr.ToSegments(
        [
            InPass(0, 0, " \n Speaker 0:Hello. \n Speaker 1:Hi."),
            InPass(1, 1, " \n Speaker 0:Later on. \n Speaker 1:Indeed."),
        ]);

        Assert.Equal(4, segs.Count);
        Assert.Equal([0, 1, 2, 3], segs.Select(s => s.Speaker));
        Assert.Equal(["Hello.", "Hi.", "Later on.", "Indeed."], segs.Select(s => s.Content));
    }

    [Fact]
    public void ATurnDoesNotRunAcrossAPassBoundary()
    {
        // Nothing survives the cache reset, so text either side of it belongs to two turns even
        // when the model names nobody in either.
        var segs = VibeVoiceStreamingAsr.ToSegments([InPass(0, 0, "before"), InPass(1, 1, "after")]);
        Assert.Equal(2, segs.Count);
        Assert.Equal("before", segs[0].Content);
        Assert.Equal("after", segs[1].Content);
        Assert.NotEqual(segs[0].Speaker, segs[1].Speaker);
    }

    [Fact]
    public void PassesThatNameNobodyStillDoNotShareALabel()
    {
        // The unnamed opening turn folds onto speaker 0 downstream, so a later pass has to start
        // above it — and a pass with no markers at all still consumes a label of its own.
        var segs = VibeVoiceStreamingAsr.ToSegments(
        [
            InPass(0, 0, "one"),
            InPass(1, 1, "two"),
            InPass(2, 2, " \n Speaker 0:three"),
        ]);
        var labels = segs.Select(s => System.Math.Max(0, s.Speaker)).ToList();
        Assert.Equal(3, segs.Count);
        Assert.Equal(labels.Count, labels.Distinct().Count());
    }

    [Fact]
    public void ASinglePassKeepsTheModelsOwnNumbering()
    {
        // The common case must be untouched by any of the above: no pass boundary, no shift.
        var segs = VibeVoiceStreamingAsr.ToSegments(Conversation);
        Assert.Equal([0, 1, 0], segs.Select(s => s.Speaker));
    }

    [Fact]
    public void ChunksWithNoSpeakerMarkerStillProduceText()
    {
        // A recording whose speaker never changes emits no markers at all; the text must not
        // be dropped for want of one.
        var segs = VibeVoiceStreamingAsr.ToSegments([C(0, "just words"), C(1, " and more")]);
        var one = Assert.Single(segs);
        Assert.Equal("just words and more", one.Content);
        Assert.Equal(-1, one.Speaker);
    }
}
