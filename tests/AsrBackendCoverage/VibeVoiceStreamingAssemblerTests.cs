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
