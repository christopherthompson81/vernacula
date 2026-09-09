using System;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// How large a KV cache a run allocates. The old answer was "the export's ceiling, always",
/// which cost 7.0 GiB on the 7B for a one-minute file and put the checkpoint out of reach of a
/// 16 GB card (issue #150). The second half of that issue is the other direction: a recording
/// too long for the cache is no longer refused, it is decoded in passes, so the planner's job
/// at the limit is to size one pass rather than to say no. These drive the arithmetic directly,
/// because every branch in it is otherwise only reachable with a particular card and a
/// particular recording in front of you.
/// </summary>
public class VibeVoiceStreamingCacheBudgetTests
{
    // The published packages: 28 layers, head dim 128, float16, both sides.
    private const long Kv7B    = 28L * 2 * 4 * 128 * 2;   // 56 KiB a position
    private const long Kv1_5B  = 28L * 2 * 2 * 128 * 2;   // 28 KiB a position
    private const int  Max7B   = 131072;
    private const int  Max1_5B = 65536;
    private const long Work7B  = (1L << 30) + 3584L * 512 * 1024;
    private const long GiB     = 1L << 30;

    private static int Plan(double seconds, long freeBytes, int fixedTokens = 0,
                            int max = Max7B, long perPos = Kv7B, long work = Work7B)
        => VibeVoiceStreamingAsr.PlanKvTokens(seconds, max, fixedTokens, perPos, work, freeBytes);

    [Fact]
    public void ShortRecordingOnASmallCardCostsAFractionOfTheCeiling()
    {
        // The reported case: a 16 GB card, the 7B loaded, a one-minute file.
        int tokens = Plan(60, freeBytes: 6L * GiB);
        Assert.True(tokens < Max7B / 10,
            $"a one-minute file should not need {tokens} of {Max7B} positions");
        Assert.True(tokens * Kv7B < 1 * GiB, "the cache for a minute of audio should be under a GiB");
    }

    [Fact]
    public void TheCeilingIsStillTheCeilingForALongRecording()
        => Assert.Equal(Max7B, Plan((Max7B - 512) / 16.0, freeBytes: 20L * GiB));

    [Fact]
    public void ARecordingLongerThanTheContextTakesTheCeilingAndIsSplit()
    {
        // It used to be refused. A cache that cannot hold the recording holds one pass of it,
        // and the run resets between passes (issue #150).
        Assert.Equal(Max7B, Plan(Max7B / 16.0 + 600, freeBytes: 40L * GiB));
    }

    [Fact]
    public void MoreIsAllocatedThanTheAverageEstimate()
    {
        // The buffer cannot grow mid-run, and the estimate is an average, so a recording denser
        // than average must still fit.
        int tokens = Plan(600, freeBytes: 20L * GiB);
        Assert.True(tokens > 600 * 16.0 * 2,
            $"{tokens} positions leaves no room for denser-than-average speech");
    }

    [Fact]
    public void AFreeVramShortfallTakesWhatTheCardHasInsteadOfRefusing()
    {
        // The reported case: room for the working set and about ten minutes of cache, asked for
        // thirty. Refusing this is what made a 28-minute file impossible on a 16 GB card; the
        // answer is now ten minutes of cache and three passes through it.
        long free   = Work7B + 10 * 60 * 16 * Kv7B;
        int  tokens = Plan(30 * 60, freeBytes: free);
        Assert.InRange(tokens, (int)(9 * 60 * 16), (int)(10 * 60 * 16));
        Assert.True(tokens * Kv7B <= free - Work7B, "the cache must still fit in what is free");
    }

    [Fact]
    public void ACardWithNoRoomForACacheAtAllIsStillRefused()
    {
        // Splitting has a floor: below a cache that holds a couple of minutes there is nothing
        // useful to split into, and the run should say so rather than thrash.
        long free = Work7B + 600 * Kv7B;
        var ex = Assert.Throws<InvalidOperationException>(() => Plan(30 * 60, freeBytes: free));
        Assert.Contains("Not enough free GPU memory", ex.Message);
    }

    [Fact]
    public void AVeryLongRecordingOnASmallCardIsSizedByTheCardNotTheRecording()
    {
        // Both limits bite at once: the card affords less than the ceiling, and the recording
        // needs more than either. The card wins, and neither is a refusal.
        long free   = Work7B + 20L * 60 * 16 * Kv7B;
        int  tokens = Plan(4 * 60 * 60, freeBytes: free);
        Assert.True(tokens < Max7B, "the ceiling is not affordable here");
        Assert.True(tokens * Kv7B <= free - Work7B, "the cache must fit in what is free");
    }

    [Fact]
    public void FreeVramClampsBelowTheSafetyFactorWithoutRefusing()
    {
        // Enough for the recording itself but not 2.5x it: take what there is rather than refuse.
        long free = Work7B + (long)(10 * 60 * 16 * 1.5) * Kv7B;
        int tokens = Plan(10 * 60, freeBytes: free);
        Assert.InRange(tokens, (int)(10 * 60 * 16), (int)(10 * 60 * 16 * 1.6));
    }

    [Fact]
    public void AGraphWithABakedInLengthStillGetsExactlyThatLength()
    {
        // Packages published before this change declare the length as a fixed dimension, and ORT
        // rejects any other size, so they keep paying for the whole ceiling.
        Assert.Equal(Max7B, Plan(60, freeBytes: 20L * GiB, fixedTokens: Max7B));

        // Nothing can be split out of a fixed length either: ORT rejects a shorter buffer, so
        // this stays the one length-related refusal.
        var ex = Assert.Throws<InvalidOperationException>(
            () => Plan(60, freeBytes: 6L * GiB, fixedTokens: Max7B));
        Assert.Contains("re-download", ex.Message);
    }

    [Fact]
    public void WithoutAnNvmlAnswerTheRecordingAloneDecides()
    {
        int tokens = Plan(60, freeBytes: 0);
        Assert.True(tokens < Max7B / 10, "an unknown card is not a reason to allocate the ceiling");
    }

    [Fact]
    public void TheSmallCheckpointIsCheaperPerPosition()
    {
        int small = Plan(600, freeBytes: 20L * GiB, max: Max1_5B, perPos: Kv1_5B,
                         work: (1L << 30) + 1536L * 512 * 1024);
        int large = Plan(600, freeBytes: 20L * GiB);
        Assert.Equal(small, large);                       // same recording, same positions
        Assert.True(small * Kv1_5B * 2 == large * Kv7B);  // half the bytes for them
    }
}
