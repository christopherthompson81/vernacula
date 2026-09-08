using Vernacula.App.Models;
using Vernacula.App.ViewModels;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// What the size picker promises about a card, against what the runtime will actually do with
/// it. These two used to be separate pieces of arithmetic with separate constants, and they
/// disagreed: the warning measured the card by its total VRAM while the runtime measures free,
/// and it compared against a rounded "2 hours" while the 7B package carries 131,072 positions
/// — about 136 minutes. A card good for 130 minutes was therefore told it was fine.
/// </summary>
public class VibeVoiceStreamingSizeWarningTests
{
    private const long GiB = 1L << 30;

    private static string Warn(VibeVoiceStreamingSize size, double freeGiB) =>
        SettingsViewModel.VibeVoiceStreamingWarningFor(size, (long)(freeGiB * GiB));

    private static long KvPerPosition(VibeVoiceStreamingSize size)
    {
        var s = SettingsViewModel.VibeVoiceStreamingShape(size);
        return VibeVoiceStreamingBudget.KvBytesPerPosition(s.Layers, s.KvHeads, s.HeadDim);
    }

    [Fact]
    public void ThePublishedShapesCostWhatTheDocumentationSays()
    {
        Assert.Equal(28 * 1024, KvPerPosition(VibeVoiceStreamingSize.Small1_5B));
        Assert.Equal(56 * 1024, KvPerPosition(VibeVoiceStreamingSize.Large7B));
    }

    [Fact]
    public void ACardShortOfTheFullContextIsWarnedRatherThanReassured()
    {
        // The reported machine: 16 GB, most of it free, running the 7B.
        string w = Warn(VibeVoiceStreamingSize.Large7B, 15.3);
        Assert.Contains("rather than the full", w);
        Assert.Contains("136", w);   // the package's own ceiling, not a rounded "120"
    }

    [Fact]
    public void ACardJustShortOfTheCeilingIsStillWarned()
    {
        // The regression the old threshold hid: between the rounded 120 it compared against and
        // the package's real 137 it said nothing at all.
        var s = SettingsViewModel.VibeVoiceStreamingShape(VibeVoiceStreamingSize.Large7B);
        long free = s.WeightBytes + VibeVoiceStreamingBudget.WorkingSetBytes(s.HiddenSize)
                  + (long)(130 * 60 * 16 + VibeVoiceStreamingBudget.PromptPositionSlack)
                    * KvPerPosition(VibeVoiceStreamingSize.Large7B);
        string w = Warn(VibeVoiceStreamingSize.Large7B, free / (double)GiB);
        Assert.Contains("rather than the full 136", w);
        Assert.Contains("130 minutes", w);
    }

    [Fact]
    public void ACardWithRoomForTheWholeContextSaysNothing()
    {
        Assert.Equal("", Warn(VibeVoiceStreamingSize.Small1_5B, 22.5));
        Assert.Equal("", Warn(VibeVoiceStreamingSize.Large7B, 21.0));
    }

    [Fact]
    public void ACardThatCannotHoldTheModelSaysSoRatherThanQuotingZeroMinutes()
    {
        string w = Warn(VibeVoiceStreamingSize.Large7B, 8.0);
        Assert.Contains("before any audio is cached", w);
        Assert.DoesNotContain("0 minutes", w);
    }

    [Fact]
    public void TheWarningAgreesWithWhatTheRuntimeWouldAllocate()
    {
        // The whole point of sharing the budget: if the picker says an hour fits, planning an
        // hour must not come back refused.
        //
        // Swept rather than sampled. The picker rounds down and the planner rounds up, so the
        // two only disagree at the boundary — a single card size passes by luck, and the first
        // version of this test did exactly that.
        foreach (var size in new[] { VibeVoiceStreamingSize.Small1_5B, VibeVoiceStreamingSize.Large7B })
        {
            var s = SettingsViewModel.VibeVoiceStreamingShape(size);
            long perPos = KvPerPosition(size);
            long floor  = s.WeightBytes + VibeVoiceStreamingBudget.WorkingSetBytes(s.HiddenSize);

            for (long extraMiB = 0; extraMiB <= 8192; extraMiB += 7)   // a prime-ish step, ~1170 cards
            {
                long free = floor + extraMiB * 1024 * 1024;
                double fits = VibeVoiceStreamingBudget.MinutesThatFit(
                    free, s.WeightBytes, s.HiddenSize, perPos, s.Ceiling);
                if (fits <= 0) continue;

                // The planner sees the card after the weights are resident.
                int planned = VibeVoiceStreamingAsr.PlanKvTokens(
                    fits * 60, s.Ceiling, 0, perPos,
                    VibeVoiceStreamingBudget.WorkingSetBytes(s.HiddenSize), free - s.WeightBytes);

                Assert.True(planned >= fits * 60 * VibeVoiceStreamingBudget.PositionsPerSecond,
                    $"{size} with {free / (double)GiB:F2} GiB free: picker promised {fits:F0} " +
                    $"minutes, planner allocated {planned} positions");
            }
        }
    }
}
