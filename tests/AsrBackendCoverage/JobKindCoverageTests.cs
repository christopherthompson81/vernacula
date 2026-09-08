using System;
using System.Collections.Generic;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.Base.Models;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// Every job kind must be described by the row model. These exist because the alternative to a
/// missing case here is not an error but a wrong answer: a new kind silently rendered as a
/// transcription, which reads as a bug in that feature rather than an omission in this one.
/// The navigation in MainViewModel now throws on an unhandled kind for the same reason.
/// </summary>
public class JobKindCoverageTests
{
    public static IEnumerable<object[]> AllKinds =>
        Enum.GetValues<JobKind>().Select(k => new object[] { k });

    private static JobRecord Row(JobKind kind) => new()
    {
        JobId = 1, Kind = kind, ResultsFile = "/tmp/r.json",
        AudioFilePath = "/tmp/a.wav", Status = JobStatus.Complete,
    };

    [Theory]
    [MemberData(nameof(AllKinds))]
    public void EveryKindHasALabelAndAnActionName(JobKind kind)
    {
        var row = Row(kind);
        Assert.False(string.IsNullOrWhiteSpace(row.KindLabel), $"{kind} has no KindLabel");
        Assert.False(string.IsNullOrWhiteSpace(row.LoadLabel), $"{kind} has no LoadLabel");
    }

    [Theory]
    [MemberData(nameof(AllKinds))]
    public void EveryKindClaimsExactlyOneProgressRow(JobKind kind)
    {
        // The two progress bars are mutually exclusive by construction; a kind that claims
        // both would draw twice, and one that claims neither would show no progress at all.
        var row = Row(kind);
        row.Status = JobStatus.Running;
        row.IsActivelyRunning = true;
        Assert.True(row.ShowAsrProgress ^ row.ShowTtsProgress,
            $"{kind} claims {(row.ShowAsrProgress ? "ASR" : "")}{(row.ShowTtsProgress ? "TTS" : "")} progress");
    }

    [Fact]
    public void KindPredicatesPartitionTheKinds()
    {
        // IsAsr/IsTts are used as an either-or across the UI; that only holds while there are
        // exactly two kinds. If a third arrives this fails, which is the point.
        foreach (var kind in Enum.GetValues<JobKind>())
        {
            var row = Row(kind);
            Assert.True(row.IsAsr ^ row.IsTts,
                $"{kind} is neither ASR nor TTS, so every `IsTts ? … : …` in the UI is now wrong for it");
        }
    }
}
