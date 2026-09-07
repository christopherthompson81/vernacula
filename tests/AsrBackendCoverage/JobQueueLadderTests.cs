using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The queue used to carry one copy of the running → complete/cancelled/failed ladder per job
/// kind, and nothing failed to compile when a copy drifted (only the TTS one recorded an output
/// duration). Issue #129 collapsed them onto <see cref="IJobRunner"/>; these pin the shared
/// ladder down so a future third kind cannot quietly skip a rung.
/// <para>
/// Both jobs here point at a file that exists but holds nothing either worker can use, so each
/// fails inside its worker without loading a model — the enqueue path hashes the file, so it has
/// to be real. That is deliberate: the failure path exercises every rung —
/// status transitions, the run-time stopwatch, the error column and the live-state cleanup —
/// while staying a unit test.
/// </para>
/// </summary>
public class JobQueueLadderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public JobQueueLadderTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>
    /// Builds a real queue over a temp control DB. Every dependency here is a field assignment —
    /// no model is touched until a job actually runs.
    /// </summary>
    private (JobQueueService queue, ControlDb db) NewQueue()
    {
        var settings = new SettingsService();
        var db       = new ControlDb(Path.Combine(_dir, "control.db"));
        var queue    = new JobQueueService(
            new TranscriptionService(settings, new LangIdService(settings)),
            new TtsJobRunner(settings),
            db,
            settings);
        return (queue, db);
    }

    /// <summary>
    /// The constructor asserts a runner exists for every <see cref="JobKind"/>, so simply
    /// building the queue is the coverage check: add a kind without a runner and this fails.
    /// </summary>
    [Fact]
    public void EveryJobKindHasARegisteredRunner()
    {
        var (queue, db) = NewQueue();
        using (db) Assert.NotNull(queue);
    }

    private static async Task WaitUntil(Func<bool> condition)
    {
        for (int i = 0; i < 500 && !condition(); i++)
            await Task.Delay(10, TestContext.Current.CancellationToken);
        Assert.True(condition(), "condition was still false after 5 s");
    }

    [Theory]
    [InlineData(JobKind.Asr)]
    [InlineData(JobKind.Tts)]
    public async Task AFailingJobWalksTheSameLadderWhateverItsKind(JobKind kind)
    {
        var (queue, db) = NewQueue();
        using var _ = db;

        var seen     = new List<JobStatus>();
        var finished = new TaskCompletionSource();
        queue.JobStatusChanged += (_, status, _, _) =>
        {
            lock (seen) seen.Add(status);
            if (status is JobStatus.Complete or JobStatus.Cancelled or JobStatus.Failed)
                finished.TrySetResult();
        };

        // Empty: the TTS runner rejects an empty document, and the ASR runner cannot decode it.
        string unusable = Path.Combine(_dir, "unusable.dat");
        await File.WriteAllTextAsync(unusable, "", TestContext.Current.CancellationToken);

        int jobId = kind == JobKind.Tts
            ? await queue.EnqueueNewTtsJobAsync(unusable, "tts", new TtsJobSettings("Kokoro", "en", "af_heart"))
            : await queue.EnqueueNewJobAsync(unusable, "asr");

        await finished.Task.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);

        // Queued, then Running, then a terminal status — the rungs the queue owns for every kind.
        lock (seen)
        {
            Assert.Equal(JobStatus.Queued,  seen[0]);
            Assert.Equal(JobStatus.Running, seen[1]);
            Assert.Equal(JobStatus.Failed,  seen[^1]);
        }

        var row = Assert.Single(db.GetJobs(), j => j.JobId == jobId);
        Assert.Equal(JobStatus.Failed, row.Status);
        Assert.Equal(kind, row.Kind);
        Assert.True(row.HasError);
        Assert.NotNull(row.RunTimeSeconds);

        // The live UI state is dropped BEFORE the status event fires, so it is already gone by
        // the time a subscriber is called — that is what lets a panel react to completion by
        // reading the database rather than the in-flight state.
        Assert.Null(queue.GetJobUiState(jobId));
        Assert.Null(queue.GetTtsJobUiState(jobId));
        Assert.Equal(0, queue.GetJobProgress(jobId));

        // The cancellation token is released just AFTER the event, deliberately: a subscriber
        // that asks IsJobActivelyRunning while handling completion still sees the job as active.
        // So it clears a moment later rather than during the callback.
        await WaitUntil(() => !queue.IsAnyJobRunning);
        Assert.False(queue.IsJobActivelyRunning(jobId));
    }
}
