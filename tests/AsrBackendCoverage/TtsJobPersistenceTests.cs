using System;
using System.IO;
using System.Linq;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The jobs table gained a kind column and per-job TTS settings when the TTS reader was folded
/// into the desktop app. These pin down the two things that would silently misbehave rather
/// than fail: an old database opening with every row still an ASR job, and a TTS job's
/// settings surviving the round trip so a requeue renders what was asked for.
/// </summary>
public class TtsJobPersistenceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));

    public TtsJobPersistenceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void AsrJobsFromBeforeTheKindColumnReadBackAsAsr()
    {
        string dbPath = Path.Combine(_dir, "control.db");
        using (var db = new ControlDb(dbPath))
            db.InsertNewJob("older", Path.Combine(_dir, "a_results.sqlite3"), "/media/a.wav", "abc", "2026-01-01 00:00:00");

        using (var reopened = new ControlDb(dbPath))
        {
            var job = Assert.Single(reopened.GetJobs());
            Assert.Equal(JobKind.Asr, job.Kind);
            Assert.False(job.IsTts);
            Assert.Equal("", job.TtsBackend);
            Assert.Null(job.OutputDurationSeconds);
        }
    }

    [Fact]
    public void TtsJobSettingsRoundTrip()
    {
        string dbPath = Path.Combine(_dir, "control.db");
        var tts = new TtsJobSettings("OmniVoice", "cy", "cy_default", Speed: 1.25f, NumStep: 24);
        string sidecar = Path.Combine(_dir, "doc_tts.json");

        using var db = new ControlDb(dbPath);
        int id = db.InsertNewTtsJob("read me", sidecar, "/docs/page.md", "0123456789abcdef", "2026-01-01 00:00:00", tts);
        db.UpdateJobOutputDuration(id, 12.5);
        db.UpdateJobStatus(id, JobStatus.Complete, runTimeSeconds: 7);

        var job = Assert.Single(db.GetJobs());
        Assert.Equal(JobKind.Tts, job.Kind);
        Assert.Equal("OmniVoice", job.TtsBackend);
        Assert.Equal("cy", job.TtsLanguage);
        Assert.Equal("cy_default", job.TtsVoice);
        Assert.Equal(1.25f, job.TtsSpeed);
        Assert.Equal(24, job.TtsNumStep);
        Assert.Equal(12.5, job.OutputDurationSeconds);
        Assert.Equal("/docs/page.md", job.AudioFilePath);
        Assert.Equal(Path.ChangeExtension(sidecar, ".wav"), job.OutputAudioPath);
        // A finished TTS job's Time column is the audio length, not the wall-clock run time.
        Assert.Equal("13s", job.RunTimeLabel);
    }

    [Fact]
    public void ReaddingTheSameDocumentWithTheSameSettingsReusesTheRow()
    {
        string dbPath = Path.Combine(_dir, "control.db");
        var tts = new TtsJobSettings("Kokoro", "", "af_heart");
        string sidecar = Path.Combine(_dir, "doc_tts.json");

        using var db = new ControlDb(dbPath);
        int first  = db.InsertNewTtsJob("one", sidecar, "/docs/page.md", "sha", "2026-01-01 00:00:00", tts);
        db.UpdateJobOutputDuration(first, 3);
        int second = db.InsertNewTtsJob("two", sidecar, "/docs/page.md", "sha", "2026-01-01 00:00:00", tts);

        Assert.Equal(first, second);
        var job = Assert.Single(db.GetJobs());
        Assert.Equal("two", job.JobTitle);
        Assert.Null(job.OutputDurationSeconds);   // a re-render starts without a stale duration
    }

    [Fact]
    public void ReaddingAFinishedJobPutsItBackInTheQueue()
    {
        // A reused row must not keep the previous run's outcome: Home would show the old result
        // (and its Resume button) while the new run overwrites the files underneath it.
        string dbPath = Path.Combine(_dir, "control.db");
        var tts = new TtsJobSettings("Kokoro", "", "af_heart");
        string sidecar = Path.Combine(_dir, "doc_tts.json");

        using var db = new ControlDb(dbPath);
        int id = db.InsertNewTtsJob("one", sidecar, "/docs/page.md", "sha", "2026-01-01 00:00:00", tts);
        db.SetJobRunning(id, "2026-01-01 00:00:01");
        db.UpdateJobOutputDuration(id, 9);
        db.UpdateJobStatus(id, JobStatus.Failed, "boom", runTimeSeconds: 4);

        db.InsertNewTtsJob("again", sidecar, "/docs/page.md", "sha", "2026-01-01 00:00:00", tts);

        var job = Assert.Single(db.GetJobs());
        Assert.Equal(JobStatus.Queued, job.Status);
        Assert.Null(job.ErrorMessage);
        Assert.Null(job.RunTimeSeconds);
        Assert.Null(job.OutputDurationSeconds);
        Assert.Null(job.TranscriptionRunDatestamp);
    }

    [Fact]
    public void ReaddingAFinishedAsrJobPutsItBackInTheQueueToo()
    {
        string dbPath = Path.Combine(_dir, "control.db");
        string results = Path.Combine(_dir, "a_results.sqlite3");
        using var db = new ControlDb(dbPath);
        int id = db.InsertNewJob("one", results, "/media/a.wav", "abc", "2026-01-01 00:00:00");
        db.UpdateJobStatus(id, JobStatus.Complete, runTimeSeconds: 12);

        db.InsertNewJob("again", results, "/media/a.wav", "abc", "2026-01-01 00:00:00");

        var job = Assert.Single(db.GetJobs());
        Assert.Equal(JobStatus.Queued, job.Status);
        Assert.Null(job.RunTimeSeconds);
    }

    [Fact]
    public void ResultsFileNameIsCultureIndependent()
    {
        // The key names a file on disk; a comma decimal separator would give the same job two
        // different names depending on the UI language.
        const string sha = "0123456789abcdef0123456789abcdef";
        var tts = new TtsJobSettings("Kokoro", "", "af_heart", Speed: 1.25f);
        var previous = System.Globalization.CultureInfo.CurrentCulture;
        try
        {
            System.Globalization.CultureInfo.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
            string german = JobQueueService.TtsResultsFileName(sha, tts);
            System.Globalization.CultureInfo.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            Assert.Equal(JobQueueService.TtsResultsFileName(sha, tts), german);
        }
        finally { System.Globalization.CultureInfo.CurrentCulture = previous; }
    }

    [Fact]
    public void ResultsFileNameKeysOnDocumentAndRenderingChoices()
    {
        const string sha = "0123456789abcdef0123456789abcdef";
        string a  = JobQueueService.TtsResultsFileName(sha, new TtsJobSettings("Kokoro", "", "af_heart"));
        string a2 = JobQueueService.TtsResultsFileName(sha, new TtsJobSettings("Kokoro", "", "af_heart"));
        string b  = JobQueueService.TtsResultsFileName(sha, new TtsJobSettings("Kokoro", "", "am_adam"));

        Assert.Equal(a, a2);
        Assert.NotEqual(a, b);
        Assert.StartsWith(sha[..16] + "_", a);
        Assert.EndsWith("_tts.json", a);
    }
}
