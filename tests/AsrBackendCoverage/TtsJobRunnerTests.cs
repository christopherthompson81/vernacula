using Vernacula.Tts.Base.Alignment;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.App.Services.Tts;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// End-to-end: the job runner turns a markdown document into a WAV + alignment sidecar the
/// reader can open. Needs a Kokoro export and the phonemizer data on this machine (the paths
/// the app itself uses — its settings.json), so it skips on a runner without them; the point
/// is a dev-box check that the queue's TTS path really renders, not CI coverage.
/// </summary>
public class TtsJobRunnerTests
{
    [Fact]
    public async Task KokoroJobWritesWavAndSidecarWithSourceText()
    {
        var settings = new SettingsService();
        settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");

        string dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string doc = Path.Combine(dir, "page.md");
            await File.WriteAllTextAsync(doc, "# A heading\n\nOne short sentence. And *another* one.\n");
            string sidecarPath = Path.Combine(dir, "page_tts.json");

            string voice = Directory.EnumerateFiles(Path.Combine(settings.GetKokoroModelsDir(), "voices"), "*.bin")
                .Select(Path.GetFileNameWithoutExtension).OrderBy(v => v).First()!;
            var tts = new TtsJobSettings("Kokoro", "", voice);

            using var runner = new TtsJobRunner(settings);
            int chunks = 0;
            var sidecar = await runner.RunAsync(doc, sidecarPath, tts,
                onChunkProduced: _ => Interlocked.Increment(ref chunks),
                onProgress: _ => { },
                CancellationToken.None);

            Assert.True(File.Exists(sidecarPath));
            Assert.True(File.Exists(Path.ChangeExtension(sidecarPath, ".wav")));
            Assert.True(chunks > 0);
            Assert.True(sidecar.AudioDurationSeconds > 0.5);
            // One aligned word per whitespace-split word of the extracted text: that 1:1 match
            // is what lets the reader attach timing by running index.
            int expectedWords = Vernacula.Tts.Base.Markdown.MarkdownTextExtractor.Extract(await File.ReadAllTextAsync(doc))
                .Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
            Assert.Equal(expectedWords, sidecar.Words.Count);

            // One paragraph per segment, each with its own WAV beside the sidecar, so a single
            // paragraph can be re-rendered later without touching the rest.
            Assert.Equal(2, sidecar.Chunks.Count);   // heading + paragraph
            Assert.Equal(new[] { "Heading", "Paragraph" }, sidecar.Chunks.Select(c => c.BlockKind));
            string segDir = AlignmentSidecar.SegmentsDirFor(sidecarPath);
            foreach (var c in sidecar.Chunks)
            {
                Assert.NotNull(c.AudioFile);
                Assert.True(File.Exists(Path.Combine(segDir, c.AudioFile!)), c.AudioFile);
            }
            Assert.Equal(sidecar.Chunks.Count, Directory.GetFiles(segDir, "seg_*.wav").Length);
            // Segment audio lengths add up to the concatenated file's duration.
            double segSum = sidecar.Chunks.Sum(c => c.AudioEndSeconds - c.AudioStartSeconds);
            Assert.InRange(segSum, sidecar.AudioDurationSeconds - 1e-6, sidecar.AudioDurationSeconds + 1e-6);

            // The reader rebuilds its view from the sidecar alone.
            var reread = AlignmentSidecar.Load(sidecarPath);
            Assert.Contains("# A heading", reread.SourceText);
            Assert.Equal(sidecar.Words.Count, reread.Words.Count);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }

    /// <summary>
    /// ⚠ A FINISHED JOB USED TO LEAVE ITS WEIGHTS RESIDENT FOR THE REST OF THE SESSION. The cache was
    /// dropped only when the settings key changed or the app exited, so a bulk synthesis held its
    /// model — ~400 MB of host memory for Kokoro, multiples of that for the others, and DEVICE memory
    /// under a GPU execution provider, which another process cannot borrow while it is held.
    /// </summary>
    [Fact]
    public async Task AnIdleRunnerReleasesItsModel()
    {
        var settings = new SettingsService();
        settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");
        settings.Current.TtsModelIdleReleaseSeconds = 1;

        string dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var (doc, sidecarPath, tts) = await SeedAsync(dir, settings);
            using var runner = new TtsJobRunner(settings);
            await runner.RunAsync(doc, sidecarPath, tts, _ => { }, _ => { }, CancellationToken.None);

            // Still held immediately after the job: the reader re-renders a paragraph through the
            // same backend as soon as the user edits, and that is exactly now.
            Assert.True(runner.IsModelLoaded);

            var deadline = DateTime.UtcNow.AddSeconds(20);
            while (runner.IsModelLoaded && DateTime.UtcNow < deadline) await Task.Delay(100);
            Assert.False(runner.IsModelLoaded);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>Zero is not the same as never: it releases as soon as the work stops.</summary>
    [Fact]
    public async Task ZeroSecondsReleasesAsSoonAsTheWorkStops()
    {
        var settings = new SettingsService();
        settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");
        settings.Current.TtsModelIdleReleaseSeconds = 0;

        string dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var (doc, sidecarPath, tts) = await SeedAsync(dir, settings);
            using var runner = new TtsJobRunner(settings);
            await runner.RunAsync(doc, sidecarPath, tts, _ => { }, _ => { }, CancellationToken.None);

            var deadline = DateTime.UtcNow.AddSeconds(10);
            while (runner.IsModelLoaded && DateTime.UtcNow < deadline) await Task.Delay(50);
            Assert.False(runner.IsModelLoaded);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    /// <summary>⚠ And a negative value keeps the behaviour the app had before the timer existed, so
    /// a user who would rather spend the memory than the reload can say so.</summary>
    [Fact]
    public async Task ANegativeValueKeepsTheModelLoaded()
    {
        var settings = new SettingsService();
        settings.Load();
        if (TtsPrerequisites.Describe(TtsBackendKind.Kokoro, settings) is { } missing)
            Assert.Skip($"Kokoro not available here: {missing}");
        settings.Current.TtsModelIdleReleaseSeconds = -1;

        string dir = Path.Combine(Path.GetTempPath(), "vernacula-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var (doc, sidecarPath, tts) = await SeedAsync(dir, settings);
            using var runner = new TtsJobRunner(settings);
            await runner.RunAsync(doc, sidecarPath, tts, _ => { }, _ => { }, CancellationToken.None);

            await Task.Delay(2000);
            Assert.True(runner.IsModelLoaded);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    private static async Task<(string Doc, string Sidecar, TtsJobSettings Tts)> SeedAsync(
        string dir, SettingsService settings)
    {
        string doc = Path.Combine(dir, "page.md");
        await File.WriteAllTextAsync(doc, "One short sentence.\n");
        string voice = Directory.EnumerateFiles(Path.Combine(settings.GetKokoroModelsDir(), "voices"), "*.bin")
            .Select(Path.GetFileNameWithoutExtension).OrderBy(v => v).First()!;
        return (doc, Path.Combine(dir, "page_tts.json"), new TtsJobSettings("Kokoro", "", voice));
    }
}
