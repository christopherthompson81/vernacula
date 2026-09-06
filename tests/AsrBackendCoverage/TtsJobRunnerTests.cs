using System;
using System.IO;
using System.Linq;
using System.Text.Json;
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
            string segDir = TtsJobRunner.SegmentsDirFor(sidecarPath);
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
            var reread = JsonSerializer.Deserialize<AlignmentSidecar>(await File.ReadAllTextAsync(sidecarPath))!;
            Assert.Contains("# A heading", reread.SourceText);
            Assert.Equal(sidecar.Words.Count, reread.Words.Count);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }
}
