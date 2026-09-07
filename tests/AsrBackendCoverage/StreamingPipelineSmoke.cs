using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.App.Models;
using Vernacula.App.Services;
using Vernacula.Base.Models;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// End-to-end run of the real transcription pipeline on the streaming backend, including the
/// post-decode database write that the unit tests never reach. Needs CUDA and an installed
/// package, so it is opt-in:
///
///   VERNACULA_STREAMING_SMOKE=1 VERNACULA_STREAMING_MODEL=&lt;package dir&gt; \
///   VERNACULA_STREAMING_AUDIO=&lt;wav&gt; dotnet test --filter StreamingPipelineSmoke
/// </summary>
public class StreamingPipelineSmoke
{
    private static string? Env(string k) => Environment.GetEnvironmentVariable(k);

    [Fact]
    public async Task RunsToCompletionAndWritesResults()
    {
        if (Env("VERNACULA_STREAMING_SMOKE") != "1") return;   // opt-in: needs CUDA and a package
        string modelDir = Env("VERNACULA_STREAMING_MODEL")!;
        string audio    = Env("VERNACULA_STREAMING_AUDIO")!;

        string tmp = Path.Combine(Path.GetTempPath(), "vernacula-stream-smoke", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(tmp, "models"));
        // The service resolves the package under <models>/vibevoice_asr_streaming_1_5b.
        Directory.CreateSymbolicLink(
            Path.Combine(tmp, "models", "vibevoice_asr_streaming_1_5b"), Path.GetFullPath(modelDir));

        var settings = new SettingsService();
        settings.Current.ModelsDir              = Path.Combine(tmp, "models");
        settings.Current.AsrBackend             = AsrBackend.VibeVoiceStreaming;
        settings.Current.VibeVoiceStreamingSize = VibeVoiceStreamingSize.Small1_5B;
        settings.Current.Segmentation           = SegmentationMode.VibeVoiceBuiltin;

        var svc = new TranscriptionService(settings, new LangIdService(settings));
        string dbPath = Path.Combine(tmp, "results.db");

        int added = 0, texts = 0;
        var progress = new Progress<TranscriptionProgress>(_ => { });
        await svc.RunAsync(
            audio, -1, dbPath, progress,
            onSegmentAdded: _ => Interlocked.Increment(ref added),
            onSegmentText:  (_, _) => Interlocked.Increment(ref texts),
            asrModelName:   AsrLanguageSupport.ModelName(AsrBackend.VibeVoiceStreaming),
            asrLanguageCode: "auto",
            ct: CancellationToken.None);

        Assert.True(added > 0, "no segments were streamed to the UI");
        Assert.True(texts > 0, "no segment text was streamed to the UI");
        Assert.True(File.Exists(dbPath), "no results database was written");
    }
}
