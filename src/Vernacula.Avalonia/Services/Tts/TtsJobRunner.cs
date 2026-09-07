using Vernacula.Tts.Base.Alignment;
using Vernacula.App.Models;
using Vernacula.Tts.Base;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Runs one text-to-speech job end to end for the job queue: reads the document, picks the
/// backend the job asked for, streams chunks out (for the reader panel) and, when done, writes
/// the rendered WAV plus an alignment sidecar next to it in the jobs directory.
/// <para>
/// The backend is cached across jobs — the model load is the sticky cost (seconds to tens of
/// seconds, and gigabytes of weights) — and rebuilt only when the next job needs a different
/// backend or the model locations in Settings changed. The queue runs one job at a time, so
/// there is never a job mid-flight when the cache is swapped.
/// </para>
/// </summary>
internal sealed class TtsJobRunner : IDisposable
{
    private readonly SettingsService _settings;

    private ITtsBackend? _backend;
    private string?      _backendKey;

    public TtsJobRunner(SettingsService settings) => _settings = settings;

    /// <summary>Sample rate of the backend a job with <paramref name="tts"/> will produce.</summary>
    public static int SampleRateFor(TtsJobSettings tts) => TtsEngines.For(tts.Backend).SampleRate;

    public static TtsBackendKind ParseBackend(string name) => TtsEngines.For(name).Kind;

    /// <summary>
    /// Synthesizes <paramref name="documentPath"/> and writes <paramref name="sidecarPath"/>
    /// (JSON) plus the WAV at the same path with a .wav extension. Returns the sidecar.
    /// </summary>
    public async Task<AlignmentSidecar> RunAsync(
        string                        documentPath,
        string                        sidecarPath,
        TtsJobSettings                tts,
        Action<ChunkProducedEvent>    onChunkProduced,
        Action<ProgressEvent>         onProgress,
        CancellationToken             ct)
    {
        string text = await File.ReadAllTextAsync(documentPath, ct);
        if (string.IsNullOrWhiteSpace(text))
            throw new InvalidOperationException($"\"{documentPath}\" is empty.");

        var backend = EnsureBackend(tts);
        string wavPath = Path.ChangeExtension(sidecarPath, ".wav");
        string segmentsDir = AlignmentSidecar.SegmentsDirFor(sidecarPath);
        Directory.CreateDirectory(Path.GetDirectoryName(sidecarPath)!);
        // A re-render starts clean: a shorter document must not leave stale paragraph files.
        if (Directory.Exists(segmentsDir)) Directory.Delete(segmentsDir, recursive: true);

        var request = TtsEngines.For(tts.Backend).BuildRequest(text, wavPath, segmentsDir, tts);

        var result = await backend.SynthesizeStreamingAsync(request, onChunkProduced, onProgress, ct);

        var sidecar = result.Alignment;
        sidecar.SourceText = text;
        sidecar.Save(sidecarPath);
        return sidecar;
    }

    /// <summary>
    /// The job's backend, built from the model locations in Settings. Missing prerequisites
    /// are reported here, before any model loads, with the path the user needs to fix.
    /// </summary>
    private ITtsBackend EnsureBackend(TtsJobSettings tts)
    {
        var engine = TtsEngines.For(tts.Backend);
        string key = engine.CacheKey(_settings);
        if (_backend is not null && _backendKey == key) return _backend;

        _backend?.Dispose();
        _backend = null;
        _backendKey = null;

        string? missing = TtsPrerequisites.Describe(engine.Kind, _settings, tts);
        if (missing is not null)
            throw new InvalidOperationException(missing);

        _backend = engine.CreateBackend(_settings);
        _backendKey = key;
        return _backend;
    }

    /// <summary>Drops the cached backend, e.g. after the model locations change in Settings.</summary>
    public void Invalidate()
    {
        _backend?.Dispose();
        _backend = null;
        _backendKey = null;
    }

    public void Dispose() => Invalidate();
}

/// <summary>
/// The "can this backend run right now" checks, shared by the job runner (fail early with a
/// path, not deep inside model loading), the New TTS Job dialog (explain a disabled Start
/// button) and the Settings → TTS tab (status line per backend).
/// </summary>
internal static class TtsPrerequisites
{
    /// <summary>
    /// Null when everything the backend needs is on disk; otherwise what is missing and where
    /// it was looked for. The on-disk part is ModelManagerService's own check — the same one
    /// the Settings rows show — so the dialog, the runner and Settings cannot disagree; only
    /// the job-specific voice/language checks are layered on here.
    /// </summary>
    public static string? Describe(TtsBackendKind kind, SettingsService s, TtsJobSettings? job = null)
    {
        var engine = TtsEngines.For(kind);
        foreach (var set in engine.RequiredSets)
        {
            var missing = set.MissingFiles(s);
            if (missing.Count > 0)
                return $"{set.Name} incomplete in {set.Dir(s)}: missing {string.Join(", ", missing)}. See Settings → Text-to-Speech.";
        }
        return job is null ? null : engine.DescribeJobIssue(s, job);
    }
}
