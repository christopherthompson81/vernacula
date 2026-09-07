using System.Text.Json;
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
    public static int SampleRateFor(TtsJobSettings tts) => ParseBackend(tts.Backend) switch
    {
        TtsBackendKind.Kokoro    => Kokoro.SampleRate,
        TtsBackendKind.OmniVoice => OmniVoiceIpaTts.SampleRate,
        _                        => ChatterboxConstants.S3GenSr,
    };

    public static TtsBackendKind ParseBackend(string name) =>
        Enum.TryParse<TtsBackendKind>(name, ignoreCase: true, out var kind) ? kind : TtsBackendKind.Chatterbox;

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
        string segmentsDir = SegmentsDirFor(sidecarPath);
        Directory.CreateDirectory(Path.GetDirectoryName(sidecarPath)!);
        // A re-render starts clean: a shorter document must not leave stale paragraph files.
        if (Directory.Exists(segmentsDir)) Directory.Delete(segmentsDir, recursive: true);

        var request = ParseBackend(tts.Backend) switch
        {
            TtsBackendKind.Kokoro    => new TtsRequest(text, wavPath, tts.Voice, tts.Speed, SegmentsDir: segmentsDir),
            TtsBackendKind.OmniVoice => new TtsRequest(text, wavPath, tts.Voice,
                                            Lang: string.IsNullOrWhiteSpace(tts.Language) ? "en" : tts.Language.Trim(),
                                            NumStep: tts.NumStep, SegmentsDir: segmentsDir),
            _                        => new TtsRequest(text, wavPath, tts.Voice, SegmentsDir: segmentsDir),
        };

        var result = await backend.SynthesizeStreamingAsync(request, onChunkProduced, onProgress, ct);

        var sidecar = result.Alignment;
        sidecar.SourceText = text;
        await using (var fs = File.Create(sidecarPath))
            await JsonSerializer.SerializeAsync(fs, sidecar, new JsonSerializerOptions { WriteIndented = false }, ct);
        return sidecar;
    }

    /// <summary>The folder of per-paragraph WAVs beside a job's sidecar: <c>{stem}_segments/</c>.</summary>
    public static string SegmentsDirFor(string sidecarPath) =>
        Path.Combine(Path.GetDirectoryName(sidecarPath)!, Path.GetFileNameWithoutExtension(sidecarPath) + "_segments");

    /// <summary>
    /// The job's backend, built from the model locations in Settings. Missing prerequisites
    /// are reported here, before any model loads, with the path the user needs to fix.
    /// </summary>
    private ITtsBackend EnsureBackend(TtsJobSettings tts)
    {
        var kind = ParseBackend(tts.Backend);
        string key = kind switch
        {
            TtsBackendKind.Kokoro    => $"kokoro|{_settings.GetKokoroModelsDir()}|{_settings.GetPhonemizerDataDir()}",
            TtsBackendKind.OmniVoice => $"omnivoice|{_settings.GetOmniVoiceModelsDir()}|{_settings.Current.OmniVoiceTokenizerJson}"
                                        + $"|{_settings.GetPhonemizerDataDir()}|{_settings.GetOmniVoiceVoiceLibDir()}",
            _                        => $"chatterbox|{_settings.GetChatterboxModelsDir()}",
        };
        if (_backend is not null && _backendKey == key) return _backend;

        _backend?.Dispose();
        _backend = null;
        _backendKey = null;

        string? missing = TtsPrerequisites.Describe(kind, _settings, tts);
        if (missing is not null)
            throw new InvalidOperationException(missing);

        _backend = kind switch
        {
            TtsBackendKind.Kokoro    => new KokoroSynthesisService(_settings.GetKokoroModelsDir(), _settings.GetPhonemizerDataDir()),
            TtsBackendKind.OmniVoice => new OmniVoiceSynthesisService(
                                            _settings.GetOmniVoiceModelsDir(),
                                            File.Exists(_settings.Current.OmniVoiceTokenizerJson) ? _settings.Current.OmniVoiceTokenizerJson : null,
                                            _settings.GetPhonemizerDataDir(),
                                            _settings.GetOmniVoiceVoiceLibDir()),
            _                        => new ChatterboxSynthesisService(
                                            _settings.GetChatterboxModelsDir(),
                                            File.Exists(Path.Combine(_settings.GetChatterboxModelsDir(), "tokenizer.json"))
                                                ? Path.Combine(_settings.GetChatterboxModelsDir(), "tokenizer.json") : null),
        };
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
    /// <summary>The model sets a backend needs on disk.</summary>
    public static ModelManagerService.TtsModelSet[] RequiredSets(TtsBackendKind kind) => kind switch
    {
        TtsBackendKind.Kokoro    => [ModelManagerService.TtsModelSet.Kokoro, ModelManagerService.TtsModelSet.PhonemizerData],
        TtsBackendKind.OmniVoice => [ModelManagerService.TtsModelSet.OmniVoice, ModelManagerService.TtsModelSet.OmniVoiceVoices,
                                     ModelManagerService.TtsModelSet.PhonemizerData],
        _                        => [ModelManagerService.TtsModelSet.Chatterbox],
    };

    /// <summary>
    /// Null when everything the backend needs is on disk; otherwise what is missing and where
    /// it was looked for. The on-disk part is ModelManagerService's own check — the same one
    /// the Settings rows show — so the dialog, the runner and Settings cannot disagree; only
    /// the job-specific voice/language checks are layered on here.
    /// </summary>
    public static string? Describe(TtsBackendKind kind, SettingsService s, TtsJobSettings? job = null)
    {
        foreach (var set in RequiredSets(kind))
        {
            var missing = ModelManagerService.GetMissingTtsFiles(set, s);
            if (missing.Count > 0)
                return $"{SetName(set)} incomplete in {ModelManagerService.TtsModelSetDir(set, s)}: missing {string.Join(", ", missing)}. See Settings → Text-to-Speech.";
        }
        if (job is null) return null;

        switch (kind)
        {
            case TtsBackendKind.Kokoro:
                if (string.IsNullOrWhiteSpace(job.Voice)) return "No Kokoro voice selected.";
                string voicePath = Path.Combine(s.GetKokoroModelsDir(), "voices", job.Voice + ".bin");
                if (!File.Exists(voicePath)) return $"Kokoro voice not found: {voicePath}";
                return null;
            case TtsBackendKind.OmniVoice:
                if (string.IsNullOrWhiteSpace(job.Voice)) return "No OmniVoice voice selected.";
                if (LanguageCatalog.ByCode(job.Language) is null) return $"Unknown language \"{job.Language}\" — pick one from the list.";
                return null;
            default:
                if (!File.Exists(job.Voice))
                    return $"Reference voice clip not found: {(string.IsNullOrWhiteSpace(job.Voice) ? "(not set)" : job.Voice)}";
                return null;
        }
    }

    private static string SetName(ModelManagerService.TtsModelSet set) => set switch
    {
        ModelManagerService.TtsModelSet.Chatterbox      => "Chatterbox bundle",
        ModelManagerService.TtsModelSet.Kokoro          => "Kokoro model",
        ModelManagerService.TtsModelSet.OmniVoice       => "OmniVoice ONNX set",
        ModelManagerService.TtsModelSet.OmniVoiceVoices => "OmniVoice voice library",
        ModelManagerService.TtsModelSet.PhonemizerData  => "Phonemizer data",
        _                                               => set.ToString(),
    };
}
