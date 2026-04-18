using Vernacula.Base;
using Vernacula.Base.Models;

namespace Vernacula.App.Services;

/// <summary>
/// Coordinates language identification for a job: picks a representative
/// clip from the VAD output, runs it through <see cref="VoxLinguaLid"/>,
/// and (optionally) escalates to a longer sample when the first pass is
/// ambiguous.
///
/// <para>
/// This service is the advisory layer — it never overrides user choices.
/// Callers take the returned <see cref="LidResult"/> and decide whether to
/// pre-fill the force-language picker, prompt the user to switch ASR
/// backends, or ignore the result.
/// </para>
///
/// <para>
/// Thread-safety: the underlying <see cref="VoxLinguaLid"/> session is
/// safe to call from multiple threads, but instances of this service
/// hold a lazily-constructed one. Construct this service once per app
/// lifetime and share; don't wrap each job in a new instance.
/// </para>
/// </summary>
internal sealed class LangIdService : IDisposable
{
    private readonly SettingsService _settings;
    private VoxLinguaLid? _lid;
    private readonly object _lock = new();
    private bool _disposed;

    public LangIdService(SettingsService settings) => _settings = settings;

    /// <summary>
    /// True when LID is enabled in settings and the model assets are
    /// present on disk. Callers should gate their pre-ASR LID step on
    /// this — otherwise fall through to whatever the user set manually.
    /// </summary>
    public bool IsAvailable
    {
        get
        {
            if (!_settings.Current.LidEnabled) return false;
            string dir = _settings.GetVoxLinguaModelsDir();
            return File.Exists(Path.Combine(dir, Config.VoxLinguaModelFile))
                && File.Exists(Path.Combine(dir, Config.VoxLinguaLangMapFile));
        }
    }

    /// <summary>
    /// Classify the audio of a job.
    ///
    /// <para>
    /// Picks the longest VAD segment, clamps to
    /// <see cref="Config.VoxLinguaDefaultClipSeconds"/>, and runs LID.
    /// If the result is ambiguous AND the longest VAD segment itself is
    /// at least <see cref="Config.VoxLinguaEscalationClipSeconds"/> long,
    /// re-runs on the longer window. The better (or longer) result is
    /// returned.
    /// </para>
    ///
    /// <para>
    /// Returns null if LID is disabled, assets are missing, or VAD
    /// produced no segments of at least 1 second.
    /// </para>
    /// </summary>
    public LidResult? DetectLanguage(
        float[] audioMono16k,
        IReadOnlyList<(double startSec, double endSec)> vadSegments)
    {
        if (!IsAvailable) return null;

        var (segStart, segEnd) = PickLongestSegment(vadSegments);
        double segDuration = segEnd - segStart;
        if (segDuration < 1.0) return null;

        var lid = GetOrCreateLid();

        // First pass at the default clip duration (15 s).
        int defaultSamples = Config.VoxLinguaDefaultClipSeconds * VoxLinguaLid.SampleRate;
        float[] initialClip = SliceCentered(audioMono16k, segStart, segEnd, defaultSamples);
        if (initialClip.Length < VoxLinguaLid.SampleRate) return null; // < 1 s of audio
        var first = lid.Classify(initialClip);

        // Escalation: if ambiguous and we have at least VoxLinguaEscalationClipSeconds
        // of audio in the same VAD segment, re-run on the longer window. The longer
        // clip often resolves confusable-family calls (ru/uk/be, de/lb/da) per Phase 6.
        if (first.IsAmbiguous
            && segDuration >= Config.VoxLinguaEscalationClipSeconds)
        {
            int longSamples = Config.VoxLinguaEscalationClipSeconds * VoxLinguaLid.SampleRate;
            float[] longClip = SliceCentered(audioMono16k, segStart, segEnd, longSamples);
            if (longClip.Length >= defaultSamples)
            {
                var escalated = lid.Classify(longClip);
                // Prefer the escalated result — longer context is what we escalated for.
                return escalated;
            }
        }

        return first;
    }

    public void Dispose()
    {
        if (_disposed) return;
        lock (_lock)
        {
            _lid?.Dispose();
            _lid = null;
            _disposed = true;
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private VoxLinguaLid GetOrCreateLid()
    {
        lock (_lock)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LangIdService));
            return _lid ??= new VoxLinguaLid(
                _settings.GetVoxLinguaModelsDir(),
                ExecutionProvider.Auto);
        }
    }

    /// <summary>Pick the longest VAD segment; empty input → (0, 0).</summary>
    private static (double start, double end) PickLongestSegment(
        IReadOnlyList<(double startSec, double endSec)> segs)
    {
        if (segs is null || segs.Count == 0) return (0, 0);
        double bestDur = -1;
        (double start, double end) best = (0, 0);
        foreach (var (s, e) in segs)
        {
            double dur = e - s;
            if (dur > bestDur) { bestDur = dur; best = (s, e); }
        }
        return best;
    }

    /// <summary>
    /// Extract up to <paramref name="targetSamples"/> samples from
    /// <paramref name="audio"/> centred on the VAD segment. Falls back to
    /// whatever the segment contains if shorter than the target.
    /// </summary>
    private static float[] SliceCentered(
        float[] audio, double segStart, double segEnd, int targetSamples)
    {
        int sr = VoxLinguaLid.SampleRate;
        int segStartSample = Math.Clamp((int)Math.Round(segStart * sr), 0, audio.Length);
        int segEndSample   = Math.Clamp((int)Math.Round(segEnd   * sr), 0, audio.Length);
        int segLen = segEndSample - segStartSample;
        if (segLen <= 0) return [];

        int take = Math.Min(targetSamples, segLen);
        // Centre the window within the VAD segment so we sample the
        // middle (the endpoints are more likely to contain onset/offset
        // artefacts that the model wasn't trained on).
        int offset = segStartSample + Math.Max(0, (segLen - take) / 2);
        var slice = new float[take];
        Array.Copy(audio, offset, slice, 0, take);
        return slice;
    }
}
