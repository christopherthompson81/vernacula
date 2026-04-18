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
        return GetOrCreateLid().ClassifyLongestSegment(audioMono16k, vadSegments);
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
}
