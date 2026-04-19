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
/// The ONNX session is created fresh for each job and disposed immediately
/// after, so it never holds GPU memory concurrently with the ASR model.
/// </para>
/// </summary>
internal sealed class LangIdService(SettingsService settings)
{
    /// <summary>
    /// True when LID is enabled in settings and the model assets are
    /// present on disk. Callers should gate their pre-ASR LID step on
    /// this — otherwise fall through to whatever the user set manually.
    /// </summary>
    public bool IsAvailable
    {
        get
        {
            if (!settings.Current.LidEnabled) return false;
            string dir = settings.GetVoxLinguaModelsDir();
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
        using var lid = new VoxLinguaLid(settings.GetVoxLinguaModelsDir());
        return lid.ClassifyLongestSegment(audioMono16k, vadSegments);
    }

    /// <summary>
    /// Run LID on every segment whose duration is at least
    /// <paramref name="minSegmentSeconds"/>. Opens one VoxLingua107 session
    /// for the whole batch (the model is the expensive part — per-call
    /// inference is fast). Returns a list parallel to <paramref name="segs"/>;
    /// indices below the minimum, or shorter than 1 s of audio after slicing,
    /// receive a null entry which the caller is expected to fill from the
    /// file-level language.
    ///
    /// <para>
    /// Each clip is centred inside its segment and clamped to
    /// <see cref="Config.VoxLinguaDefaultClipSeconds"/> (matches the
    /// file-level path so single-segment results stay comparable). Caller
    /// reports progress between segments.
    /// </para>
    /// </summary>
    public IReadOnlyList<LidResult?> ClassifyEachSegment(
        float[] audioMono16k,
        IReadOnlyList<(double startSec, double endSec)> segs,
        double minSegmentSeconds = 1.0,
        Action<int, int>? onProgress = null,
        CancellationToken ct = default)
    {
        var results = new LidResult?[segs.Count];
        if (!IsAvailable || segs.Count == 0) return results;

        using var lid = new VoxLinguaLid(settings.GetVoxLinguaModelsDir());
        int clipSeconds = Config.VoxLinguaDefaultClipSeconds;
        int sampleRate  = VoxLinguaLid.SampleRate;
        int targetSamples = clipSeconds * sampleRate;

        for (int i = 0; i < segs.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var (start, end) = segs[i];
            double dur = end - start;
            if (dur < minSegmentSeconds)
            {
                onProgress?.Invoke(i + 1, segs.Count);
                continue;
            }

            int segStart = Math.Clamp((int)Math.Round(start * sampleRate), 0, audioMono16k.Length);
            int segEnd   = Math.Clamp((int)Math.Round(end   * sampleRate), 0, audioMono16k.Length);
            int segLen   = segEnd - segStart;
            if (segLen < sampleRate)
            {
                onProgress?.Invoke(i + 1, segs.Count);
                continue;
            }

            int take = Math.Min(targetSamples, segLen);
            int offset = segStart + Math.Max(0, (segLen - take) / 2);
            try
            {
                results[i] = lid.Classify(audioMono16k.AsSpan(offset, take));
            }
            catch (ArgumentException)
            {
                // Clip slipped below the 1 s floor due to rounding; skip and
                // let the caller fall back to the file-level language.
            }
            onProgress?.Invoke(i + 1, segs.Count);
        }

        return results;
    }
}
