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
}
