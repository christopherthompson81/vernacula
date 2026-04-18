namespace Vernacula.Base.Models;

/// <summary>
/// One candidate language in a LID result, ordered by probability descending.
/// <paramref name="Iso"/> is the ISO 639-1 code when available (or ISO 639-3
/// for languages without a 1-char form — VoxLingua107's long tail).
/// </summary>
public sealed record LidCandidate(int ClassIndex, string Iso, string Name, float Probability);

/// <summary>
/// Output of a single LID classification run.
/// </summary>
public sealed record LidResult(
    LidCandidate Top,
    IReadOnlyList<LidCandidate> TopK,
    bool IsAmbiguous,
    float[] Embedding,
    float ClipDurationSeconds)
{
    /// <summary>Convenience: <see cref="Top"/>'s ISO code.</summary>
    public string Iso => Top.Iso;

    /// <summary>Convenience: <see cref="Top"/>'s probability.</summary>
    public float TopProbability => Top.Probability;

    /// <summary>
    /// Short human-readable summary, either "English (93 %)" or
    /// "ambiguous: English 48 %, Dutch 31 %, Afrikaans 14 %" when the
    /// top-1 fell below the ambiguity threshold.
    /// </summary>
    public string FormatSummary(int maxAmbiguous = 3) =>
        IsAmbiguous
            ? "ambiguous: " + string.Join(", ", TopK.Take(maxAmbiguous)
                .Select(c => $"{c.Name} {c.Probability:P0}"))
            : $"{Top.Name} ({Top.Probability:P0})";
}
