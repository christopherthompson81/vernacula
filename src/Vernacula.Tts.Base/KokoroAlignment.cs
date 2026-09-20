namespace Vernacula.Tts.Base;

/// <summary>
/// Turning Kokoro's per-phoneme-group timings into per-word ones — the half of alignment that is
/// the same on both engines.
///
/// <para>
/// ⚠ THE TWO ENGINES DIFFER IN HOW THE GROUPS ARE OBTAINED AND NOT IN WHAT IS DONE WITH THEM.
/// The ONNX path reads the model's <c>pred_dur</c> and cuts it at the space/pad tokens itself;
/// audio.cpp predicts the same durations inside the session and reports the cut groups through the
/// ABI. Either way what arrives is "group g occupies [start, end) seconds", and the phonemizer has
/// already said which source word each group came from. This is that join, written once so a
/// document highlights identically whichever backend rendered it.
/// </para>
/// </summary>
public static class KokoroAlignment
{
    /// <summary>One phoneme group's span in the rendered audio.</summary>
    public readonly record struct GroupSpan(double StartSeconds, double EndSeconds);

    /// <summary>
    /// One word per whitespace-delimited word of <paramref name="text"/>, in order.
    /// </summary>
    /// <param name="groupSourceWords">
    /// For each group in <paramref name="groups"/>, the index of the source word it came from
    /// (<see cref="KokoroPhonemization.GroupSourceWords"/>). Null, or a count that disagrees with
    /// <paramref name="groups"/>, falls back to an even split — a partial map would put every word
    /// after the gap on the wrong audio, which shows up as a highlight that is quietly one word out
    /// rather than as anything that looks like a bug.
    /// </param>
    public static IReadOnlyList<KokoroWord> WordsFromGroups(
        string text,
        IReadOnlyList<int>? groupSourceWords,
        IReadOnlyList<GroupSpan> groups,
        double totalSeconds)
    {
        var sourceWords = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var words = new List<KokoroWord>(sourceWords.Length);
        if (sourceWords.Length == 0) return words;

        if (groupSourceWords is null || groups.Count != groupSourceWords.Count)
        {
            for (var w = 0; w < sourceWords.Length; w++)
                words.Add(new KokoroWord(sourceWords[w],
                    totalSeconds * w / sourceWords.Length,
                    totalSeconds * (w + 1) / sourceWords.Length));
            return words;
        }

        // Collect each source word's group span. A word's groups are contiguous and in time order,
        // so first start / last end gives its [start, end).
        var hasRun = new bool[sourceWords.Length];
        var starts = new double[sourceWords.Length];
        var ends = new double[sourceWords.Length];
        for (var g = 0; g < groups.Count; g++)
        {
            var src = groupSourceWords[g];
            if (src < 0 || src >= sourceWords.Length) continue;
            if (!hasRun[src]) { starts[src] = groups[g].StartSeconds; hasRun[src] = true; }
            ends[src] = groups[g].EndSeconds;
        }

        // Emit one word per source word — including unpronounceable words that produced no groups
        // (zero-length marker at the running cursor), so the display shows every word and the index
        // stays 1:1 with the source-text whitespace split.
        var cursor = 0.0;
        for (var w = 0; w < sourceWords.Length; w++)
        {
            if (hasRun[w]) { words.Add(new KokoroWord(sourceWords[w], starts[w], ends[w])); cursor = ends[w]; }
            else words.Add(new KokoroWord(sourceWords[w], cursor, cursor));
        }
        return words;
    }
}
