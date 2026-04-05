using System.Collections.Generic;
using System.IO;
using Avalonia.Media;
using Vernacula.Base;

namespace Vernacula.Avalonia.Services;

/// <summary>
/// Loads the Parakeet vocabulary and provides token→text decoding and
/// confidence-to-colour mapping for the transcript editor.
/// </summary>
internal class VocabService
{
    private readonly Dictionary<int, string> _vocab;

    public VocabService(string modelsDir)
    {
        _vocab = LoadVocab(Path.Combine(modelsDir, Config.VocabFile));
    }

    // ── Vocab loading (mirrors Parakeet.GetVocab) ────────────────────────────

    private static Dictionary<int, string> LoadVocab(string vocabPath)
    {
        var vocab = new Dictionary<int, string>();
        if (!File.Exists(vocabPath)) return vocab;

        foreach (var line in File.ReadLines(vocabPath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int lastSpace = line.LastIndexOf(' ');
            if (lastSpace < 0) continue;
            string token = line[..lastSpace].Replace("\u2581", " ");
            if (int.TryParse(line[(lastSpace + 1)..], out int id))
                vocab[id] = token;
        }
        return vocab;
    }

    // ── Token decoding ────────────────────────────────────────────────────────

    public string DecodeTokens(IReadOnlyList<int> tokens)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var t in tokens)
            if (_vocab.TryGetValue(t, out var s)) sb.Append(s);
        return sb.ToString().Trim();
    }

    /// <summary>Returns (text, logprob) pairs for each token.</summary>
    public IReadOnlyList<(string text, float logprob)> GetTokenRuns(
        IReadOnlyList<int> tokens, IReadOnlyList<float> logprobs)
    {
        var runs = new List<(string, float)>(tokens.Count);
        for (int i = 0; i < tokens.Count; i++)
        {
            string text = _vocab.TryGetValue(tokens[i], out var s) ? s : "";
            float  lp   = i < logprobs.Count ? logprobs[i] : 0f;
            runs.Add((text, lp));
        }
        return runs;
    }

    // ── Confidence highlight ──────────────────────────────────────────────────

    /// <summary>
    /// Returns a semi-transparent background highlight colour for a token based on its logprob.
    /// High confidence → fully transparent; low confidence → opaque ConfidenceLowColor tint.
    /// logprob range [-2, 0]; values below -2 are clamped to maximum opacity.
    /// Uses a quadratic curve so only genuinely uncertain tokens get a visible tint.
    /// </summary>
    public static Color GetConfidenceHighlight(float logprob, Color highlightColor)
    {
        const float minLogprob = -2f;
        float linear = Math.Clamp((logprob - minLogprob) / (-minLogprob), 0f, 1f);
        // t = 1 high confidence, t = 0 low confidence
        float t     = (float)Math.Sqrt(linear);
        byte  alpha = (byte)((1f - t * t) * 200f); // quadratic: more opacity for very low confidence
        return Color.FromArgb(alpha, highlightColor.R, highlightColor.G, highlightColor.B);
    }
}
