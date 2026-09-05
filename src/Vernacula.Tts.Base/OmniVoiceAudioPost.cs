using System;
using System.Collections.Generic;

namespace Vernacula.Tts.Base;

/// <summary>
/// C# port of OmniVoice's audio pre/post-processing (omnivoice/utils/audio.py, via pydub):
/// <see cref="RemoveSilence"/> (+ edge trimming) and <see cref="FadeAndPad"/>. Silence detection
/// mirrors pydub: a 10 ms chunk is "silent" when its RMS is below the −50 dBFS threshold
/// (linear ≈ 0.00316 for float audio in [-1,1]). Operates on mono float[].
///
/// Why this matters: the reference clip is silence-trimmed BEFORE encoding in Python, so its
/// token count (which drives the duration estimate) reflects actual speech. Skipping it inflates
/// the target length → stretched/slower output. See docs/omnivoice_ipa_corpus_investigation.md.
/// </summary>
public static class OmniVoiceAudioPost
{
    private const double SilenceThreshDb = -50.0;
    private static readonly double ThreshRms = Math.Pow(10.0, SilenceThreshDb / 20.0); // ≈0.0031623

    private static double ChunkRms(float[] a, int start, int len)
    {
        int end = Math.Min(start + len, a.Length);
        if (end <= start) return 0;
        double s = 0;
        for (int i = start; i < end; i++) s += (double)a[i] * a[i];
        return Math.Sqrt(s / (end - start));
    }

    /// <summary>pydub detect_leading_silence: advance in 10 ms chunks while below threshold;
    /// returns the sample index where content starts.</summary>
    private static int DetectLeadingSilence(float[] a, int sr, int chunkMs = 10)
    {
        int chunk = Math.Max(1, chunkMs * sr / 1000);
        int trim = 0;
        while (trim < a.Length && ChunkRms(a, trim, chunk) < ThreshRms) trim += chunk;
        return Math.Min(trim, a.Length);
    }

    /// <summary>pydub detect_silence over the whole signal → contiguous silent ranges (samples).</summary>
    private static List<(int s, int e)> DetectSilence(float[] a, int sr, int minSilMs, int seekMs = 10)
    {
        int minSil = minSilMs * sr / 1000;
        int seek = Math.Max(1, seekMs * sr / 1000);
        var ranges = new List<(int, int)>();
        if (a.Length < minSil) return ranges;

        var starts = new List<int>();
        int lastStart = a.Length - minSil;
        for (int i = 0; i <= lastStart; i += seek)
            if (ChunkRms(a, i, minSil) <= ThreshRms) starts.Add(i);
        if (starts.Count == 0) return ranges;

        int rangeStart = starts[0], prev = starts[0];
        for (int j = 1; j < starts.Count; j++)
        {
            int si = starts[j];
            bool contiguous = si == prev + seek;
            bool gap = si > prev + minSil;
            if (!contiguous && gap) { ranges.Add((rangeStart, prev + minSil)); rangeStart = si; }
            prev = si;
        }
        ranges.Add((rangeStart, prev + minSil));
        return ranges;
    }

    /// <summary>Complement of the silent ranges → non-silent (speech) ranges.</summary>
    private static List<(int s, int e)> DetectNonsilent(float[] a, int sr, int minSilMs, int seekMs = 10)
    {
        var sil = DetectSilence(a, sr, minSilMs, seekMs);
        var non = new List<(int, int)>();
        if (sil.Count == 0) { if (a.Length > 0) non.Add((0, a.Length)); return non; }
        int cur = 0;
        foreach (var (s, e) in sil)
        {
            if (s > cur) non.Add((cur, s));
            cur = e;
        }
        if (cur < a.Length) non.Add((cur, a.Length));
        return non;
    }

    /// <summary>pydub split_on_silence + reconcatenation, with keep_silence padding and the
    /// midpoint-clamp on overlapping expanded ranges.</summary>
    private static float[] SplitAndConcat(float[] a, int sr, int minSilMs, int keepMs, int seekMs = 10)
    {
        var ranges = DetectNonsilent(a, sr, minSilMs, seekMs);
        if (ranges.Count == 0) return Array.Empty<float>();
        int keep = keepMs * sr / 1000;
        var outR = new List<int[]>();
        foreach (var (s, e) in ranges) outR.Add(new[] { s - keep, e + keep });
        for (int i = 0; i + 1 < outR.Count; i++)
        {
            int lastEnd = outR[i][1], nextStart = outR[i + 1][0];
            if (nextStart < lastEnd)
            {
                int mid = (lastEnd + nextStart) / 2;
                outR[i][1] = mid;
                outR[i + 1][0] = mid;
            }
        }
        var buf = new List<float>(a.Length);
        foreach (var r in outR)
        {
            int s = Math.Max(0, r[0]), e = Math.Min(a.Length, r[1]);
            for (int i = s; i < e; i++) buf.Add(a[i]);
        }
        return buf.ToArray();
    }

    /// <summary>Port of remove_silence: collapse mid-silences longer than <paramref name="midSilMs"/>
    /// (0 to skip) down to that length, then trim edge silences keeping lead/trail ms.</summary>
    public static float[] RemoveSilence(float[] audio, int sr, int midSilMs, int leadSilMs, int trailSilMs)
    {
        if (audio.Length == 0) return audio;
        var a = midSilMs > 0 ? SplitAndConcat(audio, sr, midSilMs, midSilMs) : audio;
        if (a.Length == 0) return a;

        int lead = leadSilMs * sr / 1000, trail = trailSilMs * sr / 1000;
        int start = Math.Max(0, DetectLeadingSilence(a, sr) - lead);
        var rev = (float[])a.Clone();
        Array.Reverse(rev);
        int trailStart = Math.Max(0, DetectLeadingSilence(rev, sr) - trail);
        int end = a.Length - trailStart;
        if (end <= start) return Array.Empty<float>();
        return a[start..end];
    }

    /// <summary>Port of fade_and_pad_audio: linear fade-in/out over <paramref name="fadeSec"/> and
    /// zero-pad <paramref name="padSec"/> on both ends (always applied).</summary>
    public static float[] FadeAndPad(float[] audio, int sr, double padSec = 0.1, double fadeSec = 0.1)
    {
        if (audio.Length == 0) return audio;
        int fade = (int)(fadeSec * sr), pad = (int)(padSec * sr);
        var proc = (float[])audio.Clone();
        int k = Math.Min(fade, proc.Length / 2);
        for (int i = 0; i < k; i++)
        {
            proc[i] *= i / (float)(k - 1 == 0 ? 1 : k - 1);
            proc[^ (i + 1)] *= i / (float)(k - 1 == 0 ? 1 : k - 1);
        }
        if (pad <= 0) return proc;
        var outArr = new float[pad + proc.Length + pad];
        Array.Copy(proc, 0, outArr, pad, proc.Length);
        return outArr;
    }
}
