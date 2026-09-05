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

    /// <summary>Scale IN PLACE so the loudest sample sits at <paramref name="peak"/>. Silence is
    /// left alone.</summary>
    public static void Normalize(float[] x, float peak)
    {
        float max = Peak(x);
        if (max < 1e-6f) return;
        float g = peak / max;
        for (int i = 0; i < x.Length; i++) x[i] *= g;
    }

    /// <summary>Loudest absolute sample, or 0 for an empty or silent buffer.</summary>
    public static float Peak(float[] a)
    {
        float m = 0;
        foreach (var v in a) m = Math.Max(m, Math.Abs(v));
        return m;
    }

    /// <summary>
    /// Keeps one document's chunks at consistent loudness. A document is synthesized chunk by
    /// chunk and the pieces are concatenated, so normalising each chunk to the same peak on its own
    /// would pump: a chunk whose loudest moment is a soft syllable would come out as loud as one
    /// containing a shouted word. The first chunk sets the level and the rest keep their loudness
    /// relative to it, which is what the old constant-gain chain did for free.
    ///
    /// A leveler is for ONE document. <see cref="OmniVoiceAudioPost.FinishStoredVoice"/> without
    /// one is the single-utterance case (the CLI, the demo) and normalises outright.
    /// </summary>
    public sealed class StoredVoiceLeveler
    {
        private float _firstPeak;

        /// <summary>The target peak for a chunk whose decoded peak was <paramref name="rawPeak"/>.
        /// The first chunk anchors at 0.5; a later chunk that was twice as loud would want 1.0 and
        /// is capped short of full scale rather than clipped.</summary>
        internal float TargetFor(float rawPeak)
        {
            if (_firstPeak <= 0) _firstPeak = rawPeak;
            if (_firstPeak <= 0) return 0.5f;
            return Math.Min(0.95f, 0.5f * rawPeak / _firstPeak);
        }
    }

    /// <summary>
    /// The finishing chain for a STORED voice — the demo's, and the only correct one for a
    /// reference we did not boost.
    ///
    /// ⚠ PYTHON'S VOLUME STEP UN-BOOSTS A REFERENCE THAT WAS BOOSTED AT ENCODE TIME. The stored
    /// voices never were: they carry the source clip's own level, and the corpus spans rms
    /// 0.0017–0.099, a 58× spread. Applying the un-boost to them undoes something that never
    /// happened, so a quiet source plays quiet — af (rms 0.0148) came out at 15% of level — and at
    /// the very quiet end it drops the whole utterance below the silence threshold, where
    /// <see cref="RemoveSilence"/> deletes it outright.
    ///
    /// Hence: normalise BEFORE silence removal, and again AFTER the fade — the fine-tune emits a
    /// leading transient inside the first 0.1 s, and normalising only before the fade lets that
    /// transient take the headroom which the fade then removes.
    ///
    /// The input array is not modified; the caller keeps its decoded audio.
    /// </summary>
    /// <param name="leveler">Optional, and shared across one document's chunks so they stay at
    /// consistent loudness relative to each other. Null finishes a single utterance on its own.</param>
    public static float[] FinishStoredVoice(float[] audio, int sr, StoredVoiceLeveler? leveler = null)
    {
        var target = leveler?.TargetFor(Peak(audio)) ?? 0.5f;
        audio = (float[])audio.Clone();
        Normalize(audio, 0.5f);
        audio = RemoveSilence(audio, sr, midSilMs: 500, leadSilMs: 100, trailSilMs: 100);
        audio = FadeAndPad(audio, sr);
        Normalize(audio, target);
        return audio;
    }

    /// <summary>
    /// The finishing chain in the order Python's <c>_post_process_audio</c> applies it: remove
    /// silence → volume → fade-in/out + zero-pad. Volume: with a reference (<paramref name="refRms"/>
    /// is its RMS before any boost), un-boost a reference that <see cref="OmniVoiceTts.EncodeReference"/>
    /// boosted for being quiet; without one, peak-normalise to 0.5.
    /// </summary>
    public static float[] Finish(float[] audio, int sr, float? refRms)
    {
        audio = RemoveSilence(audio, sr, midSilMs: 500, leadSilMs: 100, trailSilMs: 100);
        if (refRms is float rr)
        {
            if (rr < 0.1f) { float g = rr / 0.1f; for (int i = 0; i < audio.Length; i++) audio[i] *= g; }
        }
        else
        {
            Normalize(audio, 0.5f);
        }
        return FadeAndPad(audio, sr);
    }
}
