using Vernacula.Base.Models;
using Chatterbox.Base.Tokenization;

namespace Chatterbox.Base;

/// <summary>
/// High-level OmniVoice TTS: ties the Qwen3 tokenizer, duration estimate, text prep, the
/// three ONNX graphs, and the iterative-unmasking diffusion loop into one Speak() call.
/// This is the C# port of OmniVoice.generate / _generate_iterative for batch size 1 in the
/// deterministic greedy regime (position/class temperature = 0), validated to sound good and
/// to give exact token-field parity with the Python ONNX pipeline (OmniVoiceLoopParityTests).
///
/// The host loop owns: the CFG cond+uncond batch, the timestep/unmask schedule, classifier-
/// free-guidance mixing, the codebook layer penalty, greedy top-k position selection, and the
/// scatter of chosen tokens. The transformer (embeds+Qwen3+heads), codec encode, and codec
/// decode are the ONNX graphs in <see cref="OmniVoice"/>. Not thread-safe.
/// </summary>
public sealed class OmniVoiceTts : IDisposable
{
    public const int SampleRate = OmniVoice.SampleRate;

    private const int C = OmniVoice.NumCodebooks;        // 8
    private const int V = OmniVoice.AudioVocabSize;      // 1025
    private const int MaskId = OmniVoice.AudioMaskId;    // 1024

    private readonly OmniVoice _graphs;
    private readonly Qwen3Tokenizer _tok;
    private readonly OmniVoiceTextPrep _prep;

    public OmniVoiceTts(string onnxDir, string tokenizerJsonPath, ExecutionProvider ep,
                        SessionLoadObserver? onLoad = null)
    {
        _graphs = new OmniVoice(onnxDir, ep, onLoad);
        _tok = new Qwen3Tokenizer(tokenizerJsonPath);
        _prep = new OmniVoiceTextPrep(_tok);
    }

    /// <summary>Generation knobs (greedy regime). Defaults match OmniVoice's upstream config
    /// except the temperatures, which are 0 for determinism.</summary>
    public sealed record GenConfig(
        int NumStep = 32, float GuidanceScale = 2.0f, float TShift = 0.1f,
        float LayerPenaltyFactor = 5.0f, bool Denoise = true);

    /// <summary>Encode a 24 kHz mono reference waveform into voice-clone codes [8, Tref].
    /// The input is RMS-normalised and clipped to a multiple of the codec hop, mirroring
    /// create_voice_clone_prompt's tokenization-relevant steps.</summary>
    public long[,] EncodeReference(float[] refWav24k)
    {
        float rms = Rms(refWav24k);
        var wav = refWav24k;
        if (rms > 0 && rms < 0.1f)
        {
            wav = new float[refWav24k.Length];
            float g = 0.1f / rms;
            for (int i = 0; i < wav.Length; i++) wav[i] = refWav24k[i] * g;
        }
        int clip = wav.Length % OmniVoice.HopLength;
        if (clip > 0) wav = wav[..^clip];
        return _graphs.EncodeAudio(wav);
    }

    /// <summary>
    /// Run the greedy diffusion loop and return the generated audio codes [8, T].
    /// <paramref name="refCodes"/> is the voice-clone reference [8, Tref] or null.
    /// </summary>
    public long[,] GenerateTokens(string text, int numTargetTokens, string? refText,
                                  long[,]? refCodes, string? lang, string? instruct, GenConfig cfg)
    {
        var cond = _prep.Prepare(text, numTargetTokens, refText, refCodes, lang, instruct, cfg.Denoise);
        return RunDiffusion(cond, cfg);
    }

    /// <summary>The iterative-unmasking loop over a single prepared conditioning input.</summary>
    public long[,] RunDiffusion(OmniVoiceTextPrep.Prepared cond, GenConfig cfg)
    {
        int condLen = cond.Total, T = cond.TargetLen;
        int targetStart = condLen - T;  // = textLen + refLen

        // CFG batch [2,8,condLen]: row 0 = cond (full), row 1 = uncond (target region only,
        // rest masked). Everything starts at the mask id.
        var ids = new long[2 * C * condLen];
        Array.Fill(ids, MaskId);
        for (int cb = 0; cb < C; cb++)
            for (int p = 0; p < condLen; p++)
                ids[(0 * C + cb) * condLen + p] = cond.InputIds[cb, p];
        // uncond row: target region (all mask) at [0,T); the rest stays mask -> already filled.

        var amask = new bool[2 * condLen];
        for (int p = 0; p < condLen; p++) amask[p] = cond.AudioMask[p];            // cond
        for (int t = 0; t < T; t++) amask[condLen + t] = cond.AudioMask[targetStart + t]; // uncond (all true)

        var attn = new bool[2 * condLen * condLen];
        for (int p = 0; p < condLen; p++)
            for (int q = 0; q < condLen; q++)
                attn[p * condLen + q] = true;                                       // cond: full
        int uBase = condLen * condLen;
        for (int p = 0; p < T; p++)
            for (int q = 0; q < T; q++)
                attn[uBase + p * condLen + q] = true;                               // uncond: [:T,:T]
        for (int p = T; p < condLen; p++)
            attn[uBase + p * condLen + p] = true;                                   // uncond pad: diagonal

        var tokens = new long[C, T];
        for (int cb = 0; cb < C; cb++) for (int t = 0; t < T; t++) tokens[cb, t] = MaskId;

        var timesteps = TimeSteps(cfg.NumStep, cfg.TShift);
        var schedule = Schedule(T * C, cfg.NumStep, timesteps);

        var pred = new long[C, T];
        var score = new double[C, T];

        for (int step = 0; step < cfg.NumStep; step++)
        {
            float[] logits = _graphs.RunTransformer(ids, amask, attn, 2, condLen); // [2,8,condLen,V]

            for (int cb = 0; cb < C; cb++)
                for (int t = 0; t < T; t++)
                {
                    int condPos = targetStart + t;
                    int cOff = ((0 * C + cb) * condLen + condPos) * V;
                    int uOff = ((1 * C + cb) * condLen + t) * V;
                    var (tok, conf) = ScoreCfg(logits, cOff, uOff, cfg.GuidanceScale);
                    pred[cb, t] = tok;
                    double s = conf - cb * cfg.LayerPenaltyFactor;
                    if (tokens[cb, t] != MaskId) s = double.NegativeInfinity; // already committed
                    score[cb, t] = s;
                }

            int k = schedule[step];
            if (k <= 0) continue;
            foreach (var (cb, t) in TopK(score, T, k))
                tokens[cb, t] = pred[cb, t];

            // Write the full token field back into both CFG rows for the next step.
            for (int cb = 0; cb < C; cb++)
                for (int t = 0; t < T; t++)
                {
                    ids[(0 * C + cb) * condLen + (targetStart + t)] = tokens[cb, t];
                    ids[(1 * C + cb) * condLen + t] = tokens[cb, t];
                }
        }
        return tokens;
    }

    /// <summary>Classifier-free-guidance token + confidence for one (codebook, position).
    /// Mirrors _predict_tokens_with_scoring in the greedy (class_temperature=0) branch.</summary>
    private static (long token, double confidence) ScoreCfg(float[] logits, int cOff, int uOff, float guidance)
    {
        Span<double> cl = stackalloc double[V];
        Span<double> ul = stackalloc double[V];
        LogSoftmax(logits, cOff, cl);
        LogSoftmax(logits, uOff, ul);

        // combined = log_softmax(cl + guidance*(cl - ul)); = log_softmax((1+g)*cl - g*ul)
        Span<double> comb = stackalloc double[V];
        for (int v = 0; v < V; v++) comb[v] = (1.0 + guidance) * cl[v] - guidance * ul[v];
        LogSoftmaxInPlace(comb);
        comb[MaskId] = double.NegativeInfinity;

        long best = 0; double bestVal = double.NegativeInfinity;
        for (int v = 0; v < V; v++)
            if (comb[v] > bestVal) { bestVal = comb[v]; best = v; }
        return (best, bestVal);
    }

    private static void LogSoftmax(float[] src, int off, Span<double> dst)
    {
        double max = double.NegativeInfinity;
        for (int v = 0; v < V; v++) { double x = src[off + v]; if (x > max) max = x; }
        double sum = 0;
        for (int v = 0; v < V; v++) sum += Math.Exp(src[off + v] - max);
        double lse = max + Math.Log(sum);
        for (int v = 0; v < V; v++) dst[v] = src[off + v] - lse;
    }

    private static void LogSoftmaxInPlace(Span<double> x)
    {
        double max = double.NegativeInfinity;
        for (int v = 0; v < x.Length; v++) if (x[v] > max) max = x[v];
        double sum = 0;
        for (int v = 0; v < x.Length; v++) sum += Math.Exp(x[v] - max);
        double lse = max + Math.Log(sum);
        for (int v = 0; v < x.Length; v++) x[v] -= lse;
    }

    /// <summary>Top-k (codebook,position) slots by score; ties broken by flat index cb*T+t
    /// ascending (matches the row-major flatten the Python topk operates on).</summary>
    private static List<(int cb, int t)> TopK(double[,] score, int T, int k)
    {
        var all = new List<(double s, int idx)>(C * T);
        for (int cb = 0; cb < C; cb++)
            for (int t = 0; t < T; t++)
                all.Add((score[cb, t], cb * T + t));
        all.Sort((a, b) => a.s != b.s ? b.s.CompareTo(a.s) : a.idx.CompareTo(b.idx));
        var result = new List<(int, int)>(k);
        for (int i = 0; i < k && i < all.Count; i++)
            result.Add((all[i].idx / T, all[i].idx % T));
        return result;
    }

    // _get_time_steps: shifted linspace.
    private static double[] TimeSteps(int numStep, double tShift)
    {
        var ts = new double[numStep + 1];
        for (int i = 0; i <= numStep; i++)
        {
            double lin = (double)i / numStep;
            ts[i] = tShift * lin / (1 + (tShift - 1) * lin);
        }
        return ts;
    }

    // Per-step unmask counts: ceil(total*(t[s+1]-t[s])) clamped to the remaining budget,
    // last step takes the remainder.
    private static int[] Schedule(int totalMask, int numStep, double[] timesteps)
    {
        var sched = new int[numStep];
        int rem = totalMask;
        for (int s = 0; s < numStep; s++)
        {
            int num = s == numStep - 1
                ? rem
                : Math.Min((int)Math.Ceiling(totalMask * (timesteps[s + 1] - timesteps[s])), rem);
            sched[s] = num;
            rem -= num;
        }
        return sched;
    }

    /// <summary>Decode generated codes [8,T] to a 24 kHz waveform (no extra post-processing).</summary>
    public float[] DecodeTokens(long[,] tokens) => _graphs.DecodeAudio(tokens);

    private static float Rms(float[] x)
    {
        double s = 0;
        foreach (var v in x) s += (double)v * v;
        return x.Length == 0 ? 0f : (float)Math.Sqrt(s / x.Length);
    }

    public void Dispose() => _graphs.Dispose();
}
