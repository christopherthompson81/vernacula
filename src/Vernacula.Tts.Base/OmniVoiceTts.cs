using Vernacula.Base.Models;
using Vernacula.Tts.Base.Tokenization;

namespace Vernacula.Tts.Base;

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

    /// <summary>Diagnostic timing from the most recent <see cref="RunDiffusion"/> call:
    /// total time inside the ONNX transformer vs. total host-side scoring/bookkeeping (ms).</summary>
    public double LastTransformerMs { get; private set; }
    public double LastHostMs { get; private set; }

    public OmniVoiceTts(string onnxDir, string tokenizerJsonPath, ExecutionProvider ep,
                        SessionLoadObserver? onLoad = null, string? transformerFile = null,
                        string? diffPath = null)
    {
        _graphs = new OmniVoice(onnxDir, ep, onLoad, transformerFile, diffPath);
        _tok = new Qwen3Tokenizer(tokenizerJsonPath);
        _prep = new OmniVoiceTextPrep(_tok);
    }

    /// <summary>Generation knobs (greedy regime). Defaults match OmniVoice's upstream config
    /// except the temperatures, which are 0 for determinism.</summary>
    public sealed record GenConfig(
        int NumStep = 32, float GuidanceScale = 2.0f, float TShift = 0.1f,
        float LayerPenaltyFactor = 5.0f, bool Denoise = true);

    /// <summary>Encode a 24 kHz mono reference waveform into voice-clone codes [8, Tref].
    /// Mirrors create_voice_clone_prompt's tokenization-relevant steps: RMS-boost a quiet clip,
    /// (when <paramref name="preprocessPrompt"/>) remove silence so the token count reflects
    /// actual speech — critical, since that count drives the duration estimate — then clip to a
    /// multiple of the codec hop.</summary>
    public long[,] EncodeReference(float[] refWav24k, bool preprocessPrompt = true)
    {
        float rms = Rms(refWav24k);
        var wav = refWav24k;
        if (rms > 0 && rms < 0.1f)
        {
            wav = new float[refWav24k.Length];
            float g = 0.1f / rms;
            for (int i = 0; i < wav.Length; i++) wav[i] = refWav24k[i] * g;
        }
        // Silence removal BEFORE encoding (Python does this when preprocess_prompt): edge +
        // mid silence inflate the ref token count → an over-long duration estimate → stretched
        // output. Params match remove_silence(mid=200, lead=100, trail=200) for the reference.
        if (preprocessPrompt)
            wav = OmniVoiceAudioPost.RemoveSilence(wav, SampleRate, midSilMs: 200, leadSilMs: 100, trailSilMs: 200);
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

        // Classifier-free guidance needs two forwards per step: the conditional pass over the
        // full sequence, and the unconditional pass over ONLY the target tokens. Upstream
        // batches them as [2,8,condLen] by padding the uncond row to condLen with block-diagonal
        // attention — but that pad is discarded, so we instead run two B=1 passes at their
        // natural lengths (condLen and T), saving the wasted uncond compute.
        // Two B=1 passes per step: conditional over the full sequence, unconditional over only
        // the target tokens. We use the plain (non-IO-bound) RunTransformer: binding CPU-pinned
        // OrtValues to a CUDA session produced corrupt output, and IO binding only saved ~90 ms
        // anyway. See docs/omnivoice_onnx_investigation.md.
        var condIds = new long[C * condLen];
        Array.Fill(condIds, (long)MaskId);
        for (int cb = 0; cb < C; cb++)
            for (int p = 0; p < condLen; p++)
                condIds[cb * condLen + p] = cond.InputIds[cb, p];
        var condAmask = new bool[condLen];
        for (int i = 0; i < condLen; i++) condAmask[i] = cond.AudioMask[i];
        var condAttn = new bool[condLen * condLen];
        Array.Fill(condAttn, true);

        var uncondIds = new long[C * T];
        Array.Fill(uncondIds, (long)MaskId);
        var uncondAmask = new bool[T];
        Array.Fill(uncondAmask, true);
        var uncondAttn = new bool[T * T];
        Array.Fill(uncondAttn, true);

        var tokens = new long[C, T];
        for (int cb = 0; cb < C; cb++) for (int t = 0; t < T; t++) tokens[cb, t] = MaskId;

        var timesteps = TimeSteps(cfg.NumStep, cfg.TShift);
        var schedule = Schedule(T * C, cfg.NumStep, timesteps);

        var pred = new long[C, T];
        var score = new double[C, T];

        var swTf = new System.Diagnostics.Stopwatch();
        var swHost = new System.Diagnostics.Stopwatch();
        LastTransformerMs = 0; LastHostMs = 0;

        for (int step = 0; step < cfg.NumStep; step++)
        {
            swTf.Restart();
            float[] cLogits = _graphs.RunTransformer(condIds, condAmask, condAttn, 1, condLen);
            float[] uLogits = _graphs.RunTransformer(uncondIds, uncondAmask, uncondAttn, 1, T);
            swTf.Stop(); LastTransformerMs += swTf.Elapsed.TotalMilliseconds;
            swHost.Restart();

            // Per-(codebook,position) CFG scoring is the host hot path (each does 3 softmaxes
            // over the 1025-way vocab). Positions are independent -> parallelise across them.
            float guidance = cfg.GuidanceScale, penalty = cfg.LayerPenaltyFactor;
            Parallel.For(0, C * T, idx =>
            {
                int cb = idx / T, t = idx % T;
                int cOff = (cb * condLen + (targetStart + t)) * V;
                int uOff = (cb * T + t) * V;
                var (tok, conf) = ScoreCfg(cLogits, cOff, uLogits, uOff, guidance);
                pred[cb, t] = tok;
                score[cb, t] = tokens[cb, t] != MaskId
                    ? double.NegativeInfinity            // already committed
                    : conf - cb * penalty;
            });

            int k = schedule[step];
            if (k > 0)
            {
                foreach (var (cb, t) in TopK(score, T, k))
                    tokens[cb, t] = pred[cb, t];

                // Write the full token field back into both passes for the next step.
                for (int cb = 0; cb < C; cb++)
                    for (int t = 0; t < T; t++)
                    {
                        condIds[cb * condLen + (targetStart + t)] = tokens[cb, t];
                        uncondIds[cb * T + t] = tokens[cb, t];
                    }
            }
            swHost.Stop(); LastHostMs += swHost.Elapsed.TotalMilliseconds;
        }
        return tokens;
    }

    /// <summary>Classifier-free-guidance token + confidence for one (codebook, position).
    /// Mirrors _predict_tokens_with_scoring in the greedy (class_temperature=0) branch.</summary>
    private static (long token, double confidence) ScoreCfg(
        float[] cLogits, int cOff, float[] uLogits, int uOff, float guidance)
    {
        Span<double> cl = stackalloc double[V];
        Span<double> ul = stackalloc double[V];
        LogSoftmax(cLogits, cOff, cl);
        LogSoftmax(uLogits, uOff, ul);

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
