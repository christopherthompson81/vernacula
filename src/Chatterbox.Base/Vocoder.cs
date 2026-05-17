using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// Which cond-decoder layout was found in the model directory. The merged
/// Loop graph (Phase 2) is preferred, the 3-graph split (Phase 1) is the
/// next option, and the original monolithic conditional_decoder.onnx
/// (pre-perf-work) is the last fallback.
/// </summary>
public enum VocoderMode
{
    /// <summary>conditional_decoder_loop.onnx — single graph with embedded ONNX Loop CFM solve. Fastest single-Run path.</summary>
    Merged,
    /// <summary>flow_encoder.onnx + cfm_estimator.onnx + mel2wav.onnx — CFM solve loop orchestrated in C#.</summary>
    Split,
    /// <summary>conditional_decoder.onnx — pre-perf-work, full pipeline in one graph (slow).</summary>
    Monolithic,
}

/// <summary>
/// Wraps the cond-decoder side of the pipeline. Auto-detects layout in
/// the supplied directory at construction; subsequent
/// <see cref="Synthesize"/> calls dispatch to the appropriate path.
///
/// Not thread-safe — matches the underlying ORT <see cref="InferenceSession"/>
/// semantics. Construct one instance per concurrent caller.
/// </summary>
public sealed class Vocoder : IDisposable
{
    private readonly InferenceSession? _merged;
    private readonly InferenceSession? _flowEnc;
    private readonly InferenceSession? _cfmEst;
    private readonly InferenceSession? _m2w;
    private readonly InferenceSession? _mono;

    public VocoderMode Mode { get; }

    /// <summary>Auto-detect the available cond-decoder layout in <paramref name="onnxDir"/> and load.</summary>
    /// <param name="onLoad">Optional callback fired once per session loaded
    /// (1× for Merged/Monolithic, 3× for Split).</param>
    public Vocoder(string onnxDir, ExecutionProvider ep, SessionLoadObserver? onLoad = null)
    {
        bool mergedAvail = File.Exists(Path.Combine(onnxDir, "conditional_decoder_loop.onnx"));
        bool splitAvail = File.Exists(Path.Combine(onnxDir, "flow_encoder.onnx"))
                       && File.Exists(Path.Combine(onnxDir, "cfm_estimator.onnx"))
                       && File.Exists(Path.Combine(onnxDir, "mel2wav.onnx"));

        if (mergedAvail)
        {
            Mode = VocoderMode.Merged;
            _merged = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "conditional_decoder_loop.onnx"), ep, onLoad);
        }
        else if (splitAvail)
        {
            Mode = VocoderMode.Split;
            _flowEnc = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "flow_encoder.onnx"), ep, onLoad);
            _cfmEst = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "cfm_estimator.onnx"), ep, onLoad);
            _m2w = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "mel2wav.onnx"), ep, onLoad);
        }
        else
        {
            Mode = VocoderMode.Monolithic;
            _mono = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "conditional_decoder.onnx"), ep, onLoad);
        }
    }

    /// <summary>
    /// speech_tokens → 24 kHz mono float32 waveform. Dispatches to the
    /// mode picked at construction.
    /// </summary>
    public float[] Synthesize(long[] speechTokens, DenseTensor<float> speakerEmbeddings, DenseTensor<float> speakerFeatures)
    {
        var speechTokT = new DenseTensor<long>(speechTokens, [1, speechTokens.Length]);

        return Mode switch
        {
            VocoderMode.Merged => RunMerged(speechTokT, speakerEmbeddings, speakerFeatures),
            VocoderMode.Split => RunSplit(speechTokT, speakerEmbeddings, speakerFeatures),
            VocoderMode.Monolithic => RunMonolithic(speechTokT, speakerEmbeddings, speakerFeatures),
            _ => throw new InvalidOperationException($"Unknown VocoderMode: {Mode}"),
        };
    }

    private float[] RunMerged(DenseTensor<long> speechTokT, DenseTensor<float> spkEmbT, DenseTensor<float> spkFeatT)
    {
        using var loopOut = _merged!.Run([
            NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
            NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
        ]);
        return loopOut.First().AsTensor<float>().ToArray();
    }

    private float[] RunMonolithic(DenseTensor<long> speechTokT, DenseTensor<float> spkEmbT, DenseTensor<float> spkFeatT)
    {
        using var decOut = _mono!.Run([
            NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
            NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
        ]);
        return decOut.First().AsTensor<float>().ToArray();
    }

    /// <summary>
    /// 3-graph split path: run flow_encoder, drive the CFM solve loop
    /// (10 cosine-scheduled Euler steps with CFG, one cfm_estimator.onnx
    /// call per step), trim the prompt prefix from the final mel, then
    /// run mel2wav. Mirrors <c>parity_split_pipeline</c> in
    /// <c>scripts/chatterbox_export/test_chatterbox_parity.py</c>.
    /// </summary>
    private float[] RunSplit(DenseTensor<long> speechTokT, DenseTensor<float> spkEmbT, DenseTensor<float> spkFeatT)
    {
        const int MelBins = ChatterboxConstants.MelBins;
        const int PromptLen = ChatterboxConstants.PromptLen;
        const int CfmSteps = ChatterboxConstants.CfmSteps;
        const float CfgRate = ChatterboxConstants.CfgRate;

        // 1) flow_encoder: speech_tokens → mu, mel_mask, embedding, cond, z
        using var encOut = _flowEnc!.Run([
            NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokT),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings", spkEmbT),
            NamedOnnxValue.CreateFromTensor("speaker_features", spkFeatT),
        ]);
        var encList = encOut.ToList();
        var muT = encList[0].AsTensor<float>();       // [1, 80, T_mel]
        var maskT = encList[1].AsTensor<float>();     // [1, 1,  T_mel]
        var embedT = encList[2].AsTensor<float>();    // [1, 80]
        var condT = encList[3].AsTensor<float>();     // [1, 80, T_mel]
        var zT = encList[4].AsTensor<float>();        // [1, 80, T_mel]
        int tMel = muT.Dimensions[2];

        float[] mu = muT.ToArray();
        float[] mask = maskT.ToArray();
        float[] embed = embedT.ToArray();
        float[] cond = condT.ToArray();
        float[] x = zT.ToArray();    // CFM state, evolves each step

        // 2) CFM solve loop. Cosine-scheduled t_span; 10 Euler steps.
        var tSpan = new float[CfmSteps + 1];
        for (int i = 0; i <= CfmSteps; i++)
        {
            float linear = i / (float)CfmSteps;
            tSpan[i] = 1.0f - MathF.Cos(linear * 0.5f * MathF.PI);
        }
        float t = tSpan[0];
        float dt = tSpan[1] - tSpan[0];
        int muLen = mu.Length;
        int embedLen = embed.Length;
        for (int step = 1; step <= CfmSteps; step++)
        {
            var xIn = CatBatch(x, x);
            var maskIn = CatBatch(mask, mask);
            var muIn = CatBatch(mu, new float[muLen]);
            var condIn = CatBatch(cond, new float[cond.Length]);
            var spksIn = CatBatch(embed, new float[embedLen]);
            var tIn = new float[] { t, t };

            using var estOut = _cfmEst!.Run([
                NamedOnnxValue.CreateFromTensor("x_in",    new DenseTensor<float>(xIn, [2, MelBins, tMel])),
                NamedOnnxValue.CreateFromTensor("mask_in", new DenseTensor<float>(maskIn, [2, 1, tMel])),
                NamedOnnxValue.CreateFromTensor("mu_in",   new DenseTensor<float>(muIn, [2, MelBins, tMel])),
                NamedOnnxValue.CreateFromTensor("t_in",    new DenseTensor<float>(tIn, [2])),
                NamedOnnxValue.CreateFromTensor("spks_in", new DenseTensor<float>(spksIn, [2, MelBins])),
                NamedOnnxValue.CreateFromTensor("cond_in", new DenseTensor<float>(condIn, [2, MelBins, tMel])),
            ]);
            var dphi = estOut.First().AsTensor<float>().ToArray();  // [2, 80, T_mel]
            int half = dphi.Length / 2;
            float a = 1.0f + CfgRate;
            for (int i = 0; i < half; i++)
            {
                float c = dphi[i];
                float u = dphi[half + i];
                x[i] = x[i] + dt * (a * c - CfgRate * u);
            }
            t = tSpan[step];
            if (step < CfmSteps) dt = tSpan[step + 1] - t;
        }

        // 3) Trim mel: drop the prompt prefix (first PromptLen mel frames).
        int tTrim = tMel - PromptLen;
        var feat = new float[MelBins * tTrim];
        for (int c = 0; c < MelBins; c++)
            Array.Copy(x, c * tMel + PromptLen, feat, c * tTrim, tTrim);

        // 4) mel2wav: trimmed mel → waveform
        using var m2wOut = _m2w!.Run([
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(feat, [1, MelBins, tTrim])),
        ]);
        return m2wOut.First().AsTensor<float>().ToArray();
    }

    /// <summary>Concat two flat tensors of identical size along implicit dim 0.</summary>
    private static float[] CatBatch(float[] a, float[] b)
    {
        var r = new float[a.Length + b.Length];
        Array.Copy(a, 0, r, 0, a.Length);
        Array.Copy(b, 0, r, a.Length, b.Length);
        return r;
    }

    public void Dispose()
    {
        _merged?.Dispose();
        _flowEnc?.Dispose();
        _cfmEst?.Dispose();
        _m2w?.Dispose();
        _mono?.Dispose();
    }
}
