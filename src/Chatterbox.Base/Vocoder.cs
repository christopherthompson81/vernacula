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

    /// <summary>
    /// True when <see cref="SynthesizeBatch"/> has a working path at B&gt;1
    /// — either the merged graph carries the Path B (Run 10) metadata flag
    /// <c>supports_batched=true</c>, or all three split graphs are loaded
    /// as a fallback. False in monolithic mode or when neither route is
    /// available.
    /// </summary>
    public bool SupportsBatched => _mergedSupportsBatched
                                || (_flowEnc is not null && _cfmEst is not null && _m2w is not null);

    /// <summary>
    /// True when the loaded merged graph was exported with Path B fixes
    /// (Run 10) and accepts B&gt;1 via dynamic CFG-doubling. Detected from
    /// the model's <c>supports_batched</c> metadata prop. Older bundles
    /// without this prop fall through to the split-graph path in
    /// <see cref="SynthesizeBatch"/>.
    /// </summary>
    private readonly bool _mergedSupportsBatched;

    /// <summary>Auto-detect the available cond-decoder layout in <paramref name="onnxDir"/> and load.</summary>
    /// <param name="onLoad">Optional callback fired once per session loaded
    /// (1× for Merged-only or Monolithic; 3× for Split-only; 4× for
    /// Merged + split fallback available).</param>
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
            // Run 10 / Path B: merge_cond_decoder_loop.py now stamps a
            // `supports_batched=true` metadata prop when the graph supports
            // B>1. Older bundles don't carry it and fall back to split.
            _mergedSupportsBatched =
                _merged.ModelMetadata.CustomMetadataMap.TryGetValue("supports_batched", out var sb)
                && string.Equals(sb, "true", StringComparison.OrdinalIgnoreCase);
            // Load split graphs as a fallback for B>1 when the merged graph
            // doesn't support it. Skipping when merged is already
            // batched-capable saves the ~1.1 GB VRAM the split graphs cost
            // (per Run 6c sidebar) — that was the whole reason Path B was
            // worth doing.
            if (splitAvail && !_mergedSupportsBatched)
            {
                _flowEnc = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "flow_encoder.onnx"), ep, onLoad);
                _cfmEst = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "cfm_estimator.onnx"), ep, onLoad);
                _m2w = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "mel2wav.onnx"), ep, onLoad);
            }
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

    /// <summary>
    /// Run 10 / Path B: one merged-graph call at B>1. The graph's outer
    /// CFG-doubling, body t_in Tile, and Loop's v_initial Expand were
    /// rewritten in <c>merge_cond_decoder_loop.py</c> to scale with B,
    /// so a single .Run produces B waveforms in input order. Avoids the
    /// host-roundtrip per CFM iter that bottlenecks the split path.
    /// </summary>
    private float[][] RunMergedBatched(
        IReadOnlyList<long[]> speechTokensPerChunk,
        DenseTensor<float> speakerEmbeddings,
        DenseTensor<float> speakerFeatures,
        int B, int tTok)
    {
        // Pack speech_tokens [B, tTok].
        var speechTokensB = new long[B * tTok];
        for (int b = 0; b < B; b++)
            Array.Copy(speechTokensPerChunk[b], 0, speechTokensB, b * tTok, tTok);

        // Replicate speaker_embeddings + speaker_features across B (one voice).
        var spkEmbArr = speakerEmbeddings.ToArray();
        int spkDim = speakerEmbeddings.Dimensions[1];
        var spkEmbB = new float[B * spkDim];
        for (int b = 0; b < B; b++) Array.Copy(spkEmbArr, 0, spkEmbB, b * spkDim, spkDim);

        var spkFeatArr = speakerFeatures.ToArray();
        int spkFeatLen = spkFeatArr.Length;
        var spkFeatB = new float[B * spkFeatLen];
        for (int b = 0; b < B; b++) Array.Copy(spkFeatArr, 0, spkFeatB, b * spkFeatLen, spkFeatLen);

        var spkFeatDims = speakerFeatures.Dimensions.ToArray();

        using var loopOut = _merged!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("speech_tokens",
                new DenseTensor<long>(speechTokensB, new[] { B, tTok })),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings",
                new DenseTensor<float>(spkEmbB, new[] { B, spkDim })),
            NamedOnnxValue.CreateFromTensor("speaker_features",
                new DenseTensor<float>(spkFeatB, new[] { B, spkFeatDims[1], spkFeatDims[2] })),
        });
        var wavTensor = loopOut.First().AsTensor<float>();
        var wavDims = wavTensor.Dimensions.ToArray();
        int nSamples = wavDims[1];
        var wavFlat = wavTensor.ToArray();
        var perChunk = new float[B][];
        for (int b = 0; b < B; b++)
        {
            perChunk[b] = new float[nSamples];
            Array.Copy(wavFlat, b * nSamples, perChunk[b], 0, nSamples);
        }
        return perChunk;
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

    /// <summary>
    /// Batched vocoder synthesis. Inputs: B parallel chunks each with
    /// its own speech_tokens (same length per chunk for the MVP), one
    /// shared speaker_embeddings, one shared speaker_features. Returns
    /// B waveforms in input order.
    ///
    /// Dispatches on the loaded bundle's capabilities:
    ///   - merged graph carrying <c>supports_batched=true</c> metadata
    ///     (Path B / Run 10) → single merged-graph call at B&gt;1 via
    ///     <see cref="RunMergedBatched"/>; split graphs aren't loaded
    ///     and ~1.1 GB VRAM is saved.
    ///   - older merged graph (no metadata flag) + split graphs present
    ///     → C# orchestrated CFM solve loop on the split 3-graph path.
    /// See <c>docs/chatterbox_perf_investigation.md</c> Run 6/10 for
    /// the rationale + perf numbers.
    ///
    /// Throws if <see cref="SupportsBatched"/> is false (neither route
    /// available). Caller's recourse: re-merge with
    /// <c>scripts/_export_utils/merge_cond_decoder_loop.py</c> for the
    /// Path B graph, or re-export with <c>--split-cond-decoder</c> to
    /// get the split fallback.
    ///
    /// MVP constraint: all <paramref name="speechTokensPerChunk"/>
    /// entries must have the same length. Phase 2 will accept ragged
    /// inputs (pad + mask).
    /// </summary>
    public float[][] SynthesizeBatch(
        IReadOnlyList<long[]> speechTokensPerChunk,
        DenseTensor<float> speakerEmbeddings,
        DenseTensor<float> speakerFeatures)
    {
        if (!SupportsBatched)
            throw new InvalidOperationException(
                "Batched vocoder synthesis requires either a Path B merged graph "
                + "(conditional_decoder_loop.onnx exported via merge_cond_decoder_loop.py "
                + "with supports_batched=true metadata) OR the split 3-graph set "
                + "(flow_encoder, cfm_estimator, mel2wav). Neither was found.");

        int B = speechTokensPerChunk.Count;
        if (B == 0) return Array.Empty<float[]>();
        if (B == 1)
        {
            // Trivial case — defer to the well-tested merged or split path.
            var wav = Synthesize(speechTokensPerChunk[0], speakerEmbeddings, speakerFeatures);
            return new[] { wav };
        }

        int tTok = speechTokensPerChunk[0].Length;
        for (int b = 0; b < B; b++)
        {
            if (speechTokensPerChunk[b].Length != tTok)
                throw new ArgumentException(
                    $"All speech_tokens chunks must have the same length (MVP); got {speechTokensPerChunk[b].Length} vs {tTok} at index {b}.");
        }

        // Run 10: merged-graph batched path keeps the CFM solve on-device
        // for all B chunks at once (no host roundtrip per CFM iter, which
        // is what kept split-batched at ~514 ms/chunk in Run 6c). Routed
        // here when the merged graph carries the supports_batched=true
        // metadata flag stamped by merge_cond_decoder_loop.py.
        if (_mergedSupportsBatched)
            return RunMergedBatched(speechTokensPerChunk, speakerEmbeddings, speakerFeatures, B, tTok);

        const int MelBins = ChatterboxConstants.MelBins;
        const int PromptLen = ChatterboxConstants.PromptLen;
        const int CfmSteps = ChatterboxConstants.CfmSteps;
        const float CfgRate = ChatterboxConstants.CfgRate;

        // Build B-batched flow_encoder inputs.
        var speechTokensB = new long[B * tTok];
        for (int b = 0; b < B; b++)
            Array.Copy(speechTokensPerChunk[b], 0, speechTokensB, b * tTok, tTok);

        var spkEmbArr = speakerEmbeddings.ToArray();
        var spkFeatArr = speakerFeatures.ToArray();
        // Replicate speaker_embeddings and speaker_features across B
        // (one voice, many chunks).
        int spkDim = speakerEmbeddings.Dimensions[1];
        var spkEmbB = new float[B * spkDim];
        for (int b = 0; b < B; b++) Array.Copy(spkEmbArr, 0, spkEmbB, b * spkDim, spkDim);

        int spkFeatLen = spkFeatArr.Length;
        var spkFeatB = new float[B * spkFeatLen];
        for (int b = 0; b < B; b++) Array.Copy(spkFeatArr, 0, spkFeatB, b * spkFeatLen, spkFeatLen);

        // 1) flow_encoder at B
        using var encOut = _flowEnc!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("speech_tokens", new DenseTensor<long>(speechTokensB, new[] { B, tTok })),
            NamedOnnxValue.CreateFromTensor("speaker_embeddings",
                new DenseTensor<float>(spkEmbB, new[] { B, spkDim })),
            NamedOnnxValue.CreateFromTensor("speaker_features",
                new DenseTensor<float>(spkFeatB, speakerFeatures.Dimensions.ToArray() is var d
                    ? new[] { B, d[1], d[2] } : throw new InvalidOperationException("speaker_features must be rank-3"))),
        });
        var encList = encOut.ToList();
        var muT = encList[0].AsTensor<float>();
        int tMel = muT.Dimensions[2];

        float[] mu = muT.ToArray();             // [B, MelBins, tMel]
        float[] mask = encList[1].AsTensor<float>().ToArray();  // [B, 1, tMel]
        float[] embed = encList[2].AsTensor<float>().ToArray(); // [B, MelBins]
        float[] cond = encList[3].AsTensor<float>().ToArray();  // [B, MelBins, tMel]
        float[] xRaw = encList[4].AsTensor<float>().ToArray();  // expected [B, MelBins, tMel]
        // flow_encoder's `z` output is the CFM initial state. Run 6c
        // discovered that the export's pinned-rand_noise patch (added in
        // Run 11 of docs/chatterbox_investigation.md for parity-test
        // reproducibility) makes z appear constant from the graph's
        // perspective, so ORT emits it at [1, MelBins, tMel] regardless
        // of input batch. We detect that case and replicate to
        // [B, MelBins, tMel] manually. Per-batch-element noise is
        // intentionally identical here (same initial seed across the
        // batch); if we ever want per-element noise variation, that's a
        // separate change to the export-side patch.
        int perBatchElemSize = MelBins * tMel;
        float[] x;
        if (xRaw.Length == perBatchElemSize)
        {
            x = new float[B * perBatchElemSize];
            for (int b = 0; b < B; b++) Array.Copy(xRaw, 0, x, b * perBatchElemSize, perBatchElemSize);
        }
        else if (xRaw.Length == B * perBatchElemSize)
        {
            x = xRaw;
        }
        else
        {
            throw new InvalidOperationException(
                $"flow_encoder.onnx z output had unexpected length {xRaw.Length}; expected {perBatchElemSize} (B=1, will replicate) or {B * perBatchElemSize}.");
        }

        // 2) CFM solve loop. CFG-doubled along batch → [2*B, ...].
        var tSpan = new float[CfmSteps + 1];
        for (int i = 0; i <= CfmSteps; i++)
        {
            float linear = i / (float)CfmSteps;
            tSpan[i] = 1.0f - MathF.Cos(linear * 0.5f * MathF.PI);
        }
        float t = tSpan[0];
        float dt = tSpan[1] - tSpan[0];

        int cfgB = 2 * B;
        int muSize = MelBins * tMel;          // per-batch-elem
        int maskSize = tMel;
        int embedSize = MelBins;
        int condSize = MelBins * tMel;

        for (int step = 1; step <= CfmSteps; step++)
        {
            // Build CFG-doubled inputs: first B rows = cond, next B rows = uncond (zeros for mu/cond/spks).
            var xIn = ReplicateCfg(x, B, muSize);              // [2B, MelBins, tMel]; both halves = x (mask gets x too in upstream conv)
            var maskIn = ReplicateCfg(mask, B, maskSize);      // [2B, 1, tMel]; both halves = mask
            var muIn = CfgZeroSecondHalf(mu, B, muSize);       // [2B, MelBins, tMel]; second half zeros
            var condIn = CfgZeroSecondHalf(cond, B, condSize); // [2B, MelBins, tMel]; second half zeros
            var spksIn = CfgZeroSecondHalf(embed, B, embedSize); // [2B, MelBins]; second half zeros
            var tIn = new float[cfgB];
            Array.Fill(tIn, t);

            using var estOut = _cfmEst!.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("x_in",    new DenseTensor<float>(xIn,    new[] { cfgB, MelBins, tMel })),
                NamedOnnxValue.CreateFromTensor("mask_in", new DenseTensor<float>(maskIn, new[] { cfgB, 1, tMel })),
                NamedOnnxValue.CreateFromTensor("mu_in",   new DenseTensor<float>(muIn,   new[] { cfgB, MelBins, tMel })),
                NamedOnnxValue.CreateFromTensor("t_in",    new DenseTensor<float>(tIn,    new[] { cfgB })),
                NamedOnnxValue.CreateFromTensor("spks_in", new DenseTensor<float>(spksIn, new[] { cfgB, MelBins })),
                NamedOnnxValue.CreateFromTensor("cond_in", new DenseTensor<float>(condIn, new[] { cfgB, MelBins, tMel })),
            });
            var dphi = estOut.First().AsTensor<float>().ToArray();  // [2B, MelBins, tMel]
            int perBatchElem = muSize;
            float a = 1.0f + CfgRate;
            // CFG combine + Euler step, per batch element.
            for (int b = 0; b < B; b++)
            {
                int condOff = b * perBatchElem;
                int uncondOff = (B + b) * perBatchElem;
                for (int i = 0; i < perBatchElem; i++)
                {
                    float c = dphi[condOff + i];
                    float u = dphi[uncondOff + i];
                    x[condOff + i] = x[condOff + i] + dt * (a * c - CfgRate * u);
                }
            }
            t = tSpan[step];
            if (step < CfmSteps) dt = tSpan[step + 1] - t;
        }

        // 3) Trim mel per batch element: drop first PromptLen frames along last dim.
        int tTrim = tMel - PromptLen;
        var feat = new float[B * MelBins * tTrim];
        for (int b = 0; b < B; b++)
        {
            int srcBaseB = b * MelBins * tMel;
            int dstBaseB = b * MelBins * tTrim;
            for (int c = 0; c < MelBins; c++)
            {
                Array.Copy(x, srcBaseB + c * tMel + PromptLen,
                           feat, dstBaseB + c * tTrim, tTrim);
            }
        }

        // 4) mel2wav at B → [B, num_samples]
        using var m2wOut = _m2w!.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(feat, new[] { B, MelBins, tTrim })),
        });
        var wavTensor = m2wOut.First().AsTensor<float>();
        int numSamples = wavTensor.Dimensions[1];
        var wavFlat = wavTensor.ToArray();
        var perChunkWavs = new float[B][];
        for (int b = 0; b < B; b++)
        {
            perChunkWavs[b] = new float[numSamples];
            Array.Copy(wavFlat, b * numSamples, perChunkWavs[b], 0, numSamples);
        }
        return perChunkWavs;
    }

    /// <summary>Stack `x` (length B*elemSize) on top of itself → 2B rows.</summary>
    private static float[] ReplicateCfg(float[] x, int B, int elemSize)
    {
        var r = new float[2 * B * elemSize];
        Array.Copy(x, 0, r, 0, B * elemSize);
        Array.Copy(x, 0, r, B * elemSize, B * elemSize);
        return r;
    }

    /// <summary>Stack `x` (length B*elemSize) on top of `B*elemSize` zeros → cond + uncond CFG pair.</summary>
    private static float[] CfgZeroSecondHalf(float[] x, int B, int elemSize)
    {
        var r = new float[2 * B * elemSize];
        Array.Copy(x, 0, r, 0, B * elemSize);
        // second half stays default zero
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
