using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

// ── Buffer pooling for median filter ──────────────────────────────────────────
using System.Buffers;

namespace Vernacula.Base;

/// <summary>
/// NVIDIA Sortformer streaming speaker diarization — C# port of sortformer.py.
///
/// Key refactor vs. WPF original: <see cref="ProcessChunk"/> now takes a
/// pre-computed full-file mel spectrogram (<c>float[,,] melSpec</c>) and slices
/// the required frames directly, eliminating per-chunk FFT re-computation and the
/// overlap/halfNFft padding logic that was previously done inside this class.
///
/// Call <see cref="AudioUtils.LogMelSpectrogram"/> once on the full audio, then
/// pass the result to <see cref="GetPredParams(float[,,])"/>,
/// <see cref="GetPreds"/>, and <see cref="GetIncrementalSegments"/>.
/// The convenience <see cref="Diarize(float[], Action{int, int}?)"/> overload
/// handles mel computation internally.
/// </summary>
public sealed class SortformerStreamer : IDisposable
{
    private readonly InferenceSession _session;

    /// <summary>
    /// The CoreML steady-state graph, or null when it is not in use. See
    /// <see cref="UsesSteadyStateGraph"/> for when it is loaded and
    /// <see cref="ProcessChunk"/> for which chunks it is allowed to see.
    /// </summary>
    private InferenceSession? _steadySession;

    /// <summary>Set once the first open has been attempted, successfully or not.</summary>
    private bool _steadySessionAttempted;

    private readonly string _modelPath;
    private readonly ExecutionProvider _ep;

    /// <summary>The checkpoint's constants: <see cref="SortformerProfile.StreamingSortformerV21"/>,
    /// or whatever the graph declares in its metadata (Nemotron-3-Diarization).</summary>
    private readonly SortformerProfile _p;

    /// <inheritdoc cref="_p"/>
    public SortformerProfile Profile => _p;

    /// <summary>
    /// True when the CoreML steady-state graph is loaded alongside the stock one.
    /// Both stay resident (~1 GB combined), which is the cost of the ~3.3x chunk
    /// speedup on the Apple Neural Engine.
    /// </summary>
    public bool UsesSteadyStateGraph => _steadySession is not null;

    /// <summary>
    /// The steady-state graph, opened on first genuine need.
    /// </summary>
    /// <remarks>
    /// ⚠ DELIBERATELY NOT OPENED IN THE CONSTRUCTOR. No chunk can route here until the
    /// cache and FIFO are both full -- 188 + 124 subsampled frames, roughly 40 s of audio.
    /// TranscriptionService builds a streamer per transcription, so an eager open made
    /// every diarization of a shorter clip load, compile and dispose a ~527 MB graph that
    /// handled zero chunks. It did the same on every app launch, where Warmup() runs one
    /// chunk from a freshly reset state and therefore always routes to the stock graph.
    /// </remarks>
    private InferenceSession? SteadyStateSession()
    {
        if (_steadySessionAttempted)
            return _steadySession;

        // The variant is an export of the v2.1 graph at v2.1's fixed shapes; nothing else has one.
        if (!_p.IsLegacyV21)
        {
            _steadySessionAttempted = true;
            return null;
        }

        _steadySessionAttempted = true;
        _steadySession = TryOpenSteadyStateSession(_modelPath, _ep);
        return _steadySession;
    }

    /// <summary>Chunks sent to the steady-state graph since the last <see cref="ResetState"/>.</summary>
    public int SteadyStateChunkCount { get; private set; }

    /// <summary>Chunks sent to the stock graph since the last <see cref="ResetState"/> --
    /// warm-up, the short tail chunk, and everything when the variant is not loaded.</summary>
    public int StockChunkCount { get; private set; }

    // ── Reusable buffers for chunk processing (eliminates per-chunk allocations) ──
    private float[]? _chunkDataBuffer;
    private float[]? _predsFlatBuffer;
    private float[]? _embsFlatBuffer;
    private float[,]? _chunkEmbsBuffer;
    private float[,]? _chunkPredsBuffer;
    private float[,]? _fpBuffer;

    // ── Streaming state ───────────────────────────────────────────────────────

    private float[,,]?  _spkcache;
    private float[,,]? _spkcachePreds;
    private float[,,]?  _fifo;
    private float[,,]?  _fifoPreds;
    private float[]?    _meanSilEmb;
    private int        _nSilFrames;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Path to a cached, graph-optimised model file.  When set (the default
    /// is <c>sortformer.optimised.onnx</c> alongside the original model), ONNX
    /// Runtime will save the optimised graph on first load and load it directly
    /// on subsequent runs, skipping the expensive graph-optimisation step that
    /// causes the 10-30 second lead-in delay.  Set to <see langword="null"/> to
    /// disable caching.
    /// </summary>
    public string? OptimisedModelPath { get; set; }

    public SortformerStreamer(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto,
                              SortformerModel model = SortformerModel.StreamingSortformerV21)
    {
        var opts = new SessionOptions();

        // ⚠ THE STOCK GRAPH CAN NEVER RUN ON THE CoreML EP. It slices by tensor value, so
        // every downstream shape is data-dependent, and CoreML's MIL runtime rejects
        // unbounded dimensions outright -- session creation throws
        // "Failed to create MLModel ... error code: -14" rather than falling back. That
        // incompatibility is the entire reason the steady-state variant exists.
        //
        // So asking this class for CoreML means "use the CoreML variant where it is valid",
        // and the stock graph it falls back to for warm-up and the tail chunk has to run
        // somewhere else. Auto picks the best remaining provider (WebGPU on macOS, else CPU)
        // and never throws.
        ExecutionProvider stockEp = ep == ExecutionProvider.CoreML ? ExecutionProvider.Auto : ep;

        // CoreML / WebGpu / macOS-Auto are handled centrally -- this class is the one
        // ORT call site #164 did not route through the shared helper, so until now
        // ExecutionProvider.CoreML and .WebGpu fell through the switch below with no
        // matching case and Auto appended nothing on macOS, silently running every
        // diarization on the CPU EP. See OrtSessionBuilder.TryAppendPlatformAccelerator.
        string resolvedModelPath = Config.GetSortformerModelPath(modelPath, model);

        if (!OrtSessionBuilder.TryAppendPlatformAccelerator(opts, stockEp, resolvedModelPath))
        {
            switch (stockEp)
            {
                case ExecutionProvider.Auto:
                    if (HardwareInfo.CanProbeCudaExecutionProvider())
                    {
                        try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                    }
                    try { opts.AppendExecutionProvider_DML(0);  } catch { }
                    break;
                case ExecutionProvider.Cuda:
                    try { opts.AppendExecutionProvider_CUDA(0); }
                    catch (EntryPointNotFoundException)
                    { throw new InvalidOperationException(HardwareInfo.CudaUnavailableMessage(providerMissing: true)); }
                    break;
                case ExecutionProvider.DirectML:
                    try { opts.AppendExecutionProvider_DML(0); }
                    catch (EntryPointNotFoundException)
                    { throw new InvalidOperationException("DirectML EP not available. Build with -p:EP=DirectML (Windows only)."); }
                    break;
                case ExecutionProvider.Cpu:
                    break;
            }
        }

        // Cache the graph-optimised model on disk so that subsequent loads skip
        // the expensive ORT graph optimisation step (typically 10-30 s).
        // ONNX Runtime will run graph optimisation on first load and save the
        // result to this path; on subsequent loads it loads the optimised graph
        // directly, bypassing the optimiser entirely.
        //
        // v2.1 only. The optimised-graph file is named for v2.1 and a second model writing to
        // it would clobber it.
        if (model == SortformerModel.StreamingSortformerV21)
        {
            if (string.IsNullOrEmpty(OptimisedModelPath))
            {
                string dir  = Path.GetDirectoryName(resolvedModelPath) ?? modelPath;
                OptimisedModelPath = Path.Combine(dir, "sortformer.optimised.onnx");
            }
            opts.OptimizedModelFilePath = OptimisedModelPath;
        }

        _session = new InferenceSession(resolvedModelPath, opts);
        try
        {
            // ⚠ THE GRAPH, NOT THE CALLER, DECIDES THE PROFILE. A v2.1 export carries no
            // metadata and gets v2.1's constants; a Nemotron-3 export declares its own. Asking
            // for Nemotron-3 and getting a graph without the declaration is a wrong file (or an
            // export from before the metadata existed), and running it with v2.1's constants
            // would produce plausible-looking garbage -- 4 of 8 speaker columns, no 10 ms head.
            _p = SortformerProfile.FromMetadata(_session.ModelMetadata.CustomMetadataMap)
                 ?? (model == SortformerModel.StreamingSortformerV21
                     ? SortformerProfile.StreamingSortformerV21
                     : throw new InvalidDataException(
                         $"{resolvedModelPath} does not declare a Sortformer profile in its metadata; " +
                         "re-export it with scripts/nemo_export/export_nemotron3_diarization_to_onnx.py."));
        }
        catch
        {
            _session.Dispose();
            throw;
        }
        _modelPath = modelPath;
        _ep = ep;
        ResetState();
    }

    /// <summary>
    /// Opens the CoreML steady-state graph, or returns null to run stock-only.
    /// </summary>
    /// <remarks>
    /// Auto selects this too, not just an explicit CoreML request. `OrtSessionBuilder`'s
    /// Auto case declines CoreML because suitability is a per-model property -- but here
    /// that property is checkable rather than assumed: the artifact is present beside the
    /// stock model and its signature either matches this class's contract or it does not.
    /// Worth it only on the Neural Engine (51.5 ms vs 171.8 ms for the stock graph on CPU,
    /// M5 / ORT 1.29.0), which is why the gate still requires the CoreML EP; an explicit
    /// Cpu or WebGpu opts out even when the artifact is present.
    ///
    /// ⚠ Two things here are load-bearing:
    /// <list type="bullet">
    /// <item><c>ORT_ENABLE_BASIC</c> -- the file is already an optimized graph, and
    /// re-optimizing one at EXTENDED or above throws `AddInitializedOrtValue Attempt to
    /// replace the existing tensor` (`MatMulAddFusion`). The CoreML EP hides this by
    /// claiming the whole graph before the CPU fusions run, so it only surfaces on a
    /// fallback to CPU -- which is exactly what happens when CoreML declines.</item>
    /// <item>No <c>OptimizedModelFilePath</c> -- the stock session writes one, and pointing
    /// both at the same file would have them overwrite each other. Re-saving an already
    /// optimized graph is also the round-trip that causes the throw above.</item>
    /// </list>
    /// A failure to open is not fatal: the stock graph handles every chunk on its own.
    /// </remarks>
    private static InferenceSession? TryOpenSteadyStateSession(string modelPath, ExecutionProvider ep)
    {
        // Detected, not configured. CoreML suits a model only when that model has been
        // built for it, and here that is a checkable fact rather than a preference: the
        // steady-state artifact sits beside the stock model and either declares the exact
        // three-input signature this class feeds or it does not. Presence plus
        // SignatureMatchesSteadyStateContract IS the per-model evidence OrtSessionBuilder's
        // Auto case declines to assume, so Auto may act on it.
        //
        // Asking a user to choose would be asking them to answer a question the artifact
        // already answers. An explicit --ep still forces the matter either way: Cpu or
        // WebGpu opts out even when the variant is present.
        bool wanted = ep == ExecutionProvider.CoreML
                      || (ep == ExecutionProvider.Auto
                          && OperatingSystem.IsMacOS()
                          && OrtSessionBuilder.CoreMLProviderAvailable);
        if (!wanted)
            return null;

        string path = Config.GetSortformerCoreMLModelPath(modelPath);
        if (!File.Exists(path))
            return null;

        InferenceSession? sess = null;
        try
        {
            // Always CoreML here, never the caller's `ep`: under Auto that would build a
            // WebGPU session for a graph exported specifically for CoreML.
            var opts = OrtSessionBuilder.Create(
                ExecutionProvider.CoreML, GraphOptimizationLevel.ORT_ENABLE_BASIC,
                coreMlModelPath: path);
            sess = new InferenceSession(path, opts);

            // ⚠ THE FILENAME IS NOT THE CONTRACT. An export made before
            // --coreml-const-chunk-length existed lands at this exact path with FOUR
            // inputs (chunk_lengths still live), and one made with different
            // --fixed-*-frames has the wrong fixed dims. Both load fine and then throw
            // out of ProcessChunk on the first steady chunk -- turning a graceful
            // fall-back into a failed run. Check the signature, not the name.
            if (!SignatureMatchesSteadyStateContract(sess))
            {
                sess.Dispose();
                return null;
            }
            return sess;
        }
        catch
        {
            // Missing provider, a graph this ORT will not take, anything: fall back to
            // stock-only rather than failing diarization outright.
            sess?.Dispose();
            return null;
        }
    }

    /// <summary>
    /// Whether a candidate graph is the steady-state variant this class knows how to feed:
    /// exactly the three tensors <see cref="ProcessChunk"/> sends, at exactly the shapes it
    /// sends them, with every <c>*_lengths</c> input folded away.
    /// </summary>
    private static bool SignatureMatchesSteadyStateContract(InferenceSession sess)
    {
        var expected = new (string Name, int[] Dims)[]
        {
            ("chunk",    new[] { 1, Config.ChunkLength * Config.Subsampling, Config.NMels }),
            ("spkcache", new[] { 1, Config.SpeakerCacheLength,               Config.EmbeddingDimension }),
            ("fifo",     new[] { 1, Config.FifoLength,                       Config.EmbeddingDimension }),
        };

        var meta = sess.InputMetadata;
        if (meta.Count != expected.Length)
            return false;

        foreach (var (name, dims) in expected)
        {
            if (!meta.TryGetValue(name, out var m))
                return false;
            if (m.Dimensions.Length != dims.Length)
                return false;
            for (int i = 0; i < dims.Length; i++)
                if (m.Dimensions[i] != dims[i])   // a dynamic axis is negative here, so it fails too
                    return false;
        }
        return true;
    }

    /// <summary>
    /// Reset streaming state. Allows reusing the loaded ONNX session across
    /// multiple runs (e.g. benchmark mode) without reloading the model.
    /// </summary>
    public void ResetState()
    {
        _spkcache      = new float[1, 0, _p.EmbeddingDimension];
        _spkcachePreds = null;
        _fifo          = new float[1, 0, _p.EmbeddingDimension];
        _fifoPreds     = new float[1, 0, _p.NumSpeakers];
        _meanSilEmb    = new float[_p.EmbeddingDimension];
        _nSilFrames    = 0;
        SteadyStateChunkCount = 0;
        StockChunkCount       = 0;

        // Clear reusable buffers so they reallocate to the right size for the new run
        _chunkDataBuffer = null;
        _predsFlatBuffer = null;
        _embsFlatBuffer = null;
        _chunkEmbsBuffer = null;
        _chunkPredsBuffer = null;
        _fpBuffer = null;
    }

    /// <summary>
    /// Ensure reusable buffers are allocated with at least the given capacity.
    /// </summary>
    private void EnsureChunkBuffer(int capacity)
    {
        if (_chunkDataBuffer is null || _chunkDataBuffer.Length < capacity)
            _chunkDataBuffer = new float[capacity];
    }

    private void EnsurePredsBuffer(long length)
    {
        if (_predsFlatBuffer is null || _predsFlatBuffer.Length < length)
            _predsFlatBuffer = new float[length];
    }

    private void EnsureEmbsBuffer(long length)
    {
        if (_embsFlatBuffer is null || _embsFlatBuffer.Length < length)
            _embsFlatBuffer = new float[length];
    }

    private void EnsureChunkEmbsBuffer(int t, int d)
    {
        if (_chunkEmbsBuffer is null || _chunkEmbsBuffer.GetLength(0) < t || _chunkEmbsBuffer.GetLength(1) < d)
            _chunkEmbsBuffer = new float[t, d];
    }

    private void EnsureChunkPredsBuffer(int t, int s)
    {
        if (_chunkPredsBuffer is null || _chunkPredsBuffer.GetLength(0) < t || _chunkPredsBuffer.GetLength(1) < s)
            _chunkPredsBuffer = new float[t, s];
    }

    private void EnsureFpBuffer(int t, int s)
    {
        if (_fpBuffer is null || _fpBuffer.GetLength(0) < t || _fpBuffer.GetLength(1) < s)
            _fpBuffer = new float[t, s];
    }

    // ── Silence profile ───────────────────────────────────────────────────────

    private void UpdateSilenceProfile(float[,,] embs, float[,,] preds)
    {
        int T = embs.GetLength(1);
        int D = _p.EmbeddingDimension;

        // NeMo's _get_silence_profile sums the WHOLE chunk's silence frames and divides
        // once:
        //     sil_emb_sum = sum(emb_seq * is_sil)
        //     upd_mean    = (mean_sil_emb * n_sil_frames + sil_emb_sum) / max(upd_n, 1)
        // This re-divided and rounded to float on every frame, which drifts. It went
        // unnoticed while the -inf silence pad kept _meanSilEmb out of the cache; now that
        // the pad is +inf it populates S * SpeakerCacheSilenceFrames rows on every
        // compression, so the difference reaches the model.
        int silCount = 0;
        var silSum = new double[D];
        for (int t = 0; t < T; t++)
        {
            float probSum = 0f;
            for (int s = 0; s < _p.NumSpeakers; s++)
                probSum += preds[0, t, s];
            if (probSum >= _p.SilThreshold)
                continue;

            silCount++;
            for (int d = 0; d < D; d++)
                silSum[d] += embs[0, t, d];
        }
        if (silCount == 0)
            return;

        var meanSilEmb = _meanSilEmb!;
        int updatedN = _nSilFrames + silCount;
        for (int d = 0; d < D; d++)
            meanSilEmb[d] = (float)(((double)meanSilEmb[d] * _nSilFrames + silSum[d])
                                    / Math.Max(updatedN, 1));
        _nSilFrames = updatedN;
    }

    // ── Quality scoring ───────────────────────────────────────────────────────

    private float[,] SpeakerQualityScores(float[,] preds2d, int minPosPerSpk)
    {
        int T = preds2d.GetLength(0);
        int S = _p.NumSpeakers;
        float floor = _p.PredScoreThreshold;
        var scores = new float[T, S];

        for (int t = 0; t < T; t++)
        {
            float logOneSum = 0f;
            for (int s = 0; s < S; s++)
            {
                float p    = preds2d[t, s];
                float lp   = (float)Math.Log(Math.Max(p,       floor));
                float lo   = (float)Math.Log(Math.Max(1f - p,  floor));
                scores[t, s] = lp - lo;
                logOneSum += lo;
            }
            // NeMo's _get_log_pred_scores ends `+ log_1_probs_sum - math.log(0.5)`.
            // This subtracted log(sqrt(2)) instead, leaving every score 1.0397 too low
            // (-log(0.5) = +0.693 vs -log(sqrt 2) = -0.347). A constant offset is harmless
            // to a pure ranking, but _disable_low_scores tests `scores > 0`, so the offset
            // moved that threshold and changed which frames were disabled -- and therefore
            // which survived compression.
            float adj = logOneSum - (float)Math.Log(0.5);
            for (int s = 0; s < S; s++)
                scores[t, s] += adj;
        }

        // ⚠ ORDER MATTERS. NeMo masks non-speech to -inf FIRST, then counts positives:
        //     is_speech = preds > 0.5
        //     scores = where(is_speech, scores, -inf)
        //     is_pos = scores > 0
        //     is_nonpos_replace = ~is_pos & is_speech & (is_pos.sum(dim=1) >= min_pos)
        // Counting before the mask (as this did) lets a non-speech frame with a positive
        // raw score inflate the count, which flips the `>= minPosPerSpk` gate and disables
        // a speaker's non-positive frames that NeMo keeps.
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                if (preds2d[t, s] <= 0.5f)
                    scores[t, s] = float.NegativeInfinity;

        var posCount = new int[S];
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                if (scores[t, s] > 0f) posCount[s]++;

        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                if (preds2d[t, s] > 0.5f && scores[t, s] <= 0f && posCount[s] >= minPosPerSpk)
                    scores[t, s] = float.NegativeInfinity;

        return scores;
    }

    private static void Boost(float[,] scores, int nBoostPerSpk, float scaleFactor)
    {
        int T = scores.GetLength(0);
        int S = scores.GetLength(1);

        // NeMo's _boost_topk_scores, with its default offset=0.5:
        //     scores[..., topk_indices, ...] -= scale_factor * math.log(offset)
        // log(0.5) is NEGATIVE, so that ADDS 0.693 * scale_factor -- it boosts, as the name
        // says. This computed `0.5 * log(2)` (= +0.347) and SUBTRACTED it, so the method
        // penalised exactly the frames it was supposed to promote: wrong sign, and half the
        // magnitude. Both strong and weak boosting are affected, which is how a speaker's
        // best frames were being pushed out of the compressed cache.
        float boost = scaleFactor * (float)Math.Log(0.5);

        for (int s = 0; s < S; s++)
        {
            var col = new (float score, int t)[T];
            for (int t = 0; t < T; t++) col[t] = (scores[t, s], t);

            // NeMo picks which frames to boost with torch.topk(scores, k, dim=1), and
            // #170 recorded that torch keeps the LOWEST indices among equal values. THAT IS
            // NOT TRUE. Measured on torch 2.11.0 (x86-64) via
            // `sortformer_compress_parity.py --probe-topk-ties`, topk over tied values
            // returns a MID-RANGE SUBSET, at an offset with no simple rule and usually with
            // holes in it:
            //
            //     zeros(12)   k=5    -> 6..10        0 gaps   (not 0..4)
            //     zeros(312)  k=35   -> 196..233     1 gap
            //     zeros(312)  k=70   -> 157..233     2 gaps
            //     zeros(1248) k=188  -> 664..935     4 gaps
            //     zeros(2000) k=188  -> 1251..1499   1 gap
            //
            // It is stable across repeats and independent of thread count, which is the
            // signature of a quickselect partition rather than a documented ordering
            // guarantee -- and #170's own fixture does not reproduce, so the order very
            // likely differs by platform too. (An earlier version of this comment called the
            // result a contiguous block and quoted spans of start+k. It is not contiguous
            // and those endpoints were inferred rather than read. Quote the probe.)
            //
            // Ties are constant here: float32 sigmoid saturates to exactly 1.0 above ~16.6
            // logits, so confident frames produce bit-identical preds and bit-identical
            // scores. There is therefore no tie order that would match NeMo, and lowest-t is
            // kept because it is deterministic, unbiased over time and readable -- not
            // because it agrees with torch. See #171 and
            // docs/investigations/sortformer_topk_ties_investigation.md.
            Array.Sort(col, (a, b) =>
            {
                int byScore = b.score.CompareTo(a.score);
                return byScore != 0 ? byScore : a.t.CompareTo(b.t);
            });

            for (int i = 0; i < Math.Min(nBoostPerSpk, T); i++)
            {
                int t = col[i].t;
                // -inf stays -inf under this arithmetic anyway; the guard just makes it explicit.
                if (!float.IsNegativeInfinity(scores[t, s]))
                    scores[t, s] -= boost;
            }
        }
    }

    /// <summary>
    /// Orders the flattened (score, frame, speaker) entries so that the first
    /// <paramref name="keep"/> are the cache's picks, and returns those picks.
    /// Deterministic, and hoisted out of <see cref="CompressCache"/> so the tie behaviour
    /// can be tested directly.
    /// </summary>
    /// <remarks>
    /// ⚠ SORTS <paramref name="flat"/> IN PLACE. The production caller builds it fresh and
    /// never reads it again, but this file pools buffers elsewhere, so a future caller that
    /// hands over a reused array would find it silently reordered.
    /// </remarks>
    /// <remarks>
    /// ⚠ TIES CANNOT BE RESOLVED THE WAY NeMo RESOLVES THEM. Sortformer's float32 sigmoid
    /// saturates to exactly 1.0 above ~16.6 logits, so a confidently single-speaker stream
    /// produces bit-identical preds rows and hence bit-identical scores. NeMo's choice among
    /// those is whatever `torch.topk` happens to return, which is a quickselect artifact --
    /// a mid-range subset of the tied entries, at an offset following no rule, usually with
    /// holes in it, and not reproducing across torch builds (see the note in
    /// <see cref="Boost"/>). There is no order to match, so the port picks one that is at
    /// least well-behaved.
    ///
    /// It breaks ties FRAME-MAJOR (earliest frame first, speaker only as a final
    /// disambiguator). The obvious alternative -- the speaker-major flattened index, which
    /// this used to do -- is what the entries are laid out in, but it orders every
    /// speaker-0 entry ahead of every speaker-1 entry, so on tied scores the cache fills
    /// from the low-numbered slots down. Measured on a 100%-saturated fixture, that gave a
    /// per-speaker split of [69, 47, 36, 36] against NeMo's [36, 49, 54, 49]; frame-major
    /// gives [38, 55, 49, 46]. Neither matches NeMo and neither can, but a monotone bias
    /// toward whichever speaker landed in slot 0 is a property worth not having, and
    /// speaker IDs here are arbitrary slot assignments.
    ///
    /// This is inert outside the saturated regime: ties that actually straddle the cut need
    /// bit-identical scores, and none occur below ~20% saturated frames (measured 0 at 0%,
    /// 5% and 20% by `sortformer_compress_parity.py --tie-incidence`), which is why fidelity
    /// DER is unaffected. See #171.
    /// </remarks>
    internal static (int tIdx, int sIdx, bool disabled, int order)[] SelectCacheFrames(
        (float score, int tIdx, int sIdx)[] flat, int keep, int extT, int realFrames)
    {
        // CompressCache always has extT * S >= keep by construction, but this is reachable
        // on its own now, and slicing past the end would be a confusing IndexOutOfRange.
        if (keep < 0 || keep > flat.Length)
            throw new ArgumentOutOfRangeException(
                nameof(keep), keep, $"cannot keep {keep} of {flat.Length} scored entries.");

        // Array.Sort is an unstable introsort, so a tie rule is required for determinism at
        // all, never mind for agreement with the Python mirror.
        Array.Sort(flat, (a, b) =>
        {
            int byScore = b.score.CompareTo(a.score);
            if (byScore != 0) return byScore;
            int byFrame = a.tIdx.CompareTo(b.tIdx);
            return byFrame != 0 ? byFrame : a.sIdx.CompareTo(b.sIdx);
        });

        // NeMo's _get_topk_indices replaces any picked entry whose score is -inf with
        // max_index, which marks it DISABLED: _gather_spkcache_and_preds then substitutes
        // the mean silence embedding and zero preds, and the huge index sorts it last.
        // Treating only the silence pad (t >= realFrames) as disabled left a -inf pick
        // holding its real embedding and real preds at its natural position. It fires
        // whenever fewer than `keep` frames survive the -inf masking -- sparse or
        // low-confidence audio, and the early stream.
        //
        // Note the ORDER of the kept rows stays SPEAKER-MAJOR: that is NeMo's
        // torch.sort(topk_indices) over speaker-major flattened indices, it is not a tie
        // rule, and it is not what changed above. -inf picks go to the very end via
        // max_index, while the silence pad keeps its natural place at the tail of its
        // speaker's block. Ordering both alike is not equivalent and measurably worse.
        return flat[..keep]
            .Select(x => (
                x.tIdx,
                x.sIdx,
                disabled: float.IsNegativeInfinity(x.score) || x.tIdx >= realFrames,
                order: float.IsNegativeInfinity(x.score) ? int.MaxValue : x.sIdx * extT + x.tIdx))
            .OrderBy(x => x.order)
            .ToArray();
    }

    // ── Cache compression ─────────────────────────────────────────────────────

    private void CompressCache()
    {
        if (_spkcachePreds is null) return;

        var spkcache = _spkcache!;
        var spkcachePreds = _spkcachePreds;

        int T = spkcache.GetLength(1);
        int S = _p.NumSpeakers;
        int D = _p.EmbeddingDimension;

        var preds2d = Slice3DTo2D(spkcachePreds, T, S);

        int cachePerSpk       = _p.SpeakerCacheLength / S - _p.SpeakerCacheSilenceFrames;
        int strongBoostPerSpk = (int)(cachePerSpk * _p.StrongBoostRate);
        int weakBoostPerSpk   = (int)(cachePerSpk * _p.WeakBoostRate);
        int minPosPerSpk      = (int)(cachePerSpk * _p.MinPosScoresRate);

        float[,] scores = SpeakerQualityScores(preds2d, minPosPerSpk);

        // NeMo boosts frames newly appended to the cache before the boosts below:
        //     if self.scores_boost_latest > 0:
        //         scores[:, self.spkcache_len:, :] += self.scores_boost_latest
        // Everything past SpeakerCacheLength is what this pop just promoted out of the
        // FIFO. Without it those frames compete against already-established ones on raw
        // score alone and are evicted almost immediately, so the cache stops refreshing.
        //
        // This was missing entirely. On 90 s of real speech it is the single largest
        // divergence from NeMo's streaming: frame-level speaker agreement 89.6% -> 97.3%.
        // It is invisible on synthetic tones, which is why it survived earlier checks.
        for (int t = _p.SpeakerCacheLength; t < T; t++)
            for (int s2 = 0; s2 < S; s2++)
                scores[t, s2] += _p.ScoresBoostLatest;

        Boost(scores, strongBoostPerSpk, 2.0f);
        Boost(scores, weakBoostPerSpk,   1.0f);

        // NeMo appends spkcache_sil_frames_per_spk rows at +inf, so each speaker's block in
        // the flattened score matrix carries that many guaranteed picks and the compressed
        // cache always reserves S * that many slots for the mean silence embedding:
        //     pad = torch.full((batch, self.spkcache_sil_frames_per_spk, n_spk), float('inf'))
        //
        // This was 3 * S rows at NEGATIVE infinity -- four times as many rows, and a sign
        // that meant they were only ever picked by tying with masked frames under an
        // unstable sort, so the cache held essentially no silence frames. cachePerSpk
        // already subtracts SpeakerCacheSilenceFrames on the assumption they are spoken for.
        int silRows   = _p.SpeakerCacheSilenceFrames;
        var extScores = new float[T + silRows, S];
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                extScores[t, s] = scores[t, s];
        for (int t = T; t < T + silRows; t++)
            for (int s = 0; s < S; s++)
                extScores[t, s] = float.PositiveInfinity;

        int extT  = T + silRows;
        int total = extT * S;
        var flat  = new (float score, int tIdx, int sIdx)[total];
        for (int t = 0; t < extT; t++)
            for (int s = 0; s < S; s++)
                flat[t * S + s] = (extScores[t, s], t, s);

        int keep = _p.SpeakerCacheLength;
        var selected = SelectCacheFrames(flat, keep, extT, T);

        var newEmbs  = new float[1, keep, D];
        var newPreds = new float[1, keep, S];

        // Nemotron-3 fills disabled slots with a LEARNED embedding; NeMo's _compress_spkcache
        // substitutes `learnable_sil_emb` for mean_sil_emb whenever use_learnable_sil_emb is set.
        var meanSilEmb = _p.LearnableSilenceEmbedding ?? _meanSilEmb!;

        for (int i = 0; i < keep; i++)
        {
            int t = selected[i].tIdx;
            if (selected[i].disabled)
            {
                // mean silence embedding, and preds left at zero
                for (int d = 0; d < D; d++)
                    newEmbs[0, i, d] = meanSilEmb[d];
                continue;
            }
            for (int d = 0; d < D; d++)
                newEmbs[0, i, d] = spkcache[0, t, d];
            for (int s = 0; s < S; s++)
                newPreds[0, i, s] = spkcachePreds[0, t, s];
        }

        _spkcache      = newEmbs;
        _spkcachePreds = newPreds;
    }

    // ── Chunk processing ──────────────────────────────────────────────────────

    /// <summary>
    /// Process one chunk using a pre-computed full-file mel spectrogram.
    /// Slices frames [start, end) directly — no per-chunk FFT computation.
    /// Returns the chunk's reported predictions: (validFrames, NumSpeakers) at 80 ms, or
    /// (the chunk's mel frames, NumSpeakers) at 10 ms for a high-resolution profile.
    /// Uses reusable internal buffers to minimize per-chunk allocations.
    /// </summary>
    public float[,] ProcessChunk(int idx, int chunkStride, int totalFrames, float[,,] melSpec)
    {
        int start      = idx * chunkStride;
        int end        = Math.Min(start + chunkStride, totalFrames);
        int currentLen = end - start;
        int S          = _p.NumSpeakers;
        int D          = _p.EmbeddingDimension;
        int nMels      = _p.NMels;
        int sub        = _p.Subsampling;
        int nMelFrames = melSpec.GetLength(1);

        // ── What the graph is fed ────────────────────────────────────────────
        // v2.1: exactly `chunkStride` rows, zero-padded past `currentLen`, which the fixed-shape
        // CoreML steady-state graph requires; chunk_lengths masks the padding.
        //
        // Everything else: NeMo's streaming_feat_loader -- the chunk plus up to
        // ChunkRightContext encoder frames of look-ahead, never padded beyond the next multiple
        // of `sub`. Padding further would be wrong, not just wasteful: Nemotron-3's 10 ms head
        // is a k=3 conv over the encoder output, so the last valid frame would read the padded
        // frame's hidden state. The multiple-of-`sub` rows are exactly what FeatureStacking adds
        // itself; the export relies on the caller doing it (see the export script).
        bool legacyInput = _p.IsLegacyV21;
        int rightOffset  = legacyInput ? 0 : Math.Min(_p.ChunkRightContext * sub, totalFrames - end);
        int inputLen     = currentLen + rightOffset;
        int inputRows    = legacyInput ? chunkStride : (inputLen + sub - 1) / sub * sub;

        EnsureChunkBuffer(inputRows * nMels);
        var chunkData = _chunkDataBuffer!;
        // Clear only the portion we'll use (important for padding rows)
        Array.Clear(chunkData, 0, inputRows * nMels);

        // Slice frames [start, start + inputLen) from the pre-computed spectrogram.
        for (int t = 0; t < inputLen; t++)
        {
            int srcRow = start + t;
            if (srcRow < nMelFrames)
            {
                int dstOffset = t * nMels;
                for (int m = 0; m < nMels; m++)
                    chunkData[dstOffset + m] = melSpec[0, srcRow, m];
            }
        }

        var spkcache = _spkcache!;
        var fifo = _fifo!;
        int cacheT = spkcache.GetLength(1);
        int fifoT  = fifo.GetLength(1);

        // ── Which graph gets this chunk ──────────────────────────────────────
        // The steady-state graph has all three *_lengths baked in as constants, so it is
        // correct ONLY when the real lengths equal the baked ones. Two cases fail that:
        //
        //   * the LAST chunk of every recording, where `currentLen` is short. Its
        //     zero-padded tail would be attended to as real audio, moving that chunk's
        //     speaker probabilities by up to 0.54 (rms 0.24) on a 0..1 scale -- enough to
        //     flip speaker assignments outright. This is the whole reason for the routing.
        //   * WARM-UP, where the cache and FIFO have not filled yet. ResetState starts
        //     both at length 0 and they grow, so the fixed [1,188,512] / [1,124,512]
        //     inputs do not even match until steady state is reached.
        //
        // Anything that is not exactly steady state goes to the stock graph.

        // Shape test first, THEN open: this is the only place that knows a chunk is
        // actually eligible, and opening costs ~527 MB and up to ~3 s.
        bool steadyShapes =
            legacyInput
            && currentLen == chunkStride
            && chunkStride == Config.ChunkLength * Config.Subsampling
            && cacheT == Config.SpeakerCacheLength
            && fifoT == Config.FifoLength;

        InferenceSession? steadySession = steadyShapes ? SteadyStateSession() : null;
        bool steadyState = steadySession is not null;

        var inputs = new List<NamedOnnxValue>(steadyState ? 3 : 6)
        {
            NamedOnnxValue.CreateFromTensor("chunk",
                new DenseTensor<float>(chunkData.AsMemory(0, inputRows * nMels),
                    new[] { 1, inputRows, nMels })),
            NamedOnnxValue.CreateFromTensor("spkcache",
                new DenseTensor<float>(Flatten3D(spkcache, 1, cacheT, D),
                    new[] { 1, cacheT, D })),
            NamedOnnxValue.CreateFromTensor("fifo",
                new DenseTensor<float>(Flatten3D(fifo, 1, fifoT, D),
                    new[] { 1, fifoT, D })),
        };
        if (!steadyState)
        {
            // The steady-state graph does not declare these -- they were folded out of its
            // signature when they became constants, so passing them would be an error.
            inputs.Add(NamedOnnxValue.CreateFromTensor("chunk_lengths",
                new DenseTensor<long>(new long[] { inputLen }, new[] { 1 })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("spkcache_lengths",
                new DenseTensor<long>(new long[] { cacheT }, new[] { 1 })));
            inputs.Add(NamedOnnxValue.CreateFromTensor("fifo_lengths",
                new DenseTensor<long>(new long[] { fifoT }, new[] { 1 })));
        }

        if (steadyState) SteadyStateChunkCount++; else StockChunkCount++;

        using var results = (steadyState ? steadySession! : _session).Run(inputs);
        var predsT = results.First(r => r.Name == "spkcache_fifo_chunk_preds").AsTensor<float>();
        var embsT  = results.First(r => r.Name == "chunk_pre_encode_embs").AsTensor<float>();
        var predsHrT = _p.UpsampleFactor > 1
            ? results.First(r => r.Name == "spkcache_fifo_chunk_preds_hr").AsTensor<float>()
            : null;

        int predsLen  = (int)predsT.Length;
        int embsLen   = (int)embsT.Length;

        // Use reusable flat buffers instead of allocating per-chunk
        EnsurePredsBuffer(predsLen);
        EnsureEmbsBuffer(embsLen);
        var predsFlat = _predsFlatBuffer!;
        var embsFlat  = _embsFlatBuffer!;
        for (int i = 0; i < predsLen; i++) predsFlat[i] = predsT.GetValue(i);
        for (int i = 0; i < embsLen;  i++) embsFlat[i]  = embsT.GetValue(i);

        int predTOut    = predsLen / S;
        int embTOut     = embsLen  / D;
        // Encoder frames that belong to this chunk, i.e. excluding the right context. NeMo:
        // chunk_len = chunk.shape[1] - lc - rc, with rc = ceil(right_offset / sub).
        int validFrames = (inputLen + sub - 1) / sub - (rightOffset + sub - 1) / sub;

        static int SafeEnd(int s, int l, int max) => Math.Min(s + l, max);

        int fpStart = cacheT;
        int fpEnd   = SafeEnd(fpStart, fifoT, predTOut);
        int fpLen   = Math.Max(fpEnd - fpStart, 0);

        int cpStart = cacheT + fifoT;
        int cpEnd   = SafeEnd(cpStart, validFrames, predTOut);
        int cpLen   = Math.Max(cpEnd - cpStart, 0);

        int ceLen = Math.Min(validFrames, embTOut);

        // Use reusable output buffers
        EnsureChunkEmbsBuffer(ceLen, D);
        EnsureChunkPredsBuffer(cpLen, S);
        EnsureFpBuffer(fpLen > 0 ? fpLen : 1, S);

        var chunkEmbs  = _chunkEmbsBuffer!;
        var chunkPreds = _chunkPredsBuffer!;
        var fp = _fpBuffer!;

        for (int t = 0; t < ceLen; t++)
            for (int d = 0; d < D; d++)
                chunkEmbs[t, d] = embsFlat[t * D + d];

        for (int t = 0; t < cpLen; t++)
            for (int s = 0; s < S; s++)
                chunkPreds[t, s] = predsFlat[(cpStart + t) * S + s];

        for (int t = 0; t < fpLen; t++)
            for (int s = 0; s < S; s++)
                fp[t, s] = predsFlat[(fpStart + t) * S + s];

        var fifoCurrent = _fifo!;
        _fifo = Concat3DAxis1(fifoCurrent, Wrap2DIn3D(chunkEmbs, ceLen, D));

        // NeMo ASSIGNS fifo_preds from this pass's output, then appends the chunk's:
        //     streaming_state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
        //     streaming_state.fifo_preds = cat([fifo_preds, chunk_preds], dim=1)
        // i.e. cat(fp, chunkPreds). `fp` is the FRESH prediction for the frames already in
        // _fifo, so it REPLACES the old preds for those frames; chunkPreds belongs to the
        // embeddings just appended.
        //
        // This previously read cat(_fifoPreds, fp): it kept the previous pass's preds and
        // appended the fresh ones, so _fifoPreds[0..fifoT) described the chunk BEFORE the
        // one sitting in _fifo[0..fifoT), and chunkPreds was never stored except when the
        // FIFO was empty. popEmbs/popPreds were therefore mismatched pairs, and they feed
        // both UpdateSilenceProfile and _spkcachePreds -- which CompressCache scores to
        // decide which frames survive. It never crashed because the lengths agree in
        // steady state; it only ever produced wrong pairings.
        _fifoPreds = fpLen > 0
            ? Concat3DAxis1(Wrap2DIn3D(fp, fpLen, S), Wrap2DIn3D(chunkPreds, cpLen, S))
            : Wrap2DIn3D(chunkPreds, cpLen, S);

        int newFifoT = _fifo.GetLength(1);
        if (newFifoT > _p.FifoLength)
        {
            // NeMo's SortformerModules.streaming_update, verbatim:
            //     pop_out_len = self.spkcache_update_period
            //     pop_out_len = max(pop_out_len, max_chunk_len - max_fifo_len + fifo_len)
            //     pop_out_len = min(pop_out_len, fifo_len + chunk_len)
            // where fifo_len is the length BEFORE the chunk was appended (fifoT here) and
            // fifo_len + chunk_len is the length after (newFifoT).
            //
            // This previously read `(newFifoT - FifoLength) + newFifoT`, i.e. 2*newFifoT-124,
            // which exceeds newFifoT for every newFifoT > 124 -- so popLen always clamped to
            // the whole FIFO and _fifo drained to 0 on every pop, alternating 124, 0, 124, 0
            // instead of holding at 124. Measured against NeMo's own forward_streaming on
            // identical features, the corrected trajectory matches the reference.
            //
            // chunk_len is THIS chunk's (validFrames), as in NeMo; it differs from the schedule's
            // only on the final chunk, after which the state is never read again.
            int popLen = _p.SpeakerCacheUpdatePeriod;
            popLen = Math.Max(popLen, validFrames - _p.FifoLength + fifoT);
            popLen = Math.Min(popLen, newFifoT);

            var popEmbs  = SliceFront3D(_fifo,     popLen, D);
            var popPreds = SliceFront3D(_fifoPreds, popLen, S);

            // With a learned silence embedding NeMo never computes the running mean
            // (streaming_update skips _get_silence_profile), so neither do we.
            if (_p.LearnableSilenceEmbedding is null)
                UpdateSilenceProfile(popEmbs, popPreds);

            _fifo      = SliceTail3D(_fifo,     popLen, D);
            _fifoPreds = SliceTail3D(_fifoPreds, popLen, S);

            var spkcacheCurrent = _spkcache!;
            _spkcache = Concat3DAxis1(spkcacheCurrent, popEmbs);

            // NeMo appends only when spkcache_preds already exists, and DEFERS the first
            // seed until compression actually needs it:
            //     if spkcache_preds is not None: spkcache_preds = cat([spkcache_preds, pop_out_preds])
            //     if spkcache.shape[1] > spkcache_len:
            //         if spkcache_preds is None:
            //             spkcache_preds = cat([preds[:, :spkcache_len], pop_out_preds])
            //
            // The deferral is the point: it seeds from THIS pass's predictions for the
            // frames already in the cache. Seeding on the first pop instead (as this did)
            // leaves those rows carrying the previous chunk's preds, and CompressCache
            // scores exactly those -- so the first compression picked a different 188
            // frames and every later cache inherited it.
            if (_spkcachePreds is not null)
                _spkcachePreds = Concat3DAxis1(_spkcachePreds, popPreds);

            if (_spkcache.GetLength(1) > _p.SpeakerCacheLength)
            {
                if (_spkcachePreds is null)
                {
                    // preds rows [0, cacheT) are this pass's output for the frames that
                    // were already in the cache before popEmbs was appended.
                    var scFresh = new float[cacheT, S];
                    for (int t = 0; t < cacheT; t++)
                        for (int s2 = 0; s2 < S; s2++)
                            scFresh[t, s2] = predsFlat[t * S + s2];
                    _spkcachePreds = Concat3DAxis1(Wrap2DIn3D(scFresh, cacheT, S), popPreds);
                }
                CompressCache();
            }
        }

        if (predsHrT is not null)
        {
            // What NeMo reports for a high-resolution model: the 10 ms slice of this chunk,
            // [(cache + fifo) * U, + chunk * U), taken from the SAME pass whose 80 ms average
            // drove the state update above. Trimmed to this chunk's mel frames, so the final
            // chunk does not report past the end of the audio -- NeMo's forward_streaming cuts
            // total_preds to ceil(n_mel / output_subsampling_factor) the same way.
            int U       = _p.UpsampleFactor;
            int hrTOut  = (int)predsHrT.Length / S;
            int hrStart = (cacheT + fifoT) * U;
            int hrLen   = Math.Min((currentLen * U + sub - 1) / sub, validFrames * U);
            hrLen       = Math.Max(Math.Min(hrLen, hrTOut - hrStart), 0);

            var hr = new float[hrLen, S];
            for (int t = 0; t < hrLen; t++)
                for (int s = 0; s < S; s++)
                    hr[t, s] = predsHrT.GetValue((hrStart + t) * S + s);
            return hr;
        }

        // Return a copy since the buffer will be reused for the next chunk
        var result = new float[cpLen, S];
        for (int t = 0; t < cpLen; t++)
            for (int s = 0; s < S; s++)
                result[t, s] = chunkPreds[t, s];
        return result;
    }

    // ── Incremental segmentation ──────────────────────────────────────────────

    private (int numPredFrames, float[,] medFiltered) FilterPredsUpTo(
        List<float[,]> allPreds, int upToFrame)
    {
        int S = _p.NumSpeakers;
        var trimmed = new List<float[,]>();
        int acc = 0;
        foreach (var chunk in allPreds)
        {
            int ct = chunk.GetLength(0);
            if (acc + ct <= upToFrame)
            {
                trimmed.Add(chunk);
                acc += ct;
            }
            else
            {
                int take = upToFrame - acc;
                if (take > 0)
                {
                    var partial = new float[take, S];
                    for (int t = 0; t < take; t++)
                        for (int s = 0; s < S; s++)
                            partial[t, s] = chunk[t, s];
                    trimmed.Add(partial);
                    acc = upToFrame;
                }
                break;
            }
        }
        return FilterPreds(trimmed, acc);
    }

    /// <summary>
    /// Process all chunks and yield newly-committed segments after each one.
    /// </summary>
    public IEnumerable<IReadOnlyList<(double start, double end, string spkId)>>
        GetIncrementalSegments(float[,,] melSpec, int totalFrames, int chunkStride, int numChunks)
    {
        int half = _p.Window / 2;
        var allPreds = new List<float[,]>(numChunks);
        var emitted  = new HashSet<(double start, double end, string spkId)>();
        int accumFrames = 0;

        for (int idx = 0; idx < numChunks; idx++)
        {
            var preds = ProcessChunk(idx, chunkStride, totalFrames, melSpec);
            allPreds.Add(preds);
            accumFrames += preds.GetLength(0);

            bool isLast     = idx == numChunks - 1;
            int  safeFrames = isLast ? accumFrames : accumFrames - half;
            if (safeFrames <= 0)
            {
                yield return Array.Empty<(double, double, string)>();
                continue;
            }

            var (numPred, filtered) = isLast
                ? FilterPreds(allPreds, accumFrames)
                : FilterPredsUpTo(allPreds, safeFrames);

            var currentSegs = BinarizePredToSegments(numPred, filtered);

            double safeTime = (safeFrames - half) * _p.FrameDuration;
            var newStable = currentSegs
                .Where(s => (isLast || s.end <= safeTime) && emitted.Add(s))
                .ToList();

            yield return newStable;
        }
    }

    // ── Batch prediction ──────────────────────────────────────────────────────

    /// <summary>
    /// Compute pred params from a pre-computed mel spectrogram.
    /// </summary>
    public (int totalFrames, int chunkStride, int numChunks) GetPredParams(float[,,] melSpec)
    {
        int totalFrames = melSpec.GetLength(1);
        int chunkStride = _p.ChunkStride;
        int numChunks   = (totalFrames + chunkStride - 1) / chunkStride;
        return (totalFrames, chunkStride, numChunks);
    }

    /// <summary>
    /// Compute pred params from raw audio (convenience overload; does not compute mel).
    /// </summary>
    public (int totalFrames, int chunkStride, int numChunks) GetPredParams(float[] audio)
    {
        int paddedLen   = audio.Length + Config.NFft;
        int totalFrames = (paddedLen - Config.NFft) / Config.HopLength + 1;
        int chunkStride = _p.ChunkStride;
        int numChunks   = (totalFrames + chunkStride - 1) / chunkStride;
        return (totalFrames, chunkStride, numChunks);
    }

    public IEnumerable<(int numChunks, int idx, float[,] chunkPreds)>
        GetPreds(float[,,] melSpec, int totalFrames, int chunkStride, int numChunks)
    {
        for (int idx = 0; idx < numChunks; idx++)
        {
            var preds = ProcessChunk(idx, chunkStride, totalFrames, melSpec);
            yield return (numChunks, idx, preds);
        }
    }

    // ── Post-processing ───────────────────────────────────────────────────────

    public (int numPredFrames, float[,] medFiltered) FilterPreds(
        IReadOnlyList<float[,]> allPreds, int totalFrames)
    {
        int S    = _p.NumSpeakers;
        int totT = allPreds.Sum(p => p.GetLength(0));
        var preds = new float[totT, S];
        int offset = 0;
        foreach (var chunk in allPreds)
        {
            int ct = chunk.GetLength(0);
            for (int t = 0; t < ct; t++)
                for (int s = 0; s < S; s++)
                    preds[offset + t, s] = chunk[t, s];
            offset += ct;
        }

        int half     = _p.Window / 2;
        var filtered = new float[totT, S];

        // Single reusable window buffer — avoids allocating one per frame per speaker
        var window = new float[_p.Window];

        for (int spk = 0; spk < S; spk++)
            for (int t = 0; t < totT; t++)
            {
                int start = Math.Max(t - half, 0);
                int end   = Math.Min(t + half + 1, totT);
                int wlen  = end - start;
                for (int i = 0; i < wlen; i++)
                    window[i] = preds[start + i, spk];
                filtered[t, spk] = Median(window, wlen);
            }

        return (totT, filtered);
    }

    public List<(double start, double end, string spkId)>
        BinarizePredToSegments(int numPredFrames, float[,] medFiltered)
    {
        int S           = _p.NumSpeakers;
        var allSegments = new List<(double, double, string)>();

        for (int spk = 0; spk < S; spk++)
        {
            bool inSeg    = false;
            int  segStart = 0;
            var  tempSegs = new List<(double start, double end)>();

            for (int t = 0; t < numPredFrames; t++)
            {
                float p = medFiltered[t, spk];
                if (p >= _p.OnsetThreshold && !inSeg)
                {
                    inSeg    = true;
                    segStart = t;
                }
                else if (p < _p.OffsetThreshold && inSeg)
                {
                    inSeg = false;
                    double s = Math.Max(segStart * _p.FrameDuration - _p.PadOnset, 0.0);
                    double e = t * _p.FrameDuration + _p.PadOffset;
                    if (e - s >= _p.MinDurOn) tempSegs.Add((s, e));
                }
            }

            if (inSeg)
            {
                double s = Math.Max(segStart * _p.FrameDuration - _p.PadOnset, 0.0);
                double e = numPredFrames * _p.FrameDuration + _p.PadOffset;
                if (e - s >= _p.MinDurOn) tempSegs.Add((s, e));
            }

            var merged = new List<(double start, double end)>();
            foreach (var seg in tempSegs)
            {
                if (merged.Count == 0)
                    merged.Add(seg);
                else
                {
                    var (ps, pe) = merged[^1];
                    if (seg.start - pe < _p.MinDurOff)
                        merged[^1] = (ps, seg.end);
                    else
                        merged.Add(seg);
                }
            }

            foreach (var (s, e) in merged)
                allSegments.Add((s, e, $"speaker_{spk}"));
        }

        allSegments.Sort((a, b) => a.Item1.CompareTo(b.Item1));
        return allSegments;
    }

    /// <summary>
    /// Full diarization pipeline. Computes mel spectrogram internally.
    /// </summary>
    public List<(double start, double end, string spkId)> Diarize(
        float[] audio, Action<int, int>? progressCallback = null)
    {
        float[,,] melSpec = AudioUtils.LogMelSpectrogram(audio);
        var (totalFrames, chunkStride, numChunks) = GetPredParams(melSpec);
        var allPreds = new List<float[,]>(numChunks);

        foreach (var (nc, idx, chunkPreds) in GetPreds(melSpec, totalFrames, chunkStride, numChunks))
        {
            progressCallback?.Invoke(idx, nc);
            allPreds.Add(chunkPreds);
        }

        var (numPredFrames, medFiltered) = FilterPreds(allPreds, totalFrames);
        return BinarizePredToSegments(numPredFrames, medFiltered);
    }

    // ── Array utility helpers ─────────────────────────────────────────────────

    private static float[] Flatten3D(float[,,] a, int d0, int d1, int d2)
    {
        var flat = new float[d0 * d1 * d2];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    flat[i * d1 * d2 + j * d2 + k] = a[i, j, k];
        return flat;
    }

    private static float[,] Slice3DTo2D(float[,,] a, int T, int D)
    {
        var r = new float[T, D];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                r[t, d] = a[0, t, d];
        return r;
    }

    private static float[,,] Wrap2DIn3D(float[,] a, int T, int D)
    {
        var r = new float[1, T, D];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                r[0, t, d] = a[t, d];
        return r;
    }

    private static float[,,] Concat3DAxis1(float[,,] a, float[,,] b)
    {
        int T1 = a.GetLength(1), T2 = b.GetLength(1), D = a.GetLength(2);
        var r  = new float[1, T1 + T2, D];
        for (int t = 0; t < T1; t++)
            for (int d = 0; d < D; d++) r[0, t,      d] = a[0, t, d];
        for (int t = 0; t < T2; t++)
            for (int d = 0; d < D; d++) r[0, T1 + t, d] = b[0, t, d];
        return r;
    }

    private static float[,,] SliceFront3D(float[,,] a, int len, int D)
    {
        var r = new float[1, len, D];
        for (int t = 0; t < len; t++)
            for (int d = 0; d < D; d++) r[0, t, d] = a[0, t, d];
        return r;
    }

    private static float[,,] SliceTail3D(float[,,] a, int from, int D)
    {
        int T   = a.GetLength(1);
        int rem = T - from;
        if (rem <= 0) return new float[1, 0, D];
        var r = new float[1, rem, D];
        for (int t = 0; t < rem; t++)
            for (int d = 0; d < D; d++) r[0, t, d] = a[0, from + t, d];
        return r;
    }

    private static float Median(float[] data, int length)
    {
        if (length <= 0) return 0f;
        if (length == 1) return data[0];
        if (length == 2) return (data[0] + data[1]) * 0.5f;

        // Copy only the used portion into a sorted buffer (avoids slice allocation)
        var tmp = ArrayPool<float>.Shared.Rent(length);
        Array.Copy(data, tmp, length);

        // Insertion sort — faster than Array.Sort for small arrays (window ≤ 11)
        for (int i = 1; i < length; i++)
        {
            float key = tmp[i];
            int j = i - 1;
            while (j >= 0 && tmp[j] > key)
            {
                tmp[j + 1] = tmp[j];
                j--;
            }
            tmp[j + 1] = key;
        }

        float result;
        if (length % 2 == 0)
            result = (tmp[length / 2 - 1] + tmp[length / 2]) * 0.5f;
        else
            result = tmp[length / 2];

        ArrayPool<float>.Shared.Return(tmp);
        return result;
    }

    // ── Warmup ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Perform a single dummy inference to trigger expensive ONNX Runtime
    /// initialisation (graph optimisation, CUDA/DML provider setup, memory
    /// allocation) so that the first real chunk processes without the usual
    /// lead-in delay.
    /// </summary>
    /// <param name="ct">Optional cancellation token.</param>
    public void Warmup(CancellationToken ct = default)
    {
        // Create a tiny dummy mel spectrogram — one chunk of zeros is enough
        // to exercise the full inference path without meaningful compute.
        int chunkStride = _p.ChunkStride;
        var dummyMel = new float[1, chunkStride, _p.NMels];

        ProcessChunk(0, chunkStride, chunkStride, dummyMel);
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        _session.Dispose();
        _steadySession?.Dispose();
    }
}
