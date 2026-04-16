using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

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

    public SortformerStreamer(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = new SessionOptions();
        switch (ep)
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
                { throw new InvalidOperationException("CUDA EP not available in current OnnxRuntime build."); }
                break;
            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("DirectML EP not available. Build with -p:UseDirectML=true."); }
                break;
            case ExecutionProvider.Cpu:
                break;
        }

        string resolvedModelPath = Config.GetSortformerModelPath(modelPath);

        // Cache the graph-optimised model on disk so that subsequent loads skip
        // the expensive ORT graph optimisation step (typically 10-30 s).
        // ONNX Runtime will run graph optimisation on first load and save the
        // result to this path; on subsequent loads it loads the optimised graph
        // directly, bypassing the optimiser entirely.
        if (string.IsNullOrEmpty(OptimisedModelPath))
        {
            string dir  = Path.GetDirectoryName(resolvedModelPath) ?? modelPath;
            OptimisedModelPath = Path.Combine(dir, "sortformer.optimised.onnx");
        }
        opts.OptimizedModelFilePath = OptimisedModelPath;

        _session = new InferenceSession(resolvedModelPath, opts);
        ResetState();
    }

    /// <summary>
    /// Reset streaming state. Allows reusing the loaded ONNX session across
    /// multiple runs (e.g. benchmark mode) without reloading the model.
    /// </summary>
    public void ResetState()
    {
        _spkcache      = new float[1, 0, Config.EmbeddingDimension];
        _spkcachePreds = null;
        _fifo          = new float[1, 0, Config.EmbeddingDimension];
        _fifoPreds     = new float[1, 0, Config.NumSpeakers];
        _meanSilEmb    = new float[Config.EmbeddingDimension];
        _nSilFrames    = 0;
    }

    // ── Silence profile ───────────────────────────────────────────────────────

    private void UpdateSilenceProfile(float[,,] embs, float[,,] preds)
    {
        int T = embs.GetLength(1);
        for (int t = 0; t < T; t++)
        {
            float probSum = 0f;
            for (int s = 0; s < Config.NumSpeakers; s++)
                probSum += preds[0, t, s];

            if (probSum < Config.SilThreshold)
            {
                _nSilFrames++;
                var meanSilEmb = _meanSilEmb!;
                for (int d = 0; d < Config.EmbeddingDimension; d++)
                {
                    double oldSum = meanSilEmb[d] * (_nSilFrames - 1);
                    meanSilEmb[d] = (float)((oldSum + embs[0, t, d]) / _nSilFrames);
                }
            }
        }
    }

    // ── Quality scoring ───────────────────────────────────────────────────────

    private float[,] SpeakerQualityScores(float[,] preds2d, int minPosPerSpk)
    {
        int T = preds2d.GetLength(0);
        int S = Config.NumSpeakers;
        var scores = new float[T, S];

        for (int t = 0; t < T; t++)
        {
            float logOneSum = 0f;
            for (int s = 0; s < S; s++)
            {
                float p    = preds2d[t, s];
                float lp   = (float)Math.Log(Math.Max(p,       0.25f));
                float lo   = (float)Math.Log(Math.Max(1f - p,  0.25f));
                scores[t, s] = lp - lo;
                logOneSum += lo;
            }
            float adj = logOneSum - (float)Math.Log(Math.Sqrt(2));
            for (int s = 0; s < S; s++)
                scores[t, s] += adj;
        }

        var posCount = new int[S];
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                if (scores[t, s] > 0f) posCount[s]++;

        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
            {
                if (preds2d[t, s] <= 0.5f)
                    scores[t, s] = float.NegativeInfinity;
                else if (scores[t, s] <= 0f && posCount[s] >= minPosPerSpk)
                    scores[t, s] = float.NegativeInfinity;
            }

        return scores;
    }

    private static void Boost(float[,] scores, int nBoostPerSpk, float scaleFactor)
    {
        int T = scores.GetLength(0);
        int S = scores.GetLength(1);
        float logHalf = 0.5f * (float)Math.Log(2.0);

        for (int s = 0; s < S; s++)
        {
            var col = new (float score, int t)[T];
            for (int t = 0; t < T; t++) col[t] = (scores[t, s], t);
            Array.Sort(col, (a, b) => b.score.CompareTo(a.score));

            for (int i = 0; i < Math.Min(nBoostPerSpk, T); i++)
            {
                int t = col[i].t;
                if (!float.IsNegativeInfinity(scores[t, s]))
                    scores[t, s] -= scaleFactor * logHalf;
            }
        }
    }

    // ── Cache compression ─────────────────────────────────────────────────────

    private void CompressCache()
    {
        if (_spkcachePreds is null) return;

        var spkcache = _spkcache!;
        var spkcachePreds = _spkcachePreds;

        int T = spkcache.GetLength(1);
        int S = Config.NumSpeakers;

        var preds2d = Slice3DTo2D(spkcachePreds, T, S);

        int cachePerSpk       = Config.SpeakerCacheLength / S - 3;
        int strongBoostPerSpk = (int)(cachePerSpk * 0.75);
        int weakBoostPerSpk   = (int)(cachePerSpk * 1.5);
        int minPosPerSpk      = (int)(cachePerSpk * 0.5);

        float[,] scores = SpeakerQualityScores(preds2d, minPosPerSpk);
        Boost(scores, strongBoostPerSpk, 2.0f);
        Boost(scores, weakBoostPerSpk,   1.0f);

        int silRows   = 3 * S;
        var extScores = new float[T + silRows, S];
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                extScores[t, s] = scores[t, s];
        for (int t = T; t < T + silRows; t++)
            for (int s = 0; s < S; s++)
                extScores[t, s] = float.NegativeInfinity;

        int extT  = T + silRows;
        int total = extT * S;
        var flat  = new (float score, int tIdx, int sIdx)[total];
        for (int t = 0; t < extT; t++)
            for (int s = 0; s < S; s++)
                flat[t * S + s] = (extScores[t, s], t, s);

        Array.Sort(flat, (a, b) => b.score.CompareTo(a.score));

        int keep     = Config.SpeakerCacheLength;
        var selected = flat[..keep].OrderBy(x => x.sIdx).ThenBy(x => x.tIdx).ToArray();

        var newEmbs  = new float[1, keep, Config.EmbeddingDimension];
        var newPreds = new float[1, keep, S];
        var meanSilEmb = _meanSilEmb!;

        for (int i = 0; i < keep; i++)
        {
            int t = selected[i].tIdx;
            if (t >= T)
            {
                for (int d = 0; d < Config.EmbeddingDimension; d++)
                    newEmbs[0, i, d] = meanSilEmb[d];
                continue;
            }
            for (int d = 0; d < Config.EmbeddingDimension; d++)
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
    /// Returns chunk_preds (validFrames, NumSpeakers).
    /// </summary>
    public float[,] ProcessChunk(int idx, int chunkStride, int totalFrames, float[,,] melSpec)
    {
        int start      = idx * chunkStride;
        int end        = Math.Min(start + chunkStride, totalFrames);
        int currentLen = end - start;
        int S          = Config.NumSpeakers;
        int D          = Config.EmbeddingDimension;
        int nMelFrames = melSpec.GetLength(1);

        // Slice frames [start, end) from the pre-computed spectrogram.
        // Remaining rows stay zero (padding for the last chunk).
        var chunkData = new float[chunkStride * Config.NMels];
        for (int t = 0; t < currentLen; t++)
        {
            int srcRow = start + t;
            if (srcRow < nMelFrames)
                for (int m = 0; m < Config.NMels; m++)
                    chunkData[t * Config.NMels + m] = melSpec[0, srcRow, m];
        }

        var spkcache = _spkcache!;
        var fifo = _fifo!;
        int cacheT = spkcache.GetLength(1);
        int fifoT  = fifo.GetLength(1);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("chunk",
                new DenseTensor<float>(chunkData,
                    new[] { 1, chunkStride, Config.NMels })),
            NamedOnnxValue.CreateFromTensor("chunk_lengths",
                new DenseTensor<long>(new long[] { currentLen }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("spkcache",
                new DenseTensor<float>(Flatten3D(spkcache, 1, cacheT, D),
                    new[] { 1, cacheT, D })),
            NamedOnnxValue.CreateFromTensor("spkcache_lengths",
                new DenseTensor<long>(new long[] { cacheT }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("fifo",
                new DenseTensor<float>(Flatten3D(fifo, 1, fifoT, D),
                    new[] { 1, fifoT, D })),
            NamedOnnxValue.CreateFromTensor("fifo_lengths",
                new DenseTensor<long>(new long[] { fifoT }, new[] { 1 })),
        };

        using var results = _session.Run(inputs);
        var predsT = results.First(r => r.Name == "spkcache_fifo_chunk_preds").AsTensor<float>();
        var embsT  = results.First(r => r.Name == "chunk_pre_encode_embs").AsTensor<float>();

        int predsLen  = (int)predsT.Length;
        int embsLen   = (int)embsT.Length;
        var predsFlat = new float[predsLen];
        var embsFlat  = new float[embsLen];
        for (int i = 0; i < predsLen; i++) predsFlat[i] = predsT.GetValue(i);
        for (int i = 0; i < embsLen;  i++) embsFlat[i]  = embsT.GetValue(i);

        int predTOut    = predsLen / S;
        int embTOut     = embsLen  / D;
        int validFrames = (currentLen + Config.Subsampling - 1) / Config.Subsampling;

        static int SafeEnd(int s, int l, int max) => Math.Min(s + l, max);

        int fpStart = cacheT;
        int fpEnd   = SafeEnd(fpStart, fifoT, predTOut);
        int fpLen   = Math.Max(fpEnd - fpStart, 0);

        int cpStart = cacheT + fifoT;
        int cpEnd   = SafeEnd(cpStart, validFrames, predTOut);
        int cpLen   = Math.Max(cpEnd - cpStart, 0);

        int ceLen = Math.Min(validFrames, embTOut);

        var chunkEmbs  = new float[ceLen, D];
        var chunkPreds = new float[cpLen, S];

        for (int t = 0; t < ceLen; t++)
            for (int d = 0; d < D; d++)
                chunkEmbs[t, d] = embsFlat[t * D + d];

        for (int t = 0; t < cpLen; t++)
            for (int s = 0; s < S; s++)
                chunkPreds[t, s] = predsFlat[(cpStart + t) * S + s];

        var fp = new float[fpLen, S];
        for (int t = 0; t < fpLen; t++)
            for (int s = 0; s < S; s++)
                fp[t, s] = predsFlat[(fpStart + t) * S + s];

        var fifoCurrent = _fifo!;
        var fifoPredsCurrent = _fifoPreds!;
        _fifo = Concat3DAxis1(fifoCurrent, Wrap2DIn3D(chunkEmbs, ceLen, D));

        if (fpLen > 0)
            _fifoPreds = Concat3DAxis1(fifoPredsCurrent, Wrap2DIn3D(fp, fpLen, S));
        else
            _fifoPreds = Wrap2DIn3D(chunkPreds, cpLen, S);

        int newFifoT = _fifo.GetLength(1);
        if (newFifoT > Config.FifoLength)
        {
            int popLen = Config.SpeakerCacheUpdatePeriod;
            popLen = Math.Max(popLen, (newFifoT - Config.FifoLength) + newFifoT);
            popLen = Math.Min(popLen, newFifoT);

            var popEmbs  = SliceFront3D(_fifo,     popLen, D);
            var popPreds = SliceFront3D(_fifoPreds, popLen, S);

            UpdateSilenceProfile(popEmbs, popPreds);

            _fifo      = SliceTail3D(_fifo,     popLen, D);
            _fifoPreds = SliceTail3D(_fifoPreds, popLen, S);

            var spkcacheCurrent = _spkcache!;
            _spkcache = Concat3DAxis1(spkcacheCurrent, popEmbs);

            if (_spkcachePreds is null)
                _spkcachePreds = popPreds;
            else
                _spkcachePreds = Concat3DAxis1(_spkcachePreds, popPreds);

            if (_spkcache.GetLength(1) > Config.SpeakerCacheLength)
                CompressCache();
        }

        return chunkPreds;
    }

    // ── Incremental segmentation ──────────────────────────────────────────────

    private (int numPredFrames, float[,] medFiltered) FilterPredsUpTo(
        List<float[,]> allPreds, int upToFrame)
    {
        int S = Config.NumSpeakers;
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
        int half = Config.Window / 2;
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

            double safeTime = (safeFrames - half) * Config.FrameDuration;
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
        int chunkStride = Config.ChunkLength * Config.Subsampling;
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
        int chunkStride = Config.ChunkLength * Config.Subsampling;
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
        int S    = Config.NumSpeakers;
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

        int half     = Config.Window / 2;
        var filtered = new float[totT, S];
        var window   = new float[Config.Window];

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
        int S           = Config.NumSpeakers;
        var allSegments = new List<(double, double, string)>();

        for (int spk = 0; spk < S; spk++)
        {
            bool inSeg    = false;
            int  segStart = 0;
            var  tempSegs = new List<(double start, double end)>();

            for (int t = 0; t < numPredFrames; t++)
            {
                float p = medFiltered[t, spk];
                if (p >= Config.OnsetThreshold && !inSeg)
                {
                    inSeg    = true;
                    segStart = t;
                }
                else if (p < Config.OffsetThreshold && inSeg)
                {
                    inSeg = false;
                    double s = Math.Max(segStart * Config.FrameDuration - Config.PadOnset, 0.0);
                    double e = t * Config.FrameDuration + Config.PadOffset;
                    if (e - s >= Config.MinDurOn) tempSegs.Add((s, e));
                }
            }

            if (inSeg)
            {
                double s = Math.Max(segStart * Config.FrameDuration - Config.PadOnset, 0.0);
                double e = numPredFrames * Config.FrameDuration + Config.PadOffset;
                if (e - s >= Config.MinDurOn) tempSegs.Add((s, e));
            }

            var merged = new List<(double start, double end)>();
            foreach (var seg in tempSegs)
            {
                if (merged.Count == 0)
                    merged.Add(seg);
                else
                {
                    var (ps, pe) = merged[^1];
                    if (seg.start - pe < Config.MinDurOff)
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
        if (length == 0) return 0f;
        var tmp = data[..length];
        Array.Sort(tmp);
        int mid = length / 2;
        return (length % 2 == 0) ? (tmp[mid - 1] + tmp[mid]) / 2f : tmp[mid];
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
        int chunkStride = Config.ChunkLength * Config.Subsampling;
        var dummyMel = new float[1, chunkStride, Config.NMels];

        ProcessChunk(0, chunkStride, chunkStride, dummyMel);
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose() => _session.Dispose();
}
