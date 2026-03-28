// Adapted from Sortformer.cs in the main project.
// Key changes:
//   - EP selected at build time via #if DIRECTML (same as Program.cs)
//   - ResetState() allows reusing the loaded ONNX session across benchmark runs
//   - Uses BenchmarkAudioUtils.LogMelSpectrogram instead of AudioUtils.LogMelSpectrogram

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

internal sealed class BenchmarkSortformer : IDisposable
{
    // ── Config constants (mirrors Config.cs) ──────────────────────────────────

    private const string ModelFile               = "diar_streaming_sortformer_4spk-v2.1.onnx";
    private const int    EmbeddingDimension      = 512;
    private const int    NumSpeakers             = 4;
    private const int    SpeakerCacheLength      = 188;
    private const int    SpeakerCacheUpdatePeriod = 124;
    private const int    FifoLength              = 124;
    private const int    ChunkLength             = 124;
    private const int    Subsampling             = 8;
    private const float  SilThreshold            = 0.2f;
    private const int    Window                  = 11;
    private const float  OnsetThreshold          = 0.641f;
    private const float  OffsetThreshold         = 0.561f;
    private const double PadOnset                = 0.229;
    private const double PadOffset               = 0.079;
    private const double MinDurOn                = 0.511;
    private const double MinDurOff               = 0.296;
    private const double FrameDuration           = 0.08;

    // ── ONNX session (loaded once) ────────────────────────────────────────────

    private readonly InferenceSession _session;

    // ── Streaming state (reset between benchmark runs via ResetState()) ───────

    private float[,,]  _spkcache      = null!;
    private float[,,]? _spkcachePreds;
    private float[,,]  _fifo          = null!;
    private float[,,]  _fifoPreds     = null!;
    private float[]    _meanSilEmb    = null!;
    private int        _nSilFrames;

    // ── Construction ─────────────────────────────────────────────────────────

    public BenchmarkSortformer(string modelPath)
    {
        var opts = new SessionOptions();
#if DIRECTML
        try { opts.AppendExecutionProvider_DML(0); }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: Sortformer DirectML EP unavailable ({ex.Message}), falling back to CPU.");
        }
#elif CPU
        // CPU-only: no EP appended
#else
        try { opts.AppendExecutionProvider_CUDA(0); }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: Sortformer CUDA EP unavailable ({ex.Message}), falling back to CPU.");
        }
#endif
        _session = new InferenceSession(Path.Combine(modelPath, ModelFile), opts);
        ResetState();
    }

    /// <summary>
    /// Reset streaming state to allow re-running diarization in a subsequent benchmark run
    /// without reloading the ONNX model.
    /// </summary>
    public void ResetState()
    {
        _spkcache      = new float[1, 0, EmbeddingDimension];
        _spkcachePreds = null;
        _fifo          = new float[1, 0, EmbeddingDimension];
        _fifoPreds     = new float[1, 0, NumSpeakers];
        _meanSilEmb    = new float[EmbeddingDimension];
        _nSilFrames    = 0;
    }

    // ── Silence profile ───────────────────────────────────────────────────────

    private void UpdateSilenceProfile(float[,,] embs, float[,,] preds)
    {
        int T = embs.GetLength(1);
        for (int t = 0; t < T; t++)
        {
            float probSum = 0f;
            for (int s = 0; s < NumSpeakers; s++) probSum += preds[0, t, s];

            if (probSum < SilThreshold)
            {
                _nSilFrames++;
                for (int d = 0; d < EmbeddingDimension; d++)
                {
                    double oldSum = _meanSilEmb[d] * (_nSilFrames - 1);
                    _meanSilEmb[d] = (float)((oldSum + embs[0, t, d]) / _nSilFrames);
                }
            }
        }
    }

    // ── Quality scoring ───────────────────────────────────────────────────────

    private float[,] SpeakerQualityScores(float[,] preds2d, int minPosPerSpk)
    {
        int T      = preds2d.GetLength(0);
        int S      = NumSpeakers;
        var scores = new float[T, S];

        for (int t = 0; t < T; t++)
        {
            float logOneSum = 0f;
            for (int s = 0; s < S; s++)
            {
                float p  = preds2d[t, s];
                float lp = (float)Math.Log(Math.Max(p,       0.25f));
                float lo = (float)Math.Log(Math.Max(1f - p,  0.25f));
                scores[t, s] = lp - lo;
                logOneSum += lo;
            }
            float adj = logOneSum - (float)Math.Log(Math.Sqrt(2));
            for (int s = 0; s < S; s++) scores[t, s] += adj;
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
        int T       = scores.GetLength(0);
        int S       = NumSpeakers;
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

        int T = _spkcache.GetLength(1);
        int S = NumSpeakers;

        var preds2d = Slice3DTo2D(_spkcachePreds, T, S);

        int cachePerSpk       = SpeakerCacheLength / S - 3;
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

        int keep     = SpeakerCacheLength;
        var selected = flat[..keep].OrderBy(x => x.sIdx).ThenBy(x => x.tIdx).ToArray();

        var newEmbs  = new float[1, keep, EmbeddingDimension];
        var newPreds = new float[1, keep, S];

        for (int i = 0; i < keep; i++)
        {
            int t = selected[i].tIdx;
            if (t >= T)
            {
                for (int d = 0; d < EmbeddingDimension; d++)
                    newEmbs[0, i, d] = _meanSilEmb[d];
                continue;
            }
            for (int d = 0; d < EmbeddingDimension; d++)
                newEmbs[0, i, d] = _spkcache[0, t, d];
            for (int s = 0; s < S; s++)
                newPreds[0, i, s] = _spkcachePreds![0, t, s];
        }

        _spkcache      = newEmbs;
        _spkcachePreds = newPreds;
    }

    // ── Chunk processing ──────────────────────────────────────────────────────

    private float[,] ProcessChunk(int idx, int chunkStride, int totalFrames, float[] audio)
    {
        int start      = idx * chunkStride;
        int end        = Math.Min(start + chunkStride, totalFrames);
        int currentLen = end - start;
        int S          = NumSpeakers;
        int D          = EmbeddingDimension;

        int halfNFft    = BenchmarkAudioUtils.NFft / 2;
        int sampleStart = Math.Max(0, start * BenchmarkAudioUtils.HopLength - halfNFft);
        int sampleEnd   = Math.Min(audio.Length, end * BenchmarkAudioUtils.HopLength + halfNFft);

        int sliceLen   = sampleEnd - sampleStart;
        var audioSlice = new float[sliceLen];
        Array.Copy(audio, sampleStart, audioSlice, 0, sliceLen);

        var sliceSpec = BenchmarkAudioUtils.LogMelSpectrogram(audioSlice);
        int tSlice    = sliceSpec.GetLength(1);

        int frameOffset     = start > 0 ? halfNFft / BenchmarkAudioUtils.HopLength : 0;
        int availableFrames = tSlice - frameOffset;

        var chunkData = new float[chunkStride * BenchmarkAudioUtils.NMels];
        int copyLen   = Math.Min(currentLen, availableFrames);
        for (int t = 0; t < copyLen; t++)
        {
            int srcRow = frameOffset + t;
            if (srcRow < tSlice)
                for (int m = 0; m < BenchmarkAudioUtils.NMels; m++)
                    chunkData[t * BenchmarkAudioUtils.NMels + m] = sliceSpec[0, srcRow, m];
        }

        int cacheT      = _spkcache.GetLength(1);
        int fifoT       = _fifo.GetLength(1);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("chunk",
                new DenseTensor<float>(chunkData, new[] { 1, chunkStride, BenchmarkAudioUtils.NMels })),
            NamedOnnxValue.CreateFromTensor("chunk_lengths",
                new DenseTensor<long>(new long[] { currentLen }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("spkcache",
                new DenseTensor<float>(Flatten3D(_spkcache, 1, cacheT, D), new[] { 1, cacheT, D })),
            NamedOnnxValue.CreateFromTensor("spkcache_lengths",
                new DenseTensor<long>(new long[] { cacheT }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("fifo",
                new DenseTensor<float>(Flatten3D(_fifo, 1, fifoT, D), new[] { 1, fifoT, D })),
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
        int validFrames = (currentLen + Subsampling - 1) / Subsampling;

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

        _fifo = Concat3DAxis1(_fifo, Wrap2DIn3D(chunkEmbs, ceLen, D));

        if (fpLen > 0)
            _fifoPreds = Concat3DAxis1(_fifoPreds, Wrap2DIn3D(fp, fpLen, S));
        else
            _fifoPreds = Wrap2DIn3D(chunkPreds, cpLen, S);

        int newFifoT = _fifo.GetLength(1);
        if (newFifoT > FifoLength)
        {
            int popLen = SpeakerCacheUpdatePeriod;
            popLen = Math.Max(popLen, (newFifoT - FifoLength) + newFifoT);
            popLen = Math.Min(popLen, newFifoT);

            var popEmbs  = SliceFront3D(_fifo,      popLen, D);
            var popPreds = SliceFront3D(_fifoPreds,  popLen, S);

            UpdateSilenceProfile(popEmbs, popPreds);

            _fifo      = SliceTail3D(_fifo,      popLen, D);
            _fifoPreds = SliceTail3D(_fifoPreds,  popLen, S);

            _spkcache = Concat3DAxis1(_spkcache, popEmbs);

            if (_spkcachePreds is null)
                _spkcachePreds = popPreds;
            else
                _spkcachePreds = Concat3DAxis1(_spkcachePreds, popPreds);

            if (_spkcache.GetLength(1) > SpeakerCacheLength)
                CompressCache();
        }

        return chunkPreds;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public (int totalFrames, int chunkStride, int numChunks) GetPredParams(float[] audio)
    {
        int paddedLen   = audio.Length + BenchmarkAudioUtils.NFft;
        int totalFrames = (paddedLen - BenchmarkAudioUtils.NFft) / BenchmarkAudioUtils.HopLength + 1;
        int chunkStride = ChunkLength * Subsampling;
        int numChunks   = (totalFrames + chunkStride - 1) / chunkStride;
        return (totalFrames, chunkStride, numChunks);
    }

    public List<(double start, double end, string spkId)> Diarize(float[] audio)
    {
        var (totalFrames, chunkStride, numChunks) = GetPredParams(audio);
        var allPreds = new List<float[,]>(numChunks);

        for (int idx = 0; idx < numChunks; idx++)
            allPreds.Add(ProcessChunk(idx, chunkStride, totalFrames, audio));

        var (numPredFrames, medFiltered) = FilterPreds(allPreds, totalFrames);
        return BinarizePredToSegments(numPredFrames, medFiltered);
    }

    private (int numPredFrames, float[,] medFiltered) FilterPreds(
        IReadOnlyList<float[,]> allPreds, int totalFrames)
    {
        int S    = NumSpeakers;
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

        int half     = Window / 2;
        var filtered = new float[totT, S];
        var window   = new float[Window];

        for (int spk = 0; spk < S; spk++)
            for (int t = 0; t < totT; t++)
            {
                int start = Math.Max(t - half, 0);
                int end   = Math.Min(t + half + 1, totT);
                int wlen  = end - start;
                for (int i = 0; i < wlen; i++) window[i] = preds[start + i, spk];
                filtered[t, spk] = Median(window, wlen);
            }

        return (totT, filtered);
    }

    private List<(double start, double end, string spkId)>
        BinarizePredToSegments(int numPredFrames, float[,] medFiltered)
    {
        int S           = NumSpeakers;
        var allSegments = new List<(double, double, string)>();

        for (int spk = 0; spk < S; spk++)
        {
            bool inSeg    = false;
            int  segStart = 0;
            var  tempSegs = new List<(double start, double end)>();

            for (int t = 0; t < numPredFrames; t++)
            {
                float p = medFiltered[t, spk];
                if (p >= OnsetThreshold && !inSeg)
                {
                    inSeg    = true;
                    segStart = t;
                }
                else if (p < OffsetThreshold && inSeg)
                {
                    inSeg = false;
                    double s = Math.Max(segStart * FrameDuration - PadOnset, 0.0);
                    double e = t * FrameDuration + PadOffset;
                    if (e - s >= MinDurOn) tempSegs.Add((s, e));
                }
            }

            if (inSeg)
            {
                double s = Math.Max(segStart * FrameDuration - PadOnset, 0.0);
                double e = numPredFrames * FrameDuration + PadOffset;
                if (e - s >= MinDurOn) tempSegs.Add((s, e));
            }

            var merged = new List<(double start, double end)>();
            foreach (var seg in tempSegs)
            {
                if (merged.Count == 0)
                    merged.Add(seg);
                else
                {
                    var (ps, pe) = merged[^1];
                    if (seg.start - pe < MinDurOff)
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

    // ── Array utilities ───────────────────────────────────────────────────────

    private static float[] Flatten3D(float[,,] a, int d0, int d1, int d2)
    {
        var f = new float[d0 * d1 * d2];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    f[i * d1 * d2 + j * d2 + k] = a[i, j, k];
        return f;
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

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose() => _session.Dispose();
}
