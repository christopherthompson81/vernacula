using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;

namespace Parakeet.Base;

/// <summary>
/// DiariZen speaker diarization using ONNX Runtime.
///
/// <para><strong>Pipeline:</strong></para>
/// <list type="number">
/// <item><description>Audio is chunked into 16 s segments with 1.6 s stride (segmentation_step=0.1, matching Python)</description></item>
/// <item><description>WavLM+Conformer model outputs powerset probabilities per frame (batch=32)</description></item>
/// <item><description>Chunk predictions are aggregated via overlap-add and softmax</description></item>
/// <item><description>Median filter (7 frames) smooths frame-level noise</description></item>
/// <item><description>Thresholding extracts speaker-active regions</description></item>
/// <item><description>WeSpeaker-ResNet34 extracts 512-dim speaker embeddings per region</description></item>
/// <item><description>HAC (centroid, distance threshold 0.9) clusters regions into speakers</description></item>
/// <item><description>Temporal merging combines near-adjacent same-speaker segments</description></item>
/// </list>
///
/// <para><strong>Segmentation parameters:</strong></para>
/// <list type="bullet">
/// <item><description>Chunk duration: 16 s, stride: 1.6 s (10% step), 90% overlap</description></item>
/// <item><description>Frame rate: 50 Hz (20 ms frames), powerset classes: 11</description></item>
/// </list>
/// </summary>
public sealed class DiariZenDiarizer : IDisposable
{
    private readonly InferenceSession _segSession;
    private readonly WeSpeakerEmbedder? _embedder;
    private bool _disposed;

    // ── Model constants ────────────────────────────────────────────────────

    public const int    ChunkDurationSeconds  = 16;
    public const double SegmentationStep      = Config.DiariZenSegmentationStep; // 0.1
    public const int    SampleRate            = 16_000;
    public const int    FrameRate             = 50;
    public const int    NumPowersetClasses    = 11;
    public const int    MaxUniqueSpeakers     = 4;
    public const int    MaxSimultaneousPerFrame = 2;
    public const int    InferenceBatchSize    = 32;

    // ── Construction ───────────────────────────────────────────────────────

    public DiariZenDiarizer(
        string segmentationModelPath,
        string? embeddingModelPath = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = new SessionOptions();
        opts.IntraOpNumThreads = 4;

        switch (ep)
        {
            case ExecutionProvider.Auto:
                try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                try { opts.AppendExecutionProvider_DML(0); }  catch { }
                break;
            case ExecutionProvider.Cuda:
                try { opts.AppendExecutionProvider_CUDA(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("CUDA EP not available."); }
                break;
            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (EntryPointNotFoundException)
                { throw new InvalidOperationException("DirectML EP not available."); }
                break;
        }

        _segSession = new InferenceSession(segmentationModelPath, opts);

        if (embeddingModelPath != null && File.Exists(embeddingModelPath))
            _embedder = new WeSpeakerEmbedder(embeddingModelPath, ep);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>
    /// Perform speaker diarization on a 16 kHz mono waveform.
    /// </summary>
    public List<DiarizationSegment> Diarize(
        float[] audio,
        int   minSpeakers = 1,
        int   maxSpeakers = 8,
        float threshold   = 0.5f)
    {
        if (audio.Length == 0)
            return [];

        var (chunks, startTimes) = ChunkAudio(audio);
        var chunkScores          = SegmentBatched(chunks);

        var aggregated   = AggregateSegmentations(chunkScores, startTimes, audio.Length);
        var probabilities = ApplySoftmax(aggregated);
        var filtered      = ApplyMedianFilter(probabilities, Config.DiariZenMedianFilterSize);

        var regions = ExtractSpeakerRegions(filtered, threshold);
        if (regions.Count == 0) return [];

        var labeled = ClusterRegions(audio, regions, minSpeakers, maxSpeakers);
        return MergeAdjacentSegments(labeled);
    }

    // ── Chunking ───────────────────────────────────────────────────────────

    private (List<float[]> chunks, List<double> startTimes) ChunkAudio(float[] audio)
    {
        int chunkSamples  = ChunkDurationSeconds * SampleRate;
        int strideSamples = (int)(chunkSamples * SegmentationStep);  // 25 600 = 1.6 s

        var chunks     = new List<float[]>();
        var startTimes = new List<double>();

        for (int start = 0; start < audio.Length; start += strideSamples)
        {
            var chunk = new float[chunkSamples];
            int end   = Math.Min(start + chunkSamples, audio.Length);
            Array.Copy(audio, start, chunk, 0, end - start);

            chunks.Add(chunk);
            startTimes.Add(start / (double)SampleRate);
        }

        return (chunks, startTimes);
    }

    // ── Segmentation inference ─────────────────────────────────────────────

    private List<float[]> SegmentBatched(List<float[]> chunks)
    {
        // The exported ONNX model uses batch=1 (WavLM attention is shape-static).
        var results = new List<float[]>(chunks.Count);
        foreach (var chunk in chunks)
            results.Add(SegmentSingle(chunk));
        return results;
    }

    private float[] SegmentSingle(float[] chunk)
    {
        var tensor = new DenseTensor<float>(chunk, new[] { 1, 1, chunk.Length });
        var inputs = new List<NamedOnnxValue>
            { NamedOnnxValue.CreateFromTensor("waveform", tensor) };

        using var onnxOut = _segSession.Run(inputs);
        var outTensor     = onnxOut.First(r => r.Name == "scores").AsTensor<float>();

        int frames  = outTensor.Dimensions[1];
        int classes = outTensor.Dimensions[2];
        var scores  = new float[frames * classes];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < classes; c++)
                scores[f * classes + c] = outTensor[0, f, c];
        return scores;
    }

    // ── Aggregation ────────────────────────────────────────────────────────

    private float[] AggregateSegmentations(
        List<float[]> chunkScores,
        List<double>  startTimes,
        int           audioLength)
    {
        if (chunkScores.Count == 0) return [];

        int totalFrames    = (int)(audioLength / (double)SampleRate * FrameRate);
        var aggregated     = new float[totalFrames * NumPowersetClasses];
        var counts         = new int[totalFrames];
        int framesPerChunk = chunkScores[0].Length / NumPowersetClasses;

        for (int ci = 0; ci < chunkScores.Count; ci++)
        {
            int startFrame = (int)(startTimes[ci] * FrameRate);
            var scores     = chunkScores[ci];

            for (int t = 0; t < framesPerChunk; t++)
            {
                int gf = startFrame + t;
                if (gf >= totalFrames) break;

                int gIdx = gf * NumPowersetClasses;
                int lIdx = t  * NumPowersetClasses;
                for (int c = 0; c < NumPowersetClasses; c++)
                    aggregated[gIdx + c] += scores[lIdx + c];
                counts[gf]++;
            }
        }

        // Overlap-add normalisation
        for (int f = 0; f < totalFrames; f++)
        {
            if (counts[f] <= 0) continue;
            int gIdx = f * NumPowersetClasses;
            for (int c = 0; c < NumPowersetClasses; c++)
                aggregated[gIdx + c] /= counts[f];
        }

        return aggregated;
    }

    // ── Softmax ────────────────────────────────────────────────────────────

    private float[] ApplySoftmax(float[] logits)
    {
        if (logits.Length == 0) return logits;

        int numFrames = logits.Length / NumPowersetClasses;
        var probs     = new float[logits.Length];

        for (int f = 0; f < numFrames; f++)
        {
            int b = f * NumPowersetClasses;

            float maxVal = logits[b];
            for (int c = 1; c < NumPowersetClasses; c++)
                maxVal = Math.Max(maxVal, logits[b + c]);

            double sum = 0;
            var exps = new float[NumPowersetClasses];
            for (int c = 0; c < NumPowersetClasses; c++)
            {
                exps[c] = (float)Math.Exp(logits[b + c] - maxVal);
                sum += exps[c];
            }

            for (int c = 0; c < NumPowersetClasses; c++)
                probs[b + c] = exps[c] / (float)sum;
        }

        return probs;
    }

    // ── Median filtering ───────────────────────────────────────────────────

    /// <summary>
    /// Apply a 1-D median filter of the given window size to each powerset class
    /// independently across the time axis.  This matches pyannote's
    /// apply_median_filtering=True and smooths brief spurious activations.
    /// </summary>
    private static float[] ApplyMedianFilter(float[] probs, int windowSize)
    {
        if (windowSize <= 1 || probs.Length == 0) return probs;

        int numFrames = probs.Length / NumPowersetClasses;
        var filtered  = new float[probs.Length];
        var window    = new float[windowSize];
        int half      = windowSize / 2;

        for (int c = 0; c < NumPowersetClasses; c++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int count = 0;
                for (int k = -half; k <= half; k++)
                {
                    int fk = f + k;
                    if (fk >= 0 && fk < numFrames)
                        window[count++] = probs[fk * NumPowersetClasses + c];
                }
                Array.Sort(window, 0, count);
                filtered[f * NumPowersetClasses + c] = window[count / 2];
            }
        }

        return filtered;
    }

    // ── Region extraction ──────────────────────────────────────────────────

    private List<(int startFrame, int endFrame)> ExtractSpeakerRegions(
        float[] powersetProbs,
        float   threshold)
    {
        var activeSpeakers = PowersetDecoder.BinarizePowerset(
            powersetProbs, MaxUniqueSpeakers, MaxSimultaneousPerFrame, threshold);
        int numFrames      = activeSpeakers.Length;

        const int minDurationFrames = 25; // ~0.5 s at 50 Hz
        var allRegions     = new List<(int, int)>();

        for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
        {
            int? regionStart = null;

            for (int t = 0; t < numFrames; t++)
            {
                bool active = activeSpeakers[t].Contains(spk);

                if (active && regionStart == null)
                    regionStart = t;
                else if (!active && regionStart.HasValue)
                {
                    if (t - regionStart.Value >= minDurationFrames)
                        allRegions.Add((regionStart.Value, t));
                    regionStart = null;
                }
            }

            if (regionStart.HasValue && numFrames - regionStart.Value >= minDurationFrames)
                allRegions.Add((regionStart.Value, numFrames));
        }

        return allRegions.OrderBy(r => r.Item1).ToList();
    }

    // ── Clustering ─────────────────────────────────────────────────────────

    private List<(int startFrame, int endFrame, string speakerLabel)> ClusterRegions(
        float[]                        audio,
        List<(int startFrame, int endFrame)> regions,
        int                            minSpeakers,
        int                            maxSpeakers)
    {
        if (regions.Count == 0) return [];

        if (regions.Count == 1)
        {
            var (s, e) = regions[0];
            return [(s, e, "speaker_0")];
        }

        double[][] features = _embedder != null
            ? ExtractWeSpeakerEmbeddings(audio, regions)
            : ExtractStatisticalFeatures(audio, regions);

        var linkage = HierarchicalClustering.Linkage(features, Config.DiariZenClusteringMethod);

        int[] labels;
        if (_embedder != null)
        {
            // Threshold-based: let the data decide the number of speakers
            labels = HierarchicalClustering.FclusterThreshold(linkage, Config.DiariZenAhcThreshold);
        }
        else
        {
            // Fallback without embeddings: force at least 2 clusters
            labels = HierarchicalClustering.FclusterMaxClust(linkage, Math.Max(minSpeakers, 2));
        }

        // Enforce minSpeakers by splitting the largest cluster if needed
        // (rarely needed; just ensures we don't under-segment)
        // For now, trust the threshold result.

        var labeled = new List<(int, int, string)>(regions.Count);
        for (int i = 0; i < regions.Count; i++)
        {
            var (start, end) = regions[i];
            labeled.Add((start, end, $"speaker_{labels[i]}"));
        }

        return labeled;
    }

    // ── WeSpeaker embedding extraction ─────────────────────────────────────

    private double[][] ExtractWeSpeakerEmbeddings(
        float[] audio,
        List<(int startFrame, int endFrame)> regions)
    {
        var embeddings = new double[regions.Count][];

        for (int i = 0; i < regions.Count; i++)
        {
            var (sf, ef) = regions[i];
            int startSample = (int)((long)sf * SampleRate / FrameRate);
            int endSample   = (int)((long)ef * SampleRate / FrameRate);

            startSample = Math.Clamp(startSample, 0, audio.Length);
            endSample   = Math.Clamp(endSample,   0, audio.Length);

            float[] raw = _embedder!.ComputeEmbedding(audio, startSample, endSample);
            embeddings[i] = new double[raw.Length];
            for (int d = 0; d < raw.Length; d++)
                embeddings[i][d] = raw[d];
        }

        return embeddings;
    }

    // ── Statistical feature fallback ───────────────────────────────────────

    private static double[][] ExtractStatisticalFeatures(
        float[] audio,
        List<(int startFrame, int endFrame)> regions)
    {
        const int dim = 10;
        var features  = new double[regions.Count][];

        for (int i = 0; i < regions.Count; i++)
        {
            var (sf, ef)    = regions[i];
            int startSample = Math.Min((int)((long)sf * WeSpeakerEmbedder.SampleRate / FrameRate), audio.Length - 1);
            int endSample   = Math.Min((int)((long)ef * WeSpeakerEmbedder.SampleRate / FrameRate), audio.Length);
            int regionLen   = endSample - startSample;

            if (regionLen <= 0)
            {
                features[i] = new double[dim];
                continue;
            }

            double sum = 0, sumSq = 0;
            int zeroCrossings = 0;
            float prev = audio[startSample];
            sum += prev; sumSq += prev * prev;

            for (int j = startSample + 1; j < endSample; j++)
            {
                float curr = audio[j];
                sum   += curr;
                sumSq += curr * curr;
                if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) zeroCrossings++;
                prev = curr;
            }

            double mean   = sum / regionLen;
            double std    = Math.Sqrt(Math.Max(0, sumSq / regionLen - mean * mean));
            double energy = sumSq / regionLen;

            features[i] = new double[dim]
            {
                mean,
                std,
                zeroCrossings / (double)regionLen,
                Math.Log(energy + 1e-10),
                (ef - sf) / (double)FrameRate,
                sf / (double)(audio.Length * 1.0 / FrameRate),
                Math.Abs(mean) / (std + 1e-10),
                zeroCrossings / (std + 1e-10),
                regionLen / (double)WeSpeakerEmbedder.SampleRate,
                Math.Sqrt(energy),
            };
        }

        return features;
    }

    // ── Temporal merging ───────────────────────────────────────────────────

    private static List<DiarizationSegment> MergeAdjacentSegments(
        List<(int startFrame, int endFrame, string speakerLabel)> labeled)
    {
        if (labeled.Count == 0) return [];

        const double mergeGap = 0.5;
        var sorted = labeled.OrderBy(r => r.startFrame).ToList();
        var merged = new List<DiarizationSegment>();

        double curStart   = sorted[0].startFrame / (double)FrameRate;
        double curEnd     = sorted[0].endFrame   / (double)FrameRate;
        string curSpeaker = sorted[0].speakerLabel;

        for (int i = 1; i < sorted.Count; i++)
        {
            var (sf, ef, spk) = sorted[i];
            double startT = sf / (double)FrameRate;
            double endT   = ef / (double)FrameRate;

            if (spk == curSpeaker && startT - curEnd <= mergeGap)
                curEnd = Math.Max(curEnd, endT);
            else
            {
                merged.Add(new DiarizationSegment(curStart, curEnd, curSpeaker));
                curStart   = startT;
                curEnd     = endT;
                curSpeaker = spk;
            }
        }

        merged.Add(new DiarizationSegment(curStart, curEnd, curSpeaker));
        return merged;
    }

    // ── IDisposable ────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed)
        {
            _segSession?.Dispose();
            _embedder?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

public sealed class DiarizationSegment
{
    public double Start   { get; }
    public double End     { get; }
    public string Speaker { get; }
    public double Duration => End - Start;

    public DiarizationSegment(double start, double end, string speaker)
    {
        Start   = start;
        End     = end;
        Speaker = speaker;
    }

    public override string ToString() =>
        $"{Start:F2}s - {End:F2}: {Speaker} ({Duration:F2}s)";
}
