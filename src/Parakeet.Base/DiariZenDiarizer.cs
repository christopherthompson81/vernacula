using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;

namespace Parakeet.Base;

/// <summary>
/// DiariZen speaker diarization using ONNX Runtime.
///
/// <para><strong>Pipeline (matching pyannote/DiariZen):</strong></para>
/// <list type="number">
/// <item><description>Audio is chunked into 16 s segments with 1.6 s stride (segmentation_step=0.1)</description></item>
/// <item><description>WavLM+Conformer model outputs powerset logits per frame (batch=1)</description></item>
/// <item><description>Per chunk: softmax → argmax → powerset-to-multilabel → binary (T×4)</description></item>
/// <item><description>Per chunk: 11-frame median filter per speaker channel</description></item>
/// <item><description>Per (chunk, local_speaker) with enough active frames: extract WeSpeaker embedding</description></item>
/// <item><description>HAC (centroid linkage, distance threshold) clusters all embeddings globally</description></item>
/// <item><description>Reconstruct timeline: remap local→global speakers, overlap-add, normalise</description></item>
/// <item><description>Per global speaker: binarize at 0.5, extract active regions</description></item>
/// <item><description>Temporal merging combines near-adjacent same-speaker segments</description></item>
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

    /// <summary>
    /// Minimum active frames a local speaker must have in a chunk to warrant
    /// computing an embedding for it. 10 frames ≈ 200 ms at 50 Hz.
    /// </summary>
    private const int MinActiveFramesForEmbed = 10;

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

        // ── Step 1: Per-chunk powerset inference ──────────────────────────
        var chunkScores = SegmentBatched(chunks);

        // ── Step 2: Per-chunk decode → binary multilabel (T × 4) ─────────
        // Matches pyannote's Inference.infer with soft=False + to_multilabel,
        // followed by median_filter(size=(1,11,1)).
        var perChunkMultilabel = new List<bool[,]>(chunks.Count);
        foreach (var scores in chunkScores)
        {
            var probs    = ApplySoftmaxSingle(scores);
            var filtered = ApplyMedianFilterSingle(probs, Config.DiariZenMedianFilterSize);
            perChunkMultilabel.Add(DecodePowersetToMultilabel(filtered));
        }

        // ── Step 3: Extract per-(chunk, local_speaker) embeddings ─────────
        // Matches pyannote's get_embeddings: one embedding per (chunk, speaker) pair.
        var embeddings   = new List<double[]>();
        var embedKeys    = new List<(int chunkIdx, int speakerIdx)>();

        if (_embedder != null)
        {
            for (int ci = 0; ci < perChunkMultilabel.Count; ci++)
            {
                var multilabel = perChunkMultilabel[ci];
                int numFrames  = multilabel.GetLength(0);

                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                {
                    // Count active frames and locate first/last
                    int activeCount = 0, firstActive = -1, lastActive = -1;
                    for (int t = 0; t < numFrames; t++)
                    {
                        if (!multilabel[t, spk]) continue;
                        activeCount++;
                        if (firstActive < 0) firstActive = t;
                        lastActive = t;
                    }

                    if (activeCount < MinActiveFramesForEmbed) continue;

                    // Use the span from first to last active frame as the audio window
                    int chunkStartSample = (int)(startTimes[ci] * SampleRate);
                    int startSample = Math.Clamp(chunkStartSample + firstActive * SampleRate / FrameRate, 0, audio.Length);
                    int endSample   = Math.Clamp(chunkStartSample + (lastActive + 1) * SampleRate / FrameRate, 0, audio.Length);

                    float[] raw = _embedder.ComputeEmbedding(audio, startSample, endSample);
                    var emb = new double[raw.Length];
                    for (int d = 0; d < raw.Length; d++) emb[d] = raw[d];

                    embeddings.Add(emb);
                    embedKeys.Add((ci, spk));
                }
            }
        }

        // ── Step 4: Cluster embeddings ────────────────────────────────────
        // Build (chunkIdx, speakerIdx) → global speaker ID lookup.
        var localToGlobal = new Dictionary<(int, int), int>();

        if (embeddings.Count > 0 && _embedder != null)
        {
            int[] clusterIds;
            if (embeddings.Count == 1)
            {
                clusterIds = [0];
            }
            else
            {
                var linkage = HierarchicalClustering.Linkage(
                    embeddings.ToArray(), Config.DiariZenClusteringMethod);
                clusterIds = HierarchicalClustering.FclusterThreshold(
                    linkage, Config.DiariZenAhcThreshold);
            }

            for (int i = 0; i < embedKeys.Count; i++)
                localToGlobal[embedKeys[i]] = clusterIds[i];
        }
        else
        {
            // No embedder — assign all local speakers in each chunk to
            // their local index (will produce up to 4 "speakers")
            for (int ci = 0; ci < perChunkMultilabel.Count; ci++)
                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                    localToGlobal[(ci, spk)] = spk;
        }

        int numGlobalSpeakers = localToGlobal.Count > 0
            ? localToGlobal.Values.Max() + 1
            : MaxUniqueSpeakers;

        // ── Step 5: Reconstruct global timeline ───────────────────────────
        // For each chunk frame, map active local speakers → global speakers
        // and accumulate into a global (totalFrames × numGlobalSpeakers) array.
        // Normalise by overlap count to get a probability in [0,1].
        int totalFrames = (int)(audio.Length / (double)SampleRate * FrameRate);
        var accumulator  = new float[totalFrames * numGlobalSpeakers];
        var overlapCount = new int[totalFrames];

        for (int ci = 0; ci < perChunkMultilabel.Count; ci++)
        {
            var multilabel = perChunkMultilabel[ci];
            int numFrames  = multilabel.GetLength(0);
            int startFrame = (int)(startTimes[ci] * FrameRate);

            for (int t = 0; t < numFrames; t++)
            {
                int gf = startFrame + t;
                if (gf >= totalFrames) break;

                overlapCount[gf]++;

                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                {
                    if (!multilabel[t, spk]) continue;
                    if (!localToGlobal.TryGetValue((ci, spk), out int gs)) continue;
                    accumulator[gf * numGlobalSpeakers + gs] += 1f;
                }
            }
        }

        // Normalise
        for (int f = 0; f < totalFrames; f++)
        {
            if (overlapCount[f] <= 0) continue;
            for (int gs = 0; gs < numGlobalSpeakers; gs++)
                accumulator[f * numGlobalSpeakers + gs] /= overlapCount[f];
        }

        // ── Step 6: Binarise per speaker and extract regions ──────────────
        var labeled = new List<(int startFrame, int endFrame, string speaker)>();

        for (int gs = 0; gs < numGlobalSpeakers; gs++)
        {
            int? regionStart = null;
            for (int f = 0; f < totalFrames; f++)
            {
                bool active = accumulator[f * numGlobalSpeakers + gs] >= 0.5f;

                if (active && regionStart == null)
                    regionStart = f;
                else if (!active && regionStart.HasValue)
                {
                    labeled.Add((regionStart.Value, f, $"speaker_{gs}"));
                    regionStart = null;
                }
            }
            if (regionStart.HasValue)
                labeled.Add((regionStart.Value, totalFrames, $"speaker_{gs}"));
        }

        if (labeled.Count == 0) return [];

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

    // ── Per-chunk softmax ──────────────────────────────────────────────────

    private static float[] ApplySoftmaxSingle(float[] logits)
    {
        int numFrames = logits.Length / NumPowersetClasses;
        var probs     = new float[logits.Length];

        for (int f = 0; f < numFrames; f++)
        {
            int b = f * NumPowersetClasses;
            float maxVal = logits[b];
            for (int c = 1; c < NumPowersetClasses; c++)
                if (logits[b + c] > maxVal) maxVal = logits[b + c];

            double sum = 0;
            for (int c = 0; c < NumPowersetClasses; c++)
            {
                probs[b + c] = (float)Math.Exp(logits[b + c] - maxVal);
                sum += probs[b + c];
            }
            for (int c = 0; c < NumPowersetClasses; c++)
                probs[b + c] /= (float)sum;
        }

        return probs;
    }

    // ── Per-chunk median filter (per powerset class, within chunk) ─────────

    private static float[] ApplyMedianFilterSingle(float[] probs, int windowSize)
    {
        if (windowSize <= 1) return probs;

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
                    // "reflect" padding
                    if (fk < 0)           fk = -fk;
                    if (fk >= numFrames)  fk = 2 * numFrames - 2 - fk;
                    fk = Math.Clamp(fk, 0, numFrames - 1);
                    window[count++] = probs[fk * NumPowersetClasses + c];
                }
                Array.Sort(window, 0, count);
                filtered[f * NumPowersetClasses + c] = window[count / 2];
            }
        }

        return filtered;
    }

    // ── Powerset → per-speaker binary multilabel ──────────────────────────

    /// <summary>
    /// Replicates pyannote's Powerset.to_multilabel(soft=False):
    /// argmax over powerset classes → one-hot → mapping matrix multiply.
    /// Returns bool[numFrames, MaxUniqueSpeakers].
    /// </summary>
    private static bool[,] DecodePowersetToMultilabel(float[] probs)
    {
        var combinations = PowersetDecoder.GetPowersetCombinations(
            MaxUniqueSpeakers, MaxSimultaneousPerFrame);
        int numFrames    = probs.Length / NumPowersetClasses;
        var result       = new bool[numFrames, MaxUniqueSpeakers];

        for (int f = 0; f < numFrames; f++)
        {
            // Argmax over powerset classes for this frame
            int bestClass = 0;
            float bestProb = probs[f * NumPowersetClasses];
            for (int c = 1; c < NumPowersetClasses; c++)
            {
                float p = probs[f * NumPowersetClasses + c];
                if (p > bestProb) { bestProb = p; bestClass = c; }
            }

            // Map to active speakers
            foreach (int spk in combinations[bestClass])
                result[f, spk] = true;
        }

        return result;
    }

    // ── Temporal merging ───────────────────────────────────────────────────

    private static List<DiarizationSegment> MergeAdjacentSegments(
        List<(int startFrame, int endFrame, string speakerLabel)> labeled)
    {
        if (labeled.Count == 0) return [];

        const double mergeGap = 0.0;
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
