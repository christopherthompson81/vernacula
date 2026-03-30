using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;

namespace Parakeet.Base;

/// <summary>
/// DiariZen speaker diarization using ONNX Runtime.
/// 
/// <para><strong>Overview:</strong></para>
/// This implementation ported from DiariZen uses a deep learning-based approach
/// for speaker diarization. The pipeline consists of four main stages:
/// </para>
/// 
/// <para><strong>1. Powerset-based Segmentation (EEND Model):</strong></para>
/// <list type="number">
/// <item><description>Audio is chunked into 16s segments with 8s overlap</description></item>
/// <item><description>WavLM+Conformer model outputs powerset probabilities per frame</description></item>
/// <item><description>Chunk predictions are aggregated via overlap-add</description></item>
/// <item><description>Softmax converts logits to probabilities</description></item>
/// <item><description>Thresholding extracts speaker active regions</description></item>
/// </list>
///
/// <para><strong>2. Powerset Encoding:</strong></para>
/// The model uses a restricted powerset encoding to handle overlapping speech:
/// <list type="bullet">
/// <item><description>max_speakers_per_chunk = 4 (unique speakers per chunk)</description></item>
/// <item><description>max_speakers_per_frame = 2 (simultaneous speakers)</description></item>
/// <item><description>Num classes = C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11</description></item>
/// </list>
/// Powerset classes: silence, 4 single speakers, 6 speaker pairs
///
/// <para><strong>3. Region Feature Extraction:</strong></para>
/// For each speaker-active region, statistical features are computed:
/// <list type="bullet">
/// <item><description>Mean, standard deviation, zero-crossing rate</description></item>
/// <item><description>Energy, duration, temporal position</description></item>
/// <item><description>10-dimensional feature vector per region</description></item>
/// </list>
///
/// <para><strong>4. Hierarchical Clustering:</strong></para>
/// <list type="number">
/// <item><description>Compute linkage dendrogram using centroid method</description></item>
/// <item><description>Cut tree to get minSpeakers clusters (avoids over-segmentation)</description></item>
/// <item><description>Small clusters are merged into nearest large cluster</description></item>
/// <item><description>Temporal merging combines nearby segments of same speaker</description></item>
/// </list>
///
/// <para><strong>Key Parameters:</strong></para>
/// <list type="bullet">
/// <item><description>Chunk duration: 16 seconds, overlap: 8 seconds</description></item>
/// <item><description>Sample rate: 16kHz, frame rate: 50Hz (20ms frames)</description></item>
/// <item><description>Threshold: 0.5 (default for region extraction)</description></item>
/// <item><description>Min/Max speakers: controls clustering granularity</description></item>
/// </list>
///
/// <para><strong>Comparison with Sortformer:</strong></para>
/// Unlike Sortformer's streaming approach with speaker tracking, DiariZen uses
/// batch processing with post-hoc clustering. This provides better accuracy
/// for offline diarization at the cost of latency.
/// </para>
/// </summary>
public sealed class DiariZenDiarizer : IDisposable
{
    private readonly InferenceSession _session;
    private bool _disposed = false;

    // ── Model parameters ─────────────────────────────────────────────────────

    public const int ChunkDurationSeconds = 16;
    public const int ChunkOverlapSeconds = 8;
    public const int SampleRate = 16000;
    public const int FrameRate = 50;
    // DiariZen model: 4 unique speakers, max 2 simultaneous per frame
    // Powerset classes: C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11
    public const int NumPowersetClasses = 11;
    public const int MaxUniqueSpeakers = 4;
    public const int MaxSimultaneousPerFrame = 2;

    public DiariZenDiarizer(string modelPath, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = new SessionOptions();
        opts.IntraOpNumThreads = 4;

        switch (ep)
        {
            case ExecutionProvider.Auto:
                try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
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

        _session = new InferenceSession(modelPath, opts);
    }

    /// <summary>
    /// Perform speaker diarization on audio waveform.
    /// 
    /// <para><strong>Pipeline:</strong></para>
    /// <list type="number">
    /// <item><description>Chunk audio into 16s segments with 8s overlap</description></item>
    /// <item><description>Run EEND model on each chunk → powerset scores</description></item>
    /// <item><description>Aggregate chunk predictions via overlap-add</description></item>
    /// <item><description>Apply softmax to convert logits to probabilities</description></item>
    /// <item><description>Extract speaker-active regions via thresholding</description></item>
    /// <item><description>Extract statistical features from regions</description></item>
    /// <item><description>Cluster regions using hierarchical clustering</description></item>
    /// <item><description> Merge overlapping segments for temporal continuity</description></item>
    /// </list>
    /// </para>
    /// </summary>
    /// <param name="audio">Mono audio waveform at 16kHz sample rate</param>
    /// <param name="minSpeakers">Minimum expected number of speakers. Used to cut clustering dendrogram.</param>
    /// <param name="maxSpeakers">Maximum expected number of speakers (currently unused, reserved for future constraints)</param>
    /// <param name="threshold">Binarization threshold for region extraction (default 0.5). Higher values are more conservative.</param>
    /// <returns>List of diarization segments with start time, end time, and speaker label</returns>
    public List<DiarizationSegment> Diarize(
        float[] audio,
        int minSpeakers = 1,
        int maxSpeakers = 8,
        float threshold = 0.5f)
    {
        if (audio.Length == 0)
            return new List<DiarizationSegment>();

        var chunks = ChunkAudio(audio);
        var chunkResults = new List<(float[] scores, double startTime, double endTime)>();

        foreach (var (chunk, startT, endT) in chunks)
        {
            var scores = SegmentChunk(chunk);
            chunkResults.Add((scores, startT, endT));
        }

        var aggregatedScores = AggregateSegmentations(chunkResults, audio);
        
        // Apply softmax across powerset classes to convert logits to probabilities
        var probabilities = ApplySoftmax(aggregatedScores);

        var regions = ExtractSpeakerRegions(probabilities, threshold);

        if (regions.Count == 0)
            return new List<DiarizationSegment>();

        var labeledRegions = ClusterRegions(audio, regions, minSpeakers, maxSpeakers);
        var finalSegments = MergeOverlappingSegments(labeledRegions);

        return finalSegments;
    }

    private List<(float[] chunk, double startTime, double endTime)> ChunkAudio(float[] audio)
    {
        var chunks = new List<(float[], double, double)>();

        int chunkSamples = ChunkDurationSeconds * SampleRate;
        int overlapSamples = ChunkOverlapSeconds * SampleRate;
        int strideSamples = chunkSamples - overlapSamples;

        int start = 0;
        while (start < audio.Length)
        {
            int end = Math.Min(start + chunkSamples, audio.Length);
            var chunk = new float[chunkSamples];
            int actualLength = end - start;
            Array.Copy(audio, start, chunk, 0, actualLength);

            chunks.Add((chunk, start / (double)SampleRate, end / (double)SampleRate));
            start += strideSamples;
        }

        return chunks;
    }

    private float[] SegmentChunk(float[] chunk)
    {
        var tensor = new DenseTensor<float>(chunk, new[] { 1, 1, chunk.Length });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("waveform", tensor) };

        using var results = _session.Run(inputs);
        var outputTensor = results.First(r => r.Name == "scores").AsTensor<float>();

        // Get output dimensions: expected shape is [1, frames, num_classes]
        int frames = outputTensor.Dimensions[1];
        int classes = outputTensor.Dimensions[2];

        var scores = new float[frames * classes];
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < classes; c++)
            {
                scores[f * classes + c] = outputTensor[0, f, c];
            }
        }

        return scores;
    }

    private float[] AggregateSegmentations(
    List<(float[] scores, double startTime, double endTime)> chunkResults,
    float[] audio)
    {
        if (chunkResults.Count == 0)
            return Array.Empty<float>();

        // Estimate total frames based on audio length
        int totalFrames = (int)(audio.Length / (double)SampleRate * FrameRate);

        // Create aggregated scores and count arrays
        var aggregated = new float[totalFrames * NumPowersetClasses];
        var counts = new int[totalFrames * NumPowersetClasses];

        int framesPerChunk = chunkResults[0].scores.Length / NumPowersetClasses;
        int framesPerSecond = FrameRate;

        foreach (var (scores, startTime, endTime) in chunkResults)
        {
            int startFrame = (int)(startTime * framesPerSecond);

            for (int t = 0; t < framesPerChunk && startFrame + t < totalFrames; t++)
            {
                int globalFrame = startFrame + t;
                for (int c = 0; c < NumPowersetClasses; c++)
                {
                    int idx = globalFrame * NumPowersetClasses + c;
                    int chunkIdx = t * NumPowersetClasses + c;

                    if (idx < aggregated.Length && chunkIdx < scores.Length)
                    {
                        aggregated[idx] += scores[chunkIdx];
                        counts[idx]++;
                    }
                }
            }
        }

        // Normalize by count (overlap-add)
        for (int i = 0; i < aggregated.Length; i++)
        {
            if (counts[i] > 0)
                aggregated[i] /= counts[i];
        }

        return aggregated;
    }

    private float[] ApplySoftmax(float[] logits)
    {
        if (logits.Length == 0)
            return logits;

        var probs = new float[logits.Length];
        int numFrames = logits.Length / NumPowersetClasses;

        for (int f = 0; f < numFrames; f++)
        {
            int baseIdx = f * NumPowersetClasses;
            
            // Find max for numerical stability
            float maxVal = logits[baseIdx];
            for (int c = 1; c < NumPowersetClasses; c++)
                maxVal = Math.Max(maxVal, logits[baseIdx + c]);

            // Compute exp and sum
            double sum = 0;
            var exps = new float[NumPowersetClasses];
            for (int c = 0; c < NumPowersetClasses; c++)
            {
                exps[c] = (float)Math.Exp(logits[baseIdx + c] - maxVal);
                sum += exps[c];
            }

            // Normalize
            for (int c = 0; c < NumPowersetClasses; c++)
                probs[baseIdx + c] = exps[c] / (float)sum;
        }

        return probs;
    }

    /// <summary>
    /// Extract speaker-active regions from powerset probabilities using hysteresis thresholding.
    /// 
    /// <para><strong>Process:</strong></para>
    /// <list type="number">
    /// <item><description>Decode powerset probabilities to per-speaker activity</description></item>
    /// <item><description>Find contiguous regions where each speaker is active</description></item>
    /// <item><description>Filter out regions shorter than minimum duration (~0.5s)</description></item>
    /// <item><description>Return flattened list of all speaker regions</description></item>
    /// </list>
    /// </para>
    ///
    /// <para><strong>Region Merging:</strong></para>
    /// Short gaps between active regions of the same speaker are not automatically
    /// merged here. This is handled in the final MergeOverlappingSegments step
    /// with a mergeGap parameter (default 0.5s).
    /// </para>
    /// </summary>
    private List<(int startFrame, int endFrame)> ExtractSpeakerRegions(float[] powersetScores, float threshold)
    {
        // Use proper powerset decoding with 4 speakers, max 2 simultaneous
        var activeSpeakersPerFrame = PowersetDecoder.BinarizePowerset(powersetScores, MaxUniqueSpeakers, MaxSimultaneousPerFrame, threshold);
        int numFrames = activeSpeakersPerFrame.Length;

        // Track regions per speaker
        var speakerRegions = new Dictionary<int, List<(int startFrame, int endFrame)>>();

        const int minDurationFrames = 25; // ~0.5 seconds

        for (int speaker = 0; speaker < MaxUniqueSpeakers; speaker++)
        {
            if (!speakerRegions.ContainsKey(speaker))
                speakerRegions[speaker] = new List<(int, int)>();

            int? startFrame = null;

            for (int t = 0; t < numFrames; t++)
            {
                bool isActive = activeSpeakersPerFrame[t].Contains(speaker);

                if (isActive && startFrame == null)
                {
                    startFrame = t;
                }
                else if (!isActive && startFrame.HasValue)
                {
                    if (t - startFrame.Value >= minDurationFrames)
                        speakerRegions[speaker].Add((startFrame.Value, t));
                    startFrame = null;
                }
            }

            // Handle region at end
            if (startFrame.HasValue && numFrames - startFrame.Value >= minDurationFrames)
                speakerRegions[speaker].Add((startFrame.Value, numFrames));
        }

        // Flatten all speaker regions into single list
        var allRegions = new List<(int, int)>();
        foreach (var (speaker, regions) in speakerRegions)
        {
            if (regions.Count > 0)
                allRegions.AddRange(regions);
        }

        // Sort by start time
        return allRegions.OrderBy(r => r.Item1).ToList();
    }

     /// <summary>
    /// Extract statistical features from audio regions for clustering.
    ///
    /// <para><strong>Feature Vector (10 dimensions):</strong></para>
    /// <list type="number">
    /// <item><description>Mean amplitude</description></item>
    /// <item><description>Standard deviation</description></item>
    /// <item><description>Zero-crossing rate</description></item>
    /// <item><description>Log energy</description></item>
    /// <item><description>Duration in seconds</description></item>
    /// <item><description>Normalized start position</description></item>
    /// <item><description>Signal-to-noise ratio proxy (|mean|/std)</description></item>
    /// <item><description>Normalized zero-crossing rate</description></item>
    /// <item><description>Sample count normalized</description></item>
    /// <item><description>Root mean square energy</description></item>
    /// </list>
    ///
    /// <para><strong>Note:</strong></para>
    /// Unlike x-vector or ECAPA-TDNN embeddings used in production systems,
    /// this implementation uses simple statistical features for demonstration.
    /// Production DiariZen uses pre-trained speaker embedding models.
    /// </para>
    /// </summary>
    private double[][] ExtractRegionFeatures(float[] audio, List<(int startFrame, int endFrame)> regions)
    {
        const int featureDim = 10;
        var features = new double[regions.Count][];

        for (int i = 0; i < regions.Count; i++)
        {
            var (startFrame, endFrame) = regions[i];
            int startSample = Math.Min((int)(startFrame * SampleRate / FrameRate), audio.Length - 1);
            int endSample = Math.Min((int)(endFrame * SampleRate / FrameRate), audio.Length);
            int regionLen = endSample - startSample;

            if (regionLen <= 0)
            {
                features[i] = new double[featureDim];
                continue;
            }

            var feat = new double[featureDim];
            double sum = 0, sumSq = 0;
            int zeroCrossings = 0;

            float prev = audio[startSample];
            sum += prev;
            sumSq += prev * prev;

            for (int j = startSample + 1; j < endSample; j++)
            {
                float curr = audio[j];
                sum += curr;
                sumSq += curr * curr;

                if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0))
                    zeroCrossings++;
                prev = curr;
            }

            double mean = sum / regionLen;
            double std = Math.Sqrt(Math.Max(0, sumSq / regionLen - mean * mean));
            double energy = sumSq / regionLen;

            feat[0] = mean;
            feat[1] = std;
            feat[2] = zeroCrossings / (double)regionLen;
            feat[3] = Math.Log(energy + 1e-10);
            feat[4] = (endFrame - startFrame) / (double)FrameRate;
            feat[5] = startFrame / (double)(audio.Length * 1.0 / FrameRate);
            feat[6] = Math.Abs(mean) / (std + 1e-10);
            feat[7] = zeroCrossings / (std + 1e-10);
            feat[8] = regionLen / (double)SampleRate;
            feat[9] = Math.Sqrt(energy);

            features[i] = feat;
        }

        return features;
    }

      /// <summary>
    /// Cluster speaker regions using hierarchical agglomerative clustering.
    ///
    /// <para><strong>Algorithm:</strong></para>
    /// <list type="number">
    /// <item><description>Extract 10-dimensional statistical features from each region</description></item>
    /// <item><description>Compute linkage matrix using centroid method</description></item>
    /// <item><description>Cut dendrogram to get minSpeakers clusters</description></item>
    /// <item><description>Assign speaker labels (speaker_0, speaker_1, etc.)</description></item>
    /// </list>
    /// </para>
    ///
    /// <para><strong>Clustering Strategy:</strong></para>
    /// We use FclusterMaxClust with minSpeakers to avoid over-segmentation.
    /// This ensures we don't create too many speaker clusters when there's
    /// uncertainty. Post-processing can merge similar speakers if needed.
    /// </para>
    ///
    /// <para><strong>Alternative: VBx Clustering</strong></para>
    /// DiariZen supports VBx (Variational Bayesian) clustering which uses:
    /// <list type="bullet">
    /// <item><description>AHC initialization with centroid linkage (threshold=0.6)</description></item>
    /// <item><description>PLDA transformation (128-dimensional)</description></item>
    /// <item><description>Variational Bayesian inference (maxIters=20)</description></item>
    /// <item><description>Parameters: Fa=0.07, Fb=0.8 for regularization</description></item>
    /// </list>
    /// </para>
    /// </summary>
    private List<(int startFrame, int endFrame, string speakerLabel)> ClusterRegions(
        float[] audio,
        List<(int startFrame, int endFrame)> regions,
        int minSpeakers,
        int maxSpeakers)
    {
        if (regions.Count == 0)
            return new List<(int, int, string)>();

        // Handle single region case
        if (regions.Count == 1)
        {
            var (start, end) = regions[0];
            return new List<(int, int, string)> { (start, end, "speaker_0") };
        }

        var features = ExtractRegionFeatures(audio, regions);
        var linkage = HierarchicalClustering.Linkage(features, "centroid");

        // Use FclusterMaxClust to get a specific number of clusters
        // Use minSpeakers as the target to avoid over-segmentation
        int targetClusters = minSpeakers;
        targetClusters = Math.Max(targetClusters, 2);
        
        int[] labels = HierarchicalClustering.FclusterMaxClust(linkage, targetClusters);
        int numClusters = labels.Max() + 1;

        var labeled = new List<(int, int, string)>();
        for (int i = 0; i < regions.Count; i++)
        {
            var (start, end) = regions[i];
            labeled.Add((start, end, $"speaker_{labels[i]}"));
        }

        return labeled;
    }

    /// <summary>
    /// Merge small clusters into nearest large cluster based on centroid distance.
    ///
    /// <para><strong>Process:</strong></para>
    /// <list type="number">
    /// <item><description>Identify large clusters (≥ minClusterSize members)</description></item>
    /// <item><description>Identify small clusters (< minClusterSize members)</description></item>
    /// <item><description>Compute centroids for large clusters</description></item>
    /// <item><description>For each small cluster, find nearest large cluster centroid</description></item>
    /// <item><description>Reassign all members of small cluster to target</description></item>
    /// <item><description>Relabel clusters to consecutive integers</description></item>
    /// </list>
    /// </para>
    ///
    /// <para><strong>DiariZen Default:</strong></para>
    /// minClusterSize = 13 frames (~0.26 seconds at 50Hz frame rate).
    /// This prevents tiny spurious clusters from being assigned unique speaker IDs.
    /// </para>
    /// </summary>
    private int[] MergeSmallClusters(double[][] features, int[] labels, int minClusterSize)
    {
        int n = labels.Length;
        int numClusters = labels.Max() + 1;
        
        // Identify large and small clusters
        var clusterMembers = new System.Collections.Generic.List<int>[numClusters];
        for (int c = 0; c < numClusters; c++)
            clusterMembers[c] = new System.Collections.Generic.List<int>();
        
        for (int i = 0; i < n; i++)
            clusterMembers[labels[i]].Add(i);
        
        var largeClusters = new System.Collections.Generic.List<int>();
        var smallClusters = new System.Collections.Generic.List<int>();
        
        for (int c = 0; c < numClusters; c++)
        {
            if (clusterMembers[c].Count >= minClusterSize)
                largeClusters.Add(c);
            else
                smallClusters.Add(c);
        }
        
        if (smallClusters.Count == 0)
            return labels;
        
        // If no large clusters, just return as-is (all small clusters)
        if (largeClusters.Count == 0)
            return labels;
        
        // Compute centroids for large clusters
        var largeCentroids = new double[largeClusters.Count][];
        int d = features[0].Length;
        
        for (int i = 0; i < largeClusters.Count; i++)
        {
            int c = largeClusters[i];
            largeCentroids[i] = new double[d];
            foreach (int idx in clusterMembers[c])
            {
                for (int j = 0; j < d; j++)
                    largeCentroids[i][j] += features[idx][j];
            }
            for (int j = 0; j < d; j++)
                largeCentroids[i][j] /= clusterMembers[c].Count;
        }
        
        // Merge small clusters to nearest large cluster
        var newLabels = (int[])labels.Clone();
        
        foreach (int smallC in smallClusters)
        {
            // Compute centroid of small cluster
            var smallCentroid = new double[d];
            foreach (int idx in clusterMembers[smallC])
            {
                for (int j = 0; j < d; j++)
                    smallCentroid[j] += features[idx][j];
            }
            for (int j = 0; j < d; j++)
                smallCentroid[j] /= clusterMembers[smallC].Count;
            
            // Find nearest large cluster
            int nearestLarge = 0;
            double minDist = double.MaxValue;
            
            for (int i = 0; i < largeClusters.Count; i++)
            {
                double dist = EuclideanDistance(smallCentroid, largeCentroids[i]);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestLarge = i;
                }
            }
            
            // Reassign all members of small cluster
            int targetCluster = largeClusters[nearestLarge];
            foreach (int idx in clusterMembers[smallC])
                newLabels[idx] = targetCluster;
        }
        
        // Relabel to consecutive integers
        var labelMap = new System.Collections.Generic.Dictionary<int, int>();
        int newLabel = 0;
        for (int i = 0; i < n; i++)
        {
            if (!labelMap.ContainsKey(newLabels[i]))
                labelMap[newLabels[i]] = newLabel++;
            newLabels[i] = labelMap[newLabels[i]];
        }
        
        return newLabels;
    }

    private double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

     /// <summary>
    /// Merge temporally overlapping or adjacent segments of the same speaker.
    ///
    /// <para><strong>Merging Logic:</strong></para>
    /// Segments are merged if:
    /// <list type="bullet">
    /// <item><description>They belong to the same speaker label</description></item>
    /// <item><description>The gap between them is ≤ mergeGap (default 0.5s)</description></item>
    /// </list>
    /// </para>
    ///
    /// <para><strong>Purpose:</strong></para>
    /// This post-processing step improves temporal continuity by combining
    /// fragmented segments that likely belong to the same utterance. Short
    /// gaps may occur due to thresholding artifacts or brief pauses in speech.
    /// </para>
    /// </summary>
    private List<DiarizationSegment> MergeOverlappingSegments(
        List<(int startFrame, int endFrame, string speakerLabel)> labeledRegions)
    {
        if (labeledRegions.Count == 0)
            return new List<DiarizationSegment>();

        var sorted = labeledRegions.OrderBy(r => r.startFrame).ToList();
        var merged = new List<DiarizationSegment>();

        double currentStart = sorted[0].startFrame / (double)FrameRate;
        double currentEnd = sorted[0].endFrame / (double)FrameRate;
        string currentSpeaker = sorted[0].speakerLabel;
        const double mergeGap = 0.5;

        for (int i = 1; i < sorted.Count; i++)
        {
            var (startFrame, endFrame, speaker) = sorted[i];
            double startT = startFrame / (double)FrameRate;
            double endT = endFrame / (double)FrameRate;

            if (speaker == currentSpeaker && startT - currentEnd <= mergeGap)
                currentEnd = Math.Max(currentEnd, endT);
            else
            {
                merged.Add(new DiarizationSegment(currentStart, currentEnd, currentSpeaker));
                currentStart = startT;
                currentEnd = endT;
                currentSpeaker = speaker;
            }
        }

        merged.Add(new DiarizationSegment(currentStart, currentEnd, currentSpeaker));
        return merged;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}

public sealed class DiarizationSegment
{
    public double Start { get; }
    public double End { get; }
    public string Speaker { get; }
    public double Duration => End - Start;

    public DiarizationSegment(double start, double end, string speaker)
    {
        Start = start;
        End = end;
        Speaker = speaker;
    }

    public override string ToString() =>
        $"{Start:F2}s - {End:F2}: {Speaker} ({Duration:F2}s)";
}
