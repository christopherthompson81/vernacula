using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Parakeet.Base;

/// <summary>
/// DiariZen speaker diarization using ONNX Runtime.
/// 
/// Features:
/// - ONNX-based segmentation (WavLM + Conformer)
/// - Statistical feature extraction from audio regions
/// - Agglomerative hierarchical clustering for speaker assignment
/// - Segment merging for temporal continuity
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
    public const int NumPowersetClasses = 11;
    // Number of simultaneous speakers the model's powerset was trained for.
    // This is a model architecture constant — changing it requires retraining.
    // Total unique speakers across a recording is unconstrained; clustering handles that.
    public const int MaxSimultaneousSpeakersPerChunk = 4;

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
        var regions = ExtractSpeakerRegions(aggregatedScores, threshold);
        
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
        
        int numElements = 1;
        foreach (var dim in outputTensor.Dimensions)
            numElements *= dim;
        
        var scores = new float[numElements];
        for (int i = 0; i < numElements; i++)
            scores[i] = outputTensor[i];
        
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

    private List<(int startFrame, int endFrame)> ExtractSpeakerRegions(float[] powersetScores, float threshold)
    {
        // Use proper powerset decoding
        var activeSpeakersPerFrame = PowersetDecoder.BinarizePowerset(powersetScores, MaxSimultaneousSpeakersPerChunk, threshold);
        int numFrames = activeSpeakersPerFrame.Length;

        // Track regions per speaker
        var speakerRegions = new Dictionary<int, List<(int startFrame, int endFrame)>>();

        const int minDurationFrames = 25; // ~0.5 seconds
        
        for (int speaker = 0; speaker < MaxSimultaneousSpeakersPerChunk; speaker++)
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
        return allRegions.OrderBy(r => r.startFrame).ToList();
    }

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

    private List<(int startFrame, int endFrame, string speakerLabel)> ClusterRegions(
        float[] audio,
        List<(int startFrame, int endFrame)> regions,
        int minSpeakers,
        int maxSpeakers)
    {
        if (regions.Count == 0)
            return new List<(int, int, string)>();

        var features = ExtractRegionFeatures(audio, regions);
        var linkage = HierarchicalClustering.Linkage(features, "centroid");
        
        double[] distances = linkage.Select(r => r[2]).ToArray();
        Array.Sort(distances);
        double threshold = distances.Length > 0 ? distances[distances.Length / 2] : 1.0;
        
        int[] labels = HierarchicalClustering.FclusterThreshold(linkage, threshold);
        int numClusters = labels.Max() + 1;
        
        if (numClusters < minSpeakers)
            labels = HierarchicalClustering.FclusterMaxClust(linkage, minSpeakers);
        else if (numClusters > maxSpeakers)
            labels = HierarchicalClustering.FclusterMaxClust(linkage, maxSpeakers);

        var labeled = new List<(int, int, string)>();
        for (int i = 0; i < regions.Count; i++)
        {
            var (start, end) = regions[i];
            labeled.Add((start, end, $"speaker_{labels[i]}"));
        }

        return labeled;
    }

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
