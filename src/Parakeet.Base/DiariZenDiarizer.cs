using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Parakeet.Base;

/// <summary>
/// DiariZen speaker diarization using ONNX Runtime.
/// 
/// This class wraps the exported DiariZen segmentation model (WavLM + Conformer)
/// and implements a simplified clustering pipeline for C#.
/// 
/// The full DiariZen pipeline includes:
/// 1. Chunk audio into overlapping segments (16s chunks, 8s overlap)
/// 2. Run segmentation model to get per-frame speaker activity scores
/// 3. Aggregate scores across chunks
/// 4. Binarize and extract active speaker regions
/// 5. Cluster regions to assign speaker labels
/// 6. Merge overlapping segments
/// 
/// For C# portability, we use a simplified approach:
/// - Skip separate embedding model (use temporal clustering only)
/// - Simple threshold-based binarization
/// - Basic region merging
/// </summary>
public sealed class DiariZenDiarizer : IDisposable
{
    private readonly InferenceSession _session;
    private bool _disposed = false;

    // ── Model parameters (from config.toml) ───────────────────────────────────
    
    /// <summary>Chunk duration in seconds (DiariZen default: 16s)</summary>
    public const int ChunkDurationSeconds = 16;
    
    /// <summary>Overlap between chunks in seconds (half of chunk duration)</summary>
    public const int ChunkOverlapSeconds = 8;
    
    /// <summary>Sample rate (must be 16kHz for DiariZen)</summary>
    public const int SampleRate = 16000;
    
    /// <summary>Frame rate of output (~50Hz from WavLM)</summary>
    public const int FrameRate = 50;
    
    /// <summary>Number of powerset classes (for max 4 speakers: 2^4 - 1 = 15, but model uses 11)</summary>
    public const int NumPowersetClasses = 11;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Initialize the DiariZen diarizer with an exported ONNX model.
    /// </summary>
    /// <param name="modelPath">Path to diarizen_segmentation.onnx</param>
    /// <param name="ep">Execution provider (Auto = try CUDA/DML, fallback to CPU)</param>
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

        _session = new InferenceSession(modelPath, opts);
        
        // Verify model inputs/outputs
        var input = _session.InputMetadata.First();
        if (input.Key != "waveform")
            throw new InvalidOperationException($"Expected input name 'waveform', got '{input.Key}'");
    }

    /// <summary>
    /// Run diarization on raw audio data.
    /// </summary>
    /// <param name="audio">16kHz mono audio as float32 samples in [-1, 1]</param>
    /// <param name="minSpeakers">Minimum number of speakers to detect</param>
    /// <param name="maxSpeakers">Maximum number of speakers to detect</param>
    /// <param name="threshold">Binarization threshold for segmentation scores (default: 0.5)</param>
    /// <returns>List of diarization segments with start/end times and speaker labels</returns>
    public List<DiarizationSegment> Diarize(
        float[] audio,
        int minSpeakers = 1,
        int maxSpeakers = 8,
        float threshold = 0.5f)
    {
        if (audio.Length == 0)
            return new List<DiarizationSegment>();

        // Step 1: Chunk audio
        var chunks = ChunkAudio(audio);
        
        // Step 2: Run segmentation on each chunk
        var chunkResults = new List<(float[] scores, double startTime, double endTime)>();
        foreach (var (chunk, startT, endT) in chunks)
        {
            var scores = SegmentChunk(chunk);
            chunkResults.Add((scores, startT, endT));
        }

        // Step 3: Aggregate scores across chunks
        var aggregatedScores = AggregateSegmentations(chunkResults);
        
        // Step 4: Binarize and extract regions
        var binaryActivity = BinarizeSegmentation(aggregatedScores, threshold);
        var regions = ExtractSpeakerRegions(binaryActivity);
        
        if (regions.Count == 0)
            return new List<DiarizationSegment>();

        // Step 5: Simple temporal clustering (no embeddings for now)
        var labeledRegions = ClusterRegionsTemporally(regions, minSpeakers, maxSpeakers);
        
        // Step 6: Merge overlapping segments
        var finalSegments = MergeOverlappingSegments(labeledRegions);
        
        return finalSegments;
    }

    // ── Audio chunking ───────────────────────────────────────────────────────

    /// <summary>
    /// Split audio into overlapping chunks.
    /// </summary>
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
            
            // Create chunk with zero padding if needed
            var chunk = new float[chunkSamples];
            int actualLength = end - start;
            Array.Copy(audio, start, chunk, 0, actualLength);
            
            double startTime = start / (double)SampleRate;
            double endTime = end / (double)SampleRate;
            
            chunks.Add((chunk, startTime, endTime));
            start += strideSamples;
        }

        return chunks;
    }

    // ── Segmentation inference ───────────────────────────────────────────────

    /// <summary>
    /// Run segmentation model on a single audio chunk.
    /// </summary>
    /// <param name="chunk">Audio chunk of exactly ChunkDurationSeconds at 16kHz</param>
    /// <returns>Segmentation scores of shape [numFrames, numPowersetClasses]</returns>
    private float[] SegmentChunk(float[] chunk)
    {
        // Reshape for ONNX input: (batch=1, channels=1, samples)
        var tensor = new DenseTensor<float>(chunk, new[] { 1, 1, chunk.Length });
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("waveform", tensor)
        };

        using var results = _session.Run(inputs);
        
        // Extract output: (batch=1, frames, classes) -> (frames * classes,)
        var outputTensor = results.First(r => r.Name == "scores").AsTensor<float>();
        
        int numElements = 1;
        foreach (var dim in outputTensor.Dimensions)
            numElements *= dim;
        
        var scores = new float[numElements];
        for (int i = 0; i < numElements; i++)
            scores[i] = outputTensor[i];
        
        return scores;
    }

    // ── Score aggregation ────────────────────────────────────────────────────

    /// <summary>
    /// Aggregate segmentation scores from all chunks into a single array.
    /// 
    /// Simplified approach: just concatenate (proper implementation would align
    /// and average overlapping regions based on timestamps).
    /// </summary>
    private float[] AggregateSegmentations(
        List<(float[] scores, double startTime, double endTime)> chunkResults)
    {
        if (chunkResults.Count == 0)
            return Array.Empty<float>();

        // Calculate total length
        int totalLength = chunkResults.Sum(c => c.scores.Length);
        var aggregated = new float[totalLength];
        
        int offset = 0;
        foreach (var (scores, _, _) in chunkResults)
        {
            Array.Copy(scores, 0, aggregated, offset, scores.Length);
            offset += scores.Length;
        }

        return aggregated;
    }

    // ── Binarization ─────────────────────────────────────────────────────────

    /// <summary>
    /// Convert soft segmentation scores to binary speaker activity.
    /// 
    /// Simplified: apply threshold to each powerset class independently.
    /// Proper implementation would decode the powerset representation.
    /// </summary>
    private bool[] BinarizeSegmentation(float[] scores, float threshold)
    {
        // Reshape: scores is (totalFrames * numClasses,) -> treat as flat array
        // For simplicity, we'll just threshold the max activation per frame
        
        int numFrames = scores.Length / NumPowersetClasses;
        var binary = new bool[numFrames];
        
        for (int t = 0; t < numFrames; t++)
        {
            // Find max activation across all classes for this frame
            float maxAct = 0f;
            for (int c = 0; c < NumPowersetClasses; c++)
            {
                int idx = t * NumPowersetClasses + c;
                if (idx < scores.Length && scores[idx] > maxAct)
                    maxAct = scores[idx];
            }
            
            binary[t] = maxAct >= threshold;
        }

        return binary;
    }

    // ── Region extraction ────────────────────────────────────────────────────

    /// <summary>
    /// Extract contiguous active speaker regions from binary activity.
    /// </summary>
    private List<(int startFrame, int endFrame)> ExtractSpeakerRegions(bool[] binaryActivity)
    {
        var regions = new List<(int, int)>();
        
        const int minDurationFrames = 25; // ~0.5 seconds at 50Hz
        
        int? startFrame = null;
        
        for (int t = 0; t < binaryActivity.Length; t++)
        {
            if (binaryActivity[t] && startFrame == null)
            {
                startFrame = t;
            }
            else if (!binaryActivity[t] && startFrame.HasValue)
            {
                int duration = t - startFrame.Value;
                if (duration >= minDurationFrames)
                {
                    regions.Add((startFrame.Value, t));
                }
                startFrame = null;
            }
        }
        
        // Handle region at end
        if (startFrame.HasValue)
        {
            int duration = binaryActivity.Length - startFrame.Value;
            if (duration >= minDurationFrames)
            {
                regions.Add((startFrame.Value, binaryActivity.Length));
            }
        }

        return regions;
    }

    // ── Temporal clustering ─────────────────────────────────────────────────

    /// <summary>
    /// Simple temporal clustering: assign speaker labels based on region order.
    /// 
    /// This is a placeholder that alternates between speakers. A proper
    /// implementation would use embeddings and hierarchical clustering.
    /// </summary>
    private List<(int startFrame, int endFrame, string speakerLabel)> ClusterRegionsTemporally(
        List<(int startFrame, int endFrame)> regions,
        int minSpeakers,
        int maxSpeakers)
    {
        // For now, use a simple alternating scheme
        // In production, this should be replaced with proper clustering
        
        var labeled = new List<(int, int, string)>();
        int currentSpeaker = 0;
        
        for (int i = 0; i < regions.Count; i++)
        {
            var (start, end) = regions[i];
            
            // Simple heuristic: if gap to previous region is large, might be different speaker
            if (i > 0)
            {
                var prevEnd = regions[i - 1].endFrame;
                int gapFrames = start - prevEnd;
                
                // If gap > 2 seconds, possibly new speaker
                if (gapFrames > FrameRate * 2 && currentSpeaker < maxSpeakers - 1)
                {
                    currentSpeaker = (currentSpeaker + 1) % maxSpeakers;
                }
            }
            
            labeled.Add((start, end, $"speaker_{currentSpeaker}"));
        }

        return labeled;
    }

    // ── Segment merging ─────────────────────────────────────────────────────

    /// <summary>
    /// Merge adjacent/overlapping segments from the same speaker.
    /// </summary>
    private List<DiarizationSegment> MergeOverlappingSegments(
        List<(int startFrame, int endFrame, string speakerLabel)> labeledRegions)
    {
        if (labeledRegions.Count == 0)
            return new List<DiarizationSegment>();

        // Sort by start time
        var sorted = labeledRegions.OrderBy(r => r.startFrame).ToList();
        
        var merged = new List<DiarizationSegment>();
        
        double currentStart = sorted[0].startFrame / (double)FrameRate;
        double currentEnd = sorted[0].endFrame / (double)FrameRate;
        string currentSpeaker = sorted[0].speakerLabel;
        
        const double mergeGap = 0.5; // Merge if gap < 0.5 seconds
        
        for (int i = 1; i < sorted.Count; i++)
        {
            var (startFrame, endFrame, speaker) = sorted[i];
            double startT = startFrame / (double)FrameRate;
            double endT = endFrame / (double)FrameRate;
            
            // Same speaker and overlapping/adjacent?
            if (speaker == currentSpeaker && startT - currentEnd <= mergeGap)
            {
                currentEnd = Math.Max(currentEnd, endT);
            }
            else
            {
                // Output current segment
                merged.Add(new DiarizationSegment(currentStart, currentEnd, currentSpeaker));
                
                // Start new segment
                currentStart = startT;
                currentEnd = endT;
                currentSpeaker = speaker;
            }
        }
        
        // Don't forget last segment
        merged.Add(new DiarizationSegment(currentStart, currentEnd, currentSpeaker));

        return merged;
    }

    // ── IDisposable ─────────────────────────────────────────────────────────

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

/// <summary>
/// Represents a single speaker segment from diarization.
/// </summary>
public sealed class DiarizationSegment
{
    /// <summary>Start time in seconds</summary>
    public double Start { get; }
    
    /// <summary>End time in seconds</summary>
    public double End { get; }
    
    /// <summary>Speaker label (e.g., "speaker_0")</summary>
    public string Speaker { get; }
    
    /// <summary>Segment duration in seconds</summary>
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
