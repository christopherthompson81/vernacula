using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;
using System.Collections.Concurrent;
using System.Text.Json;

namespace Parakeet.Base;

/// <summary>
/// DiariZen speaker diarization using ONNX Runtime.
///
/// <para><strong>Pipeline (matching pyannote/DiariZen):</strong></para>
/// <list type="number">
/// <item><description>Audio is chunked into 16 s segments with 1.6 s stride (segmentation_step=0.1)</description></item>
/// <item><description>WavLM+Conformer model outputs powerset logits per frame (batch=1)</description></item>
/// <item><description>Per chunk: softmax → median filter → soft per-speaker scores (T×4) by summing powerset class probabilities</description></item>
/// <item><description>Per (chunk, local_speaker) with enough active frames: extract WeSpeaker embedding</description></item>
/// <item><description>HAC (centroid linkage, distance threshold) clusters all embeddings globally</description></item>
/// <item><description>Reconstruct timeline: remap local→global speakers, overlap-add, normalise</description></item>
/// <item><description>Per global speaker: binarize at 0.5, extract active regions</description></item>
/// <item><description>Temporal merging combines near-adjacent same-speaker segments</description></item>
/// </list>
/// </summary>
public sealed class DiariZenDiarizer : IDisposable
{
    private readonly InferenceSession[] _segSessions;
    private readonly WeSpeakerEmbedder? _embedder;
    private readonly string _segInputName;
    private readonly string _segOutputName;
    private readonly bool _segSupportsBatching;
    private readonly int _segBatchSize;
    private int _segBatchingDisabled;
    private readonly bool _preferGpuEmbeddingPipeline;
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
    private const int DefaultMinActiveFramesForEmbed = 10;
    private const double MinCleanFrameRatioForClustering = 0.1;
    private static readonly int MaxEmbeddingWorkers =
        Math.Max(1, Config.GetDiariZenEmbeddingMaxWorkers() ?? Math.Max(1, Environment.ProcessorCount - 2));
    private static readonly int[][] PowersetCombinations =
        PowersetDecoder.GetPowersetCombinations(MaxUniqueSpeakers, MaxSimultaneousPerFrame)
            .Select(combo => combo.ToArray())
            .ToArray();
    private static readonly bool UseConstrainedCentroidAssignment =
        !string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DISABLE_CONSTRAINED_ASSIGNMENT"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool DisablePostSmoothing =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DISABLE_POST_SMOOTHING"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool DisableGapFill =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DISABLE_GAP_FILL"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool DisableShortRegionRemoval =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DISABLE_SHORT_REGION_REMOVAL"),
            "1",
            StringComparison.Ordinal);
    private static readonly int MinActiveFramesForEmbed = GetMinActiveFramesForEmbed();
    // ── Construction ───────────────────────────────────────────────────────

    public DiariZenDiarizer(
        string segmentationModelPath,
        string? embeddingModelPath = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _segSessions = new InferenceSession[DetermineSegmentationWorkerCount(ep)];
        for (int i = 0; i < _segSessions.Length; i++)
            _segSessions[i] = CreateSegmentationSession(segmentationModelPath, ep);
        _segInputName = _segSessions[0].InputMetadata.Keys.First();
        _segOutputName = _segSessions[0].OutputMetadata.Keys.First();
        _segSupportsBatching = SupportsDynamicBatch(_segSessions[0], _segInputName);
        _segBatchSize = _segSupportsBatching ? Config.GetDiariZenSegmentationBatchSize() : 1;

        if (embeddingModelPath != null && File.Exists(embeddingModelPath))
        {
            // Look for LDA transform in a "plda" sibling directory next to the model file
            string? ldaDir = null;
            string? modelDir = Path.GetDirectoryName(embeddingModelPath);
            if (modelDir != null)
            {
                string candidate = Path.Combine(modelDir, Config.DiariZenLdaDir);
                if (Directory.Exists(candidate)) ldaDir = candidate;
            }
            _embedder = new WeSpeakerEmbedder(embeddingModelPath, ldaDir, ep);
        }

        _preferGpuEmbeddingPipeline = _embedder?.PreferGpuBatching == true;
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>
    /// Perform speaker diarization on a 16 kHz mono waveform.
    /// </summary>
    public List<DiarizationSegment> Diarize(
        float[] audio,
        int   minSpeakers  = 1,
        int   maxSpeakers  = 8,
        float threshold    = 0.5f,
        float ahcThreshold = -1f,
        Action<string>? progress = null)   // -1 = use Config default
    {
        if (audio.Length == 0)
            return [];

        int chunkCount = GetChunkCount(audio.Length);
        progress?.Invoke($"chunked audio into {chunkCount} window(s)");

        var startTimes = new List<double>(chunkCount);
        var perChunkBinary = new List<float[,]>(chunkCount);

        // ── Step 3: Extract per-(chunk, local_speaker) embeddings ─────────
        // embeddings — L2-normed 256-dim, used for AHC initialisation
        // xvecs      — 128-dim xvec (after xvec_tf but before batch plda_tf), collected for VBx
        var embeddings = new List<double[]>();
        var xvecs      = new List<float[]>();   // raw xvec before batch plda_tf
        var embedKeys  = new List<(int chunkIdx, int speakerIdx)>();
        var results    = Array.Empty<EmbeddingJobResult>();

        progress?.Invoke("running segmentation model");
        List<EmbeddingJob>? jobs = _embedder != null ? new List<EmbeddingJob>() : null;
        BlockingCollection<ChunkEmbeddingPrepRequest>? prepQueue = null;
        ConcurrentDictionary<int, WeSpeakerEmbedder.PreparedEmbeddingInput>? preparedMap = null;
        Task[] prepWorkers = [];
        var completedPrepJobs = new[] { 0 };
        var lastReportedPrepJobs = new[] { 0 };

        if (_embedder != null)
        {
            progress?.Invoke("extracting speaker embeddings");
            prepQueue = new BlockingCollection<ChunkEmbeddingPrepRequest>(Math.Max(1, MaxEmbeddingWorkers * 2));
            preparedMap = new ConcurrentDictionary<int, WeSpeakerEmbedder.PreparedEmbeddingInput>();
            prepWorkers = StartEmbeddingPrepWorkers(
                _embedder,
                prepQueue,
                preparedMap,
                () => jobs!.Count,
                progress,
                completedPrepJobs,
                lastReportedPrepJobs);
        }

        int decodedChunks = 0;
        int batchStartChunkIndex = 0;
        foreach (var chunkBatch in EnumerateChunkBatches(audio, _segBatchSize))
        {
            var batchScores = SegmentBatch(chunkBatch.Select(item => item.Chunk).ToList());
            for (int bi = 0; bi < chunkBatch.Count; bi++)
            {
                var chunkInfo = chunkBatch[bi];
                var rawScores  = batchScores[bi];
                var chunkScore = DecodeChunkBinary(
                    rawScores,
                    threshold,
                    Config.DiariZenMedianFilterSize);
                perChunkBinary.Add(chunkScore);
                startTimes.Add(chunkInfo.StartTime);

                int ci = batchStartChunkIndex + bi;
                if (ShouldReportProgress(ci, chunkCount))
                    progress?.Invoke($"decoded chunk {ci + 1}/{chunkCount}");

                if (_embedder == null || jobs == null || prepQueue == null)
                    continue;

                int numFrames  = chunkScore.GetLength(0);
                var overlapCounts = new int[numFrames];
                var chunkJobs = new List<EmbeddingJob>(MaxUniqueSpeakers);
                for (int t = 0; t < numFrames; t++)
                for (int other = 0; other < MaxUniqueSpeakers; other++)
                    if (chunkScore[t, other] > 0f)
                        overlapCounts[t]++;

                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                {
                    var activeMask = new bool[numFrames];
                    var cleanMask  = new bool[numFrames];
                    int activeCount = 0;
                    int cleanCount  = 0;
                    for (int t = 0; t < numFrames; t++)
                    {
                        if (chunkScore[t, spk] <= 0f) continue;
                        activeMask[t] = true;
                        activeCount++;
                        if (overlapCounts[t] == 1)
                        {
                            cleanMask[t] = true;
                            cleanCount++;
                        }
                    }

                    if (activeCount == 0) continue;

                    bool[] selectedMask;
                    if (_embedder.SupportsWeights)
                    {
                        // Match pyannote speaker_diarization: derive the minimum
                        // clean-mask length from the embedder's minimum accepted
                        // waveform size, use clean speech only when it is strictly
                        // longer than that threshold, otherwise fall back to the
                        // full speaker mask without dropping the local speaker.
                        int minMaskFrames = Math.Max(
                            1,
                            (int)Math.Ceiling(
                                numFrames * (_embedder.MinNumSamples / (double)chunkInfo.Chunk.Length)));
                        selectedMask = cleanCount > minMaskFrames ? cleanMask : activeMask;
                    }
                    else
                    {
                        if (activeCount < MinActiveFramesForEmbed) continue;
                        selectedMask = cleanCount >= MinActiveFramesForEmbed ? cleanMask : activeMask;
                    }

                    int sequenceIndex = jobs.Count;
                    var job = new EmbeddingJob(sequenceIndex, ci, spk, selectedMask, numFrames, cleanCount);
                    jobs.Add(job);
                    chunkJobs.Add(job);
                }

                if (chunkJobs.Count > 0)
                    prepQueue.Add(new ChunkEmbeddingPrepRequest(ci, chunkInfo.Chunk, chunkJobs));

                if (ShouldReportProgress(ci, chunkCount))
                    progress?.Invoke($"processed embeddings for chunk {ci + 1}/{chunkCount}");
            }
            decodedChunks += chunkBatch.Count;
            batchStartChunkIndex += chunkBatch.Count;
            progress?.Invoke($"segmented chunk {decodedChunks}/{chunkCount}");
        }

        if (_embedder != null && jobs != null && prepQueue != null && preparedMap != null)
        {
            prepQueue.CompleteAdding();
            Task.WaitAll(prepWorkers);
            results = ComputeEmbeddingResults(_embedder, jobs, preparedMap, progress);
            MaterializeEmbeddingOutputs(results, embeddings, xvecs, embedKeys);

            WriteEmbeddingDebug(startTimes, jobs, results);
        }

        progress?.Invoke($"collected {embeddings.Count} embedding(s) from {perChunkBinary.Count} chunk(s)");
        WriteLocalMaskDebug(startTimes, perChunkBinary);

        // ── Step 4: Cluster embeddings ────────────────────────────────────
        // Build (chunkIdx, speakerIdx) → global speaker ID lookup.
        var localToGlobal = new Dictionary<(int, int), int>();

        if (embeddings.Count > 0 && _embedder != null)
        {
            progress?.Invoke($"clustering {embeddings.Count} embedding(s)");
            int[] clusterIds;
            if (embeddings.Count == 1)
            {
                clusterIds = [0];
            }
            else
            {
                int[] trainIndices = SelectTrainingEmbeddingIndices(results, MinCleanFrameRatioForClustering);
                var trainEmbeddings = trainIndices.Select(index => embeddings[index]).ToArray();

                // AHC initialisation on L2-normed 256-dim (matches Python VBxClustering)
                float activeThreshold = ahcThreshold > 0 ? ahcThreshold : Config.DiariZenAhcThreshold;
                var linkage = HierarchicalClustering.Linkage(
                    trainEmbeddings, Config.DiariZenClusteringMethod);
                clusterIds = HierarchicalClustering.FclusterThreshold(linkage, activeThreshold);
                int ahcClusters = clusterIds.Max() + 1;
                progress?.Invoke($"AHC produced {ahcClusters} cluster(s) at threshold {activeThreshold:0.###}");

                // VBx refinement: apply batch plda_tf then run VBx
                if (_embedder != null && _embedder.HasPlda && _embedder.VbxPhi != null
                    && xvecs.Count == embeddings.Count
                    && _embedder.PldaTr != null && _embedder.PldaMu != null)
                {
                    var trainXvecs = trainIndices.Select(index => xvecs[index]).ToArray();
                    // plda_tf: (xvecs - plda_mu) @ plda_tr.T  (no normalization)
                    double[][] feaBatch = WeSpeakerEmbedder.ApplyPldaTfBatch(
                        _embedder.PldaTr, _embedder.PldaMu, trainXvecs);

                    var phi = Array.ConvertAll(_embedder.VbxPhi, x => (double)x);
                    clusterIds = VbxClustering.Cluster(
                        feaBatch, phi, clusterIds,
                        fa: Config.DiariZenVbxFa, fb: Config.DiariZenVbxFb,
                        maxIters: Config.DiariZenVbxMaxIters);
                    clusterIds = MergeSmallClusters(
                        clusterIds,
                        trainEmbeddings,
                        Config.DiariZenMinEmbeddingClusterSize);
                    var centroids = ComputeClusterCentroids(trainEmbeddings, clusterIds);
                    clusterIds = AssignEmbeddingsToCentroids(
                        results,
                        embeddings.ToArray(),
                        centroids,
                        constrained: UseConstrainedCentroidAssignment);
                    progress?.Invoke($"VBx refined clustering to {clusterIds.Distinct().Count()} speaker(s)");
                }
                else
                {
                    clusterIds = MergeSmallClusters(clusterIds, trainEmbeddings, Config.DiariZenMinClusterSize);
                    var centroids = ComputeClusterCentroids(trainEmbeddings, clusterIds);
                    clusterIds = AssignEmbeddingsToCentroids(results, embeddings.ToArray(), centroids, constrained: false);
                    progress?.Invoke($"small-cluster merge produced {clusterIds.Distinct().Count()} speaker(s)");
                }
            }

            for (int i = 0; i < embedKeys.Count; i++)
            {
                if (clusterIds[i] < 0) continue;
                localToGlobal[embedKeys[i]] = clusterIds[i];
            }

            WriteAssignmentDebug(startTimes, localToGlobal);

            progress?.Invoke($"clustered to {clusterIds.Distinct().Count()} speaker(s)");
        }
        else
        {
            // No embedder — assign all local speakers in each chunk to
            // their local index (will produce up to 4 "speakers")
            for (int ci = 0; ci < perChunkBinary.Count; ci++)
                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                    localToGlobal[(ci, spk)] = spk;
        }

        int numGlobalSpeakers = localToGlobal.Count > 0
            ? localToGlobal.Values.Where(id => id >= 0).DefaultIfEmpty(0).Max() + 1
            : MaxUniqueSpeakers;

        // ── Step 5: Reconstruct global timeline ───────────────────────────
        // Mirror pyannote's reconstruct/to_diarization flow:
        // 1. Within each chunk, merge local speakers assigned to the same global
        //    speaker using max over local tracks.
        // 2. Aggregate chunk activations across time by overlap-add (no averaging).
        // 3. Estimate per-frame speaker count by averaging the number of active local
        //    speakers across overlapping chunks, then rounding.
        // 4. For each frame, activate the top-N global speakers where N = count[t].
        int totalFrames = (int)Math.Ceiling(audio.Length / (double)SampleRate * FrameRate);
        var activationSums = new float[totalFrames * numGlobalSpeakers];
        var activationContrib = new int[totalFrames * numGlobalSpeakers];
        var countSums = new float[totalFrames];
        var countContrib = new int[totalFrames];
        var perChunkGlobal = new float[numGlobalSpeakers];
        var chunkHasGlobalSpeaker = new bool[perChunkBinary.Count, numGlobalSpeakers];

        foreach (var ((chunkIdx, _), globalSpeaker) in localToGlobal)
        {
            if (globalSpeaker >= 0)
                chunkHasGlobalSpeaker[chunkIdx, globalSpeaker] = true;
        }

        progress?.Invoke("reconstructing global speaker timeline");
        for (int ci = 0; ci < perChunkBinary.Count; ci++)
        {
            var chunkScore = perChunkBinary[ci];
            int numFrames  = chunkScore.GetLength(0);
            int startFrame = (int)Math.Round(startTimes[ci] * FrameRate);

            for (int t = 0; t < numFrames; t++)
            {
                int gf = startFrame + t;
                if (gf >= totalFrames) break;

                int activeLocalSpeakers = 0;
                Array.Clear(perChunkGlobal, 0, perChunkGlobal.Length);
                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                {
                    float localScore = chunkScore[t, spk];
                    if (localScore > 0f)
                        activeLocalSpeakers++;

                    if (localScore <= 0f) continue;
                    if (!localToGlobal.TryGetValue((ci, spk), out int gs)) continue;
                    perChunkGlobal[gs] = Math.Max(perChunkGlobal[gs], localScore);
                }

                for (int gs = 0; gs < numGlobalSpeakers; gs++)
                {
                    if (!chunkHasGlobalSpeaker[ci, gs]) continue;

                    int flatIndex = gf * numGlobalSpeakers + gs;
                    activationSums[flatIndex] += perChunkGlobal[gs];
                    activationContrib[flatIndex]++;
                }

                countSums[gf] += activeLocalSpeakers;
                countContrib[gf]++;
            }

            if (ShouldReportProgress(ci, perChunkBinary.Count))
                progress?.Invoke($"reconstructed chunk {ci + 1}/{perChunkBinary.Count}");
        }

        // ── Step 6: Select top-N speakers per frame and extract regions ────
        var labeled = new List<(int startFrame, int endFrame, string speaker)>();
        var binary  = new bool[totalFrames * numGlobalSpeakers];
        var ranked  = new (float score, int speaker)[numGlobalSpeakers];
        var debugFrames = CreateReconstructionDebugFrames(totalFrames);

        for (int f = 0; f < totalFrames; f++)
        {
            if (countContrib[f] <= 0) continue;

            double averageCount = countSums[f] / countContrib[f];
            int count = (int)Math.Round(averageCount, MidpointRounding.AwayFromZero);
            count = Math.Clamp(
                count,
                0,
                Math.Min(maxSpeakers, numGlobalSpeakers));

            for (int gs = 0; gs < numGlobalSpeakers; gs++)
            {
                int flatIndex = f * numGlobalSpeakers + gs;
                float score = activationContrib[flatIndex] > 0
                    ? activationSums[flatIndex] / activationContrib[flatIndex]
                    : 0f;
                ranked[gs] = (score, gs);
            }

            Array.Sort(ranked, (a, b) => b.score.CompareTo(a.score));
            for (int i = 0; i < count; i++)
            {
                if (ranked[i].score <= 0f) break;
                binary[f * numGlobalSpeakers + ranked[i].speaker] = true;
            }

            CaptureReconstructionDebugFrame(
                debugFrames,
                f,
                (float)averageCount,
                count,
                ranked,
                binary,
                numGlobalSpeakers);
        }

        if (!DisablePostSmoothing)
        {
            SmoothBinaryTimeline(
                binary,
                totalFrames,
                numGlobalSpeakers,
                Config.GetDiariZenFillShortGapFrames(),
                Config.GetDiariZenMinRegionFrames());
        }

        for (int gs = 0; gs < numGlobalSpeakers; gs++)
        {
            int? regionStart = null;
            for (int f = 0; f < totalFrames; f++)
            {
                bool active = binary[f * numGlobalSpeakers + gs];

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

        WriteReconstructionDebugFrames(debugFrames);
        progress?.Invoke($"extracted {labeled.Count} raw segment(s)");
        if (labeled.Count == 0) return [];

        return MergeAdjacentSegments(labeled);
    }

    private static void SmoothBinaryTimeline(
        bool[] binary,
        int totalFrames,
        int numGlobalSpeakers,
        int fillShortGapFrames,
        int minRegionFrames)
    {
        if (totalFrames <= 0 || numGlobalSpeakers <= 0)
            return;

        for (int gs = 0; gs < numGlobalSpeakers; gs++)
        {
            if (!DisableGapFill)
                FillShortFalseGaps(binary, totalFrames, numGlobalSpeakers, gs, fillShortGapFrames);
            if (!DisableShortRegionRemoval)
                RemoveShortTrueRegions(binary, totalFrames, numGlobalSpeakers, gs, minRegionFrames);
        }
    }

    private static List<ReconstructionDebugFrame>? CreateReconstructionDebugFrames(int totalFrames)
    {
        if (!TryGetDebugReconstructionWindow(out int startFrame, out int endFrame))
            return null;

        startFrame = Math.Clamp(startFrame, 0, totalFrames);
        endFrame = Math.Clamp(endFrame, startFrame, totalFrames);
        return endFrame > startFrame ? new List<ReconstructionDebugFrame>(endFrame - startFrame) : null;
    }

    private static void CaptureReconstructionDebugFrame(
        List<ReconstructionDebugFrame>? debugFrames,
        int frameIndex,
        float averageCount,
        int selectedCount,
        (float score, int speaker)[] ranked,
        bool[] binary,
        int numGlobalSpeakers)
    {
        if (debugFrames is null || !TryGetDebugReconstructionWindow(out int startFrame, out int endFrame))
            return;
        if (frameIndex < startFrame || frameIndex >= endFrame)
            return;

        var scores = new float[numGlobalSpeakers];
        var selected = new bool[numGlobalSpeakers];
        for (int i = 0; i < ranked.Length; i++)
            scores[ranked[i].speaker] = ranked[i].score;
        for (int gs = 0; gs < numGlobalSpeakers; gs++)
            selected[gs] = binary[frameIndex * numGlobalSpeakers + gs];

        debugFrames.Add(new ReconstructionDebugFrame(
            frameIndex,
            frameIndex / (double)FrameRate,
            averageCount,
            selectedCount,
            scores,
            selected));
    }

    private static void WriteReconstructionDebugFrames(List<ReconstructionDebugFrame>? debugFrames)
    {
        if (debugFrames is null)
            return;

        string? outputPath = Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DEBUG_RECON_PATH");
        if (string.IsNullOrWhiteSpace(outputPath))
            return;

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(outputPath, JsonSerializer.Serialize(debugFrames, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
    }

    private static bool TryGetDebugReconstructionWindow(out int startFrame, out int endFrame)
    {
        startFrame = 0;
        endFrame = 0;

        string? raw = Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DEBUG_RECON_WINDOW");
        if (string.IsNullOrWhiteSpace(raw))
            return false;

        string[] parts = raw.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2)
            return false;

        if (!double.TryParse(parts[0], out double startSeconds) ||
            !double.TryParse(parts[1], out double endSeconds))
            return false;

        startFrame = (int)Math.Floor(startSeconds * FrameRate);
        endFrame = (int)Math.Ceiling(endSeconds * FrameRate);
        return endFrame > startFrame;
    }

    private static int GetMinActiveFramesForEmbed()
    {
        string? raw = Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_MIN_ACTIVE_FRAMES_FOR_EMBED");
        return int.TryParse(raw, out int parsed) && parsed >= 0
            ? parsed
            : DefaultMinActiveFramesForEmbed;
    }

    private static void WriteAssignmentDebug(
        List<double> startTimes,
        Dictionary<(int chunkIdx, int localSpeaker), int> localToGlobal)
    {
        string? outputPath = Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DEBUG_ASSIGN_PATH");
        if (string.IsNullOrWhiteSpace(outputPath))
            return;

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        var payload = startTimes.Select((startTime, chunkIdx) => new
        {
            chunkIdx,
            startTime,
            assignments = localToGlobal
                .Where(kv => kv.Key.chunkIdx == chunkIdx)
                .OrderBy(kv => kv.Key.localSpeaker)
                .Select(kv => new
                {
                    localSpeaker = kv.Key.localSpeaker,
                    globalSpeaker = kv.Value
                })
                .ToArray()
        });

        File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
    }

    private static void WriteLocalMaskDebug(
        List<double> startTimes,
        List<float[,]> perChunkBinary)
    {
        string? outputPath = Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DEBUG_LOCAL_PATH");
        if (string.IsNullOrWhiteSpace(outputPath))
            return;

        if (!TryGetDebugReconstructionWindow(out int startFrame, out int endFrame))
            return;

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        var payload = new List<object>();
        for (int ci = 0; ci < perChunkBinary.Count; ci++)
        {
            float[,] chunkBinary = perChunkBinary[ci];
            int numFrames = chunkBinary.GetLength(0);
            int chunkStartFrame = (int)Math.Round(startTimes[ci] * FrameRate);
            var frames = new List<object>();

            for (int t = 0; t < numFrames; t++)
            {
                int globalFrame = chunkStartFrame + t;
                if (globalFrame < startFrame || globalFrame >= endFrame)
                    continue;

                var mask = new float[MaxUniqueSpeakers];
                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                    mask[spk] = chunkBinary[t, spk];

                frames.Add(new
                {
                    localFrame = t,
                    globalFrame,
                    globalTimeSeconds = globalFrame / (double)FrameRate,
                    localMask = mask
                });
            }

            if (frames.Count == 0)
                continue;

            payload.Add(new
            {
                chunkIdx = ci,
                startTime = startTimes[ci],
                frames
            });
        }

        File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
    }

    private static void WriteEmbeddingDebug(
        List<double> startTimes,
        List<EmbeddingJob> jobs,
        EmbeddingJobResult[] results)
    {
        string? outputPath = Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DEBUG_EMBED_PATH");
        if (string.IsNullOrWhiteSpace(outputPath))
            return;

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        var payload = new List<object>(results.Length);
        for (int i = 0; i < results.Length; i++)
        {
            var job = jobs[i];
            var result = results[i];
            payload.Add(new
            {
                index = i,
                chunkIdx = result.ChunkIdx,
                chunkStartTime = startTimes[result.ChunkIdx],
                localSpeaker = result.SpeakerIdx,
                numFrames = result.NumFrames,
                cleanFrameCount = result.CleanFrameCount,
                selectedFrameCount = job.SelectedMask.Count(v => v),
                selectedMask = job.SelectedMask,
                l2Embedding = result.L2Embedding,
                xvec = result.Xvec,
            });
        }

        File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true
        }));
    }

    private static void FillShortFalseGaps(
        bool[] binary,
        int totalFrames,
        int numGlobalSpeakers,
        int speakerIndex,
        int maxGapFrames)
    {
        if (maxGapFrames <= 0)
            return;

        int frame = 0;
        while (frame < totalFrames)
        {
            while (frame < totalFrames && !binary[frame * numGlobalSpeakers + speakerIndex])
                frame++;

            if (frame >= totalFrames)
                break;

            while (frame < totalFrames && binary[frame * numGlobalSpeakers + speakerIndex])
                frame++;

            int gapStart = frame;
            while (frame < totalFrames && !binary[frame * numGlobalSpeakers + speakerIndex])
                frame++;

            if (gapStart == 0 || frame >= totalFrames)
                continue;

            int gapLength = frame - gapStart;
            if (gapLength <= maxGapFrames)
            {
                for (int fill = gapStart; fill < frame; fill++)
                    binary[fill * numGlobalSpeakers + speakerIndex] = true;
            }
        }
    }

    private static void RemoveShortTrueRegions(
        bool[] binary,
        int totalFrames,
        int numGlobalSpeakers,
        int speakerIndex,
        int minRegionFrames)
    {
        if (minRegionFrames <= 1)
            return;

        int frame = 0;
        while (frame < totalFrames)
        {
            while (frame < totalFrames && !binary[frame * numGlobalSpeakers + speakerIndex])
                frame++;

            if (frame >= totalFrames)
                break;

            int regionStart = frame;
            while (frame < totalFrames && binary[frame * numGlobalSpeakers + speakerIndex])
                frame++;

            int regionLength = frame - regionStart;
            if (regionLength <= minRegionFrames)
            {
                for (int clear = regionStart; clear < frame; clear++)
                    binary[clear * numGlobalSpeakers + speakerIndex] = false;
            }
        }
    }

    // ── Chunking ───────────────────────────────────────────────────────────

    private IEnumerable<ChunkInfo> EnumerateChunks(float[] audio)
    {
        int chunkSamples  = ChunkDurationSeconds * SampleRate;
        int strideSamples = (int)(chunkSamples * SegmentationStep);  // 25 600 = 1.6 s

        int chunkIndex = 0;
        for (int start = 0; start < audio.Length; start += strideSamples)
        {
            var chunk = new float[chunkSamples];
            int end   = Math.Min(start + chunkSamples, audio.Length);
            Array.Copy(audio, start, chunk, 0, end - start);
            yield return new ChunkInfo(chunkIndex++, chunk, start / (double)SampleRate);
        }
    }

    // ── Segmentation inference ─────────────────────────────────────────────

    private IEnumerable<List<ChunkInfo>> EnumerateChunkBatches(float[] audio, int batchSize)
    {
        batchSize = Math.Max(1, batchSize);
        var batch = new List<ChunkInfo>(batchSize);
        foreach (var chunk in EnumerateChunks(audio))
        {
            batch.Add(chunk);
            if (batch.Count == batchSize)
            {
                yield return batch;
                batch = new List<ChunkInfo>(batchSize);
            }
        }

        if (batch.Count > 0)
            yield return batch;
    }

    private static bool ShouldReportProgress(int index, int total)
    {
        if (total <= 0) return false;
        if (index == total - 1) return true;
        if (total <= 10) return true;

        int interval = Math.Max(1, total / 10);
        return index == 0 || ((index + 1) % interval) == 0;
    }

    private readonly record struct ChunkInfo(
        int ChunkIndex,
        float[] Chunk,
        double StartTime);
    private readonly record struct EmbeddingJob(
        int SequenceIndex,
        int ChunkIdx,
        int SpeakerIdx,
        bool[] SelectedMask,
        int NumFrames,
        int CleanFrameCount);
    private readonly record struct ChunkEmbeddingPrepRequest(
        int ChunkIdx,
        float[] ChunkWaveform,
        List<EmbeddingJob> Jobs);
    private readonly record struct EmbeddingJobResult(
        int SequenceIndex,
        int ChunkIdx,
        int SpeakerIdx,
        int NumFrames,
        int CleanFrameCount,
        float[] L2Embedding,
        float[]? Xvec);

    private float[] SegmentSingle(InferenceSession session, float[] chunk)
    {
        var tensor = new DenseTensor<float>(chunk, new[] { 1, 1, chunk.Length });
        var inputs = new List<NamedOnnxValue>
            { NamedOnnxValue.CreateFromTensor(_segInputName, tensor) };

        using var onnxOut = session.Run(inputs);
        var outTensor     = onnxOut.First(r => r.Name == _segOutputName).AsTensor<float>();

        int frames  = outTensor.Dimensions[1];
        int classes = outTensor.Dimensions[2];
        var scores  = new float[frames * classes];
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < classes; c++)
                scores[f * classes + c] = outTensor[0, f, c];
        return scores;
    }

    private List<float[]> SegmentBatch(IReadOnlyList<float[]> chunks)
    {
        if (chunks.Count == 0)
            return [];

        if (_segBatchSize > 1 && chunks.Count > 1 && Volatile.Read(ref _segBatchingDisabled) == 0)
        {
            try
            {
                return SegmentBatchSingleSession(chunks);
            }
            catch (OnnxRuntimeException)
            {
                Interlocked.Exchange(ref _segBatchingDisabled, 1);
            }
        }

        if (_segSessions.Length == 1 || chunks.Count == 1)
            return chunks.Select(chunk => SegmentSingle(_segSessions[0], chunk)).ToList();

        var results = new float[chunks.Count][];
        int nextSession = -1;
        using var sessionLease = new ThreadLocal<InferenceSession>(
            () => _segSessions[Interlocked.Increment(ref nextSession) % _segSessions.Length]);

        Parallel.For(
            0,
            chunks.Count,
            new ParallelOptions { MaxDegreeOfParallelism = _segSessions.Length },
            ci => results[ci] = SegmentSingle(sessionLease.Value!, chunks[ci]));

        return new List<float[]>(results);
    }

    private List<float[]> SegmentBatchSingleSession(IReadOnlyList<float[]> chunks)
    {
        int batch = chunks.Count;
        int chunkLength = chunks[0].Length;
        var data = new float[batch * chunkLength];
        for (int i = 0; i < batch; i++)
            Array.Copy(chunks[i], 0, data, i * chunkLength, chunkLength);

        var tensor = new DenseTensor<float>(data, new[] { batch, 1, chunkLength });
        var inputs = new List<NamedOnnxValue>
            { NamedOnnxValue.CreateFromTensor(_segInputName, tensor) };

        using var onnxOut = _segSessions[0].Run(inputs);
        var outTensor = onnxOut.First(r => r.Name == _segOutputName).AsTensor<float>();
        int frames = outTensor.Dimensions[1];
        int classes = outTensor.Dimensions[2];
        var results = new List<float[]>(batch);
        for (int b = 0; b < batch; b++)
        {
            var scores = new float[frames * classes];
            for (int f = 0; f < frames; f++)
                for (int c = 0; c < classes; c++)
                    scores[f * classes + c] = outTensor[b, f, c];
            results.Add(scores);
        }

        return results;
    }

    private static int DetermineSegmentationWorkerCount(ExecutionProvider ep)
    {
        int? configuredWorkers = Config.GetDiariZenSegmentationMaxWorkers();
        if (configuredWorkers.HasValue)
            return configuredWorkers.Value;

        if (ep == ExecutionProvider.Cuda || ep == ExecutionProvider.DirectML)
            return 1;
        if (ep == ExecutionProvider.Auto && HardwareInfo.CanProbeCudaExecutionProvider())
            return 1;

        int intraOpThreads = Math.Max(1, Config.GetDiariZenSegmentationIntraOpThreads());
        int availableCores = Math.Max(1, Environment.ProcessorCount);
        return Math.Clamp(availableCores / intraOpThreads, 1, 4);
    }

    private static int GetChunkCount(int audioLength)
    {
        if (audioLength <= 0)
            return 0;

        int chunkSamples = ChunkDurationSeconds * SampleRate;
        int strideSamples = (int)(chunkSamples * SegmentationStep);
        return 1 + Math.Max(0, (audioLength - 1) / strideSamples);
    }

    private static bool SupportsDynamicBatch(InferenceSession session, string inputName)
    {
        if (!session.InputMetadata.TryGetValue(inputName, out var metadata))
            return false;

        if (metadata.Dimensions.Length == 0)
            return false;

        int batchDim = metadata.Dimensions[0];
        return batchDim <= 0;
    }

    private static Task[] StartEmbeddingPrepWorkers(
        WeSpeakerEmbedder embedder,
        BlockingCollection<ChunkEmbeddingPrepRequest> prepQueue,
        ConcurrentDictionary<int, WeSpeakerEmbedder.PreparedEmbeddingInput> preparedMap,
        Func<int> totalJobsProvider,
        Action<string>? progress,
        int[] completedJobs,
        int[] lastReportedJobs)
    {
        var workers = new Task[MaxEmbeddingWorkers];
        for (int workerIndex = 0; workerIndex < workers.Length; workerIndex++)
        {
            workers[workerIndex] = Task.Run(() =>
            {
                foreach (var request in prepQueue.GetConsumingEnumerable())
                {
                    float[,] fbank = embedder.ComputeChunkFbank(request.ChunkWaveform);
                    foreach (var job in request.Jobs)
                    {
                        var prepared = embedder.PrepareEmbeddingInput(fbank, job.SelectedMask);
                        if (prepared == null)
                            continue;

                        preparedMap[job.SequenceIndex] = prepared.Value;
                        int done = Interlocked.Increment(ref completedJobs[0]);
                        int totalJobs = totalJobsProvider();
                        int reportInterval = Math.Max(1, Math.Max(done, totalJobs) / 10);
                        if (done == totalJobs || done - Volatile.Read(ref lastReportedJobs[0]) >= reportInterval)
                        {
                            int previous = Interlocked.Exchange(ref lastReportedJobs[0], done);
                            if (done != previous)
                                progress?.Invoke($"prepared embeddings {done}/{totalJobs}");
                        }
                    }
                }
            });
        }

        return workers;
    }

    private EmbeddingJobResult[] ComputeEmbeddingResults(
        WeSpeakerEmbedder embedder,
        List<EmbeddingJob> jobs,
        ConcurrentDictionary<int, WeSpeakerEmbedder.PreparedEmbeddingInput> preparedMap,
        Action<string>? progress)
    {
        var results = new EmbeddingJobResult[jobs.Count];
        if (_preferGpuEmbeddingPipeline)
        {
            int maxBatchSize = Config.GetDiariZenEmbeddingGpuMaxBatchSize();
            int maxBatchFrames = embedder.ComputeSuggestedGpuBatchFrameLimit();
            var preparedJobs = jobs
                .Where(job => preparedMap.ContainsKey(job.SequenceIndex))
                .Select(job => new WeSpeakerEmbedder.PreparedEmbeddingJob(
                    job.SequenceIndex,
                    preparedMap[job.SequenceIndex]))
                .ToList();

            int completed = 0;
            foreach (var batch in MakeEmbeddingBatches(preparedJobs, maxBatchSize, maxBatchFrames))
            {
                foreach (var batchResult in embedder.ComputeEmbeddings(batch))
                {
                    var job = jobs[batchResult.SequenceIndex];
                    results[job.SequenceIndex] = new EmbeddingJobResult(
                        job.SequenceIndex,
                        job.ChunkIdx,
                        job.SpeakerIdx,
                        job.NumFrames,
                        job.CleanFrameCount,
                        batchResult.L2Embedding,
                        embedder.ComputeXvec(batchResult.RawEmbedding));
                    completed++;
                    if (completed == jobs.Count || completed % Math.Max(1, jobs.Count / 10) == 0)
                        progress?.Invoke($"computed embeddings {completed}/{jobs.Count}");
                }
            }

            return results;
        }

        Parallel.For(
            0,
            jobs.Count,
            new ParallelOptions { MaxDegreeOfParallelism = MaxEmbeddingWorkers },
            jobIndex =>
            {
                var job = jobs[jobIndex];
                if (!preparedMap.TryGetValue(job.SequenceIndex, out var prepared))
                    return;
                float[] l2emb = embedder.ComputeEmbedding(prepared, out float[] rawEmb);
                results[job.SequenceIndex] = new EmbeddingJobResult(
                    job.SequenceIndex,
                    job.ChunkIdx,
                    job.SpeakerIdx,
                    job.NumFrames,
                    job.CleanFrameCount,
                    l2emb,
                    embedder.ComputeXvec(rawEmb));
            });

        return results;
    }

    private static IEnumerable<List<WeSpeakerEmbedder.PreparedEmbeddingJob>> MakeEmbeddingBatches(
        List<WeSpeakerEmbedder.PreparedEmbeddingJob> jobs,
        int maxBatchSize,
        int maxBatchFrames)
    {
        var current = new List<WeSpeakerEmbedder.PreparedEmbeddingJob>();
        int currentMaxFrames = 0;

        foreach (var job in jobs)
        {
            int frames = job.Prepared.Fbank.GetLength(0);
            int projectedMaxFrames = Math.Max(currentMaxFrames, frames);
            long projectedFrames = (long)(current.Count + 1) * projectedMaxFrames;
            bool sizeLimit = current.Count >= maxBatchSize;
            bool frameLimit = current.Count > 0 && projectedFrames > maxBatchFrames;
            if (sizeLimit || frameLimit)
            {
                yield return current;
                current = new List<WeSpeakerEmbedder.PreparedEmbeddingJob>();
                currentMaxFrames = 0;
            }

            current.Add(job);
            currentMaxFrames = Math.Max(currentMaxFrames, frames);
        }

        if (current.Count > 0)
            yield return current;
    }

    private static void MaterializeEmbeddingOutputs(
        EmbeddingJobResult[] results,
        List<double[]> embeddings,
        List<float[]> xvecs,
        List<(int chunkIdx, int speakerIdx)> embedKeys)
    {
        for (int i = 0; i < results.Length; i++)
        {
            var result = results[i];
            var emb = new double[result.L2Embedding.Length];
            for (int d = 0; d < result.L2Embedding.Length; d++) emb[d] = result.L2Embedding[d];
            embeddings.Add(emb);
            if (result.Xvec != null) xvecs.Add(result.Xvec);
            embedKeys.Add((result.ChunkIdx, result.SpeakerIdx));
        }
    }

    private static InferenceSession CreateSegmentationSession(
        string segmentationModelPath,
        ExecutionProvider ep)
    {
        var opts = new SessionOptions
        {
            IntraOpNumThreads = Config.GetDiariZenSegmentationIntraOpThreads()
        };

        switch (ep)
        {
            case ExecutionProvider.Auto:
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try { opts.AppendExecutionProvider_CUDA(0); } catch { }
                }
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

        return new InferenceSession(segmentationModelPath, opts);
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

    private static float[,] DecodeChunkBinary(float[] logits, float threshold, int medianFilterSize)
    {
        float[] probs = ApplySoftmaxSingle(logits);
        int numFrames = probs.Length / NumPowersetClasses;
        var scores = new float[numFrames * MaxUniqueSpeakers];

        if (medianFilterSize <= 1)
        {
            AccumulateSpeakerScores(probs, scores);
            return BinarizeSpeakerScores(scores, numFrames, threshold);
        }

        int half = medianFilterSize / 2;
        var window = new float[medianFilterSize];
        for (int c = 0; c < NumPowersetClasses; c++)
        {
            int[] speakers = PowersetCombinations[c];
            for (int f = 0; f < numFrames; f++)
            {
                int count = 0;
                for (int k = -half; k <= half; k++)
                {
                    int fk = f + k;
                    if (fk < 0) fk = -fk;
                    if (fk >= numFrames) fk = 2 * numFrames - 2 - fk;
                    fk = Math.Clamp(fk, 0, numFrames - 1);
                    window[count++] = probs[fk * NumPowersetClasses + c];
                }

                Array.Sort(window, 0, count);
                float median = window[count / 2];
                if (median == 0f)
                    continue;

                int scoreBase = f * MaxUniqueSpeakers;
                for (int i = 0; i < speakers.Length; i++)
                    scores[scoreBase + speakers[i]] += median;
            }
        }

        return BinarizeSpeakerScores(scores, numFrames, threshold);
    }

    private static void AccumulateSpeakerScores(float[] probs, float[] scores)
    {
        int numFrames = probs.Length / NumPowersetClasses;
        for (int f = 0; f < numFrames; f++)
        {
            int powersetBase = f * NumPowersetClasses;
            int scoreBase = f * MaxUniqueSpeakers;
            for (int c = 0; c < NumPowersetClasses; c++)
            {
                float p = probs[powersetBase + c];
                if (p == 0f)
                    continue;

                int[] speakers = PowersetCombinations[c];
                for (int i = 0; i < speakers.Length; i++)
                    scores[scoreBase + speakers[i]] += p;
            }
        }
    }

    private static float[,] BinarizeSpeakerScores(float[] scores, int numFrames, float threshold)
    {
        var binary = new float[numFrames, MaxUniqueSpeakers];
        for (int f = 0; f < numFrames; f++)
        {
            int scoreBase = f * MaxUniqueSpeakers;
            for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                binary[f, spk] = scores[scoreBase + spk] >= threshold ? 1f : 0f;
        }

        return binary;
    }

    // ── Powerset → soft per-speaker scores ────────────────────────────────

    /// <summary>
    /// Replicates pyannote's Powerset.to_multilabel(soft=True):
    /// for each speaker s, score[t, s] = Σ_c p(c|t) for all powerset classes c where s is active.
    /// Returns float[numFrames, MaxUniqueSpeakers] with values in [0, 1].
    /// Using soft scores (rather than hard argmax) allows overlap regions where multiple
    /// speakers each exceed the binarization threshold simultaneously.
    /// </summary>
    private static float[,] ComputePerSpeakerScores(float[] probs)
    {
        var combinations = PowersetDecoder.GetPowersetCombinations(
            MaxUniqueSpeakers, MaxSimultaneousPerFrame);
        int numFrames    = probs.Length / NumPowersetClasses;
        var result       = new float[numFrames, MaxUniqueSpeakers];

        for (int f = 0; f < numFrames; f++)
        {
            int b = f * NumPowersetClasses;
            for (int c = 0; c < NumPowersetClasses; c++)
            {
                float p = probs[b + c];
                if (p == 0f) continue;
                foreach (int spk in combinations[c])
                    result[f, spk] += p;
            }
        }

        return result;
    }

    private static float[,] BinarizePerSpeakerScores(float[,] scores, float threshold)
    {
        int numFrames = scores.GetLength(0);
        int numSpeakers = scores.GetLength(1);
        var binary = new float[numFrames, numSpeakers];

        for (int t = 0; t < numFrames; t++)
        for (int spk = 0; spk < numSpeakers; spk++)
            binary[t, spk] = scores[t, spk] >= threshold ? 1f : 0f;

        return binary;
    }

    // ── Small-cluster absorption ───────────────────────────────────────────

    /// <summary>
    /// Reassigns embeddings in clusters smaller than <paramref name="minClusterSize"/>
    /// to the nearest large-cluster centroid (Euclidean on L2-normalised vectors).
    /// Mirrors pyannote's AgglomerativeClustering min_cluster_size post-processing.
    /// Returns a compact 0-based relabelling.
    /// </summary>
    private static int[] MergeSmallClusters(int[] clusterIds, double[][] embeddings, int minClusterSize)
    {
        // Count embeddings per cluster
        var counts = new Dictionary<int, int>();
        foreach (int id in clusterIds)
            counts[id] = counts.GetValueOrDefault(id, 0) + 1;

        var largeSet = new HashSet<int>(counts.Where(kv => kv.Value >= minClusterSize).Select(kv => kv.Key));
        var smallSet = new HashSet<int>(counts.Where(kv => kv.Value <  minClusterSize).Select(kv => kv.Key));

        // Nothing to do if all clusters are large (or there are no large clusters to absorb into)
        if (smallSet.Count == 0 || largeSet.Count == 0) return CompactIds(clusterIds);

        int dim = embeddings[0].Length;

        // Compute centroid for each large cluster
        var centroids = new Dictionary<int, double[]>();
        foreach (int lc in largeSet)
        {
            var c = new double[dim];
            int n = 0;
            for (int i = 0; i < clusterIds.Length; i++)
            {
                if (clusterIds[i] != lc) continue;
                for (int d = 0; d < dim; d++) c[d] += embeddings[i][d];
                n++;
            }
            for (int d = 0; d < dim; d++) c[d] /= n;
            centroids[lc] = c;
        }

        // Reassign small-cluster embeddings to nearest large centroid
        var result = (int[])clusterIds.Clone();
        for (int i = 0; i < result.Length; i++)
        {
            if (!smallSet.Contains(result[i])) continue;

            double bestDist = double.MaxValue;
            int    bestId   = largeSet.First();
            foreach (var (lc, centroid) in centroids)
            {
                double dist = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = embeddings[i][d] - centroid[d];
                    dist += diff * diff;
                }
                if (dist < bestDist) { bestDist = dist; bestId = lc; }
            }
            result[i] = bestId;
        }

        return CompactIds(result);
    }

    private static int[] CompactIds(int[] ids)
    {
        var map = new Dictionary<int, int>();
        int next = 0;
        var result = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            if (!map.TryGetValue(ids[i], out int mapped))
                map[ids[i]] = mapped = next++;
            result[i] = mapped;
        }
        return result;
    }

    private static int[] SelectTrainingEmbeddingIndices(
        EmbeddingJobResult[] results,
        double minCleanFrameRatio)
    {
        if (results.Length <= 1)
            return Enumerable.Range(0, results.Length).ToArray();

        var indices = new List<int>(results.Length);
        for (int i = 0; i < results.Length; i++)
        {
            int minCleanFrames = Math.Max(1, (int)Math.Round(results[i].NumFrames * minCleanFrameRatio));
            if (results[i].CleanFrameCount >= minCleanFrames)
                indices.Add(i);
        }

        if (indices.Count >= 2)
            return indices.ToArray();

        return Enumerable.Range(0, results.Length).ToArray();
    }

    private static double[][] ComputeClusterCentroids(double[][] embeddings, int[] clusterIds)
    {
        int numClusters = clusterIds.Max() + 1;
        int dim = embeddings[0].Length;
        var centroids = new double[numClusters][];
        var counts = new int[numClusters];

        for (int k = 0; k < numClusters; k++)
            centroids[k] = new double[dim];

        for (int i = 0; i < embeddings.Length; i++)
        {
            int cluster = clusterIds[i];
            counts[cluster]++;
            for (int d = 0; d < dim; d++)
                centroids[cluster][d] += embeddings[i][d];
        }

        for (int k = 0; k < numClusters; k++)
        {
            if (counts[k] == 0) continue;
            for (int d = 0; d < dim; d++)
                centroids[k][d] /= counts[k];
        }

        return centroids;
    }

    private static int[] AssignEmbeddingsToCentroids(
        EmbeddingJobResult[] results,
        double[][] embeddings,
        double[][] centroids,
        bool constrained)
    {
        if (results.Length == 0 || centroids.Length == 0)
            return [];

        var assigned = Enumerable.Repeat(-1, results.Length).ToArray();
        var byChunk = new Dictionary<int, List<int>>();
        for (int i = 0; i < results.Length; i++)
        {
            if (!byChunk.TryGetValue(results[i].ChunkIdx, out var list))
            {
                list = [];
                byChunk[results[i].ChunkIdx] = list;
            }
            list.Add(i);
        }

        foreach (var (_, chunkIndices) in byChunk)
        {
            if (!constrained || chunkIndices.Count == 1 || centroids.Length == 1)
            {
                foreach (int index in chunkIndices)
                    assigned[index] = BestCentroidIndex(embeddings[index], centroids);
                continue;
            }

            var scoreMatrix = new double[chunkIndices.Count, centroids.Length];
            for (int row = 0; row < chunkIndices.Count; row++)
                for (int col = 0; col < centroids.Length; col++)
                    scoreMatrix[row, col] = CosineSimilarity(embeddings[chunkIndices[row]], centroids[col]);

            var rowToCluster = BestUniqueAssignment(scoreMatrix);
            for (int row = 0; row < chunkIndices.Count; row++)
            {
                int cluster = rowToCluster[row];
                if (cluster >= 0)
                    assigned[chunkIndices[row]] = cluster;
            }
        }

        var activeAssignments = assigned.Where(id => id >= 0).ToArray();
        if (activeAssignments.Length == 0)
            return Enumerable.Repeat(0, assigned.Length).ToArray();

        var compactMap = new Dictionary<int, int>();
        int nextId = 0;
        for (int i = 0; i < assigned.Length; i++)
        {
            int id = assigned[i];
            if (id < 0) continue;
            if (!compactMap.TryGetValue(id, out int mapped))
                compactMap[id] = mapped = nextId++;
            assigned[i] = mapped;
        }

        return assigned;
    }

    private static int BestCentroidIndex(double[] embedding, double[][] centroids)
    {
        int bestIndex = 0;
        double bestScore = double.NegativeInfinity;
        for (int i = 0; i < centroids.Length; i++)
        {
            double score = CosineSimilarity(embedding, centroids[i]);
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private static int[] BestUniqueAssignment(double[,] scoreMatrix)
    {
        int rows = scoreMatrix.GetLength(0);
        int cols = scoreMatrix.GetLength(1);
        var assignment = Enumerable.Repeat(-1, rows).ToArray();
        var best = Enumerable.Repeat(-1, rows).ToArray();
        var usedCols = new bool[cols];
        double bestScore = double.NegativeInfinity;
        int targetAssignments = Math.Min(rows, cols);

        void Search(int row, int assignedCount, double totalScore)
        {
            if (row == rows)
            {
                if (assignedCount == targetAssignments && totalScore > bestScore)
                {
                    bestScore = totalScore;
                    Array.Copy(assignment, best, rows);
                }
                return;
            }

            if (rows - row < targetAssignments - assignedCount)
                return;

            if (assignedCount < targetAssignments)
            {
                for (int col = 0; col < cols; col++)
                {
                    if (usedCols[col]) continue;
                    usedCols[col] = true;
                    assignment[row] = col;
                    Search(row + 1, assignedCount + 1, totalScore + scoreMatrix[row, col]);
                    assignment[row] = -1;
                    usedCols[col] = false;
                }
            }

            if (assignedCount + (rows - row - 1) >= targetAssignments)
                Search(row + 1, assignedCount, totalScore);
        }

        Search(0, 0, 0.0);
        return best;
    }

    private static double CosineSimilarity(double[] a, double[] b)
    {
        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA <= 0.0 || normB <= 0.0)
            return double.NegativeInfinity;

        return dot / Math.Sqrt(normA * normB);
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
            foreach (var segSession in _segSessions)
                segSession.Dispose();
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

internal sealed record ReconstructionDebugFrame(
    int Frame,
    double TimeSeconds,
    float AverageCount,
    int SelectedCount,
    float[] Scores,
    bool[] Selected);
