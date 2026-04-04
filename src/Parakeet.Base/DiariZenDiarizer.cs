using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Parakeet.Base.Models;
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
    private const int DefaultMinActiveFramesForEmbed = 10;
    private const double MinCleanFrameRatioForClustering = 0.1;
    private static readonly int MaxEmbeddingWorkers = Math.Max(1, Environment.ProcessorCount - 2);
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
    private static readonly bool DisableTinySpeakerPruning =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DISABLE_TINY_SPEAKER_PRUNING"),
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
    private static readonly bool DisableAdjacentMerge =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_DISABLE_ADJACENT_MERGE"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool UseHardPowersetMasks =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_USE_HARD_POWERSET_MASKS"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool UseHardPowersetCount =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_USE_HARD_POWERSET_COUNT"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool ProjectSoftMasksToHardCount =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_PROJECT_SOFT_MASKS_TO_COUNT"),
            "1",
            StringComparison.Ordinal);
    private static readonly bool UseSummedActivations =
        string.Equals(
            Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_USE_SUMMED_ACTIVATIONS"),
            "1",
            StringComparison.Ordinal);
    private static readonly string CountRoundingMode =
        (Environment.GetEnvironmentVariable("PARAKEET_DIARIZEN_COUNT_ROUNDING") ?? "round")
        .Trim()
        .ToLowerInvariant();
    private static readonly int MinActiveFramesForEmbed = GetMinActiveFramesForEmbed();

    // ── Construction ───────────────────────────────────────────────────────

    public DiariZenDiarizer(
        string segmentationModelPath,
        string? embeddingModelPath = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        var opts = new SessionOptions();
        opts.IntraOpNumThreads = Config.GetDiariZenSegmentationIntraOpThreads();

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

        _segSession = new InferenceSession(segmentationModelPath, opts);

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

        var (chunks, startTimes) = ChunkAudio(audio);
        progress?.Invoke($"chunked audio into {chunks.Count} window(s)");

        // ── Step 1: Per-chunk powerset inference ──────────────────────────
        progress?.Invoke("running segmentation model");
        var chunkScores = SegmentBatched(chunks, progress);

        // ── Step 2: Per-chunk decode → soft per-speaker scores (T × 4) ──
        // Matches pyannote's Inference.infer with soft=True + to_multilabel:
        // score[t, spk] = Σ_c p(c|t)  for all powerset classes c containing spk.
        // Followed by median_filter(size=(1,11,1)).
        var perChunkScores = new List<float[,]>(chunks.Count);
        var perChunkBinary = new List<float[,]>(chunks.Count);
        var perChunkCounts = new List<int[]>(chunks.Count);
        for (int ci = 0; ci < chunkScores.Count; ci++)
        {
            var rawScores = chunkScores[ci];
            var probs    = ApplySoftmaxSingle(rawScores);
            var filtered = ApplyMedianFilterSingle(probs, Config.DiariZenMedianFilterSize);
            var softScores = ComputePerSpeakerScores(filtered);
            perChunkScores.Add(softScores);
            int[] hardCounts = DecodePowersetCardinality(filtered, threshold);
            perChunkCounts.Add(hardCounts);
            perChunkBinary.Add(
                UseHardPowersetMasks
                    ? DecodeHardPowersetMasks(filtered, threshold)
                    : ProjectSoftMasksToHardCount
                        ? ProjectSoftScoresToFrameCount(softScores, hardCounts, threshold)
                        : BinarizePerSpeakerScores(softScores, threshold));
            if (ShouldReportProgress(ci, chunkScores.Count))
                progress?.Invoke($"decoded chunk {ci + 1}/{chunkScores.Count}");
        }

        // ── Step 3: Extract per-(chunk, local_speaker) embeddings ─────────
        // embeddings — L2-normed 256-dim, used for AHC initialisation
        // xvecs      — 128-dim xvec (after xvec_tf but before batch plda_tf), collected for VBx
        var embeddings = new List<double[]>();
        var xvecs      = new List<float[]>();   // raw xvec before batch plda_tf
        var embedKeys  = new List<(int chunkIdx, int speakerIdx)>();
        var results    = Array.Empty<EmbeddingJobResult>();

        if (_embedder != null)
        {
            progress?.Invoke("extracting speaker embeddings");
            var jobs = new List<EmbeddingJob>();
            for (int ci = 0; ci < perChunkScores.Count; ci++)
            {
                var chunkScore = perChunkBinary[ci];
                int numFrames  = chunkScore.GetLength(0);
                var overlapCounts = new int[numFrames];
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
                                numFrames * (_embedder.MinNumSamples / (double)chunks[ci].Length)));
                        selectedMask = cleanCount > minMaskFrames ? cleanMask : activeMask;
                    }
                    else
                    {
                        if (activeCount < MinActiveFramesForEmbed) continue;
                        selectedMask = cleanCount >= MinActiveFramesForEmbed ? cleanMask : activeMask;
                    }

                    jobs.Add(new EmbeddingJob(ci, spk, chunks[ci], selectedMask, numFrames, cleanCount));
                }

                if (ShouldReportProgress(ci, perChunkScores.Count))
                    progress?.Invoke($"processed embeddings for chunk {ci + 1}/{perChunkScores.Count}");
            }

            results = new EmbeddingJobResult[jobs.Count];
            int completedJobs = 0;
            int lastReported = 0;
            int reportInterval = Math.Max(1, jobs.Count / 10);

            Parallel.For(
                0,
                jobs.Count,
                new ParallelOptions { MaxDegreeOfParallelism = MaxEmbeddingWorkers },
                jobIndex =>
                {
                    var job = jobs[jobIndex];
                    float[] l2emb = _embedder.ComputeEmbedding(job.ChunkWaveform, job.SelectedMask, out float[] rawEmb);
                    float[]? xv = _embedder.ComputeXvec(rawEmb);
                    results[jobIndex] = new EmbeddingJobResult(
                        job.ChunkIdx,
                        job.SpeakerIdx,
                        job.NumFrames,
                        job.CleanFrameCount,
                        l2emb,
                        xv);

                    int done = Interlocked.Increment(ref completedJobs);
                    if (done == jobs.Count || done - Volatile.Read(ref lastReported) >= reportInterval)
                    {
                        int previous = Interlocked.Exchange(ref lastReported, done);
                        if (done != previous)
                            progress?.Invoke($"computed embeddings {done}/{jobs.Count}");
                    }
                });

            for (int i = 0; i < results.Length; i++)
            {
                var result = results[i];
                var emb = new double[result.L2Embedding.Length];
                for (int d = 0; d < result.L2Embedding.Length; d++) emb[d] = result.L2Embedding[d];
                embeddings.Add(emb);
                if (result.Xvec != null) xvecs.Add(result.Xvec);
                embedKeys.Add((result.ChunkIdx, result.SpeakerIdx));
            }

            WriteEmbeddingDebug(startTimes, jobs, results);
        }

        progress?.Invoke($"collected {embeddings.Count} embedding(s) from {perChunkScores.Count} chunk(s)");
        WriteLocalMaskDebug(startTimes, perChunkBinary, perChunkCounts);

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
            for (int ci = 0; ci < perChunkScores.Count; ci++)
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
        var chunkHasGlobalSpeaker = new bool[perChunkScores.Count, numGlobalSpeakers];

        foreach (var ((chunkIdx, _), globalSpeaker) in localToGlobal)
        {
            if (globalSpeaker >= 0)
                chunkHasGlobalSpeaker[chunkIdx, globalSpeaker] = true;
        }

        progress?.Invoke("reconstructing global speaker timeline");
        for (int ci = 0; ci < perChunkScores.Count; ci++)
        {
            var chunkScore = perChunkBinary[ci];
            int numFrames  = chunkScore.GetLength(0);
            int startFrame = (int)Math.Round(startTimes[ci] * FrameRate);

            for (int t = 0; t < numFrames; t++)
            {
                int gf = startFrame + t;
                if (gf >= totalFrames) break;

                int activeLocalSpeakers = UseHardPowersetCount ? perChunkCounts[ci][t] : 0;
                Array.Clear(perChunkGlobal, 0, perChunkGlobal.Length);
                for (int spk = 0; spk < MaxUniqueSpeakers; spk++)
                {
                    float localScore = chunkScore[t, spk];
                    if (!UseHardPowersetCount && localScore > 0f)
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

            if (ShouldReportProgress(ci, perChunkScores.Count))
                progress?.Invoke($"reconstructed chunk {ci + 1}/{perChunkScores.Count}");
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
            int count = CountRoundingMode switch
            {
                "floor" => (int)Math.Floor(averageCount),
                "ceil" => (int)Math.Ceiling(averageCount),
                _ => (int)Math.Round(averageCount, MidpointRounding.AwayFromZero),
            };
            count = Math.Clamp(
                count,
                0,
                Math.Min(maxSpeakers, numGlobalSpeakers));

            for (int gs = 0; gs < numGlobalSpeakers; gs++)
            {
                int flatIndex = f * numGlobalSpeakers + gs;
                float score = activationContrib[flatIndex] > 0
                    ? (UseSummedActivations
                        ? activationSums[flatIndex]
                        : activationSums[flatIndex] / activationContrib[flatIndex])
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

        return DisableAdjacentMerge ? ToDiarizationSegments(labeled) : MergeAdjacentSegments(labeled);
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
        List<float[,]> perChunkBinary,
        List<int[]> perChunkCounts)
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
            int[] chunkCounts = perChunkCounts[ci];
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
                    decodedCount = chunkCounts[t],
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

    private List<float[]> SegmentBatched(List<float[]> chunks, Action<string>? progress = null)
    {
        var results = new List<float[]>(chunks.Count);
        for (int ci = 0; ci < chunks.Count; ci++)
        {
            var chunk = chunks[ci];
            results.Add(SegmentSingle(chunk));
            if (ShouldReportProgress(ci, chunks.Count))
                progress?.Invoke($"segmented chunk {ci + 1}/{chunks.Count}");
        }
        return results;
    }

    private static bool ShouldReportProgress(int index, int total)
    {
        if (total <= 0) return false;
        if (index == total - 1) return true;
        if (total <= 10) return true;

        int interval = Math.Max(1, total / 10);
        return index == 0 || ((index + 1) % interval) == 0;
    }

    private readonly record struct EmbeddingJob(
        int ChunkIdx,
        int SpeakerIdx,
        float[] ChunkWaveform,
        bool[] SelectedMask,
        int NumFrames,
        int CleanFrameCount);
    private readonly record struct EmbeddingJobResult(
        int ChunkIdx,
        int SpeakerIdx,
        int NumFrames,
        int CleanFrameCount,
        float[] L2Embedding,
        float[]? Xvec);

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

    private static float[,] ProjectSoftScoresToFrameCount(float[,] scores, int[] frameCounts, float threshold)
    {
        int numFrames = scores.GetLength(0);
        int numSpeakers = scores.GetLength(1);
        var binary = new float[numFrames, numSpeakers];
        var ranked = new (float score, int speaker)[numSpeakers];

        for (int t = 0; t < numFrames; t++)
        {
            int keep = Math.Clamp(frameCounts[t], 0, numSpeakers);
            for (int spk = 0; spk < numSpeakers; spk++)
                ranked[spk] = (scores[t, spk], spk);

            Array.Sort(ranked, (a, b) => b.score.CompareTo(a.score));
            for (int i = 0; i < keep; i++)
            {
                if (ranked[i].score < threshold)
                    break;
                binary[t, ranked[i].speaker] = 1f;
            }
        }

        return binary;
    }

    private static float[,] DecodeHardPowersetMasks(float[] powersetScores, float threshold)
    {
        var activeSpeakers = PowersetDecoder.BinarizePowerset(
            powersetScores,
            MaxUniqueSpeakers,
            MaxSimultaneousPerFrame,
            threshold);
        int numFrames = activeSpeakers.Length;
        var binary = new float[numFrames, MaxUniqueSpeakers];

        for (int t = 0; t < numFrames; t++)
        {
            foreach (int spk in activeSpeakers[t])
                binary[t, spk] = 1f;
        }

        return binary;
    }

    private static int[] DecodePowersetCardinality(float[] powersetScores, float threshold)
    {
        var activeSpeakers = PowersetDecoder.BinarizePowerset(
            powersetScores,
            MaxUniqueSpeakers,
            MaxSimultaneousPerFrame,
            threshold);
        var counts = new int[activeSpeakers.Length];
        for (int t = 0; t < activeSpeakers.Length; t++)
            counts[t] = activeSpeakers[t].Count;
        return counts;
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

    private static List<DiarizationSegment> ToDiarizationSegments(
        List<(int startFrame, int endFrame, string speakerLabel)> labeled)
    {
        var sorted = labeled.OrderBy(r => r.startFrame).ToList();
        var segments = new List<DiarizationSegment>(sorted.Count);
        foreach (var (startFrame, endFrame, speakerLabel) in sorted)
        {
            segments.Add(new DiarizationSegment(
                startFrame / (double)FrameRate,
                endFrame / (double)FrameRate,
                speakerLabel));
        }

        return segments;
    }

    private static List<(int startFrame, int endFrame, string speakerLabel)> PruneTinySpeakers(
        List<(int startFrame, int endFrame, string speakerLabel)> labeled,
        double minDurationSeconds)
    {
        if (labeled.Count == 0 || minDurationSeconds <= 0)
            return labeled;

        int minFrames = Math.Max(1, (int)Math.Round(minDurationSeconds * FrameRate));
        var durationBySpeaker = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (startFrame, endFrame, speakerLabel) in labeled)
            durationBySpeaker[speakerLabel] = durationBySpeaker.GetValueOrDefault(speakerLabel) + (endFrame - startFrame);

        var tinySpeakers = new HashSet<string>(
            durationBySpeaker.Where(kv => kv.Value < minFrames).Select(kv => kv.Key),
            StringComparer.Ordinal);

        if (tinySpeakers.Count == 0)
            return labeled;

        return labeled.Where(seg => !tinySpeakers.Contains(seg.speakerLabel)).ToList();
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

internal sealed record ReconstructionDebugFrame(
    int Frame,
    double TimeSeconds,
    float AverageCount,
    int SelectedCount,
    float[] Scores,
    bool[] Selected);
