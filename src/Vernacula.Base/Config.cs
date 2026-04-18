using Vernacula.Base.Models;

namespace Vernacula.Base;

/// <summary>
/// Central configuration — mirrors config.py exactly.
/// </summary>
public static class Config
{
    // ── Audio ────────────────────────────────────────────────────────────────
    public const int SampleRate = 16_000;

    // ── Diarization (Sortformer) ─────────────────────────────────────────────
    public const string SortformerSubDir = "sortformer";
    public const string SortformerFile = "diar_streaming_sortformer_4spk-v2.1.onnx";
    public const string SortformerDataFile = "diar_streaming_sortformer_4spk-v2.1.onnx.data";
    public const string SortformerModelOverrideEnvVar = "VERNACULA_SORTFORMER_MODEL_FILE";
    /// <summary>
    /// Expected MD5 hash of the Sortformer model file as published in the
    /// remote manifest.  Used by the model-version guard to detect outdated
    /// or corrupted model files on disk.
    /// </summary>
    public const string SortformerModelExpectedMd5 = "647a22cef31f59dc2c314fa783b2581d";

    public const int    NFft            = 512;
    public const int    WinLength       = 400;
    public const int    HopLength       = 160;
    public const int    NMels           = 128;
    public const float  Preemph         = 0.97f;
    public const float  LogZeroGuard    = 5.960464478e-8f;

    public const int    ChunkLength              = 124;  // frames per chunk
    public const int    FifoLength               = 124;
    public const int    Subsampling              = 8;
    public const int    EmbeddingDimension       = 512;
    public const int    NumSpeakers              = 4;
    public const int    SpeakerCacheLength       = 188;
    public const int    SpeakerCacheUpdatePeriod = 124;
    public const double FrameDuration            = 0.08; // 80 ms per frame

    public const float  SilThreshold    = 0.2f;
    public const int    Window          = 11;
    public const float  OnsetThreshold  = 0.641f;
    public const float  OffsetThreshold = 0.561f;
    public const double PadOnset        = 0.229;
    public const double PadOffset       = 0.079;
    public const double MinDurOn        = 0.511;
    public const double MinDurOff       = 0.296;

    // ── Diarization (DiariZen) ───────────────────────────────────────────────
    /// <summary>
    /// DiariZen segmentation model file.
    /// </summary>
    public const string DiariZenFile          = "diarizen_segmentation.onnx";
    /// <summary>
    /// pyannote/wespeaker-voxceleb-resnet34-LM: 80-bin Fbank input, 256-dim raw embedding,
    /// with frame-weight support matching pyannote's native weighted pooling semantics.
    /// Export via: python scripts/diarizen_export/export_pyannote_wespeaker_onnx.py --weighted
    /// </summary>
    public const string DiariZenEmbedderFile  = "wespeaker_pyannote_weighted.onnx";

    /// <summary>
    /// Directory (relative to model dir) containing mean1.bin, lda.bin, mean2.bin
    /// for the 256→128 LDA transform used by VBx clustering.
    /// Export via: python scripts/diarizen_export/export_lda_transform.py
    /// </summary>
    public const string DiariZenLdaDir       = "plda";

    /// <summary>
    /// Maximum number of unique speakers per chunk (default DiariZen configuration).
    /// </summary>
    public const int DiariZenMaxSpeakersPerChunk = 4;

    /// <summary>
    /// Maximum number of simultaneous speakers per frame (default DiariZen configuration).
    /// </summary>
    public const int DiariZenMaxSpeakersPerFrame = 2;

    /// <summary>
    /// Total number of powerset classes: C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11.
    /// </summary>
    public const int DiariZenNumPowersetClasses = 11;

    /// <summary>
    /// DiariZen chunk duration in seconds.
    /// </summary>
    public const int DiariZenChunkDurationSeconds = 16;

    /// <summary>
    /// Fraction of chunk duration to advance each step (0.1 = 10%, matching Python segmentation_step).
    /// stride = DiariZenChunkDurationSeconds * DiariZenSegmentationStep = 1.6 s
    /// </summary>
    public const double DiariZenSegmentationStep = 0.1;

    /// <summary>
    /// DiariZen frame rate: 50 frames per second (20ms frames).
    /// </summary>
    public const int DiariZenFrameRate = 50;

    /// <summary>
    /// Default threshold for binarizing powerset probabilities (0.5).
    /// </summary>
    public const float DiariZenDefaultThreshold = 0.5f;

    /// <summary>
    /// Default upper bound for adaptive ONNX Runtime intra-op threading used by
    /// DiariZen segmentation inference on CPU.
    /// </summary>
    public const int DiariZenSegmentationMaxIntraOpThreads = 12;

    /// <summary>
    /// Median filter half-width (frames) applied to powerset probabilities before binarisation.
    /// 11 frames ≈ 220 ms at 50 Hz, matching DiariZen's median_filter(size=(1, 11, 1)).
    /// </summary>
    public const int DiariZenMedianFilterSize = 11;

    /// <summary>
    /// Hierarchical clustering parameters for DiariZen.
    /// </summary>
    public const string DiariZenClusteringMethod = "centroid";
    public const int DiariZenMinClusterSize = 13; // ~0.26s at 50Hz

    /// <summary>
    /// AHC distance threshold used to initialise VBx.
    /// In PLDA-projected 128-dim space; matches Python VBxClustering ahc_threshold=0.6.
    /// </summary>
    public const float DiariZenAhcThreshold = 0.6f;

    /// <summary>
    /// VBx clustering parameters (alternative to HAC).
    /// </summary>
    public const float DiariZenVbxFa = 0.07f; // scales sufficient statistics
    public const float DiariZenVbxFb = 0.8f;  // speaker regularization
    public const int DiariZenVbxLdaDim = 128; // PLDA dimensionality
    public const int DiariZenVbxMaxIters = 20;
    public const int DiariZenMinEmbeddingClusterSize = 8;
    public const double DiariZenMinSpeakerDurationSeconds = 2.0;
    public const int DiariZenFillShortGapFrames = 2;
    public const int DiariZenMinRegionFrames = 2;
    public const int DiariZenMergeGapFrames = 50;
    public const int DiariZenEmbeddingGpuSafetyMb = 1024;
    public const int DiariZenEmbeddingGpuMaxBatchSize = 32;
    public const int DiariZenEmbeddingGpuMaxBatchFrames = 32_000;
    public const int DiariZenEmbeddingGpuBytesPerFrameEstimate = 65_536;
    public const int DiariZenEmbeddingMaxIntraOpThreads = 8;

    // Slaney mel-scale parameters
    public const double FMin      = 0.0;
    public const double FSp       = 200.0 / 3.0;
    public const double MinLogHz  = 1000.0;
    public static readonly double MinLogMel = (1000.0 - 0.0) / (200.0 / 3.0);
    public static readonly double LogStep   = Math.Log(6.4) / 27.0;

    // ── Denoiser (DeepFilterNet3) ─────────────────────────────────────────────
    public const string Dfn3SubDir = "deepfilternet3";
    /// <summary>DeepFilterNet3 encoder model (ERB+spec features → skip connections + embedding).</summary>
    public const string Dfn3EncFile    = "enc.onnx";
    /// <summary>DeepFilterNet3 ERB decoder model (embedding+skips → ERB mask).</summary>
    public const string Dfn3ErbDecFile = "erb_dec.onnx";
    /// <summary>DeepFilterNet3 DF decoder model (embedding+c0 → deep filter coefficients).</summary>
    public const string Dfn3DfDecFile  = "df_dec.onnx";

    // ── VAD (Silero) ──────────────────────────────────────────────────────────
    public const string VadSubDir         = "silero";
    public const string VadFile            = "silero_vad.onnx";
    public const int    VadWindowSamples   = 512;
    public const int    VadContextSamples  = 64;
    public const float  VadOnsetThreshold  = 0.5f;
    public const float  VadOffsetThreshold = 0.35f;
    public const int    VadMinSpeechMs     = 250;
    public const int    VadMinSilenceMs    = 100;
    public const int    VadSpeechPadMs     = 30;

    // ── Model manifest ───────────────────────────────────────────────────────
    public const string ManifestUrl =
        "https://huggingface.co/christopherthompson81/sortformer_parakeet_onnx/resolve/main/manifest.json";

    // ── ASR (VibeVoice) ──────────────────────────────────────────────────────

    /// <summary>
    /// Default subdirectory name (under the main model dir) where VibeVoice-ASR
    /// models are expected: audio_encoder.onnx, decoder_single.onnx, export-report.json, tokenizer.json.
    /// </summary>
    public const string VibeVoiceSubDir = "vibevoice_asr";

    // ── ASR (Qwen3-ASR) ──────────────────────────────────────────────────────
    public const string Qwen3AsrSubDir = "qwen3asr";

    // ── Language identification (VoxLingua107 ECAPA-TDNN) ────────────────────
    /// <summary>
    /// Directory (relative to models dir) for the language-ID model.
    /// </summary>
    public const string VoxLinguaSubDir   = "voxlingua107";
    /// <summary>
    /// Single-file FP32 ONNX with Conv1D-based STFT and MVN folded in.
    /// Input: <c>audio [batch, samples]</c> float32 at 16 kHz, mono.
    /// Outputs: <c>logits [batch, 107]</c> and <c>embedding [batch, 256]</c>.
    /// </summary>
    public const string VoxLinguaModelFile  = "voxlingua107.onnx";
    /// <summary>
    /// Companion JSON mapping class index (0..106) → {iso, name}.
    /// </summary>
    public const string VoxLinguaLangMapFile = "lang_map.json";
    /// <summary>
    /// Default clip duration fed to the model. Phase 6 of the perf
    /// investigation landed on 15 s as the accuracy plateau — longer
    /// clips don't meaningfully help, and doubling to 30 s doubles
    /// compute for ≤ 0.5 pp accuracy gain.
    /// </summary>
    public const int    VoxLinguaDefaultClipSeconds = 15;
    /// <summary>
    /// If the top-1 softmax probability falls below this threshold on a
    /// 15 s clip, the result is flagged ambiguous and the caller may
    /// escalate to a longer sample (if one is available) before
    /// committing to a language. Phase 6 showed confusable languages
    /// (Slavic family, Germanic family) sit around 0.55, confident
    /// picks at ≥ 0.8.
    /// </summary>
    public const float  VoxLinguaAmbiguityThreshold = 0.60f;
    /// <summary>
    /// Clip duration used for the escalation re-run when a 15 s
    /// classification is ambiguous. Capped by the actual available
    /// longest VAD segment.
    /// </summary>
    public const int    VoxLinguaEscalationClipSeconds = 60;

    // ── ASR (Parakeet) ───────────────────────────────────────────────────────
    public const string ParakeetSubDir       = "parakeet";
    public const string PreprocessorFile     = "nemo128.onnx";
    public const string EncoderFile          = "encoder-model.onnx";
    public const string DecoderJointFile     = "decoder_joint-model.onnx";
    public const string EncoderFileInt8      = "encoder-model.int8.onnx";
    public const string DecoderJointFileInt8 = "decoder_joint-model.int8.onnx";
    public const string VocabFile            = "vocab.txt";
    public const string AsrConfigFile        = "config.json";
    public const int    MaxTokensPerStep     = 10;
    public const int    MaxBatchSize         = 32;

    // VRAM formula for encoder batch sizing (RTX 3090 empirical fit, March 2026)
    public const double VramSlopePerSample = 0.001682;
    public const double VramInterceptMb    = -953.0;
    public const double VramWeightsMb      = 3500.0;
    public const double VramSafetyMb       = 2000.0;
    public const long   FallbackMaxFrames  = 1_600_000;

    public static (string encoder, string decoderJoint) GetAsrFiles(ModelPrecision precision) =>
        precision == ModelPrecision.Int8
            ? (EncoderFileInt8, DecoderJointFileInt8)
            : (EncoderFile,     DecoderJointFile);

    public static string GetSortformerModelPath(string modelDir)
    {
        string? overridePath = Environment.GetEnvironmentVariable(SortformerModelOverrideEnvVar);
        if (!string.IsNullOrWhiteSpace(overridePath))
        {
            return Path.IsPathRooted(overridePath)
                ? overridePath
                : Path.Combine(modelDir, overridePath);
        }

        string subDirPath = Path.Combine(modelDir, SortformerSubDir, SortformerFile);
        return File.Exists(subDirPath)
            ? subDirPath
            : Path.Combine(modelDir, SortformerFile);
    }

    public static int GetDiariZenSegmentationIntraOpThreads()
    {
        const string envVar = "VERNACULA_DIARIZEN_SEG_THREADS";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return GetAdaptiveDiariZenSegmentationIntraOpThreads();
    }

    public static int? GetDiariZenSegmentationMaxWorkers()
    {
        const string envVar = "VERNACULA_DIARIZEN_SEG_MAX_WORKERS";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return null;
    }

    public static int GetDiariZenSegmentationBatchSize()
    {
        const string envVar = "VERNACULA_DIARIZEN_SEG_BATCH_SIZE";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return 1;
    }

    public static int GetDiariZenEmbeddingGpuSafetyMb()
    {
        const string envVar = "VERNACULA_DIARIZEN_EMBED_GPU_SAFETY_MB";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed >= 0)
            return parsed;

        return DiariZenEmbeddingGpuSafetyMb;
    }

    public static int GetDiariZenEmbeddingIntraOpThreads()
    {
        const string envVar = "VERNACULA_DIARIZEN_EMBED_THREADS";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return GetAdaptiveDiariZenEmbeddingIntraOpThreads();
    }

    public static int? GetDiariZenEmbeddingMaxWorkers()
    {
        const string envVar = "VERNACULA_DIARIZEN_EMBED_MAX_WORKERS";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        int logicalCpuCount = Math.Max(1, Environment.ProcessorCount);
        int embedThreads = Math.Max(1, GetDiariZenEmbeddingIntraOpThreads());
        int usableCores = Math.Max(1, logicalCpuCount - 2);
        if (embedThreads >= 8)
            return logicalCpuCount >= 16 ? 2 : 1;

        if (embedThreads >= 4)
            return Math.Max(1, Math.Min(2, usableCores / embedThreads));

        return Math.Max(1, Math.Min(3, usableCores / embedThreads));
    }

    public static int GetDiariZenEmbeddingGpuMaxBatchSize()
    {
        const string envVar = "VERNACULA_DIARIZEN_EMBED_GPU_MAX_BATCH_SIZE";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return DiariZenEmbeddingGpuMaxBatchSize;
    }

    public static int GetDiariZenEmbeddingGpuMaxBatchFrames()
    {
        const string envVar = "VERNACULA_DIARIZEN_EMBED_GPU_MAX_BATCH_FRAMES";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return DiariZenEmbeddingGpuMaxBatchFrames;
    }

    public static int GetDiariZenEmbeddingGpuBytesPerFrameEstimate()
    {
        const string envVar = "VERNACULA_DIARIZEN_EMBED_GPU_BYTES_PER_FRAME";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed > 0)
            return parsed;

        return DiariZenEmbeddingGpuBytesPerFrameEstimate;
    }

    public static int GetDiariZenFillShortGapFrames()
    {
        const string envVar = "VERNACULA_DIARIZEN_FILL_SHORT_GAP_FRAMES";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed >= 0)
            return parsed;

        return DiariZenFillShortGapFrames;
    }

    public static int GetDiariZenMergeGapFrames()
    {
        const string envVar = "VERNACULA_DIARIZEN_MERGE_GAP_FRAMES";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed >= 0)
            return parsed;

        return DiariZenMergeGapFrames;
    }

    private static int GetAdaptiveDiariZenSegmentationIntraOpThreads()
    {
        int logicalCpuCount = Math.Max(1, Environment.ProcessorCount);
        long totalMemoryMb = HardwareInfo.GetTotalSystemMemoryMb();

        if (logicalCpuCount <= 4)
            return Math.Clamp(2, 1, logicalCpuCount);

        if (logicalCpuCount <= 8)
            return Math.Clamp(logicalCpuCount - 1, 2, logicalCpuCount);

        int preferredThreads = Math.Max(2, (int)Math.Floor(logicalCpuCount * 0.75));

        // Keep memory-constrained systems from leaning too hard on CPU
        // parallelism, even when the host reports many logical cores.
        if (totalMemoryMb > 0 && totalMemoryMb <= 4 * 1024)
            return Math.Min(4, preferredThreads);

        if (totalMemoryMb > 0 && totalMemoryMb <= 8 * 1024)
            return Math.Min(8, preferredThreads);

        return Math.Min(DiariZenSegmentationMaxIntraOpThreads, preferredThreads);
    }

    private static int GetAdaptiveDiariZenEmbeddingIntraOpThreads()
    {
        int logicalCpuCount = Math.Max(1, Environment.ProcessorCount);
        long totalMemoryMb = HardwareInfo.GetTotalSystemMemoryMb();

        if (logicalCpuCount <= 4)
            return 1;

        if (logicalCpuCount <= 8)
            return 2;

        if (logicalCpuCount <= 12)
            return totalMemoryMb > 0 && totalMemoryMb <= 8 * 1024 ? 2 : 4;

        if (totalMemoryMb > 0 && totalMemoryMb <= 8 * 1024)
            return 4;

        return Math.Min(DiariZenEmbeddingMaxIntraOpThreads, 8);
    }

    public static int GetDiariZenMinRegionFrames()
    {
        const string envVar = "VERNACULA_DIARIZEN_MIN_REGION_FRAMES";
        string? raw = Environment.GetEnvironmentVariable(envVar);
        if (int.TryParse(raw, out int parsed) && parsed >= 1)
            return parsed;

        return DiariZenMinRegionFrames;
    }
}
