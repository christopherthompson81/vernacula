using Parakeet.Base.Models;

namespace Parakeet.Base;

/// <summary>
/// Central configuration — mirrors config.py exactly.
/// </summary>
public static class Config
{
    // ── Audio ────────────────────────────────────────────────────────────────
    public const int SampleRate = 16_000;

    // ── Diarization (Sortformer) ─────────────────────────────────────────────
    public const string SortformerFile = "diar_streaming_sortformer_4spk-v2.1.onnx";

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
    public const string DiariZenFile = "diarizen_segmentation.onnx";

    // Slaney mel-scale parameters
    public const double FMin      = 0.0;
    public const double FSp       = 200.0 / 3.0;
    public const double MinLogHz  = 1000.0;
    public static readonly double MinLogMel = (1000.0 - 0.0) / (200.0 / 3.0);
    public static readonly double LogStep   = Math.Log(6.4) / 27.0;

    // ── VAD (Silero) ──────────────────────────────────────────────────────────
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

    // ── ASR (Parakeet) ───────────────────────────────────────────────────────
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
}
