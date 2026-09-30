using System.Buffers.Binary;
using System.Globalization;

namespace Vernacula.Base;

/// <summary>Which Sortformer-family checkpoint a <see cref="SortformerStreamer"/> runs.</summary>
public enum SortformerModel
{
    /// <summary>nvidia/diar_streaming_sortformer_4spk-v2.1 — the default.</summary>
    StreamingSortformerV21,

    /// <summary>nvidia/Nemotron-3-Diarization — 8 speakers, 10 ms output.</summary>
    Nemotron3,
}

/// <summary>
/// Every constant the Sortformer streaming loop and its post-processing depend on.
/// </summary>
/// <remarks>
/// Streaming Sortformer v2.1 and Nemotron-3-Diarization run the same streaming algorithm
/// (speaker cache + FIFO + score-based cache compression) with different numbers, plus two
/// differences in kind that Nemotron-3 introduces:
/// <list type="bullet">
/// <item>a LEARNED silence embedding fills the cache's disabled slots, instead of the running
/// mean of popped silence frames (<see cref="LearnableSilenceEmbedding"/>);</item>
/// <item>a 10 ms output head (<see cref="UpsampleFactor"/> = 8) beside the 80 ms predictions
/// that still drive the cache and FIFO.</item>
/// </list>
/// v2.1's values are <see cref="Config"/>'s constants. Nemotron-3's are read from the ONNX
/// metadata written by <c>scripts/nemo_export/export_nemotron3_diarization_to_onnx.py</c>, so the
/// artifact is the one source for its schedule and its learned silence embedding.
/// </remarks>
public sealed record SortformerProfile
{
    public required int NumSpeakers { get; init; }
    public required int EmbeddingDimension { get; init; }
    public required int NMels { get; init; }

    /// <summary>Mel frames per encoder (80 ms) frame.</summary>
    public required int Subsampling { get; init; }

    /// <summary>
    /// Reported frames per encoder frame: 1 when the graph reports at 80 ms (v2.1), 8 when it
    /// also emits <c>spkcache_fifo_chunk_preds_hr</c> at 10 ms (Nemotron-3).
    /// </summary>
    public int UpsampleFactor { get; init; } = 1;

    // ── Streaming schedule, in encoder frames ───────────────────────────────
    public required int ChunkLength { get; init; }
    public required int FifoLength { get; init; }
    public required int SpeakerCacheLength { get; init; }
    public required int SpeakerCacheUpdatePeriod { get; init; }

    /// <summary>Look-ahead frames attended to but not scored; they open the next chunk.</summary>
    public int ChunkRightContext { get; init; }

    // ── Cache compression (NeMo SortformerModules) ──────────────────────────
    public required int SpeakerCacheSilenceFrames { get; init; }
    public required float SilThreshold { get; init; }
    public required float ScoresBoostLatest { get; init; }
    public float PredScoreThreshold { get; init; } = 0.25f;
    public float StrongBoostRate { get; init; } = 0.75f;
    public float WeakBoostRate { get; init; } = 1.5f;
    public float MinPosScoresRate { get; init; } = 0.5f;

    /// <summary>
    /// The learned embedding for disabled cache slots, or null to use the running mean of
    /// popped silence frames (v2.1).
    /// </summary>
    public float[]? LearnableSilenceEmbedding { get; init; }

    // ── Post-processing (median filter + hysteresis binarisation) ───────────
    public required int Window { get; init; }
    public required float OnsetThreshold { get; init; }
    public required float OffsetThreshold { get; init; }
    public required double PadOnset { get; init; }
    public required double PadOffset { get; init; }
    public required double MinDurOn { get; init; }
    public required double MinDurOff { get; init; }

    /// <summary>Mel frames per graph call, excluding right context.</summary>
    public int ChunkStride => ChunkLength * Subsampling;

    /// <summary>Seconds per reported frame.</summary>
    public double FrameDuration => Config.HopLength / (double)Config.SampleRate * Subsampling / UpsampleFactor;

    /// <summary>True for the v2.1 graph, whose CoreML steady-state variant this profile's numbers describe.</summary>
    public bool IsLegacyV21 { get; init; }

    /// <summary>diar_streaming_sortformer_4spk-v2.1, exactly as Vernacula has always run it.</summary>
    public static SortformerProfile StreamingSortformerV21 { get; } = new()
    {
        IsLegacyV21              = true,
        NumSpeakers              = Config.NumSpeakers,
        EmbeddingDimension       = Config.EmbeddingDimension,
        NMels                    = Config.NMels,
        Subsampling              = Config.Subsampling,
        ChunkLength              = Config.ChunkLength,
        FifoLength               = Config.FifoLength,
        SpeakerCacheLength       = Config.SpeakerCacheLength,
        SpeakerCacheUpdatePeriod = Config.SpeakerCacheUpdatePeriod,
        SpeakerCacheSilenceFrames = Config.SpeakerCacheSilenceFrames,
        SilThreshold             = Config.SilThreshold,
        ScoresBoostLatest        = Config.ScoresBoostLatest,
        Window                   = Config.Window,
        OnsetThreshold           = Config.OnsetThreshold,
        OffsetThreshold          = Config.OffsetThreshold,
        PadOnset                 = Config.PadOnset,
        PadOffset                = Config.PadOffset,
        MinDurOn                 = Config.MinDurOn,
        MinDurOff                = Config.MinDurOff,
    };

    public const string MetadataPrefix = "vernacula.diar.";
    public const string HighResolutionContract = "sortformer-hr-1";

    /// <summary>
    /// The profile a graph declares in its metadata, or null when it declares none (a v2.1
    /// export, which predates the metadata and is described by <see cref="StreamingSortformerV21"/>).
    /// </summary>
    /// <exception cref="InvalidDataException">The graph declares a contract this build does not know,
    /// or a required key is missing or malformed.</exception>
    public static SortformerProfile? FromMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        if (!metadata.TryGetValue(MetadataPrefix + "contract", out string? contract))
            return null;
        if (contract != HighResolutionContract)
            throw new InvalidDataException(
                $"Diarization model declares contract '{contract}'; this build understands '{HighResolutionContract}'.");

        string Get(string key) => metadata.TryGetValue(MetadataPrefix + key, out string? v)
            ? v
            : throw new InvalidDataException($"Diarization model metadata is missing '{MetadataPrefix}{key}'.");
        int I(string key) => int.Parse(Get(key), NumberStyles.Integer, CultureInfo.InvariantCulture);
        float F(string key) => float.Parse(Get(key), NumberStyles.Float, CultureInfo.InvariantCulture);

        int emb = I("emb_dim");
        byte[] silBytes = Convert.FromBase64String(Get("learnable_sil_emb_f32le_b64"));
        if (silBytes.Length != emb * sizeof(float))
            throw new InvalidDataException(
                $"learnable_sil_emb has {silBytes.Length} bytes; expected {emb * sizeof(float)}.");
        var sil = new float[emb];
        for (int d = 0; d < emb; d++)
            sil[d] = BinaryPrimitives.ReadSingleLittleEndian(silBytes.AsSpan(d * 4, 4));

        return new SortformerProfile
        {
            NumSpeakers              = I("num_speakers"),
            EmbeddingDimension       = emb,
            NMels                    = I("n_mels"),
            Subsampling              = I("subsampling"),
            UpsampleFactor           = I("upsample_factor"),
            ChunkLength              = I("chunk_len"),
            ChunkRightContext        = I("chunk_right_context"),
            FifoLength               = I("fifo_len"),
            SpeakerCacheLength       = I("spkcache_len"),
            SpeakerCacheUpdatePeriod = I("spkcache_update_period"),
            SpeakerCacheSilenceFrames = I("spkcache_sil_frames_per_spk"),
            SilThreshold             = F("sil_threshold"),
            ScoresBoostLatest        = F("scores_boost_latest"),
            PredScoreThreshold       = F("pred_score_threshold"),
            StrongBoostRate          = F("strong_boost_rate"),
            WeakBoostRate            = F("weak_boost_rate"),
            MinPosScoresRate         = F("min_pos_scores_rate"),
            LearnableSilenceEmbedding = sil,
            // NeMo's and Transformers' defaults for this model -- a plain 0.5 threshold on the
            // 10 ms probabilities, no median filter, no padding -- with ONE departure: a
            // speaker's segments separated by less than 0.5 s are merged. The model's
            // boundaries follow pauses closely (it is trained on forced-aligned labels), and at
            // NeMo's MinDurOff = 0 five minutes of far-field meeting audio came out as 251
            // segments, each of which goes to ASR on its own. Merging cut that to 174 and did
            // not change speaker attribution: DER within 0.3 points on VoxConverse, lower on
            // AMI. See docs/investigations/nemotron3_diarization_onnx_investigation.md, Run 6.
            Window          = 1,
            OnsetThreshold  = 0.5f,
            OffsetThreshold = 0.5f,
            PadOnset        = 0.0,
            PadOffset       = 0.0,
            MinDurOn        = 0.0,
            MinDurOff       = 0.5,
        };
    }
}
