using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Nemotron-3-Diarization's streaming schedule and learned silence embedding live in its ONNX
/// metadata (written by scripts/nemo_export/export_nemotron3_diarization_to_onnx.py), so the
/// parser is the only thing standing between a malformed or foreign graph and a streaming run
/// with the wrong constants -- which would not crash, only produce plausible-looking garbage.
/// </summary>
public class SortformerProfileTests
{
    private const string P = SortformerProfile.MetadataPrefix;

    private static Dictionary<string, string> Metadata(int embDim = 4, float[]? sil = null)
    {
        sil ??= Enumerable.Range(0, embDim).Select(i => 0.25f * i - 0.5f).ToArray();
        var bytes = new byte[sil.Length * 4];
        for (int i = 0; i < sil.Length; i++)
            BitConverter.TryWriteBytes(bytes.AsSpan(i * 4, 4), sil[i]);   // little-endian on every supported host

        return new Dictionary<string, string>
        {
            [P + "contract"]                    = SortformerProfile.HighResolutionContract,
            [P + "num_speakers"]                = "8",
            [P + "emb_dim"]                     = embDim.ToString(),
            [P + "n_mels"]                      = "128",
            [P + "subsampling"]                 = "8",
            [P + "upsample_factor"]             = "8",
            [P + "spkcache_len"]                = "264",
            [P + "fifo_len"]                    = "40",
            [P + "chunk_len"]                   = "340",
            [P + "chunk_right_context"]         = "40",
            [P + "spkcache_update_period"]      = "300",
            [P + "spkcache_sil_frames_per_spk"] = "1",
            [P + "sil_threshold"]               = "0.2",
            [P + "scores_boost_latest"]         = "0.05",
            [P + "pred_score_threshold"]        = "0.25",
            [P + "strong_boost_rate"]           = "0.75",
            [P + "weak_boost_rate"]             = "1.5",
            [P + "min_pos_scores_rate"]         = "0.5",
            [P + "learnable_sil_emb_f32le_b64"] = Convert.ToBase64String(bytes),
        };
    }

    [Fact]
    public void V21Graph_DeclaresNothing_AndGetsNull()
    {
        // The v2.1 export predates the metadata; SortformerStreamer falls back to v2.1's
        // constants for it. Anything else carrying a vernacula.diar.* key is not v2.1.
        Assert.Null(SortformerProfile.FromMetadata(new Dictionary<string, string>()));
        Assert.Null(SortformerProfile.FromMetadata(new Dictionary<string, string> { ["producer"] = "pytorch" }));
    }

    [Fact]
    public void Nemotron3Metadata_RoundTrips()
    {
        var sil = new[] { 0.125f, -3.5f, 1e-7f, float.Epsilon };
        var p = SortformerProfile.FromMetadata(Metadata(sil: sil))!;

        Assert.Equal(8, p.NumSpeakers);
        Assert.Equal(340, p.ChunkLength);
        Assert.Equal(40, p.ChunkRightContext);
        Assert.Equal(40, p.FifoLength);
        Assert.Equal(264, p.SpeakerCacheLength);
        Assert.Equal(300, p.SpeakerCacheUpdatePeriod);
        Assert.Equal(1, p.SpeakerCacheSilenceFrames);
        Assert.Equal(340 * 8, p.ChunkStride);
        Assert.Equal(0.01, p.FrameDuration, 12);
        Assert.False(p.IsLegacyV21);
        // Bit-exact, including a denormal: the embedding goes straight into the speaker cache.
        Assert.Equal(sil, p.LearnableSilenceEmbedding);
    }

    [Fact]
    public void V21Profile_KeepsItsHistoricalConstants()
    {
        var p = SortformerProfile.StreamingSortformerV21;
        Assert.True(p.IsLegacyV21);
        Assert.Equal(4, p.NumSpeakers);
        Assert.Equal(0.08, p.FrameDuration, 12);
        Assert.Null(p.LearnableSilenceEmbedding);
        Assert.Equal(0, p.ChunkRightContext);
        Assert.Equal(1, p.UpsampleFactor);
    }

    [Fact]
    public void UnknownContract_IsRejected()
    {
        var md = Metadata();
        md[P + "contract"] = "sortformer-hr-2";
        var ex = Assert.Throws<InvalidDataException>(() => SortformerProfile.FromMetadata(md));
        Assert.Contains("sortformer-hr-2", ex.Message);
    }

    [Fact]
    public void MissingKey_IsRejected_ByName()
    {
        var md = Metadata();
        md.Remove(P + "fifo_len");
        var ex = Assert.Throws<InvalidDataException>(() => SortformerProfile.FromMetadata(md));
        Assert.Contains("fifo_len", ex.Message);
    }

    [Theory]
    [InlineData("fifo_len", "40.0")]
    [InlineData("chunk_len", "")]
    [InlineData("sil_threshold", "0,2")]
    [InlineData("learnable_sil_emb_f32le_b64", "not base64!")]
    public void MalformedValue_IsRejected_AsInvalidData_ByName(string key, string value)
    {
        // Same exception type as a missing key, so one catch covers "re-export the model".
        var md = Metadata();
        md[P + key] = value;
        var ex = Assert.Throws<InvalidDataException>(() => SortformerProfile.FromMetadata(md));
        Assert.Contains(key, ex.Message);
    }

    [Theory]
    [InlineData("80")]
    [InlineData("160")]
    public void MelBinCountOtherThanTheFrontends_IsRejected(string nMels)
    {
        var md = Metadata();
        md[P + "n_mels"] = nMels;
        Assert.Throws<InvalidDataException>(() => SortformerProfile.FromMetadata(md));
    }

    [Fact]
    public void SilenceEmbeddingOfTheWrongSize_IsRejected()
    {
        var md = Metadata(embDim: 4, sil: new float[3]);
        Assert.Throws<InvalidDataException>(() => SortformerProfile.FromMetadata(md));
    }
}
