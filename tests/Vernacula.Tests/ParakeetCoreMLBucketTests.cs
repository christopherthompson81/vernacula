using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// The CoreML encoder buckets have no <c>encoded_lengths</c> output, so
/// <see cref="Config.ParakeetEncodedFrames"/> is the ONLY thing deciding how much of the
/// encoder output gets decoded. If it drifts from the exporter's subsampler arithmetic
/// nothing throws: the decode is silently truncated or runs off into padding.
/// </summary>
public class ParakeetCoreMLBucketTests
{
    /// <summary>
    /// The frame counts the exporter recorded in `coreml-encoder-report.json` for the
    /// published buckets. These are the shapes the graphs actually declare — the session's
    /// output tensor is `[1, 1024, T]` for exactly these T — so a mismatch here is a
    /// mismatch with the artifact.
    /// </summary>
    [Theory]
    [InlineData(400, 50)]
    [InlineData(1000, 125)]
    [InlineData(2000, 250)]
    [InlineData(3000, 375)]
    public void EncodedFrames_MatchesThePublishedBuckets(int melFrames, int expected)
        => Assert.Equal(expected, Config.ParakeetEncodedFrames(melFrames));

    /// <summary>
    /// NeMo's `calc_length` applied three times, with kernel 3 / stride 2 / padding 1 each
    /// side: <c>out = (in + 2 - 3) / 2 + 1</c>. Spelled out independently here so the test
    /// fails if <see cref="Config.ParakeetEncodedFrames"/> is "simplified" into something
    /// that only agrees on multiples of 8.
    /// </summary>
    [Fact]
    public void EncodedFrames_AgreesWithNeMoArithmeticAtEveryLength()
    {
        for (int melFrames = 1; melFrames <= 3000; melFrames++)
        {
            int expected = melFrames;
            for (int i = 0; i < 3; i++)
                expected = (expected + 2 - 3) / 2 + 1;

            Assert.Equal(expected, Config.ParakeetEncodedFrames(melFrames));
        }
    }

    /// <summary>
    /// A true length that is not a multiple of the ×8 subsampling must still round UP, not
    /// truncate: the trailing partial window is a real encoder frame carrying real audio,
    /// and dropping it clips the end of the segment's transcript.
    /// </summary>
    [Theory]
    [InlineData(300, 38)]     // 300/8 = 37.5
    [InlineData(301, 38)]
    [InlineData(390, 49)]     // 390/8 = 48.75
    [InlineData(950, 119)]
    public void EncodedFrames_RoundsUpOnAPartialWindow(int melFrames, int expected)
        => Assert.Equal(expected, Config.ParakeetEncodedFrames(melFrames));

    /// <summary>
    /// The ladder has to be ascending for the "smallest bucket that fits" scan to pick the
    /// smallest, and every entry needs a real encoder frame count.
    /// </summary>
    [Fact]
    public void BucketLadder_IsAscendingAndUsable()
    {
        int[] frames = Config.ParakeetCoreMLEncoderFrames;
        Assert.NotEmpty(frames);
        for (int i = 1; i < frames.Length; i++)
            Assert.True(frames[i] > frames[i - 1],
                $"ladder must ascend: {frames[i - 1]} then {frames[i]}");
        foreach (int f in frames)
            Assert.True(Config.ParakeetEncodedFrames(f) > 0);
    }

    /// <summary>The filename the app looks for is the one the exporter writes.</summary>
    [Fact]
    public void BucketFileName_MatchesTheExporter()
        => Assert.Equal("encoder-model.coreml-1000.onnx", Config.ParakeetCoreMLEncoderFile(1000));
}
