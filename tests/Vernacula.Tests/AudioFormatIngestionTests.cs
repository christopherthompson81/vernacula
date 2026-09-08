using System.Diagnostics;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Format coverage for <see cref="AudioUtils.ReadAudio"/> (issue #156).
///
/// The regression these lock down: NAudio 3's cross-platform build decodes only
/// PCM/IEEE-float WAV, so everything else — MP3, FLAC, M4A, non-PCM WAV — threw
/// NotSupportedException out of the CLI surface on every platform. Reading those
/// now goes through FFmpeg.
///
/// Fixtures are synthesised with ffmpeg rather than committed, so there is no
/// binary test data in the repo and every case is generated from the same
/// known-good source tone. Skips when ffmpeg is absent, as on a hosted runner.
/// </summary>
public class AudioFormatIngestionTests : IDisposable
{
    private const int SourceRate = 44100;
    private const double DurationSec = 0.5;

    private readonly string _dir = Directory.CreateTempSubdirectory("vernacula_audio_fmt_").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
        GC.SuppressFinalize(this);
    }

    private static void RequireFfmpeg()
    {
        if (!FfmpegAudioDecoder.IsAvailable)
            Assert.Skip("ffmpeg/ffprobe not on PATH.");
    }

    /// <summary>Synthesise a 440 Hz tone in the requested container/codec.</summary>
    private string MakeFixture(string fileName, params string[] extraArgs)
    {
        string path = Path.Combine(_dir, fileName);
        var psi = new ProcessStartInfo("ffmpeg")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        // ⚠ volume=6 IS LOAD-BEARING. lavfi's sine source emits at -18 dBFS (peak 0.125),
        // and a resample can land the stereo fixture near 0.088 — close enough to the
        // "is this silent?" assertion below to make it flap. Lifting the source to ~0.75
        // keeps that check meaningful instead of tuning the threshold down to meet it.
        foreach (string a in new[] { "-v", "error", "-y", "-f", "lavfi", "-i",
                                     $"sine=frequency=440:sample_rate={SourceRate}:duration={DurationSec.ToString(System.Globalization.CultureInfo.InvariantCulture)}",
                                     "-af", "volume=6" })
            psi.ArgumentList.Add(a);
        foreach (string a in extraArgs) psi.ArgumentList.Add(a);
        psi.ArgumentList.Add(path);

        using var proc = Process.Start(psi)!;
        proc.StandardOutput.ReadToEnd();
        string err = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        if (proc.ExitCode != 0)
        {
            // Encoder sets differ between ffmpeg builds (libmp3lame, libvorbis and libopus
            // are all optional). That's a gap in the local ffmpeg, not in ReadAudio, so it
            // skips rather than reporting a failure against this code.
            if (err.Contains("Unknown encoder", StringComparison.OrdinalIgnoreCase)
                || err.Contains("Encoder not found", StringComparison.OrdinalIgnoreCase))
                Assert.Skip($"local ffmpeg cannot encode {fileName}: {err.Trim()}");

            Assert.Fail($"ffmpeg could not build fixture {fileName}: {err}");
        }
        return path;
    }

    /// <summary>A tone must come back as a sane number of non-silent samples.</summary>
    private static void AssertDecodedTone((float[] samples, int sampleRate, int channels) got,
                                          int expectedChannels = 1)
    {
        Assert.Equal(expectedChannels, got.channels);
        Assert.True(got.sampleRate > 0, "sample rate should be positive");

        int frames = got.samples.Length / got.channels;
        double seconds = (double)frames / got.sampleRate;
        // Lossy encoders pad, so this is a sanity band rather than an equality.
        Assert.InRange(seconds, DurationSec * 0.8, DurationSec * 1.5);

        float peak = 0f;
        foreach (float s in got.samples) peak = Math.Max(peak, Math.Abs(s));
        Assert.True(peak > 0.1f, $"decoded audio is silent or near-silent (peak {peak})");
        Assert.True(peak <= 1.01f, $"samples should be normalised to [-1, 1] (peak {peak})");
    }

    // ── The in-process path: PCM WAV never needed ffmpeg and still must not use it ──

    /// <summary>
    /// Run <paramref name="read"/> and assert it never reached ffmpeg.
    /// <para>
    /// ⚠ THE DELTA IS THE POINT. Asserting only that the tone decoded would pass
    /// identically if ReadAudio started routing every WAV through the subprocess —
    /// which is the regression these two tests exist to catch. Watching the decode
    /// counter is what makes them load-bearing. Safe against xunit's parallelism:
    /// this is the only class in the assembly that decodes audio, and xunit runs the
    /// tests within one class sequentially.
    /// </para>
    /// </summary>
    private static void AssertReadWithoutFfmpeg(Func<(float[], int, int)> read, int expectedChannels = 1)
    {
        int before = FfmpegAudioDecoder.DecodeInvocations;
        var got = read();
        int after = FfmpegAudioDecoder.DecodeInvocations;

        Assert.True(before == after,
            $"expected the in-process NAudio path, but ffmpeg was invoked {after - before} time(s)");
        AssertDecodedTone(got, expectedChannels);
    }

    [Fact]
    public void PcmWav_ReadsWithoutFfmpeg()
    {
        RequireFfmpeg();  // only to build the fixture
        string path = MakeFixture("tone_pcm16.wav", "-acodec", "pcm_s16le");

        AssertReadWithoutFfmpeg(() => AudioUtils.ReadAudio(path));
    }

    [Fact]
    public void FloatWav_ReadsWithoutFfmpeg()
    {
        RequireFfmpeg();
        string path = MakeFixture("tone_f32.wav", "-acodec", "pcm_f32le");

        AssertReadWithoutFfmpeg(() => AudioUtils.ReadAudio(path));
    }

    /// <summary>Native rate and channel count survive; ReadAudio must not resample or downmix.</summary>
    [Fact]
    public void StereoWav_PreservesNativeRateAndChannels()
    {
        RequireFfmpeg();
        string path = MakeFixture("tone_stereo_48k.wav", "-acodec", "pcm_s16le", "-ac", "2", "-ar", "48000");

        var got = AudioUtils.ReadAudio(path);
        Assert.Equal(2, got.channels);
        Assert.Equal(48000, got.sampleRate);
        AssertDecodedTone(got, expectedChannels: 2);
    }

    // ── The formats #156 is about: these all threw NotSupportedException before ──

    [Theory]
    [InlineData("tone.mp3", new[] { "-acodec", "libmp3lame" })]
    [InlineData("tone.flac", new[] { "-acodec", "flac" })]
    [InlineData("tone.ogg", new[] { "-acodec", "libvorbis" })]
    [InlineData("tone.m4a", new[] { "-c:a", "aac" })]
    [InlineData("tone.opus", new[] { "-c:a", "libopus" })]
    public void CompressedFormats_DecodeThroughFfmpeg(string fileName, string[] codecArgs)
    {
        RequireFfmpeg();
        string path = MakeFixture(fileName, codecArgs);

        AssertDecodedTone(AudioUtils.ReadAudio(path));
    }

    /// <summary>
    /// The reported sample rate must be the DECODER's, not the source file's.
    /// <para>
    /// Opus always decodes at 48 kHz whatever went in, so a 44.1 kHz tone encoded to
    /// Opus and read back must report 48000. An earlier draft took the rate from a
    /// separate ffprobe call, which can disagree with what ffmpeg actually emits -
    /// implicit-SBR HE-AAC being the case that silently doubles it. This build has no
    /// HE-AAC encoder (no libfdk_aac), so that exact case is not covered here; this
    /// pins the general property that rate follows the decoder.
    /// </para>
    /// </summary>
    [Fact]
    public void Opus_ReportsTheDecoderSampleRateNotTheSourceRate()
    {
        RequireFfmpeg();
        Assert.NotEqual(48000, SourceRate);  // the test is meaningless if these match
        string path = MakeFixture("tone_rate.opus", "-c:a", "libopus");

        var got = AudioUtils.ReadAudio(path);
        Assert.Equal(48000, got.sampleRate);
        AssertDecodedTone(got);
    }

    /// <summary>
    /// A .wav that isn't PCM. The extension sends it down the NAudio path first, which
    /// throws, and it has to land on ffmpeg rather than propagating the exception.
    /// </summary>
    [Theory]
    [InlineData("tone_mulaw.wav", "pcm_mulaw")]
    [InlineData("tone_alaw.wav", "pcm_alaw")]
    [InlineData("tone_adpcm.wav", "adpcm_ima_wav")]
    public void NonPcmWav_FallsBackToFfmpeg(string fileName, string codec)
    {
        RequireFfmpeg();
        string path = MakeFixture(fileName, "-acodec", codec);

        int before = FfmpegAudioDecoder.DecodeInvocations;
        var got = AudioUtils.ReadAudio(path);
        Assert.True(FfmpegAudioDecoder.DecodeInvocations > before,
            "a non-PCM .wav should have fallen through to ffmpeg");
        AssertDecodedTone(got);
    }

    // ── Failure modes should say something useful ──

    [Fact]
    public void MissingFile_ThrowsFileNotFound()
    {
        Assert.Throws<FileNotFoundException>(
            () => AudioUtils.ReadAudio(Path.Combine(_dir, "does_not_exist.mp3")));
    }

    [Fact]
    public void FileWithNoAudioStream_ThrowsWithAnExplanation()
    {
        RequireFfmpeg();
        string path = Path.Combine(_dir, "not_audio.mp3");
        File.WriteAllText(path, "this is not an audio file");

        var ex = Assert.ThrowsAny<InvalidOperationException>(() => AudioUtils.ReadAudio(path));
        Assert.Contains("not_audio.mp3", ex.Message);
    }
}
