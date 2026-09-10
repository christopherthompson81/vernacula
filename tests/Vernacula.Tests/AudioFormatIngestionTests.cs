using System.Diagnostics;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// Format coverage for <see cref="AudioUtils.ReadAudio"/> (issues #156 and #176).
///
/// The regression these lock down: NAudio 3's cross-platform build decodes only
/// PCM/IEEE-float WAV, so everything else — MP3, FLAC, M4A, non-PCM WAV — threw
/// NotSupportedException out of the CLI surface on every platform. WAV, MP3, AIFF,
/// Ogg Vorbis and Ogg Opus are decoded in-process now; the rest goes through FFmpeg.
///
/// Fixtures for the FFmpeg formats are synthesised with ffmpeg rather than committed,
/// so every case is generated from the same known-good source tone. Those skip when
/// ffmpeg is absent, as on a hosted runner.
///
/// ⚠ THE IN-PROCESS FORMATS USE COMMITTED FIXTURES, AND THEY HAVE TO. What those cases
/// assert is "decodes with no ffmpeg installed", and the machine that is true on is
/// precisely the machine where a synthesised fixture cannot be built — a skipping test
/// there proves nothing, which is how #176 shipped. They read from tests/fixtures/
/// instead, and assert the decode never reached the subprocess. See
/// tests/fixtures/README.md.
///
/// ⚠ EVERY AUDIO DECODE IN THIS ASSEMBLY LIVES IN THIS ONE CLASS, DELIBERATELY. The
/// routing assertions watch process-wide decode counters, so a second test class
/// decoding audio in parallel would make them flap; xunit runs the tests within a
/// single class sequentially. Add audio cases here rather than in a new class.
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
            Assert.Skip("ffmpeg not found on PATH or in the managed directory.");
    }

    /// <summary>Synthesise a 440 Hz tone in the requested container/codec.</summary>
    private string MakeFixture(string fileName, params string[] extraArgs)
    {
        string path = Path.Combine(_dir, fileName);
        // ⚠ RESOLVE IT THE WAY THE CODE UNDER TEST DOES. RequireFfmpeg above asks
        // FfmpegAudioDecoder, which since #176's follow-up also finds a copy the desktop app
        // downloaded into the managed directory. A bare "ffmpeg" here would disagree with
        // that gate on exactly those machines: the skip would not fire and every fixture
        // build would throw Win32Exception instead.
        var psi = new ProcessStartInfo(FfmpegBinaries.ResolveExecutable("ffmpeg"))
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

    // MP3 is deliberately absent from this theory: it decodes in-process now, and the cases
    // below assert that it does. Leaving it here would keep passing either way, since this
    // theory only checks that the tone came back.
    [Theory]
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

    // ── MP3 (#176): in-process, on every platform, with no ffmpeg anywhere near it ──

    private const string MonoMp3   = "tone_mono_44100.mp3";
    private const string StereoMp3 = "tone_stereo_48000.mp3";
    private const string Lsf16kMp3 = "tone_mono_16000.mp3";
    private const string Mpeg25Mp3 = "tone_mono_8000.mp3";

    /// <summary>Committed fixture, copied next to the test assembly by the csproj.</summary>
    private static string Fixture(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "fixtures", name);
        Assert.True(File.Exists(path),
            $"missing committed fixture '{name}' at {path} — check the <None Include=\"..\\fixtures\\*.mp3\"> "
            + "item in Vernacula.Tests.csproj");
        return path;
    }

    /// <summary>
    /// Run <paramref name="read"/> and assert it took an in-process decoder, not ffmpeg.
    /// <para>
    /// ⚠ TWO COUNTERS, NOT ONE. "ffmpeg wasn't called" would also hold if ReadAudio threw
    /// before reaching it, and "the managed decoder ran" would hold if ReadAudio then fell
    /// through to ffmpeg anyway. #176 is the second shape: MP3 decoded fine, through the
    /// wrong route, on a machine that happened to have ffmpeg installed — which is every CI
    /// runner, so the weaker assertion would have stayed green through the whole regression.
    /// </para>
    /// </summary>
    private static (float[] samples, int sampleRate, int channels)
        ReadAssertingManaged(Func<(float[], int, int)> read, Func<int> managedCounter)
    {
        int ffmpegBefore  = FfmpegAudioDecoder.DecodeInvocations;
        int managedBefore = managedCounter();

        var got = read();

        Assert.True(FfmpegAudioDecoder.DecodeInvocations == ffmpegBefore,
            $"this format must not need ffmpeg, but it was invoked "
            + $"{FfmpegAudioDecoder.DecodeInvocations - ffmpegBefore} time(s)");
        Assert.True(managedCounter() == managedBefore + 1,
            "the file should have gone through its in-process decoder exactly once");
        return got;
    }

    private static (float[] samples, int sampleRate, int channels)
        ReadAssertingManagedMp3(Func<(float[], int, int)> read) =>
        ReadAssertingManaged(read, () => Mp3Decoder.DecodeInvocations);

    private static (float[] samples, int sampleRate, int channels)
        ReadAssertingManagedOgg(Func<(float[], int, int)> read) =>
        ReadAssertingManaged(read, () => OggDecoder.DecodeInvocations);

    /// <summary>
    /// Goertzel magnitude at <paramref name="hz"/>. Used to check the decoder produced the
    /// tone rather than any plausible-looking buffer of the right size.
    /// </summary>
    private static double EnergyAt(float[] mono, int sampleRate, double hz)
    {
        double coeff = 2.0 * Math.Cos(2.0 * Math.PI * hz / sampleRate);
        double s1 = 0.0, s2 = 0.0;
        foreach (float x in mono)
        {
            double s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        return s1 * s1 + s2 * s2 - coeff * s1 * s2;
    }

    /// <summary>The fixtures are 440 Hz sines; assert that is still what came out.</summary>
    private static void AssertIs440HzTone((float[] samples, int sampleRate, int channels) got)
    {
        float[] mono = AudioUtils.DownmixToMono(got.samples, got.channels);
        double at440 = EnergyAt(mono, got.sampleRate, 440);
        foreach (double other in new[] { 220.0, 660.0, 880.0, 1320.0, 2000.0 })
        {
            double elsewhere = EnergyAt(mono, got.sampleRate, other);
            Assert.True(at440 > 10 * elsewhere,
                $"decoded audio does not look like a 440 Hz tone: energy at {other} Hz "
                + $"({elsewhere:G3}) rivals 440 Hz ({at440:G3})");
        }
    }

    /// <summary>
    /// The #176 case itself: an MP3 must read on a machine with no ffmpeg installed, at its
    /// own rate and channel count, in full.
    ///
    /// <para>
    /// ⚠ ALL FOUR RATES, BECAUSE THEY ARE THREE DIFFERENT DECODERS. 44.1/48 kHz are MPEG-1;
    /// 16 kHz is MPEG-2 (LSF, 576 samples per frame rather than 1152); 8 kHz is MPEG-2.5.
    /// NLayer 1.16.0 returns roughly half of any file at or below 24 kHz while still
    /// reporting the full duration, so a suite testing only 44.1 kHz would pass against a
    /// decoder that silently drops half of every 16 kHz voice memo — the shape an ASR tool
    /// is most often handed. The length assertion in <see cref="AssertDecodedTone"/> is what
    /// catches that, so keep its lower bound well above half.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(MonoMp3, 44100, 1)]     // MPEG-1
    [InlineData(StereoMp3, 48000, 2)]   // MPEG-1, stereo
    [InlineData(Lsf16kMp3, 16000, 1)]   // MPEG-2 LSF
    [InlineData(Mpeg25Mp3, 8000, 1)]    // MPEG-2.5
    public void Mp3_ReadsWithoutFfmpegAtItsNativeLayout(string fixture, int rate, int channels)
    {
        var got = ReadAssertingManagedMp3(() => AudioUtils.ReadAudio(Fixture(fixture)));

        Assert.Equal(channels, got.channels);
        Assert.Equal(rate, got.sampleRate);
        AssertDecodedTone(got, expectedChannels: channels);
        AssertIs440HzTone(got);
    }

    /// <summary>
    /// Real-world MP3s carry tags, and a tag is bytes that are not MPEG frames sitting where
    /// the decoder looks for a frame sync. Wrapping a known-good fixture in both tag formats
    /// must change nothing about what comes out.
    /// <para>
    /// The ID3v2 payload here is 4 KB of 0xFF, which is the hostile case on purpose: 0xFF is
    /// the first byte of an MPEG frame sync, so a decoder that scanned for a sync instead of
    /// honouring the tag's declared length would start decoding garbage.
    /// </para>
    /// </summary>
    [Fact]
    public void TaggedMp3_DecodesToTheSameSamples()
    {
        var plain = ReadAssertingManagedMp3(() => AudioUtils.ReadAudio(Fixture(MonoMp3)));

        byte[] audio = File.ReadAllBytes(Fixture(MonoMp3));
        const int payload = 4096;

        // ID3v2.3 header: "ID3", version, flags, then a 4-byte syncsafe size (7 bits per byte)
        // covering everything after the 10-byte header.
        var tagged = new List<byte>(payload + audio.Length + 138);
        tagged.AddRange("ID3"u8.ToArray());
        tagged.AddRange([3, 0, 0]);
        tagged.AddRange([
            (byte)((payload >> 21) & 0x7F), (byte)((payload >> 14) & 0x7F),
            (byte)((payload >> 7)  & 0x7F), (byte)( payload        & 0x7F),
        ]);
        tagged.AddRange(Enumerable.Repeat((byte)0xFF, payload));
        tagged.AddRange(audio);
        // ID3v1 trailer: the last 128 bytes of the file, starting "TAG".
        tagged.AddRange("TAG"u8.ToArray());
        tagged.AddRange(Enumerable.Repeat((byte)0x20, 125));

        string path = Path.Combine(_dir, "tagged.mp3");
        File.WriteAllBytes(path, tagged.ToArray());

        var got = ReadAssertingManagedMp3(() => AudioUtils.ReadAudio(path));

        Assert.Equal(plain.sampleRate, got.sampleRate);
        Assert.Equal(plain.channels, got.channels);
        Assert.Equal(plain.samples.Length, got.samples.Length);
        Assert.Equal(plain.samples, got.samples);
    }

    /// <summary>
    /// <see cref="Mp3Decoder"/> is the public entry point behind that routing, and it reports
    /// the file's own layout rather than anything resampled.
    /// </summary>
    [Fact]
    public void Mp3Decoder_ReturnsTheFilesNativeLayout()
    {
        var got = Mp3Decoder.Decode(Fixture(StereoMp3));

        Assert.Equal(48000, got.sampleRate);
        Assert.Equal(2, got.channels);
        Assert.Equal(0, got.samples.Length % 2);
        AssertDecodedTone(got, expectedChannels: 2);
    }

    /// <summary>
    /// An .mp3 whose bytes are not MPEG audio — a renamed WAV — still has to read. The
    /// extension picks the decoder to try first; the content decides what actually works.
    /// </summary>
    [Fact]
    public void RenamedNonMp3_FallsBackToFfmpeg()
    {
        RequireFfmpeg();
        string wav = MakeFixture("actually_a_wav.wav", "-acodec", "pcm_s16le");
        string path = Path.Combine(_dir, "renamed.mp3");
        File.Copy(wav, path);

        int before = FfmpegAudioDecoder.DecodeInvocations;
        var got = AudioUtils.ReadAudio(path);

        Assert.True(FfmpegAudioDecoder.DecodeInvocations > before,
            "an .mp3 that is really a WAV should have fallen through to ffmpeg");
        AssertDecodedTone(got);
    }

    /// <summary>
    /// And when neither decoder can read it, the error names both attempts. Reporting only
    /// the ffmpeg failure would send a Windows user off installing ffmpeg for a file that is
    /// simply corrupt — the exact wrong conclusion, and the one #176 made easy to reach.
    /// </summary>
    [Fact]
    public void UnreadableMp3_ErrorNamesBothDecoders()
    {
        string path = Path.Combine(_dir, "corrupt.mp3");
        File.WriteAllText(path, "this is not an audio file");

        var ex = Assert.ThrowsAny<InvalidOperationException>(() => AudioUtils.ReadAudio(path));
        Assert.Contains("corrupt.mp3", ex.Message);
        Assert.Contains("MP3", ex.Message);
        Assert.Contains("FFmpeg", ex.Message);
    }

    /// <summary>A missing .mp3 is a missing file, not a format problem, and must not spawn ffmpeg.</summary>
    [Fact]
    public void MissingMp3_ThrowsFileNotFoundWithoutFfmpeg()
    {
        int before = FfmpegAudioDecoder.DecodeInvocations;

        Assert.Throws<FileNotFoundException>(
            () => AudioUtils.ReadAudio(Path.Combine(_dir, "nope.mp3")));

        Assert.Equal(before, FfmpegAudioDecoder.DecodeInvocations);
    }

    // ── The rest of what Vernacula decodes without ffmpeg ──

    /// <summary>
    /// ⚠ AIFF NEEDED NO NEW DEPENDENCY AND STILL DID NOT WORK. NAudio.Core has always carried
    /// AiffFileReader, on every platform; ReadAudio simply never routed to it, so an
    /// uncompressed AIFF — a file format that is a header and raw PCM — was being handed to a
    /// subprocess, and failed outright when that subprocess was not installed.
    /// </summary>
    [Fact]
    public void Aiff_ReadsWithoutFfmpeg()
    {
        int before = FfmpegAudioDecoder.DecodeInvocations;
        var got = AudioUtils.ReadAudio(Fixture("tone_mono_8000.aiff"));

        Assert.Equal(before, FfmpegAudioDecoder.DecodeInvocations);
        Assert.Equal(8000, got.sampleRate);
        Assert.Equal(1, got.channels);
        AssertDecodedTone(got);
        AssertIs440HzTone(got);
    }

    /// <summary>Ogg Vorbis, decoded by NVorbis in-process.</summary>
    [Fact]
    public void OggVorbis_ReadsWithoutFfmpeg()
    {
        var got = ReadAssertingManagedOgg(() => AudioUtils.ReadAudio(Fixture("tone_mono_44100.ogg")));

        Assert.Equal(44100, got.sampleRate);
        Assert.Equal(1, got.channels);
        AssertDecodedTone(got);
        AssertIs440HzTone(got);
    }

    /// <summary>
    /// Ogg Opus, decoded by Concentus in-process. Opus only ever decodes at 48 kHz, so that
    /// is what must be reported — the decoder's rate, not the source's, which is the same
    /// contract Opus_ReportsTheDecoderSampleRateNotTheSourceRate pins for the ffmpeg path.
    /// </summary>
    [Fact]
    public void OggOpus_ReadsWithoutFfmpegAt48kHz()
    {
        var got = ReadAssertingManagedOgg(() => AudioUtils.ReadAudio(Fixture("tone_stereo_48000.opus")));

        Assert.Equal(48000, got.sampleRate);
        Assert.Equal(2, got.channels);
        AssertDecodedTone(got, expectedChannels: 2);
        AssertIs440HzTone(got);
    }

    /// <summary>
    /// ⚠ .ogg IS A CONTAINER, NOT A CODEC, AND THIS IS THE CASE THAT PROVES IT MATTERS.
    /// Messaging apps hand out Opus under a .ogg extension constantly. A decoder chosen from
    /// the extension would give this file to Vorbis, which would reject it — so the fixture
    /// here is Opus wearing the wrong name, and it has to come back as 48 kHz Opus.
    /// </summary>
    [Fact]
    public void OpusInsideAnOggExtension_IsSniffedNotAssumed()
    {
        var got = ReadAssertingManagedOgg(() => AudioUtils.ReadAudio(Fixture("tone_opus_in_ogg.ogg")));

        Assert.Equal(48000, got.sampleRate);   // Vorbis would have reported the source rate
        Assert.Equal(1, got.channels);
        AssertDecodedTone(got);
        AssertIs440HzTone(got);
    }

    /// <summary>
    /// The list docs/installation.md promises works on a machine with no FFmpeg at all. If an
    /// entry drops out of the managed table, the docs become wrong and this fails.
    /// </summary>
    [Theory]
    [InlineData(".wav")]
    [InlineData(".mp3")]
    [InlineData(".aiff")]
    [InlineData(".aif")]
    [InlineData(".ogg")]
    [InlineData(".oga")]
    [InlineData(".opus")]
    public void ManagedTable_ClaimsTheFormatsTheDocsPromise(string extension)
    {
        Assert.True(ManagedAudioDecoders.Handles(extension),
            $"{extension} is documented as decoding without FFmpeg, but no in-process decoder claims it");
    }

    /// <summary>And it must not claim the ones it cannot actually read.</summary>
    [Theory]
    [InlineData(".flac")]
    [InlineData(".m4a")]
    [InlineData(".aac")]
    [InlineData(".wma")]
    [InlineData(".mp4")]
    [InlineData(".mkv")]
    public void ManagedTable_DoesNotClaimFormatsThatNeedFfmpeg(string extension)
    {
        Assert.False(ManagedAudioDecoders.Handles(extension),
            $"{extension} has no in-process decoder, so claiming it would route the file to a "
            + "decoder that cannot read it instead of to FFmpeg");
    }

    /// <summary>
    /// An .ogg carrying something that is neither Vorbis nor Opus — FLAC-in-Ogg, Speex — has
    /// to reach ffmpeg rather than failing outright. FLAC-in-Ogg is the fixture because it is
    /// lossless, so the tone assertions stay exact; Speex overshoots [-1, 1] slightly, which
    /// would mean loosening a bound that is doing real work elsewhere.
    /// </summary>
    [Fact]
    public void OggWithAnUnsupportedCodec_FallsBackToFfmpeg()
    {
        RequireFfmpeg();
        string path = MakeFixture("flac_in.ogg", "-c:a", "flac");

        int before = FfmpegAudioDecoder.DecodeInvocations;
        var got = AudioUtils.ReadAudio(path);

        Assert.True(FfmpegAudioDecoder.DecodeInvocations > before,
            "an Ogg stream with no in-process decoder should have fallen through to ffmpeg");
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
