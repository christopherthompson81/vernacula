using System;
using System.IO;
using Vernacula.Base;
using Xunit;

// Vernacula.App and Vernacula.Base each declare an AudioUtils, and the point of this file is
// that they are different classes with different routing. Alias rather than import.
using AppAudioUtils = Vernacula.App.AudioUtils;
using FFmpegDecoder  = Vernacula.App.FFmpegDecoder;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// In-process audio routing for the desktop app's own <c>ReadAudio</c> (issue #176 and follow-up).
///
/// <para>
/// ⚠ THE DESKTOP APP HAS A SECOND ReadAudio, AND THAT IS THE WHOLE REASON THIS FILE EXISTS.
/// <c>Vernacula.App.AudioUtils.ReadAudio</c> is not <c>Vernacula.Base.AudioUtils.ReadAudio</c>:
/// it takes a stream index and routes video containers and FFmpeg-only formats through
/// FFmpeg.AutoGen in-process rather than through the ffmpeg executable. So the format-routing
/// tests in tests/Vernacula.Tests cover the CLI and say nothing about what the app does with
/// the same file, and #176 was reported against a released build of the app.
/// </para>
///
/// <para>
/// What is pinned here is the route, not the decode: MP3 must reach the managed NLayer
/// decoder. Both routes it could take instead need something the user may not have — the
/// FFmpeg shared libraries for AutoGen, or MediaFoundation, which NAudio 3 hands only to the
/// Windows target framework — and this project builds the net10.0 flavour, which has neither.
/// See tests/fixtures/README.md for where the fixture comes from.
/// </para>
/// </summary>
public class DesktopAudioRoutingTests
{
    private static string Fixture(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "fixtures", name);
        Assert.True(File.Exists(path),
            $"missing committed fixture '{name}' at {path} — check the <None Include=\"..\\fixtures\\*.mp3\"> "
            + "item in AsrBackendCoverage.csproj");
        return path;
    }

    // MP3: 44.1/48 kHz are MPEG-1, 16 kHz is MPEG-2 LSF, 8 kHz is MPEG-2.5 — different decode
    // paths, and the low-rate ones are where a bad decoder loses half the audio silently.
    // See the note in tests/Vernacula.Tests/AudioFormatIngestionTests.cs.
    [Theory]
    [InlineData("tone_mono_44100.mp3", 44100, 1)]
    [InlineData("tone_stereo_48000.mp3", 48000, 2)]
    [InlineData("tone_mono_16000.mp3", 16000, 1)]
    [InlineData("tone_mono_8000.mp3", 8000, 1)]
    public void Mp3_DecodesInProcessWithNativeLayout(string fixture, int expectedRate, int expectedChannels)
    {
        int before = Mp3Decoder.DecodeInvocations;

        var (samples, sampleRate, channels) = AppAudioUtils.ReadAudio(Fixture(fixture));

        Assert.True(Mp3Decoder.DecodeInvocations > before,
            "the desktop ReadAudio should decode MP3 through the in-process NLayer decoder, "
            + "not through FFmpeg or MediaFoundation");
        Assert.Equal(expectedRate, sampleRate);
        Assert.Equal(expectedChannels, channels);

        // Half a second of a 440 Hz tone: enough samples for the duration to be right, and
        // loud enough that a decoder returning a correctly-sized buffer of nothing fails.
        AssertHalfSecondTone(samples, sampleRate, channels);
    }

    /// <summary>
    /// The formats the managed table owns all have to reach it through the app's ReadAudio,
    /// not just MP3 — Ogg Opus is what a WhatsApp voice note arrives as, and AIFF needs no
    /// decoder the app did not already have.
    /// </summary>
    [Theory]
    [InlineData("tone_mono_44100.ogg", 44100, 1)]     // Ogg Vorbis
    [InlineData("tone_stereo_48000.opus", 48000, 2)]  // Ogg Opus, always 48 kHz
    [InlineData("tone_opus_in_ogg.ogg", 48000, 1)]    // Opus under a .ogg extension
    public void Ogg_DecodesInProcessWithNativeLayout(string fixture, int expectedRate, int expectedChannels)
    {
        int before = OggDecoder.DecodeInvocations;

        var (samples, sampleRate, channels) = AppAudioUtils.ReadAudio(Fixture(fixture));

        Assert.True(OggDecoder.DecodeInvocations > before,
            "the desktop ReadAudio should decode Ogg through the in-process decoder");
        Assert.Equal(expectedRate, sampleRate);
        Assert.Equal(expectedChannels, channels);
        AssertHalfSecondTone(samples, sampleRate, channels);
    }

    /// <summary>AIFF is read by NAudio.Core, which the app has always had.</summary>
    [Fact]
    public void Aiff_DecodesInProcess()
    {
        var (samples, sampleRate, channels) = AppAudioUtils.ReadAudio(Fixture("tone_mono_8000.aiff"));

        Assert.Equal(8000, sampleRate);
        Assert.Equal(1, channels);
        AssertHalfSecondTone(samples, sampleRate, channels);
    }

    /// <summary>
    /// ⚠ THE TWO ROUTING LISTS MUST NOT OVERLAP, AND THIS IS THE GUARD. #176 was two lists
    /// that disagreed about MP3. An extension named in both the managed table and the app's
    /// FFmpeg-only set is that same bug in miniature: whichever check runs first silently
    /// wins, and the loser is dead configuration that reads as intent.
    /// </summary>
    [Fact]
    public void FfmpegOnlyExtensions_DoNotOverlapTheManagedTable()
    {
        foreach (string ext in FFmpegDecoder.FfmpegAudioExtensions)
            Assert.False(ManagedAudioDecoders.Handles(ext),
                $"{ext} is claimed both by the managed decoder table and by the app's "
                + "FFmpeg-only list; exactly one of them should own it");

        foreach (string ext in FFmpegDecoder.VideoExtensions)
            Assert.False(ManagedAudioDecoders.Handles(ext),
                $"{ext} is a video container and cannot be decoded in-process, but the "
                + "managed table claims it");
    }

    private static void AssertHalfSecondTone(float[] samples, int sampleRate, int channels)
    {
        double seconds = (double)(samples.Length / channels) / sampleRate;
        Assert.InRange(seconds, 0.4, 0.75);

        float peak = 0f;
        foreach (float s in samples) peak = Math.Max(peak, Math.Abs(s));
        Assert.InRange(peak, 0.1f, 1.01f);
    }

    /// <summary>
    /// A stream index means "pick stream N of a multi-stream file", which is FFmpeg's job.
    /// The MP3 shortcut must not swallow that request and quietly return stream 0 instead.
    /// <para>
    /// Whether the FFmpeg route then succeeds depends on what is installed on the machine, so
    /// only the routing is asserted — the decode is allowed to fail.
    /// </para>
    /// </summary>
    [Fact]
    public void Mp3_WithAnExplicitStreamIndex_StaysOnTheFFmpegRoute()
    {
        int before = Mp3Decoder.DecodeInvocations;

        try
        {
            AppAudioUtils.ReadAudio(Fixture("tone_mono_44100.mp3"), streamIndex: 0);
        }
        catch
        {
            // No FFmpeg libraries or executable on this machine. Not what is under test.
        }

        Assert.Equal(before, Mp3Decoder.DecodeInvocations);
    }
}
