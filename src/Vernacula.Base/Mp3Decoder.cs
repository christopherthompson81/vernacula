using NLayer;

namespace Vernacula.Base;

/// <summary>
/// Decodes MP3 in-process with NLayer, a pure-managed MPEG-1/2/2.5 Layer I–III decoder.
/// <para>
/// ⚠ THIS EXISTS BECAUSE NAudio 3 TOOK MP3 AWAY, AND FFMPEG IS NOT A SUBSTITUTE FOR IT.
/// NAudio 2's net6.0 assembly still carried <c>Mp3FileReader</c> and its ACM/MediaFoundation
/// P/Invokes, so a net10.0 resolve got in-process MP3 decoding that worked on Windows and
/// threw on Linux. NAudio 3 moved those into <c>NAudio.WinMM</c>/<c>NAudio.Wasapi</c>, which
/// ship only to a Windows target framework, so this project has none of them on any host.
/// #156 answered that by routing MP3 to the FFmpeg executable — which fixed Linux and broke
/// Windows, where MP3 had never needed an external binary and most users have none installed
/// (#176). A managed decoder is the only answer that holds on all three platforms.
/// </para>
/// <para>
/// ⚠ NAudio.Core STILL HAS THE MP3 *PLUMBING*, JUST NO DECODER: <c>Mp3FileReaderBase</c> and
/// <c>IMp3FrameDecompressor</c> are there, waiting for a frame decompressor to be supplied.
/// Going through them would mean re-implementing NLayer's <c>IMpegFrame</c> over NAudio's
/// <c>Mp3Frame</c> for the sake of a <see cref="NAudio.Wave.WaveStream"/> nothing here wants:
/// every caller reads the whole file to a float[]. NLayer's own <see cref="MpegFile"/> does
/// that directly, and handles ID3v1/ID3v2 tags and Xing/LAME headers on the way.
/// </para>
/// <para>
/// ⚠ THE NLayer VERSION IS LOAD-BEARING — 1.16.0 LOSES HALF OF EVERY LOW-RATE FILE. It
/// decodes about half of any MPEG-2 / MPEG-2.5 stream (every sample rate at or below 24 kHz)
/// and reports the full duration while doing it, so a 16 kHz mono voice memo — the shape an
/// ASR tool is most often handed — comes back at half length with no error raised anywhere.
/// The pin lives in Vernacula.Base.csproj; tests/fixtures carries 8 kHz and 16 kHz fixtures
/// so a downgrade fails a test rather than silently truncating someone's transcript.
/// </para>
/// <para>
/// ⚠ ENCODER DELAY IS NOT TRIMMED, DELIBERATELY. NLayer decodes every frame the file
/// contains, so the ~1100 samples (~25 ms) of encoder priming a LAME file starts with come
/// back as leading near-silence; FFmpeg would consume the LAME gapless tag and drop them.
/// That shifts word timestamps by under 25 ms, and it is exactly what NAudio 2 did on
/// Windows for years, so it is not a change anyone is calibrated against.
/// </para>
/// <para>
/// Nothing is resampled or downmixed: callers get the decoder's native layout, the same
/// contract as <see cref="AudioUtils.ReadAudio"/>'s other paths.
/// </para>
/// </summary>
public static class Mp3Decoder
{
    /// <summary>Number of decodes performed. Test seam for asserting routing.</summary>
    internal static int DecodeInvocations;

    /// <summary>
    /// Decode <paramref name="path"/> to interleaved float samples in [-1, 1], with the
    /// file's own sample rate and channel count.
    /// </summary>
    /// <exception cref="FileNotFoundException">No file at <paramref name="path"/>.</exception>
    /// <exception cref="InvalidDataException">
    /// The bytes are not MPEG audio, or carry no decodable frames.
    /// </exception>
    public static (float[] samples, int sampleRate, int channels) Decode(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Audio file not found: {path}", path);

        Interlocked.Increment(ref DecodeInvocations);

        using var stream = File.OpenRead(path);
        // Throws InvalidDataException("Not a valid MPEG file!") when no frame sync is found,
        // which is what an .mp3 that is really something else looks like from here.
        using var mpeg = new MpegFile(stream);

        int sampleRate = mpeg.SampleRate;
        int channels   = mpeg.Channels;
        if (sampleRate <= 0 || channels <= 0)
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' has no usable MPEG audio format "
                + $"(rate {sampleRate}, {channels} channels).");

        // mpeg.Length is the decoded size in BYTES of float32 output, from the Xing/LAME
        // frame count when there is one. A hint only: it is absent on a CBR file with no
        // Xing header, and overstates a truncated file, so the array still grows on demand.
        long hint = mpeg.Length / sizeof(float);
        var samples = new float[hint is > 0 and < (1 << 24) ? (int)hint : 1 << 16];
        int count = 0;

        while (true)
        {
            if (count == samples.Length)
                Array.Resize(ref samples, samples.Length * 2);

            int read = mpeg.ReadSamples(samples, count, samples.Length - count);
            if (read <= 0) break;
            count += read;
        }

        if (count == 0)
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' parsed as MPEG audio but decoded to no samples.");

        if (count != samples.Length)
            Array.Resize(ref samples, count);
        return (samples, sampleRate, channels);
    }
}
