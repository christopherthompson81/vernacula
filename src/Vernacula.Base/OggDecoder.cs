using Concentus;
using Concentus.Oggfile;
using NAudio.Vorbis;
using NAudio.Wave;

namespace Vernacula.Base;

/// <summary>
/// Decodes Ogg-contained audio in-process: Vorbis via NVorbis, Opus via Concentus.
/// Both are pure managed, so this works on every platform with no FFmpeg install.
/// <para>
/// ⚠ THE EXTENSION DOES NOT NAME THE CODEC, SO THIS SNIFFS. Ogg is a container, and
/// <c>.ogg</c> holds Vorbis or Opus about equally often in the wild — messaging apps
/// hand out Opus under both <c>.ogg</c> and <c>.opus</c>. Dispatching on the extension
/// would decode a large share of real files with the wrong decoder, so the first page
/// is read and the codec taken from its identification header. <c>.opus</c> gets the
/// same treatment rather than being assumed: it costs one 128-byte read.
/// </para>
/// <para>
/// ⚠ OPUS ALWAYS COMES BACK AT 48 kHz, whatever the source was, because that is the
/// only rate libopus decodes to. That is the decoder's rate and therefore the honest
/// answer here — the same property <see cref="FfmpegAudioDecoder"/> is careful to
/// preserve, and the reason ReadAudio's contract is "the decoder's native layout"
/// rather than "the file's declared layout".
/// </para>
/// </summary>
public static class OggDecoder
{
    /// <summary>Number of decodes performed. Test seam for asserting routing.</summary>
    internal static int DecodeInvocations;

    /// <summary>Opus decodes at 48 kHz and nothing else.</summary>
    private const int OpusRate = 48_000;

    /// <summary>Enough to cover the first Ogg page header, its segment table, and the codec header.</summary>
    private const int SniffBytes = 256;

    /// <summary>
    /// Decode <paramref name="path"/> to interleaved float samples in [-1, 1], with the
    /// decoder's own sample rate and channel count.
    /// </summary>
    /// <exception cref="FileNotFoundException">No file at <paramref name="path"/>.</exception>
    /// <exception cref="InvalidDataException">
    /// Not an Ogg stream, or one carrying a codec other than Vorbis or Opus.
    /// </exception>
    public static (float[] samples, int sampleRate, int channels) Decode(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Audio file not found: {path}", path);

        byte[] head = ReadHead(path);
        if (!head.AsSpan().StartsWith("OggS"u8))
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' is not an Ogg stream (no OggS capture pattern).");

        Interlocked.Increment(ref DecodeInvocations);

        int opusHead = IndexOf(head, "OpusHead"u8);
        if (opusHead >= 0)
            return DecodeOpus(path, ChannelsFromOpusHead(head, opusHead, path));

        // Vorbis identification header: packet type 0x01 followed by "vorbis".
        if (IndexOf(head, "\x01vorbis"u8) >= 0)
            return DecodeVorbis(path);

        throw new InvalidDataException(
            $"'{Path.GetFileName(path)}' is an Ogg stream, but carries neither Vorbis nor Opus "
            + "(Speex, FLAC-in-Ogg and video codecs are not decoded in-process).");
    }

    private static byte[] ReadHead(string path)
    {
        using var stream = File.OpenRead(path);
        var head = new byte[SniffBytes];
        int filled = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        return filled == head.Length ? head : head[..filled];
    }

    /// <summary>
    /// OpusHead is magic(8) + version(1) + channel count(1) + pre-skip(2) + input rate(4) + …
    /// <para>
    /// ⚠ THE CHANNEL COUNT HAS TO COME FROM HERE, NOT FROM A DEFAULT. Concentus fixes the
    /// channel count when the decoder is constructed, which is before OpusOggReadStream has
    /// parsed anything. Guessing stereo would decode a mono file into duplicated channels and
    /// report 2 — audio that sounds right, twice the samples, and a channel count the rest of
    /// the pipeline would then average back down.
    /// </para>
    /// </summary>
    private static int ChannelsFromOpusHead(byte[] head, int offset, string path)
    {
        int channelIndex = offset + 9;
        if (channelIndex >= head.Length)
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' has a truncated OpusHead header.");

        int channels = head[channelIndex];
        if (channels is < 1 or > 8)
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' declares {channels} Opus channels, which is out of range.");

        return channels;
    }

    private static (float[] samples, int sampleRate, int channels) DecodeOpus(string path, int channels)
    {
        using var stream = File.OpenRead(path);
        var ogg = new OpusOggReadStream(OpusCodecFactory.CreateDecoder(OpusRate, channels), stream);

        var samples = new float[OpusRate * channels];  // one second, then grow
        int count = 0;

        // ⚠ A BOUND ON EMPTY PACKETS, BECAUSE THE LOOP CONDITION IS NOT OURS. HasNextPacket
        // belongs to the Ogg reader, and DecodeNextPacket returning null is how it reports a
        // packet it could not use. Trusting it to always advance would make a malformed or
        // truncated file — which is to say, a file a user could plausibly hand us — able to
        // spin this thread forever with no output and no error. A run of empty packets this
        // long is a broken stream either way, so stop and let the caller fall back to FFmpeg.
        const int maxConsecutiveEmpty = 64;
        int emptyRun = 0;

        while (ogg.HasNextPacket)
        {
            short[] pcm = ogg.DecodeNextPacket();
            if (pcm is null || pcm.Length == 0)
            {
                if (++emptyRun > maxConsecutiveEmpty) break;
                continue;
            }
            emptyRun = 0;

            if (count + pcm.Length > samples.Length)
                Array.Resize(ref samples, Math.Max(samples.Length * 2, count + pcm.Length));

            for (int i = 0; i < pcm.Length; i++)
                samples[count + i] = pcm[i] / 32768f;
            count += pcm.Length;
        }

        if (count == 0)
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' parsed as Ogg Opus but decoded to no samples.");

        if (count != samples.Length)
            Array.Resize(ref samples, count);
        return (samples, OpusRate, channels);
    }

    private static (float[] samples, int sampleRate, int channels) DecodeVorbis(string path)
    {
        using var reader = new VorbisWaveReader(path);
        int sampleRate = reader.WaveFormat.SampleRate;
        int channels   = reader.WaveFormat.Channels;

        // ⚠ THROUGH THE INTERFACE, for the same reason as the WAV path in AudioUtils:
        // VorbisWaveReader carries both a float and a byte Read overload.
        ISampleProvider provider = reader;

        var samples = new float[Math.Max(sampleRate * channels, 1 << 16)];
        int count = 0;

        while (true)
        {
            if (count == samples.Length)
                Array.Resize(ref samples, samples.Length * 2);

            int read = provider.Read(samples.AsSpan(count));
            if (read <= 0) break;
            count += read;
        }

        if (count == 0)
            throw new InvalidDataException(
                $"'{Path.GetFileName(path)}' parsed as Ogg Vorbis but decoded to no samples.");

        if (count != samples.Length)
            Array.Resize(ref samples, count);
        return (samples, sampleRate, channels);
    }

    private static int IndexOf(byte[] haystack, ReadOnlySpan<byte> needle)
    {
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
            if (haystack.AsSpan(i, needle.Length).SequenceEqual(needle))
                return i;
        return -1;
    }
}
