using NAudio.Wave;

namespace Vernacula.Base;

/// <summary>
/// The formats Vernacula decodes in-process, and which decoder owns each one.
/// <para>
/// ⚠ ONE TABLE, BECAUSE THE ROUTING IS THE THING THAT KEEPS BREAKING. #156 and #176 were
/// both routing bugs, not decoding bugs: a format went to the wrong decoder, or to one that
/// had quietly stopped existing. Both <see cref="AudioUtils.ReadAudio"/> and the desktop
/// app's own ReadAudio consult this table, so there is one answer to "what needs FFmpeg?"
/// rather than two lists that drifted apart — which is how the desktop app kept working for
/// MP3 on Windows for a while after the CLI stopped.
/// </para>
/// <para>
/// Everything absent from this table — FLAC, M4A, AAC, WMA, non-PCM WAV, and every video
/// container — is decoded by FFmpeg. See <see cref="FfmpegAudioDecoder"/>.
/// </para>
/// </summary>
public static class ManagedAudioDecoders
{
    /// <summary>A decoder and the name to use for it when reporting a failure.</summary>
    public readonly record struct Entry(string Name, Func<string, (float[], int, int)> Decode);

    private static readonly Dictionary<string, Entry> ByExtension =
        new(StringComparer.OrdinalIgnoreCase)
        {
            // PCM and IEEE-float WAV. NAudio.Core reads these on every platform; anything else
            // in a .wav container (mu-law, A-law, ADPCM, MP3-in-WAV) is ACM's job and falls
            // through to FFmpeg.
            [".wav"]  = new("WAV", ReadWithNAudio),

            // MPEG-1/2/2.5 Layer I-III, via NLayer. See Mp3Decoder for why this is not NAudio.
            [".mp3"]  = new("MP3", Mp3Decoder.Decode),

            // ⚠ AIFF NEEDS NOTHING NEW. AiffFileReader is in NAudio.Core and has always worked
            // on every platform; ReadAudio simply never routed to it, so uncompressed AIFF was
            // being handed to FFmpeg for no reason at all.
            [".aif"]  = new("AIFF", ReadWithNAudio),
            [".aiff"] = new("AIFF", ReadWithNAudio),

            // Ogg containers. OggDecoder sniffs the codec rather than trusting the extension,
            // because .ogg carries Opus at least as often as Vorbis.
            [".ogg"]  = new("Ogg", OggDecoder.Decode),
            [".oga"]  = new("Ogg", OggDecoder.Decode),
            [".opus"] = new("Ogg", OggDecoder.Decode),
        };

    /// <summary>True when <paramref name="extension"/> has an in-process decoder.</summary>
    public static bool Handles(string extension) => ByExtension.ContainsKey(extension);

    /// <summary>
    /// The in-process decoder for <paramref name="extension"/>, if there is one.
    /// Public because the desktop app's own ReadAudio routes through this same table —
    /// that is the point of the table.
    /// </summary>
    public static bool TryGet(string extension, out Entry entry) =>
        ByExtension.TryGetValue(extension, out entry);

    /// <summary>
    /// Exceptions that mean "this decoder cannot read these bytes", as opposed to "something
    /// went wrong". Only these hand the file on to FFmpeg.
    /// <para>
    /// ⚠ ArgumentException IS IN THE LIST ON PURPOSE, AND IT IS THE LOOSE ONE. NVorbis reports
    /// a stream it cannot open that way, so leaving it out would turn a renamed file into a
    /// hard failure. The cost of it being too loose is bounded: the file goes to FFmpeg, and
    /// if that fails too the combined error names this exception and its message, so a genuine
    /// bug in our own argument handling still surfaces rather than being swallowed.
    /// </para>
    /// <para>
    /// FileNotFoundException is deliberately NOT here. A missing file is not a format question,
    /// and callers already handle it.
    /// </para>
    /// </summary>
    public static bool IsFormatRejection(Exception ex) =>
        ex is NotSupportedException
           or FormatException
           or InvalidDataException
           or EndOfStreamException
           or ArgumentException;

    /// <summary>
    /// WAV and AIFF, both read by NAudio's AudioFileReader.
    /// </summary>
    private static (float[] samples, int sampleRate, int channels) ReadWithNAudio(string path)
    {
        using var reader = new AudioFileReader(path);
        int sampleRate = reader.WaveFormat.SampleRate;
        int channels   = reader.WaveFormat.Channels;

        var list   = new List<float>(sampleRate * channels * 10);
        var buffer = new float[8192];
        int read;
        // ⚠ THROUGH THE INTERFACE, DELIBERATELY. AudioFileReader carries both
        // Read(Span<float>) (ISampleProvider) and Read(Span<byte>) (WaveStream) in NAudio 3;
        // going through the interface pins the float overload rather than leaving it to
        // overload resolution on a call whose failure mode is a silent byte-wise read.
        ISampleProvider readerSamples = reader;
        while ((read = readerSamples.Read(buffer)) > 0)
            for (int i = 0; i < read; i++) list.Add(buffer[i]);

        return (list.ToArray(), sampleRate, channels);
    }
}
