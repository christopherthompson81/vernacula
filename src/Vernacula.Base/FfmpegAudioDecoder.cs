using System.Buffers.Binary;
using System.Diagnostics;
using System.Text;

namespace Vernacula.Base;

/// <summary>
/// Decodes audio by shelling out to <c>ffmpeg</c>, for everything NAudio's
/// cross-platform build can't read.
/// <para>
/// ⚠ SUBPROCESS, NOT A LIBRARY BINDING, DELIBERATELY. <c>Vernacula.Avalonia</c>
/// uses FFmpeg.AutoGen in-process, but <c>Vernacula.Base</c> is referenced by
/// every CLI and test project; a native package reference here would land in
/// all of them. The <c>ffmpeg</c> EXECUTABLE has to be on PATH — which is a
/// different requirement from the FFmpeg shared libraries AutoGen needs, and
/// docs/installation.md now asks for both. See issue #156.
/// </para>
/// <para>
/// ⚠ METADATA COMES FROM THE WAV HEADER FFMPEG EMITS, NOT FROM <c>ffprobe</c>.
/// An earlier draft probed for the rate and channel count and then decoded raw
/// <c>f32le</c>. That silently desynchronises whenever the decoder's output rate
/// differs from the container's declared rate — implicit-SBR HE-AAC is the
/// common case, where ffmpeg emits at double the probed rate. Callers would then
/// resample from the wrong rate and get audio at the wrong pitch with timestamps
/// off by 2x, with nothing anywhere reporting an error. Reading the header of the
/// stream we are about to consume makes the two impossible to disagree.
/// </para>
/// <para>
/// Nothing is resampled or downmixed: callers get the decoder's native layout,
/// the same contract as the NAudio path.
/// </para>
/// </summary>
public static class FfmpegAudioDecoder
{
    /// <summary>Number of decodes performed. Test seam for asserting routing.</summary>
    internal static int DecodeInvocations;

    /// <summary>True when <c>ffmpeg</c> can be launched.</summary>
    /// <remarks>
    /// Not cached: one process spawn, and callers use this for test gating and
    /// error messages rather than in a loop.
    /// </remarks>
    public static bool IsAvailable
    {
        get
        {
            try
            {
                using var probe = Process.Start(new ProcessStartInfo("ffmpeg", "-version")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                });
                if (probe is null) return false;
                probe.StandardOutput.ReadToEnd();
                probe.StandardError.ReadToEnd();
                probe.WaitForExit();
                return probe.ExitCode == 0;
            }
            catch
            {
                // Win32Exception (not on PATH) and anything else the platform throws
                // while spawning both mean the same thing here: no usable ffmpeg.
                return false;
            }
        }
    }

    /// <summary>
    /// Decode <paramref name="path"/> to interleaved float samples in [-1, 1],
    /// with the decoder's own sample rate and channel count.
    /// </summary>
    /// <param name="streamIndex">Audio stream to decode, for files carrying more
    /// than one. 0 is the first audio stream, not the first stream.</param>
    /// <exception cref="InvalidOperationException">ffmpeg is missing, the file has
    /// no decodable audio stream, or the decode failed.</exception>
    public static (float[] samples, int sampleRate, int channels) Decode(string path, int streamIndex = 0)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Audio file not found: {path}", path);

        Interlocked.Increment(ref DecodeInvocations);

        // -f wav with pcm_f32le: the header carries the rate and channel count the
        // decoder actually produced, and the payload is plain float32 after it.
        // On a pipe ffmpeg cannot backfill the RIFF/data sizes, so those fields are
        // placeholders -- we read the payload to EOF and ignore them.
        var psi = new ProcessStartInfo("ffmpeg")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string a in new[]
                 {
                     "-v", "error", "-nostdin", "-i", path,
                     "-map", $"0:a:{streamIndex}",
                     "-f", "wav", "-acodec", "pcm_f32le", "-",
                 })
            psi.ArgumentList.Add(a);

        Process proc;
        try
        {
            proc = Process.Start(psi)
                   ?? throw new InvalidOperationException("Process.Start returned null for ffmpeg.");
        }
        catch (Exception ex) when (ex is not InvalidOperationException)
        {
            throw new InvalidOperationException(
                $"Could not run 'ffmpeg', needed to decode '{Path.GetFileName(path)}'. "
                + "Vernacula decodes everything except PCM/IEEE-float WAV by running the ffmpeg "
                + "executable, which must be on PATH (see the prerequisites in docs/installation.md). "
                + "Alternatively, convert the file to PCM WAV.", ex);
        }

        using (proc)
        {
            // ⚠ DRAIN stderr ON ANOTHER THREAD. A decode of any real length fills the
            // stdout pipe; if we blocked reading stderr first, ffmpeg would block writing
            // stdout and neither side would move. Reading stdout to completion first has
            // the mirror-image deadlock when ffmpeg is chatty on stderr.
            var stderrTask = proc.StandardError.ReadToEndAsync();
            try
            {
                var stream = proc.StandardOutput.BaseStream;
                var (sampleRate, channels) = ReadWavHeader(stream, proc, stderrTask, path);
                float[] samples = ReadSamplesToEnd(stream);

                proc.WaitForExit();
                string stderr = stderrTask.GetAwaiter().GetResult();
                if (proc.ExitCode != 0)
                    throw Failed(path, proc.ExitCode, stderr);

                return (samples, sampleRate, channels);
            }
            catch
            {
                // Don't leave ffmpeg running on the way out. Disposing the Process does
                // not kill the child, and an abandoned decode holds a pipe and a core.
                try { if (!proc.HasExited) proc.Kill(entireProcessTree: true); } catch { /* best-effort */ }
                throw;
            }
        }
    }

    private static InvalidOperationException Failed(string path, int exitCode, string stderr) =>
        new($"ffmpeg failed (exit {exitCode}) decoding '{Path.GetFileName(path)}'"
            + (stderr.Trim().Length > 0 ? $": {stderr.Trim()}" : "."));

    /// <summary>
    /// Consume the RIFF/WAVE header, returning the format ffmpeg actually emitted and
    /// leaving <paramref name="stream"/> positioned at the first sample byte.
    /// </summary>
    private static (int sampleRate, int channels) ReadWavHeader(
        Stream stream, Process proc, Task<string> stderrTask, string path)
    {
        Span<byte> riff = stackalloc byte[12];
        if (!TryReadExactly(stream, riff))
        {
            // No header at all: ffmpeg produced nothing, which for a file with no
            // decodable audio stream is the normal failure. Its stderr says why.
            proc.WaitForExit();
            throw Failed(path, proc.ExitCode, stderrTask.GetAwaiter().GetResult());
        }

        if (!riff[..4].SequenceEqual("RIFF"u8) || !riff[8..12].SequenceEqual("WAVE"u8))
            throw new InvalidOperationException(
                $"ffmpeg did not emit a WAV stream for '{Path.GetFileName(path)}'.");

        Span<byte> chunkHeader = stackalloc byte[8];
        int sampleRate = 0, channels = 0;

        while (TryReadExactly(stream, chunkHeader))
        {
            uint chunkSize = BinaryPrimitives.ReadUInt32LittleEndian(chunkHeader[4..]);

            if (chunkHeader[..4].SequenceEqual("fmt "u8))
            {
                // Only the first 8 bytes matter here: format tag, channels, sample rate.
                // The codec is pcm_f32le because we asked for it, so the rest is noise.
                var fmt = new byte[chunkSize];
                if (!TryReadExactly(stream, fmt))
                    throw new InvalidOperationException(
                        $"ffmpeg emitted a truncated WAV header for '{Path.GetFileName(path)}'.");

                channels = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(2));
                sampleRate = (int)BinaryPrimitives.ReadUInt32LittleEndian(fmt.AsSpan(4));
            }
            else if (chunkHeader[..4].SequenceEqual("data"u8))
            {
                if (sampleRate <= 0 || channels <= 0)
                    throw new InvalidOperationException(
                        $"ffmpeg emitted a WAV stream with no format chunk for '{Path.GetFileName(path)}'.");
                return (sampleRate, channels);
            }
            else
            {
                // LIST/fact/whatever. Chunks are word-aligned, so an odd size is padded.
                Skip(stream, chunkSize + (chunkSize & 1));
            }
        }

        throw new InvalidOperationException(
            $"ffmpeg emitted a WAV stream with no data chunk for '{Path.GetFileName(path)}'.");
    }

    /// <summary>
    /// Read float32 samples to end of stream.
    /// </summary>
    /// <remarks>
    /// ⚠ NOT VIA MemoryStream. Buffering the payload and calling ToArray() would hold two
    /// full copies at peak, and MemoryStream tops out at int.MaxValue bytes - about 1.5
    /// hours of 48 kHz stereo, which is an ordinary meeting recording for an ASR tool.
    /// Growing the float[] directly keeps one copy, and the ceiling becomes the array
    /// limit rather than half of it.
    /// </remarks>
    private static float[] ReadSamplesToEnd(Stream stream)
    {
        var samples = new float[1 << 16];
        int sampleCount = 0;

        var chunk = new byte[1 << 16];
        int carry = 0;  // bytes of a split sample left over from the previous read

        while (true)
        {
            int read = stream.Read(chunk, carry, chunk.Length - carry);
            if (read <= 0) break;

            int available = carry + read;
            int whole = available / sizeof(float);

            if (sampleCount + whole > samples.Length)
                Array.Resize(ref samples, Math.Max(samples.Length * 2, sampleCount + whole));

            Buffer.BlockCopy(chunk, 0, samples, sampleCount * sizeof(float), whole * sizeof(float));
            sampleCount += whole;

            // A read can end mid-sample; keep the tail for the next pass.
            carry = available - whole * sizeof(float);
            if (carry > 0)
                Buffer.BlockCopy(chunk, whole * sizeof(float), chunk, 0, carry);
        }

        if (sampleCount != samples.Length)
            Array.Resize(ref samples, sampleCount);
        return samples;
    }

    private static bool TryReadExactly(Stream stream, Span<byte> buffer)
    {
        int filled = 0;
        while (filled < buffer.Length)
        {
            int read = stream.Read(buffer[filled..]);
            if (read <= 0) return false;
            filled += read;
        }
        return true;
    }

    private static void Skip(Stream stream, long count)
    {
        Span<byte> scratch = stackalloc byte[4096];
        while (count > 0)
        {
            int read = stream.Read(scratch[..(int)Math.Min(count, scratch.Length)]);
            if (read <= 0) return;
            count -= read;
        }
    }
}
