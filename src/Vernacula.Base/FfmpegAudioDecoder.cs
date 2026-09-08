using System.Diagnostics;
using System.Globalization;

namespace Vernacula.Base;

/// <summary>
/// Decodes audio by shelling out to <c>ffmpeg</c>, for everything NAudio's
/// cross-platform build can't read.
/// <para>
/// ⚠ SUBPROCESS, NOT A LIBRARY BINDING, DELIBERATELY. <c>Vernacula.Avalonia</c>
/// uses FFmpeg.AutoGen in-process, but <c>Vernacula.Base</c> is referenced by
/// every CLI and test project; a native package reference here would land in
/// all of them. FFmpeg is already a documented prerequisite (see README), and
/// this matches how the rest of the CLI surface treats it. See issue #156.
/// </para>
/// <para>
/// Sample rate and channel count come from <c>ffprobe</c> and the decode does
/// not resample or downmix — callers get the file's native layout, same as the
/// NAudio path, and do their own conversion.
/// </para>
/// </summary>
public static class FfmpegAudioDecoder
{
    /// <summary>True when both <c>ffmpeg</c> and <c>ffprobe</c> can be launched.</summary>
    /// <remarks>
    /// Not cached: a probe costs one process spawn, and callers use this for
    /// test gating and error messages rather than in a loop.
    /// </remarks>
    public static bool IsAvailable => CanRun("ffmpeg") && CanRun("ffprobe");

    private static bool CanRun(string exe)
    {
        try
        {
            using var probe = Process.Start(new ProcessStartInfo(exe, "-version")
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

    /// <summary>
    /// Decode <paramref name="path"/> to interleaved float samples in [-1, 1].
    /// Returns the file's native sample rate and channel count.
    /// </summary>
    /// <param name="streamIndex">Audio stream to decode, for files carrying
    /// more than one. 0 is the first audio stream, not the first stream.</param>
    /// <exception cref="InvalidOperationException">ffmpeg/ffprobe is missing,
    /// the file has no audio stream, or the decode failed.</exception>
    public static (float[] samples, int sampleRate, int channels) Decode(string path, int streamIndex = 0)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Audio file not found: {path}", path);

        var (sampleRate, channels) = Probe(path, streamIndex);

        // -f f32le with no -ar/-ac keeps the stream's own rate and layout, so the
        // ffprobe numbers above describe what comes back on stdout.
        byte[] raw = RunToBytes(
            "ffmpeg",
            ["-v", "error", "-nostdin", "-i", path, "-map", $"0:a:{streamIndex}",
             "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            path);

        var samples = new float[raw.Length / sizeof(float)];
        Buffer.BlockCopy(raw, 0, samples, 0, samples.Length * sizeof(float));
        return (samples, sampleRate, channels);
    }

    private static (int sampleRate, int channels) Probe(string path, int streamIndex)
    {
        byte[] stdout = RunToBytes(
            "ffprobe",
            ["-v", "error", "-select_streams", $"a:{streamIndex}",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", path],
            path);

        // "48000,2" — one line per selected stream; -select_streams pins it to one.
        string text = System.Text.Encoding.UTF8.GetString(stdout).Trim();
        string[] fields = text.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                              .FirstOrDefault()?
                              .Trim()
                              .Split(',') ?? [];

        if (fields.Length < 2
            || !int.TryParse(fields[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int sampleRate)
            || !int.TryParse(fields[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int channels)
            || sampleRate <= 0 || channels <= 0)
        {
            throw new InvalidOperationException(
                $"No decodable audio stream (index {streamIndex}) in '{path}'. "
                + $"ffprobe reported: {(text.Length == 0 ? "<nothing>" : text)}");
        }

        return (sampleRate, channels);
    }

    private static byte[] RunToBytes(string exe, string[] args, string path)
    {
        var psi = new ProcessStartInfo(exe)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string a in args) psi.ArgumentList.Add(a);

        Process proc;
        try
        {
            proc = Process.Start(psi)
                   ?? throw new InvalidOperationException($"Failed to start {exe}.");
        }
        catch (Exception ex) when (ex is not InvalidOperationException)
        {
            throw new InvalidOperationException(
                $"Could not run '{exe}', needed to decode '{Path.GetFileName(path)}'. "
                + "Vernacula decodes everything except PCM/IEEE-float WAV through FFmpeg. "
                + "Install FFmpeg and make sure it is on PATH (see the prerequisites in "
                + "README.md), or convert the file to PCM WAV.", ex);
        }

        using (proc)
        {
            // ⚠ DRAIN stderr ON ANOTHER THREAD. A decode of any real length fills the
            // stdout pipe; if we blocked reading stderr first, ffmpeg would block writing
            // stdout and neither side would move. Reading stdout to completion first has
            // the mirror-image deadlock when ffmpeg is chatty on stderr.
            var stderrTask = proc.StandardError.ReadToEndAsync();

            using var buffer = new MemoryStream();
            proc.StandardOutput.BaseStream.CopyTo(buffer);
            proc.WaitForExit();
            string stderr = stderrTask.GetAwaiter().GetResult();

            if (proc.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    $"{exe} failed (exit {proc.ExitCode}) decoding '{Path.GetFileName(path)}'"
                    + (stderr.Length > 0 ? $": {stderr.Trim()}" : "."));
            }

            return buffer.ToArray();
        }
    }
}
