using System.Diagnostics;

namespace Vernacula.Base;

/// <summary>
/// Finds the <c>ffmpeg</c> and <c>ffprobe</c> executables.
/// <para>
/// ⚠ NOT JUST "ffmpeg" ANY MORE, BECAUSE PATH IS NOT THE ONLY PLACE THEY LIVE. The desktop
/// app can download a copy on demand (the same way it downloads models) into a directory
/// beside its model cache. Every spawn in this repo resolves through here so that a copy the
/// app fetched is also found by the CLI, by the TTS tools, and by tests — one install, every
/// surface. A system-wide install still wins, so a user who manages FFmpeg themselves keeps
/// exactly the behaviour they had.
/// </para>
/// <para>
/// ⚠ RESOLUTION ORDER IS DELIBERATE. The environment variable comes first so a developer or
/// a packager can pin a specific build without touching settings; PATH comes next so the
/// system install beats a stale download; the managed copy is the fallback that makes the
/// app work on a machine with no FFmpeg at all.
/// </para>
/// </summary>
public static class FfmpegBinaries
{
    /// <summary>Overrides every other source. Set it to a directory holding the executables.</summary>
    public const string DirectoryEnvironmentVariable = "VERNACULA_FFMPEG_DIR";

    /// <summary>
    /// Where the desktop app puts a downloaded copy: beside the model cache, so uninstalling
    /// means deleting one directory. Shared with the CLI by construction rather than by
    /// configuration — both compute the same path.
    /// </summary>
    public static string ManagedDirectory => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Vernacula", "tools", "ffmpeg");

    /// <summary>
    /// What to pass to <see cref="ProcessStartInfo"/> for <paramref name="name"/> — an
    /// absolute path when a managed copy is present, otherwise the bare name for PATH to
    /// resolve. Never throws: a missing binary is reported when the spawn fails, where the
    /// message can say what was actually being attempted.
    /// </summary>
    public static string ResolveExecutable(string name)
    {
        string fileName = OperatingSystem.IsWindows() ? name + ".exe" : name;

        string? pinned = Environment.GetEnvironmentVariable(DirectoryEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(pinned))
        {
            string candidate = Path.Combine(pinned, fileName);
            if (File.Exists(candidate)) return candidate;
        }

        if (ExistsOnPath(name)) return name;

        string managed = Path.Combine(ManagedDirectory, fileName);
        return File.Exists(managed) ? managed : name;
    }

    /// <summary>True when <paramref name="name"/> can actually be launched.</summary>
    /// <remarks>
    /// Not cached: one process spawn, and callers use this for provisioning decisions,
    /// test gating and error messages rather than in a loop.
    /// </remarks>
    public static bool IsAvailable(string name)
    {
        try
        {
            using var probe = Process.Start(new ProcessStartInfo(ResolveExecutable(name), "-version")
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
            // Win32Exception (not on PATH) and anything else the platform throws while
            // spawning both mean the same thing here: no usable binary by that name.
            return false;
        }
    }

    /// <summary>
    /// The message to show when <paramref name="name"/> could not be launched. Written once
    /// here because every caller needs to say the same three things, and the raw
    /// Win32Exception ("An error occurred trying to start process 'ffprobe'") says none of
    /// them.
    /// </summary>
    public static string MissingBinaryMessage(string name, string path) =>
        $"Could not run '{name}', needed to decode '{Path.GetFileName(path)}'. "
        + "Vernacula decodes WAV, MP3, AIFF, Ogg Vorbis and Ogg Opus on its own, but this "
        + "format needs FFmpeg. Install it and make sure it is on PATH (see the prerequisites "
        + "in docs/installation.md), let the desktop app download a copy for you, or convert "
        + "the file to WAV or MP3.";

    private static bool ExistsOnPath(string name)
    {
        string fileName = OperatingSystem.IsWindows() ? name + ".exe" : name;
        string? path = Environment.GetEnvironmentVariable("PATH");
        if (path is null) return false;

        foreach (string entry in path.Split(Path.PathSeparator))
        {
            // ⚠ TRIM THE QUOTES. Windows accepts quoted PATH entries ("C:\Program Files\..."),
            // and Path.Combine on one keeps the quote, so File.Exists says no. Missing a real
            // install that way is not harmless: resolution falls through to the downloaded
            // copy, silently shadowing the FFmpeg the user chose to install — the opposite of
            // what docs/installation.md promises.
            string dir = entry.Trim().Trim('"');
            if (dir.Length == 0) continue;
            try
            {
                if (File.Exists(Path.Combine(dir, fileName))) return true;
            }
            catch
            {
                // A malformed PATH entry is not worth failing over; skip it.
            }
        }
        return false;
    }
}
