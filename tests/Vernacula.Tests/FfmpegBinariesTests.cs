using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests;

/// <summary>
/// How the FFmpeg executables are located.
///
/// <para>
/// ⚠ WHY THIS IS NOT JUST "ffmpeg". The desktop app can download a copy on demand, and the
/// point of putting the lookup in <see cref="Vernacula.Base"/> is that the CLI, the TTS tools
/// and the tests then find that same copy — one install, every surface. A regression here is
/// silent in the worst way: everything keeps working on a developer machine with FFmpeg on
/// PATH, and only the machines the download exists for are affected.
/// </para>
///
/// <para>
/// ⚠ THE FIXTURES USE A MADE-UP TOOL NAME ON PURPOSE. These tests set a process-wide
/// environment variable, and xunit runs classes in parallel. Pointing the variable at a
/// directory holding a fake "ffmpeg" would hand that fake to any decode running concurrently.
/// A name nothing else looks up exercises the same resolution order with nothing to break.
/// </para>
/// </summary>
public class FfmpegBinariesTests : IDisposable
{
    private const string ToolName = "vernacula-test-tool";

    private readonly string _dir = Directory.CreateTempSubdirectory("vernacula_ffbin_").FullName;
    private readonly string? _originalDir =
        Environment.GetEnvironmentVariable(FfmpegBinaries.DirectoryEnvironmentVariable);

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(FfmpegBinaries.DirectoryEnvironmentVariable, _originalDir);
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort */ }
        GC.SuppressFinalize(this);
    }

    private string CreateFakeTool()
    {
        string fileName = OperatingSystem.IsWindows() ? ToolName + ".exe" : ToolName;
        string path = Path.Combine(_dir, fileName);
        File.WriteAllText(path, "not a real executable");
        return path;
    }

    [Fact]
    public void EnvironmentVariable_WinsWhenItPointsAtTheBinary()
    {
        string expected = CreateFakeTool();
        Environment.SetEnvironmentVariable(FfmpegBinaries.DirectoryEnvironmentVariable, _dir);

        Assert.Equal(expected, FfmpegBinaries.ResolveExecutable(ToolName));
    }

    /// <summary>
    /// A variable pointing somewhere useless must not shadow a working install. Otherwise a
    /// stale value in a shell profile turns into "FFmpeg is not installed" on a machine where
    /// it plainly is.
    /// </summary>
    [Fact]
    public void EnvironmentVariable_IsIgnoredWhenTheBinaryIsNotThere()
    {
        Environment.SetEnvironmentVariable(FfmpegBinaries.DirectoryEnvironmentVariable, _dir);

        Assert.Equal(ToolName, FfmpegBinaries.ResolveExecutable(ToolName));
    }

    /// <summary>
    /// With nothing found anywhere, the bare name is returned so the spawn fails with the
    /// platform's own "not found" — which the callers turn into an actionable message.
    /// </summary>
    [Fact]
    public void UnknownTool_FallsBackToTheBareName()
    {
        Environment.SetEnvironmentVariable(FfmpegBinaries.DirectoryEnvironmentVariable, null);

        Assert.Equal(ToolName, FfmpegBinaries.ResolveExecutable(ToolName));
        Assert.False(FfmpegBinaries.IsAvailable(ToolName));
    }

    /// <summary>
    /// The managed directory sits under the app's own data directory, beside the model cache,
    /// so uninstalling is deleting one tree. Both the app and the CLI compute it here rather
    /// than agreeing on a string in two places.
    /// </summary>
    [Fact]
    public void ManagedDirectory_SitsUnderTheAppDataDirectory()
    {
        string expected = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Vernacula", "tools", "ffmpeg");

        Assert.Equal(expected, FfmpegBinaries.ManagedDirectory);
    }

    /// <summary>
    /// ⚠ THE MESSAGE IS THE FEATURE. What the desktop app used to surface when ffprobe was
    /// missing was "An error occurred trying to start process 'ffprobe' ... The system cannot
    /// find the file specified", which names neither the file being opened nor anything the
    /// user could do. Everything a reader needs has to be in here.
    /// </summary>
    [Fact]
    public void MissingBinaryMessage_SaysWhatFailedAndWhatToDo()
    {
        // ⚠ BUILT WITH Path.Combine, NOT WRITTEN AS A LITERAL. Path.GetFileName splits on the
        // running platform's separators only, so a hard-coded Windows path stays whole under
        // Linux and the "does not contain the directory" assertion fails there — on the test,
        // not on the code. The message itself is fine: its paths come from the real OS.
        string directory = Path.Combine(Path.GetTempPath(), "recordings");
        string message = FfmpegBinaries.MissingBinaryMessage(
            "ffprobe", Path.Combine(directory, "meeting.m4a"));

        Assert.Contains("ffprobe", message);          // which tool
        Assert.Contains("meeting.m4a", message);      // which file
        Assert.DoesNotContain(directory, message);    // but not the whole path
        Assert.Contains("FFmpeg", message);           // what to install
        Assert.Contains("docs/installation.md", message);
        Assert.Contains("MP3", message);              // and the way out that needs nothing
    }
}
