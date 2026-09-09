using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.App.Services;
using Vernacula.Base;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// What the CUDA check is allowed to depend on.
///
/// It once loaded the Parakeet preprocessor, which made "is CUDA working?" a question about the
/// models directory: a machine with a perfect CUDA install but no Parakeet bundle -- a fresh
/// install, a models directory pointed elsewhere, or any VibeVoice user, since that backend never
/// downloads Parakeet -- was told CUDA was unavailable. The Setup panel's CUDA and cuDNN rows,
/// which ask the hardware, said the opposite on the same screen, and the streaming backend was
/// labelled "(CPU - very slow)" on a working GPU.
/// </summary>
public class CudaCheckTests
{
    /// <summary>
    /// The embedded probe graph has to be a loadable model. It is base64 in a source file, where
    /// nothing else would notice it rotting -- and a blob that only failed at session creation
    /// would be indistinguishable from a machine with no CUDA, which is the one answer this check
    /// must never invent.
    /// </summary>
    [Fact]
    public void TheProbeGraphIsAModelOnnxRuntimeCanRun()
    {
        var field = typeof(ModelManagerService).GetField(
            "CudaProbeModelBase64", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(field);

        var model = Convert.FromBase64String((string)field!.GetRawConstantValue()!);

        var input = new DenseTensor<float>(new float[] { 1f }, new[] { 1, 1, 1, 1 });

        using var opts = new SessionOptions();
        using var session = new InferenceSession(model, opts);
        using var results = session.Run([NamedOnnxValue.CreateFromTensor("X", input)]);

        Assert.Equal(1f, results.First().AsTensor<float>().GetValue(0));
    }

    /// <summary>
    /// Same machine, two empty models directories: the verdict and the reason must be identical,
    /// and neither may name a model file. Asserting the verdict rather than just the message is
    /// what stops the dependency coming back in a different shape.
    /// </summary>
    [Fact]
    public void TheAnswerDoesNotDependOnWhatIsDownloaded()
    {
        using var first = new TempModelsDir();
        using var second = new TempModelsDir();

        var a = CheckIn(first.Path);
        var b = CheckIn(second.Path);

        Assert.Equal(a.Available, b.Available);
        Assert.Equal(a.Message, b.Message);

        Assert.DoesNotContain(Config.PreprocessorFile, a.Message, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(first.Path, a.Message, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(a.Message);
    }

    /// <summary>
    /// The registered directories have to end up in FRONT of whatever is already on PATH, not
    /// merely present on it.
    ///
    /// The NVIDIA installer commonly leaves both toolkits on PATH with the older one first. An
    /// implementation that skips a directory it finds already there writes nothing at all on that
    /// machine, and the loader goes on binding the CUDA 12 copy of every library whose name does
    /// not carry its major -- cudnn64_9.dll, curand64_10.dll -- which is the failure the whole
    /// registration exists to prevent, in its least visible form.
    /// </summary>
    [Fact]
    public void TheCudaDirectoriesGoToTheFrontOfPathEvenWhenAlreadyOnIt()
    {
        string? original = Environment.GetEnvironmentVariable("PATH");
        try
        {
            // Built from the platform's own separators: the method is not Windows-only even though
            // only Windows calls it, and CI runs this on Linux, where ';' and '\' are ordinary
            // filename characters rather than separators.
            string sep = Path.DirectorySeparatorChar.ToString();
            string cuda13 = Path.Combine("cuda13", "bin", "x64");
            string cudnn9 = Path.Combine("cudnn9", "bin");
            string cuda12 = Path.Combine("cuda12", "bin");
            string unrelated = Path.Combine("usr", "bin");

            string[] wanted = [cuda13, cudnn9];
            // The older toolkit first, the newer one already present (with a trailing separator, as
            // PATH entries are often written), and an unrelated entry that must survive.
            Environment.SetEnvironmentVariable(
                "PATH", string.Join(Path.PathSeparator, [cuda12, cuda13 + sep, unrelated]));

            PrependToProcessPath(wanted);
            var after = Environment.GetEnvironmentVariable("PATH")!.Split(Path.PathSeparator);

            Assert.Equal(wanted[0], after[0]);
            Assert.Equal(wanted[1], after[1]);
            Assert.Contains(cuda12, after);          // unrelated entries survive
            Assert.Contains(unrelated, after);
            Assert.DoesNotContain(cuda13 + sep, after); // and the stale copy is gone, not duplicated

            // A Re-check runs this again; it must not grow PATH or reorder anything a second time.
            string once = Environment.GetEnvironmentVariable("PATH")!;
            PrependToProcessPath(wanted);
            Assert.Equal(once, Environment.GetEnvironmentVariable("PATH"));
        }
        finally
        {
            Environment.SetEnvironmentVariable("PATH", original);
        }
    }

    private static void PrependToProcessPath(string[] directories)
    {
        var method = typeof(ModelManagerService).GetMethod(
            "PrependToProcessPath", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        method!.Invoke(null, [directories]);
    }

    /// <summary>
    /// ⚠ THE REAL CHECK, WITH ITS SIDE EFFECTS PUT BACK. Calling it for real is the point -- a test
    /// against a reimplementation would not have caught the dependency this file exists for -- but
    /// it writes the developer's own %LOCALAPPDATA%\Vernacula\cuda_debug.txt and edits the test
    /// process's PATH. Leaving either changed means a test run silently overwrites the diagnostic a
    /// user is about to attach to a bug report, with a failure from a CPU-only test host.
    /// </summary>
    private static ModelManagerService.CudaCheck CheckIn(string modelsDir)
    {
        string logPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Vernacula", "cuda_debug.txt");
        byte[]? log = File.Exists(logPath) ? File.ReadAllBytes(logPath) : null;
        string? path = Environment.GetEnvironmentVariable("PATH");

        try
        {
            var settings = new SettingsService();
            settings.Current.ModelsDir = modelsDir;
            return new ModelManagerService(settings).CheckCuda();
        }
        finally
        {
            Environment.SetEnvironmentVariable("PATH", path);
            try
            {
                if (log is not null) File.WriteAllBytes(logPath, log);
                else if (File.Exists(logPath)) File.Delete(logPath);
            }
            catch { /* best-effort: a restore failure must not fail the test it is cleaning up after */ }
        }
    }

    private sealed class TempModelsDir : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "vernacula-cuda-check", Guid.NewGuid().ToString("N"));
        public TempModelsDir() => Directory.CreateDirectory(Path);
        public void Dispose() { try { Directory.Delete(Path, true); } catch { } }
    }
}
