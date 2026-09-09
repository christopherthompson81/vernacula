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

    private static ModelManagerService.CudaCheck CheckIn(string modelsDir)
    {
        var settings = new SettingsService();
        settings.Current.ModelsDir = modelsDir;
        return new ModelManagerService(settings).CheckCuda();
    }

    private sealed class TempModelsDir : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "vernacula-cuda-check", Guid.NewGuid().ToString("N"));
        public TempModelsDir() => Directory.CreateDirectory(Path);
        public void Dispose() { try { Directory.Delete(Path, true); } catch { } }
    }
}
