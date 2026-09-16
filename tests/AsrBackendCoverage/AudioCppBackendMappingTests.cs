using Vernacula.App.Services;
using Vernacula.Base.Models;
using Xunit;

namespace Vernacula.Tests.AsrBackendCoverage;

/// <summary>
/// The app's execution provider → audio.cpp's backend names. Both audio.cpp backends read this,
/// so a change here moves recognition and synthesis together, which is the point of it existing.
/// </summary>
public class AudioCppBackendMappingTests
{
    /// <summary>
    /// The regression the mapping's own comment describes: Auto is the out-of-the-box state, and
    /// an earlier version treated everything that was not explicitly Cuda as CPU — so a default
    /// install ran audio.cpp on the CPU while every ONNX backend took CUDA from the same
    /// setting. It worked, and looked merely slow, which is why it survived.
    /// </summary>
    [Fact]
    public void AutoTriesTheGpuBackendsBeforeFallingBackToCpu()
    {
        var auto = AudioCppBackends.For(ExecutionProvider.Auto);
        Assert.Contains("cuda", auto);
        Assert.Equal("cpu", auto[^1]);
        Assert.NotEqual(0, auto.AsSpan().IndexOf("cpu"));   // not first: it is the fallback
    }

    [Theory]
    [InlineData(ExecutionProvider.Cuda,   "cuda")]
    [InlineData(ExecutionProvider.CoreML, "metal")]
    [InlineData(ExecutionProvider.WebGpu, "vulkan")]
    public void AGpuProviderIsTriedFirstAndCpuCatchesABuildWithoutIt(ExecutionProvider ep, string first)
    {
        var backends = AudioCppBackends.For(ep);
        Assert.Equal(first, backends[0]);
        // Always a list, never a single name: whether a backend is registered depends on how the
        // engine was BUILT, and the only way to find out is to try it.
        Assert.Equal("cpu", backends[^1]);
    }

    [Fact]
    public void CpuAsksForNothingElse()
    {
        Assert.Equal(["cpu"], AudioCppBackends.For(ExecutionProvider.Cpu));
    }
}
