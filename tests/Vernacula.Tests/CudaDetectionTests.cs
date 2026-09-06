using Xunit;
using Vernacula.Base;

namespace Vernacula.Tests;

/// <summary>
/// The path rules behind Windows CUDA detection. They exist because two cuDNN layouts are both
/// legitimate and a rule that suits one breaks the other — which is exactly how earlier attempts
/// at this failed, each time on the layout the previous fix had not considered.
/// </summary>
public class CudaDetectionTests
{
    private static string P(params string[] parts) => string.Join(System.IO.Path.DirectorySeparatorChar, parts);

    [Fact]
    public void StandaloneCudnnTreeNamesTheCudaVersionInItsPath()
    {
        // C:\Program Files\NVIDIA\CUDNN\v9.10\bin\13.0\ — holds cudnn64_9.dll and nothing else, so
        // a rule that wants a cudart beside it would drop the whole install.
        Assert.True(HardwareInfo.NamesRequiredMajor(P("C:", "Program Files", "NVIDIA", "CUDNN", "v9.10", "bin", "13.0")));
        Assert.True(HardwareInfo.NamesRequiredMajor(P("opt", "cudnn", "13")));
    }

    [Fact]
    public void ACudnnTreeForAnotherCudaIsNotOurs()
    {
        Assert.False(HardwareInfo.NamesRequiredMajor(P("C:", "Program Files", "NVIDIA", "CUDNN", "v9.10", "bin", "12.8")));
        Assert.False(HardwareInfo.NamesRequiredMajor(P("C:", "Program Files", "NVIDIA", "CUDNN", "v9.10", "bin")));
    }

    [Fact]
    public void ToolkitBinAndItsX64SubdirectoryAreTheSameInstall()
    {
        // The other legitimate layout: cuDNN copied into <toolkit>\bin, while CUDA 13 keeps its
        // runtime DLLs in <toolkit>\bin\x64. A rule requiring the same directory drops this one.
        var toolkit = P("C:", "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v13.0");
        Assert.Equal(toolkit, HardwareInfo.ToolkitRootOf(P(toolkit, "bin")));
        Assert.Equal(toolkit, HardwareInfo.ToolkitRootOf(P(toolkit, "bin", "x64")));
    }

    [Fact]
    public void SeparateToolkitsDoNotShareARoot()
    {
        var v13 = P("C:", "CUDA", "v13.0", "bin", "x64");
        var v12 = P("C:", "CUDA", "v12.8", "bin");
        Assert.NotEqual(HardwareInfo.ToolkitRootOf(v13), HardwareInfo.ToolkitRootOf(v12));
    }

    [Fact]
    public void ADirectoryWithNoBinSegmentIsItsOwnRoot()
    {
        var loose = P("C:", "some", "place");
        Assert.Equal(loose, HardwareInfo.ToolkitRootOf(loose));
    }

    [Fact]
    public void TheProbeAnswersAndSaysWhyWhenItSaysNo()
    {
        // Whatever this machine has, the invariant holds: a false answer carries a reason, and a
        // true one does not need one. This is what a settings window shows the user.
        HardwareInfo.InvalidateCudaProbes();
        var runtime = HardwareInfo.IsCudaToolkitInstalled();
        var cudnn = HardwareInfo.IsCudnnInstalled();

        Assert.Equal(runtime, HardwareInfo.CudaRuntimeNote is null);
        Assert.Equal(cudnn, HardwareInfo.CudnnNote is null);
        Assert.Contains($"CUDA {HardwareInfo.RequiredCudaMajor}", HardwareInfo.CudaUnavailableMessage());
    }

    [Fact]
    public void InvalidatingMakesTheNextAnswerFresh()
    {
        var before = HardwareInfo.IsCudaToolkitInstalled();
        HardwareInfo.InvalidateCudaProbes();
        Assert.Equal(before, HardwareInfo.IsCudaToolkitInstalled());   // same machine, same answer
    }
}
