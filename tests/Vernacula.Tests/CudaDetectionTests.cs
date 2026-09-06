using Xunit;
using Vernacula.Base;

namespace Vernacula.Tests;

/// <summary>
/// What CUDA detection promises: an answer about the major this build actually links, and a reason
/// whenever the answer is no.
/// </summary>
public class CudaDetectionTests
{
    [Fact]
    public void TheRequiredMajorsAreWhatOnnxRuntime129Links()
    {
        // The provider names these in its own dependencies: libcudart.so.13 and libcudnn.so.9.
        // If a future runtime moves either, these constants are the one place to change.
        Assert.Equal(13, HardwareInfo.RequiredCudaMajor);
        Assert.Equal(9, HardwareInfo.RequiredCudnnMajor);
    }

    [Fact]
    public void TheProbeAnswersAndSaysWhyWhenItSaysNo()
    {
        // Only Windows and Linux are probed at all; elsewhere there is nothing to explain.
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            Assert.Skip("CUDA detection only applies to Windows and Linux.");

        HardwareInfo.InvalidateCudaProbes();
        var runtime = HardwareInfo.IsCudaToolkitInstalled();
        var cudnn = HardwareInfo.IsCudnnInstalled();

        // A "no" always carries a reason a user can act on; a "yes" needs none.
        Assert.Equal(runtime, HardwareInfo.CudaRuntimeNote is null);
        Assert.Equal(cudnn, HardwareInfo.CudnnNote is null);
    }

    [Fact]
    public void TheMessageLeadsWithWhateverTheProbeFound()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            Assert.Skip("CUDA detection only applies to Windows and Linux.");

        HardwareInfo.InvalidateCudaProbes();
        var message = HardwareInfo.CudaUnavailableMessage();
        var note = HardwareInfo.CudaProbeNote;

        if (note is not null)
        {
            Assert.Contains(note, message);
        }
        else
        {
            // Both halves checked out, so the major version is not the likely cause and must not be
            // blamed: the message should point at the driver, the GPU, or the build instead.
            Assert.Contains("driver", message);
            Assert.DoesNotContain("cannot load it", message);   // the major must not be blamed here
        }
    }

    [Fact]
    public void ADownloadIsOnlyOfferedForSomethingActuallyMissing()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            Assert.Skip("CUDA detection only applies to Windows and Linux.");

        HardwareInfo.InvalidateCudaProbes();

        // "Present" is the weaker claim, so a usable install is necessarily present. The gap
        // between them — present but unusable — is where a download link would contradict the note
        // telling the user the library is already there.
        if (HardwareInfo.IsCudaToolkitInstalled()) Assert.True(HardwareInfo.IsCudaRuntimePresent);
        if (HardwareInfo.IsCudnnInstalled()) Assert.True(HardwareInfo.IsCudnnPresent);
    }

    [Fact]
    public void InvalidatingReallyThrowsTheAnswerAway()
    {
        // Asserting only that the answer is stable would pass for a cache that never refreshes at
        // all, which is the bug this exists to prevent: Re-check has to re-check.
        var before = HardwareInfo.IsCudaToolkitInstalled();
        var generation = HardwareInfo.ProbeGeneration;

        HardwareInfo.InvalidateCudaProbes();

        Assert.True(HardwareInfo.ProbeGeneration > generation, "invalidation must discard the cached probe");
        Assert.Equal(before, HardwareInfo.IsCudaToolkitInstalled());   // same machine, same answer
    }
}
