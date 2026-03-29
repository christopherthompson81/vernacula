using System.Runtime.InteropServices;

namespace Parakeet.Base;

/// <summary>
/// NVML-based GPU memory query and CUDA / cuDNN installation detection.
/// All methods are safe to call when the relevant hardware or libraries are absent;
/// they return neutral values (0 / false) rather than throwing.
/// </summary>
public static class HardwareInfo
{
    // ── NVML P/Invokes ────────────────────────────────────────────────────────

    [DllImport("nvml.dll", EntryPoint = "nvmlInit_v2")]
    private static extern int NvmlInit();

    [DllImport("nvml.dll", EntryPoint = "nvmlShutdown")]
    private static extern int NvmlShutdown();

    [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetHandleByIndex_v2")]
    private static extern int NvmlDeviceGetHandleByIndex(uint index, out IntPtr device);

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemory { public ulong total, free, used; }

    [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetMemoryInfo")]
    private static extern int NvmlDeviceGetMemoryInfo(IntPtr device, out NvmlMemory memory);

    // ── GPU memory ────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns (TotalMb, FreeMb) for the specified GPU.
    /// Both are 0 if NVML is unavailable or the query fails.
    /// Call this after ONNX models are loaded so FreeMb reflects actual remaining space.
    /// </summary>
    public static (long TotalMb, long FreeMb) GetGpuMemoryMb(int gpuId = 0)
    {
        try
        {
            if (NvmlInit() != 0) return (0, 0);
            try
            {
                if (NvmlDeviceGetHandleByIndex((uint)gpuId, out var device) != 0) return (0, 0);
                if (NvmlDeviceGetMemoryInfo(device, out var mem) != 0) return (0, 0);
                return ((long)(mem.total / (1024UL * 1024UL)),
                        (long)(mem.free  / (1024UL * 1024UL)));
            }
            finally { NvmlShutdown(); }
        }
        catch { return (0, 0); }
    }

    // ── CUDA Toolkit ──────────────────────────────────────────────────────────

    /// <summary>
    /// True if CUDA_PATH is set and cudart64_*.dll exists in its bin directory.
    /// </summary>
    public static bool IsCudaToolkitInstalled()
    {
        string? cudaPath = GetCudaToolkitPath();
        if (string.IsNullOrEmpty(cudaPath)) return false;
        var binDir = Path.Combine(cudaPath, "bin");
        return Directory.Exists(binDir)
            && Directory.GetFiles(binDir, "cudart64_*.dll").Length > 0;
    }

    /// <summary>Returns the CUDA_PATH environment variable, or null if not set.</summary>
    public static string? GetCudaToolkitPath() =>
        Environment.GetEnvironmentVariable("CUDA_PATH");

    // ── cuDNN ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// True if cudnn64_*.dll or cudnn_*.dll is found in CUDA_PATH\bin or any PATH entry.
    /// </summary>
    public static bool IsCudnnInstalled()
    {
        string? cudaPath = GetCudaToolkitPath();
        if (!string.IsNullOrEmpty(cudaPath))
        {
            var binDir = Path.Combine(cudaPath, "bin");
            if (Directory.Exists(binDir) && HasCudnnDll(binDir))
                return true;
        }

        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv is null) return false;
        foreach (var entry in pathEnv.Split(';'))
        {
            if (string.IsNullOrWhiteSpace(entry)) continue;
            try
            {
                if (Directory.Exists(entry) && HasCudnnDll(entry))
                    return true;
            }
            catch { }
        }
        return false;
    }

    private static bool HasCudnnDll(string dir) =>
        Directory.GetFiles(dir, "cudnn64_*.dll").Length > 0 ||
        Directory.GetFiles(dir, "cudnn_*.dll").Length > 0;
}
