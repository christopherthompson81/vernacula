using System.Runtime.InteropServices;

namespace Parakeet.Base;

/// <summary>
/// NVML-based GPU memory query and CUDA / cuDNN installation detection.
/// All methods are safe to call when the relevant hardware or libraries are absent;
/// they return neutral values (0 / false) rather than throwing.
/// </summary>
public static class HardwareInfo
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    private struct MemoryStatusEx
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx lpBuffer);

    // ── NVML P/Invokes ────────────────────────────────────────────────────────

    [DllImport("nvml.dll", EntryPoint = "nvmlInit_v2")]
    private static extern int NvmlInitWindows();

    [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlInit_v2")]
    private static extern int NvmlInitLinux();

    [DllImport("nvml.dll", EntryPoint = "nvmlShutdown")]
    private static extern int NvmlShutdownWindows();

    [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlShutdown")]
    private static extern int NvmlShutdownLinux();

    [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetHandleByIndex_v2")]
    private static extern int NvmlDeviceGetHandleByIndexWindows(uint index, out IntPtr device);

    [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetHandleByIndex_v2")]
    private static extern int NvmlDeviceGetHandleByIndexLinux(uint index, out IntPtr device);

    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlMemory { public ulong total, free, used; }

    [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetMemoryInfo")]
    private static extern int NvmlDeviceGetMemoryInfoWindows(IntPtr device, out NvmlMemory memory);

    [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetMemoryInfo")]
    private static extern int NvmlDeviceGetMemoryInfoLinux(IntPtr device, out NvmlMemory memory);

    // ── GPU memory ────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns (TotalMb, FreeMb) for the specified GPU.
    /// Both are 0 if NVML is unavailable or the query fails.
    /// Call this after ONNX models are loaded so FreeMb reflects actual remaining space.
    /// </summary>
    public static (long TotalMb, long FreeMb) GetGpuMemoryMb(int gpuId = 0)
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return (0, 0);

        try
        {
            if (NvmlInitPlatform() != 0) return (0, 0);
            try
            {
                if (NvmlDeviceGetHandleByIndexPlatform((uint)gpuId, out var device) != 0) return (0, 0);
                if (NvmlDeviceGetMemoryInfoPlatform(device, out var mem) != 0) return (0, 0);
                return ((long)(mem.total / (1024UL * 1024UL)),
                        (long)(mem.free  / (1024UL * 1024UL)));
            }
            finally { NvmlShutdownPlatform(); }
        }
        catch { return (0, 0); }
    }

    /// <summary>
    /// Returns approximate total physical system memory in megabytes.
    /// Returns 0 when the platform query is unavailable or fails.
    /// </summary>
    public static long GetTotalSystemMemoryMb()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                var status = new MemoryStatusEx
                {
                    dwLength = (uint)Marshal.SizeOf<MemoryStatusEx>()
                };

                if (GlobalMemoryStatusEx(ref status))
                    return (long)(status.ullTotalPhys / (1024UL * 1024UL));
            }

            if (OperatingSystem.IsLinux())
            {
                const string memInfoPath = "/proc/meminfo";
                if (!File.Exists(memInfoPath))
                    return 0;

                foreach (string line in File.ReadLines(memInfoPath))
                {
                    if (!line.StartsWith("MemTotal:", StringComparison.Ordinal))
                        continue;

                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2 || !long.TryParse(parts[1], out long kb))
                        return 0;

                    return kb / 1024;
                }
            }
        }
        catch
        {
        }

        return 0;
    }

    // ── CUDA Toolkit ──────────────────────────────────────────────────────────

    /// <summary>
    /// True if the platform's CUDA runtime library can be found.
    /// </summary>
    public static bool IsCudaToolkitInstalled()
    {
        if (OperatingSystem.IsWindows())
        {
            string? cudaPath = GetCudaToolkitPath();
            if (string.IsNullOrEmpty(cudaPath)) return false;

            var binDir = Path.Combine(cudaPath, "bin");
            return Directory.Exists(binDir)
                && Directory.GetFiles(binDir, "cudart64_*.dll").Length > 0;
        }

        if (OperatingSystem.IsLinux())
        {
            foreach (var dir in GetLinuxCudaLibraryDirs())
            {
                if (HasFile(dir, "libcudart.so") || HasFile(dir, "libcudart.so.*"))
                    return true;
            }
        }

        return false;
    }

    /// <summary>Returns the configured CUDA toolkit root for the current platform, or null if none is known.</summary>
    public static string? GetCudaToolkitPath()
    {
        if (OperatingSystem.IsWindows())
            return Environment.GetEnvironmentVariable("CUDA_PATH");

        if (OperatingSystem.IsLinux())
        {
            string? envPath = Environment.GetEnvironmentVariable("CUDA_PATH")
                           ?? Environment.GetEnvironmentVariable("CUDA_HOME");

            if (!string.IsNullOrWhiteSpace(envPath))
                return envPath;

            const string defaultCudaPath = "/usr/local/cuda";
            if (Directory.Exists(defaultCudaPath))
                return defaultCudaPath;
        }

        return null;
    }

    // ── cuDNN ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// True if the platform's cuDNN runtime library can be found.
    /// </summary>
    public static bool IsCudnnInstalled()
    {
        if (OperatingSystem.IsWindows())
        {
            string? cudaPath = GetCudaToolkitPath();
            if (!string.IsNullOrEmpty(cudaPath))
            {
                var binDir = Path.Combine(cudaPath, "bin");
                if (Directory.Exists(binDir) && HasWindowsCudnnDll(binDir))
                    return true;
            }

            string? pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (pathEnv is null) return false;
            foreach (var entry in pathEnv.Split(Path.PathSeparator))
            {
                if (string.IsNullOrWhiteSpace(entry)) continue;
                try
                {
                    if (Directory.Exists(entry) && HasWindowsCudnnDll(entry))
                        return true;
                }
                catch { }
            }

            return false;
        }

        if (OperatingSystem.IsLinux())
        {
            foreach (var dir in GetLinuxCudaLibraryDirs())
            {
                if (HasFile(dir, "libcudnn.so") || HasFile(dir, "libcudnn.so.*"))
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// True when the current machine appears capable of initializing CUDA execution:
    /// the CUDA runtime is installed, cuDNN is present, and at least one NVIDIA GPU
    /// is visible through NVML.
    /// </summary>
    public static bool CanProbeCudaExecutionProvider()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return false;

        if (!IsCudaToolkitInstalled() || !IsCudnnInstalled())
            return false;

        var (totalMb, _) = GetGpuMemoryMb();
        return totalMb > 0;
    }

    private static int NvmlInitPlatform() =>
        OperatingSystem.IsWindows() ? NvmlInitWindows() : NvmlInitLinux();

    private static int NvmlShutdownPlatform() =>
        OperatingSystem.IsWindows() ? NvmlShutdownWindows() : NvmlShutdownLinux();

    private static int NvmlDeviceGetHandleByIndexPlatform(uint index, out IntPtr device) =>
        OperatingSystem.IsWindows()
            ? NvmlDeviceGetHandleByIndexWindows(index, out device)
            : NvmlDeviceGetHandleByIndexLinux(index, out device);

    private static int NvmlDeviceGetMemoryInfoPlatform(IntPtr device, out NvmlMemory memory) =>
        OperatingSystem.IsWindows()
            ? NvmlDeviceGetMemoryInfoWindows(device, out memory)
            : NvmlDeviceGetMemoryInfoLinux(device, out memory);

    private static IEnumerable<string> GetLinuxCudaLibraryDirs()
    {
        var dirs = new HashSet<string>(StringComparer.Ordinal);

        string? cudaPath = GetCudaToolkitPath();
        if (!string.IsNullOrWhiteSpace(cudaPath))
        {
            dirs.Add(Path.Combine(cudaPath, "lib64"));
            dirs.Add(Path.Combine(cudaPath, "lib"));
            dirs.Add(Path.Combine(cudaPath, "targets", "x86_64-linux", "lib"));
        }

        string? ldLibraryPath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH");
        if (!string.IsNullOrWhiteSpace(ldLibraryPath))
        {
            foreach (var entry in ldLibraryPath.Split(Path.PathSeparator))
            {
                if (!string.IsNullOrWhiteSpace(entry))
                    dirs.Add(entry.Trim());
            }
        }

        dirs.Add("/usr/local/cuda/lib64");
        dirs.Add("/usr/local/cuda/lib");
        dirs.Add("/usr/lib/x86_64-linux-gnu");
        dirs.Add("/usr/lib/wsl/lib");
        dirs.Add("/usr/lib64");

        return dirs.Where(Directory.Exists);
    }

    private static bool HasWindowsCudnnDll(string dir) =>
        Directory.GetFiles(dir, "cudnn64_*.dll").Length > 0 ||
        Directory.GetFiles(dir, "cudnn_*.dll").Length > 0;

    private static bool HasFile(string dir, string pattern)
    {
        try
        {
            return Directory.Exists(dir) && Directory.GetFiles(dir, pattern).Length > 0;
        }
        catch
        {
            return false;
        }
    }
}
