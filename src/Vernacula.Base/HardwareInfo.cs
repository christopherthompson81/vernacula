using System.Runtime.InteropServices;

namespace Vernacula.Base;

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

    [DllImport("nvml.dll", EntryPoint = "nvmlDeviceGetCudaComputeCapability")]
    private static extern int NvmlDeviceGetCudaComputeCapabilityWindows(IntPtr device, out int major, out int minor);

    [DllImport("libnvidia-ml.so.1", EntryPoint = "nvmlDeviceGetCudaComputeCapability")]
    private static extern int NvmlDeviceGetCudaComputeCapabilityLinux(IntPtr device, out int major, out int minor);

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
            // Search recursively: CUDA 13 relocated the Windows runtime DLLs out of
            // <CUDA_PATH>\bin into a bin\x64 subfolder, so a flat glob on bin\ misses them.
            return GetWindowsCudaSearchRoots()
                .Any(dir => AnyFileMatches(dir, "cudart64_*.dll"));
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
            // cuDNN 9 installs to its own tree (C:\Program Files\NVIDIA\CUDNN\vX.Y\bin\<cuda>\),
            // not into the CUDA Toolkit bin and not on PATH by default — so search recursively
            // across both the Toolkit and the standalone cuDNN install roots.
            return GetWindowsCudaSearchRoots()
                .Any(dir => AnyFileMatches(dir, "cudnn64_*.dll")
                         || AnyFileMatches(dir, "cudnn_*.dll"));
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

    /// <summary>
    /// Returns (major, minor) CUDA compute capability of <paramref name="gpuId"/>,
    /// or (0, 0) if NVML is unavailable or the query fails. Compute capability ≥ 8.0
    /// indicates an Ampere-class or newer architecture with hardware BF16 tensor
    /// cores; older parts (Volta/Turing 7.x, Pascal 6.x) lack accelerated BF16 and
    /// will run BF16 ops via slower fallback paths.
    /// </summary>
    public static (int Major, int Minor) GetCudaComputeCapability(int gpuId = 0)
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return (0, 0);

        try
        {
            if (NvmlInitPlatform() != 0) return (0, 0);
            try
            {
                if (NvmlDeviceGetHandleByIndexPlatform((uint)gpuId, out var device) != 0) return (0, 0);
                int rc = OperatingSystem.IsWindows()
                    ? NvmlDeviceGetCudaComputeCapabilityWindows(device, out int major, out int minor)
                    : NvmlDeviceGetCudaComputeCapabilityLinux(device, out major, out minor);
                if (rc != 0) return (0, 0);
                return (major, minor);
            }
            finally { NvmlShutdownPlatform(); }
        }
        catch { return (0, 0); }
    }

    /// <summary>
    /// True if the system has CUDA available AND a GPU with hardware-accelerated
    /// BF16 (Ampere or newer, compute capability ≥ 8.0). Used to gate selection of
    /// BF16 ONNX bundles like Granite Speech 4.1: on hardware that lacks BF16
    /// tensor cores the BF16 kernels fall back to slower paths or are unavailable
    /// (the LM decoder ops, the encoder Conv, and ORT's CPU EP <c>Where(BF16)</c>
    /// all have known coverage gaps), so the FP32 bundle is the safe default.
    ///
    /// Result is cached after the first call: the answer is fixed for the
    /// lifetime of the process (GPUs don't hot-swap), and the underlying
    /// NVML init/get/shutdown cycle is non-trivial when called repeatedly
    /// from settings/UI flows.
    /// </summary>
    public static bool SupportsBf16Acceleration() => _bf16Cache.Value;

    private static readonly Lazy<bool> _bf16Cache = new(() =>
    {
        if (!CanProbeCudaExecutionProvider()) return false;
        var (major, _) = GetCudaComputeCapability();
        return major >= 8;
    });

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

    /// <summary>
    /// Windows: every directory under which CUDA Toolkit or cuDNN runtime DLLs may live.
    /// Each returned directory is meant to be searched <em>recursively</em>, because the
    /// runtime DLLs no longer sit directly in a flat bin\ folder:
    /// <list type="bullet">
    ///   <item>CUDA 13 moved the Toolkit runtime DLLs into a bin\x64 subfolder.</item>
    ///   <item>cuDNN 9 installs its DLLs under bin\&lt;cuda-major.minor&gt;\.</item>
    /// </list>
    /// Covers CUDA_PATH, any versioned CUDA_PATH_V* (e.g. CUDA_PATH_V13_3), the default
    /// Toolkit install root (in case no env var is set), the standalone cuDNN install
    /// tree, and any CUDA/cuDNN directories already on PATH.
    /// </summary>
    private static IEnumerable<string> GetWindowsCudaSearchRoots()
    {
        var roots = new List<string>();

        void AddToolkitBin(string? toolkitRoot)
        {
            if (!string.IsNullOrWhiteSpace(toolkitRoot))
                roots.Add(Path.Combine(toolkitRoot, "bin"));
        }

        // CUDA Toolkit roots from the environment.
        AddToolkitBin(Environment.GetEnvironmentVariable("CUDA_PATH"));
        try
        {
            foreach (System.Collections.DictionaryEntry e in Environment.GetEnvironmentVariables())
            {
                if (e.Key is string key
                    && key.StartsWith("CUDA_PATH_V", StringComparison.OrdinalIgnoreCase))
                    AddToolkitBin(e.Value as string);
            }
        }
        catch { }

        string programFiles = Environment.GetEnvironmentVariable("ProgramFiles")
                              ?? @"C:\Program Files";

        // Default Toolkit install location, in case CUDA_PATH is unset.
        foreach (var root in SafeGetDirectories(
                     Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA"), "v*"))
            AddToolkitBin(root);

        // Standalone cuDNN install tree: C:\Program Files\NVIDIA\CUDNN\vX.Y\bin\<cuda>\.
        foreach (var root in SafeGetDirectories(
                     Path.Combine(programFiles, "NVIDIA", "CUDNN"), "v*"))
            roots.Add(Path.Combine(root, "bin"));

        // PATH entries the user may have added manually for CUDA or cuDNN.
        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv != null)
        {
            foreach (var entry in pathEnv.Split(Path.PathSeparator))
            {
                if (!string.IsNullOrWhiteSpace(entry) &&
                    (entry.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
                     entry.Contains("CUDNN", StringComparison.OrdinalIgnoreCase)))
                    roots.Add(entry.Trim());
            }
        }

        return roots.Where(Directory.Exists).Distinct(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Windows: the concrete directories that actually contain a CUDA or cuDNN runtime DLL.
    /// Unlike <see cref="GetWindowsCudaSearchRoots"/> (search roots to scan recursively),
    /// these are the leaf directories to register with AddDllDirectory so an MSIX-packaged
    /// onnxruntime can locate cudart / cudnn at load time.
    /// </summary>
    public static IReadOnlyCollection<string> GetWindowsCudaDllDirectories()
    {
        var result = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!OperatingSystem.IsWindows())
            return result;

        foreach (var root in GetWindowsCudaSearchRoots())
        {
            foreach (var pattern in new[] { "cudart64_*.dll", "cudnn64_*.dll", "cudnn_*.dll" })
            {
                try
                {
                    foreach (var file in Directory.EnumerateFiles(root, pattern, SearchOption.AllDirectories))
                    {
                        var dir = Path.GetDirectoryName(file);
                        if (!string.IsNullOrEmpty(dir))
                            result.Add(dir);
                    }
                }
                catch { }
            }
        }

        return result;
    }

    private static IEnumerable<string> SafeGetDirectories(string parent, string pattern)
    {
        try
        {
            return Directory.Exists(parent)
                ? Directory.GetDirectories(parent, pattern)
                : Array.Empty<string>();
        }
        catch { return Array.Empty<string>(); }
    }

    /// <summary>True if <paramref name="dir"/> or any subdirectory contains a file matching <paramref name="pattern"/>.</summary>
    private static bool AnyFileMatches(string dir, string pattern)
    {
        try
        {
            return Directory.Exists(dir)
                && Directory.EnumerateFiles(dir, pattern, SearchOption.AllDirectories).Any();
        }
        catch { return false; }
    }

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
