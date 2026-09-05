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
            return _windowsCudaScan.Value.HasCudart;

        if (OperatingSystem.IsLinux())
            return _linuxCudaRuntimeLoadable.Value;

        return false;
    }

    /// <summary>
    /// Whether the dynamic loader can open the CUDA runtime this build needs.
    ///
    /// ⚠ ASK THE LOADER, NOT THE FILESYSTEM. The CUDA provider dlopens libcudart by soname, so the
    /// only question that matters is whether the loader can find it -- which is not the same as the
    /// file existing somewhere. A side-by-side CUDA 13 that is not in the ldconfig cache or on
    /// LD_LIBRARY_PATH is a file the provider cannot open, and answering "installed" for it puts us
    /// back to trying CUDA and silently landing on the CPU.
    ///
    /// ⚠ CACHED, like the Windows scan beside it. <see cref="CanProbeCudaExecutionProvider"/> is
    /// called at every model-init site, and this does a dlopen plus a directory sweep; without the
    /// cache a diarization run would repeat both dozens of times, and print the diagnostic each time.
    /// </summary>
    private static readonly Lazy<bool> _linuxCudaRuntimeLoadable = new(() =>
    {
        if (NativeLibrary.TryLoad($"libcudart.so.{RequiredCudaMajor}", out var handle))
        {
            NativeLibrary.Free(handle);
            return true;
        }

        // Present but unreachable. Still false -- the provider will fail exactly as we just did --
        // but worth saying out loud, because the fix is one ldconfig line rather than an install.
        foreach (var dir in GetLinuxCudaLibraryDirs())
        {
            foreach (var file in SafeGetFiles(dir, $"libcudart.so.{RequiredCudaMajor}*"))
            {
                if (!NativeLibrary.TryLoad(file, out var byPath)) continue;
                NativeLibrary.Free(byPath);
                Note($"CUDA {RequiredCudaMajor} was found at {file} but is not on the loader path, "
                     + "so the CUDA execution provider cannot load it. Add its directory to "
                     + "/etc/ld.so.conf.d (then run ldconfig) or to LD_LIBRARY_PATH.");
                return false;
            }

            if (SafeGetFiles(dir, $"libcudart.so.{RequiredCudaMajor}*").Any())
                Note($"CUDA {RequiredCudaMajor} was found in {dir} but could not be loaded even by "
                     + "full path, which usually means one of its own dependencies is missing.");
        }

        if (CudaProbeNote is null)
            Note($"No CUDA {RequiredCudaMajor} runtime found. This build links CUDA "
                 + $"{RequiredCudaMajor}; an older CUDA cannot load it.");
        return false;
    });

    /// <summary>
    /// Why CUDA was not available, when the probe worked it out — a library present but off the
    /// loader path, say. Null until <see cref="IsCudaToolkitInstalled"/> has run, and only ever set
    /// on Linux; <see cref="CudaUnavailableMessage"/> folds it into the error a caller throws, so
    /// the reason reaches a user who has no console to read stderr from.
    /// </summary>
    public static string? CudaProbeNote { get; private set; }

    /// <summary>
    /// The explanation to give when the CUDA execution provider will not start. One place, because
    /// four call sites threw their own version of it and only one of them said anything useful.
    /// </summary>
    public static string CudaUnavailableMessage() =>
        "Could not initialise the CUDA execution provider. This build links CUDA "
        + $"{RequiredCudaMajor}, and an older CUDA runtime cannot load it (the major version is part "
        + "of the library name), so that is the first thing to check; a missing driver, no visible "
        + "GPU, or a CPU-only build of ONNX Runtime reports the same failure."
        + (CudaProbeNote is null ? "" : $" Probe found: {CudaProbeNote}");

    private static void Note(string message)
    {
        CudaProbeNote = message;
        Console.Error.WriteLine($"[HardwareInfo] {message}");
    }

    private static IEnumerable<string> SafeGetFiles(string dir, string pattern)
    {
        try { return Directory.Exists(dir) ? Directory.GetFiles(dir, pattern) : Array.Empty<string>(); }
        catch { return Array.Empty<string>(); }
    }

    /// <summary>
    /// The CUDA major version the bundled ONNX Runtime links against.
    ///
    /// ⚠ THE MAJOR IS PART OF THE ANSWER, NOT A DETAIL. ONNX Runtime moved from CUDA 12 to CUDA 13
    /// at 1.27, and the two are not interchangeable: the provider library names its dependencies
    /// (libcudart.so.13, libcublas.so.13) with the major in the soname, so a machine with only
    /// CUDA 12 cannot load it. Reporting "CUDA is installed" from a bare libcudart.so of ANY major
    /// meant <see cref="ExecutionProvider.Auto"/> tried CUDA, failed to load the provider, silently
    /// fell back to CPU, and left the user wondering why their GPU was idle.
    /// </summary>
    public const int RequiredCudaMajor = 13;

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
            return _windowsCudaScan.Value.HasCudnn;

        if (OperatingSystem.IsLinux())
            return _linuxCudnnLoadable.Value;

        return false;
    }

    /// <summary>
    /// Whether the loader can open cuDNN, asked the same way as the CUDA runtime.
    ///
    /// ⚠ A FILESYSTEM MATCH WAS NOT ENOUGH, AND WAS ACTIVELY MISLEADING. The directories searched
    /// include the toolkit that /usr/local/cuda and CUDA_PATH point at, which on a side-by-side
    /// install is usually the OLDER one -- so a cuDNN sitting in the CUDA 12 tree answered for a
    /// runtime that links CUDA 13, and the provider then failed to load it with the failure
    /// swallowed. Asking the loader at least means the answer describes a library that can actually
    /// be opened.
    ///
    /// It still cannot tell which CUDA major a given cuDNN was built against -- the soname is
    /// libcudnn.so.9 either way -- so a wrong-major cuDNN that is on the loader path will still get
    /// past this and fail at provider init.
    /// </summary>
    private static readonly Lazy<bool> _linuxCudnnLoadable = new(() =>
    {
        foreach (var soname in new[] { "libcudnn.so.9", "libcudnn.so" })
        {
            if (!NativeLibrary.TryLoad(soname, out var handle)) continue;
            NativeLibrary.Free(handle);
            return true;
        }
        return false;
    });

    /// <summary>
    /// True when the current machine appears capable of initializing CUDA execution:
    /// the CUDA runtime is installed, cuDNN is present, and at least one NVIDIA GPU
    /// is visible through NVML.
    /// </summary>
    /// <remarks>
    /// cuDNN is still required, even though ONNX Runtime 1.29 loads it lazily rather than linking
    /// it: the models here run convolutions in the codec decoder, which is exactly what the CUDA
    /// provider needs cuDNN for. A machine with CUDA but no cuDNN would load the provider and then
    /// fail on the first session, which is a worse place to find out.
    /// </remarks>
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

        // ⚠ VERSIONED ROOTS TOO, NOT JUST THE `cuda` SYMLINK. Installing CUDA 13 beside an existing
        // CUDA 12 leaves /usr/local/cuda (and often CUDA_PATH) pointing at the older toolkit, so a
        // machine that DOES meet the requirement would look like it does not, and fall to the CPU
        // for the same invisible reason the major check exists to prevent.
        foreach (var root in SafeGetDirectories("/usr/local", $"cuda-{RequiredCudaMajor}*"))
        {
            dirs.Add(Path.Combine(root, "lib64"));
            dirs.Add(Path.Combine(root, "lib"));
            dirs.Add(Path.Combine(root, "targets", "x86_64-linux", "lib"));
        }
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
    /// Windows: the concrete directories that actually contain a CUDA or cuDNN runtime DLL
    /// (cudart, cublas, cufft, nvrtc, cudnn, …). Unlike <see cref="GetWindowsCudaSearchRoots"/>
    /// (roots to scan), these are the leaf directories to register with AddDllDirectory so an
    /// MSIX-packaged onnxruntime can locate the CUDA EP's dependencies at load time.
    /// </summary>
    public static IReadOnlyCollection<string> GetWindowsCudaDllDirectories() =>
        _windowsCudaScan.Value.DllDirectories;

    /// <summary>
    /// Result of a single recursive scan of the Windows CUDA/cuDNN search roots:
    /// whether the Toolkit runtime (cudart) and cuDNN are present, and the leaf
    /// directories holding the runtime DLLs the CUDA execution provider needs.
    /// </summary>
    private sealed record WindowsCudaScan(
        bool HasCudart,
        bool HasCudnn,
        IReadOnlyCollection<string> DllDirectories);

    // Toolkit / cuDNN presence is fixed for the lifetime of the process (installs don't
    // appear or vanish mid-run), and CanProbeCudaExecutionProvider fans this query out
    // across every ONNX model-init site — so the recursive scan is done once and cached,
    // mirroring _bf16Cache below. A newly-installed CUDA is picked up on next launch.
    private static readonly Lazy<WindowsCudaScan> _windowsCudaScan = new(ScanWindowsCuda);

    private static WindowsCudaScan ScanWindowsCuda()
    {
        var dllDirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!OperatingSystem.IsWindows())
            return new WindowsCudaScan(false, false, dllDirs);

        // Bounded recursion: the DLLs sit at most ~2 levels below a search root
        // (Toolkit bin\x64, cuDNN bin\<cuda>\). Capping depth + ignoring inaccessible
        // and reparse-point dirs keeps an over-broad PATH root (e.g. one whose name merely
        // contains "cuda") from triggering an unbounded walk or aborting on a single
        // permission error mid-enumeration.
        var options = new EnumerationOptions
        {
            RecurseSubdirectories = true,
            MaxRecursionDepth     = 3,
            IgnoreInaccessible    = true,
            AttributesToSkip      = FileAttributes.ReparsePoint
                                  | FileAttributes.Hidden
                                  | FileAttributes.System,
        };

        // Per directory, so the decisions below can be made about a DIRECTORY rather than about
        // one file at a time -- which is what the cuDNN rule needs.
        var cudartDirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var depDirs    = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var cudnnDirs  = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var root in GetWindowsCudaSearchRoots())
        {
            try
            {
                foreach (var file in Directory.EnumerateFiles(root, "*.dll", options))
                {
                    var name = Path.GetFileName(file);
                    var dir = Path.GetDirectoryName(file);
                    if (string.IsNullOrEmpty(dir)) continue;

                    // cudart64_13.dll, not any cudart64_*.dll: see RequiredCudaMajor.
                    if (name.StartsWith($"cudart64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase))
                    { cudartDirs.Add(dir); depDirs.Add(dir); continue; }

                    // The other libraries whose name tracks the CUDA major. cuFFT and cuRAND
                    // version independently (CUDA 12 ships cuFFT 11, CUDA 13 ships 12), so gating
                    // on them would match nothing; they sit beside cudart anyway.
                    if (name.StartsWith($"cublas64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase)
                        || name.StartsWith($"cublasLt64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase)
                        || name.StartsWith($"nvrtc64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase))
                    { depDirs.Add(dir); continue; }

                    if (name.StartsWith("cudnn", StringComparison.OrdinalIgnoreCase))
                        cudnnDirs.Add(dir);
                }
            }
            catch { }
        }

        // ⚠ cuDNN'S FILE NAME SAYS NOTHING ABOUT WHICH CUDA IT IS FOR, so the directory has to.
        // Two shapes are legitimate: NVIDIA's standalone tree, which names the CUDA version in the
        // directory itself (…\CUDNN9.xin.0\), and the documented copy-into-the-toolkit
        // install, where cuDNN sits beside a matching cudart. Anything else -- typically a cuDNN
        // left in an older toolkit's bin -- is neither counted as cuDNN nor added to the search
        // path, because loading it would fail against a runtime that links CUDA 13.
        //
        // Excluding cuDNN from the search path outright, as an earlier pass did, broke the
        // standalone tree: that directory holds cudnn64_9.dll and nothing else, so it stopped being
        // registered at all and the provider could not resolve cuDNN on a correctly set up machine.
        bool IsForRequiredMajor(string dir) =>
            cudartDirs.Contains(dir)
            || Path.GetFileName(dir).StartsWith($"{RequiredCudaMajor}.", StringComparison.Ordinal);

        bool hasCudart = cudartDirs.Count > 0;
        bool hasCudnn = false;
        foreach (var dir in cudnnDirs.Where(IsForRequiredMajor))
        {
            hasCudnn = true;
            depDirs.Add(dir);
        }

        foreach (var dir in depDirs) dllDirs.Add(dir);

        return new WindowsCudaScan(hasCudart, hasCudnn, dllDirs);
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

}
