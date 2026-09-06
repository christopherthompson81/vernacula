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
    /// The CUDA major version the bundled ONNX Runtime links against.
    ///
    /// ⚠ THE MAJOR IS PART OF THE ANSWER, NOT A DETAIL. ONNX Runtime moved from CUDA 12 to CUDA 13
    /// at 1.27, and the two are not interchangeable: the provider names its dependencies with the
    /// major in the library name (libcudart.so.13, cudart64_13.dll), so a machine with only CUDA 12
    /// cannot load it. Answering "CUDA is installed" for any major meant the app tried CUDA, failed
    /// to load the provider, had the failure swallowed, and ran on the CPU with the GPU idle.
    /// </summary>
    public const int RequiredCudaMajor = 13;

    /// <summary>
    /// True if the CUDA runtime this build needs can actually be loaded.
    /// </summary>
    public static bool IsCudaToolkitInstalled() => Probe.Runtime;

    // ── The probe ─────────────────────────────────────────────────────────────

    /// <summary>What one look at the machine found. Computed together and published as a unit, so a
    /// reader never sees half of a refresh.</summary>
    private sealed record CudaProbeResult(
        bool Runtime,
        bool Cudnn,
        string? RuntimeNote,
        string? CudnnNote,
        IReadOnlyCollection<string> DllDirectories);

    private static CudaProbeResult? _probe;

    /// <summary>Bumped by every invalidation, so a probe that began earlier does not publish its
    /// answer over a newer one. Without it a Re-check could be undone by an in-flight probe and go
    /// on reporting the pre-install state for the rest of the process.</summary>
    private static int _probeGeneration;

    private static CudaProbeResult Probe
    {
        get
        {
            var current = Volatile.Read(ref _probe);
            if (current is not null) return current;

            var startedAt = Volatile.Read(ref _probeGeneration);
            var fresh = RunProbe();
            // Publish only if nothing invalidated while we were looking. Two probes of the same
            // generation produce equivalent records, so either may win.
            if (Volatile.Read(ref _probeGeneration) == startedAt)
                Volatile.Write(ref _probe, fresh);
            return fresh;
        }
    }

    private static CudaProbeResult RunProbe() =>
        OperatingSystem.IsWindows() ? ProbeWindows()
        : OperatingSystem.IsLinux() ? ProbeLinux()
        : new CudaProbeResult(false, false, null, null, Array.Empty<string>());

    /// <summary>
    /// Linux: ask the dynamic loader, not the filesystem.
    ///
    /// ⚠ THE PROVIDER dlopens BY SONAME, so the only question that matters is whether the loader
    /// can find the library. A side-by-side CUDA 13 that is not in the ldconfig cache or on
    /// LD_LIBRARY_PATH is a file we can see and the provider cannot open; answering "installed" for
    /// it lands us on the CPU with no explanation. Where that is the case we say so, because the
    /// fix is one ldconfig line rather than an install.
    /// </summary>
    private static CudaProbeResult ProbeLinux()
    {
        var (runtime, runtimeNote) = ProbeLinuxLibrary(
            $"libcudart.so.{RequiredCudaMajor}",
            $"libcudart.so.{RequiredCudaMajor}*",
            $"CUDA {RequiredCudaMajor}",
            $"No CUDA {RequiredCudaMajor} runtime was found. This build links CUDA {RequiredCudaMajor}; an older CUDA cannot load it.");

        // ⚠ THE SONAME, NOT THE BARE libcudnn.so. That one is the development symlink and on a
        // cuDNN 8 install it points at libcudnn.so.8, which the provider cannot use.
        var (cudnn, cudnnNote) = ProbeLinuxLibrary(
            "libcudnn.so.9",
            "libcudnn.so.9*",
            "cuDNN 9",
            $"No cuDNN 9 was found. The CUDA execution provider needs cuDNN 9 built for CUDA {RequiredCudaMajor}.");

        return new CudaProbeResult(runtime, cudnn, runtimeNote, cudnnNote, Array.Empty<string>());
    }

    private static (bool Found, string? Note) ProbeLinuxLibrary(
        string soname, string filePattern, string label, string absentNote)
    {
        if (NativeLibrary.TryLoad(soname, out var handle))
        {
            NativeLibrary.Free(handle);
            return (true, null);
        }

        // Every candidate across every directory, then decide: returning on the first directory
        // that merely CONTAINS a match let one broken copy hide both a good copy further down and
        // the "not on the loader path" diagnosis, which is the one with an actionable fix.
        var candidates = GetLinuxCudaLibraryDirs().SelectMany(d => SafeGetFiles(d, filePattern)).ToList();
        foreach (var file in candidates)
        {
            if (!NativeLibrary.TryLoad(file, out var byPath)) continue;
            NativeLibrary.Free(byPath);
            return (false, $"{label} was found at {file} but is not on the loader path, so the "
                         + "CUDA execution provider cannot load it. Add its directory to "
                         + "/etc/ld.so.conf.d (then run ldconfig) or to LD_LIBRARY_PATH.");
        }

        if (candidates.Count > 0)
            return (false, $"{label} was found ({candidates[0]}) but could not be loaded, even by "
                         + "full path, which usually means one of its own dependencies is missing.");

        return (false, absentNote);
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
    /// True if cuDNN can actually be loaded.
    /// </summary>
    public static bool IsCudnnInstalled() => Probe.Cudnn;

    /// <summary>Why the CUDA runtime could not be used, when the probe worked it out. Null when it
    /// could. Readable by a UI, because a desktop app has no console to read stderr from.</summary>
    public static string? CudaRuntimeNote => Probe.RuntimeNote;

    /// <summary>Why cuDNN could not be used, when the probe worked it out. Kept apart from
    /// <see cref="CudaRuntimeNote"/> so a cuDNN problem is never reported as the runtime's.</summary>
    public static string? CudnnNote => Probe.CudnnNote;

    /// <summary>Both notes, for a caller that just wants to say what is wrong.</summary>
    public static string? CudaProbeNote
    {
        get
        {
            // One snapshot: reading the two notes through their own properties could take them from
            // different generations if a refresh landed between the reads.
            var probe = Probe;
            var notes = new[] { probe.RuntimeNote, probe.CudnnNote }.Where(n => n is not null).ToArray();
            return notes.Length == 0 ? null : string.Join(" ", notes);
        }
    }

    /// <summary>
    /// The explanation to give when the CUDA execution provider will not start, including whatever
    /// the probe found. One place, because several call sites each threw their own version and only
    /// one of them said anything useful.
    /// </summary>
    public static string CudaUnavailableMessage()
    {
        var note = CudaProbeNote;
        return "Could not initialise the CUDA execution provider. This build links CUDA "
            + $"{RequiredCudaMajor}, and an older CUDA runtime cannot load it, because the major "
            + "version is part of the library name."
            + (note is null ? " A missing driver, no visible GPU, or a CPU-only build of ONNX "
                              + "Runtime will also report this." : $" {note}");
    }

    /// <summary>
    /// Discard what the probe found, so the next question is asked afresh.
    ///
    /// ⚠ THE UI PROMISES THIS. The settings window's "Re-check" button, and the help text
    /// describing it, say detection re-runs without restarting the application -- which a
    /// process-lifetime cache silently broke. Installing cuDNN or running ldconfig while the app is
    /// open is exactly when someone presses it. (An LD_LIBRARY_PATH change still needs a restart:
    /// the loader read it at process start.)
    /// </summary>
    public static void InvalidateCudaProbes()
    {
        Interlocked.Increment(ref _probeGeneration);
        Volatile.Write(ref _probe, null);
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
    /// Windows: the concrete directories that actually contain a CUDA or cuDNN runtime DLL
    /// (cudart, cublas, cufft, nvrtc, cudnn, …). Unlike <see cref="GetWindowsCudaSearchRoots"/>
    /// (roots to scan), these are the leaf directories to register with AddDllDirectory so an
    /// MSIX-packaged onnxruntime can locate the CUDA EP's dependencies at load time.
    /// </summary>
    public static IReadOnlyCollection<string> GetWindowsCudaDllDirectories() => Probe.DllDirectories;


    /// <summary>
    /// Windows: which directories hold a CUDA of the major this build needs, and whether cuDNN is
    /// among them.
    ///
    /// ⚠ TWO cuDNN LAYOUTS ARE BOTH LEGITIMATE, AND A RULE THAT SUITS ONE BREAKS THE OTHER:
    ///
    ///   • NVIDIA's standalone tree, C:\Program Files\NVIDIA\CUDNN\v9.x\bin\13.0\, holds
    ///     cudnn64_9.dll and nothing else. Requiring a cudart beside it drops this one.
    ///   • The documented copy-into-the-toolkit install puts cudnn64_9.dll in &lt;toolkit&gt;\bin, while
    ///     CUDA 13 puts its runtime DLLs in &lt;toolkit&gt;\bin\x64. Requiring the SAME directory as
    ///     cudart drops this one.
    ///
    /// So a cuDNN directory qualifies when it names the CUDA version itself, or when it belongs to
    /// a toolkit whose runtime is the major we need. Nothing else does -- a cuDNN left behind in an
    /// older toolkit is not a cuDNN we can use, and its directory must not join the search path
    /// either, since the provider would load the wrong major from it.
    /// </summary>
    private static CudaProbeResult ProbeWindows()
    {
        if (!OperatingSystem.IsWindows())
            return new CudaProbeResult(false, false, null, null, Array.Empty<string>());

        // Bounded recursion: the DLLs sit at most ~2 levels below a search root (Toolkit bin\x64,
        // cuDNN bin\<cuda>). Capping depth and skipping inaccessible or reparse-point directories
        // keeps an over-broad PATH root from triggering an unbounded walk.
        var options = new EnumerationOptions
        {
            RecurseSubdirectories = true,
            MaxRecursionDepth     = 3,
            IgnoreInaccessible    = true,
            AttributesToSkip      = FileAttributes.ReparsePoint
                                  | FileAttributes.Hidden
                                  | FileAttributes.System,
        };

        var cudartDirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var otherDeps  = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var cudnnDirs  = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var anyCudart  = false;
        var anyCudnn   = false;

        foreach (var root in GetWindowsCudaSearchRoots())
        {
            try
            {
                foreach (var file in Directory.EnumerateFiles(root, "*.dll", options))
                {
                    var name = Path.GetFileName(file);
                    var dir = Path.GetDirectoryName(file);
                    if (string.IsNullOrEmpty(dir)) continue;

                    if (name.StartsWith("cudart64_", StringComparison.OrdinalIgnoreCase))
                    {
                        anyCudart = true;
                        if (name.StartsWith($"cudart64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase))
                            cudartDirs.Add(dir);
                    }
                    else if (name.StartsWith("cudnn", StringComparison.OrdinalIgnoreCase))
                    {
                        anyCudnn = true;
                        // cudnn64_9.dll specifically, matching what Linux asks the loader for. A
                        // cuDNN 8 copied into a CUDA 13 toolkit satisfies every path rule below and
                        // still cannot be loaded by the provider, so accepting it would put the
                        // wrong major on the DLL search path, which is what this exists to prevent.
                        if (name.StartsWith("cudnn64_9", StringComparison.OrdinalIgnoreCase))
                            cudnnDirs.Add(dir);
                    }
                    // cuFFT and cuRAND version independently of the CUDA major (CUDA 12 ships cuFFT
                    // 11, CUDA 13 ships 12), so they cannot identify a directory; these can.
                    else if (name.StartsWith($"cublas64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase)
                          || name.StartsWith($"cublasLt64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase)
                          || name.StartsWith($"nvrtc64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase))
                    {
                        otherDeps.Add(dir);
                    }
                }
            }
            catch { }
        }

        var toolkitRoots = new HashSet<string>(cudartDirs.Select(ToolkitRootOf), StringComparer.OrdinalIgnoreCase);
        var usableCudnn = cudnnDirs.Where(d => NamesRequiredMajor(d) || toolkitRoots.Contains(ToolkitRootOf(d)))
                                   .ToList();

        var dllDirs = new HashSet<string>(cudartDirs, StringComparer.OrdinalIgnoreCase);
        foreach (var d in otherDeps) dllDirs.Add(d);
        foreach (var d in usableCudnn) dllDirs.Add(d);

        var runtime = cudartDirs.Count > 0;
        var cudnn = usableCudnn.Count > 0;

        var runtimeNote = runtime ? null
            : anyCudart
                ? $"A CUDA runtime was found, but not CUDA {RequiredCudaMajor}, which this build links. "
                  + $"Install the CUDA {RequiredCudaMajor} runtime."
                : $"No CUDA {RequiredCudaMajor} runtime was found.";
        var cudnnNote = cudnn ? null
            : anyCudnn
                ? $"cuDNN was found, but not cuDNN 9 belonging to a CUDA {RequiredCudaMajor} "
                  + "install, so the CUDA execution provider cannot load it."
                : "No cuDNN was found. The CUDA execution provider needs cuDNN 9 built for CUDA "
                  + $"{RequiredCudaMajor}.";

        return new CudaProbeResult(runtime, cudnn, runtimeNote, cudnnNote, dllDirs);
    }

    /// <summary>True when a path segment names the CUDA version we need, as the standalone cuDNN
    /// tree does (…\CUDNN\v9.x\bin\13.0\).</summary>
    internal static bool NamesRequiredMajor(string dir) =>
        dir.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
           .Any(seg => seg == RequiredCudaMajor.ToString()
                    || seg.StartsWith($"{RequiredCudaMajor}.", StringComparison.Ordinal));

    /// <summary>
    /// The install a directory belongs to: the parent of its `bin`, so &lt;toolkit&gt;\bin and
    /// &lt;toolkit&gt;\bin\x64 resolve to the same root and cuDNN copied into the toolkit is
    /// recognised as belonging to the CUDA there.
    /// </summary>
    internal static string ToolkitRootOf(string dir)
    {
        for (var d = dir; !string.IsNullOrEmpty(d); d = Path.GetDirectoryName(d) ?? "")
        {
            if (string.Equals(Path.GetFileName(d), "bin", StringComparison.OrdinalIgnoreCase))
                return Path.GetDirectoryName(d) ?? d;
        }
        return dir;
    }

    private static IEnumerable<string> SafeGetFiles(string dir, string pattern)
    {
        try { return Directory.Exists(dir) ? Directory.GetFiles(dir, pattern) : Array.Empty<string>(); }
        catch { return Array.Empty<string>(); }
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
