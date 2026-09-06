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

    /// <summary>The cuDNN major the CUDA provider loads (libcudnn.so.9 / cudnn64_9.dll). Named for
    /// the same reason as <see cref="RequiredCudaMajor"/>: when ONNX Runtime moves to cuDNN 10, a
    /// literal 9 scattered through this file would report "no cuDNN" on a correct install.</summary>
    public const int RequiredCudnnMajor = 9;

    /// <summary>
    /// True if the CUDA runtime this build needs can actually be loaded.
    /// </summary>
    public static bool IsCudaToolkitInstalled() => Probe.Runtime;

    // ── The probe ─────────────────────────────────────────────────────────────

    /// <summary>What one look at the machine found. Computed together and published as a unit, so a
    /// reader never sees half of a refresh.</summary>
    /// <param name="RuntimePresent">The runtime is on the machine, whether or not it can be used.
    /// Distinguishing that from "absent" is what stops the UI offering a download link beside a note
    /// saying the library is already installed and merely unreachable.</param>
    private sealed record CudaProbeResult(
        bool Runtime,
        bool Cudnn,
        string? RuntimeNote,
        string? CudnnNote,
        IReadOnlyCollection<string> DllDirectories,
        bool RuntimePresent = false,
        bool CudnnPresent = false);

    /// <summary>
    /// ⚠ A Lazy, NOT A HAND-ROLLED CACHE. CanProbeCudaExecutionProvider is asked at every
    /// model-init site, and those run in parallel; on Windows the probe is a recursive walk of
    /// every CUDA, cuDNN and PATH root. A check-then-run cache lets every caller that arrives
    /// while it is empty run the whole walk. ExecutionAndPublication means one thread does it and
    /// the others wait, and swapping the Lazy wholesale on invalidation keeps Re-check honest
    /// without reintroducing that herd.
    /// </summary>
    private static Lazy<CudaProbeResult> _probe = NewProbe();

    private static Lazy<CudaProbeResult> NewProbe() =>
        new(RunProbe, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>Counts invalidations. Not needed for correctness now that each refresh is its own
    /// Lazy, but it makes "the answer was actually thrown away" observable to a test.</summary>
    internal static int ProbeGeneration => Volatile.Read(ref _probeGeneration);

    private static int _probeGeneration;

    private static CudaProbeResult Probe => Volatile.Read(ref _probe).Value;

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
        var (runtime, runtimeNote, runtimePresent) = ProbeLinuxLibrary(
            $"libcudart.so.{RequiredCudaMajor}",
            $"libcudart.so.{RequiredCudaMajor}*",
            $"CUDA {RequiredCudaMajor}",
            $"No CUDA {RequiredCudaMajor} runtime was found. This build links CUDA {RequiredCudaMajor}; an older CUDA cannot load it.");

        // ⚠ THE SONAME, NOT THE BARE libcudnn.so. That one is the development symlink and on a
        // cuDNN 8 install it points at libcudnn.so.8, which the provider cannot use.
        var (cudnn, cudnnNote, cudnnPresent) = ProbeLinuxLibrary(
            $"libcudnn.so.{RequiredCudnnMajor}",
            $"libcudnn.so.{RequiredCudnnMajor}*",
            $"cuDNN {RequiredCudnnMajor}",
            $"No cuDNN {RequiredCudnnMajor} was found. The CUDA execution provider needs cuDNN "
            + $"{RequiredCudnnMajor} built for CUDA {RequiredCudaMajor}.");

        return new CudaProbeResult(runtime, cudnn, runtimeNote, cudnnNote, Array.Empty<string>(),
                                   runtimePresent, cudnnPresent);
    }

    private static (bool Found, string? Note, bool Present) ProbeLinuxLibrary(
        string soname, string filePattern, string label, string absentNote)
    {
        if (NativeLibrary.TryLoad(soname, out var handle))
        {
            NativeLibrary.Free(handle);
            return (true, null, true);
        }

        // ⚠ LOOK, DO NOT LOAD. Loading a candidate by full path would register it under its
        // soname for the rest of the process, so the NEXT probe would find it by name and report
        // "installed" -- losing the ldconfig advice that the first probe correctly gave. The answer
        // has to mean the same thing every time the user presses Re-check.
        // Normalised on both sides: LD_LIBRARY_PATH is commonly written with a trailing slash while
        // the same directory arrives here from CUDA_PATH without one, and a raw string comparison
        // would then give exactly the circular advice this distinction exists to avoid.
        var onLdPath = new HashSet<string>(
            (Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "")
                .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
                .Select(e => Normalise(e.Trim())),
            StringComparer.Ordinal);

        foreach (var dir in GetLinuxCudaLibraryDirs())
        {
            var hit = SafeGetFiles(dir, filePattern).FirstOrDefault();
            if (hit is null) continue;

            // ⚠ DO NOT TELL SOMEONE TO ADD A DIRECTORY THAT IS ALREADY THERE. When the file turned
            // up in an LD_LIBRARY_PATH entry, the loader has already looked and failed, so the
            // cause is the library itself -- a missing dependency of its own, or the wrong
            // architecture -- and the ldconfig advice would send them in a circle.
            return (Found: false, Note: onLdPath.Contains(Normalise(dir))
                ? $"{label} was found at {hit}, in a directory that is already on LD_LIBRARY_PATH, "
                  + "and still could not be loaded. One of its own dependencies is probably "
                  + "missing, or it was built for a different architecture."
                : $"{label} was found at {hit} but the loader could not open it by name, so the "
                  + "CUDA execution provider cannot either. Add its directory to "
                  + "/etc/ld.so.conf.d (then run ldconfig) or to LD_LIBRARY_PATH.",
                Present: true);
        }

        return (false, absentNote, false);
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
    /// <param name="providerMissing">True when the ONNX Runtime binary has no CUDA provider entry
    /// point at all, which is what an EntryPointNotFoundException means: a CPU-only build.
    /// ⚠ THE PROBE'S NOTE IS IRRELEVANT THEN. No amount of installing CUDA or cuDNN puts a provider
    /// into a binary that does not contain one, so leading with "no cuDNN found" sends that user
    /// somewhere that cannot help them.</param>
    public static string CudaUnavailableMessage(bool providerMissing = false)
    {
        // The platform question comes first: on macOS there is nothing to probe and nothing to say
        // about majors. (Asking for the note here would also force a probe whose answer is unused.)
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return "The CUDA execution provider is not available on this platform; CUDA builds of "
                 + "ONNX Runtime exist for Windows and Linux only.";

        if (providerMissing)
            return "This build of ONNX Runtime has no CUDA execution provider in it: it is a "
                 + "CPU-only build. Installing CUDA or cuDNN will not change that; the application "
                 + "needs to be built with -p:EP=Cuda.";

        // With a note, the probe knows what is wrong and says it. Without one, CUDA and cuDNN both
        // checked out, so the major version is NOT the likely cause and must not lead.
        var note = CudaProbeNote;
        return note is not null
            ? $"Could not initialise the CUDA execution provider. {note}"
            : "Could not initialise the CUDA execution provider. The CUDA "
              + $"{RequiredCudaMajor} runtime and cuDNN {RequiredCudnnMajor} were both found, so "
              + "the likely causes are a missing or too-old driver, no GPU visible to this process "
              + "(a container without the NVIDIA devices passed through, say), or a CPU-only build "
              + "of ONNX Runtime.";
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
        Volatile.Write(ref _probe, NewProbe());
        // ⚠ AND THE THINGS COMPUTED FROM IT. SupportsBf16Acceleration latches an answer derived
        // from this probe, and it decides which model bundle is selected; leaving it behind meant a
        // user could install CUDA, press Re-check, be told CUDA works, and still get the FP32
        // bundle for the rest of the session.
        Volatile.Write(ref _bf16Cache, NewBf16Cache());
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

        // One snapshot: asking through the two properties could take the answers from either side
        // of a Re-check landing between them.
        var probe = Probe;
        if (!probe.Runtime || !probe.Cudnn)
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
    /// from settings/UI flows. <see cref="InvalidateCudaProbes"/> clears it too,
    /// because a CUDA installed mid-session changes this answer as well.
    /// </summary>
    public static bool SupportsBf16Acceleration() => Volatile.Read(ref _bf16Cache).Value;

    /// <summary>The CUDA runtime is on this machine, even if it is the wrong major or cannot be
    /// loaded. Offering a download link in that case tells the user to install what they have.</summary>
    public static bool IsCudaRuntimePresent => Probe.RuntimePresent;

    /// <summary>cuDNN is on this machine, even if it is not usable. See
    /// <see cref="IsCudaRuntimePresent"/>.</summary>
    public static bool IsCudnnPresent => Probe.CudnnPresent;

    /// <summary>Whether anything has asked for <see cref="SupportsBf16Acceleration"/> yet. Lets a
    /// caller find out whether the answer CHANGED without forcing the probe that computes it --
    /// which, on a cold cache, would run the whole scan only to throw it away.</summary>
    public static bool IsBf16AnswerKnown => Volatile.Read(ref _bf16Cache).IsValueCreated;

    private static Lazy<bool> _bf16Cache = NewBf16Cache();

    private static Lazy<bool> NewBf16Cache() => new(() =>
    {
        if (!CanProbeCudaExecutionProvider()) return false;
        var (major, _) = GetCudaComputeCapability();
        return major >= 8;
    }, LazyThreadSafetyMode.ExecutionAndPublication);

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
    /// Windows: is there a CUDA of the major this build needs, and a cuDNN it can use?
    ///
    /// ⚠ THE MAJOR IS CHECKED ON cudart, WHICH CARRIES IT, AND NOT ON cuDNN, WHICH DOES NOT.
    /// cudart64_13.dll names its CUDA; cudnn64_9.dll is called that whether it was built for CUDA
    /// 12 or 13, so no rule over file names or directory layout can tell those apart. An earlier
    /// attempt inferred it from the directory instead -- a cuDNN counted only if its path named the
    /// CUDA version or it sat under the same toolkit as a matching cudart -- and that rejected the
    /// commonest Windows install of all: unzip cuDNN to C:\cudnn and put it on PATH. Being strict
    /// about something unknowable cost a working configuration and bought nothing.
    ///
    /// So cuDNN is answered the way Linux answers it: is a cuDNN of the required major there at
    /// all. A cuDNN built for the wrong CUDA still gets past this and fails at provider init, on
    /// both platforms, and that limit is real rather than papered over.
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

        var cudartDirs   = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var depDirs      = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var unversioned  = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var cudnnDirs    = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var anyCudart    = false;
        var anyCudnn     = false;
        var otherMajor   = false;

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
                        else
                            otherMajor = true;   // another CUDA is installed alongside
                    }
                    else if (name.StartsWith("cudnn", StringComparison.OrdinalIgnoreCase))
                    {
                        anyCudnn = true;
                        if (name.StartsWith($"cudnn64_{RequiredCudnnMajor}", StringComparison.OrdinalIgnoreCase))
                            cudnnDirs.Add(dir);
                    }
                    // ⚠ ONLY LIBRARIES THAT NAME THE MAJOR MAY IDENTIFY A DIRECTORY. cuFFT and
                    // cuRAND version independently -- curand64_10.dll is called that under both
                    // CUDA 12 and 13 -- so a directory known only by them could be CUDA 12's bin,
                    // and adding it would let the provider bind CUDA 12's cuRAND beside a CUDA 13
                    // cudart. In a stock toolkit they sit in the cudart directory anyway, which is
                    // already on the list.
                    else if (name.StartsWith($"cublas64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase)
                          || name.StartsWith($"cublasLt64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase)
                          || name.StartsWith($"nvrtc64_{RequiredCudaMajor}", StringComparison.OrdinalIgnoreCase))
                    {
                        depDirs.Add(dir);
                    }
                    // cuFFT and cuRAND keep their names across CUDA majors (curand64_10.dll under
                    // both 12 and 13), so they identify a directory only when no other CUDA is
                    // installed to confuse it with. The provider does load them, and a packaged app
                    // cannot fall back on PATH, so they are worth having when they are unambiguous.
                    else if (name.StartsWith("cufft", StringComparison.OrdinalIgnoreCase)
                          || name.StartsWith("curand", StringComparison.OrdinalIgnoreCase))
                    {
                        unversioned.Add(dir);
                    }
                }
            }
            catch { }
        }

        var runtime = cudartDirs.Count > 0;
        var cudnn = cudnnDirs.Count > 0;

        // Only from an install we are actually going to use, and in a deliberate order: a HashSet
        // enumerates unpredictably, and AddDllDirectory order decides which of two cuDNNs the
        // loader binds. Ours first, then cuDNN sitting with it, then anything else.
        var dllDirs = new List<string>();
        if (runtime)
        {
            void Add(IEnumerable<string> dirs)
            {
                foreach (var d in dirs)
                    if (!dllDirs.Contains(d, StringComparer.OrdinalIgnoreCase))
                        dllDirs.Add(d);
            }

            // ⚠ ON PATH BOUNDARIES, NOT CHARACTERS. A bare StartsWith makes C:\cuda12\bin look
            // like it sits under C:\cuda, so a CUDA 12 cuDNN would be ordered ahead of the one
            // actually paired with our runtime -- the opposite of what this ordering is for.
            bool BesideOurCuda(string dir)
            {
                var d = Normalise(dir);
                return cudartDirs.Select(Normalise)
                                 .Any(c => c.StartsWith(d, StringComparison.OrdinalIgnoreCase)
                                        || d.StartsWith(c, StringComparison.OrdinalIgnoreCase));
            }

            // ⚠ THE DIRECTORY NAME ORDERS cuDNN; IT DOES NOT DECIDE IT. NVIDIA's standalone package
            // installs into per-CUDA subdirectories -- …\CUDNN\v9.x\bin\12.9\ AND …\bin\13.0\ --
            // so both land in the set, neither sits beside our cudart, and enumeration order would
            // hand the loader 12.9 first. That name is real evidence about which CUDA a cuDNN was
            // built for; it is just not reliable enough to reject an install over (a cuDNN unzipped
            // to C:\cudnn names nothing at all), which is why it ranks rather than filters.
            int CudnnRank(string dir) =>
                NamesMajor(dir, RequiredCudaMajor) ? 0
                : BesideOurCuda(dir)              ? 1
                : NamesSomeOtherMajor(dir)        ? 3
                                                  : 2;

            Add(cudartDirs);
            Add(depDirs);
            // Beside our own runtime, these are unambiguous whatever else is installed; further
            // afield they are only safe when no other CUDA is present to confuse them with, because
            // curand64_10.dll carries that name under every major.
            Add(unversioned.Where(BesideOurCuda));
            if (!otherMajor) Add(unversioned);
            Add(cudnnDirs.OrderBy(CudnnRank));
        }

        var runtimeNote = runtime ? null
            : anyCudart
                ? $"A CUDA runtime was found, but not CUDA {RequiredCudaMajor}, which this build "
                  + $"links. Install the CUDA {RequiredCudaMajor} runtime."
                : $"No CUDA {RequiredCudaMajor} runtime was found.";
        var cudnnNote = cudnn ? null
            : anyCudnn
                ? $"cuDNN was found, but not cuDNN {RequiredCudnnMajor}, which the CUDA execution "
                  + "provider loads."
                : $"No cuDNN was found. The CUDA execution provider needs cuDNN {RequiredCudnnMajor} "
                  + $"built for CUDA {RequiredCudaMajor}.";

        return new CudaProbeResult(runtime, cudnn, runtimeNote, cudnnNote, dllDirs,
                                   RuntimePresent: anyCudart, CudnnPresent: anyCudnn);
    }

    /// <summary>True when a path segment names the given CUDA major, as NVIDIA's standalone cuDNN
    /// tree does (…\CUDNN\v9.x\bin\13.0\).</summary>
    private static bool NamesMajor(string dir, int major) =>
        dir.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
           .Any(seg => seg == major.ToString()
                    || seg.StartsWith($"{major}.", StringComparison.Ordinal));

    /// <summary>True when a path segment names a CUDA major that is not the one we need — evidence
    /// that this directory belongs to another install, and should be tried last.</summary>
    private static bool NamesSomeOtherMajor(string dir) =>
        Enumerable.Range(9, 12).Any(m => m != RequiredCudaMajor && NamesMajor(dir, m));

    /// <summary>A directory in comparable form: absolute, with exactly one trailing separator, so
    /// two spellings of the same directory match and a prefix test lands on a path boundary.</summary>
    private static string Normalise(string dir)
    {
        try { return Path.TrimEndingDirectorySeparator(Path.GetFullPath(dir)) + Path.DirectorySeparatorChar; }
        catch { return dir; }
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
