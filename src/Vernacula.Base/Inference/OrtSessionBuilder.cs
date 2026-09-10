using System.Security.Cryptography;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Vernacula.Base.Models;

namespace Vernacula.Base.Inference;

/// <summary>
/// Unified ONNX Runtime <see cref="SessionOptions"/> factory shared by every ASR
/// backend. Replaces the per-backend <c>MakeSessionOptions</c> copies that had
/// drifted into slightly different error messages, exception filters, and
/// execution-provider ordering.
/// </summary>
public static class OrtSessionBuilder
{
    /// <summary>
    /// Build a <see cref="SessionOptions"/> for the requested execution provider.
    /// In <see cref="ExecutionProvider.Auto"/> mode CUDA is tried first (guarded
    /// by <see cref="HardwareInfo.CanProbeCudaExecutionProvider"/>) and DirectML
    /// is registered as a fallback; either or both may silently fail. Strict
    /// modes throw a descriptive <see cref="InvalidOperationException"/> if the
    /// requested provider is unavailable in the current ORT build.
    /// </summary>
    public static SessionOptions Create(
        ExecutionProvider ep,
        GraphOptimizationLevel optLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        bool enableProfiling = false,
        string? coreMlModelPath = null)
        => Create(ep, optLevel, enableProfiling, out _, coreMlModelPath: coreMlModelPath);

    /// <inheritdoc cref="Create(ExecutionProvider, GraphOptimizationLevel, bool)"/>
    /// <param name="usedCuda">True when the CUDA execution provider was
    /// successfully appended. Callers use this to gate CUDA-only paths
    /// (IOBinding, CUDA graphs) without re-probing.</param>
    public static SessionOptions Create(
        ExecutionProvider ep,
        GraphOptimizationLevel optLevel,
        bool enableProfiling,
        out bool usedCuda,
        bool disableTf32 = false,
        string? coreMlModelPath = null)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = optLevel };
        if (enableProfiling)
            opts.EnableProfiling = true;

        // VERNACULA_ORT_VERBOSE=1 turns on ORT INFO-level logging across every
        // session built through this factory. Used to diagnose graph-level
        // surprises like "5 Memcpy nodes are added to the graph"
        // (issue #41 perf round 2 / Run 12). Default is ORT's normal warning
        // floor; opt-in only because INFO is chatty.
        if (Environment.GetEnvironmentVariable("VERNACULA_ORT_VERBOSE") == "1")
            opts.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;

        usedCuda = false;
        switch (ep)
        {
            case ExecutionProvider.Auto:
                // macOS has neither CUDA nor DirectML. The osx-arm64 ORT build
                // ships CoreML + WebGPU; Auto picks WebGPU because it is the safe
                // choice for ANY graph, including the stock dynamic-shape exports
                // that CoreML cannot compile at all. CoreML is faster once a model
                // has been through docs/coreml_onnx_playbook.md, but that is a
                // per-model property, so selecting it is left explicit.
                if (OperatingSystem.IsMacOS())
                {
                    // Gated on the provider actually being in this build: Auto is what the
                    // desktop app uses everywhere and it has no EP setting, so a throw here
                    // would break every model load with no way out. See ProviderAvailable.
                    if (ProviderAvailable(WebGpuProviderName))
                    {
                        try { AppendWebGpu(opts); }
                        catch { }
                    }
                    break;
                }
                if (HardwareInfo.CanProbeCudaExecutionProvider())
                {
                    try
                    {
                        AppendCuda(opts, disableTf32);
                        usedCuda = true;
                    }
                    catch { }
                }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;

            case ExecutionProvider.Cuda:
                try
                {
                    AppendCuda(opts, disableTf32);
                    usedCuda = true;
                }
                // ⚠ BEFORE THE BROAD CATCH. EntryPointNotFoundException means the binary has no
                // CUDA provider in it, and it would otherwise be handed the probe's note -- telling
                // someone to install cuDNN when nothing they install can add a provider.
                catch (EntryPointNotFoundException ex)
                {
                    throw new InvalidOperationException(
                        HardwareInfo.CudaUnavailableMessage(providerMissing: true), ex);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        HardwareInfo.CudaUnavailableMessage()
                        + " Run on the CPU instead with the application's Cpu execution-provider "
                        + "setting, or build with -p:EP=Cpu.", ex);
                }
                break;

            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        "DirectML EP not available. Build with -p:EP=DirectML (Windows only).", ex);
                }
                break;

            case ExecutionProvider.CoreML:
                try { AppendCoreML(opts, coreMlModelPath); }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        "CoreML EP not available in the current ONNX Runtime build.", ex);
                }
                break;

            case ExecutionProvider.WebGpu:
                try { opts.AppendExecutionProvider("WebGPU", new Dictionary<string, string>()); }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        "WebGPU EP not available in the current ONNX Runtime build.", ex);
                }
                break;

            case ExecutionProvider.Cpu:
                break;
        }

        return opts;
    }

    // Append the CoreML EP. Uses the ML Program format (CoreML's current IR --
    // the legacy NeuralNetwork format is frozen) and lets CoreML pick among CPU,
    // GPU and ANE. Note that CoreML silently declines any node whose shape has an
    // unbounded dimension, so graphs with a dynamic time axis end up heavily
    // partitioned; measure before preferring this over CPU.
    private static void AppendCoreML(SessionOptions opts, string? modelPath = null)
    {
        var cfg = new Dictionary<string, string>
        {
            ["ModelFormat"] = "MLProgram",
            ["MLComputeUnits"] = "ALL",
        };

        // Without this the EP recompiles the CoreML model on EVERY session creation.
        // Measured on the Sortformer steady-state graph, session load drops 2.99 s -> 0.85 s,
        // and TranscriptionService builds a streamer per transcription, so it is ~2 s of
        // per-run latency for a directory. Best-effort -- if the cache dir cannot be
        // created, drop the option rather than fail the session.
        //
        // ⚠ IT IS NOT SMALL AND NOTHING PRUNES IT. The compiled .mlmodelc for the 527 MB
        // Sortformer variant alone is ~1.0 GB, and ORT keys entries by graph, so every model
        // and every re-export of one adds another. Trading ~2 s per run for unbounded disk
        // is the right default for a desktop app, but it wants a cap or a cleanup path.
        string? dir = CoreMLCacheDirectoryFor(modelPath);
        if (dir is not null)
            cfg["ModelCacheDirectory"] = dir;

        opts.AppendExecutionProvider("CoreML", cfg);
    }

    /// <summary>
    /// A cache directory private to this exact model file, or null to cache nothing.
    /// </summary>
    /// <remarks>
    /// ⚠ ORT DERIVES ITS CACHE KEY FROM THE MODEL PATH, NOT ITS CONTENT. Verified: the same
    /// bytes at two paths compile twice and produce two entries, so an in-place replacement
    /// at ONE path -- exactly what ModelManagerService does when it re-downloads a changed
    /// asset to the same filename -- would silently reuse the previous export's compiled
    /// bundle. Wrong outputs, or a hard failure if the export's fixed dims moved.
    ///
    /// So the directory is namespaced by (length, last-write-time) of the file, which both
    /// change when it is replaced and cost no hashing of a half-gigabyte model. Callers that
    /// do not name a model get no cache rather than a shared one: a stale compiled graph is
    /// a correctness bug, and ~2 s of compile is not worth risking it.
    /// </remarks>
    private static string? CoreMLCacheDirectoryFor(string? modelPath)
    {
        if (string.IsNullOrEmpty(modelPath))
            return null;

        string? root = CoreMLCacheRoot.Value;
        if (root is null)
            return null;

        try
        {
            var info = new FileInfo(modelPath);
            if (!info.Exists)
                return null;

            // The prefix identifies the model FILE (path), the suffix its CONTENT version.
            // Splitting them that way is what lets the prune below remove previous versions
            // of this file without touching a different model that happens to share a
            // basename -- two model roots each holding a diar_..._coreml.onnx, say.
            string prefix = $"{StableHash(info.FullName):x8}-{Path.GetFileNameWithoutExtension(modelPath)}";
            string token  = $"{prefix}-{info.Length:x}-{info.LastWriteTimeUtc.Ticks:x}";
            string dir    = Path.Combine(root, token);
            Directory.CreateDirectory(dir);

            PruneStaleVersions(root, prefix, keep: token);
            return dir;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Deletes compiled caches for previous versions of one model file.
    /// </summary>
    /// <remarks>
    /// Namespacing the cache per content version fixed stale-graph reuse, but on its own it
    /// turns re-exports into an unbounded pile: each is ~1 GB for a model this size and
    /// nothing reclaims the last one. Only directories under our own cache root whose prefix
    /// marks them as an older version of THIS file are removed.
    ///
    /// Best-effort by design. A concurrent session may still hold an old directory open; the
    /// delete fails, is swallowed, and the worst outcome is that ORT recompiles later.
    /// </remarks>
    private static void PruneStaleVersions(string root, string prefix, string keep)
    {
        try
        {
            foreach (string dir in Directory.EnumerateDirectories(root, prefix + "-*"))
            {
                if (string.Equals(Path.GetFileName(dir), keep, StringComparison.Ordinal))
                    continue;
                try { Directory.Delete(dir, recursive: true); }
                catch { /* in use, or gone already */ }
            }
        }
        catch { /* the root vanished; nothing to prune */ }
    }

    /// <summary>FNV-1a over the string, for a short stable directory prefix. Not a digest.</summary>
    private static uint StableHash(string value)
    {
        unchecked
        {
            uint h = 2166136261;
            foreach (char c in value)
            {
                h ^= c;
                h *= 16777619;
            }
            return h;
        }
    }

    /// <summary>
    /// Where the CoreML EP caches compiled `.mlmodelc` bundles, or null if it cannot be
    /// created. Lazy because every model-init site asks and those run in parallel.
    /// </summary>
    private static readonly Lazy<string?> CoreMLCacheRoot = new(() =>
    {
        try
        {
            // ⚠ On Unix this returns "" when neither XDG_DATA_HOME nor HOME is set (launchd
            // agents, some sandboxes). Path.Combine would then yield the RELATIVE
            // "Vernacula/coreml-cache", CreateDirectory would succeed, and ORT would write
            // a multi-GB cache into whatever the process CWD happens to be.
            string root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (string.IsNullOrEmpty(root))
                return null;

            string dir = Path.Combine(root, "Vernacula", "coreml-cache");
            Directory.CreateDirectory(dir);
            return dir;
        }
        catch
        {
            return null;
        }
    });

    // The short names AppendExecutionProvider takes, and the long names
    // GetAvailableProviders reports. They are not the same strings.
    private const string WebGpuProviderName = "WebGpuExecutionProvider";
    private const string CoreMLProviderName = "CoreMLExecutionProvider";

    private static void AppendWebGpu(SessionOptions opts)
        => opts.AppendExecutionProvider("WebGPU", new Dictionary<string, string>());

    /// <summary>
    /// Whether this ONNX Runtime build actually carries the named provider.
    /// </summary>
    /// <remarks>
    /// ⚠ A Lazy for the same reason as HardwareInfo's CUDA probe: this is asked at every
    /// model-init site and those run in parallel.
    ///
    /// This matters most for Auto. Registering an absent provider throws, which the Auto path
    /// swallows -- but on macOS the desktop app has no EP setting (every service defaults to
    /// Auto), so anything Auto cannot recover from takes down every model load with no way for
    /// the user to opt out. Asking ORT what it has is cheaper and more honest than registering
    /// blind and catching. Notably the macOS-x64 pin (ORT 1.19.2) predates the WebGPU EP
    /// entirely, and a Windows/Linux EP=DirectML or EP=Cuda build has no WebGPU either.
    ///
    /// It answers "is the provider IN this build", not "will a device be found". A build that
    /// has the provider but no usable Metal adapter still fails at session creation.
    /// </remarks>
    private static readonly Lazy<HashSet<string>> _availableProviders = new(
        () =>
        {
            try
            {
                return new HashSet<string>(
                    OrtEnv.Instance().GetAvailableProviders(), StringComparer.OrdinalIgnoreCase);
            }
            catch
            {
                // Never let a probe failure be worse than not probing.
                return new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            }
        },
        LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Fail an explicitly requested provider with the reason, rather than letting
    /// AppendExecutionProvider throw something that does not say what to do about it.
    /// </summary>
    private static void RequireProvider(string providerName, string display)
    {
        if (ProviderAvailable(providerName))
            return;

        string built = string.Join(", ", _availableProviders.Value.OrderBy(p => p));
        throw new InvalidOperationException(
            $"{display} is not in this ONNX Runtime build (it has: {built}). {display} is macOS "
          + "arm64 only, and comes from the plain Microsoft.ML.OnnxRuntime package -- build with "
          + "-p:EP=Cpu on Apple Silicon. A Cuda or DirectML build carries no macOS accelerator.");
    }

    private static bool ProviderAvailable(string name)
    {
        var have = _availableProviders.Value;
        // An empty set means the probe itself failed; fall back to trying, which is the
        // pre-probe behaviour.
        return have.Count == 0 || have.Contains(name);
    }

    /// <summary>
    /// Append the macOS accelerator implied by <paramref name="ep"/>, returning true when one
    /// was actually registered.
    /// </summary>
    /// <remarks>
    /// The DiariZen-family backends (WeSpeakerEmbedder, VoxLinguaLid, DiariZenDiarizer) build
    /// their own <see cref="SessionOptions"/> because each sets its own thread counts, so they
    /// cannot route through <see cref="Create(ExecutionProvider)"/>. They still have to agree
    /// with it about what Auto/CoreML/WebGpu mean, and three hand-rolled switches had already
    /// drifted once. This is the shared tail they call instead of growing a fourth copy.
    /// A false return means "nothing macOS-specific applies" -- the caller should run its
    /// ordinary CUDA/DirectML path, which lands on CPU here.
    /// </remarks>
    public static bool TryAppendPlatformAccelerator(
        SessionOptions opts, ExecutionProvider ep, string? coreMlModelPath = null)
    {
        switch (ep)
        {
            case ExecutionProvider.CoreML:
                RequireProvider(CoreMLProviderName, "CoreML");
                AppendCoreML(opts, coreMlModelPath);
                return true;

            case ExecutionProvider.WebGpu:
                RequireProvider(WebGpuProviderName, "WebGPU");
                AppendWebGpu(opts);
                return true;

            // Auto prefers WebGPU on macOS (see the Auto case in Create). Best effort:
            // if the provider is not in this ORT build, fall back rather than throw.
            case ExecutionProvider.Auto when OperatingSystem.IsMacOS():
                if (!ProviderAvailable(WebGpuProviderName))
                    return false;
                try { AppendWebGpu(opts); return true; }
                catch { return false; }

            default:
                return false;
        }
    }

    // Append the CUDA EP, optionally forcing full-fp32 matmul (use_tf32=0). TF32's ~1e-2
    // error is fine for one-shot models but COMPOUNDS catastrophically through OmniVoice's
    // iterative diffusion loop (audible noise) — see docs/omnivoice_onnx_investigation.md.
    private static void AppendCuda(SessionOptions opts, bool disableTf32)
    {
        if (!disableTf32)
        {
            opts.AppendExecutionProvider_CUDA(0);
            return;
        }
        using var cuda = new OrtCUDAProviderOptions();
        cuda.UpdateOptions(new Dictionary<string, string> { ["device_id"] = "0", ["use_tf32"] = "0" });
        if (Environment.GetEnvironmentVariable("VERNACULA_ORT_VERBOSE") == "1")
            Console.Error.WriteLine($"[OrtSessionBuilder] CUDA use_tf32=0 -> {cuda.GetOptions()}");
        opts.AppendExecutionProvider_CUDA(cuda);
    }

    /// <summary>
    /// Create an <see cref="InferenceSession"/> backed by a disk-cached
    /// post-optimization graph. First call loads <paramref name="modelPath"/>,
    /// runs graph optimization at <paramref name="optLevel"/>, and writes the
    /// optimized graph next to the source. Subsequent calls find the cached
    /// file, skip optimization (<c>ORT_DISABLE_ALL</c>), and load directly —
    /// typically 5–10× faster for large graphs.
    ///
    /// <b>Layered cache format.</b> Four states per cache key, advanced by
    /// observed round-trip behavior:
    /// <list type="number">
    ///   <item><description><b>.onnx + sidecar</b> (primary). Fast: ORT can
    ///     lazy-load weights from a <c>_data</c> external-initializer
    ///     sidecar. Works for almost every graph.</description></item>
    ///   <item><description><b>.onnx inline</b> (marked by a
    ///     <c>.no-ext-init</c> hint file). Same format, written without the
    ///     external-initializer session entries. Used when the primary tier
    ///     can't round-trip, which happens for any graph carrying a
    ///     control-flow subgraph (<c>Loop</c>, <c>If</c>): ORT's
    ///     external-initializer writer appends a subgraph's initializers to
    ///     that subgraph a second time instead of clearing first, and then
    ///     rejects its own output with "<c>&lt;name&gt; initializer name is
    ///     not unique</c>" (issue #60, still open upstream as of ORT 1.29).
    ///     Dropping the sidecar entries sidesteps it. Counter-intuitively
    ///     this is also the *fastest* tier for a graph whose source already
    ///     has a <c>_data</c> sidecar: the cache file keeps pointing at the
    ///     source's weights instead of copying them, so it's both smaller on
    ///     disk and quicker to load than tier 1 would have been.</description></item>
    ///   <item><description><b>.ort</b> (marked by a <c>.use-ort</c> hint
    ///     file). ORT's binary format — a different serializer that
    ///     round-trips at every opt level. Reached either when the inline
    ///     tier also fails to round-trip, or when its write fails outright,
    ///     which is what a >2 GB graph does once the sidecar is gone
    ///     (protobuf message limit). Embeds all initializers inline, so
    ///     loads are slower for large graphs. Only used when
    ///     needed.</description></item>
    ///   <item><description><b>No cache</b> (last resort, marked by a
    ///     <c>.cache-disabled</c> sentinel). Used when even <c>.ort</c>
    ///     can't round-trip. Stops the broken write→fail→delete cycle.
    ///     Has not been observed in practice but the path exists as
    ///     defense-in-depth.</description></item>
    /// </list>
    ///
    /// Convergence costs one extra cache miss per state transition: a
    /// subgraph-bearing graph takes 2 runs to reach a steady-state cache hit
    /// (run 1: write .onnx+sidecar → run 2: reload fails, escalate to the
    /// inline hint and rewrite .onnx in the same call → run 3+: cache HIT).
    /// Most graphs stay on tier 1 forever.
    ///
    /// Cache key (in the path stem) embeds EP, ORT version, and source-file
    /// mtime+size, so source changes or ORT upgrades automatically invalidate
    /// everything including hints and sentinels. Stale files are NOT
    /// auto-cleaned — callers can `rm &lt;stem&gt;.opt.*` to reset.
    ///
    /// Set <c>VERNACULA_ORT_NO_CACHE=1</c> to bypass the cache entirely
    /// (forces a fresh full-optimization load every time; useful when debugging
    /// graph-level surprises).
    /// </summary>
    public static InferenceSession CreateCachedSession(
        string modelPath,
        ExecutionProvider ep,
        GraphOptimizationLevel optLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        long externalInitializersMinBytes = 1024 * 1024)
        => CreateCachedSession(modelPath, ep, out _, out _, optLevel, externalInitializersMinBytes);

    /// <inheritdoc cref="CreateCachedSession(string, ExecutionProvider, GraphOptimizationLevel, long)"/>
    /// <param name="cacheHit">True if a valid pre-optimized file was found and
    /// reused; false if a fresh optimization happened (and was written to disk
    /// for next time).</param>
    public static InferenceSession CreateCachedSession(
        string modelPath,
        ExecutionProvider ep,
        out bool cacheHit,
        GraphOptimizationLevel optLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        long externalInitializersMinBytes = 1024 * 1024)
        => CreateCachedSession(modelPath, ep, out cacheHit, out _, optLevel, externalInitializersMinBytes);

    /// <inheritdoc cref="CreateCachedSession(string, ExecutionProvider, GraphOptimizationLevel, long)"/>
    /// <param name="cacheHit">True if a valid pre-optimized file was found and
    /// reused; false if a fresh optimization happened (and was written to disk
    /// for next time).</param>
    /// <param name="usedCuda">True when the CUDA execution provider was
    /// successfully appended to the session that actually backs the returned
    /// <see cref="InferenceSession"/>. False for CPU, DirectML, or when
    /// <see cref="ExecutionProvider.Auto"/> fell back to DirectML because
    /// CUDA was unavailable. Callers use this (rather than the *requested*
    /// EP) to gate CUDA-only paths like IOBinding without re-probing.</param>
    public static InferenceSession CreateCachedSession(
        string modelPath,
        ExecutionProvider ep,
        out bool cacheHit,
        out bool usedCuda,
        GraphOptimizationLevel optLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        long externalInitializersMinBytes = 1024 * 1024,
        bool disableTf32 = false)
    {
        cacheHit = false;
        usedCuda = false;
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        bool bypassCache = Environment.GetEnvironmentVariable("VERNACULA_ORT_NO_CACHE") == "1";
        string cacheBase = bypassCache ? "" : ComputeCacheBasePath(modelPath, ep, optLevel);
        // All cache-key-derived paths. The six files form a small state
        // machine; the per-key disposition is encoded by which marker
        // (if any) exists alongside the actual cache file(s).
        string cachePathOnnx = cacheBase + ".onnx";   // primary cache (with _data sidecar)
        string cacheDataPath = cachePathOnnx + "_data";
        string cachePathOrt = cacheBase + ".ort";     // last-resort fallback for subgraph-bearing graphs
        string noExtInitHintPath = cacheBase + ".no-ext-init"; // ".onnx+sidecar round-trip failed, write .onnx inline"
        string useOrtHintPath = cacheBase + ".use-ort";       // ".onnx round-trip failed both ways, escalate to .ort"
        string cacheDisabledPath = cacheBase + ".cache-disabled"; // "both formats failed, skip caching"

        bool cacheDisabled = !bypassCache && File.Exists(cacheDisabledPath);
        bool useOrtFormat = !bypassCache && !cacheDisabled && File.Exists(useOrtHintPath);
        // .use-ort wins if both hints are somehow present — it's the later
        // (more conservative) state in the escalation order.
        bool noExtInit = !bypassCache && !cacheDisabled && !useOrtFormat
                         && File.Exists(noExtInitHintPath);

        // One-time migration for caches written before issue #60 was
        // understood. A .use-ort hint with no .no-ext-init beside it was
        // written by the old two-tier ladder, which jumped straight to .ort on
        // a failed .onnx round-trip. We now know the sidecar is the trigger and
        // that the inline tier is both faster and smaller, so retry it once.
        // Costs one cache miss. A graph that genuinely needs .ort re-escalates
        // on its own — and the .no-ext-init hint left behind stops this from
        // firing a second time.
        // A .cache-disabled key from that era is migrated too: it means both
        // old tiers failed, and the inline tier — which neither of them was —
        // may well work. If it doesn't, the ladder walks the key straight back
        // to disabled on its own.
        if ((useOrtFormat || cacheDisabled) && !bypassCache && !File.Exists(noExtInitHintPath))
        {
            try
            {
                // Order matters: clear the old markers BEFORE writing the new
                // one. If a delete throws (locked file, EACCES) we're left
                // with no hints at all, which replays the ladder from tier 1
                // next run. Writing .no-ext-init first and then failing the
                // delete would strand the key on .ort permanently, since the
                // guard above would never be true again.
                if (File.Exists(useOrtHintPath)) File.Delete(useOrtHintPath);
                if (File.Exists(cacheDisabledPath)) File.Delete(cacheDisabledPath);
                File.WriteAllText(noExtInitHintPath, "");
                useOrtFormat = false;
                cacheDisabled = false;
                noExtInit = true;
                try { if (File.Exists(cachePathOrt)) File.Delete(cachePathOrt); } catch { /* best-effort */ }
                Console.WriteLine(
                    $"[cache-format] {Path.GetFileName(cachePathOnnx)}: " +
                    "retrying inline .onnx in place of the older fallback (see issue #60); one cache miss to converge.");
            }
            catch
            {
                // Couldn't rewrite the markers. Whatever survived on disk still
                // describes a valid state, so just carry on with it.
            }
        }

        if (cacheDisabled)
        {
            // Defensive cleanup: if a previous catch-block's File.Delete
            // silently failed (file locked, EACCES, etc.) we'd be leaking
            // hundreds of MB per affected model. Retry the deletes now that
            // we know the cache is permanently disabled for this key.
            try { if (File.Exists(cachePathOnnx)) File.Delete(cachePathOnnx); } catch { /* best-effort */ }
            try { if (File.Exists(cacheDataPath)) File.Delete(cacheDataPath); } catch { /* best-effort */ }
            try { if (File.Exists(cachePathOrt))  File.Delete(cachePathOrt);  } catch { /* best-effort */ }
        }

        string activeCachePath = useOrtFormat ? cachePathOrt : cachePathOnnx;

        if (!bypassCache && !cacheDisabled && File.Exists(activeCachePath))
        {
            // Cache hit: load pre-optimized graph with optimization DISABLED
            // (the graph is already optimized; re-running passes is wasted work
            // and may hit unsupported-op errors on a fused graph).
            var hitOpts = Create(ep, GraphOptimizationLevel.ORT_DISABLE_ALL, enableProfiling: false, out var hitUsedCuda, disableTf32);
            try
            {
                var session = new InferenceSession(activeCachePath, hitOpts);
                cacheHit = true;
                usedCuda = hitUsedCuda;
                return session;
            }
            catch
            {
                hitOpts.Dispose();
                if (useOrtFormat)
                {
                    // .ort fallback also can't round-trip. Disable caching
                    // for this key entirely. Logged once (not per-load) so
                    // the silent transition is visible.
                    Console.WriteLine(
                        $"[cache-disabled] {Path.GetFileName(cachePathOrt)}: " +
                        ".ort fallback ALSO failed to round-trip; disabling cache for this model (see issue #56).");
                    try { File.Delete(cachePathOrt); } catch { /* best-effort */ }
                    try { File.WriteAllText(cacheDisabledPath, ""); } catch { /* best-effort */ }
                    cacheDisabled = true;
                }
                else if (noExtInit)
                {
                    // Inline .onnx ALSO failed to round-trip; escalate to .ort.
                    // Not observed for any graph we ship — the inline tier
                    // fixes both known cases — but the rung exists so a new
                    // graph with some third serializer defect still ends up
                    // cached rather than re-optimized on every load.
                    Console.WriteLine(
                        $"[cache-format] {Path.GetFileName(cachePathOnnx)}: " +
                        "inline .onnx round-trip ALSO failed; switching to .ort for this model (see issue #60).");
                    try { File.Delete(cachePathOnnx); } catch { /* best-effort */ }
                    // The inline tier never writes a _data sidecar, but a stale
                    // one from a tier-1 write whose cleanup failed can still be
                    // sitting here. Nothing downstream cleans it up once the key
                    // escalates to .ort, so drop it now.
                    try { File.Delete(cacheDataPath); } catch { /* best-effort */ }
                    try { File.WriteAllText(useOrtHintPath, ""); } catch { /* best-effort */ }
                    useOrtFormat = true;
                }
                else
                {
                    // .onnx + external-initializer sidecar failed to round-trip.
                    // That combination is the actual ORT bug (issue #60): the
                    // external-initializer writer appends a subgraph's
                    // initializers to that subgraph a second time instead of
                    // clearing first, so ORT rejects its own output with
                    // "<name> initializer name is not unique". Dropping just
                    // the sidecar entries fixes it — and is *faster* than the
                    // .ort fallback, because the inline graph keeps pointing at
                    // the source model's existing _data sidecar instead of
                    // copying every weight into the cache file.
                    //
                    // noExtInit=true makes the fall-through cache-write path
                    // re-emit .onnx inline this run — so by the end of *this*
                    // call the usable cache is on disk and the next call HITs.
                    Console.WriteLine(
                        $"[cache-format] {Path.GetFileName(cachePathOnnx)}: " +
                        ".onnx round-trip failed (subgraph-bearing graph); rewriting without the external-initializer sidecar (see issue #60).");
                    try { File.Delete(cachePathOnnx); } catch { /* best-effort */ }
                    try { File.Delete(cacheDataPath); } catch { /* best-effort */ }
                    try { File.WriteAllText(noExtInitHintPath, ""); } catch { /* best-effort */ }
                    noExtInit = true;
                }
            }
        }

        // Cache miss (or bypass, or this-model-disabled): load source,
        // optimize, optionally save the result for next time.
        SessionOptions BuildWriteOptions(out bool cudaUsed)
        {
            var o = Create(ep, optLevel, enableProfiling: false, out cudaUsed, disableTf32);
            if (bypassCache || cacheDisabled)
                return o;
            o.OptimizedModelFilePath = useOrtFormat ? cachePathOrt : cachePathOnnx;
            if (useOrtFormat)
            {
                // .ort embeds all initializers inline; no _data sidecar.
                o.AddSessionConfigEntry("session.save_model_format", "ORT");
            }
            else if (!noExtInit)
            {
                // For >2GB graphs, force the optimized weights to an external-data
                // sidecar; otherwise the serializer hits protobuf's 2GB limit.
                // Skipped on the inline tier — this sidecar is what trips the
                // ORT subgraph-initializer bug (issue #60).
                o.AddSessionConfigEntry(
                    "session.optimized_model_external_initializers_file_name",
                    Path.GetFileName(cacheDataPath));
                o.AddSessionConfigEntry(
                    "session.optimized_model_external_initializers_min_size_in_bytes",
                    externalInitializersMinBytes.ToString());
            }
            return o;
        }

        Exception? inlineWriteFailure = null;
        if (noExtInit && !useOrtFormat)
        {
            // The inline tier is the one write that can fail on its own terms:
            // without the sidecar, a graph whose optimized initializers exceed
            // protobuf's 2 GB message limit can't serialize at all. Retry on
            // .ort rather than surfacing a cache-write failure as a model-load
            // failure.
            var inlineOpts = BuildWriteOptions(out var inlineUsedCuda);
            try
            {
                usedCuda = inlineUsedCuda;
                return new InferenceSession(modelPath, inlineOpts);
            }
            catch (Exception ex)
            {
                inlineOpts.Dispose();
                inlineWriteFailure = ex;
                try { File.Delete(cachePathOnnx); } catch { /* best-effort */ }
                useOrtFormat = true;
            }
        }

        var opts = BuildWriteOptions(out var freshUsedCuda);
        usedCuda = freshUsedCuda;
        var freshSession = new InferenceSession(modelPath, opts);

        // Only now is the demotion justified. The catch above can't tell a real
        // serializer limit from something transient and unrelated — CUDA OOM
        // under memory pressure, a briefly unreadable source sidecar — and the
        // .use-ort hint is sticky for the life of the cache key. Writing it only
        // once .ort has demonstrably succeeded where inline failed keeps a bad
        // afternoon from permanently downgrading a model that was fine.
        if (inlineWriteFailure is not null)
        {
            Console.WriteLine(
                $"[cache-format] {Path.GetFileName(cachePathOnnx)}: " +
                $"inline .onnx write failed ({inlineWriteFailure.GetType().Name}), .ort succeeded; " +
                "using .ort for this model (see issue #60).");
            try { File.WriteAllText(useOrtHintPath, ""); } catch { /* best-effort */ }
        }
        return freshSession;
    }

    // Returns the cache-key path STEM (no extension). Caller appends
    // ".onnx" / ".ort" / ".no-ext-init" / ".use-ort" / ".cache-disabled"
    // as needed.
    private static string ComputeCacheBasePath(
        string modelPath, ExecutionProvider ep, GraphOptimizationLevel optLevel)
    {
        var fi = new FileInfo(modelPath);
        var ortVer = typeof(InferenceSession).Assembly.GetName().Version?.ToString() ?? "unknown";
        var epTag = ep switch
        {
            ExecutionProvider.Cpu => "cpu",
            // Auto resolves to a different provider per platform, so its tag has to
            // follow -- otherwise a macOS Auto run and an explicit WebGpu run build
            // two copies of the same optimised graph under different keys.
            ExecutionProvider.Auto => OperatingSystem.IsMacOS() ? "webgpu" : "cuda",
            ExecutionProvider.Cuda => "cuda",
            ExecutionProvider.DirectML => "dml",
            ExecutionProvider.CoreML => "coreml",
            ExecutionProvider.WebGpu => "webgpu",
            _ => "auto",
        };
        // Include mtime+size in a short hash so source edits invalidate.
        // The source's external-data sidecar counts as source: the inline tier
        // writes a cache file that still references it by name rather than
        // copying the weights, so a sidecar swapped under an untouched .onnx
        // would otherwise be a silent stale-weights HIT.
        var sidecar = new FileInfo(modelPath + "_data");
        var sidecarKey = sidecar.Exists
            ? $"{sidecar.LastWriteTimeUtc.Ticks}|{sidecar.Length}"
            : "no-sidecar";
        var keyBytes = Encoding.UTF8.GetBytes(
            $"{fi.LastWriteTimeUtc.Ticks}|{fi.Length}|{sidecarKey}|{epTag}|{optLevel}|{ortVer}");
        var hash = SHA256.HashData(keyBytes).AsSpan(0, 6);
        var hashHex = Convert.ToHexString(hash).ToLowerInvariant();
        var dir = fi.DirectoryName ?? ".";
        var stem = Path.GetFileNameWithoutExtension(modelPath);
        return Path.Combine(dir, $"{stem}.opt.{epTag}.{hashHex}");
    }
}
