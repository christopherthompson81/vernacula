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
        bool enableProfiling = false)
        => Create(ep, optLevel, enableProfiling, out _);

    /// <inheritdoc cref="Create(ExecutionProvider, GraphOptimizationLevel, bool)"/>
    /// <param name="usedCuda">True when the CUDA execution provider was
    /// successfully appended. Callers use this to gate CUDA-only paths
    /// (IOBinding, CUDA graphs) without re-probing.</param>
    public static SessionOptions Create(
        ExecutionProvider ep,
        GraphOptimizationLevel optLevel,
        bool enableProfiling,
        out bool usedCuda,
        bool disableTf32 = false)
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

            case ExecutionProvider.Cpu:
                break;
        }

        return opts;
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
    /// <b>Layered cache format.</b> Three states per cache key, advanced by
    /// observed round-trip behavior:
    /// <list type="number">
    ///   <item><description><b>.onnx</b> (primary). Fast: ORT can lazy-load
    ///     weights from a <c>_data</c> external-initializer sidecar. Works
    ///     for almost every graph.</description></item>
    ///   <item><description><b>.ort</b> (fallback, marked by a
    ///     <c>.use-ort</c> hint file). Used when the <c>.onnx</c> writer
    ///     mishandles the graph — currently any graph containing a Loop
    ///     subgraph (issue #56 — the <c>.onnx</c> serializer duplicates
    ///     body-scope initializers into the outer scope). The <c>.ort</c>
    ///     binary writer encodes subgraphs faithfully and round-trips at
    ///     every opt level, but embeds all initializers inline so loads
    ///     are slower for large graphs. Only used when needed.</description></item>
    ///   <item><description><b>No cache</b> (last resort, marked by a
    ///     <c>.cache-disabled</c> sentinel). Used when even <c>.ort</c>
    ///     can't round-trip. Stops the broken write→fail→delete cycle.
    ///     Has not been observed in practice but the path exists as
    ///     defense-in-depth.</description></item>
    /// </list>
    ///
    /// Convergence costs one extra cache miss per state transition: a
    /// Loop-bearing graph takes 2 runs to reach a steady-state cache hit
    /// (run 1: write .onnx → run 2: .onnx reload fails, escalate to .ort
    /// hint, write .ort in the same call → run 3+: .ort cache HIT).
    /// Most graphs stay on the .onnx path forever.
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
        // All cache-key-derived paths. The five files form a small state
        // machine; the per-key disposition is encoded by which marker
        // (if any) exists alongside the actual cache file(s).
        string cachePathOnnx = cacheBase + ".onnx";   // primary cache (with _data sidecar)
        string cacheDataPath = cachePathOnnx + "_data";
        string cachePathOrt = cacheBase + ".ort";     // fallback for Loop-bearing graphs
        string useOrtHintPath = cacheBase + ".use-ort";       // ".onnx round-trip failed, escalate to .ort"
        string cacheDisabledPath = cacheBase + ".cache-disabled"; // "both formats failed, skip caching"

        bool cacheDisabled = !bypassCache && File.Exists(cacheDisabledPath);
        bool useOrtFormat = !bypassCache && !cacheDisabled && File.Exists(useOrtHintPath);

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
                else
                {
                    // .onnx round-trip failed; escalate to .ort fallback.
                    // useOrtFormat=true makes the fall-through cache-write
                    // path emit .ort this run — so by the end of *this* call
                    // the .ort file is on disk, and the very next call sees
                    // a cache HIT instead of having to repeat the cycle.
                    Console.WriteLine(
                        $"[cache-format] {Path.GetFileName(cachePathOnnx)}: " +
                        ".onnx round-trip failed (likely a Loop-subgraph graph); switching to .ort for this model (see issue #56).");
                    try { File.Delete(cachePathOnnx); } catch { /* best-effort */ }
                    try { File.Delete(cacheDataPath); } catch { /* best-effort */ }
                    try { File.WriteAllText(useOrtHintPath, ""); } catch { /* best-effort */ }
                    useOrtFormat = true;
                }
            }
        }

        // Cache miss (or bypass, or this-model-disabled): load source,
        // optimize, optionally save the result for next time.
        var opts = Create(ep, optLevel, enableProfiling: false, out var freshUsedCuda, disableTf32);
        usedCuda = freshUsedCuda;
        if (!bypassCache && !cacheDisabled)
        {
            opts.OptimizedModelFilePath = useOrtFormat ? cachePathOrt : cachePathOnnx;
            if (useOrtFormat)
            {
                // .ort embeds all initializers inline; no _data sidecar.
                opts.AddSessionConfigEntry("session.save_model_format", "ORT");
            }
            else
            {
                // For >2GB graphs, force the optimized weights to an external-data
                // sidecar; otherwise the serializer hits protobuf's 2GB limit.
                opts.AddSessionConfigEntry(
                    "session.optimized_model_external_initializers_file_name",
                    Path.GetFileName(cacheDataPath));
                opts.AddSessionConfigEntry(
                    "session.optimized_model_external_initializers_min_size_in_bytes",
                    externalInitializersMinBytes.ToString());
            }
        }
        return new InferenceSession(modelPath, opts);
    }

    // Returns the cache-key path STEM (no extension). Caller appends
    // ".onnx" / ".ort" / ".use-ort" / ".cache-disabled" as needed.
    private static string ComputeCacheBasePath(
        string modelPath, ExecutionProvider ep, GraphOptimizationLevel optLevel)
    {
        var fi = new FileInfo(modelPath);
        var ortVer = typeof(InferenceSession).Assembly.GetName().Version?.ToString() ?? "unknown";
        var epTag = ep switch
        {
            ExecutionProvider.Cpu => "cpu",
            ExecutionProvider.Cuda or ExecutionProvider.Auto => "cuda",
            ExecutionProvider.DirectML => "dml",
            _ => "auto",
        };
        // Include mtime+size in a short hash so source edits invalidate.
        var keyBytes = Encoding.UTF8.GetBytes(
            $"{fi.LastWriteTimeUtc.Ticks}|{fi.Length}|{epTag}|{optLevel}|{ortVer}");
        var hash = SHA256.HashData(keyBytes).AsSpan(0, 6);
        var hashHex = Convert.ToHexString(hash).ToLowerInvariant();
        var dir = fi.DirectoryName ?? ".";
        var stem = Path.GetFileNameWithoutExtension(modelPath);
        return Path.Combine(dir, $"{stem}.opt.{epTag}.{hashHex}");
    }
}
