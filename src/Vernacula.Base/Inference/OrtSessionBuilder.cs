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
        out bool usedCuda)
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
                        opts.AppendExecutionProvider_CUDA(0);
                        usedCuda = true;
                    }
                    catch { }
                }
                try { opts.AppendExecutionProvider_DML(0); } catch { }
                break;

            case ExecutionProvider.Cuda:
                try
                {
                    opts.AppendExecutionProvider_CUDA(0);
                    usedCuda = true;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        "CUDA EP not available in the current ONNX Runtime build.", ex);
                }
                break;

            case ExecutionProvider.DirectML:
                try { opts.AppendExecutionProvider_DML(0); }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        "DirectML EP not available. Build with -p:UseDirectML=true.", ex);
                }
                break;

            case ExecutionProvider.Cpu:
                break;
        }

        return opts;
    }

    /// <summary>
    /// Create an <see cref="InferenceSession"/> backed by a disk-cached
    /// post-optimization graph. First call loads <paramref name="modelPath"/>,
    /// runs graph optimization at <paramref name="optLevel"/>, and writes the
    /// optimized graph next to the source as <c>{stem}.opt.{ep}.{ortver}.onnx</c>
    /// (with a <c>.opt_data</c> sidecar for tensors above
    /// <paramref name="externalInitializersMinBytes"/>). Subsequent calls find
    /// the cached file, skip optimization (<c>ORT_DISABLE_ALL</c>), and load
    /// directly — typically 5–10× faster for large graphs.
    ///
    /// Cache key embeds EP, ORT version, and source-file mtime+size, so source
    /// changes or ORT upgrades automatically invalidate. Stale optimized files
    /// are NOT auto-cleaned — callers can `rm <stem>.opt.*` to reset.
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
        => CreateCachedSession(modelPath, ep, out _, optLevel, externalInitializersMinBytes);

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
    {
        cacheHit = false;
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        bool bypassCache = Environment.GetEnvironmentVariable("VERNACULA_ORT_NO_CACHE") == "1";
        string cachePath = bypassCache ? "" : ComputeCachePath(modelPath, ep, optLevel);
        string cacheDataPath = cachePath + "_data";

        if (!bypassCache && File.Exists(cachePath))
        {
            // Cache hit: load pre-optimized graph with optimization DISABLED
            // (the graph is already optimized; re-running passes is wasted work
            // and may hit unsupported-op errors on a fused graph).
            var hitOpts = Create(ep, GraphOptimizationLevel.ORT_DISABLE_ALL, enableProfiling: false, out _);
            try
            {
                var session = new InferenceSession(cachePath, hitOpts);
                cacheHit = true;
                return session;
            }
            catch
            {
                // Cache file is corrupt or incompatible — fall through to fresh load.
                hitOpts.Dispose();
                try { File.Delete(cachePath); } catch { /* best-effort */ }
                try { File.Delete(cacheDataPath); } catch { /* best-effort */ }
            }
        }

        // Cache miss (or bypass): load source, optimize, save the result.
        var opts = Create(ep, optLevel, enableProfiling: false, out _);
        if (!bypassCache)
        {
            opts.OptimizedModelFilePath = cachePath;
            // For >2GB graphs, force the optimized weights to an external-data
            // sidecar; otherwise the serializer hits protobuf's 2GB limit.
            opts.AddSessionConfigEntry(
                "session.optimized_model_external_initializers_file_name",
                Path.GetFileName(cacheDataPath));
            opts.AddSessionConfigEntry(
                "session.optimized_model_external_initializers_min_size_in_bytes",
                externalInitializersMinBytes.ToString());
        }
        return new InferenceSession(modelPath, opts);
    }

    private static string ComputeCachePath(
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
        return Path.Combine(dir, $"{stem}.opt.{epTag}.{hashHex}.onnx");
    }
}
