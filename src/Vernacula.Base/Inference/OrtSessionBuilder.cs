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
}
