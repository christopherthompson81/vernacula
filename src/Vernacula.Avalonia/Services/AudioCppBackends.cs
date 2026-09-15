using Vernacula.Base.Models;

namespace Vernacula.App.Services;

/// <summary>
/// The app's execution-provider setting translated into audio.cpp's backend names.
/// </summary>
/// <remarks>
/// Shared by both audio.cpp backends — recognition and synthesis — because a second copy is how
/// one of them ends up on the CPU while the other takes CUDA from the same setting, which is a
/// difference that shows up as "slow" and never as "wrong".
/// </remarks>
internal static class AudioCppBackends
{
    /// <summary>
    /// The backends to try, in order, taking the first that opens.
    ///
    /// <para>
    /// ⚠ Auto is the DEFAULT and it is not Cuda. An earlier version read
    /// <c>== Cuda ? "cuda" : "cpu"</c>, so every install that had never set an execution provider
    /// explicitly — which is the out-of-the-box state — ran audio.cpp on the CPU while the ONNX
    /// backends took CUDA through the same Auto. It transcribed correctly and looked merely slow.
    /// </para>
    ///
    /// <para>
    /// The engine's backends are not ONNX Runtime's, so this maps rather than casts, and Auto
    /// becomes an ordered list: whether CUDA is registered depends on how the engine was BUILT,
    /// which nothing here can see, so the only reliable test is to ask it.
    /// </para>
    /// </summary>
    public static string[] For(ExecutionProvider ep) => ep switch
    {
        ExecutionProvider.Cpu    => ["cpu"],
        ExecutionProvider.Cuda   => ["cuda", "cpu"],
        ExecutionProvider.CoreML => ["metal", "cpu"],
        // No WebGPU in the engine; Vulkan is the nearest portable GPU backend it does have, and
        // CPU catches a build without either.
        ExecutionProvider.WebGpu => ["vulkan", "cpu"],
        _                        => ["cuda", "metal", "vulkan", "cpu"],
    };
}
