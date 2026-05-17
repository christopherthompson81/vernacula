using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// Internal helper that wraps <see cref="OrtSessionBuilder.CreateCachedSession"/>
/// with consistent timing, source-size capture, and observer notification.
/// Used by <see cref="SpeakerEmbedder"/>, <see cref="AcousticLM"/>, and
/// <see cref="Vocoder"/> so they all build the same
/// <see cref="SessionLoadEvent"/> shape — one place to change if the event
/// gains new fields or the timing semantics shift.
/// </summary>
internal static class SessionLoader
{
    /// <summary>
    /// Load an ONNX session via the cached-session builder, time the load,
    /// and (if <paramref name="onLoad"/> is non-null) fire one
    /// <see cref="SessionLoadEvent"/> for it.
    /// </summary>
    /// <param name="usedCuda">Effective-EP flag from
    /// <see cref="OrtSessionBuilder.CreateCachedSession"/> — true iff the
    /// CUDA EP was successfully appended to the session backing the
    /// returned <see cref="InferenceSession"/>.</param>
    public static InferenceSession LoadAndReport(
        string path, ExecutionProvider ep, SessionLoadObserver? onLoad, out bool usedCuda)
    {
        var sw = Stopwatch.StartNew();
        var session = OrtSessionBuilder.CreateCachedSession(path, ep, out var hit, out usedCuda);
        sw.Stop();
        onLoad?.Invoke(new SessionLoadEvent(
            Path.GetFileName(path), sw.ElapsedMilliseconds, hit, usedCuda,
            new FileInfo(path).Length));
        return session;
    }

    /// <summary>
    /// Overload for callers that don't need the effective-EP flag
    /// (i.e. classes that don't dispatch on CUDA-only paths like IoBinding).
    /// </summary>
    public static InferenceSession LoadAndReport(
        string path, ExecutionProvider ep, SessionLoadObserver? onLoad)
        => LoadAndReport(path, ep, onLoad, out _);
}
