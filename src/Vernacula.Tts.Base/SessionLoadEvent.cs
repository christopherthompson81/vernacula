namespace Vernacula.Tts.Base;

/// <summary>
/// One ONNX session-load event reported via <see cref="SessionLoadObserver"/>.
/// Designed for observability — letting consumers print per-session timing,
/// cache state, and disk footprint without the library owning a logger.
/// </summary>
/// <param name="FileName">Bare name of the source file, e.g. <c>speech_encoder.onnx</c>.</param>
/// <param name="ElapsedMs">Wall-clock time spent constructing the <c>InferenceSession</c>.</param>
/// <param name="CacheHit">
/// True when an optimized cache file was loaded; false when a fresh
/// optimization happened. See <see cref="Vernacula.Base.Inference.OrtSessionBuilder"/>
/// for the layered cache semantics.
/// </param>
/// <param name="UsedCuda">
/// True iff the CUDA execution provider was successfully appended to this
/// session. Distinct from the *requested* EP because
/// <see cref="Vernacula.Base.Models.ExecutionProvider.Auto"/> can silently
/// fall back to DirectML when CUDA is unavailable.
/// </param>
/// <param name="SourceSizeBytes">Size of the source <c>.onnx</c> on disk, for context.</param>
public sealed record SessionLoadEvent(
    string FileName,
    long ElapsedMs,
    bool CacheHit,
    bool UsedCuda,
    long SourceSizeBytes);

/// <summary>
/// Callback invoked once per ONNX session loaded by a Vernacula.Tts.Base
/// component (SpeakerEmbedder, AcousticLM, Vocoder, ChatterboxPipeline).
/// Pass <c>null</c> to opt out of session logging. Fired synchronously
/// inside the constructor; don't perform expensive work in the handler.
/// </summary>
public delegate void SessionLoadObserver(SessionLoadEvent e);
