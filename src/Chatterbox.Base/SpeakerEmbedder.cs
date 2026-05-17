using System.Diagnostics;
using Chatterbox.Base.AudioIo;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// Outputs of <c>speech_encoder.onnx</c>. Owned by the caller; not
/// disposable because they're plain CPU arrays — ORT's <see cref="Tensor{T}"/>
/// instances were copied out at <see cref="SpeakerEmbedder.Embed(float[])"/>
/// time so the calling session can be safely disposed before these are used.
///
/// Field types are deliberately heterogeneous: the three tensor fields stay
/// as <see cref="DenseTensor{T}"/> because their consumers (<see cref="AcousticLM"/>
/// and <see cref="Vocoder"/>) re-wrap them with <c>NamedOnnxValue.CreateFromTensor</c>
/// — handing them across as <c>float[]</c>+dims would force the consumer
/// to reconstruct the tensor. <see cref="AudioTokens"/> is a bare
/// <c>long[]</c> because <see cref="AcousticLmResult.BuildSpeechTokens"/>
/// uses it as a plain array. Don't "fix" the inconsistency without
/// checking the call sites first.
/// </summary>
/// <param name="CondEmb">
/// <c>[1, S_cond, 1024]</c> — speaker conditioning embedding the LM
/// prepends to its text-embedding prompt.
/// </param>
/// <param name="AudioTokens">
/// <c>[1, T_audio]</c> — discrete audio tokens of the reference clip.
/// Prepended to the LM-generated tokens before the vocoder runs.
/// </param>
/// <param name="SpeakerEmbeddings"><c>[1, 192]</c> — speaker vector for the vocoder.</param>
/// <param name="SpeakerFeatures">
/// <c>[1, PromptLen, 80]</c> — mel features of the prompt; the vocoder
/// trims the first PromptLen mel frames after CFM solve.
/// </param>
public sealed record SpeakerEmbedding(
    DenseTensor<float> CondEmb,
    long[] AudioTokens,
    DenseTensor<float> SpeakerEmbeddings,
    DenseTensor<float> SpeakerFeatures);

/// <summary>
/// Wraps <c>speech_encoder.onnx</c>. Given a voice-prompt WAV (or a
/// pre-resampled 24 kHz mono buffer), produces the speaker conditioning
/// outputs the LM and vocoder consume.
///
/// Stateless apart from the owned <see cref="InferenceSession"/>. Safe to
/// reuse across multiple <see cref="Embed(string)"/> / <see cref="Embed(float[])"/>
/// calls; not thread-safe (matches ORT session semantics).
/// </summary>
public sealed class SpeakerEmbedder : IDisposable
{
    private readonly InferenceSession _session;

    /// <summary>
    /// Load <c>speech_encoder.onnx</c> from <paramref name="onnxPath"/>
    /// via the cached-session builder. Disposes the underlying session
    /// when this object is disposed.
    /// </summary>
    /// <param name="onLoad">Optional callback fired once with timing,
    /// cache state, and effective-EP info for the loaded session.</param>
    public SpeakerEmbedder(string onnxPath, ExecutionProvider ep, SessionLoadObserver? onLoad = null)
    {
        var sw = Stopwatch.StartNew();
        _session = OrtSessionBuilder.CreateCachedSession(onnxPath, ep, out var hit, out var usedCuda);
        sw.Stop();
        onLoad?.Invoke(new SessionLoadEvent(
            Path.GetFileName(onnxPath), sw.ElapsedMilliseconds, hit, usedCuda,
            new FileInfo(onnxPath).Length));
    }

    /// <summary>Load + resample the WAV at <paramref name="voicePath"/>, then embed.</summary>
    public SpeakerEmbedding Embed(string voicePath)
        => Embed(VoicePromptLoader.Load(voicePath));

    /// <summary>
    /// Run <c>speech_encoder.onnx</c> on a 24 kHz mono float32 buffer
    /// of length <see cref="ChatterboxConstants.DummyAudioSamples"/>.
    /// (Use <see cref="VoicePromptLoader.Load(string)"/> to produce one
    /// from an arbitrary-format WAV.)
    /// </summary>
    public SpeakerEmbedding Embed(float[] audio24kMono)
    {
        if (audio24kMono.Length != ChatterboxConstants.DummyAudioSamples)
            throw new ArgumentException(
                $"audio must be exactly {ChatterboxConstants.DummyAudioSamples} samples at 24 kHz; got {audio24kMono.Length}. " +
                "Use VoicePromptLoader.Load() to pad/crop a WAV.", nameof(audio24kMono));

        var audioT = new DenseTensor<float>(audio24kMono, [1, audio24kMono.Length]);
        using var output = _session.Run([NamedOnnxValue.CreateFromTensor("audio_values", audioT)]);
        var list = output.ToList();

        // Copy each tensor out of the session-owned DisposableNamedOnnxValue
        // collection so the caller can safely dispose the session before
        // using the returned record.
        var condEmb = CopyToDense<float>(list[0].AsTensor<float>());
        var audioTokens = list[1].AsTensor<long>().ToArray();
        var spkEmb = CopyToDense<float>(list[2].AsTensor<float>());
        var spkFeat = CopyToDense<float>(list[3].AsTensor<float>());

        return new SpeakerEmbedding(condEmb, audioTokens, spkEmb, spkFeat);
    }

    private static DenseTensor<T> CopyToDense<T>(Tensor<T> src)
        => new(src.ToArray(), src.Dimensions.ToArray());

    public void Dispose() => _session.Dispose();
}
