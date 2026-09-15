#if AUDIOCPP_BACKEND
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.AudioCpp;
using Vernacula.Phonemizer;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Alignment;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Kokoro-82M through audio.cpp's C ABI, beside the ONNX Kokoro rather than instead of it.
/// Same segmented-synthesis contract as every other engine — one paragraph per segment, streamed
/// as it is rendered, per-segment WAVs, one sidecar at the end.
///
/// <para>
/// ⚠ THE WORD TIMINGS HERE ARE ESTIMATES, and that is a property of the ABI rather than a
/// shortcut. The <c>kokoro_tts</c> family reports <c>SupportsTimestamps=false</c> and returns no
/// words, so unlike the ONNX Kokoro — which reads the model's own predicted per-token durations —
/// there is nothing to measure. The sidecar's aligner is named
/// <c>audiocpp_proportional</c> so that is legible to whoever reads it later.
/// </para>
///
/// <para>
/// Compiled only when the AudioCpp-Bindings submodule is present; see the project file. The
/// model loads lazily on the first call and is reused.
/// </para>
/// </summary>
public sealed class AudioCppSynthesisService : ITtsBackend
{
    private readonly string   _modelPath;
    private readonly string[] _backends;
    private readonly int      _threads;

    private AudioCppTts? _tts;
    private readonly object _gate = new();

    public AudioCppSynthesisService(string modelPath, string[] backends, int threads)
    {
        _modelPath = modelPath;
        _backends  = backends;
        _threads   = threads;
    }

    public int SampleRate => AudioCppTts.SampleRate;

    public async Task<SynthesisResult> SynthesizeStreamingAsync(
        TtsRequest request,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var voice = request.Voice;
        var speed = request.Speed;

        onProgress?.Invoke(new ProgressEvent("loading models"));
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        return await Task.Run(() =>
        {
            var tts = _tts!;

            (float[] Audio, IReadOnlyList<AlignedWord> Words) SynthesizeSegment(
                Vernacula.Tts.Base.Markdown.TextSegment seg, Action<string> warn)
            {
                cancellationToken.ThrowIfCancellationRequested();
                // No chunker: the family splits on its own text_chunk_size and joins, so a
                // paragraph of any length comes back as one buffer (investigation Run 4).
                var audio = tts.Speak(seg.Text, voice, speed);
                return (audio, EstimateWords(seg.Text, audio.Length / (double)SampleRate));
            }

            // No batch synthesizer: the ABI takes one request at a time, so there is nothing to
            // batch and offering a delegate that loops would only disable the reuse decorator.
            return SegmentedSynthesis.Run(request, SampleRate, "audiocpp_proportional",
                SynthesizeSegment, onChunkProduced, onProgress, cancellationToken);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Spreads the segment's words across its duration, weighted by word length.
    ///
    /// <para>
    /// Reuses OmniVoice's estimator rather than growing a second one. That class is deliberately
    /// separate from its engine so it can run without a model, and its untraced path — which is
    /// what an empty <see cref="PhonemeTrace"/> selects — is precisely "weight each word by its
    /// length", which is all that can be known here: audio.cpp phonemizes inside the engine with
    /// eSpeak-ng and hands back no trace, no phonemes and no timings.
    /// </para>
    /// </summary>
    internal static IReadOnlyList<AlignedWord> EstimateWords(string text, double totalSeconds) =>
        [.. OmniVoiceIpaAlignment.Proportional(text, new PhonemeTrace(), totalSeconds)
                .Select(w => new AlignedWord { Text = w.Text, StartSeconds = w.StartSec, EndSeconds = w.EndSec })];

    private void EnsureLoaded()
    {
        if (_tts is not null) return;
        lock (_gate)
        {
            _tts ??= new AudioCppTts(_modelPath, _backends, _threads);
        }
    }

    public void Dispose() => _tts?.Dispose();
}
#endif
