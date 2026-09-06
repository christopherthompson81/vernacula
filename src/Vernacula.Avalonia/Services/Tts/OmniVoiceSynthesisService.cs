using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.App.Models;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Models;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// OmniVoice-IPA streaming backend: text → IPA (vernacula-phonemizer, any language) → the IPA
/// fine-tune, in a stored voice from the web demo's library. Same per-chunk contract as the other
/// backends; <see cref="TtsRequest.Voice"/> is a library voice id, <see cref="TtsRequest.Lang"/>
/// the phonemizer language, <see cref="TtsRequest.NumStep"/> the diffusion steps.
///
/// Word timings are proportional estimates, not measurements — see <see cref="OmniVoiceIpaTts"/>.
/// The 2.45 GB transformer loads lazily on the first call and is reused.
/// </summary>
public sealed class OmniVoiceSynthesisService : ITtsBackend
{
    private readonly string _onnxDir;
    private readonly string? _tokenizerJson;
    private readonly string? _phonemizerDataDir;
    private readonly string _voiceLibDir;
    private readonly ExecutionProvider _ep;

    private OmniVoiceIpaTts? _tts;
    private readonly object _gate = new();

    public OmniVoiceSynthesisService(string onnxDir, string? tokenizerJson, string? phonemizerDataDir,
                                     string voiceLibDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _onnxDir = onnxDir;
        _tokenizerJson = tokenizerJson;
        _phonemizerDataDir = phonemizerDataDir;
        _voiceLibDir = voiceLibDir;
        _ep = ep;
    }

    public int SampleRate => OmniVoiceIpaTts.SampleRate;

    public async Task<SynthesisResult> SynthesizeStreamingAsync(
        TtsRequest request,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var lang = string.IsNullOrWhiteSpace(request.Lang) ? "en" : request.Lang!;

        onProgress?.Invoke(new ProgressEvent("loading models"));
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        return await Task.Run(() =>
        {
            var tts = _tts!;
            // The voice is per request, the loaded model is not: swapping voices is a JSON read.
            if (tts.Voice?.Id != request.Voice)
                tts.Voice = StoredVoice.Load(_voiceLibDir, request.Voice)
                    ?? throw new InvalidOperationException($"voice \"{request.Voice}\" not found in {_voiceLibDir}");

            // One leveler for the whole document: the segments are concatenated, so each must
            // not be normalised to the same peak on its own or the level pumps between them.
            var leveler = new OmniVoiceAudioPost.StoredVoiceLeveler();

            // One paragraph per segment; a paragraph over the token budget is split on sentences
            // by the chunker and joined back.
            (float[] Audio, IReadOnlyList<AlignedWord> Words) SynthesizeSegment(Vernacula.Tts.Base.Markdown.TextSegment seg, Action<string> warn)
            {
                var parts = new List<(float[], IReadOnlyList<(string, double, double)>)>();
                foreach (var chunkText in tts.ChunkForSynthesis(seg.Text, lang))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var sp = tts.SpeakAligned(chunkText, lang, request.NumStep, leveler);
                    parts.Add((sp.Audio, sp.Words.Select(w => (w.Text, w.StartSec, w.EndSec)).ToList()));
                }
                return SegmentedSynthesis.Join(parts, SampleRate);
            }

            return SegmentedSynthesis.Run(request, SampleRate, "omnivoice_ipa_proportional",
                SynthesizeSegment, onChunkProduced, onProgress, cancellationToken);
        }, cancellationToken).ConfigureAwait(false);
    }

    private void EnsureLoaded()
    {
        if (_tts is not null) return;
        lock (_gate)
        {
            _tts ??= new OmniVoiceIpaTts(_onnxDir, _tokenizerJson, _phonemizerDataDir, _ep);
        }
    }

    public void Dispose() => _tts?.Dispose();
}
