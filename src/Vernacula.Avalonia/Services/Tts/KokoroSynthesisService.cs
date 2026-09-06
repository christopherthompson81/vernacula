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
/// Kokoro-82M streaming backend. Same per-chunk streaming contract as
/// <see cref="SynthesisService"/>, but synthesis goes through <see cref="KokoroTts"/>
/// (named voice + speed) and word timings come from the model's per-token predicted
/// durations rather than LM cross-attention. <see cref="TtsRequest.Voice"/> is a voice
/// name (e.g. "af_heart"); en-gb pronunciation is inferred from a bf_/bm_ prefix.
///
/// The model loads lazily on the first call and is reused (the ONNX session + phonemizer
/// language data are the sticky cost).
/// </summary>
public sealed class KokoroSynthesisService : ITtsBackend
{
    private readonly string _onnxDir;
    private readonly string? _dataDir;
    private readonly ExecutionProvider _ep;

    private KokoroTts? _tts;
    private readonly object _gate = new();

    public KokoroSynthesisService(string onnxDir, string? phonemizerDataDir,
                                  ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _onnxDir = onnxDir;
        _dataDir = phonemizerDataDir;
        _ep = ep;
    }

    public int SampleRate => Kokoro.SampleRate;

    public async Task<SynthesisResult> SynthesizeStreamingAsync(
        TtsRequest request,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var voice = request.Voice;
        var speed = request.Speed;
        bool british = voice.StartsWith("bf_", StringComparison.Ordinal)
                    || voice.StartsWith("bm_", StringComparison.Ordinal);

        onProgress?.Invoke(new ProgressEvent("loading models"));
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        return await Task.Run(() =>
        {
            var tts = _tts!;

            // One paragraph per segment; a paragraph longer than Kokoro's 512-token window is
            // split by the token-aware chunker and joined back (char-based chunking alone can
            // overflow the graph on a long paragraph).
            (float[] Audio, IReadOnlyList<AlignedWord> Words) SynthesizeSegment(Vernacula.Tts.Base.Markdown.TextSegment seg, Action<string> warn)
            {
                var parts = new List<(float[], IReadOnlyList<(string, double, double)>)>();
                foreach (var chunkText in tts.ChunkForSynthesis(seg.Text, british))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var sp = tts.SpeakAligned(chunkText, voice, speed, british);
                    parts.Add((sp.Audio, sp.Words.Select(w => (w.Text, w.StartSec, w.EndSec)).ToList()));
                }
                return SegmentedSynthesis.Join(parts, SampleRate);
            }

            return SegmentedSynthesis.Run(request, SampleRate, "kokoro_duration",
                SynthesizeSegment, onChunkProduced, onProgress, cancellationToken);
        }, cancellationToken).ConfigureAwait(false);
    }

    private void EnsureLoaded()
    {
        if (_tts is not null) return;
        lock (_gate)
        {
            _tts ??= new KokoroTts(_onnxDir, _dataDir, _ep);
        }
    }

    public void Dispose() => _tts?.Dispose();
}
