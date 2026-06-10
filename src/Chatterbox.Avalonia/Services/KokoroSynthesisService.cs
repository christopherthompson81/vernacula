using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Chatterbox.App.Models;
using Chatterbox.Base;
using Chatterbox.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Models;

namespace Chatterbox.App.Services;

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
    private readonly string _dataDir;
    private readonly ExecutionProvider _ep;

    private KokoroTts? _tts;
    private readonly object _gate = new();

    public KokoroSynthesisService(string onnxDir, string phonemizerDataDir,
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

        var preparedText = MarkdownTextExtractor.Extract(request.Text).Text;
        if (string.IsNullOrWhiteSpace(preparedText))
            throw new InvalidOperationException("Text is empty after markdown extraction.");

        var chunks = ParagraphChunker.Chunk(preparedText);
        int totalChunks = chunks.Count;
        onProgress?.Invoke(new ProgressEvent($"chunked into {totalChunks} paragraphs"));

        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            var tts = _tts!;

            var chunkAudios = new float[totalChunks][];
            var sidecarChunks = new List<ChunkRecord>(totalChunks);
            var allWords = new List<AlignedWord>();
            int sampleOffsetCursor = 0;

            for (int idx = 0; idx < totalChunks; idx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                onProgress?.Invoke(new ProgressEvent("synthesizing", idx + 1, totalChunks));

                string chunkText = chunks[idx];
                var sp = tts.SpeakAligned(chunkText, voice, speed, british);
                var wav = sp.Audio;

                chunkAudios[idx] = wav;
                double chunkStartSec = sampleOffsetCursor / (double)SampleRate;
                double chunkEndSec = (sampleOffsetCursor + wav.Length) / (double)SampleRate;
                sampleOffsetCursor += wav.Length;

                var absWords = new List<AlignedWord>(sp.Words.Count);
                foreach (var w in sp.Words)
                {
                    absWords.Add(new AlignedWord
                    {
                        Text = w.Text,
                        StartSeconds = chunkStartSec + w.StartSec,
                        EndSeconds = chunkStartSec + w.EndSec,
                        ChunkIndex = idx,
                    });
                }
                allWords.AddRange(absWords);
                sidecarChunks.Add(new ChunkRecord
                {
                    Index = idx,
                    AudioStartSeconds = chunkStartSec,
                    AudioEndSeconds = chunkEndSec,
                    Text = chunkText,
                    WordCount = absWords.Count,
                });

                onProgress?.Invoke(new ProgressEvent($"chunk {idx + 1}/{totalChunks} ready", idx + 1, totalChunks));
                onChunkProduced?.Invoke(new ChunkProducedEvent(
                    idx, totalChunks, wav, chunkText, chunkStartSec, absWords));
            }

            int totalSamples = chunkAudios.Sum(c => c.Length);
            var allSamples = new float[totalSamples];
            int off = 0;
            for (int i = 0; i < totalChunks; i++)
            {
                Array.Copy(chunkAudios[i], 0, allSamples, off, chunkAudios[i].Length);
                off += chunkAudios[i].Length;
            }
            var fmt = WaveFormat.CreateIeeeFloatWaveFormat(SampleRate, 1);
            using (var writer = new WaveFileWriter(request.OutWavPath, fmt))
                writer.WriteSamples(allSamples, 0, allSamples.Length);

            var sidecar = new AlignmentSidecar
            {
                AudioPath = request.OutWavPath,
                SampleRate = SampleRate,
                AudioDurationSeconds = totalSamples / (double)SampleRate,
                Aligner = "kokoro_duration",
                Chunks = sidecarChunks,
                Words = allWords,
            };
            return new SynthesisResult(request.OutWavPath, sidecar);
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
