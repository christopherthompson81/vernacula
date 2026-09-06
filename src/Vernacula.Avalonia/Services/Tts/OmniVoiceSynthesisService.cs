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

        var preparedText = MarkdownTextExtractor.Extract(request.Text).Text;
        if (string.IsNullOrWhiteSpace(preparedText))
            throw new InvalidOperationException("Text is empty after markdown extraction.");

        return await Task.Run(() =>
        {
            var tts = _tts!;
            // The voice is per request, the loaded model is not: swapping voices is a JSON read.
            if (tts.Voice?.Id != request.Voice)
                tts.Voice = StoredVoice.Load(_voiceLibDir, request.Voice)
                    ?? throw new InvalidOperationException($"voice \"{request.Voice}\" not found in {_voiceLibDir}");

            var chunks = tts.ChunkForSynthesis(preparedText, lang);
            int totalChunks = chunks.Count;
            onProgress?.Invoke(new ProgressEvent($"chunked into {totalChunks} pieces", null, totalChunks));

            // One leveler for the whole document: the chunks are concatenated, so each one must
            // not be normalised to the same peak on its own or the level pumps between them.
            var leveler = new OmniVoiceAudioPost.StoredVoiceLeveler();
            var chunkAudios = new float[totalChunks][];
            var sidecarChunks = new List<ChunkRecord>(totalChunks);
            var allWords = new List<AlignedWord>();
            int sampleOffsetCursor = 0;

            for (int idx = 0; idx < totalChunks; idx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                onProgress?.Invoke(new ProgressEvent($"synthesizing ({request.NumStep} steps)", idx + 1, totalChunks));

                string chunkText = chunks[idx];
                var sp = tts.SpeakAligned(chunkText, lang, request.NumStep, leveler);
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
                Aligner = "omnivoice_ipa_proportional",
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
            _tts ??= new OmniVoiceIpaTts(_onnxDir, _tokenizerJson, _phonemizerDataDir, _ep);
        }
    }

    public void Dispose() => _tts?.Dispose();
}
