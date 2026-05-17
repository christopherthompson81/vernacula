using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Chatterbox.App.Models;
using Chatterbox.Base;
using Chatterbox.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Alignment;
using Vernacula.Base.Models;

namespace Chatterbox.App.Services;

/// <summary>
/// Streaming synthesis + alignment for the reader UI. The non-streaming
/// MVP path (single returned <see cref="SynthesisResult"/>) is still
/// here as a convenience; the new streaming flow is what wires up to
/// per-chunk playback and per-word highlighting.
///
/// Sessions are lazily loaded on the first call and reused across
/// subsequent invocations — the pipeline load is the big sticky cost
/// (~5-10 s for Chatterbox + ~1 s for NFA), and rebuilding it per
/// click would make the UI feel slow.
/// </summary>
public sealed class SynthesisService : IDisposable
{
    private readonly string _onnxBundleDir;
    private readonly string? _nfaBundleDir;
    private readonly string? _tokenizerJsonPath;
    private readonly ExecutionProvider _ep;

    private ChatterboxPipeline? _pipeline;
    private NemoNfaAligner? _aligner;
    private readonly object _gate = new();

    public SynthesisService(
        string onnxBundleDir,
        string? nfaBundleDir = null,
        string? tokenizerJsonPath = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _onnxBundleDir = onnxBundleDir;
        _nfaBundleDir = nfaBundleDir;
        _tokenizerJsonPath = tokenizerJsonPath;
        _ep = ep;
    }

    public sealed record SynthesisResult(string AudioPath, AlignmentSidecar Alignment);

    /// <summary>Per-chunk streaming event payload.</summary>
    /// <param name="ChunkIndex">0-based index in input order.</param>
    /// <param name="TotalChunks">Total chunks the synthesis will produce.</param>
    /// <param name="Audio24k">Raw float32 mono samples at 24 kHz (Chatterbox's native rate).</param>
    /// <param name="ChunkText">The reference text for this chunk (post-markdown-extraction).</param>
    /// <param name="AudioStartSeconds">Absolute start in the concatenated audio timeline.</param>
    /// <param name="Words">Per-word alignments with absolute timings, or empty when no NFA bundle is configured.</param>
    public sealed record ChunkProducedEvent(
        int ChunkIndex,
        int TotalChunks,
        float[] Audio24k,
        string ChunkText,
        double AudioStartSeconds,
        IReadOnlyList<AlignedWord> Words);

    /// <summary>Status-line events for the UI. Phase is a free-form short string
    /// like "loading models", "synthesizing chunk 3/8", "aligning chunk 4/8".</summary>
    public sealed record ProgressEvent(string Phase, int? ChunkIndex = null, int? TotalChunks = null);

    /// <summary>
    /// Synthesize text in <paramref name="voicePath"/>'s voice and stream
    /// chunks back via <paramref name="onChunkProduced"/> as each one
    /// completes. Returns the final concatenated WAV path + full
    /// alignment sidecar once everything is done.
    ///
    /// <paramref name="onProgress"/> fires for major phase transitions
    /// (model load, chunk start/done, alignment start/done) and for
    /// every chunk index — bind it to the UI's status line so the user
    /// sees forward motion through long runs.
    ///
    /// Cancellation: checked at chunk boundaries and after the synthesis
    /// call. The LM rollout (~30 s per chunk worst case) is uninterruptible
    /// once started; a cancel during a chunk's rollout takes effect only
    /// after that chunk completes.
    /// </summary>
    public async Task<SynthesisResult> SynthesizeStreamingAsync(
        string voicePath,
        string text,
        string outWavPath,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        onProgress?.Invoke(new ProgressEvent("loading models"));
        // EnsureLoaded is sync + heavy; offload to the threadpool so the
        // UI dispatcher doesn't stall on the first call (~5-10 s).
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        // Always route through MarkdownTextExtractor — see the inline note
        // in the pre-streaming version for why we don't heuristic-detect.
        var preparedText = MarkdownTextExtractor.Extract(text).Text;
        if (string.IsNullOrWhiteSpace(preparedText))
            throw new InvalidOperationException("Text is empty after markdown extraction.");

        var chunks = ParagraphChunker.Chunk(preparedText);
        int totalChunks = chunks.Count;
        onProgress?.Invoke(new ProgressEvent($"chunked into {totalChunks} paragraphs"));

        // Per-chunk state, shared between the synth callback and the
        // post-synth concatenation + alignment-sidecar build. The
        // ConcurrentDictionary holds the raw audio keyed by chunk index
        // so we can both stream-emit and concatenate at the end.
        var chunkAudios = new ConcurrentDictionary<int, float[]>();
        var chunkTexts = chunks;  // input order; index aligns with idx
        var allWords = new List<AlignedWord>();
        var sidecarChunks = new List<ChunkRecord>();
        int sampleOffsetCursor = 0;

        return await Task.Run(async () =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            var spk = _pipeline!.Embedder.Embed(voicePath);
            cancellationToken.ThrowIfCancellationRequested();

            // Alignment is non-trivial CPU/GPU work (~25% of vocoder wall
            // time on a typical chunk). Running it on the synth thread
            // would block the next chunk's vocoder pass. Instead we chain
            // alignment+emit onto a ContinueWith pipeline so the synth
            // thread can return immediately after stamping the cursor.
            // ContinueWith preserves input order, so allWords / sidecarChunks
            // appends are lock-free.
            Task alignChain = Task.CompletedTask;

            void EmitChunk(int idx, float[] wav, ChunkTiming _t)
            {
                cancellationToken.ThrowIfCancellationRequested();
                chunkAudios[idx] = wav;
                // Cursor stamp is cheap; do it on the synth thread so we
                // can hand the absolute timing to the alignment task.
                // ChunkedSynthesizer calls EmitChunk in input order from a
                // single thread, so no lock needed.
                double chunkStartSec = sampleOffsetCursor / (double)ChatterboxConstants.S3GenSr;
                double chunkEndSec = (sampleOffsetCursor + wav.Length) / (double)ChatterboxConstants.S3GenSr;
                sampleOffsetCursor += wav.Length;

                alignChain = alignChain.ContinueWith(
                    _ => AlignAndEmit(idx, wav, chunkStartSec, chunkEndSec),
                    cancellationToken,
                    TaskContinuationOptions.None,
                    TaskScheduler.Default);
            }

            void AlignAndEmit(int idx, float[] wav, double chunkStartSec, double chunkEndSec)
            {
                cancellationToken.ThrowIfCancellationRequested();
                IReadOnlyList<AlignedWord> wordsForChunk;
                if (_aligner is not null)
                {
                    onProgress?.Invoke(new ProgressEvent("aligning", idx + 1, totalChunks));
                    var wav16k = Vernacula.Base.AudioUtils.AudioTo16000Mono(
                        wav, ChatterboxConstants.S3GenSr, channels: 1);
                    var localWords = _aligner.Align(wav16k, chunkTexts[idx], "en");
                    var absWords = new List<AlignedWord>(localWords.Count);
                    foreach (var w in localWords)
                    {
                        absWords.Add(new AlignedWord
                        {
                            Text = w.Text,
                            StartSeconds = chunkStartSec + w.StartSeconds,
                            EndSeconds = chunkStartSec + w.EndSeconds,
                            ChunkIndex = idx,
                        });
                    }
                    allWords.AddRange(absWords);
                    sidecarChunks.Add(new ChunkRecord
                    {
                        Index = idx,
                        AudioStartSeconds = chunkStartSec,
                        AudioEndSeconds = chunkEndSec,
                        Text = chunkTexts[idx],
                        WordCount = absWords.Count,
                    });
                    wordsForChunk = absWords;
                }
                else
                {
                    sidecarChunks.Add(new ChunkRecord
                    {
                        Index = idx,
                        AudioStartSeconds = chunkStartSec,
                        AudioEndSeconds = chunkEndSec,
                        Text = chunkTexts[idx],
                        WordCount = 0,
                    });
                    wordsForChunk = Array.Empty<AlignedWord>();
                }
                onProgress?.Invoke(new ProgressEvent($"chunk {idx + 1}/{totalChunks} ready", idx + 1, totalChunks));
                onChunkProduced?.Invoke(new ChunkProducedEvent(
                    idx, totalChunks, wav, chunkTexts[idx], chunkStartSec, wordsForChunk));
            }

            // Dispatch. Single-chunk path uses the bare pipeline (no
            // ChunkedSynthesizer overhead); long-form uses the streaming
            // callback. Both end up populating chunkAudios + invoking
            // EmitChunk in input order.
            if (totalChunks <= 1)
            {
                onProgress?.Invoke(new ProgressEvent("synthesizing", 1, 1));
                var tokenIds = _pipeline.Tokenizer!.WrapForLm(preparedText);
                var lmResult = _pipeline.Lm.Generate(spk.CondEmb, tokenIds);
                var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
                var samples = _pipeline.Vocoder.Synthesize(
                    speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
                EmitChunk(0, samples,
                    new ChunkTiming(0, 0, lmResult.Steps, 0, samples.Length));
            }
            else
            {
                onProgress?.Invoke(new ProgressEvent($"synthesizing {totalChunks} chunks"));
                var tokensPerChunk = chunks.Select(c => _pipeline.Tokenizer!.WrapForLm(c)).ToArray();
                var synth = new ChunkedSynthesizer(_pipeline);
                synth.Synthesize(spk, tokensPerChunk, chunkProduced: EmitChunk);
            }

            // Drain the alignment chain before we build the sidecar — the
            // last chunk's alignment is still in flight after Synthesize
            // returns.
            await alignChain.ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();

            // Concatenate all chunks in input order + write the final WAV.
            int totalSamples = chunkAudios.Sum(kv => kv.Value.Length);
            var allSamples = new float[totalSamples];
            int off = 0;
            for (int i = 0; i < totalChunks; i++)
            {
                var w = chunkAudios[i];
                Array.Copy(w, 0, allSamples, off, w.Length);
                off += w.Length;
            }
            var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
            using (var writer = new WaveFileWriter(outWavPath, fmt))
                writer.WriteSamples(allSamples, 0, allSamples.Length);

            // The ContinueWith alignment chain runs in input order so
            // sidecarChunks should already be sorted; the explicit sort
            // is a future-proof against a reordering callback policy.
            sidecarChunks.Sort((a, b) => a.Index.CompareTo(b.Index));

            var sidecar = new AlignmentSidecar
            {
                AudioPath = outWavPath,
                SampleRate = ChatterboxConstants.S3GenSr,
                AudioDurationSeconds = totalSamples / (double)ChatterboxConstants.S3GenSr,
                Aligner = _aligner is null ? "none" : "nemo_nfa",
                Chunks = sidecarChunks,
                Words = allWords,
            };
            return new SynthesisResult(outWavPath, sidecar);
        }, cancellationToken).ConfigureAwait(false);
    }

    private void EnsureLoaded()
    {
        if (_pipeline is not null) return;
        lock (_gate)
        {
            if (_pipeline is not null) return;
            _pipeline = new ChatterboxPipeline(_onnxBundleDir, _ep, _tokenizerJsonPath);
            if (_nfaBundleDir is not null && Directory.Exists(_nfaBundleDir))
                _aligner = new NemoNfaAligner(_nfaBundleDir, _ep);
        }
    }

    public void Dispose()
    {
        _aligner?.Dispose();
        _pipeline?.Dispose();
    }
}
