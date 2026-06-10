using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Chatterbox.App.Models;
using Chatterbox.Base;
using Chatterbox.Base.Alignment;
using Chatterbox.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Models;

namespace Chatterbox.App.Services;

/// <summary>
/// Streaming synthesis + cross-attention alignment for the reader UI.
/// Per-chunk synthesis emits audio + word timings together; word timings
/// come from the Chatterbox LM's own self-attention via
/// <see cref="ChatterboxAttentionAligner"/> rather than a separate ASR
/// forced-alignment pass. See
/// <c>docs/tts_alignment_desync_investigation.md</c> for the why.
///
/// Sessions are lazily loaded on the first call and reused across
/// subsequent invocations — the pipeline load is the big sticky cost
/// (~5-10 s), and rebuilding it per click would make the UI feel slow.
/// </summary>
public sealed class SynthesisService : ITtsBackend
{
    private readonly string _onnxBundleDir;
    private readonly string? _tokenizerJsonPath;
    private readonly ExecutionProvider _ep;

    private ChatterboxPipeline? _pipeline;
    private readonly object _gate = new();

    public SynthesisService(
        string onnxBundleDir,
        string? tokenizerJsonPath = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _onnxBundleDir = onnxBundleDir;
        _tokenizerJsonPath = tokenizerJsonPath;
        _ep = ep;
    }

    public int SampleRate => ChatterboxConstants.S3GenSr;

    /// <summary>
    /// Synthesize <paramref name="request"/>.Text in the voice WAV at
    /// <paramref name="request"/>.Voice and stream chunks back via
    /// <paramref name="onChunkProduced"/> as each one completes. Returns the final
    /// concatenated WAV path + full alignment sidecar once everything is done.
    /// (Chatterbox ignores <c>request.Speed</c>.)
    ///
    /// Per-chunk flow: LM.Generate(captureAlignment: true) produces both
    /// the speech tokens and the cross-attention matrix in one rollout.
    /// Vocoder runs on the tokens. Aligner converts the matrix into
    /// word timings. All synchronous per chunk — no separate alignment
    /// thread, no ContinueWith chain, no NFA model load.
    ///
    /// Cancellation: checked at chunk boundaries. The LM rollout itself
    /// is uninterruptible once started; cancel during a rollout takes
    /// effect after the current chunk completes.
    /// </summary>
    public async Task<SynthesisResult> SynthesizeStreamingAsync(
        TtsRequest request,
        Action<ChunkProducedEvent>? onChunkProduced = null,
        Action<ProgressEvent>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var voicePath = request.Voice;
        var text = request.Text;
        var outWavPath = request.OutWavPath;
        onProgress?.Invoke(new ProgressEvent("loading models"));
        // EnsureLoaded is sync + heavy; offload to the threadpool so the
        // UI dispatcher doesn't stall on the first call (~5-10 s).
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        // Always route through MarkdownTextExtractor (see prior comment
        // in the pre-attention version for why we don't heuristic-detect).
        var preparedText = MarkdownTextExtractor.Extract(text).Text;
        if (string.IsNullOrWhiteSpace(preparedText))
            throw new InvalidOperationException("Text is empty after markdown extraction.");

        var chunks = ParagraphChunker.Chunk(preparedText);
        int totalChunks = chunks.Count;
        onProgress?.Invoke(new ProgressEvent($"chunked into {totalChunks} paragraphs"));

        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pipeline = _pipeline!;
            var tokenizer = pipeline.Tokenizer
                ?? throw new InvalidOperationException(
                    "ChatterboxPipeline has no tokenizer — required for attention alignment. "
                    + "Pass tokenizerJsonPath to the SynthesisService constructor, or ensure "
                    + "the HF cache contains the Chatterbox tokenizer.json.");
            var spk = pipeline.Embedder.Embed(voicePath);
            cancellationToken.ThrowIfCancellationRequested();

            // Per-chunk synthesis loop. Each iteration: LM rollout with
            // attention capture → vocoder → aligner. No separate
            // alignment thread; alignment is part of the same per-chunk
            // sequence as synthesis (~ms compared to seconds of LM/vocoder).
            var chunkAudios = new float[totalChunks][];
            var sidecarChunks = new List<ChunkRecord>(totalChunks);
            var allWords = new List<AlignedWord>();
            int sampleOffsetCursor = 0;

            const int maxLmSteps = 1024;
            for (int idx = 0; idx < totalChunks; idx++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                onProgress?.Invoke(new ProgressEvent($"synthesizing", idx + 1, totalChunks));

                string chunkText = chunks[idx];
                var tokenIds = tokenizer.WrapForLm(chunkText);
                var lmResult = pipeline.Lm.Generate(
                    spk.CondEmb, tokenIds, maxSteps: maxLmSteps, captureAlignment: true);

                // Truncation check: if the LM hit maxSteps without
                // emitting StopSpeechToken, the chunk's audio is shorter
                // than the text demands. The alignment matrix will only
                // cover the spoken prefix; the aligner's interpolation
                // will spread the un-spoken trailing words across a tiny
                // window at the end. Surface this so the user can see
                // which chunk got cut.
                bool truncated = lmResult.Steps >= maxLmSteps
                    && lmResult.RawGeneratedTokens[^1] != ChatterboxConstants.StopSpeechToken;
                if (truncated)
                {
                    onProgress?.Invoke(new ProgressEvent(
                        $"⚠ chunk {idx + 1}/{totalChunks} hit LM step cap — audio may be truncated",
                        idx + 1, totalChunks));
                }

                var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
                var wav = pipeline.Vocoder.Synthesize(
                    speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);

                chunkAudios[idx] = wav;
                double chunkStartSec = sampleOffsetCursor / (double)ChatterboxConstants.S3GenSr;
                double chunkEndSec = (sampleOffsetCursor + wav.Length) / (double)ChatterboxConstants.S3GenSr;
                sampleOffsetCursor += wav.Length;

                // Cross-attention alignment. lmResult.Alignment is
                // guaranteed non-null because we passed captureAlignment.
                var localWords = ChatterboxAttentionAligner.Align(
                    lmResult.Alignment!, chunkText, tokenizer, wav.Length, ChatterboxConstants.S3GenSr);
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
                    Text = chunkText,
                    WordCount = absWords.Count,
                });

                onProgress?.Invoke(new ProgressEvent($"chunk {idx + 1}/{totalChunks} ready", idx + 1, totalChunks));
                onChunkProduced?.Invoke(new ChunkProducedEvent(
                    idx, totalChunks, wav, chunkText, chunkStartSec, absWords));
            }

            // Concatenate all chunks in input order + write the final WAV.
            int totalSamples = chunkAudios.Sum(c => c.Length);
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

            var sidecar = new AlignmentSidecar
            {
                AudioPath = outWavPath,
                SampleRate = ChatterboxConstants.S3GenSr,
                AudioDurationSeconds = totalSamples / (double)ChatterboxConstants.S3GenSr,
                Aligner = "chatterbox_attention",
                Chunks = sidecarChunks,
                Words = allWords,
            };
            return new SynthesisResult(outWavPath, sidecar);
        }, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Concatenate <paramref name="chunks"/> in the given order
    /// and write a 24 kHz mono float32 WAV. Used to persist a partial
    /// result when the user cancels mid-synthesis — Chatterbox chunks
    /// can be concatenated directly because the LM rollout state isn't
    /// shared across chunks (each chunk is a self-contained synthesis
    /// against the same conditioning).</summary>
    public static void WriteWavFromChunks(string outPath, IReadOnlyList<float[]> chunks, int sampleRate)
    {
        int total = 0;
        foreach (var c in chunks) total += c.Length;
        var all = new float[total];
        int off = 0;
        foreach (var c in chunks)
        {
            Array.Copy(c, 0, all, off, c.Length);
            off += c.Length;
        }
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        using var writer = new WaveFileWriter(outPath, fmt);
        writer.WriteSamples(all, 0, all.Length);
    }

    private void EnsureLoaded()
    {
        if (_pipeline is not null) return;
        lock (_gate)
        {
            if (_pipeline is not null) return;
            _pipeline = new ChatterboxPipeline(_onnxBundleDir, _ep, _tokenizerJsonPath);
        }
    }

    public void Dispose()
    {
        _pipeline?.Dispose();
    }
}
