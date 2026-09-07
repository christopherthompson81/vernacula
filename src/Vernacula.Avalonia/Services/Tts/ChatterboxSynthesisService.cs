using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Vernacula.App.Models;
using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Alignment;
using Vernacula.Tts.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Models;

namespace Vernacula.App.Services.Tts;

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
public sealed class ChatterboxSynthesisService : ITtsBackend
{
    private readonly string _onnxBundleDir;
    private readonly string? _tokenizerJsonPath;
    private readonly ExecutionProvider _ep;

    private ChatterboxPipeline? _pipeline;
    private readonly object _gate = new();

    public ChatterboxSynthesisService(
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
        onProgress?.Invoke(new ProgressEvent("loading models"));
        // EnsureLoaded is sync + heavy; offload to the threadpool so the
        // UI dispatcher doesn't stall on the first call (~5-10 s).
        await Task.Run(EnsureLoaded, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        return await Task.Run(() =>
        {
            var pipeline = _pipeline!;
            var tokenizer = pipeline.Tokenizer
                ?? throw new InvalidOperationException(
                    "ChatterboxPipeline has no tokenizer — required for attention alignment. "
                    + "Pass tokenizerJsonPath to the ChatterboxSynthesisService constructor, or ensure "
                    + "the HF cache contains the Chatterbox tokenizer.json.");
            var spk = pipeline.Embedder.Embed(request.Voice);
            cancellationToken.ThrowIfCancellationRequested();
            const int maxLmSteps = 1024;

            // A segment (one paragraph) is rendered as one or more LM rollouts — the chunker
            // keeps each rollout under the LM's comfortable length — and joined back into the
            // paragraph's audio. Per rollout: LM.Generate(captureAlignment: true) produces the
            // speech tokens and the cross-attention matrix in one pass; the vocoder runs on the
            // tokens; the aligner turns the matrix into word timings.
            (float[] Audio, IReadOnlyList<AlignedWord> Words) SynthesizeSegment(Vernacula.Tts.Base.Markdown.TextSegment seg, Action<string> warn)
            {
                var chunks = ParagraphChunker.Chunk(seg.Text);
                var parts = new List<(float[], IReadOnlyList<(string, double, double)>)>(chunks.Count);
                for (int c = 0; c < chunks.Count; c++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string chunkText = chunks[c];
                    var tokenIds = tokenizer.WrapForLm(chunkText);
                    var lmResult = pipeline.Lm.Generate(
                        spk.CondEmb, tokenIds, maxSteps: maxLmSteps, captureAlignment: true);

                    // If the LM hit maxSteps without emitting StopSpeechToken, the audio is
                    // shorter than the text demands and the aligner will squeeze the unspoken
                    // tail into the end. Say so, per paragraph, so the user can see which.
                    bool truncated = lmResult.Steps >= maxLmSteps
                        && lmResult.RawGeneratedTokens[^1] != ChatterboxConstants.StopSpeechToken;
                    if (truncated)
                        warn($"⚠ paragraph {seg.Index + 1} hit the LM step cap — audio may be truncated");

                    var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
                    var wav = pipeline.Vocoder.Synthesize(
                        speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
                    var localWords = ChatterboxAttentionAligner.Align(
                        lmResult.Alignment!, chunkText, tokenizer, wav.Length, ChatterboxConstants.S3GenSr);
                    parts.Add((wav, localWords.Select(w => (w.Text, w.StartSeconds, w.EndSeconds)).ToList()));
                }
                return SegmentedSynthesis.Join(parts, ChatterboxConstants.S3GenSr);
            }

            return SegmentedSynthesis.Run(request, ChatterboxConstants.S3GenSr, "chatterbox_attention",
                SynthesizeSegment, onChunkProduced, onProgress, cancellationToken);
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
